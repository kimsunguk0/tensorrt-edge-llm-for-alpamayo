from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import os
import threading
import time
import termios
from typing import Any

import numpy as np


A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

BAUD_RATES = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}


@dataclass(frozen=True, slots=True)
class GnssSerialConfig:
    device: str = "/dev/ttyUSB0"
    baud: int = 115200
    max_fixes: int = 2048
    reconnect_s: float = 2.0


@dataclass(frozen=True, slots=True)
class GnssFix:
    timestamp_utc_ns: int
    timestamp_monotonic_ns: int
    lat: float
    lon: float
    alt: float
    fix_type: int | None = None
    num_sats: int | None = None
    hdop: float | None = None
    speed_mps: float | None = None
    course_rad: float | None = None


def _nmea_checksum_ok(sentence: str) -> bool:
    if "*" not in sentence:
        return True
    body, checksum_text = sentence.strip()[1:].split("*", 1)
    checksum_text = checksum_text[:2]
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        expected = int(checksum_text, 16)
    except ValueError:
        return False
    return checksum == expected


def _split_nmea(sentence: str) -> tuple[str, list[str]] | None:
    sentence = sentence.strip()
    if not sentence.startswith("$") or len(sentence) < 6:
        return None
    if not _nmea_checksum_ok(sentence):
        return None
    body = sentence[1:].split("*", 1)[0]
    parts = body.split(",")
    if not parts:
        return None
    return parts[0][-3:].upper(), parts[1:]


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_lat_lon(value: str, hemisphere: str) -> float | None:
    if not value or not hemisphere:
        return None
    if hemisphere in ("N", "S"):
        degree_digits = 2
    elif hemisphere in ("E", "W"):
        degree_digits = 3
    else:
        return None
    try:
        degrees = int(value[:degree_digits])
        minutes = float(value[degree_digits:])
    except ValueError:
        return None
    coord = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        coord *= -1.0
    return coord


def _parse_rmc_date(value: str) -> date | None:
    if len(value) != 6:
        return None
    try:
        day = int(value[0:2])
        month = int(value[2:4])
        year_2 = int(value[4:6])
    except ValueError:
        return None
    year = 2000 + year_2 if year_2 < 80 else 1900 + year_2
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_nmea_time_ns(time_text: str, day: date) -> int | None:
    if len(time_text) < 6:
        return None
    try:
        hour = int(time_text[0:2])
        minute = int(time_text[2:4])
        sec_float = float(time_text[4:])
    except ValueError:
        return None
    second = int(sec_float)
    frac_ns = int(round((sec_float - second) * 1_000_000_000))
    try:
        dt = datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1_000_000_000) + frac_ns


def _geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)
    normal = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * slat * slat)
    x = (normal + alt_m) * clat * clon
    y = (normal + alt_m) * clat * slon
    z = (normal * (1.0 - E2_WGS84) + alt_m) * slat
    return np.stack([x, y, z], axis=-1)


def _ecef_to_enu(xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref_xyz = _geodetic_to_ecef(
        np.asarray([ref_lat_deg], dtype=np.float64),
        np.asarray([ref_lon_deg], dtype=np.float64),
        np.asarray([ref_alt_m], dtype=np.float64),
    )[0]
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)
    rot = np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )
    return (xyz - ref_xyz) @ rot.T


def _yaw_to_rot(yaw_rad: np.ndarray) -> np.ndarray:
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    rot = np.zeros((len(yaw_rad), 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 1] = -sin_y
    rot[:, 1, 0] = sin_y
    rot[:, 1, 1] = cos_y
    rot[:, 2, 2] = 1.0
    return rot


def _dedupe_fixes(fixes: list[GnssFix]) -> list[GnssFix]:
    by_timestamp: dict[int, GnssFix] = {}
    for fix in fixes:
        if math.isfinite(fix.lat) and math.isfinite(fix.lon) and math.isfinite(fix.alt):
            by_timestamp[fix.timestamp_utc_ns] = fix
    return [by_timestamp[key] for key in sorted(by_timestamp)]


class GnssSerialReader:
    def __init__(self, config: GnssSerialConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._fixes: list[GnssFix] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_rmc_date: date | None = None
        self._last_rmc_speed_mps: float | None = None
        self._last_rmc_course_rad: float | None = None
        self._latest_error: str | None = None
        self._line_count = 0
        self._fix_count = 0
        self._running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gnss-serial-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._fixes[-1] if self._fixes else None
            age_ms = None
            latest_payload = None
            if latest is not None:
                age_ms = max((time.monotonic_ns() - latest.timestamp_monotonic_ns) / 1e6, 0.0)
                latest_payload = {
                    "timestamp_utc_ns": latest.timestamp_utc_ns,
                    "lat": latest.lat,
                    "lon": latest.lon,
                    "alt": latest.alt,
                    "fix_type": latest.fix_type,
                    "num_sats": latest.num_sats,
                    "hdop": latest.hdop,
                }
            return {
                "enabled": True,
                "running": self._running,
                "device": self.config.device,
                "baud": self.config.baud,
                "line_count": self._line_count,
                "fix_count": self._fix_count,
                "buffered_fixes": len(self._fixes),
                "latest_fix_age_ms": age_ms,
                "latest_fix": latest_payload,
                "last_error": self._latest_error,
            }

    def parse_sentence(self, sentence: str, *, monotonic_ns: int | None = None) -> GnssFix | None:
        parsed = _split_nmea(sentence)
        if parsed is None:
            return None
        sentence_type, fields = parsed
        monotonic = time.monotonic_ns() if monotonic_ns is None else monotonic_ns

        if sentence_type == "RMC":
            self._parse_rmc(fields)
            return None
        if sentence_type == "GGA":
            return self._parse_gga(fields, monotonic)
        return None

    def build_ego_history(
        self,
        *,
        t0_utc_ns: int | None,
        history_len: int = 16,
        dt_s: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
        dt_ns = int(round(dt_s * 1e9))
        min_span_ns = (history_len - 1) * dt_ns
        with self._lock:
            fixes = _dedupe_fixes(list(self._fixes))
        if len(fixes) < 2:
            return None

        utc = np.asarray([fix.timestamp_utc_ns for fix in fixes], dtype=np.int64)
        lat = np.asarray([fix.lat for fix in fixes], dtype=np.float64)
        lon = np.asarray([fix.lon for fix in fixes], dtype=np.float64)
        alt = np.asarray([fix.alt for fix in fixes], dtype=np.float64)
        if int(utc[-1] - utc[0]) < min_span_ns:
            return None

        requested_t0 = int(t0_utc_ns) if t0_utc_ns is not None else int(utc[-1])
        t0 = int(np.clip(requested_t0, int(utc[0] + min_span_ns), int(utc[-1])))
        hist_times = np.asarray([t0 - (history_len - 1 - i) * dt_ns for i in range(history_len)], dtype=np.int64)

        interp_lat = np.interp(hist_times, utc, lat)
        interp_lon = np.interp(hist_times, utc, lon)
        interp_alt = np.interp(hist_times, utc, alt)
        ref_lla = (
            float(np.interp([t0], utc, lat)[0]),
            float(np.interp([t0], utc, lon)[0]),
            float(np.interp([t0], utc, alt)[0]),
        )

        world_xyz = _ecef_to_enu(_geodetic_to_ecef(interp_lat, interp_lon, interp_alt), *ref_lla).astype(np.float32)
        vel_xy = np.zeros((len(world_xyz), 2), dtype=np.float32)
        vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_s)
        vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_s)
        vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_s)
        yaw = np.arctan2(vel_xy[:, 1], vel_xy[:, 0]).astype(np.float32)
        world_rot = _yaw_to_rot(yaw)

        p0 = world_xyz[-1]
        r0 = world_rot[-1]
        local_xyz = ((world_xyz - p0) @ r0).astype(np.float32)
        local_rot = np.einsum("ij,tjk->tik", r0.T, world_rot).astype(np.float32)
        info = {
            "requested_t0_utc_ns": requested_t0,
            "used_t0_utc_ns": t0,
            "oldest_fix_utc_ns": int(utc[0]),
            "latest_fix_utc_ns": int(utc[-1]),
            "history_len": history_len,
            "dt_s": float(dt_s),
            "fix_count": len(fixes),
            "clamped_t0": bool(t0 != requested_t0),
        }
        return local_xyz[None, None, ...], local_rot[None, None, ...], info

    def _run(self) -> None:
        while not self._stop.is_set():
            fd: int | None = None
            try:
                fd = os.open(self.config.device, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
                self._configure(fd)
                self._set_error(None)
                self._running = True
                self._read_loop(fd)
            except Exception as exc:
                self._running = False
                self._set_error(str(exc))
                self._stop.wait(self.config.reconnect_s)
            finally:
                self._running = False
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def _read_loop(self, fd: int) -> None:
        pending = b""
        while not self._stop.is_set():
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                self._stop.wait(0.05)
                continue
            if not chunk:
                self._stop.wait(0.05)
                continue
            pending += chunk
            while b"\n" in pending:
                raw_line, pending = pending.split(b"\n", 1)
                line = raw_line.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                with self._lock:
                    self._line_count += 1
                fix = self.parse_sentence(line)
                if fix is not None:
                    self._append_fix(fix)

    def _configure(self, fd: int) -> None:
        if self.config.baud not in BAUD_RATES:
            raise ValueError(f"Unsupported GNSS baud: {self.config.baud}")
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        attrs[3] = 0
        attrs[4] = BAUD_RATES[self.config.baud]
        attrs[5] = BAUD_RATES[self.config.baud]
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 10
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

    def _parse_rmc(self, fields: list[str]) -> None:
        if len(fields) < 9 or fields[1] != "A":
            return
        parsed_date = _parse_rmc_date(fields[8])
        if parsed_date is not None:
            self._last_rmc_date = parsed_date
        speed_knots = _parse_float(fields[6])
        course_deg = _parse_float(fields[7])
        self._last_rmc_speed_mps = None if speed_knots is None else float(speed_knots * 0.514444)
        self._last_rmc_course_rad = None if course_deg is None else math.radians(course_deg)

    def _parse_gga(self, fields: list[str], monotonic_ns: int) -> GnssFix | None:
        if len(fields) < 9:
            return None
        current_day = self._last_rmc_date or datetime.now(timezone.utc).date()
        utc_ns = _parse_nmea_time_ns(fields[0], current_day)
        lat = _parse_lat_lon(fields[1], fields[2])
        lon = _parse_lat_lon(fields[3], fields[4])
        fix_type = _parse_int(fields[5])
        num_sats = _parse_int(fields[6])
        hdop = _parse_float(fields[7])
        alt = _parse_float(fields[8])
        if utc_ns is None or lat is None or lon is None or alt is None:
            return None
        return GnssFix(
            timestamp_utc_ns=utc_ns,
            timestamp_monotonic_ns=monotonic_ns,
            lat=float(lat),
            lon=float(lon),
            alt=float(alt),
            fix_type=fix_type,
            num_sats=num_sats,
            hdop=hdop,
            speed_mps=self._last_rmc_speed_mps,
            course_rad=self._last_rmc_course_rad,
        )

    def _append_fix(self, fix: GnssFix) -> None:
        with self._lock:
            self._fix_count += 1
            self._fixes.append(fix)
            if len(self._fixes) > self.config.max_fixes:
                del self._fixes[: len(self._fixes) - self.config.max_fixes]

    def _set_error(self, value: str | None) -> None:
        with self._lock:
            self._latest_error = value
