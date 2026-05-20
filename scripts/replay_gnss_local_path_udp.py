#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any
import urllib.request

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_team_replay_common import COORD_MODE_LOCAL, FLAG_VALID, build_packet_dict, wrap_angles  # noqa: E402
from planner_live.gnss_serial import GnssSerialConfig, GnssSerialReader  # noqa: E402


A_WGS84 = 6378137.0
E2_WGS84 = 6.69437999014e-3
DEFAULT_DATASET_ROOT = Path("/workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GNSS rows for selected dataset chunks, convert global lat/lon/alt "
            "to ego-local path packets, save them, and optionally replay as UDP JSON."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--global-path-csv",
        type=Path,
        default=None,
        help="Use this recorded global GNSS path CSV as the route instead of reading a dataset chunk.",
    )
    parser.add_argument("--chunks", default="0,1", help="Chunk list/ranges, e.g. '0,1' or '0-1'.")
    parser.add_argument("--camera-name", default="front", help="Camera frame table used to define chunk time windows.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/output/gnss_local_path_chunks00_01"),
    )
    parser.add_argument("--plan-points", type=int, default=64, help="Future path points excluding origin.")
    parser.add_argument("--plan-dt-s", type=float, default=0.1)
    parser.add_argument(
        "--max-forward-m",
        type=float,
        default=20.0,
        help="Only send local path points in front of the vehicle up to this x distance. Use <=0 to disable.",
    )
    parser.add_argument("--send-hz", type=float, default=10.0)
    parser.add_argument("--udp-host", default="10.179.113.253")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--send", action="store_true", help="Actually send saved payloads over UDP.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of payloads for smoke tests.")
    parser.add_argument("--loop", action="store_true", help="Loop UDP replay until interrupted.")
    parser.add_argument(
        "--log-actual-gnss",
        action="store_true",
        help="Poll a live sensor health endpoint while replaying and save the driven GNSS path.",
    )
    parser.add_argument(
        "--live-follow",
        action="store_true",
        help=(
            "Use live GNSS as the current ego pose and continuously send the selected "
            "dataset GNSS route ahead of the vehicle."
        ),
    )
    parser.add_argument(
        "--sensor-health-url",
        default="http://127.0.0.1:18080/healthz",
        help="Sensor/planner health endpoint containing live GNSS lat/lon/alt.",
    )
    parser.add_argument("--gnss-source", choices=("healthz", "serial"), default="healthz")
    parser.add_argument("--gnss-device", default="/dev/ttyUSB0")
    parser.add_argument("--gnss-baud", type=int, default=115200)
    parser.add_argument("--gnss-fix-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--live-heading-source",
        choices=("auto", "ins", "yaw", "velocity", "history", "course", "route"),
        default="auto",
        help="Heading source for live-follow local-frame conversion.",
    )
    parser.add_argument("--live-min-heading-distance-m", type=float, default=0.5)
    parser.add_argument("--live-min-velocity-heading-mps", type=float, default=0.2)
    parser.add_argument(
        "--live-stationary-speed-mps",
        type=float,
        default=0.15,
        help="Below this live GNSS speed, hold heading instead of trusting jittery yaw/course.",
    )
    parser.add_argument(
        "--live-stationary-distance-m",
        type=float,
        default=0.2,
        help="If recent live GNSS movement is below this distance, also treat the vehicle as stationary.",
    )
    parser.add_argument(
        "--live-heading-filter-alpha",
        type=float,
        default=0.35,
        help="Moving-state yaw low-pass alpha. 1.0 disables filtering.",
    )
    parser.add_argument(
        "--disable-stationary-heading-hold",
        action="store_true",
        help="Use raw selected heading even while stationary.",
    )
    parser.add_argument("--enable-opencv-ui", action="store_true")
    parser.add_argument("--ui-width", type=int, default=1280)
    parser.add_argument("--ui-height", type=int, default=720)
    parser.add_argument("--ui-history-points", type=int, default=120)
    parser.add_argument("--ui-window-name", default="GNSS live follow")
    parser.add_argument("--actual-log-hz", type=float, default=10.0)
    parser.add_argument("--actual-health-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--allow-tail-clamp",
        action="store_true",
        help="Keep rows whose future horizon extends past available GNSS data by clamping to the last row.",
    )
    parser.add_argument(
        "--include-global-points",
        action="store_true",
        help="Include future global lat/lon/alt points in each JSON payload. Useful for debug, larger UDP payloads.",
    )
    return parser.parse_args()


def parse_chunks(value: str) -> list[int]:
    chunks: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chunks.extend(range(int(start), int(end) + 1))
        else:
            chunks.append(int(token))
    return sorted(set(chunks))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
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


def ecef_to_enu(xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref_xyz = geodetic_to_ecef(
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
    rot = np.asarray(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )
    return (xyz - ref_xyz) @ rot.T


def select_pose_gnss_rows(gnss: pd.DataFrame) -> pd.DataFrame:
    base = gnss.dropna(subset=["timestamp_utc_ns", "lat", "lon", "alt"]).copy()
    if base.empty:
        raise RuntimeError("No GNSS rows with timestamp_utc_ns, lat, lon, and alt")

    quality_mask = np.zeros(len(base), dtype=bool)
    if "num_sats" in base.columns:
        quality_mask |= base["num_sats"].notna().to_numpy()
    if "hdop" in base.columns:
        quality_mask |= base["hdop"].notna().to_numpy()
    if int(quality_mask.sum()) >= 2:
        base = base.loc[quality_mask].copy()

    sort_cols = ["timestamp_utc_ns", "row_id"] if "row_id" in base.columns else ["timestamp_utc_ns"]
    base = base.sort_values(sort_cols).drop_duplicates(subset=["timestamp_utc_ns"], keep="last")
    if len(base) < 2:
        raise RuntimeError("Need at least two unique GNSS timestamps")
    return base.reset_index(drop=True)


def load_chunk_windows(dataset_root: Path, camera_name: str, chunks: list[int]) -> pd.DataFrame:
    frames_path = dataset_root / "sensors" / f"camera_{camera_name}" / "frames.parquet"
    if not frames_path.exists():
        raise FileNotFoundError(f"Camera frames parquet not found: {frames_path}")
    frames = pd.read_parquet(frames_path)
    frames = frames[frames["chunk_id"].isin(chunks)].copy()
    if frames.empty:
        raise RuntimeError(f"No camera frames for chunks {chunks} in {frames_path}")
    windows = (
        frames.groupby("chunk_id")["timestamp_utc_ns"]
        .agg(chunk_start_utc_ns="min", chunk_end_utc_ns="max")
        .reset_index()
        .sort_values("chunk_id")
    )
    return windows


def assign_chunk_ids(times_ns: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    chunk_ids = np.full(times_ns.shape, -1, dtype=np.int32)
    for row in windows.itertuples(index=False):
        mask = (times_ns >= int(row.chunk_start_utc_ns)) & (times_ns <= int(row.chunk_end_utc_ns))
        chunk_ids[mask] = int(row.chunk_id)
    return chunk_ids


def load_recorded_global_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Global path CSV not found: {path}")
    route = pd.read_csv(path)
    route = route.dropna(subset=["lat", "lon", "alt"]).copy()
    if route.empty:
        raise RuntimeError(f"No valid lat/lon/alt rows in {path}")
    if "timestamp_utc_ns" not in route.columns:
        if "receive_time_utc_ns" in route.columns:
            route["timestamp_utc_ns"] = route["receive_time_utc_ns"]
        else:
            base_ns = int(time.time_ns())
            route["timestamp_utc_ns"] = base_ns + np.arange(len(route), dtype=np.int64) * 100_000_000
    route["timestamp_utc_ns"] = pd.to_numeric(route["timestamp_utc_ns"], errors="coerce")
    route = route.dropna(subset=["timestamp_utc_ns"]).copy()
    route["timestamp_utc_ns"] = route["timestamp_utc_ns"].astype(np.int64)
    if "chunk_id" not in route.columns:
        route["chunk_id"] = 0
    sort_cols = ["timestamp_utc_ns"]
    if "seq" in route.columns:
        sort_cols.append("seq")
    route = route.sort_values(sort_cols).drop_duplicates(subset=["timestamp_utc_ns"], keep="last")
    if len(route) < 2:
        raise RuntimeError(f"Need at least two unique GNSS rows in {path}")
    return route.reset_index(drop=True)


def compute_world_arrays(gnss: pd.DataFrame) -> dict[str, np.ndarray | tuple[float, float, float]]:
    ref_lla = (
        float(gnss["lat"].iloc[0]),
        float(gnss["lon"].iloc[0]),
        float(gnss["alt"].iloc[0]),
    )
    xyz = ecef_to_enu(
        geodetic_to_ecef(
            gnss["lat"].to_numpy(dtype=np.float64),
            gnss["lon"].to_numpy(dtype=np.float64),
            gnss["alt"].to_numpy(dtype=np.float64),
        ),
        *ref_lla,
    ).astype(np.float32)
    times_s = (gnss["timestamp_utc_ns"].to_numpy(dtype=np.int64) - int(gnss["timestamp_utc_ns"].iloc[0])) / 1e9
    times_s = times_s.astype(np.float64)
    vx = np.gradient(xyz[:, 0].astype(np.float64), times_s).astype(np.float32)
    vy = np.gradient(xyz[:, 1].astype(np.float64), times_s).astype(np.float32)
    yaw = np.arctan2(vy, vx).astype(np.float32)
    speed = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    delta_xy = np.diff(xyz[:, :2].astype(np.float64), axis=0)
    route_s = np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.cumsum(np.linalg.norm(delta_xy, axis=1))]
    ).astype(np.float64)
    return {
        "ref_lla": ref_lla,
        "xyz_enu": xyz,
        "times_s": times_s,
        "route_s_m": route_s,
        "yaw_world": yaw,
        "speed_mps": speed,
    }


def interp_angle(times_s: np.ndarray, yaw_rad: np.ndarray, target_times_s: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(yaw_rad.astype(np.float64))
    return wrap_angles(np.interp(target_times_s, times_s, unwrapped).astype(np.float32))


def build_local_packet(
    *,
    gnss: pd.DataFrame,
    arrays: dict[str, Any],
    row_index: int,
    tx_seq: int,
    chunk_id: int,
    plan_points: int,
    plan_dt_s: float,
    max_forward_m: float,
    include_global_points: bool,
) -> dict[str, Any]:
    utc = gnss["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    t0_ns = int(utc[row_index])
    source_t0_us = t0_ns // 1000
    base_ns = int(utc[0])
    target_times_s = (t0_ns - base_ns) / 1e9 + np.arange(plan_points + 1, dtype=np.float64) * float(plan_dt_s)

    world = arrays["xyz_enu"]
    times_s = arrays["times_s"]
    future_enu = np.stack(
        [
            np.interp(target_times_s, times_s, world[:, 0]),
            np.interp(target_times_s, times_s, world[:, 1]),
            np.interp(target_times_s, times_s, world[:, 2]),
        ],
        axis=-1,
    ).astype(np.float32)

    p0 = future_enu[0]
    yaw0 = float(interp_angle(times_s, arrays["yaw_world"], np.asarray([target_times_s[0]], dtype=np.float64))[0])
    c0 = math.cos(yaw0)
    s0 = math.sin(yaw0)
    rot0 = np.asarray([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    local_xyz = ((future_enu - p0) @ rot0).astype(np.float32)

    vx = np.gradient(local_xyz[:, 0].astype(np.float64), target_times_s).astype(np.float32)
    vy = np.gradient(local_xyz[:, 1].astype(np.float64), target_times_s).astype(np.float32)
    speed = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    yaw_local = wrap_angles(np.arctan2(vy, vx).astype(np.float32))
    yaw_local[0] = 0.0
    yaw_unwrapped = np.unwrap(yaw_local.astype(np.float64)).astype(np.float32)
    yaw_rate = np.gradient(yaw_unwrapped, target_times_s).astype(np.float32)
    curvature = np.where(speed > 1e-6, yaw_rate / np.maximum(speed, 1e-6), 0.0).astype(np.float32)

    point_mask = np.ones(local_xyz.shape[0], dtype=bool)
    if float(max_forward_m) > 0.0:
        x_forward = local_xyz[:, 0].astype(np.float32)
        point_mask = (x_forward >= -1e-3) & (x_forward <= float(max_forward_m) + 1e-3)
        point_mask[0] = True
        if int(point_mask.sum()) < 2 and local_xyz.shape[0] >= 2:
            point_mask[1] = True

    local_xyz_packet = local_xyz[point_mask]
    yaw_packet = yaw_local[point_mask]
    speed_packet = speed[point_mask]
    curvature_packet = curvature[point_mask]
    target_times_packet_s = target_times_s[point_mask]

    packet = build_packet_dict(
        tx_seq=tx_seq,
        plan_seq=tx_seq,
        sample_id=int(row_index),
        source_t0_us=source_t0_us,
        tx_time_us=source_t0_us,
        coord_mode=COORD_MODE_LOCAL,
        dt_s=float(plan_dt_s),
        x=local_xyz_packet[:, 0],
        y=local_xyz_packet[:, 1],
        yaw=yaw_packet,
        v=speed_packet,
        curvature=curvature_packet,
        flags=FLAG_VALID,
    )
    row = gnss.iloc[row_index]
    future_points = packet["points"][1:]
    global_indices = np.flatnonzero(point_mask)
    payload: dict[str, Any] = {
        "label": "gnss_local_path",
        "payload_format": "gnss_local_path_json",
        "path_type": "gnss_local_path",
        "udp_mode": "text_json_gnss_local_path_replay",
        "chunk_id": int(chunk_id),
        "sample_id": int(row_index),
        "front_frame_id": int(row_index),
        "t0_utc_ns": t0_ns,
        "t0_us": source_t0_us,
        "target_offset_s": float((t0_ns - int(utc[0])) / 1e9),
        "actual_offset_s": float((t0_ns - int(utc[0])) / 1e9),
        "plan_dt_s": float(plan_dt_s),
        "requested_plan_points_no_origin": int(plan_points),
        "requested_traj_points_with_origin": int(plan_points + 1),
        "plan_points_no_origin": max(int(packet["header"]["num_points"]) - 1, 0),
        "traj_points_with_origin": int(packet["header"]["num_points"]),
        "max_forward_m": float(max_forward_m) if float(max_forward_m) > 0.0 else None,
        "cropped_by_forward_m": bool(float(max_forward_m) > 0.0),
        "coord_mode": "local",
        "coordinate_note": "ego-local: +x forward, +y left, yaw/curvature positive left",
        "packet_header": packet["header"],
        "packet_points": packet["points"],
        "pred_xyz": local_xyz_packet[1:].astype(float).tolist(),
        "pred_yaw_rad": [float(point["yaw_rad"]) for point in future_points],
        "pred_v_mps": [float(point["v_mps"]) for point in future_points],
        "pred_curvature": [float(point["curvature"]) for point in future_points],
        "source_gnss": {
            "timestamp_utc_ns": t0_ns,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "alt": float(row["alt"]),
            "yaw_world_rad": yaw0,
        },
        "reference_lla": {
            "lat": float(arrays["ref_lla"][0]),
            "lon": float(arrays["ref_lla"][1]),
            "alt": float(arrays["ref_lla"][2]),
        },
    }
    if include_global_points:
        lat = gnss["lat"].to_numpy(dtype=np.float64)
        lon = gnss["lon"].to_numpy(dtype=np.float64)
        alt = gnss["alt"].to_numpy(dtype=np.float64)
        payload["global_points"] = [
            {
                "t_s": float(target_times_packet_s[out_idx] - target_times_packet_s[0]),
                "source_index": int(global_idx),
                "lat": float(np.interp(target_times_s[global_idx], times_s, lat)),
                "lon": float(np.interp(target_times_s[global_idx], times_s, lon)),
                "alt": float(np.interp(target_times_s[global_idx], times_s, alt)),
            }
            for out_idx, global_idx in enumerate(global_indices)
        ]
    return payload


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


_FLOAT_TEXT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _pick_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in mapping:
            parsed = _coerce_float(mapping[key])
            if parsed is not None:
                return parsed
    return None


def _pick_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in mapping:
            parsed = _coerce_int(mapping[key])
            if parsed is not None:
                return parsed
    return None


def _extract_gnss_from_dict(mapping: dict[str, Any]) -> dict[str, Any] | None:
    lat = _pick_float(mapping, ("lat", "latitude"))
    lon = _pick_float(mapping, ("lon", "lng", "longitude"))
    alt = _pick_float(mapping, ("alt", "alt_m", "altitude", "height_m"))
    if lat is None or lon is None or alt is None:
        return None

    fix: dict[str, Any] = {"lat": lat, "lon": lon, "alt": alt}
    timestamp_utc_ns = _pick_int(mapping, ("timestamp_utc_ns", "utc_ns", "time_utc_ns"))
    timestamp_utc_us = _pick_int(mapping, ("timestamp_utc_us", "utc_us", "latest_utc_us", "time_utc_us"))
    if timestamp_utc_ns is not None:
        fix["timestamp_utc_ns"] = timestamp_utc_ns
    elif timestamp_utc_us is not None:
        fix["timestamp_utc_ns"] = int(timestamp_utc_us * 1000)

    yaw_rad = _pick_float(mapping, ("yaw_rad", "heading_rad", "heading_enu_rad"))
    if yaw_rad is not None:
        fix["yaw_rad"] = yaw_rad
    for out_key, keys in (
        ("fix_type", ("fix_type", "quality", "fix_quality")),
        ("num_sats", ("num_sats", "satellites", "sat_count")),
    ):
        parsed = _pick_int(mapping, keys)
        if parsed is not None:
            fix[out_key] = parsed
    for out_key, keys in (
        ("hdop", ("hdop",)),
        ("speed_mps", ("speed_mps", "speed")),
        ("course_rad", ("course_rad", "course_true_rad")),
        ("fix_age_ms", ("fix_age_ms", "latest_fix_age_ms", "age_ms")),
    ):
        parsed = _pick_float(mapping, keys)
        if parsed is not None:
            fix[out_key] = parsed
    course_deg = _pick_float(mapping, ("course_deg", "course_true_deg"))
    if course_deg is not None and "course_rad" not in fix:
        fix["course_rad"] = math.radians(course_deg)

    for key in ("vel", "velocity", "velocity_mps"):
        value = mapping.get(key)
        if isinstance(value, (list, tuple)) and value:
            vector = [_coerce_float(item) for item in value]
            fix["velocity_mps"] = [float(item) for item in vector if item is not None]
            break
    for key in ("covariance", "position_covariance", "pos_covariance", "position_cov", "cov"):
        value = mapping.get(key)
        if isinstance(value, (list, tuple)) and len(value) in (4, 6, 9):
            cov_values = [_coerce_float(item) for item in value]
            if all(item is not None for item in cov_values):
                fix["position_covariance"] = [float(item) for item in cov_values if item is not None]
                break
    for out_key, keys in (
        ("hacc_m", ("hacc_m", "horizontal_accuracy_m", "accuracy_m", "eph")),
        ("vacc_m", ("vacc_m", "vertical_accuracy_m", "epv")),
    ):
        parsed = _pick_float(mapping, keys)
        if parsed is not None:
            fix[out_key] = parsed
    return fix


def _parse_text_gnss(text: str) -> dict[str, Any] | None:
    def field(names: tuple[str, ...]) -> float | None:
        for name in names:
            match = re.search(rf"(?:^|[\s,{{]){re.escape(name)}\s*[:=]\s*({_FLOAT_TEXT})", text)
            if match:
                return _coerce_float(match.group(1))
        return None

    lat = field(("lat", "latitude"))
    lon = field(("lon", "lng", "longitude"))
    alt = field(("alt", "alt_m", "altitude", "height_m"))
    if lat is None or lon is None or alt is None:
        return None
    fix: dict[str, Any] = {"lat": lat, "lon": lon, "alt": alt}

    timestamp_utc_ns = field(("timestamp_utc_ns", "utc_ns", "time_utc_ns"))
    timestamp_utc_us = field(("timestamp_utc_us", "utc_us", "latest_utc_us", "time_utc_us"))
    if timestamp_utc_ns is not None:
        fix["timestamp_utc_ns"] = int(timestamp_utc_ns)
    elif timestamp_utc_us is not None:
        fix["timestamp_utc_ns"] = int(timestamp_utc_us * 1000)

    yaw_rad = field(("yaw_rad", "heading_rad", "heading_enu_rad"))
    if yaw_rad is not None:
        fix["yaw_rad"] = float(yaw_rad)
    for out_key, names, parser in (
        ("fix_type", ("fix_type", "quality", "fix_quality"), _coerce_int),
        ("num_sats", ("num_sats", "satellites", "sat_count"), _coerce_int),
        ("hdop", ("hdop",), _coerce_float),
        ("speed_mps", ("speed_mps", "speed"), _coerce_float),
        ("course_rad", ("course_rad", "course_true_rad"), _coerce_float),
        ("fix_age_ms", ("fix_age_ms", "latest_fix_age_ms", "age_ms"), _coerce_float),
    ):
        parsed_value = field(names)
        parsed = parser(parsed_value)
        if parsed is not None:
            fix[out_key] = parsed
    course_deg = field(("course_deg", "course_true_deg"))
    if course_deg is not None and "course_rad" not in fix:
        fix["course_rad"] = math.radians(float(course_deg))

    vel_match = re.search(r"(?:vel|velocity|velocity_mps)\s*[:=]\s*\[([^\]]+)\]", text)
    if vel_match:
        velocity = [_coerce_float(item.strip()) for item in vel_match.group(1).split(",")]
        fix["velocity_mps"] = [float(item) for item in velocity if item is not None]
    cov_match = re.search(r"(?:position_covariance|pos_covariance|position_cov|covariance|cov)\s*[:=]\s*\[([^\]]+)\]", text)
    if cov_match:
        covariance = [_coerce_float(item.strip()) for item in cov_match.group(1).split(",")]
        covariance = [float(item) for item in covariance if item is not None]
        if len(covariance) in (4, 6, 9):
            fix["position_covariance"] = covariance
    for out_key, names in (
        ("hacc_m", ("hacc_m", "horizontal_accuracy_m", "accuracy_m", "eph")),
        ("vacc_m", ("vacc_m", "vertical_accuracy_m", "epv")),
    ):
        parsed = field(names)
        if parsed is not None:
            fix[out_key] = float(parsed)
    return fix


def _parse_vector_field(text: str, names: tuple[str, ...]) -> list[float] | None:
    for name in names:
        match = re.search(rf"(?:^|[\s,{{]){re.escape(name)}\s*[:=]\s*\[([^\]]+)\]", text)
        if match:
            values = [_coerce_float(item.strip()) for item in match.group(1).split(",")]
            parsed = [float(item) for item in values if item is not None]
            if parsed:
                return parsed
    return None


def _extract_imu_heading(payload: Any, *, _depth: int = 0) -> dict[str, Any]:
    if _depth > 5 or payload is None:
        return {}
    if isinstance(payload, bytes):
        return _extract_imu_heading(payload.decode("utf-8", errors="replace"), _depth=_depth + 1)
    if isinstance(payload, str):
        euler_deg = _parse_vector_field(payload, ("euler_deg", "euler"))
        heading: dict[str, Any] = {}
        if euler_deg is not None and len(euler_deg) >= 3:
            yaw_deg = float(euler_deg[2])
            heading["ins_euler_deg"] = euler_deg[:3]
            heading["ins_yaw_deg"] = yaw_deg
            heading["ins_yaw_rad"] = math.radians(yaw_deg)
        yaw_rad_match = re.search(rf"(?:^|[\s,{{])(?:ins_yaw_rad|yaw_rad)\s*[:=]\s*({_FLOAT_TEXT})", payload)
        if yaw_rad_match:
            heading["ins_yaw_rad"] = float(yaw_rad_match.group(1))
            heading["ins_yaw_deg"] = math.degrees(float(yaw_rad_match.group(1)))
        yaw_deg_match = re.search(rf"(?:^|[\s,{{])(?:ins_yaw_deg|yaw_deg|heading_deg)\s*[:=]\s*({_FLOAT_TEXT})", payload)
        if yaw_deg_match:
            heading["ins_yaw_deg"] = float(yaw_deg_match.group(1))
            heading["ins_yaw_rad"] = math.radians(float(yaw_deg_match.group(1)))
        return heading
    if isinstance(payload, dict):
        heading: dict[str, Any] = {}
        euler = payload.get("euler_deg") or payload.get("euler")
        if isinstance(euler, (list, tuple)) and len(euler) >= 3:
            euler_values = [_coerce_float(item) for item in euler[:3]]
            if all(item is not None for item in euler_values):
                yaw_deg = float(euler_values[2])
                heading["ins_euler_deg"] = [float(item) for item in euler_values if item is not None]
                heading["ins_yaw_deg"] = yaw_deg
                heading["ins_yaw_rad"] = math.radians(yaw_deg)
        for key in ("ins_yaw_rad", "yaw_rad"):
            value = _coerce_float(payload.get(key))
            if value is not None:
                heading["ins_yaw_rad"] = float(value)
                heading["ins_yaw_deg"] = math.degrees(float(value))
        for key in ("ins_yaw_deg", "yaw_deg", "heading_deg"):
            value = _coerce_float(payload.get(key))
            if value is not None:
                heading["ins_yaw_deg"] = float(value)
                heading["ins_yaw_rad"] = math.radians(float(value))
        for key in ("imu", "orientation", "ins"):
            nested = _extract_imu_heading(payload.get(key), _depth=_depth + 1)
            heading.update({k: v for k, v in nested.items() if k not in heading})
        if heading:
            return heading
        for value in payload.values():
            nested = _extract_imu_heading(value, _depth=_depth + 1)
            if nested:
                return nested
    return {}


def parse_health_gnss(payload: Any, *, _depth: int = 0) -> dict[str, Any] | None:
    if _depth > 6 or payload is None:
        return None
    if isinstance(payload, bytes):
        return parse_health_gnss(payload.decode("utf-8", errors="replace"), _depth=_depth + 1)
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            parsed_json = json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_text_gnss(stripped)
        return parse_health_gnss(parsed_json, _depth=_depth + 1)
    if isinstance(payload, dict):
        direct = _extract_gnss_from_dict(payload)
        if direct is not None:
            return direct
        for key in ("latest_fix", "gnss_latest", "gnss_fix", "fix", "gnss", "gnss_serial", "localization"):
            if key in payload:
                nested = parse_health_gnss(payload[key], _depth=_depth + 1)
                if nested is not None:
                    return nested
        for value in payload.values():
            nested = parse_health_gnss(value, _depth=_depth + 1)
            if nested is not None:
                return nested
        return None
    if isinstance(payload, list):
        for value in payload:
            nested = parse_health_gnss(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


def lla_to_enu(lat: float, lon: float, alt: float, ref_lla: tuple[float, float, float]) -> tuple[float, float, float]:
    enu = ecef_to_enu(
        geodetic_to_ecef(
            np.asarray([lat], dtype=np.float64),
            np.asarray([lon], dtype=np.float64),
            np.asarray([alt], dtype=np.float64),
        ),
        *ref_lla,
    )[0]
    return float(enu[0]), float(enu[1]), float(enu[2])


def fetch_health_gnss(health_url: str, timeout_s: float) -> dict[str, Any]:
    receive_time_utc_ns = time.time_ns()
    with urllib.request.urlopen(health_url, timeout=float(timeout_s)) as response:
        raw = response.read()
    try:
        payload: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        payload = raw
    fix = parse_health_gnss(payload)
    if fix is None:
        raise RuntimeError("health response did not contain GNSS lat/lon/alt")
    fix.update(_extract_imu_heading(payload))
    fix["receive_time_utc_ns"] = receive_time_utc_ns
    fix["receive_monotonic_ns"] = time.monotonic_ns()
    return fix


class LiveGnssSource:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read_fix(self, timeout_s: float) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError


class HealthzGnssSource(LiveGnssSource):
    def __init__(self, health_url: str) -> None:
        self.health_url = health_url

    @property
    def label(self) -> str:
        return self.health_url

    def read_fix(self, timeout_s: float) -> dict[str, Any]:
        fix = fetch_health_gnss(self.health_url, timeout_s)
        fix["gnss_source"] = "healthz"
        return fix


class SerialGnssSource(LiveGnssSource):
    def __init__(self, device: str, baud: int) -> None:
        self.device = device
        self.baud = int(baud)
        self.reader = GnssSerialReader(GnssSerialConfig(device=device, baud=int(baud), max_fixes=4096))

    @property
    def label(self) -> str:
        return f"serial:{self.device}@{self.baud}"

    def start(self) -> None:
        self.reader.start()

    def stop(self) -> None:
        self.reader.stop()

    def read_fix(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        last_snapshot: dict[str, Any] | None = None
        while True:
            snapshot = self.reader.snapshot()
            last_snapshot = snapshot
            latest = snapshot.get("latest_fix")
            if isinstance(latest, dict) and latest.get("lat") is not None:
                fix = dict(latest)
                fix["gnss_source"] = "serial"
                fix["device"] = self.device
                fix["baud"] = self.baud
                fix["fix_age_ms"] = snapshot.get("latest_fix_age_ms")
                fix["receive_time_utc_ns"] = time.time_ns()
                fix["receive_monotonic_ns"] = time.monotonic_ns()
                return fix
            if time.monotonic() >= deadline:
                last_error = None if last_snapshot is None else last_snapshot.get("last_error")
                line_count = None if last_snapshot is None else last_snapshot.get("line_count")
                fix_count = None if last_snapshot is None else last_snapshot.get("fix_count")
                raise RuntimeError(
                    f"serial GNSS fix unavailable device={self.device} baud={self.baud} "
                    f"line_count={line_count} fix_count={fix_count} last_error={last_error}"
                )
            time.sleep(0.02)


def build_live_gnss_source(args: argparse.Namespace) -> LiveGnssSource:
    if args.gnss_source == "serial":
        return SerialGnssSource(str(args.gnss_device), int(args.gnss_baud))
    return HealthzGnssSource(str(args.sensor_health_url))


def actual_gnss_log_row(
    *,
    fix: dict[str, Any],
    enu: tuple[float, float, float],
    ref_lla: tuple[float, float, float],
    source_label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": str(fix.get("gnss_source", "sensor_healthz")),
        "source_label": source_label,
        "receive_time_utc_ns": int(fix.get("receive_time_utc_ns", time.time_ns())),
        "wall_time_unix_s": int(fix.get("receive_time_utc_ns", time.time_ns())) / 1e9,
        "monotonic_ns": int(fix.get("receive_monotonic_ns", time.monotonic_ns())),
        "lat": float(fix["lat"]),
        "lon": float(fix["lon"]),
        "alt": float(fix["alt"]),
        "enu_x_m": float(enu[0]),
        "enu_y_m": float(enu[1]),
        "enu_z_m": float(enu[2]),
        "reference_lla": {
            "lat": float(ref_lla[0]),
            "lon": float(ref_lla[1]),
            "alt": float(ref_lla[2]),
        },
    }
    for key in (
        "timestamp_utc_ns",
        "yaw_rad",
        "ins_euler_deg",
        "ins_yaw_deg",
        "ins_yaw_rad",
        "fix_type",
        "num_sats",
        "hdop",
        "speed_mps",
        "course_rad",
        "fix_age_ms",
        "velocity_mps",
        "position_covariance",
        "hacc_m",
        "vacc_m",
    ):
        if key in fix:
            row[key] = fix[key]
    if extra:
        row.update(extra)
    return row


def _course_to_enu_yaw(course_rad: float) -> float:
    return float(wrap_angles(np.asarray([math.pi / 2.0 - float(course_rad)], dtype=np.float32))[0])


def _wrap_one(angle_rad: float) -> float:
    return float(wrap_angles(np.asarray([float(angle_rad)], dtype=np.float32))[0])


def _angle_lerp(previous_rad: float, current_rad: float, alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    delta = _wrap_one(float(current_rad) - float(previous_rad))
    return _wrap_one(float(previous_rad) + delta * alpha)


def live_fix_speed_mps(fix: dict[str, Any]) -> float | None:
    speed = _coerce_float(fix.get("speed_mps"))
    if speed is not None:
        return abs(float(speed))
    velocity = fix.get("velocity_mps")
    if isinstance(velocity, list) and len(velocity) >= 2:
        vx = _coerce_float(velocity[0])
        vy = _coerce_float(velocity[1])
        vz = _coerce_float(velocity[2]) if len(velocity) >= 3 else 0.0
        if vx is not None and vy is not None:
            return math.sqrt(vx * vx + vy * vy + (0.0 if vz is None else vz * vz))
    return None


class LiveHeadingStabilizer:
    def __init__(
        self,
        *,
        stationary_speed_mps: float,
        stationary_distance_m: float,
        filter_alpha: float,
        enabled: bool,
    ) -> None:
        self.stationary_speed_mps = float(stationary_speed_mps)
        self.stationary_distance_m = float(stationary_distance_m)
        self.filter_alpha = float(filter_alpha)
        self.enabled = bool(enabled)
        self.last_yaw_rad: float | None = None
        self.last_yaw_source: str | None = None
        self.last_moving_yaw_rad: float | None = None
        self.last_stationary: bool | None = None

    def _history_movement_m(self, current_enu: tuple[float, float, float], history: list[dict[str, Any]]) -> float | None:
        if not history:
            return None
        current_xy = np.asarray([float(current_enu[0]), float(current_enu[1])], dtype=np.float64)
        prev_xy = np.asarray([float(history[-1]["enu_x_m"]), float(history[-1]["enu_y_m"])], dtype=np.float64)
        return float(np.linalg.norm(current_xy - prev_xy))

    def is_stationary(self, fix: dict[str, Any], current_enu: tuple[float, float, float], history: list[dict[str, Any]]) -> bool:
        speed = live_fix_speed_mps(fix)
        speed_stationary = speed is not None and speed < self.stationary_speed_mps
        movement = self._history_movement_m(current_enu, history)
        movement_stationary = movement is not None and movement < self.stationary_distance_m
        if speed is not None and movement is not None:
            return bool(speed_stationary and movement_stationary)
        if speed is not None:
            return bool(speed_stationary)
        if movement is not None:
            return bool(movement_stationary)
        return False

    def _last_source_label(self) -> str:
        source = self.last_yaw_source or "last"
        while source.startswith("stationary_hold:"):
            source = source.split(":", 1)[1]
        return source

    def resolve(
        self,
        *,
        raw_yaw_rad: float,
        raw_source: str,
        route_yaw_rad: float,
        fix: dict[str, Any],
        current_enu: tuple[float, float, float],
        history: list[dict[str, Any]],
    ) -> tuple[float, str, bool]:
        if not self.enabled:
            yaw = _wrap_one(raw_yaw_rad)
            self.last_yaw_rad = yaw
            self.last_yaw_source = raw_source
            self.last_moving_yaw_rad = yaw
            self.last_stationary = False
            return yaw, raw_source, False

        stationary = self.is_stationary(fix, current_enu, history)
        if stationary:
            if self.last_moving_yaw_rad is not None:
                yaw = self.last_moving_yaw_rad
                source = f"stationary_hold:{self._last_source_label() or 'last_moving'}"
            elif self.last_yaw_rad is not None:
                yaw = self.last_yaw_rad
                source = f"stationary_hold:{self._last_source_label()}"
            else:
                yaw = _wrap_one(route_yaw_rad)
                source = "stationary_route_tangent"
            self.last_yaw_rad = yaw
            self.last_yaw_source = source
            self.last_stationary = True
            return yaw, source, True

        raw_yaw = _wrap_one(raw_yaw_rad)
        if self.last_yaw_rad is None or self.filter_alpha >= 1.0:
            yaw = raw_yaw
        else:
            yaw = _angle_lerp(self.last_yaw_rad, raw_yaw, self.filter_alpha)
        self.last_yaw_rad = yaw
        self.last_yaw_source = raw_source if yaw == raw_yaw else f"filtered:{raw_source}"
        self.last_moving_yaw_rad = yaw
        self.last_stationary = False
        return yaw, self.last_yaw_source, False


def resolve_live_yaw(
    *,
    fix: dict[str, Any],
    current_enu: tuple[float, float, float],
    history: list[dict[str, Any]],
    route_yaw_rad: float,
    heading_source: str,
    min_history_distance_m: float,
    min_velocity_mps: float,
) -> tuple[float, str]:
    if heading_source in ("auto", "ins") and "ins_yaw_rad" in fix:
        yaw = _coerce_float(fix.get("ins_yaw_rad"))
        if yaw is not None:
            return float(wrap_angles(np.asarray([yaw], dtype=np.float32))[0]), "ins_euler_yaw"

    if heading_source in ("auto", "yaw") and "yaw_rad" in fix:
        yaw = _coerce_float(fix.get("yaw_rad"))
        if yaw is not None:
            return float(wrap_angles(np.asarray([yaw], dtype=np.float32))[0]), "yaw_rad"

    if heading_source in ("auto", "velocity"):
        velocity = fix.get("velocity_mps")
        if isinstance(velocity, list) and len(velocity) >= 2:
            vx = _coerce_float(velocity[0])
            vy = _coerce_float(velocity[1])
            if vx is not None and vy is not None:
                speed = math.hypot(vx, vy)
                if speed >= float(min_velocity_mps):
                    return float(math.atan2(vy, vx)), "velocity_mps_xy"

    if heading_source in ("auto", "history"):
        cx, cy = float(current_enu[0]), float(current_enu[1])
        for prev in reversed(history):
            dx = cx - float(prev["enu_x_m"])
            dy = cy - float(prev["enu_y_m"])
            if math.hypot(dx, dy) >= float(min_history_distance_m):
                return float(math.atan2(dy, dx)), "gnss_history"

    if heading_source in ("auto", "course") and "course_rad" in fix:
        course = _coerce_float(fix.get("course_rad"))
        if course is not None:
            return _course_to_enu_yaw(course), "course_rad_north_cw"

    return float(route_yaw_rad), "route_tangent"


def nearest_route_index(arrays: dict[str, Any], current_enu: tuple[float, float, float]) -> int:
    route_xy = arrays["xyz_enu"][:, :2].astype(np.float64)
    current_xy = np.asarray([current_enu[0], current_enu[1]], dtype=np.float64)
    distances_sq = np.sum((route_xy - current_xy) ** 2, axis=1)
    return int(np.argmin(distances_sq))


def covariance_xy_from_fix(fix: dict[str, Any]) -> np.ndarray | None:
    cov = fix.get("position_covariance")
    if isinstance(cov, list):
        values = [_coerce_float(item) for item in cov]
        values = [float(item) for item in values if item is not None]
        if len(values) == 4:
            return np.asarray([[values[0], values[1]], [values[2], values[3]]], dtype=np.float64)
        if len(values) == 6:
            return np.asarray([[values[0], values[1]], [values[1], values[3]]], dtype=np.float64)
        if len(values) == 9:
            return np.asarray([[values[0], values[1]], [values[3], values[4]]], dtype=np.float64)
    hacc = _coerce_float(fix.get("hacc_m"))
    if hacc is not None:
        variance = float(hacc) * float(hacc)
        return np.asarray([[variance, 0.0], [0.0, variance]], dtype=np.float64)
    return None


def rolling_covariance_xy(history: list[dict[str, Any]], min_points: int = 4) -> np.ndarray | None:
    if len(history) < min_points:
        return None
    xy = np.asarray([[float(row["enu_x_m"]), float(row["enu_y_m"])] for row in history], dtype=np.float64)
    if xy.shape[0] < min_points:
        return None
    cov = np.cov(xy.T)
    if not np.all(np.isfinite(cov)):
        return None
    return cov


def covariance_summary(cov_xy: np.ndarray | None) -> dict[str, Any] | None:
    if cov_xy is None:
        return None
    values, _ = np.linalg.eigh(cov_xy.astype(np.float64))
    values = np.maximum(values, 0.0)
    return {
        "cov_xx_m2": float(cov_xy[0, 0]),
        "cov_xy_m2": float(cov_xy[0, 1]),
        "cov_yy_m2": float(cov_xy[1, 1]),
        "sigma_major_m": float(math.sqrt(float(values[-1]))),
        "sigma_minor_m": float(math.sqrt(float(values[0]))),
    }


class LiveFollowOpenCvUi:
    def __init__(
        self,
        *,
        arrays: dict[str, Any],
        width: int,
        height: int,
        window_name: str,
        history_points: int,
    ) -> None:
        os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
        import cv2  # type: ignore

        self.cv2 = cv2
        build_info = cv2.getBuildInformation()
        if "GUI:                           NONE" in build_info or "GUI: NONE" in build_info:
            raise RuntimeError(
                "OpenCV GUI is unavailable because this Python environment is using opencv-python-headless. "
                "Install a GUI-enabled OpenCV build, for example: "
                "python3 -m pip uninstall -y opencv-python-headless && "
                "python3 -m pip install opencv-python"
            )
        self.width = int(width)
        self.height = int(height)
        self.window_name = window_name
        self.history_points = int(history_points)
        self.route_xy = arrays["xyz_enu"][:, :2].astype(np.float64)
        self.route_min = np.min(self.route_xy, axis=0)
        self.route_max = np.max(self.route_xy, axis=0)
        try:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(self.window_name, self.width, self.height)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV HighGUI window creation failed. The container likely lacks a GUI-enabled OpenCV "
                "or display backend."
            ) from exc

    def update(
        self,
        *,
        payload: dict[str, Any],
        live_enu: tuple[float, float, float],
        history: list[dict[str, Any]],
        reported_cov_xy: np.ndarray | None,
        rolling_cov_xy: np.ndarray | None,
        fix: dict[str, Any],
    ) -> bool:
        cv2 = self.cv2
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (18, 20, 24)
        split = self.width // 2
        left = (24, 64, split - 24, self.height - 48)
        right = (split + 24, 64, self.width - 24, self.height - 48)
        cv2.putText(img, "GLOBAL ROUTE / LIVE GNSS", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
        cv2.putText(img, "LOCAL ROUTE / SENT PATH", (split + 24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
        cv2.rectangle(img, (left[0], left[1]), (left[2], left[3]), (70, 70, 70), 1)
        cv2.rectangle(img, (right[0], right[1]), (right[2], right[3]), (70, 70, 70), 1)

        live_xy = np.asarray([live_enu[0], live_enu[1]], dtype=np.float64)
        target_points = payload.get("packet_points") or []
        local_xy = np.asarray([[float(p["x_m"]), float(p["y_m"])] for p in target_points], dtype=np.float64)
        live_heading_rad = _coerce_float((payload.get("live_gnss") or {}).get("yaw_world_rad"))
        local_route_xy = np.empty((0, 2), dtype=np.float64)
        if live_heading_rad is not None:
            delta_route = self.route_xy - live_xy.reshape(1, 2)
            c0 = math.cos(float(live_heading_rad))
            s0 = math.sin(float(live_heading_rad))
            local_route_all = delta_route @ np.asarray([[c0, -s0], [s0, c0]], dtype=np.float64)
            max_forward = _coerce_float(payload.get("max_forward_m")) or 20.0
            route_mask = (
                (local_route_all[:, 0] >= -2.0)
                & (local_route_all[:, 0] <= max(float(max_forward), 20.0) + 2.0)
                & (np.abs(local_route_all[:, 1]) <= 12.0)
            )
            local_route_xy = local_route_all[route_mask]
        nearest = payload.get("nearest_route") or {}
        nearest_xy = np.asarray(
            [float(nearest.get("enu_x_m", live_xy[0])), float(nearest.get("enu_y_m", live_xy[1]))], dtype=np.float64
        )

        pad = np.asarray([30.0, 30.0], dtype=np.float64)
        global_min = np.minimum(self.route_min, live_xy) - pad
        global_max = np.maximum(self.route_max, live_xy) + pad

        def world_to_px(points: np.ndarray) -> np.ndarray:
            span = np.maximum(global_max - global_min, 1e-3)
            sx = (left[2] - left[0]) / span[0]
            sy = (left[3] - left[1]) / span[1]
            scale = min(sx, sy)
            center = (global_min + global_max) * 0.5
            px_center = np.asarray([(left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5], dtype=np.float64)
            out = np.empty_like(points, dtype=np.int32)
            out[:, 0] = np.round(px_center[0] + (points[:, 0] - center[0]) * scale).astype(np.int32)
            out[:, 1] = np.round(px_center[1] - (points[:, 1] - center[1]) * scale).astype(np.int32)
            return out

        def draw_poly(points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
            if len(points) >= 2:
                cv2.polylines(img, [points.reshape((-1, 1, 2))], False, color, thickness, cv2.LINE_AA)

        route_px = world_to_px(self.route_xy)
        draw_poly(route_px, (0, 0, 255), 3)
        if len(route_px) > 0:
            cv2.circle(img, tuple(route_px[0]), 5, (80, 220, 80), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(route_px[-1]), 5, (80, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "START", tuple(route_px[0] + np.asarray([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 240, 120), 1, cv2.LINE_AA)
            cv2.putText(img, "END", tuple(route_px[-1] + np.asarray([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 255), 1, cv2.LINE_AA)
        if history:
            hist_xy = np.asarray(
                [[float(row["enu_x_m"]), float(row["enu_y_m"])] for row in history[-self.history_points :]],
                dtype=np.float64,
            )
            draw_poly(world_to_px(hist_xy), (0, 180, 255), 2)

        live_px = world_to_px(live_xy.reshape(1, 2))[0]
        nearest_px = world_to_px(nearest_xy.reshape(1, 2))[0]
        cv2.line(img, tuple(live_px), tuple(nearest_px), (100, 100, 100), 1, cv2.LINE_AA)
        route_dist_m = float(nearest.get("distance_to_live_m", 0.0))
        mid_px = ((live_px.astype(np.int32) + nearest_px.astype(np.int32)) // 2).astype(np.int32)
        cv2.putText(img, f"{route_dist_m:.1f}m to route", tuple(mid_px + np.asarray([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(nearest_px), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "nearest route", tuple(nearest_px + np.asarray([8, 14])), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(live_px), 9, (255, 90, 40), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(live_px), 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "ME", tuple(live_px + np.asarray([10, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 180), 2, cv2.LINE_AA)
        if live_heading_rad is not None:
            arrow_len_px = 48
            arrow_end = (
                int(round(live_px[0] + math.cos(live_heading_rad) * arrow_len_px)),
                int(round(live_px[1] - math.sin(live_heading_rad) * arrow_len_px)),
            )
            cv2.arrowedLine(img, tuple(live_px), arrow_end, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cov_for_draw = reported_cov_xy if reported_cov_xy is not None else rolling_cov_xy
        self._draw_covariance(img, live_xy, cov_for_draw, world_to_px, color=(0, 220, 220))

        self._draw_local_panel(img, right, local_xy, local_route_xy)

        cov_source = "reported" if reported_cov_xy is not None else "rolling" if rolling_cov_xy is not None else "none"
        cov_info = covariance_summary(reported_cov_xy if reported_cov_xy is not None else rolling_cov_xy)
        live_gnss = payload.get("live_gnss") or {}
        yaw_deg = _coerce_float(live_gnss.get("yaw_world_deg"))
        raw_yaw_deg = _coerce_float(live_gnss.get("raw_yaw_world_deg"))
        ins_yaw_deg = _coerce_float(fix.get("ins_yaw_deg"))
        texts = [
            f"lat={float(fix['lat']):.7f} lon={float(fix['lon']):.7f} alt={float(fix['alt']):.2f}m",
            f"fix={fix.get('fix_type')} age_ms={fix.get('fix_age_ms')} vel={fix.get('velocity_mps')}",
            f"nearest_idx={nearest.get('index')} route_dist={float(nearest.get('distance_to_live_m', 0.0)):.2f}m",
            f"points={len(target_points)} heading={yaw_deg:.1f}deg source={live_gnss.get('yaw_source')} stationary={live_gnss.get('stationary')} cov={cov_source}"
            if yaw_deg is not None
            else f"points={len(target_points)} heading=NA source={live_gnss.get('yaw_source')} stationary={live_gnss.get('stationary')} cov={cov_source}",
        ]
        if raw_yaw_deg is not None:
            texts.append(f"raw heading={raw_yaw_deg:.1f}deg source={live_gnss.get('raw_yaw_source')}")
        if ins_yaw_deg is not None:
            texts.append(f"INS euler yaw={ins_yaw_deg:.1f}deg")
        if cov_info is not None:
            texts.append(
                f"cov xx={cov_info['cov_xx_m2']:.3f} xy={cov_info['cov_xy_m2']:.3f} yy={cov_info['cov_yy_m2']:.3f} m^2"
            )
            texts.append(f"sigma major/minor={cov_info['sigma_major_m']:.2f}/{cov_info['sigma_minor_m']:.2f}m")
        else:
            texts.append("GNSS covariance: not provided; waiting for rolling history")
        y = self.height - 132
        for text in texts:
            cv2.putText(img, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
            y += 20
        cv2.putText(img, "Press q or ESC to quit", (self.width - 250, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        cv2.putText(img, "global path", (left[0] + 12, left[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "global route segment", (right[0] + 12, right[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "sent local path", (right[0] + 12, right[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 120), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        self.cv2.destroyWindow(self.window_name)

    def _draw_covariance(
        self,
        img: np.ndarray,
        center_xy: np.ndarray,
        cov_xy: np.ndarray | None,
        world_to_px: Any,
        *,
        color: tuple[int, int, int],
    ) -> None:
        if cov_xy is None:
            return
        values, vectors = np.linalg.eigh(cov_xy.astype(np.float64))
        values = np.maximum(values, 0.0)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        major = max(math.sqrt(float(values[0])) * 2.0, 0.1)
        minor = max(math.sqrt(float(values[1])) * 2.0, 0.1)
        angle = math.atan2(float(vectors[1, 0]), float(vectors[0, 0]))
        center_px = world_to_px(center_xy.reshape(1, 2))[0]
        edge_major = world_to_px((center_xy + vectors[:, 0] * major).reshape(1, 2))[0]
        edge_minor = world_to_px((center_xy + vectors[:, 1] * minor).reshape(1, 2))[0]
        axes = (
            max(int(np.linalg.norm(edge_major - center_px)), 1),
            max(int(np.linalg.norm(edge_minor - center_px)), 1),
        )
        self.cv2.ellipse(img, tuple(center_px), axes, -math.degrees(angle), 0, 360, color, 2, self.cv2.LINE_AA)

    def _draw_local_panel(
        self,
        img: np.ndarray,
        box: tuple[int, int, int, int],
        local_xy: np.ndarray,
        local_route_xy: np.ndarray,
    ) -> None:
        cv2 = self.cv2
        x0, y0, x1, y1 = box
        origin_px = np.asarray([(x0 + x1) // 2, y1 - 40], dtype=np.int32)
        max_forward = 20.0
        lateral = 10.0
        sx = (x1 - x0 - 40) / (2.0 * lateral)
        sy = (y1 - y0 - 70) / max_forward
        scale = min(sx, sy)

        def local_to_px(points: np.ndarray) -> np.ndarray:
            out = np.empty_like(points, dtype=np.int32)
            out[:, 0] = np.round(origin_px[0] - points[:, 1] * scale).astype(np.int32)
            out[:, 1] = np.round(origin_px[1] - points[:, 0] * scale).astype(np.int32)
            return out

        axis_color = (220, 220, 220)
        axis_grid_color = (80, 80, 80)
        cv2.line(img, (origin_px[0], y0 + 20), (origin_px[0], y1 - 20), axis_grid_color, 1)
        cv2.line(img, (x0 + 20, origin_px[1]), (x1 - 20, origin_px[1]), axis_grid_color, 1)
        cv2.arrowedLine(img, tuple(origin_px), (origin_px[0], y0 + 24), axis_color, 2, tipLength=0.04)
        cv2.putText(img, "+x forward", (origin_px[0] + 8, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, axis_color, 1)
        cv2.putText(img, "+y left", (x0 + 24, origin_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, axis_color, 1)
        if len(local_route_xy) >= 2:
            px = local_to_px(local_route_xy)
            cv2.polylines(img, [px.reshape((-1, 1, 2))], False, (0, 0, 255), 2, cv2.LINE_AA)
            for pt in px[:: max(len(px) // 12, 1)]:
                cv2.circle(img, tuple(pt), 2, (80, 80, 255), -1, cv2.LINE_AA)
        if len(local_xy) >= 2:
            px = local_to_px(local_xy)
            cv2.polylines(img, [px.reshape((-1, 1, 2))], False, (60, 220, 120), 3, cv2.LINE_AA)
            for pt in px[:: max(len(px) // 12, 1)]:
                cv2.circle(img, tuple(pt), 3, (80, 255, 160), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(origin_px), 7, (40, 80, 255), -1, cv2.LINE_AA)


def build_live_follow_packet(
    *,
    gnss: pd.DataFrame,
    arrays: dict[str, Any],
    live_fix: dict[str, Any],
    live_enu: tuple[float, float, float],
    live_yaw_rad: float,
    live_yaw_source: str,
    nearest_idx: int,
    tx_seq: int,
    plan_points: int,
    plan_dt_s: float,
    max_forward_m: float,
    include_global_points: bool,
) -> dict[str, Any]:
    utc = gnss["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    base_ns = int(utc[0])
    times_s = arrays["times_s"]
    route_time_s = float(times_s[nearest_idx])
    target_times_s = route_time_s + np.arange(plan_points, dtype=np.float64) * float(plan_dt_s)
    world = arrays["xyz_enu"]
    route_enu = np.stack(
        [
            np.interp(target_times_s, times_s, world[:, 0]),
            np.interp(target_times_s, times_s, world[:, 1]),
            np.interp(target_times_s, times_s, world[:, 2]),
        ],
        axis=-1,
    ).astype(np.float32)

    origin = np.asarray(live_enu, dtype=np.float32)
    c0 = math.cos(live_yaw_rad)
    s0 = math.sin(live_yaw_rad)
    rot0 = np.asarray([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    route_local = ((route_enu - origin) @ rot0).astype(np.float32)

    route_yaw_world = interp_angle(times_s, arrays["yaw_world"], target_times_s)
    route_yaw_local = wrap_angles(route_yaw_world - float(live_yaw_rad))
    speed = np.interp(target_times_s, times_s, arrays["speed_mps"]).astype(np.float32)
    yaw_unwrapped = np.unwrap(route_yaw_world.astype(np.float64)).astype(np.float32)
    yaw_rate = np.gradient(yaw_unwrapped, target_times_s).astype(np.float32)
    curvature = np.where(speed > 1e-6, yaw_rate / np.maximum(speed, 1e-6), 0.0).astype(np.float32)

    point_mask = np.ones(route_local.shape[0], dtype=bool)
    if float(max_forward_m) > 0.0:
        point_mask = (route_local[:, 0] >= -1e-3) & (route_local[:, 0] <= float(max_forward_m) + 1e-3)
    route_indices = np.flatnonzero(point_mask)
    if route_indices.size == 0:
        route_indices = np.asarray([0], dtype=np.int64)

    local_xyz_packet = np.concatenate(
        [np.zeros((1, 3), dtype=np.float32), route_local[route_indices].astype(np.float32)],
        axis=0,
    )
    yaw_packet = np.concatenate(
        [np.asarray([0.0], dtype=np.float32), route_yaw_local[route_indices].astype(np.float32)]
    )
    speed_packet = np.concatenate(
        [
            np.asarray([float(speed[route_indices[0]]) if route_indices.size else 0.0], dtype=np.float32),
            speed[route_indices].astype(np.float32),
        ]
    )
    curvature_packet = np.concatenate(
        [
            np.asarray([float(curvature[route_indices[0]]) if route_indices.size else 0.0], dtype=np.float32),
            curvature[route_indices].astype(np.float32),
        ]
    )

    source_t0_ns = int(live_fix.get("timestamp_utc_ns", live_fix.get("receive_time_utc_ns", time.time_ns())))
    tx_time_us = time.time_ns() // 1000
    packet = build_packet_dict(
        tx_seq=tx_seq,
        plan_seq=tx_seq,
        sample_id=int(nearest_idx),
        source_t0_us=source_t0_ns // 1000,
        tx_time_us=tx_time_us,
        coord_mode=COORD_MODE_LOCAL,
        dt_s=float(plan_dt_s),
        x=local_xyz_packet[:, 0],
        y=local_xyz_packet[:, 1],
        yaw=yaw_packet,
        v=speed_packet,
        curvature=curvature_packet,
        flags=FLAG_VALID,
    )

    nearest_row = gnss.iloc[nearest_idx]
    future_points = packet["points"][1:]
    payload: dict[str, Any] = {
        "label": "gnss_live_follow_path",
        "payload_format": "gnss_local_path_json",
        "path_type": "gnss_live_follow_path",
        "udp_mode": "text_json_gnss_live_follow",
        "chunk_id": int(nearest_row["chunk_id"]),
        "sample_id": int(nearest_idx),
        "front_frame_id": int(nearest_idx),
        "t0_utc_ns": source_t0_ns,
        "t0_us": source_t0_ns // 1000,
        "target_offset_s": float((int(utc[nearest_idx]) - base_ns) / 1e9),
        "actual_offset_s": 0.0,
        "plan_dt_s": float(plan_dt_s),
        "requested_plan_points_no_origin": int(plan_points),
        "requested_traj_points_with_origin": int(plan_points + 1),
        "plan_points_no_origin": max(int(packet["header"]["num_points"]) - 1, 0),
        "traj_points_with_origin": int(packet["header"]["num_points"]),
        "max_forward_m": float(max_forward_m) if float(max_forward_m) > 0.0 else None,
        "cropped_by_forward_m": bool(float(max_forward_m) > 0.0),
        "coord_mode": "local",
        "coordinate_note": "ego-local from live GNSS: +x forward, +y left, yaw/curvature positive left",
        "packet_header": packet["header"],
        "packet_points": packet["points"],
        "pred_xyz": local_xyz_packet[1:].astype(float).tolist(),
        "pred_yaw_rad": [float(point["yaw_rad"]) for point in future_points],
        "pred_v_mps": [float(point["v_mps"]) for point in future_points],
        "pred_curvature": [float(point["curvature"]) for point in future_points],
        "source_gnss": {
            "timestamp_utc_ns": source_t0_ns,
            "lat": float(live_fix["lat"]),
            "lon": float(live_fix["lon"]),
            "alt": float(live_fix["alt"]),
            "yaw_world_rad": float(live_yaw_rad),
            "yaw_source": live_yaw_source,
            "enu_x_m": float(live_enu[0]),
            "enu_y_m": float(live_enu[1]),
            "enu_z_m": float(live_enu[2]),
        },
        "live_gnss": {
            "timestamp_utc_ns": source_t0_ns,
            "lat": float(live_fix["lat"]),
            "lon": float(live_fix["lon"]),
            "alt": float(live_fix["alt"]),
            "enu_x_m": float(live_enu[0]),
            "enu_y_m": float(live_enu[1]),
            "enu_z_m": float(live_enu[2]),
            "yaw_world_rad": float(live_yaw_rad),
            "yaw_world_deg": math.degrees(float(live_yaw_rad)),
            "yaw_source": live_yaw_source,
            "ins_yaw_deg": live_fix.get("ins_yaw_deg"),
            "ins_yaw_rad": live_fix.get("ins_yaw_rad"),
        },
        "nearest_route": {
            "index": int(nearest_idx),
            "timestamp_utc_ns": int(utc[nearest_idx]),
            "lat": float(nearest_row["lat"]),
            "lon": float(nearest_row["lon"]),
            "alt": float(nearest_row["alt"]),
            "enu_x_m": float(world[nearest_idx, 0]),
            "enu_y_m": float(world[nearest_idx, 1]),
            "route_s_m": float(arrays["route_s_m"][nearest_idx]),
            "distance_to_live_m": float(
                np.linalg.norm(world[nearest_idx, :2].astype(np.float64) - np.asarray(live_enu[:2], dtype=np.float64))
            ),
        },
        "reference_lla": {
            "lat": float(arrays["ref_lla"][0]),
            "lon": float(arrays["ref_lla"][1]),
            "alt": float(arrays["ref_lla"][2]),
        },
    }
    if include_global_points:
        lat = gnss["lat"].to_numpy(dtype=np.float64)
        lon = gnss["lon"].to_numpy(dtype=np.float64)
        alt = gnss["alt"].to_numpy(dtype=np.float64)
        payload["global_points"] = [
            {
                "t_s": 0.0,
                "source": "live_origin",
                "lat": float(live_fix["lat"]),
                "lon": float(live_fix["lon"]),
                "alt": float(live_fix["alt"]),
            }
        ]
        payload["global_points"].extend(
            {
                "t_s": float(target_times_s[route_idx] - route_time_s),
                "source": "route",
                "source_index": int(route_idx),
                "lat": float(np.interp(target_times_s[route_idx], times_s, lat)),
                "lon": float(np.interp(target_times_s[route_idx], times_s, lon)),
                "alt": float(np.interp(target_times_s[route_idx], times_s, alt)),
            }
            for route_idx in route_indices
        )
    return payload


class ActualGnssLogger:
    def __init__(
        self,
        *,
        health_url: str,
        output_path: Path,
        ref_lla: tuple[float, float, float],
        log_hz: float,
        timeout_s: float,
    ) -> None:
        self.health_url = health_url
        self.output_path = output_path
        self.ref_lla = ref_lla
        self.log_hz = log_hz
        self.timeout_s = timeout_s
        self.rows_written = 0
        self.error_count = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="actual-gnss-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        period_s = 1.0 / max(float(self.log_hz), 1e-6)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as log_file:
            while not self._stop.is_set():
                started_s = time.monotonic()
                receive_time_utc_ns = time.time_ns()
                try:
                    with urllib.request.urlopen(self.health_url, timeout=float(self.timeout_s)) as response:
                        raw = response.read()
                    fix = parse_health_gnss(raw)
                    if fix is None:
                        raise RuntimeError("health response did not contain GNSS lat/lon/alt")
                    enu_x, enu_y, enu_z = lla_to_enu(
                        float(fix["lat"]),
                        float(fix["lon"]),
                        float(fix["alt"]),
                        self.ref_lla,
                    )
                    row = {
                        "source": "sensor_healthz",
                        "health_url": self.health_url,
                        "receive_time_utc_ns": receive_time_utc_ns,
                        "wall_time_unix_s": receive_time_utc_ns / 1e9,
                        "monotonic_ns": time.monotonic_ns(),
                        "lat": float(fix["lat"]),
                        "lon": float(fix["lon"]),
                        "alt": float(fix["alt"]),
                        "enu_x_m": enu_x,
                        "enu_y_m": enu_y,
                        "enu_z_m": enu_z,
                        "reference_lla": {
                            "lat": float(self.ref_lla[0]),
                            "lon": float(self.ref_lla[1]),
                            "alt": float(self.ref_lla[2]),
                        },
                    }
                    for key in (
                        "timestamp_utc_ns",
                        "ins_euler_deg",
                        "ins_yaw_deg",
                        "ins_yaw_rad",
                        "fix_type",
                        "num_sats",
                        "hdop",
                        "speed_mps",
                        "course_rad",
                        "fix_age_ms",
                        "velocity_mps",
                        "position_covariance",
                        "hacc_m",
                        "vacc_m",
                    ):
                        if key in fix:
                            row[key] = fix[key]
                    log_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    log_file.flush()
                    self.rows_written += 1
                    self.last_error = None
                except Exception as exc:
                    self.error_count += 1
                    self.last_error = str(exc)
                elapsed_s = time.monotonic() - started_s
                self._stop.wait(max(period_s - elapsed_s, 0.0))


def write_target_tx_log(log_file: Any, payload: dict[str, Any], host: str, port: int, tx_index: int, byte_count: int) -> None:
    row = {
        "tx_index": int(tx_index),
        "udp_host": host,
        "udp_port": int(port),
        "byte_count": int(byte_count),
        "send_time_utc_ns": time.time_ns(),
        "wall_time_unix_s": time.time(),
        "monotonic_ns": time.monotonic_ns(),
        "chunk_id": int(payload["chunk_id"]),
        "sample_id": int(payload["sample_id"]),
        "t0_utc_ns": int(payload["t0_utc_ns"]),
        "t0_us": int(payload["t0_us"]),
        "source_gnss": payload.get("source_gnss"),
        "live_gnss": payload.get("live_gnss"),
        "nearest_route": payload.get("nearest_route"),
        "reference_lla": payload.get("reference_lla"),
        "packet_header": payload.get("packet_header"),
        "packet_points": payload.get("packet_points"),
    }
    log_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    log_file.flush()


def send_payloads(
    payloads: list[dict[str, Any]],
    host: str,
    port: int,
    send_hz: float,
    loop: bool,
    target_log_path: Path | None = None,
) -> None:
    period_s = 1.0 / max(float(send_hz), 1e-6)
    target = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_log_file = target_log_path.open("w", encoding="utf-8") if target_log_path is not None else None
    tx_index = 0
    try:
        while True:
            for payload in payloads:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                sock.sendto(data, target)
                if target_log_file is not None:
                    write_target_tx_log(target_log_file, payload, host, port, tx_index, len(data))
                print(
                    "sent",
                    f"chunk={payload['chunk_id']}",
                    f"sample={payload['sample_id']}",
                    f"bytes={len(data)}",
                    f"target={host}:{port}",
                    flush=True,
                )
                tx_index += 1
                time.sleep(period_s)
            if not loop:
                break
    finally:
        if target_log_file is not None:
            target_log_file.close()
        sock.close()


def send_live_follow(
    *,
    gnss: pd.DataFrame,
    arrays: dict[str, Any],
    gnss_source: LiveGnssSource,
    fix_timeout_s: float,
    host: str,
    port: int,
    send_hz: float,
    limit: int,
    plan_points: int,
    plan_dt_s: float,
    max_forward_m: float,
    heading_source: str,
    min_heading_distance_m: float,
    min_velocity_heading_mps: float,
    stationary_heading_hold: bool,
    stationary_speed_mps: float,
    stationary_distance_m: float,
    heading_filter_alpha: float,
    include_global_points: bool,
    enable_opencv_ui: bool,
    ui_width: int,
    ui_height: int,
    ui_window_name: str,
    ui_history_points: int,
    target_log_path: Path,
    actual_log_path: Path,
) -> dict[str, Any]:
    period_s = 1.0 / max(float(send_hz), 1e-6)
    target = (host, int(port))
    ref_lla = tuple(float(item) for item in arrays["ref_lla"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    history: list[dict[str, Any]] = []
    tx_count = 0
    error_count = 0
    last_error: str | None = None
    interrupted = False
    ui_quit = False
    ui: LiveFollowOpenCvUi | None = None
    heading_stabilizer = LiveHeadingStabilizer(
        stationary_speed_mps=float(stationary_speed_mps),
        stationary_distance_m=float(stationary_distance_m),
        filter_alpha=float(heading_filter_alpha),
        enabled=bool(stationary_heading_hold),
    )
    if enable_opencv_ui:
        ui = LiveFollowOpenCvUi(
            arrays=arrays,
            width=int(ui_width),
            height=int(ui_height),
            window_name=str(ui_window_name),
            history_points=int(ui_history_points),
        )
    target_log_path.parent.mkdir(parents=True, exist_ok=True)
    actual_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gnss_source.start()
        with target_log_path.open("w", encoding="utf-8") as target_log, actual_log_path.open(
            "w", encoding="utf-8"
        ) as actual_log:
            while limit <= 0 or tx_count < int(limit):
                tick_start_s = time.monotonic()
                try:
                    fix = gnss_source.read_fix(fix_timeout_s)
                    live_enu = lla_to_enu(float(fix["lat"]), float(fix["lon"]), float(fix["alt"]), ref_lla)
                    nearest_idx = nearest_route_index(arrays, live_enu)
                    route_yaw = float(arrays["yaw_world"][nearest_idx])
                    raw_live_yaw, raw_live_yaw_source = resolve_live_yaw(
                        fix=fix,
                        current_enu=live_enu,
                        history=history,
                        route_yaw_rad=route_yaw,
                        heading_source=heading_source,
                        min_history_distance_m=min_heading_distance_m,
                        min_velocity_mps=min_velocity_heading_mps,
                    )
                    live_yaw, live_yaw_source, live_stationary = heading_stabilizer.resolve(
                        raw_yaw_rad=raw_live_yaw,
                        raw_source=raw_live_yaw_source,
                        route_yaw_rad=route_yaw,
                        fix=fix,
                        current_enu=live_enu,
                        history=history,
                    )
                    payload = build_live_follow_packet(
                        gnss=gnss,
                        arrays=arrays,
                        live_fix=fix,
                        live_enu=live_enu,
                        live_yaw_rad=live_yaw,
                        live_yaw_source=live_yaw_source,
                        nearest_idx=nearest_idx,
                        tx_seq=tx_count,
                        plan_points=plan_points,
                        plan_dt_s=plan_dt_s,
                        max_forward_m=max_forward_m,
                        include_global_points=include_global_points,
                    )
                    payload["live_gnss"]["raw_yaw_world_rad"] = float(raw_live_yaw)
                    payload["live_gnss"]["raw_yaw_world_deg"] = math.degrees(float(raw_live_yaw))
                    payload["live_gnss"]["raw_yaw_source"] = raw_live_yaw_source
                    payload["live_gnss"]["stationary"] = bool(live_stationary)
                    payload["live_gnss"]["speed_mps"] = live_fix_speed_mps(fix)
                    pending_history = history + [
                        {
                            "enu_x_m": float(live_enu[0]),
                            "enu_y_m": float(live_enu[1]),
                            "enu_z_m": float(live_enu[2]),
                            "monotonic_ns": int(fix.get("receive_monotonic_ns", time.monotonic_ns())),
                        }
                    ]
                    reported_cov_xy = covariance_xy_from_fix(fix)
                    rolling_cov_xy = rolling_covariance_xy(pending_history[-int(max(ui_history_points, 4)) :])
                    cov_source = "reported" if reported_cov_xy is not None else "rolling" if rolling_cov_xy is not None else "none"
                    cov_info = covariance_summary(reported_cov_xy if reported_cov_xy is not None else rolling_cov_xy)
                    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    sock.sendto(data, target)
                    write_target_tx_log(target_log, payload, host, port, tx_count, len(data))
                    actual_log.write(
                        json.dumps(
                            actual_gnss_log_row(
                                fix=fix,
                                enu=live_enu,
                                ref_lla=ref_lla,
                                source_label=gnss_source.label,
                                extra={
                                    "mode": "live_follow",
                                    "tx_seq": int(tx_count),
                                    "heading_source": live_yaw_source,
                                    "raw_heading_source": raw_live_yaw_source,
                                    "stationary": bool(live_stationary),
                                    "nearest_route_index": int(nearest_idx),
                                    "nearest_route_distance_m": payload["nearest_route"]["distance_to_live_m"],
                                    "covariance_source": cov_source,
                                    "covariance_xy_m2": None
                                    if cov_info is None
                                    else {
                                        "xx": cov_info["cov_xx_m2"],
                                        "xy": cov_info["cov_xy_m2"],
                                        "yy": cov_info["cov_yy_m2"],
                                        "sigma_major_m": cov_info["sigma_major_m"],
                                        "sigma_minor_m": cov_info["sigma_minor_m"],
                                    },
                                },
                            ),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    actual_log.flush()
                    history.append(
                        {
                            "enu_x_m": float(live_enu[0]),
                            "enu_y_m": float(live_enu[1]),
                            "enu_z_m": float(live_enu[2]),
                            "monotonic_ns": int(fix.get("receive_monotonic_ns", time.monotonic_ns())),
                        }
                    )
                    if len(history) > 100:
                        del history[: len(history) - 100]
                    print(
                        "live_sent",
                        f"seq={tx_count}",
                        f"points={payload['packet_header']['num_points']}",
                        f"nearest={nearest_idx}",
                        f"route_dist_m={payload['nearest_route']['distance_to_live_m']:.2f}",
                        f"yaw={live_yaw_source}",
                        f"raw_yaw={raw_live_yaw_source}",
                        f"stationary={int(live_stationary)}",
                        f"heading_deg={math.degrees(float(live_yaw)):.1f}",
                        f"cov={cov_source}",
                        f"sigma={cov_info['sigma_major_m']:.2f}m" if cov_info is not None else "sigma=NA",
                        f"bytes={len(data)}",
                        f"target={host}:{port}",
                        flush=True,
                    )
                    tx_count += 1
                    last_error = None
                    if ui is not None:
                        keep_running = ui.update(
                            payload=payload,
                            live_enu=live_enu,
                            history=history,
                            reported_cov_xy=reported_cov_xy,
                            rolling_cov_xy=rolling_cov_xy,
                            fix=fix,
                        )
                        if not keep_running:
                            ui_quit = True
                            break
                except KeyboardInterrupt:
                    interrupted = True
                    raise
                except Exception as exc:
                    error_count += 1
                    last_error = str(exc)
                    print(f"live_follow_error count={error_count} error={last_error}", flush=True)
                    if limit > 0 and error_count >= int(limit) and tx_count == 0:
                        break

                elapsed_s = time.monotonic() - tick_start_s
                time.sleep(max(period_s - elapsed_s, 0.0))
                if ui_quit:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; stopping live-follow sender", flush=True)
    finally:
        gnss_source.stop()
        if ui is not None:
            ui.close()
        sock.close()

    return {
        "mode": "live_follow",
        "interrupted": interrupted,
        "ui_quit": ui_quit,
        "target_tx_log": str(target_log_path),
        "actual_gnss_log": str(actual_log_path),
        "gnss_source": gnss_source.label,
        "sent_packets": int(tx_count),
        "errors": int(error_count),
        "last_error": last_error,
        "send_hz": float(send_hz),
    }


def main() -> None:
    args = parse_args()
    chunks = parse_chunks(args.chunks)
    ensure_dir(args.output_root)

    if args.global_path_csv is not None:
        windows = pd.DataFrame()
        selected = load_recorded_global_path(args.global_path_csv)
        route_source = str(args.global_path_csv)
    else:
        windows = load_chunk_windows(args.dataset_root, args.camera_name, chunks)
        gnss_path = args.dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
        gnss = select_pose_gnss_rows(pd.read_parquet(gnss_path))
        times_ns = gnss["timestamp_utc_ns"].to_numpy(dtype=np.int64)
        chunk_ids = assign_chunk_ids(times_ns, windows)
        selected = gnss.loc[chunk_ids >= 0].copy()
        selected["chunk_id"] = chunk_ids[chunk_ids >= 0]
        selected = selected.reset_index(drop=True)
        route_source = str(args.dataset_root)
        if selected.empty:
            raise RuntimeError(f"No GNSS rows inside chunk windows {chunks}")

    horizon_ns = int(round(float(args.plan_points) * float(args.plan_dt_s) * 1e9))
    if not args.allow_tail_clamp and not args.live_follow:
        max_t = int(selected["timestamp_utc_ns"].max())
        selected = selected[selected["timestamp_utc_ns"].to_numpy(dtype=np.int64) + horizon_ns <= max_t].copy()
        selected = selected.reset_index(drop=True)
    arrays = compute_world_arrays(selected)
    payload_row_count = len(selected) if args.live_follow else min(len(selected), int(args.limit)) if args.limit > 0 else len(selected)
    payload_rows = selected.head(payload_row_count).copy()
    selected_csv = args.output_root / "gnss_global_selected.csv"
    payload_rows.to_csv(selected_csv, index=False)

    xyz = arrays["xyz_enu"]
    selected_with_enu = payload_rows.copy()
    selected_with_enu["enu_x_m"] = xyz[:payload_row_count, 0]
    selected_with_enu["enu_y_m"] = xyz[:payload_row_count, 1]
    selected_with_enu["enu_z_m"] = xyz[:payload_row_count, 2]
    selected_with_enu.to_csv(args.output_root / "gnss_global_enu_selected.csv", index=False)

    payloads = [
        build_local_packet(
            gnss=selected,
            arrays=arrays,
            row_index=idx,
            tx_seq=idx,
            chunk_id=int(selected["chunk_id"].iloc[idx]),
            plan_points=int(args.plan_points),
            plan_dt_s=float(args.plan_dt_s),
            max_forward_m=float(args.max_forward_m),
            include_global_points=bool(args.include_global_points),
        )
        for idx in range(payload_row_count)
    ]

    jsonl_path = args.output_root / "gnss_local_path_packets.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    latest_payload_path = args.output_root / "latest_udp_payload.json"
    if payloads:
        write_json(latest_payload_path, payloads[0])

    target_tx_log_path = args.output_root / "target_tx_log.jsonl"
    actual_gnss_log_path = args.output_root / "actual_gnss_log.jsonl"
    point_counts = [int(payload["traj_points_with_origin"]) for payload in payloads]
    manifest = {
        "dataset_root": str(args.dataset_root),
        "route_source": route_source,
        "global_path_csv": str(args.global_path_csv) if args.global_path_csv is not None else None,
        "chunks": chunks,
        "camera_name": args.camera_name,
        "chunk_windows": windows.to_dict(orient="records") if not windows.empty else [],
        "gnss_context_rows": int(len(selected)),
        "gnss_payload_rows": int(len(payloads)),
        "requested_plan_points_no_origin": int(args.plan_points),
        "requested_traj_points_with_origin": int(args.plan_points + 1),
        "plan_points_no_origin": max(int(payloads[0]["traj_points_with_origin"]) - 1, 0) if payloads else 0,
        "traj_points_with_origin": int(payloads[0]["traj_points_with_origin"]) if payloads else 0,
        "traj_points_with_origin_min": min(point_counts) if point_counts else 0,
        "traj_points_with_origin_max": max(point_counts) if point_counts else 0,
        "plan_dt_s": float(args.plan_dt_s),
        "max_forward_m": float(args.max_forward_m) if float(args.max_forward_m) > 0.0 else None,
        "cropped_by_forward_m": bool(float(args.max_forward_m) > 0.0),
        "coordinate_note": "ego-local: +x forward, +y left, yaw/curvature positive left",
        "selected_csv": str(selected_csv),
        "selected_enu_csv": str(args.output_root / "gnss_global_enu_selected.csv"),
        "packets_jsonl": str(jsonl_path),
        "latest_udp_payload": str(latest_payload_path),
        "udp_target": f"{args.udp_host}:{args.udp_port}",
        "send": bool(args.send),
        "live_follow": bool(args.live_follow),
        "gnss_source": args.gnss_source,
        "gnss_device": args.gnss_device if args.gnss_source == "serial" else None,
        "gnss_baud": int(args.gnss_baud) if args.gnss_source == "serial" else None,
        "live_send_hz": float(args.send_hz),
        "live_heading_source": args.live_heading_source,
        "stationary_heading_hold": not bool(args.disable_stationary_heading_hold),
        "live_stationary_speed_mps": float(args.live_stationary_speed_mps),
        "live_stationary_distance_m": float(args.live_stationary_distance_m),
        "live_heading_filter_alpha": float(args.live_heading_filter_alpha),
        "opencv_ui": bool(args.enable_opencv_ui),
        "covariance_note": (
            "Uses sensor-reported position covariance/hacc when provided; otherwise displays rolling ENU covariance "
            "from recent live GNSS samples."
        ),
        "target_tx_log": str(target_tx_log_path) if args.send else None,
        "actual_gnss_log": str(actual_gnss_log_path) if args.send and (args.log_actual_gnss or args.live_follow) else None,
        "actual_gnss_source": (
            f"serial:{args.gnss_device}@{args.gnss_baud}"
            if args.gnss_source == "serial"
            else args.sensor_health_url
        )
        if (args.log_actual_gnss or args.live_follow)
        else None,
    }
    write_json(args.output_root / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.log_actual_gnss and not args.send:
        print("--log-actual-gnss was requested, but --send is off; skipping live actual-path logging.", flush=True)
    if args.send:
        if args.live_follow:
            live_gnss_source = build_live_gnss_source(args)
            run_summary = send_live_follow(
                gnss=selected,
                arrays=arrays,
                gnss_source=live_gnss_source,
                fix_timeout_s=float(args.gnss_fix_timeout_s),
                host=str(args.udp_host),
                port=int(args.udp_port),
                send_hz=float(args.send_hz),
                limit=int(args.limit),
                plan_points=int(args.plan_points),
                plan_dt_s=float(args.plan_dt_s),
                max_forward_m=float(args.max_forward_m),
                heading_source=str(args.live_heading_source),
                min_heading_distance_m=float(args.live_min_heading_distance_m),
                min_velocity_heading_mps=float(args.live_min_velocity_heading_mps),
                stationary_heading_hold=not bool(args.disable_stationary_heading_hold),
                stationary_speed_mps=float(args.live_stationary_speed_mps),
                stationary_distance_m=float(args.live_stationary_distance_m),
                heading_filter_alpha=float(args.live_heading_filter_alpha),
                include_global_points=bool(args.include_global_points),
                enable_opencv_ui=bool(args.enable_opencv_ui),
                ui_width=int(args.ui_width),
                ui_height=int(args.ui_height),
                ui_window_name=str(args.ui_window_name),
                ui_history_points=int(args.ui_history_points),
                target_log_path=target_tx_log_path,
                actual_log_path=actual_gnss_log_path,
            )
            write_json(args.output_root / "replay_run_summary.json", run_summary)
            print(json.dumps(run_summary, indent=2, ensure_ascii=False), flush=True)
            return

        actual_logger: ActualGnssLogger | None = None
        if args.log_actual_gnss:
            ref_lla = tuple(float(item) for item in arrays["ref_lla"])
            actual_logger = ActualGnssLogger(
                health_url=str(args.sensor_health_url),
                output_path=actual_gnss_log_path,
                ref_lla=ref_lla,
                log_hz=float(args.actual_log_hz),
                timeout_s=float(args.actual_health_timeout_s),
            )
            actual_logger.start()
            print(f"actual GNSS logging -> {actual_gnss_log_path}", flush=True)
        interrupted = False
        try:
            send_payloads(
                payloads,
                args.udp_host,
                args.udp_port,
                args.send_hz,
                args.loop,
                target_log_path=target_tx_log_path,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("interrupted; stopping replay/logging", flush=True)
        finally:
            if actual_logger is not None:
                actual_logger.stop()
            run_summary = {
                "interrupted": interrupted,
                "target_tx_log": str(target_tx_log_path),
                "actual_gnss_log": str(actual_gnss_log_path) if args.log_actual_gnss else None,
                "actual_gnss_rows": 0 if actual_logger is None else int(actual_logger.rows_written),
                "actual_gnss_errors": 0 if actual_logger is None else int(actual_logger.error_count),
                "actual_gnss_last_error": None if actual_logger is None else actual_logger.last_error,
            }
            write_json(args.output_root / "replay_run_summary.json", run_summary)
            print(json.dumps(run_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
