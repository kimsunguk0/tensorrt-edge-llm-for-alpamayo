#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replay_gnss_local_path_udp import (  # noqa: E402
    A_WGS84,
    E2_WGS84,
    HealthzGnssSource,
    SerialGnssSource,
    covariance_summary,
    covariance_xy_from_fix,
    ensure_dir,
    geodetic_to_ecef,
    lla_to_enu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a live GNSS global path for later local-path replay.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/output/live_global_path"),
    )
    parser.add_argument("--output-name", default="global_path")
    parser.add_argument("--gnss-source", choices=("healthz", "serial"), default="healthz")
    parser.add_argument("--sensor-health-url", default="http://127.0.0.1:18080/healthz")
    parser.add_argument("--gnss-device", default="/dev/ttyUSB0")
    parser.add_argument("--gnss-baud", type=int, default=115200)
    parser.add_argument("--record-hz", type=float, default=10.0)
    parser.add_argument("--fix-timeout-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means record until Ctrl+C.")
    parser.add_argument("--limit", type=int, default=0, help="0 means no row limit.")
    parser.add_argument("--min-distance-m", type=float, default=0.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-smooth", action="store_true", help="Skip writing the smoothed final path.")
    parser.add_argument("--smooth-spacing-m", type=float, default=0.5)
    parser.add_argument("--smooth-window-m", type=float, default=2.0)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--smooth-min-step-m", type=float, default=0.05)
    parser.add_argument("--no-review-ui", action="store_true", help="Do not open the OpenCV path review window.")
    parser.add_argument("--review-ui-width", type=int, default=1100)
    parser.add_argument("--review-ui-height", type=int, default=800)
    parser.add_argument("--review-window-name", default="GNSS global path review")
    return parser.parse_args()


def build_source(args: argparse.Namespace) -> HealthzGnssSource | SerialGnssSource:
    if args.gnss_source == "serial":
        return SerialGnssSource(str(args.gnss_device), int(args.gnss_baud))
    return HealthzGnssSource(str(args.sensor_health_url))


def _value(fix: dict[str, Any], key: str) -> Any:
    value = fix.get(key)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return value


def enu_to_lla(points_enu: np.ndarray, ref_lla: tuple[float, float, float]) -> np.ndarray:
    ref_lat_deg, ref_lon_deg, ref_alt_m = ref_lla
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
    xyz = points_enu.astype(np.float64) @ rot + ref_xyz
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    out_lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    out_lat = np.arctan2(z, p * (1.0 - E2_WGS84))
    out_alt = np.zeros_like(out_lat)
    for _ in range(8):
        sin_lat = np.sin(out_lat)
        normal = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat * sin_lat)
        out_alt = p / np.maximum(np.cos(out_lat), 1e-12) - normal
        out_lat = np.arctan2(z, p * (1.0 - E2_WGS84 * normal / np.maximum(normal + out_alt, 1e-12)))
    sin_lat = np.sin(out_lat)
    normal = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat * sin_lat)
    out_alt = p / np.maximum(np.cos(out_lat), 1e-12) - normal
    return np.stack([np.rad2deg(out_lat), np.rad2deg(out_lon), out_alt], axis=-1)


def _moving_average(values: np.ndarray, window: int, iterations: int) -> np.ndarray:
    if values.shape[0] < 3 or window <= 1 or iterations <= 0:
        return values.copy()
    window = min(int(window), int(values.shape[0]))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    weights = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    out = values.astype(np.float64).copy()
    first = out[0].copy()
    last = out[-1].copy()
    for _ in range(int(iterations)):
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.empty_like(out)
        for dim in range(out.shape[1]):
            smoothed[:, dim] = np.convolve(padded[:, dim], weights, mode="valid")
        smoothed[0] = first
        smoothed[-1] = last
        out = smoothed
    return out


def _dedupe_by_distance(points_enu: np.ndarray, timestamps_ns: np.ndarray, min_step_m: float) -> np.ndarray:
    if len(points_enu) <= 2:
        return np.ones(len(points_enu), dtype=bool)
    keep = np.zeros(len(points_enu), dtype=bool)
    keep[0] = True
    last = points_enu[0, :2].astype(np.float64)
    for idx in range(1, len(points_enu) - 1):
        current = points_enu[idx, :2].astype(np.float64)
        timestamp_ok = int(timestamps_ns[idx]) > int(timestamps_ns[np.flatnonzero(keep)[-1]])
        if timestamp_ok and np.linalg.norm(current - last) >= float(min_step_m):
            keep[idx] = True
            last = current
    keep[-1] = True
    return keep


def smooth_global_path_csv(
    *,
    raw_csv_path: Path,
    smoothed_csv_path: Path,
    spacing_m: float,
    window_m: float,
    iterations: int,
    min_step_m: float,
) -> dict[str, Any]:
    raw = pd.read_csv(raw_csv_path)
    raw = raw.dropna(subset=["lat", "lon", "alt"]).copy()
    if raw.empty:
        raise RuntimeError(f"No valid GNSS rows to smooth: {raw_csv_path}")
    if "timestamp_utc_ns" not in raw.columns:
        if "receive_time_utc_ns" in raw.columns:
            raw["timestamp_utc_ns"] = raw["receive_time_utc_ns"]
        else:
            raw["timestamp_utc_ns"] = time.time_ns() + np.arange(len(raw), dtype=np.int64) * 100_000_000
    raw["timestamp_utc_ns"] = pd.to_numeric(raw["timestamp_utc_ns"], errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc_ns"]).copy()
    raw["timestamp_utc_ns"] = raw["timestamp_utc_ns"].astype(np.int64)
    raw = raw.sort_values(["timestamp_utc_ns", "seq"] if "seq" in raw.columns else ["timestamp_utc_ns"])
    raw = raw.drop_duplicates(subset=["timestamp_utc_ns"], keep="last").reset_index(drop=True)
    if len(raw) < 2:
        raise RuntimeError("Need at least two unique GNSS timestamps for a replayable path.")

    ref_lla = (float(raw["lat"].iloc[0]), float(raw["lon"].iloc[0]), float(raw["alt"].iloc[0]))
    raw_enu = np.asarray(
        [
            lla_to_enu(float(row.lat), float(row.lon), float(row.alt), ref_lla)
            for row in raw.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    raw_timestamps = raw["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    keep = _dedupe_by_distance(raw_enu, raw_timestamps, float(min_step_m))
    filtered = raw.loc[keep].reset_index(drop=True)
    filtered_enu = raw_enu[keep]
    filtered_timestamps = raw_timestamps[keep]
    if len(filtered_enu) < 2:
        filtered = raw.iloc[[0, len(raw) - 1]].reset_index(drop=True)
        filtered_enu = raw_enu[[0, len(raw) - 1]]
        filtered_timestamps = raw_timestamps[[0, len(raw) - 1]]

    deltas = np.diff(filtered_enu[:, :2], axis=0)
    route_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))]).astype(np.float64)
    total_m = float(route_s[-1])
    if total_m <= 1e-6:
        smoothed = filtered.copy()
        smoothed["seq"] = np.arange(len(smoothed), dtype=np.int64)
        smoothed["smooth_source"] = "stationary_raw_copy"
        smoothed["smooth_s_m"] = 0.0
        smoothed.to_csv(smoothed_csv_path, index=False)
        return {
            "raw_rows": int(len(raw)),
            "filtered_rows": int(len(filtered)),
            "smoothed_rows": int(len(smoothed)),
            "total_distance_m": total_m,
            "raw_csv": str(raw_csv_path),
            "smoothed_csv": str(smoothed_csv_path),
            "ref_lla": {"lat": ref_lla[0], "lon": ref_lla[1], "alt": ref_lla[2]},
            "smoothed_preview_png": None,
            "raw_enu": raw_enu,
            "smoothed_enu": filtered_enu,
        }

    spacing = max(float(spacing_m), 0.05)
    target_s = np.arange(0.0, total_m + spacing * 0.5, spacing, dtype=np.float64)
    if target_s[-1] < total_m:
        target_s = np.append(target_s, total_m)
    target_s[-1] = total_m
    resampled = np.stack(
        [
            np.interp(target_s, route_s, filtered_enu[:, 0]),
            np.interp(target_s, route_s, filtered_enu[:, 1]),
            np.interp(target_s, route_s, filtered_enu[:, 2]),
        ],
        axis=-1,
    )
    base_timestamp_ns = int(filtered_timestamps[0])
    timestamp_offset_float = np.interp(
        target_s,
        route_s,
        (filtered_timestamps - base_timestamp_ns).astype(np.float64),
    )
    timestamp_ns = base_timestamp_ns + np.maximum.accumulate(np.round(timestamp_offset_float).astype(np.int64))
    if len(timestamp_ns) > 1:
        for idx in range(1, len(timestamp_ns)):
            if timestamp_ns[idx] <= timestamp_ns[idx - 1]:
                timestamp_ns[idx] = timestamp_ns[idx - 1] + 1_000_000

    window_points = int(round(max(float(window_m), 0.0) / spacing))
    if window_points % 2 == 0:
        window_points += 1
    smooth_enu = _moving_average(resampled, window_points, int(iterations))
    smooth_lla = enu_to_lla(smooth_enu, ref_lla)
    out = pd.DataFrame(
        {
            "seq": np.arange(len(smooth_lla), dtype=np.int64),
            "timestamp_utc_ns": timestamp_ns,
            "receive_time_utc_ns": timestamp_ns,
            "lat": smooth_lla[:, 0],
            "lon": smooth_lla[:, 1],
            "alt": smooth_lla[:, 2],
            "chunk_id": 0,
            "smooth_s_m": target_s,
            "enu_x_m": smooth_enu[:, 0],
            "enu_y_m": smooth_enu[:, 1],
            "enu_z_m": smooth_enu[:, 2],
            "smooth_source": "resample_moving_average",
        }
    )
    out.to_csv(smoothed_csv_path, index=False)
    return {
        "raw_rows": int(len(raw)),
        "filtered_rows": int(len(filtered)),
        "smoothed_rows": int(len(out)),
        "total_distance_m": total_m,
        "spacing_m": spacing,
        "window_m": float(window_m),
        "window_points": int(window_points),
        "iterations": int(iterations),
        "raw_csv": str(raw_csv_path),
        "smoothed_csv": str(smoothed_csv_path),
        "ref_lla": {"lat": ref_lla[0], "lon": ref_lla[1], "alt": ref_lla[2]},
        "raw_enu": raw_enu,
        "smoothed_enu": smooth_enu,
    }


def show_path_review(
    *,
    raw_enu: np.ndarray,
    smoothed_enu: np.ndarray,
    stats: dict[str, Any],
    preview_png_path: Path,
    width: int,
    height: int,
    window_name: str,
) -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
    import cv2  # type: ignore

    img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    img[:, :] = (18, 20, 24)
    plot_box = (48, 72, int(width) - 48, int(height) - 150)
    cv2.rectangle(img, (plot_box[0], plot_box[1]), (plot_box[2], plot_box[3]), (70, 70, 70), 1)

    all_xy = np.concatenate([raw_enu[:, :2], smoothed_enu[:, :2]], axis=0)
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    center = (min_xy + max_xy) * 0.5
    pad = span * 0.15 + 2.0
    min_xy = center - span * 0.5 - pad
    max_xy = center + span * 0.5 + pad

    def world_to_px(points: np.ndarray) -> np.ndarray:
        span_xy = np.maximum(max_xy - min_xy, 1e-3)
        sx = (plot_box[2] - plot_box[0]) / span_xy[0]
        sy = (plot_box[3] - plot_box[1]) / span_xy[1]
        scale = min(sx, sy)
        px_center = np.asarray([(plot_box[0] + plot_box[2]) * 0.5, (plot_box[1] + plot_box[3]) * 0.5])
        out = np.empty((len(points), 2), dtype=np.int32)
        out[:, 0] = np.round(px_center[0] + (points[:, 0] - center[0]) * scale).astype(np.int32)
        out[:, 1] = np.round(px_center[1] - (points[:, 1] - center[1]) * scale).astype(np.int32)
        return out

    def draw_path(points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        if len(points) >= 2:
            px = world_to_px(points[:, :2])
            cv2.polylines(img, [px.reshape((-1, 1, 2))], False, color, thickness, cv2.LINE_AA)
            for pt in px[:: max(len(px) // 18, 1)]:
                cv2.circle(img, tuple(pt), 3, color, -1, cv2.LINE_AA)

    draw_path(raw_enu, (0, 0, 255), 2)
    draw_path(smoothed_enu, (60, 230, 120), 3)
    for label, point, color in (
        ("START", smoothed_enu[0, :2], (120, 240, 120)),
        ("END", smoothed_enu[-1, :2], (120, 120, 255)),
    ):
        px = world_to_px(point.reshape(1, 2))[0]
        cv2.circle(img, tuple(px), 7, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, tuple(px + np.asarray([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    cv2.putText(img, "GLOBAL PATH REVIEW", (48, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(img, "raw GNSS", (int(width) - 260, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "smoothed final path", (int(width) - 260, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 120), 2, cv2.LINE_AA)
    texts = [
        f"raw rows={stats.get('raw_rows')} filtered={stats.get('filtered_rows')} smoothed={stats.get('smoothed_rows')}",
        f"distance={float(stats.get('total_distance_m', 0.0)):.2f}m spacing={float(stats.get('spacing_m', 0.0)):.2f}m window={float(stats.get('window_m', 0.0)):.2f}m iter={stats.get('iterations')}",
        f"final CSV: {stats.get('smoothed_csv')}",
        "Press q, ESC, or ENTER to close",
    ]
    y = int(height) - 108
    for text in texts:
        cv2.putText(img, text, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24

    cv2.imwrite(str(preview_png_path), img)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(width), int(height))
        cv2.imshow(window_name, img)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (27, 10, 13, ord("q")):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    csv_path = args.output_root / f"{args.output_name}.csv"
    jsonl_path = args.output_root / f"{args.output_name}.jsonl"
    manifest_path = args.output_root / f"{args.output_name}_manifest.json"
    smoothed_csv_path = args.output_root / f"{args.output_name}_smoothed.csv"
    preview_png_path = args.output_root / f"{args.output_name}_smoothed_preview.png"
    source = build_source(args)
    period_s = 1.0 / max(float(args.record_hz), 1e-6)
    mode = "a" if args.append and csv_path.exists() else "w"
    write_header = mode == "w"
    fields = [
        "seq",
        "timestamp_utc_ns",
        "receive_time_utc_ns",
        "lat",
        "lon",
        "alt",
        "distance_from_prev_m",
        "total_distance_m",
        "fix_type",
        "fix_age_ms",
        "yaw_deg",
        "yaw_rad",
        "ins_yaw_deg",
        "ins_yaw_rad",
        "vel_x_mps",
        "vel_y_mps",
        "vel_z_mps",
        "utm_easting_m",
        "utm_northing_m",
        "utm_alt_m",
        "utm_zone",
        "utm_northp",
        "hacc_m",
        "vacc_m",
        "cov_xx_m2",
        "cov_xy_m2",
        "cov_yy_m2",
        "cov_zz_m2",
        "sigma_major_m",
        "sigma_minor_m",
        "gnss_source",
        "source_label",
    ]
    rows_written = 0
    errors = 0
    last_error: str | None = None
    interrupted = False
    ref_lla: tuple[float, float, float] | None = None
    last_enu: tuple[float, float, float] | None = None
    total_distance_m = 0.0
    started_s = time.monotonic()
    source.start()
    try:
        with csv_path.open(mode, newline="", encoding="utf-8") as csv_file, jsonl_path.open(mode, encoding="utf-8") as jsonl:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            if write_header:
                writer.writeheader()
            while True:
                tick_s = time.monotonic()
                if args.duration_s > 0 and tick_s - started_s >= float(args.duration_s):
                    break
                if args.limit > 0 and rows_written >= int(args.limit):
                    break
                try:
                    fix = source.read_fix(float(args.fix_timeout_s))
                    timestamp_utc_ns = int(fix.get("timestamp_utc_ns") or fix.get("receive_time_utc_ns") or time.time_ns())
                    lat = float(fix["lat"])
                    lon = float(fix["lon"])
                    alt = float(fix["alt"])
                    if ref_lla is None:
                        ref_lla = (lat, lon, alt)
                    enu = lla_to_enu(lat, lon, alt, ref_lla)
                    distance_from_prev = 0.0
                    if last_enu is not None:
                        distance_from_prev = math.hypot(float(enu[0] - last_enu[0]), float(enu[1] - last_enu[1]))
                        if distance_from_prev < float(args.min_distance_m):
                            time.sleep(max(period_s - (time.monotonic() - tick_s), 0.0))
                            continue
                    total_distance_m += distance_from_prev
                    last_enu = enu
                    cov = covariance_summary(covariance_xy_from_fix(fix))
                    velocity = fix.get("velocity_mps") if isinstance(fix.get("velocity_mps"), list) else []
                    utm = fix.get("gnss_utm") if isinstance(fix.get("gnss_utm"), list) else []
                    cov_full = fix.get("gnss_covariance_enu_m2") if isinstance(fix.get("gnss_covariance_enu_m2"), list) else []
                    row = {
                        "seq": rows_written,
                        "timestamp_utc_ns": timestamp_utc_ns,
                        "receive_time_utc_ns": int(fix.get("receive_time_utc_ns") or time.time_ns()),
                        "lat": lat,
                        "lon": lon,
                        "alt": alt,
                        "distance_from_prev_m": distance_from_prev,
                        "total_distance_m": total_distance_m,
                        "fix_type": _value(fix, "fix_type"),
                        "fix_age_ms": _value(fix, "fix_age_ms"),
                        "yaw_deg": _value(fix, "yaw_deg"),
                        "yaw_rad": _value(fix, "yaw_rad"),
                        "ins_yaw_deg": _value(fix, "ins_yaw_deg"),
                        "ins_yaw_rad": _value(fix, "ins_yaw_rad"),
                        "vel_x_mps": velocity[0] if len(velocity) > 0 else None,
                        "vel_y_mps": velocity[1] if len(velocity) > 1 else None,
                        "vel_z_mps": velocity[2] if len(velocity) > 2 else None,
                        "utm_easting_m": utm[0] if len(utm) > 0 else None,
                        "utm_northing_m": utm[1] if len(utm) > 1 else None,
                        "utm_alt_m": utm[2] if len(utm) > 2 else None,
                        "utm_zone": _value(fix, "gnss_utm_zone"),
                        "utm_northp": _value(fix, "gnss_utm_northp"),
                        "hacc_m": _value(fix, "hacc_m"),
                        "vacc_m": _value(fix, "vacc_m"),
                        "cov_xx_m2": None if cov is None else cov["cov_xx_m2"],
                        "cov_xy_m2": None if cov is None else cov["cov_xy_m2"],
                        "cov_yy_m2": None if cov is None else cov["cov_yy_m2"],
                        "cov_zz_m2": cov_full[8] if len(cov_full) >= 9 else None,
                        "sigma_major_m": None if cov is None else cov["sigma_major_m"],
                        "sigma_minor_m": None if cov is None else cov["sigma_minor_m"],
                        "gnss_source": fix.get("gnss_source", args.gnss_source),
                        "source_label": source.label,
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    jsonl.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    jsonl.flush()
                    print(
                        "recorded",
                        f"seq={rows_written}",
                        f"lat={lat:.7f}",
                        f"lon={lon:.7f}",
                        f"alt={alt:.2f}",
                        f"step_m={distance_from_prev:.2f}",
                        f"total_m={total_distance_m:.2f}",
                        flush=True,
                    )
                    rows_written += 1
                    last_error = None
                except KeyboardInterrupt:
                    interrupted = True
                    raise
                except Exception as exc:
                    errors += 1
                    last_error = str(exc)
                    print(f"record_error count={errors} error={last_error}", flush=True)
                time.sleep(max(period_s - (time.monotonic() - tick_s), 0.0))
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; stopping global path recording", flush=True)
    finally:
        source.stop()
        smooth_result: dict[str, Any] | None = None
        postprocess_error: str | None = None
        review_ui_error: str | None = None
        if not args.no_smooth:
            try:
                smooth_result = smooth_global_path_csv(
                    raw_csv_path=csv_path,
                    smoothed_csv_path=smoothed_csv_path,
                    spacing_m=float(args.smooth_spacing_m),
                    window_m=float(args.smooth_window_m),
                    iterations=int(args.smooth_iterations),
                    min_step_m=float(args.smooth_min_step_m),
                )
                raw_enu = smooth_result.pop("raw_enu", None)
                smoothed_enu = smooth_result.pop("smoothed_enu", None)
                smooth_result["smoothed_preview_png"] = str(preview_png_path)
                print(
                    "smoothed_global_path",
                    f"raw_rows={smooth_result['raw_rows']}",
                    f"smoothed_rows={smooth_result['smoothed_rows']}",
                    f"distance_m={smooth_result['total_distance_m']:.2f}",
                    f"csv={smoothed_csv_path}",
                    flush=True,
                )
                if not args.no_review_ui and raw_enu is not None and smoothed_enu is not None:
                    try:
                        show_path_review(
                            raw_enu=raw_enu,
                            smoothed_enu=smoothed_enu,
                            stats=smooth_result,
                            preview_png_path=preview_png_path,
                            width=int(args.review_ui_width),
                            height=int(args.review_ui_height),
                            window_name=str(args.review_window_name),
                        )
                    except Exception as exc:
                        review_ui_error = str(exc)
                        print(f"review_ui_error error={review_ui_error}", flush=True)
                    except KeyboardInterrupt:
                        review_ui_error = "interrupted while review UI was open"
                        print(f"review_ui_error error={review_ui_error}", flush=True)
            except Exception as exc:
                postprocess_error = str(exc)
                print(f"smooth_error error={postprocess_error}", flush=True)
        manifest = {
            "csv_path": str(csv_path),
            "jsonl_path": str(jsonl_path),
            "smoothed_csv_path": str(smoothed_csv_path) if smooth_result is not None else None,
            "recommended_replay_csv": str(smoothed_csv_path) if smooth_result is not None else str(csv_path),
            "smoothed_preview_png": str(preview_png_path) if smooth_result is not None else None,
            "smooth_result": smooth_result,
            "postprocess_error": postprocess_error,
            "review_ui_error": review_ui_error,
            "gnss_source": source.label,
            "record_hz": float(args.record_hz),
            "rows_written_this_run": int(rows_written),
            "errors": int(errors),
            "last_error": last_error,
            "interrupted": interrupted,
            "total_distance_m": float(total_distance_m),
            "reference_lla": None
            if ref_lla is None
            else {"lat": float(ref_lla[0]), "lon": float(ref_lla[1]), "alt": float(ref_lla[2])},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
