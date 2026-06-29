# !/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare live UDP Alpamayo paths against a simultaneously recorded GNSS trajectory."
    )
    parser.add_argument(
        "--path-log",
        type=Path,
        required=True,
        help="udp_sent_paths.jsonl from --udp-save-path-log.",
    )
    parser.add_argument(
        "--gnss-csv",
        type=Path,
        required=True,
        help="Raw gnss_gt.csv from scripts/record_live_gnss_global_path.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "live_udp_vs_gnss_analysis",
    )
    parser.add_argument("--max-overlay-paths", type=int, default=36)
    parser.add_argument("--min-path-arc-m", type=float, default=2.0)
    parser.add_argument("--max-latency-age-s", type=float, default=4.0)
    parser.add_argument("--compare-horizons-s", default="0.5,1,2,3")
    parser.add_argument("--ld-m", default="1.5,2,2.5,3,5")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("expected at least one comma-separated value")
    return sorted(set(out))


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def wrap_angle_rad(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception as exc:
                raise RuntimeError(f"failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def read_gnss_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp_ns = as_float(row.get("timestamp_utc_ns"))
            east = as_float(row.get("utm_easting_m"))
            north = as_float(row.get("utm_northing_m"))
            if not (math.isfinite(timestamp_ns) and math.isfinite(east) and math.isfinite(north)):
                continue
            vel_x = as_float(row.get("vel_x_mps"))
            vel_y = as_float(row.get("vel_y_mps"))
            yaw = as_float(row.get("ins_yaw_rad"))
            if not math.isfinite(yaw) and math.isfinite(vel_x) and math.isfinite(vel_y):
                if math.hypot(vel_x, vel_y) > 0.2:
                    yaw = math.atan2(vel_y, vel_x)
            rows.append(
                {
                    "timestamp_ns": int(timestamp_ns),
                    "east": east,
                    "north": north,
                    "yaw": yaw,
                    "speed": math.hypot(vel_x, vel_y) if math.isfinite(vel_x) and math.isfinite(vel_y) else float("nan"),
                    "total_distance_m": as_float(row.get("total_distance_m")),
                }
            )
    if len(rows) < 2:
        raise RuntimeError(f"not enough GNSS rows in {path}")

    rows.sort(key=lambda item: int(item["timestamp_ns"]))
    t_abs_s = np.asarray([int(row["timestamp_ns"]) / 1_000_000_000.0 for row in rows], dtype=np.float64)
    t0_abs_s = float(t_abs_s[0])
    east = np.asarray([float(row["east"]) for row in rows], dtype=np.float64)
    north = np.asarray([float(row["north"]) for row in rows], dtype=np.float64)
    yaw_raw = np.asarray([float(row["yaw"]) for row in rows], dtype=np.float64)
    speed = np.asarray([float(row["speed"]) for row in rows], dtype=np.float64)
    total_distance_m = np.asarray([float(row["total_distance_m"]) for row in rows], dtype=np.float64)

    yaw = yaw_raw.copy()
    finite_yaw = np.isfinite(yaw)
    if np.count_nonzero(finite_yaw) >= 2:
        finite_idx = np.flatnonzero(finite_yaw)
        yaw_unwrapped = np.unwrap(yaw[finite_idx])
        yaw = np.interp(np.arange(yaw.size), finite_idx, yaw_unwrapped)
    elif np.count_nonzero(finite_yaw) == 1:
        yaw[:] = yaw[finite_yaw][0]
    else:
        raise RuntimeError("GNSS CSV has no usable yaw/velocity heading")

    return {
        "t_abs_s": t_abs_s,
        "t_rel_s": t_abs_s - t0_abs_s,
        "t0_abs_s": np.asarray([t0_abs_s], dtype=np.float64),
        "x_m": east - east[0],
        "y_m": north - north[0],
        "yaw_unwrapped_rad": yaw,
        "speed_mps": speed,
        "total_distance_m": total_distance_m,
    }


def interp_series(times_s: np.ndarray, values: np.ndarray, target_s: np.ndarray | float) -> np.ndarray:
    target = np.asarray(target_s, dtype=np.float64)
    out = np.full(target.shape, np.nan, dtype=np.float64)
    mask = (target >= float(times_s[0])) & (target <= float(times_s[-1]))
    if np.any(mask):
        out[mask] = np.interp(target[mask], times_s, values)
    return out


def interp_pose(gnss: dict[str, np.ndarray], target_rel_s: float) -> tuple[float, float, float, float] | None:
    target = np.asarray([float(target_rel_s)], dtype=np.float64)
    x = interp_series(gnss["t_rel_s"], gnss["x_m"], target)[0]
    y = interp_series(gnss["t_rel_s"], gnss["y_m"], target)[0]
    yaw = interp_series(gnss["t_rel_s"], gnss["yaw_unwrapped_rad"], target)[0]
    speed = interp_series(gnss["t_rel_s"], gnss["speed_mps"], target)[0]
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
        return None
    return float(x), float(y), float(wrap_angle_rad(yaw)), float(speed)


def transform_local_to_global(
    local_xy: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    yaw_rad: float,
) -> np.ndarray:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    return np.column_stack(
        [
            origin_x + x * cos_yaw - y * sin_yaw,
            origin_y + x * sin_yaw + y * cos_yaw,
        ]
    )


def global_delta_to_local(delta_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    dx = delta_xy[:, 0]
    dy = delta_xy[:, 1]
    return np.column_stack([dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw])


def nearest_signed_distance_to_polyline(points_xy: np.ndarray, polyline_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0 or polyline_xy.shape[0] < 2:
        return np.full((points_xy.shape[0],), np.nan, dtype=np.float64)
    p0 = polyline_xy[:-1]
    p1 = polyline_xy[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg * seg, axis=1)
    valid_seg = seg_len2 > EPS
    if not np.any(valid_seg):
        return np.full((points_xy.shape[0],), np.nan, dtype=np.float64)
    p0 = p0[valid_seg]
    seg = seg[valid_seg]
    seg_len2 = seg_len2[valid_seg]
    signed = np.full((points_xy.shape[0],), np.nan, dtype=np.float64)
    for idx, point in enumerate(points_xy):
        rel = point[None, :] - p0
        proj_t = np.clip(np.sum(rel * seg, axis=1) / seg_len2, 0.0, 1.0)
        proj = p0 + proj_t[:, None] * seg
        diff = point[None, :] - proj
        dist = np.hypot(diff[:, 0], diff[:, 1])
        best = int(np.argmin(dist))
        cross = seg[best, 0] * diff[best, 1] - seg[best, 1] * diff[best, 0]
        sign = 1.0 if cross >= 0.0 else -1.0
        signed[idx] = sign * float(dist[best])
    return signed


def cumulative_arc(local_xy: np.ndarray) -> np.ndarray:
    if local_xy.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    if local_xy.shape[0] == 1:
        return np.asarray([0.0], dtype=np.float64)
    steps = np.hypot(np.diff(local_xy[:, 0]), np.diff(local_xy[:, 1]))
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(steps)])


def interp_y_at_arc(local_xy: np.ndarray, target_s_m: float) -> float | None:
    arc = cumulative_arc(local_xy)
    if arc.size < 2 or float(arc[-1]) < float(target_s_m):
        return None
    return float(np.interp(float(target_s_m), arc, local_xy[:, 1]))


def packet_local_xy(record: dict[str, Any]) -> np.ndarray:
    points = record.get("packet_points") or []
    xy = np.asarray([[as_float(point.get("x_m")), as_float(point.get("y_m"))] for point in points], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float64)
    return xy[np.isfinite(xy).all(axis=1)]


def analyze(path_rows: list[dict[str, Any]], gnss: dict[str, np.ndarray], horizons_s: list[float], lds_m: list[float], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overlay_items: list[dict[str, Any]] = []
    gt_t0_abs_s = float(gnss["t0_abs_s"][0])

    for idx, record in enumerate(path_rows):
        header = dict(record.get("packet_header") or {})
        tx_time_us = int(header.get("tx_time_us", 0))
        source_t0_us = int(header.get("source_t0_us", 0))
        tx_abs_s = tx_time_us / 1_000_000.0
        tx_rel_s = tx_abs_s - gt_t0_abs_s
        pose = interp_pose(gnss, tx_rel_s)
        local_xy = packet_local_xy(record)
        arc = cumulative_arc(local_xy)
        path_arc_m = float(arc[-1]) if arc.size else 0.0
        dt_s = float(header.get("dt_s", record.get("plan_dt_s", 0.1)) or 0.1)
        actual_offset_s = float(record.get("actual_offset_s", header.get("latency_compensation_age_s", math.nan)))
        latency_age_s = as_float(header.get("latency_compensation_age_s"))
        plan_seq = int(header.get("plan_seq", record.get("sample_id", -1)))

        row: dict[str, Any] = {
            "idx": idx,
            "plan_seq": plan_seq,
            "tx_time_us": tx_time_us,
            "source_t0_us": source_t0_us,
            "tx_rel_s": tx_rel_s,
            "actual_offset_s": actual_offset_s,
            "latency_compensation_age_s": latency_age_s,
            "packet_points": int(local_xy.shape[0]),
            "path_arc_m": path_arc_m,
            "path_x_last_m": float(local_xy[-1, 0]) if local_xy.size else math.nan,
            "path_y_last_m": float(local_xy[-1, 1]) if local_xy.size else math.nan,
            "path_x_max_m": float(np.max(local_xy[:, 0])) if local_xy.size else math.nan,
            "path_y_min_m": float(np.min(local_xy[:, 1])) if local_xy.size else math.nan,
            "path_y_max_m": float(np.max(local_xy[:, 1])) if local_xy.size else math.nan,
            "has_gnss_pose_at_tx": pose is not None,
        }
        for ld_m in lds_m:
            y_at_ld = interp_y_at_arc(local_xy, ld_m)
            row[f"path_y_at_arc_{ld_m:g}m"] = y_at_ld
            row[f"path_pp_kappa_at_arc_{ld_m:g}m"] = None if y_at_ld is None else 2.0 * float(y_at_ld) / max(float(ld_m) ** 2, EPS)

        if pose is None or local_xy.shape[0] < 2:
            row["usable"] = False
            row["skip_reason"] = "no_gnss_pose_or_path"
            rows.append(row)
            continue

        origin_x, origin_y, origin_yaw, origin_speed = pose
        row["gnss_speed_at_tx_mps"] = origin_speed
        row["origin_yaw_rad"] = origin_yaw
        pred_global = transform_local_to_global(local_xy, origin_x=origin_x, origin_y=origin_y, yaw_rad=origin_yaw)
        point_rel_times = tx_rel_s + np.arange(local_xy.shape[0], dtype=np.float64) * dt_s
        gt_x = interp_series(gnss["t_rel_s"], gnss["x_m"], point_rel_times)
        gt_y = interp_series(gnss["t_rel_s"], gnss["y_m"], point_rel_times)
        valid = np.isfinite(gt_x) & np.isfinite(gt_y)
        row["future_points_with_gt"] = int(np.count_nonzero(valid))
        if np.count_nonzero(valid) < 3:
            row["usable"] = False
            row["skip_reason"] = "not_enough_future_gt"
            rows.append(row)
            continue

        gt_global = np.column_stack([gt_x, gt_y])
        delta = gt_global - pred_global
        local_error = global_delta_to_local(delta, origin_yaw)
        gt_local = global_delta_to_local(gt_global - np.asarray([[origin_x, origin_y]], dtype=np.float64), origin_yaw)
        geom_signed_cross = nearest_signed_distance_to_polyline(gt_local, local_xy)
        error_m = np.hypot(delta[:, 0], delta[:, 1])
        for horizon_s in horizons_s:
            hmask = valid & (point_rel_times - tx_rel_s <= horizon_s + 1e-6)
            hidx = int(round(float(horizon_s) / max(dt_s, EPS)))
            key = f"{horizon_s:g}s"
            if np.any(hmask):
                row[f"ade_{key}_m"] = float(np.mean(error_m[hmask]))
                row[f"mean_cross_{key}_m"] = float(np.mean(local_error[hmask, 1]))
                row[f"mean_abs_cross_{key}_m"] = float(np.mean(np.abs(local_error[hmask, 1])))
                row[f"max_error_{key}_m"] = float(np.max(error_m[hmask]))
                geom_vals = geom_signed_cross[hmask]
                geom_vals = geom_vals[np.isfinite(geom_vals)]
                if geom_vals.size:
                    row[f"geom_mean_cross_{key}_m"] = float(np.mean(geom_vals))
                    row[f"geom_mean_abs_cross_{key}_m"] = float(np.mean(np.abs(geom_vals)))
                else:
                    row[f"geom_mean_cross_{key}_m"] = None
                    row[f"geom_mean_abs_cross_{key}_m"] = None
            else:
                row[f"ade_{key}_m"] = None
                row[f"mean_cross_{key}_m"] = None
                row[f"mean_abs_cross_{key}_m"] = None
                row[f"max_error_{key}_m"] = None
                row[f"geom_mean_cross_{key}_m"] = None
                row[f"geom_mean_abs_cross_{key}_m"] = None
            if hidx < error_m.size and valid[hidx]:
                row[f"error_at_{key}_m"] = float(error_m[hidx])
                row[f"cross_at_{key}_m"] = float(local_error[hidx, 1])
                row[f"along_at_{key}_m"] = float(local_error[hidx, 0])
                row[f"geom_cross_at_{key}_m"] = float(geom_signed_cross[hidx])
                row[f"geom_abs_cross_at_{key}_m"] = float(abs(geom_signed_cross[hidx]))
            else:
                row[f"error_at_{key}_m"] = None
                row[f"cross_at_{key}_m"] = None
                row[f"along_at_{key}_m"] = None
                row[f"geom_cross_at_{key}_m"] = None
                row[f"geom_abs_cross_at_{key}_m"] = None

        row["usable"] = bool(
            path_arc_m >= float(args.min_path_arc_m)
            and actual_offset_s <= float(args.max_latency_age_s)
            and np.count_nonzero(valid) >= 3
        )
        if not row["usable"]:
            if path_arc_m < float(args.min_path_arc_m):
                row["skip_reason"] = "short_or_degenerate_path"
            elif actual_offset_s > float(args.max_latency_age_s):
                row["skip_reason"] = "stale_packet_age"
            else:
                row["skip_reason"] = "filtered"
        else:
            row["skip_reason"] = ""

        overlay_items.append(
            {
                "idx": idx,
                "usable": row["usable"],
                "tx_rel_s": tx_rel_s,
                "plan_seq": plan_seq,
                "pred_global": pred_global,
                "valid": valid,
            }
        )
        rows.append(row)

    summary = build_summary(rows, gnss, horizons_s, lds_m)
    return rows, overlay_items, summary


def finite_values(rows: list[dict[str, Any]], key: str, *, usable_only: bool = True) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if usable_only and not row.get("usable"):
            continue
        value = row.get(key)
        if value is None:
            continue
        value_f = as_float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return np.asarray(values, dtype=np.float64)


def stats(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def build_summary(rows: list[dict[str, Any]], gnss: dict[str, np.ndarray], horizons_s: list[float], lds_m: list[float]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("usable")]
    skipped: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("skip_reason") or "")
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1

    summary: dict[str, Any] = {
        "path_packets_total": len(rows),
        "path_packets_usable": len(usable),
        "path_packets_skipped": skipped,
        "gnss_rows": int(gnss["t_rel_s"].size),
        "gnss_duration_s": float(gnss["t_rel_s"][-1] - gnss["t_rel_s"][0]),
        "gnss_distance_m": float(np.nanmax(gnss["total_distance_m"])),
        "tx_rel_range_s": [
            float(np.nanmin([row["tx_rel_s"] for row in rows])) if rows else None,
            float(np.nanmax([row["tx_rel_s"] for row in rows])) if rows else None,
        ],
        "usable_tx_rel_range_s": [
            float(np.nanmin([row["tx_rel_s"] for row in usable])) if usable else None,
            float(np.nanmax([row["tx_rel_s"] for row in usable])) if usable else None,
        ],
        "actual_offset_s": stats(finite_values(rows, "actual_offset_s", usable_only=False)),
        "actual_offset_s_usable": stats(finite_values(rows, "actual_offset_s", usable_only=True)),
        "path_arc_m_usable": stats(finite_values(rows, "path_arc_m", usable_only=True)),
        "gnss_speed_at_tx_mps_usable": stats(finite_values(rows, "gnss_speed_at_tx_mps", usable_only=True)),
    }
    for horizon_s in horizons_s:
        key = f"{horizon_s:g}s"
        summary[f"ade_{key}_m"] = stats(finite_values(rows, f"ade_{key}_m"))
        summary[f"error_at_{key}_m"] = stats(finite_values(rows, f"error_at_{key}_m"))
        summary[f"abs_cross_at_{key}_m"] = stats(np.abs(finite_values(rows, f"cross_at_{key}_m")))
        summary[f"cross_at_{key}_m"] = stats(finite_values(rows, f"cross_at_{key}_m"))
        summary[f"geom_abs_cross_at_{key}_m"] = stats(finite_values(rows, f"geom_abs_cross_at_{key}_m"))
        summary[f"geom_cross_at_{key}_m"] = stats(finite_values(rows, f"geom_cross_at_{key}_m"))
        summary[f"geom_mean_abs_cross_{key}_m"] = stats(finite_values(rows, f"geom_mean_abs_cross_{key}_m"))
    for ld_m in lds_m:
        key = f"{ld_m:g}m"
        summary[f"path_y_at_arc_{key}_usable"] = stats(finite_values(rows, f"path_y_at_arc_{ld_m:g}m"))
        summary[f"path_pp_kappa_at_arc_{key}_usable"] = stats(finite_values(rows, f"path_pp_kappa_at_arc_{ld_m:g}m"))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(
    output_path: Path,
    *,
    rows: list[dict[str, Any]],
    overlay_items: list[dict[str, Any]],
    gnss: dict[str, np.ndarray],
    horizons_s: list[float],
    lds_m: list[float],
    max_overlay_paths: int,
) -> None:
    usable_rows = [row for row in rows if row.get("usable")]
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=150)

    ax = axs[0, 0]
    ax.plot(gnss["x_m"], gnss["y_m"], color="black", linewidth=2.2, label="GNSS actual")
    usable_items = [item for item in overlay_items if item.get("usable")]
    if usable_items:
        selected = np.linspace(0, len(usable_items) - 1, min(len(usable_items), max_overlay_paths), dtype=np.int64)
        cmap = plt.get_cmap("viridis")
        for rank, item_idx in enumerate(selected):
            item = usable_items[int(item_idx)]
            pred_global = item["pred_global"]
            color = cmap(rank / max(len(selected) - 1, 1))
            ax.plot(pred_global[:, 0], pred_global[:, 1], color=color, alpha=0.62, linewidth=1.2)
            ax.scatter(pred_global[0, 0], pred_global[0, 1], s=8, color=color, alpha=0.8)
    ax.set_title("UDP sent paths transformed to GNSS frame")
    ax.set_xlabel("UTM east relative [m]")
    ax.set_ylabel("UTM north relative [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[0, 1]
    x = np.asarray([row["tx_rel_s"] for row in usable_rows], dtype=np.float64)
    for horizon_s in horizons_s:
        key = f"{horizon_s:g}s"
        y = np.asarray([as_float(row.get(f"error_at_{key}_m")) for row in usable_rows], dtype=np.float64)
        if np.isfinite(y).any():
            ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, label=f"error @{key}")
    ax.set_title("Future GNSS error at horizons")
    ax.set_xlabel("tx time since GNSS start [s]")
    ax.set_ylabel("error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[1, 0]
    for horizon_s in horizons_s:
        key = f"{horizon_s:g}s"
        y = np.asarray([as_float(row.get(f"cross_at_{key}_m")) for row in usable_rows], dtype=np.float64)
        if np.isfinite(y).any():
            ax.plot(x, y, marker=".", linewidth=1.0, label=f"cross @{key}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-track error, positive means GT left of sent path")
    ax.set_xlabel("tx time since GNSS start [s]")
    ax.set_ylabel("cross-track [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[1, 1]
    primary_lds = lds_m[:4]
    for ld_m in primary_lds:
        y = np.asarray([as_float(row.get(f"path_y_at_arc_{ld_m:g}m")) for row in usable_rows], dtype=np.float64)
        if np.isfinite(y).any():
            ax.plot(x, y, marker=".", linewidth=1.0, label=f"path y @{ld_m:g}m arc")
    age = np.asarray([as_float(row.get("actual_offset_s")) for row in usable_rows], dtype=np.float64)
    if np.isfinite(age).any():
        ax2 = ax.twinx()
        ax2.plot(x, age, color="gray", alpha=0.35, linewidth=1.0, label="actual offset")
        ax2.set_ylabel("actual offset [s]", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sent path lateral command and packet age")
    ax.set_xlabel("tx time since GNSS start [s]")
    ax.set_ylabel("local y [m] (+ left)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], plot_path: Path) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "NA"
        try:
            value_f = float(value)
        except Exception:
            return str(value)
        if not math.isfinite(value_f):
            return "NA"
        return f"{value_f:.{digits}f}"

    def stat_line(name: str, item: dict[str, Any], unit: str = "") -> str:
        if not item or item.get("count", 0) == 0:
            return f"- {name}: count 0"
        suffix = f" {unit}" if unit else ""
        return (
            f"- {name}: mean {fmt(item.get('mean'))}{suffix}, "
            f"median {fmt(item.get('median'))}{suffix}, p90 {fmt(item.get('p90'))}{suffix}, "
            f"max {fmt(item.get('max'))}{suffix}, n={item.get('count')}"
        )

    stale = [row for row in rows if str(row.get("skip_reason")) == "stale_packet_age"]
    degenerate = [row for row in rows if str(row.get("skip_reason")) == "short_or_degenerate_path"]
    lines = [
        "# Live UDP Path vs GNSS Analysis",
        "",
        f"- Total UDP packets: {summary['path_packets_total']}",
        f"- Usable packets: {summary['path_packets_usable']}",
        f"- Skipped packets: {summary['path_packets_skipped']}",
        f"- GNSS duration: {fmt(summary['gnss_duration_s'])} s",
        f"- GNSS distance: {fmt(summary['gnss_distance_m'])} m",
        f"- TX relative range: {summary['tx_rel_range_s']}",
        f"- Usable TX relative range: {summary['usable_tx_rel_range_s']}",
        f"- Plot: {plot_path}",
        "",
        "## Latency And Path Horizon",
        "",
        stat_line("actual_offset_s all", summary.get("actual_offset_s", {}), "s"),
        stat_line("actual_offset_s usable", summary.get("actual_offset_s_usable", {}), "s"),
        stat_line("path_arc_m usable", summary.get("path_arc_m_usable", {}), "m"),
        stat_line("GNSS speed at TX usable", summary.get("gnss_speed_at_tx_mps_usable", {}), "m/s"),
        "",
        "## Future Tracking Error",
        "",
    ]
    for key in sorted(k for k in summary if k.startswith("error_at_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Cross Track Error", ""])
    for key in sorted(k for k in summary if k.startswith("cross_at_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Geometric Path Error", ""])
    for key in sorted(k for k in summary if k.startswith("geom_abs_cross_at_")):
        lines.append(stat_line(key, summary[key], "m"))
    for key in sorted(k for k in summary if k.startswith("geom_mean_abs_cross_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Sent Path Lateral Shape", ""])
    for key in sorted(k for k in summary if k.startswith("path_y_at_arc_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Notable Packets", ""])
    if stale:
        lines.append("- Stale packets:")
        for row in stale[:5]:
            lines.append(
                f"  - idx {row['idx']} seq {row['plan_seq']}: age {fmt(row['actual_offset_s'])}s, arc {fmt(row['path_arc_m'])}m"
            )
    if degenerate:
        lines.append("- Degenerate/short packets:")
        for row in degenerate[:5]:
            lines.append(
                f"  - idx {row['idx']} seq {row['plan_seq']}: age {fmt(row['actual_offset_s'])}s, arc {fmt(row['path_arc_m'])}m"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    horizons_s = parse_float_list(args.compare_horizons_s)
    lds_m = parse_float_list(args.ld_m)
    path_rows = read_jsonl(args.path_log)
    gnss = read_gnss_csv(args.gnss_csv)
    rows, overlay_items, summary = analyze(path_rows, gnss, horizons_s, lds_m, args)

    metrics_csv = args.output_dir / "udp_vs_gnss_metrics.csv"
    summary_json = args.output_dir / "udp_vs_gnss_summary.json"
    plot_png = args.output_dir / "udp_vs_gnss_analysis.png"
    report_md = args.output_dir / "udp_vs_gnss_report.md"
    write_csv(metrics_csv, rows)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_analysis(
        plot_png,
        rows=rows,
        overlay_items=overlay_items,
        gnss=gnss,
        horizons_s=horizons_s,
        lds_m=lds_m,
        max_overlay_paths=int(args.max_overlay_paths),
    )
    write_markdown_report(report_md, summary, rows, plot_png)
    print(json.dumps({"summary_json": str(summary_json), "metrics_csv": str(metrics_csv), "plot_png": str(plot_png), "report_md": str(report_md), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
