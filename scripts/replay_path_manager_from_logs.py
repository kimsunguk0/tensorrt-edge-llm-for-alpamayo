# !/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner_live.live_path_manager import _path_arc, _project_to_path  # noqa: E402
from scripts.analyze_live_udp_path_vs_gnss import (  # noqa: E402
    as_float,
    cumulative_arc,
    global_delta_to_local,
    interp_pose,
    interp_series,
    nearest_signed_distance_to_polyline,
    packet_local_xy,
    parse_float_list,
    read_gnss_csv,
    read_jsonl,
    stats,
    transform_local_to_global,
)


EPS = 1e-9


@dataclass(slots=True)
class ReplayPlan:
    idx: int
    plan_seq: int
    tx_rel_s: float
    tx_abs_s: float
    source_t0_abs_s: float
    actual_offset_s: float
    dt_s: float
    path_arc_m: float
    global_xy: np.ndarray
    local_xy_at_tx: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay for the 10 Hz GNSS-projection path-manager mode using "
            "previous udp_sent_paths.jsonl and GNSS logs."
        )
    )
    parser.add_argument("--path-log", type=Path, required=True)
    parser.add_argument("--gnss-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--max-plan-age-s", type=float, default=3.0)
    parser.add_argument("--min-plan-arc-m", type=float, default=2.0)
    parser.add_argument("--min-remaining-distance-m", type=float, default=2.0)
    parser.add_argument("--max-projection-distance-m", type=float, default=5.0)
    parser.add_argument("--output-points", type=int, default=65)
    parser.add_argument("--compare-horizons-s", default="0.5,1,2,3")
    parser.add_argument("--ld-m", default="4,6,8")
    parser.add_argument("--max-overlay-paths", type=int, default=48)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pad_or_trim_xy(values: np.ndarray, output_points: int) -> np.ndarray:
    target = max(int(output_points), 2)
    if values.shape[0] >= target:
        return values[:target]
    if values.shape[0] == 0:
        return np.zeros((target, 2), dtype=np.float64)
    pad = np.repeat(values[-1:], target - values.shape[0], axis=0)
    return np.concatenate([values, pad], axis=0)


def interp_y_at_arc(local_xy: np.ndarray, target_s_m: float) -> float | None:
    arc = cumulative_arc(local_xy)
    if arc.size < 2 or float(arc[-1]) + EPS < float(target_s_m):
        return None
    return float(np.interp(float(target_s_m), arc, local_xy[:, 1]))


def build_replay_plans(
    path_rows: list[dict[str, Any]],
    gnss: dict[str, np.ndarray],
    *,
    min_plan_arc_m: float,
) -> tuple[list[ReplayPlan], list[dict[str, Any]]]:
    gt_t0_abs_s = float(gnss["t0_abs_s"][0])
    accepted: list[ReplayPlan] = []
    plan_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(path_rows):
        header = dict(record.get("packet_header") or {})
        tx_time_us = int(header.get("tx_time_us", 0))
        source_t0_us = int(header.get("source_t0_us", record.get("t0_us", 0)))
        if tx_time_us <= 0:
            plan_rows.append({"idx": idx, "accepted": False, "reason": "missing_tx_time_us"})
            continue
        tx_abs_s = tx_time_us / 1_000_000.0
        tx_rel_s = tx_abs_s - gt_t0_abs_s
        pose = interp_pose(gnss, tx_rel_s)
        local_xy = packet_local_xy(record)
        arc = cumulative_arc(local_xy)
        path_arc_m = float(arc[-1]) if arc.size else 0.0
        plan_seq = int(header.get("plan_seq", record.get("sample_id", idx)))
        actual_offset_s = float(record.get("actual_offset_s", header.get("latency_compensation_age_s", math.nan)))
        dt_s = float(header.get("dt_s", record.get("plan_dt_s", 0.1)) or 0.1)
        row = {
            "idx": idx,
            "plan_seq": plan_seq,
            "tx_rel_s": tx_rel_s,
            "source_t0_rel_s": source_t0_us / 1_000_000.0 - gt_t0_abs_s,
            "actual_offset_s": actual_offset_s,
            "path_arc_m": path_arc_m,
            "packet_points": int(local_xy.shape[0]),
        }
        if pose is None:
            row.update({"accepted": False, "reason": "no_gnss_pose_at_tx"})
            plan_rows.append(row)
            continue
        if local_xy.shape[0] < 2 or path_arc_m < float(min_plan_arc_m):
            row.update({"accepted": False, "reason": "short_or_degenerate_path"})
            plan_rows.append(row)
            continue

        origin_x, origin_y, origin_yaw, _origin_speed = pose
        global_xy = transform_local_to_global(
            local_xy,
            origin_x=origin_x,
            origin_y=origin_y,
            yaw_rad=origin_yaw,
        )
        plan = ReplayPlan(
            idx=idx,
            plan_seq=plan_seq,
            tx_rel_s=tx_rel_s,
            tx_abs_s=tx_abs_s,
            source_t0_abs_s=source_t0_us / 1_000_000.0,
            actual_offset_s=actual_offset_s,
            dt_s=dt_s,
            path_arc_m=path_arc_m,
            global_xy=global_xy.astype(np.float64),
            local_xy_at_tx=local_xy.astype(np.float64),
        )
        accepted.append(plan)
        row.update({"accepted": True, "reason": ""})
        plan_rows.append(row)

    accepted.sort(key=lambda item: item.tx_rel_s)
    return accepted, plan_rows


def replay(
    plans: list[ReplayPlan],
    gnss: dict[str, np.ndarray],
    *,
    rate_hz: float,
    max_plan_age_s: float,
    min_remaining_distance_m: float,
    max_projection_distance_m: float,
    output_points: int,
    horizons_s: list[float],
    lds_m: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plans:
        return [], []

    gt_t0_abs_s = float(gnss["t0_abs_s"][0])
    start_s = max(float(gnss["t_rel_s"][0]), float(plans[0].tx_rel_s))
    end_s = min(float(gnss["t_rel_s"][-1]), float(plans[-1].tx_rel_s + max_plan_age_s))
    if end_s <= start_s:
        return [], []

    period_s = 1.0 / max(float(rate_hz), EPS)
    tick_times = np.arange(start_s, end_s + period_s * 0.5, period_s, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    current: ReplayPlan | None = None
    plan_idx = 0

    for tick_idx, tick_rel_s in enumerate(tick_times):
        while plan_idx < len(plans) and plans[plan_idx].tx_rel_s <= tick_rel_s + 1e-6:
            current = plans[plan_idx]
            plan_idx += 1

        pose = interp_pose(gnss, float(tick_rel_s))
        row: dict[str, Any] = {
            "tick_idx": tick_idx,
            "tick_rel_s": float(tick_rel_s),
            "published": False,
            "plan_seq": None if current is None else current.plan_seq,
            "plan_log_idx": None if current is None else current.idx,
        }
        if current is None:
            row["skip_reason"] = "no_plan"
            rows.append(row)
            continue
        if pose is None:
            row["skip_reason"] = "no_gnss_pose"
            rows.append(row)
            continue

        tick_abs_s = gt_t0_abs_s + float(tick_rel_s)
        plan_age_s = max(0.0, tick_abs_s - current.source_t0_abs_s)
        row["plan_age_s"] = plan_age_s
        row["plan_tx_age_s"] = max(0.0, float(tick_rel_s) - current.tx_rel_s)
        row["source_actual_offset_s"] = current.actual_offset_s
        row["source_path_arc_m"] = current.path_arc_m
        if plan_age_s > float(max_plan_age_s):
            row["skip_reason"] = "plan_age_exceeded"
            rows.append(row)
            continue

        origin_x, origin_y, origin_yaw, origin_speed = pose
        plan_arc = _path_arc(current.global_xy)
        projection = _project_to_path(np.asarray([origin_x, origin_y], dtype=np.float64), current.global_xy, plan_arc)
        if projection is None:
            row["skip_reason"] = "projection_unavailable"
            rows.append(row)
            continue

        remaining_distance_m = float(plan_arc[-1] - projection.arc_m)
        row["projection_distance_m"] = float(projection.distance_m)
        row["projection_arc_m"] = float(projection.arc_m)
        row["remaining_distance_m"] = remaining_distance_m
        row["gnss_speed_mps"] = origin_speed
        if projection.distance_m > float(max_projection_distance_m):
            row["skip_reason"] = "projection_distance_exceeded"
            rows.append(row)
            continue
        if remaining_distance_m < float(min_remaining_distance_m):
            row["skip_reason"] = "remaining_distance_short"
            rows.append(row)
            continue

        idx = projection.segment_idx
        remaining_global = np.concatenate([projection.point_xy.reshape(1, 2), current.global_xy[idx + 1 :]], axis=0)
        local_xy = global_delta_to_local(
            remaining_global - np.asarray([[origin_x, origin_y]], dtype=np.float64),
            origin_yaw,
        )
        local_xy = pad_or_trim_xy(local_xy, output_points)
        out_arc = cumulative_arc(local_xy)
        row["published"] = True
        row["skip_reason"] = ""
        row["output_points"] = int(local_xy.shape[0])
        row["output_arc_m"] = float(out_arc[-1]) if out_arc.size else 0.0
        row["local_start_x_m"] = float(local_xy[0, 0])
        row["local_start_y_m"] = float(local_xy[0, 1])
        row["local_end_x_m"] = float(local_xy[-1, 0])
        row["local_end_y_m"] = float(local_xy[-1, 1])
        for ld_m in lds_m:
            y_at_ld = interp_y_at_arc(local_xy, ld_m)
            row[f"path_y_at_arc_{ld_m:g}m"] = y_at_ld
            row[f"path_pp_kappa_at_arc_{ld_m:g}m"] = None if y_at_ld is None else 2.0 * y_at_ld / max(ld_m * ld_m, EPS)

        point_rel_times = float(tick_rel_s) + np.arange(local_xy.shape[0], dtype=np.float64) * current.dt_s
        gt_x = interp_series(gnss["t_rel_s"], gnss["x_m"], point_rel_times)
        gt_y = interp_series(gnss["t_rel_s"], gnss["y_m"], point_rel_times)
        valid = np.isfinite(gt_x) & np.isfinite(gt_y)
        gt_global = np.column_stack([gt_x, gt_y])
        sent_global = transform_local_to_global(
            local_xy,
            origin_x=origin_x,
            origin_y=origin_y,
            yaw_rad=origin_yaw,
        )
        if np.count_nonzero(valid) >= 3:
            delta = gt_global - sent_global
            local_error = global_delta_to_local(delta, origin_yaw)
            gt_local = global_delta_to_local(gt_global - np.asarray([[origin_x, origin_y]], dtype=np.float64), origin_yaw)
            geom_signed_cross = nearest_signed_distance_to_polyline(gt_local[valid], local_xy)
            full_geom_cross = np.full((local_xy.shape[0],), np.nan, dtype=np.float64)
            full_geom_cross[valid] = geom_signed_cross
            error_m = np.hypot(delta[:, 0], delta[:, 1])
            for horizon_s in horizons_s:
                hkey = f"{horizon_s:g}s"
                hmask = valid & (point_rel_times - float(tick_rel_s) <= horizon_s + 1e-6)
                hidx = int(round(float(horizon_s) / max(current.dt_s, EPS)))
                if np.any(hmask):
                    row[f"ade_{hkey}_m"] = float(np.mean(error_m[hmask]))
                    row[f"mean_cross_{hkey}_m"] = float(np.mean(local_error[hmask, 1]))
                    row[f"mean_abs_cross_{hkey}_m"] = float(np.mean(np.abs(local_error[hmask, 1])))
                    geom_vals = full_geom_cross[hmask]
                    geom_vals = geom_vals[np.isfinite(geom_vals)]
                    row[f"geom_mean_abs_cross_{hkey}_m"] = (
                        float(np.mean(np.abs(geom_vals))) if geom_vals.size else None
                    )
                if hidx < error_m.size and valid[hidx]:
                    row[f"error_at_{hkey}_m"] = float(error_m[hidx])
                    row[f"cross_at_{hkey}_m"] = float(local_error[hidx, 1])
                    row[f"along_at_{hkey}_m"] = float(local_error[hidx, 0])
                    row[f"geom_cross_at_{hkey}_m"] = float(full_geom_cross[hidx])
                    row[f"geom_abs_cross_at_{hkey}_m"] = float(abs(full_geom_cross[hidx]))

        overlays.append(
            {
                "tick_idx": tick_idx,
                "tick_rel_s": float(tick_rel_s),
                "plan_seq": current.plan_seq,
                "sent_global": sent_global,
            }
        )
        rows.append(row)

    return rows, overlays


def finite_values(rows: list[dict[str, Any]], key: str, *, published_only: bool = True) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if published_only and not row.get("published"):
            continue
        value = row.get(key)
        if value is None:
            continue
        value_f = as_float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    return np.asarray(values, dtype=np.float64)


def summarize(
    *,
    plan_rows: list[dict[str, Any]],
    tick_rows: list[dict[str, Any]],
    gnss: dict[str, np.ndarray],
    horizons_s: list[float],
    lds_m: list[float],
    rate_hz: float,
) -> dict[str, Any]:
    accepted_plans = [row for row in plan_rows if row.get("accepted")]
    published_rows = [row for row in tick_rows if row.get("published")]
    skipped_plans: dict[str, int] = {}
    for row in plan_rows:
        if row.get("accepted"):
            continue
        reason = str(row.get("reason") or "unknown")
        skipped_plans[reason] = skipped_plans.get(reason, 0) + 1
    skipped_ticks: dict[str, int] = {}
    for row in tick_rows:
        if row.get("published"):
            continue
        reason = str(row.get("skip_reason") or "unknown")
        skipped_ticks[reason] = skipped_ticks.get(reason, 0) + 1

    publish_times = finite_values(tick_rows, "tick_rel_s")
    gaps = np.diff(publish_times) if publish_times.size >= 2 else np.asarray([], dtype=np.float64)
    summary: dict[str, Any] = {
        "rate_hz": float(rate_hz),
        "input_plan_packets_total": len(plan_rows),
        "input_plan_packets_accepted": len(accepted_plans),
        "input_plan_packets_rejected": skipped_plans,
        "replay_ticks_total": len(tick_rows),
        "replay_ticks_published": len(published_rows),
        "replay_ticks_dropped": skipped_ticks,
        "publish_interval_s": stats(gaps),
        "gnss_duration_s": float(gnss["t_rel_s"][-1] - gnss["t_rel_s"][0]),
        "gnss_distance_m": float(np.nanmax(gnss["total_distance_m"])),
        "projection_distance_m": stats(finite_values(tick_rows, "projection_distance_m")),
        "remaining_distance_m": stats(finite_values(tick_rows, "remaining_distance_m")),
        "output_arc_m": stats(finite_values(tick_rows, "output_arc_m")),
        "local_start_y_m": stats(finite_values(tick_rows, "local_start_y_m")),
        "plan_age_s": stats(finite_values(tick_rows, "plan_age_s")),
        "plan_tx_age_s": stats(finite_values(tick_rows, "plan_tx_age_s")),
    }
    for horizon_s in horizons_s:
        key = f"{horizon_s:g}s"
        summary[f"error_at_{key}_m"] = stats(finite_values(tick_rows, f"error_at_{key}_m"))
        summary[f"geom_abs_cross_at_{key}_m"] = stats(finite_values(tick_rows, f"geom_abs_cross_at_{key}_m"))
        summary[f"geom_mean_abs_cross_{key}_m"] = stats(finite_values(tick_rows, f"geom_mean_abs_cross_{key}_m"))
        summary[f"along_at_{key}_m"] = stats(finite_values(tick_rows, f"along_at_{key}_m"))
    for ld_m in lds_m:
        key = f"{ld_m:g}m"
        summary[f"path_y_at_arc_{key}"] = stats(finite_values(tick_rows, f"path_y_at_arc_{ld_m:g}m"))
        summary[f"path_pp_kappa_at_arc_{key}"] = stats(finite_values(tick_rows, f"path_pp_kappa_at_arc_{ld_m:g}m"))
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


def plot_replay(
    output_path: Path,
    *,
    gnss: dict[str, np.ndarray],
    tick_rows: list[dict[str, Any]],
    overlays: list[dict[str, Any]],
    max_overlay_paths: int,
) -> None:
    published = [row for row in tick_rows if row.get("published")]
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=150)

    ax = axs[0, 0]
    ax.plot(gnss["x_m"], gnss["y_m"], color="black", linewidth=2.2, label="GNSS actual")
    if overlays:
        selected = np.linspace(0, len(overlays) - 1, min(len(overlays), max_overlay_paths), dtype=np.int64)
        cmap = plt.get_cmap("viridis")
        for rank, overlay_idx in enumerate(selected):
            item = overlays[int(overlay_idx)]
            path = item["sent_global"]
            color = cmap(rank / max(len(selected) - 1, 1))
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.55, linewidth=1.0)
            ax.scatter(path[0, 0], path[0, 1], s=7, color=color, alpha=0.8)
    ax.set_title("Offline path-manager outputs in GNSS frame")
    ax.set_xlabel("UTM east relative [m]")
    ax.set_ylabel("UTM north relative [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[0, 1]
    x = np.asarray([row["tick_rel_s"] for row in tick_rows], dtype=np.float64)
    pub = np.asarray([1.0 if row.get("published") else 0.0 for row in tick_rows], dtype=np.float64)
    ax.plot(x, pub, drawstyle="steps-post", linewidth=1.0, label="published")
    if published:
        age = np.asarray([as_float(row.get("plan_age_s")) for row in published], dtype=np.float64)
        px = np.asarray([row["tick_rel_s"] for row in published], dtype=np.float64)
        ax2 = ax.twinx()
        ax2.plot(px, age, color="tab:orange", linewidth=1.0, alpha=0.8, label="plan age")
        ax2.set_ylabel("plan age [s]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("10 Hz publish/drop timeline")
    ax.set_xlabel("time since GNSS start [s]")
    ax.set_ylabel("published")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    ax = axs[1, 0]
    if published:
        px = np.asarray([row["tick_rel_s"] for row in published], dtype=np.float64)
        proj = np.asarray([as_float(row.get("projection_distance_m")) for row in published], dtype=np.float64)
        rem = np.asarray([as_float(row.get("remaining_distance_m")) for row in published], dtype=np.float64)
        ax.plot(px, proj, linewidth=1.0, label="projection distance")
        ax.plot(px, rem, linewidth=1.0, label="remaining distance")
    ax.set_title("Projection and remaining horizon")
    ax.set_xlabel("time since GNSS start [s]")
    ax.set_ylabel("distance [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[1, 1]
    if published:
        px = np.asarray([row["tick_rel_s"] for row in published], dtype=np.float64)
        for key in ("geom_abs_cross_at_1s_m", "geom_abs_cross_at_2s_m", "geom_abs_cross_at_3s_m"):
            y = np.asarray([as_float(row.get(key)) for row in published], dtype=np.float64)
            if np.isfinite(y).any():
                ax.plot(px, y, linewidth=1.0, label=key.replace("geom_abs_cross_at_", "geom @").replace("_m", ""))
    ax.set_title("Geometric lateral error to future GNSS")
    ax.set_xlabel("time since GNSS start [s]")
    ax.set_ylabel("abs cross-track [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], plot_path: Path) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
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

    lines = [
        "# Offline Path Manager Replay",
        "",
        f"- Input plan packets: {summary['input_plan_packets_total']}",
        f"- Accepted plan packets: {summary['input_plan_packets_accepted']}",
        f"- Rejected plan packets: {summary['input_plan_packets_rejected']}",
        f"- Replay ticks: {summary['replay_ticks_total']}",
        f"- Published ticks: {summary['replay_ticks_published']}",
        f"- Dropped ticks: {summary['replay_ticks_dropped']}",
        f"- GNSS duration: {fmt(summary['gnss_duration_s'])} s",
        f"- GNSS distance: {fmt(summary['gnss_distance_m'])} m",
        f"- Plot: {plot_path}",
        "",
        "## Runtime Shape",
        "",
        stat_line("publish interval", summary["publish_interval_s"], "s"),
        stat_line("plan age", summary["plan_age_s"], "s"),
        stat_line("plan tx age", summary["plan_tx_age_s"], "s"),
        stat_line("projection distance", summary["projection_distance_m"], "m"),
        stat_line("remaining distance", summary["remaining_distance_m"], "m"),
        stat_line("output arc", summary["output_arc_m"], "m"),
        stat_line("local start y", summary["local_start_y_m"], "m"),
        "",
        "## Geometric Error",
        "",
    ]
    for key in sorted(k for k in summary if k.startswith("geom_abs_cross_at_")):
        lines.append(stat_line(key, summary[key], "m"))
    for key in sorted(k for k in summary if k.startswith("geom_mean_abs_cross_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Time Indexed Error", ""])
    for key in sorted(k for k in summary if k.startswith("error_at_")):
        lines.append(stat_line(key, summary[key], "m"))
    lines.extend(["", "## Lookahead Signals", ""])
    for key in sorted(k for k in summary if k.startswith("path_y_at_arc_")):
        lines.append(stat_line(key, summary[key], "m"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    horizons_s = parse_float_list(args.compare_horizons_s)
    lds_m = parse_float_list(args.ld_m)
    path_rows = read_jsonl(args.path_log)
    gnss = read_gnss_csv(args.gnss_csv)
    plans, plan_summary_rows = build_replay_plans(
        path_rows,
        gnss,
        min_plan_arc_m=float(args.min_plan_arc_m),
    )
    tick_rows, overlays = replay(
        plans,
        gnss,
        rate_hz=float(args.rate_hz),
        max_plan_age_s=float(args.max_plan_age_s),
        min_remaining_distance_m=float(args.min_remaining_distance_m),
        max_projection_distance_m=float(args.max_projection_distance_m),
        output_points=int(args.output_points),
        horizons_s=horizons_s,
        lds_m=lds_m,
    )
    summary = summarize(
        plan_rows=plan_summary_rows,
        tick_rows=tick_rows,
        gnss=gnss,
        horizons_s=horizons_s,
        lds_m=lds_m,
        rate_hz=float(args.rate_hz),
    )

    plan_csv = args.output_dir / "path_manager_replay_input_plans.csv"
    tick_csv = args.output_dir / "path_manager_replay_ticks.csv"
    summary_json = args.output_dir / "path_manager_replay_summary.json"
    plot_path = args.output_dir / "path_manager_replay.png"
    report_path = args.output_dir / "path_manager_replay_report.md"
    write_csv(plan_csv, plan_summary_rows)
    write_csv(tick_csv, tick_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_replay(
        plot_path,
        gnss=gnss,
        tick_rows=tick_rows,
        overlays=overlays,
        max_overlay_paths=int(args.max_overlay_paths),
    )
    write_report(report_path, summary, plot_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
