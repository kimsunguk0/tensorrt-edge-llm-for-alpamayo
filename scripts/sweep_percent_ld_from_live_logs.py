#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_live_udp_path_vs_gnss import (
    as_float,
    cumulative_arc,
    global_delta_to_local,
    interp_pose,
    interp_series,
    packet_local_xy,
    read_gnss_csv,
    read_jsonl,
    transform_local_to_global,
)


EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep variable lookahead points chosen as a percentage of sent path arc length, "
            "and compare them with future GNSS at the same traveled arc distance."
        )
    )
    parser.add_argument("--path-log", type=Path, required=True)
    parser.add_argument("--gnss-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-path-arc-m", type=float, default=4.0)
    parser.add_argument("--max-packet-age-s", type=float, default=4.0)
    parser.add_argument("--min-speed-mps", type=float, default=0.5)
    parser.add_argument("--percent-min", type=float, default=10.0)
    parser.add_argument("--percent-max", type=float, default=95.0)
    parser.add_argument("--percent-step", type=float, default=5.0)
    parser.add_argument("--fixed-ld-m", default="2,4,6,8")
    parser.add_argument("--control-arc-min-m", type=float, default=4.0)
    parser.add_argument("--control-arc-max-m", type=float, default=8.0)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def point_at_arc(points_xy: np.ndarray, arc_m: float) -> np.ndarray | None:
    arc = cumulative_arc(points_xy)
    if arc.size < 2 or float(arc[-1]) < float(arc_m):
        return None
    x = float(np.interp(float(arc_m), arc, points_xy[:, 0]))
    y = float(np.interp(float(arc_m), arc, points_xy[:, 1]))
    return np.asarray([x, y], dtype=np.float64)


def gnss_future_local_at_arc(
    gnss: dict[str, np.ndarray],
    *,
    tx_rel_s: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    target_arc_m: float,
    max_future_s: float,
) -> np.ndarray | None:
    if tx_rel_s < float(gnss["t_rel_s"][0]) or tx_rel_s >= float(gnss["t_rel_s"][-1]):
        return None
    sample_dt = 0.05
    future_t = tx_rel_s + np.arange(0.0, max_future_s + sample_dt * 0.5, sample_dt, dtype=np.float64)
    future_t = future_t[future_t <= float(gnss["t_rel_s"][-1])]
    if future_t.size < 3:
        return None
    gx = interp_series(gnss["t_rel_s"], gnss["x_m"], future_t)
    gy = interp_series(gnss["t_rel_s"], gnss["y_m"], future_t)
    valid = np.isfinite(gx) & np.isfinite(gy)
    if np.count_nonzero(valid) < 3:
        return None
    gx = gx[valid]
    gy = gy[valid]
    gt_global = np.column_stack([gx, gy])
    gt_local = global_delta_to_local(
        gt_global - np.asarray([[origin_x, origin_y]], dtype=np.float64),
        origin_yaw,
    )
    gt_arc = cumulative_arc(gt_local)
    if gt_arc.size < 2 or float(gt_arc[-1]) < float(target_arc_m):
        return None
    return point_at_arc(gt_local, target_arc_m)


def stat(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(abs_arr)),
        "median_abs": float(np.median(abs_arr)),
        "p90_abs": float(np.percentile(abs_arr, 90.0)),
        "max_abs": float(np.max(abs_arr)),
    }


def summarize_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return stat([float(row[key]) for row in rows if row.get(key) is not None])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_rows = read_jsonl(args.path_log)
    gnss = read_gnss_csv(args.gnss_csv)
    gt_t0_abs_s = float(gnss["t0_abs_s"][0])
    percents = list(
        np.arange(float(args.percent_min), float(args.percent_max) + 1e-6, float(args.percent_step))
    )
    fixed_lds = parse_float_list(args.fixed_ld_m)

    usable_records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for idx, record in enumerate(path_rows):
        header = dict(record.get("packet_header") or {})
        tx_time_us = int(header.get("tx_time_us", 0))
        tx_rel_s = tx_time_us / 1_000_000.0 - gt_t0_abs_s
        pose = interp_pose(gnss, tx_rel_s)
        local_xy = packet_local_xy(record)
        path_arc = float(cumulative_arc(local_xy)[-1]) if local_xy.shape[0] >= 2 else 0.0
        packet_age = as_float(header.get("latency_compensation_age_s"))
        plan_seq = int(header.get("plan_seq", record.get("sample_id", -1)))
        if pose is None or local_xy.shape[0] < 2:
            skipped["no_pose_or_path"] = skipped.get("no_pose_or_path", 0) + 1
            continue
        if path_arc < float(args.min_path_arc_m):
            skipped["short_path"] = skipped.get("short_path", 0) + 1
            continue
        if math.isfinite(packet_age) and packet_age > float(args.max_packet_age_s):
            skipped["stale_packet_age"] = skipped.get("stale_packet_age", 0) + 1
            continue
        origin_x, origin_y, origin_yaw, speed = pose
        if math.isfinite(speed) and speed < float(args.min_speed_mps):
            skipped["low_speed"] = skipped.get("low_speed", 0) + 1
            continue
        usable_records.append(
            {
                "idx": idx,
                "plan_seq": plan_seq,
                "tx_rel_s": tx_rel_s,
                "local_xy": local_xy,
                "path_arc_m": path_arc,
                "packet_age_s": packet_age,
                "origin_x": origin_x,
                "origin_y": origin_y,
                "origin_yaw": origin_yaw,
                "speed_mps": speed,
            }
        )

    percent_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    sample_errors_by_percent: dict[float, list[float]] = {p: [] for p in percents}
    sample_errors_by_fixed: dict[float, list[float]] = {ld: [] for ld in fixed_lds}

    for item in usable_records:
        local_xy = item["local_xy"]
        path_arc = float(item["path_arc_m"])
        max_future_s = max(8.0, path_arc / max(float(item["speed_mps"]), 0.3) + 2.0)
        for pct in percents:
            target_arc = path_arc * float(pct) / 100.0
            pred_local = point_at_arc(local_xy, target_arc)
            gt_local = gnss_future_local_at_arc(
                gnss,
                tx_rel_s=float(item["tx_rel_s"]),
                origin_x=float(item["origin_x"]),
                origin_y=float(item["origin_y"]),
                origin_yaw=float(item["origin_yaw"]),
                target_arc_m=target_arc,
                max_future_s=max_future_s,
            )
            if pred_local is None or gt_local is None:
                continue
            err_vec = pred_local - gt_local
            error = float(np.linalg.norm(err_vec))
            cross = float(err_vec[1])
            sample_errors_by_percent[pct].append(error)
            percent_rows.append(
                {
                    "kind": "percent",
                    "percent": pct,
                    "target_arc_m": target_arc,
                    "idx": item["idx"],
                    "plan_seq": item["plan_seq"],
                    "tx_rel_s": item["tx_rel_s"],
                    "path_arc_m": path_arc,
                    "speed_mps": item["speed_mps"],
                    "pred_x_m": float(pred_local[0]),
                    "pred_y_m": float(pred_local[1]),
                    "gt_x_m": float(gt_local[0]),
                    "gt_y_m": float(gt_local[1]),
                    "error_m": error,
                    "cross_error_m": cross,
                }
            )
        for ld_m in fixed_lds:
            if path_arc < ld_m:
                continue
            pred_local = point_at_arc(local_xy, ld_m)
            gt_local = gnss_future_local_at_arc(
                gnss,
                tx_rel_s=float(item["tx_rel_s"]),
                origin_x=float(item["origin_x"]),
                origin_y=float(item["origin_y"]),
                origin_yaw=float(item["origin_yaw"]),
                target_arc_m=ld_m,
                max_future_s=max_future_s,
            )
            if pred_local is None or gt_local is None:
                continue
            err_vec = pred_local - gt_local
            error = float(np.linalg.norm(err_vec))
            sample_errors_by_fixed[ld_m].append(error)
            fixed_rows.append(
                {
                    "kind": "fixed_ld",
                    "ld_m": ld_m,
                    "target_arc_m": ld_m,
                    "idx": item["idx"],
                    "plan_seq": item["plan_seq"],
                    "tx_rel_s": item["tx_rel_s"],
                    "path_arc_m": path_arc,
                    "speed_mps": item["speed_mps"],
                    "pred_x_m": float(pred_local[0]),
                    "pred_y_m": float(pred_local[1]),
                    "gt_x_m": float(gt_local[0]),
                    "gt_y_m": float(gt_local[1]),
                    "error_m": error,
                    "cross_error_m": float(err_vec[1]),
                }
            )

    percent_summary: list[dict[str, Any]] = []
    for pct in percents:
        rows_for_pct = [row for row in percent_rows if abs(float(row["percent"]) - pct) < 1e-9]
        if not rows_for_pct:
            continue
        err = summarize_rows(rows_for_pct, "error_m")
        cross = stat([float(row["cross_error_m"]) for row in rows_for_pct])
        arcs = stat([float(row["target_arc_m"]) for row in rows_for_pct])
        mean_arc = arcs.get("mean_abs")
        in_control = (
            mean_arc is not None
            and float(args.control_arc_min_m) <= float(mean_arc) <= float(args.control_arc_max_m)
        )
        percent_summary.append(
            {
                "percent": pct,
                "n": err.get("n", 0),
                "target_error_mean_m": err.get("mean"),
                "target_error_mean_abs_m": err.get("mean_abs"),
                "target_error_median_abs_m": err.get("median_abs"),
                "target_error_p90_abs_m": err.get("p90_abs"),
                "cross_error_mean_m": cross.get("mean"),
                "cross_error_mean_abs_m": cross.get("mean_abs"),
                "target_arc_mean_m": mean_arc,
                "target_arc_p90_m": arcs.get("p90_abs"),
                "control_arc_window": bool(in_control),
            }
        )

    fixed_summary: list[dict[str, Any]] = []
    for ld_m in fixed_lds:
        rows_for_ld = [row for row in fixed_rows if abs(float(row["ld_m"]) - ld_m) < 1e-9]
        if not rows_for_ld:
            continue
        err = summarize_rows(rows_for_ld, "error_m")
        cross = stat([float(row["cross_error_m"]) for row in rows_for_ld])
        fixed_summary.append(
            {
                "ld_m": ld_m,
                "n": err.get("n", 0),
                "target_error_mean_m": err.get("mean"),
                "target_error_mean_abs_m": err.get("mean_abs"),
                "target_error_median_abs_m": err.get("median_abs"),
                "target_error_p90_abs_m": err.get("p90_abs"),
                "cross_error_mean_m": cross.get("mean"),
                "cross_error_mean_abs_m": cross.get("mean_abs"),
            }
        )

    best_any = min(
        (row for row in percent_summary if int(row["n"]) > 0),
        key=lambda row: float(row["target_error_mean_abs_m"]),
        default=None,
    )
    control_candidates = [
        row for row in percent_summary if row["control_arc_window"] and int(row["n"]) > 0
    ]
    best_control = min(
        control_candidates,
        key=lambda row: float(row["target_error_mean_abs_m"]),
        default=None,
    )

    with (args.output_dir / "percent_ld_sweep_rows.csv").open("w", newline="", encoding="utf-8") as f:
        if percent_rows:
            writer = csv.DictWriter(f, fieldnames=list(percent_rows[0].keys()))
            writer.writeheader()
            writer.writerows(percent_rows)
    with (args.output_dir / "percent_ld_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        if percent_summary:
            writer = csv.DictWriter(f, fieldnames=list(percent_summary[0].keys()))
            writer.writeheader()
            writer.writerows(percent_summary)

    summary = {
        "path_log": str(args.path_log),
        "gnss_csv": str(args.gnss_csv),
        "usable_records": len(usable_records),
        "skipped": skipped,
        "min_path_arc_m": args.min_path_arc_m,
        "max_packet_age_s": args.max_packet_age_s,
        "min_speed_mps": args.min_speed_mps,
        "control_arc_window_m": [args.control_arc_min_m, args.control_arc_max_m],
        "best_percent_any": best_any,
        "best_percent_control_arc_window": best_control,
        "percent_summary": percent_summary,
        "fixed_ld_summary": fixed_summary,
    }
    (args.output_dir / "percent_ld_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)
    pct_x = [float(row["percent"]) for row in percent_summary]
    ade_y = [float(row["target_error_mean_abs_m"]) for row in percent_summary]
    p90_y = [float(row["target_error_p90_abs_m"]) for row in percent_summary]
    arc_y = [float(row["target_arc_mean_m"]) for row in percent_summary]
    axes[0].plot(pct_x, ade_y, marker="o", label="mean target error")
    axes[0].plot(pct_x, p90_y, marker=".", label="p90 target error")
    if best_any:
        axes[0].axvline(float(best_any["percent"]), color="#64748b", linestyle="--", label="best any")
    if best_control:
        axes[0].axvline(float(best_control["percent"]), color="#16a34a", linestyle="--", label="best 4-8m arc")
    axes[0].set_ylabel("target error vs GNSS [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(pct_x, arc_y, marker="o", color="#f97316", label="mean target arc")
    axes[1].axhspan(args.control_arc_min_m, args.control_arc_max_m, color="#dcfce7", alpha=0.45, label="4-8m window")
    axes[1].set_xlabel("lookahead percent of sent path length [%]")
    axes[1].set_ylabel("target arc [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "percent_ld_sweep.png")
    plt.close(fig)

    lines = [
        "# Percent Lookahead Sweep",
        "",
        f"- path_log: `{args.path_log}`",
        f"- gnss_csv: `{args.gnss_csv}`",
        f"- usable records: {len(usable_records)}",
        f"- skipped: `{skipped}`",
        f"- plot: `{args.output_dir / 'percent_ld_sweep.png'}`",
        "",
        "Metric: pick the point at p% of sent path arc length, then compare it to the future GNSS point at the same traveled arc distance from the TX pose.",
        "",
        "## Best",
        "",
        f"- best overall percent: `{best_any}`",
        f"- best percent with mean target arc inside {args.control_arc_min_m:g}-{args.control_arc_max_m:g}m: `{best_control}`",
        "",
        "## Percent Summary",
        "",
        "| percent | n | mean target arc m | mean error m | median error m | p90 error m | mean cross m |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in percent_summary:
        lines.append(
            f"| {float(row['percent']):.1f} | {int(row['n'])} | "
            f"{float(row['target_arc_mean_m']):.3f} | {float(row['target_error_mean_abs_m']):.3f} | "
            f"{float(row['target_error_median_abs_m']):.3f} | {float(row['target_error_p90_abs_m']):.3f} | "
            f"{float(row['cross_error_mean_m']):.3f} |"
        )
    lines.extend(["", "## Fixed LD Summary", ""])
    lines.append("| LD m | n | mean error m | median error m | p90 error m | mean cross m |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in fixed_summary:
        lines.append(
            f"| {float(row['ld_m']):.1f} | {int(row['n'])} | "
            f"{float(row['target_error_mean_abs_m']):.3f} | {float(row['target_error_median_abs_m']):.3f} | "
            f"{float(row['target_error_p90_abs_m']):.3f} | {float(row['cross_error_mean_m']):.3f} |"
        )
    (args.output_dir / "percent_ld_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(args.output_dir / "percent_ld_sweep_summary.json"), "report_md": str(args.output_dir / "percent_ld_sweep_report.md")}, indent=2))


if __name__ == "__main__":
    main()
