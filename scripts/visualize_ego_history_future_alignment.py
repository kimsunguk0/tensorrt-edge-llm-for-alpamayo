#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay ego history with GT/model future paths and report t0 alignment metrics."
    )
    parser.add_argument(
        "--run-root",
        default="/workspace/alpamayo_vlm/output/live_capture_dynamic_inference_20260526_063851_064421_latest_model/run_20260526_063851",
    )
    parser.add_argument(
        "--capture-root",
        default="/workspace/live_camera_ego_history_capture/0526_1",
        help="Capture root containing ego_history/*.npy. This may differ from stale paths in request metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/alpamayo_vlm/output/ego_history_future_alignment_20260526_063851",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-future-distance-m", type=float, default=5.0)
    parser.add_argument("--history-heading-window-m", type=float, default=0.5)
    parser.add_argument("--future-heading-window-m", type=float, default=0.5)
    parser.add_argument("--xlim", type=float, default=25.0)
    parser.add_argument("--ylim", type=float, default=8.0)
    return parser.parse_args()


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float64)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_xyz(path: Path) -> np.ndarray:
    data = load_json(path)
    xyz = np.asarray(data.get("pred_xyz", []), dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    if xyz.shape[1] == 2:
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return xyz[:, :3]


def with_origin(xyz: np.ndarray) -> np.ndarray:
    origin = np.zeros((1, 3), dtype=np.float64)
    return np.concatenate([origin, xyz], axis=0)


def choose_evenly(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    used = set()
    chosen = []
    for idx in indices:
        if int(idx) in used:
            continue
        used.add(int(idx))
        chosen.append(rows[int(idx)])
    return chosen


def first_index_at_distance(xy: np.ndarray, distance_m: float) -> int:
    if len(xy) == 0:
        return 0
    dist = np.linalg.norm(xy[:, :2] - xy[0, :2], axis=1)
    candidates = np.where(dist >= float(distance_m))[0]
    if candidates.size:
        return int(candidates[0])
    return int(len(xy) - 1)


def history_heading(hist_xyz: np.ndarray, window_m: float) -> tuple[float | None, float]:
    end = hist_xyz[-1, :2]
    dist_from_end = np.linalg.norm(hist_xyz[:, :2] - end, axis=1)
    candidates = np.where(dist_from_end >= float(window_m))[0]
    if candidates.size:
        idx = int(candidates[-1])
    else:
        idx = max(0, len(hist_xyz) - 2)
    vec = end - hist_xyz[idx, :2]
    length = float(np.linalg.norm(vec))
    if length < 1e-4:
        return None, length
    return float(math.atan2(vec[1], vec[0])), length


def future_heading(future_xyz_with_origin: np.ndarray, window_m: float) -> tuple[float | None, float]:
    if len(future_xyz_with_origin) < 2:
        return None, 0.0
    idx = first_index_at_distance(future_xyz_with_origin[:, :2], window_m)
    idx = max(1, idx)
    vec = future_xyz_with_origin[idx, :2] - future_xyz_with_origin[0, :2]
    length = float(np.linalg.norm(vec))
    if length < 1e-4:
        return None, length
    return float(math.atan2(vec[1], vec[0])), length


def interp_y_at_x(path_with_origin: np.ndarray, x_m: float) -> float | None:
    x = path_with_origin[:, 0]
    y = path_with_origin[:, 1]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    if unique_x.size < 2 or x_m < float(unique_x[0]) or x_m > float(unique_x[-1]):
        return None
    return float(np.interp(float(x_m), unique_x, y_sorted[unique_idx]))


def select_rows(run_root: Path, min_future_distance_m: float, num_samples: int) -> list[dict[str, Any]]:
    schedule = load_json(run_root / "evaluation" / "selected_schedule.json")
    valid = []
    for row in schedule:
        artifact_root = Path(row["artifact_root"])
        gt_path = artifact_root / "gt_path.json"
        final_path = artifact_root / "final_path.json"
        if not gt_path.exists() or not final_path.exists():
            continue
        gt = path_xyz(gt_path)
        if len(gt) == 0:
            continue
        distance = float(np.sum(np.linalg.norm(np.diff(with_origin(gt)[:, :2], axis=0), axis=1)))
        if distance < float(min_future_distance_m):
            continue
        row = dict(row)
        row["future_distance_m"] = distance
        valid.append(row)
    return choose_evenly(valid, num_samples)


def ego_paths(capture_root: Path, sample_id: int, t0_us: int) -> tuple[Path, Path]:
    ego_root = capture_root / "ego_history"
    xyz = ego_root / f"sample_{sample_id:06d}_t0_{t0_us}_ego_history_xyz.npy"
    rot = ego_root / f"sample_{sample_id:06d}_t0_{t0_us}_ego_history_rot.npy"
    if xyz.exists() and rot.exists():
        return xyz, rot
    matches = sorted(ego_root.glob(f"sample_{sample_id:06d}_t0_*_ego_history_xyz.npy"))
    if not matches:
        raise FileNotFoundError(f"missing ego history for sample {sample_id}")
    xyz = matches[0]
    rot = Path(str(xyz).replace("_xyz.npy", "_rot.npy"))
    return xyz, rot


def analyze_row(row: dict[str, Any], capture_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    sample_id = int(row["sample_id"])
    t0_us = int(row["t0_us"])
    xyz_path, rot_path = ego_paths(capture_root, sample_id, t0_us)
    hist = np.load(xyz_path).astype(np.float64)[0, 0]
    rot = np.load(rot_path).astype(np.float64)[0, 0]
    hist_yaw = yaw_from_rot(rot)

    artifact_root = Path(row["artifact_root"])
    gt = with_origin(path_xyz(artifact_root / "gt_path.json"))
    final = with_origin(path_xyz(artifact_root / "final_path.json"))

    hist_head, hist_head_len = history_heading(hist, args.history_heading_window_m)
    fut_head, fut_head_len = future_heading(gt, args.future_heading_window_m)
    yaw_diff = None
    if hist_head is not None and fut_head is not None:
        yaw_diff = math.degrees(wrap_angle(fut_head - hist_head))

    hist_step_speed = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]) / 0.1) if len(hist) >= 2 else 0.0
    gt_step_speed = float(np.linalg.norm(gt[1, :2] - gt[0, :2]) / 0.1) if len(gt) >= 2 else 0.0
    final_step_speed = float(np.linalg.norm(final[1, :2] - final[0, :2]) / 0.1) if len(final) >= 2 else 0.0
    final_minus_gt_y_10 = None
    gt_y_10 = interp_y_at_x(gt, 10.0)
    final_y_10 = interp_y_at_x(final, 10.0)
    if gt_y_10 is not None and final_y_10 is not None:
        final_minus_gt_y_10 = final_y_10 - gt_y_10

    return {
        "sample_id": sample_id,
        "selected_index": int(row["selected_index"]),
        "t0_us": t0_us,
        "artifact_root": str(artifact_root),
        "ego_history_xyz": str(xyz_path),
        "ego_history_rot": str(rot_path),
        "hist_end_xy_error_m": float(np.linalg.norm(hist[-1, :2])),
        "hist_end_yaw_deg": float(math.degrees(hist_yaw[-1])),
        "hist_step_speed_mps": hist_step_speed,
        "gt_first_step_speed_mps": gt_step_speed,
        "final_first_step_speed_mps": final_step_speed,
        "history_heading_deg": None if hist_head is None else float(math.degrees(hist_head)),
        "future_heading_deg": None if fut_head is None else float(math.degrees(fut_head)),
        "junction_yaw_diff_deg": yaw_diff,
        "history_heading_window_m": hist_head_len,
        "future_heading_window_m": fut_head_len,
        "gt_y_at_10m": gt_y_10,
        "final_y_at_10m": final_y_10,
        "final_minus_gt_y_at_10m": final_minus_gt_y_10,
        "future_distance_m": float(row["future_distance_m"]),
        "_hist": hist,
        "_gt": gt,
        "_final": final,
    }


def finite_values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for rec in records:
        value = rec.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "hist_end_xy_error_m",
        "hist_end_yaw_deg",
        "hist_step_speed_mps",
        "gt_first_step_speed_mps",
        "junction_yaw_diff_deg",
        "gt_y_at_10m",
        "final_y_at_10m",
        "final_minus_gt_y_at_10m",
    ]
    out: dict[str, Any] = {"num_samples": len(records)}
    for key in keys:
        arr = finite_values(records, key)
        if arr.size == 0:
            continue
        out[key] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
        }
    return out


def plot_records(records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    cols = 4
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, rec in enumerate(records):
        ax = axes.ravel()[idx]
        ax.axis("on")
        hist = rec["_hist"]
        gt = rec["_gt"]
        final = rec["_final"]
        ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", marker="o", markersize=2.5, linewidth=1.5, label="ego history")
        ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", linewidth=1.8, label="GT future")
        ax.plot(final[:, 0], final[:, 1], color="#d62728", linewidth=1.4, alpha=0.85, label="model final")
        ax.scatter([0.0], [0.0], color="black", s=18, zorder=5)
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.axvline(0.0, color="0.8", linewidth=0.8)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-4.0, args.xlim)
        ax.set_ylim(-args.ylim, args.ylim)
        yaw_diff = rec.get("junction_yaw_diff_deg")
        yaw_text = "na" if yaw_diff is None else f"{yaw_diff:+.1f}deg"
        y10 = rec.get("final_minus_gt_y_at_10m")
        y10_text = "na" if y10 is None else f"{y10:+.2f}m"
        ax.set_title(
            f"sid {rec['sample_id']} | join {yaw_text} | dY10 {y10_text}",
            fontsize=9,
        )
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Ego history vs GT future vs model final path", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "ego_history_future_alignment_grid.png", dpi=180)
    plt.close(fig)

    indiv = output_dir / "samples"
    indiv.mkdir(parents=True, exist_ok=True)
    for rec in records:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        hist = rec["_hist"]
        gt = rec["_gt"]
        final = rec["_final"]
        ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", marker="o", markersize=3, linewidth=1.8, label="ego history")
        ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", linewidth=2.2, label="GT future")
        ax.plot(final[:, 0], final[:, 1], color="#d62728", linewidth=1.8, alpha=0.9, label="model final")
        ax.scatter([0.0], [0.0], color="black", s=25, label="t0 origin")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.axvline(0.0, color="0.8", linewidth=0.8)
        ax.grid(True, color="0.9")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-4.0, args.xlim)
        ax.set_ylim(-args.ylim, args.ylim)
        ax.set_xlabel("x forward [m]")
        ax.set_ylabel("y left [m]")
        ax.legend(loc="upper left")
        ax.set_title(f"sample {rec['sample_id']} t0={rec['t0_us']}")
        fig.tight_layout()
        fig.savefig(indiv / f"sample_{rec['sample_id']:06d}_alignment.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    capture_root = Path(args.capture_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_rows(run_root, args.min_future_distance_m, args.num_samples)
    records = [analyze_row(row, capture_root, args) for row in selected]
    plot_records(records, output_dir, args)

    public_records = []
    for rec in records:
        public_records.append({k: v for k, v in rec.items() if not k.startswith("_")})

    with (output_dir / "alignment_metrics.csv").open("w", newline="") as fh:
        fieldnames = list(public_records[0].keys()) if public_records else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_records)

    summary = summarize(public_records)
    summary["run_root"] = str(run_root)
    summary["capture_root"] = str(capture_root)
    summary["grid_png"] = str(output_dir / "ego_history_future_alignment_grid.png")
    summary["metrics_csv"] = str(output_dir / "alignment_metrics.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
