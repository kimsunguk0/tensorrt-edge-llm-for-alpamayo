#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_ROOTS = [
    REPO_ROOT / "output" / "raw_dataset_one_shot" / "chunk0001_offset_49p0",
    REPO_ROOT / "output" / "raw_dataset_one_shot" / "chunk0001_offset_52p0",
    REPO_ROOT / "output" / "raw_dataset_one_shot" / "chunk0002_offset_20p0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a yaw diagnostic visualization for a few straight-ish one-shot samples."
    )
    parser.add_argument(
        "--sample-root",
        action="append",
        type=Path,
        default=[],
        help="One-shot sample root such as output/raw_dataset_one_shot/chunk0002_offset_20p0. May be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "reports" / "straight_yaw_diagnostic",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def find_request_json(sample_root: Path) -> Path:
    request_paths = sorted((sample_root / "request_bank" / "requests").glob("request_*.json"))
    if len(request_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one request JSON under {sample_root}, found {len(request_paths)}")
    return request_paths[0]


def extract_front_image(request_json: Path) -> Path:
    request = load_json(request_json)
    items = request["requests"][0]["messages"][1]["content"]
    current_camera = ""
    front_image: Path | None = None
    for item in items:
        if item["type"] == "text":
            text = item["text"].strip()
            if text.endswith("camera:"):
                current_camera = text.rstrip(":")
        elif item["type"] == "image" and current_camera == "Front camera":
            front_image = Path(item["image"])
    if front_image is None:
        raise FileNotFoundError(f"Could not find latest front camera image in {request_json}")
    return front_image


def load_history(request_json: Path) -> tuple[np.ndarray, np.ndarray]:
    request = load_json(request_json)
    req = request["requests"][0]
    hist_xyz = np.load(req["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    hist_rot = np.load(req["ego_history_rot_npy"]).astype(np.float32)[0, 0]
    return hist_xyz, hist_rot


def load_path_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    if "pred_xyz" not in obj or "pred_yaw_rad" not in obj:
        raise ValueError(f"Unexpected path summary format: {path}")
    return obj


def build_row(sample_root: Path) -> dict[str, Any]:
    request_json = find_request_json(sample_root)
    hist_xyz, hist_rot = load_history(request_json)
    final_summary = load_path_summary(sample_root / "artifacts" / "final_path.json")
    gt_summary = load_path_summary(sample_root / "artifacts" / "gt_path.json")

    history_yaw = wrap_angles(yaw_from_rot(hist_rot))
    history_yaw_unwrapped = unwrap_angles(history_yaw)
    history_yaw_rel = history_yaw_unwrapped - history_yaw_unwrapped[-1]

    gen_yaw = np.asarray(final_summary["pred_yaw_rad"], dtype=np.float32)
    gt_yaw = np.asarray(gt_summary["pred_yaw_rad"], dtype=np.float32)
    gen_yaw_rel = unwrap_angles(gen_yaw) - float(history_yaw_unwrapped[-1])
    gt_yaw_rel = unwrap_angles(gt_yaw) - float(history_yaw_unwrapped[-1])

    hist_times = (np.arange(len(history_yaw_rel), dtype=np.float32) - (len(history_yaw_rel) - 1)) * 0.1
    future_dt = float(final_summary["plan_dt_s"])
    future_times = (np.arange(len(gen_yaw_rel), dtype=np.float32) + 1.0) * future_dt

    gen_xyz = np.asarray(final_summary["pred_xyz"], dtype=np.float32)
    gt_xyz = np.asarray(gt_summary["pred_xyz"], dtype=np.float32)

    return {
        "sample_root": sample_root,
        "label": sample_root.name,
        "front_image": extract_front_image(request_json),
        "history_xyz": hist_xyz,
        "history_yaw_rel": history_yaw_rel.astype(np.float32),
        "hist_times": hist_times.astype(np.float32),
        "gen_xyz": gen_xyz,
        "gt_xyz": gt_xyz,
        "gen_yaw_rel": gen_yaw_rel.astype(np.float32),
        "gt_yaw_rel": gt_yaw_rel.astype(np.float32),
        "future_times": future_times.astype(np.float32),
        "final_output": final_summary.get("final_output", ""),
        "actual_offset_s": float(final_summary.get("actual_offset_s", 0.0)),
        "gen_yaw_range": float(np.ptp(gen_yaw_rel)) if len(gen_yaw_rel) else 0.0,
        "gt_yaw_range": float(np.ptp(gt_yaw_rel)) if len(gt_yaw_rel) else 0.0,
        "gen_final_y": float(gen_xyz[-1, 1]) if len(gen_xyz) else 0.0,
        "gt_final_y": float(gt_xyz[-1, 1]) if len(gt_xyz) else 0.0,
    }


def plot_topdown(ax: plt.Axes, row: dict[str, Any]) -> None:
    hist_xyz = np.asarray(row["history_xyz"], dtype=np.float32)
    gen_xyz = np.asarray(row["gen_xyz"], dtype=np.float32)
    gt_xyz = np.asarray(row["gt_xyz"], dtype=np.float32)

    ax.set_facecolor("#0f1117")
    ax.plot(-hist_xyz[:, 1], hist_xyz[:, 0], color="#9ca3af", linewidth=1.7, label="ego history")
    ax.plot(-gen_xyz[:, 1], gen_xyz[:, 0], color="#60a5fa", linewidth=2.2, label="generated")
    ax.plot(-gt_xyz[:, 1], gt_xyz[:, 0], color="#34d399", linewidth=1.9, linestyle="--", label="GT")
    ax.scatter([0], [0], color="white", s=24, marker="x", zorder=5)

    all_x = np.concatenate([-hist_xyz[:, 1], -gen_xyz[:, 1], -gt_xyz[:, 1]])
    all_y = np.concatenate([hist_xyz[:, 0], gen_xyz[:, 0], gt_xyz[:, 0]])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad_x = max(0.8, 0.12 * (x_max - x_min))
    pad_y = max(1.0, 0.12 * (y_max - y_min))
    ax.set_xlim(x_max + pad_x, x_min - pad_x)
    ax.set_ylim(y_min - 0.5, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(which="major", color="#334155", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", color="#1f2937", linewidth=0.35, alpha=0.4)
    ax.tick_params(colors="#e5e7eb", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.set_xlabel("Left +y [m]", color="#e5e7eb", fontsize=8)
    ax.set_ylabel("Forward +x [m]", color="#e5e7eb", fontsize=8)


def plot_yaw(ax: plt.Axes, row: dict[str, Any]) -> None:
    ax.set_facecolor("#0f1117")
    ax.axvline(0.0, color="#475569", linewidth=1.0, linestyle=":")
    ax.plot(row["hist_times"], row["history_yaw_rel"], color="#9ca3af", linewidth=1.8, label="ego_history yaw")
    ax.plot(row["future_times"], row["gen_yaw_rel"], color="#60a5fa", linewidth=2.2, label="generated yaw")
    ax.plot(row["future_times"], row["gt_yaw_rel"], color="#34d399", linewidth=1.9, linestyle="--", label="GT yaw")
    ax.set_xlim(float(row["hist_times"][0]), float(row["future_times"][-1]))
    all_y = np.concatenate([row["history_yaw_rel"], row["gen_yaw_rel"], row["gt_yaw_rel"]])
    max_abs = max(0.05, float(np.max(np.abs(all_y))) * 1.15)
    ax.set_ylim(-max_abs, max_abs)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(which="major", color="#334155", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", color="#1f2937", linewidth=0.35, alpha=0.4)
    ax.tick_params(colors="#e5e7eb", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.set_xlabel("Time from t0 [s]", color="#e5e7eb", fontsize=8)
    ax.set_ylabel("Yaw rel. to t0 [rad]", color="#e5e7eb", fontsize=8)


def build_figure(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 4.7 * len(rows)), dpi=180)
    fig.patch.set_facecolor("#0b1117")
    gs = GridSpec(len(rows), 3, figure=fig, width_ratios=[1.15, 1.0, 1.25], hspace=0.28, wspace=0.18)

    for row_idx, row in enumerate(rows):
        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(Image.open(row["front_image"]).convert("RGB"))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_facecolor("#111827")
        for spine in ax_img.spines.values():
            spine.set_color("#374151")
        ax_img.set_title(
            f"{row['label']}  |  front image @ {row['actual_offset_s']:.3f}s",
            fontsize=10,
            color="white",
            pad=8,
        )

        ax_traj = fig.add_subplot(gs[row_idx, 1])
        plot_topdown(ax_traj, row)
        if row_idx == 0:
            ax_traj.legend(facecolor="#111827", edgecolor="#374151", framealpha=0.95, fontsize=7, loc="upper left")
        ax_traj.set_title("Top-down: history vs generated vs GT", fontsize=10, color="white", pad=8)

        ax_yaw = fig.add_subplot(gs[row_idx, 2])
        plot_yaw(ax_yaw, row)
        ax_yaw.set_title("Yaw diagnostic", fontsize=10, color="white", pad=8)
        text = (
            f"Final: {row['final_output']}\n"
            f"gen yaw range={row['gen_yaw_range']:.3f} rad | gt yaw range={row['gt_yaw_range']:.3f} rad\n"
            f"gen final y={row['gen_final_y']:.3f} m | gt final y={row['gt_final_y']:.3f} m"
        )
        ax_yaw.text(
            0.02,
            0.98,
            text,
            transform=ax_yaw.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=8,
            bbox={"facecolor": "#111827", "edgecolor": "#374151", "boxstyle": "round,pad=0.3", "alpha": 0.9},
        )

    fig.suptitle(
        "Straight-Sample Diagnostic: Front Image + Top-down Path + Ego/Generated/GT Yaw",
        color="white",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_roots = args.sample_root or DEFAULT_SAMPLE_ROOTS
    rows = [build_row(sample_root) for sample_root in sample_roots]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "straight_yaw_diagnostic.png"
    summary_path = args.out_dir / "summary.json"
    build_figure(rows, out_path)

    summary = {
        "samples": [
            {
                "sample_root": str(row["sample_root"]),
                "front_image": str(row["front_image"]),
                "actual_offset_s": row["actual_offset_s"],
                "final_output": row["final_output"],
                "gen_yaw_range": row["gen_yaw_range"],
                "gt_yaw_range": row["gt_yaw_range"],
                "gen_final_y": row["gen_final_y"],
                "gt_final_y": row["gt_final_y"],
            }
            for row in rows
        ],
        "diagnostic_png": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
