#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fm_model_defaults import first_existing_fm_engine
from run_raw_dataset_one_shot_udp import build_result_artifacts, load_request_bank


DEFAULT_REQUEST_BANK_ROOTS = sorted((REPO_ROOT / "output" / "live_chunk_udp_replay").glob("chunk*/request_bank"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select straight-ish samples, run the current model on them, and summarize lateral drift bias."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--request-bank-root", action="append", type=Path, default=[])
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-per-chunk", type=int, default=2)
    parser.add_argument("--future-len", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--min-forward-m", type=float, default=20.0)
    parser.add_argument("--max-abs-final-y-m", type=float, default=2.5)
    parser.add_argument("--max-yaw-range-rad", type=float, default=0.10)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "reports" / "straight_bias_report")
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=first_existing_fm_engine())
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def build_env(plugin_lib: Path) -> dict[str, str]:
    env = dict(os.environ)
    if plugin_lib.exists():
        env["EDGELLM_PLUGIN_PATH"] = str(plugin_lib)
        build_dir = str(plugin_lib.parent)
        env["LD_LIBRARY_PATH"] = build_dir + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    return env


def load_gnss_arrays(dataset_root: Path) -> dict[str, np.ndarray]:
    import pandas as pd

    gnss_path = dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
    ensure_exists(gnss_path, "gnss parquet")
    gnss = pd.read_parquet(gnss_path)
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    return {
        "utc": gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64),
        "lat": gnss_valid["lat"].to_numpy(dtype=np.float64),
        "lon": gnss_valid["lon"].to_numpy(dtype=np.float64),
        "alt": gnss_valid["alt"].to_numpy(dtype=np.float64),
    }


def build_gt_future_local_pose_fast(
    request_bank: Any,
    gnss_arrays: dict[str, np.ndarray],
    *,
    t0_utc_ns: int,
    future_len: int,
    dt_s: float,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    utc = gnss_arrays["utc"]
    lat = gnss_arrays["lat"]
    lon = gnss_arrays["lon"]
    alt = gnss_arrays["alt"]

    ref_lla = (
        float(np.interp([t0_utc_ns], utc, lat)[0]),
        float(np.interp([t0_utc_ns], utc, lon)[0]),
        float(np.interp([t0_utc_ns], utc, alt)[0]),
    )
    dt_ns = int(round(dt_s * 1e9))
    hist_times = np.asarray([t0_utc_ns - (history_len - 1 - i) * dt_ns for i in range(history_len)], dtype=np.int64)
    fut_times = np.asarray([t0_utc_ns + (i + 1) * dt_ns for i in range(future_len)], dtype=np.int64)
    all_times = np.concatenate([hist_times, fut_times], axis=0)

    interp_lat = np.interp(all_times, utc, lat)
    interp_lon = np.interp(all_times, utc, lon)
    interp_alt = np.interp(all_times, utc, alt)
    world_xyz = request_bank.ecef_to_enu(
        request_bank.geodetic_to_ecef(interp_lat, interp_lon, interp_alt),
        *ref_lla,
    ).astype(np.float32)

    world_vel_xy = np.zeros((len(all_times), 2), dtype=np.float32)
    world_vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_ns / 1e9)
    world_vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_ns / 1e9)
    world_vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_ns / 1e9)
    world_yaw = np.arctan2(world_vel_xy[:, 1], world_vel_xy[:, 0]).astype(np.float32)
    world_rot = request_bank.yaw_to_rot(world_yaw)

    hist_world_xyz = world_xyz[:history_len]
    fut_world_xyz = world_xyz[history_len:]
    hist_world_rot = world_rot[:history_len]
    fut_world_rot = world_rot[history_len:]
    p0 = hist_world_xyz[-1]
    r0 = hist_world_rot[-1]
    r0_t = r0.T
    future_local_xyz = ((fut_world_xyz - p0) @ r0).astype(np.float32)
    future_local_rot = np.einsum("ij,tjk->tik", r0_t, fut_world_rot).astype(np.float32)
    return future_local_xyz, future_local_rot


def compute_gt_straight_metrics(gt_xyz: np.ndarray, gt_rot: np.ndarray) -> dict[str, float]:
    gt_yaw = yaw_from_rot(gt_rot)
    gt_yaw_range = float(np.ptp(unwrap_angles(gt_yaw))) if len(gt_yaw) else 0.0
    return {
        "gt_final_x_m": float(gt_xyz[-1, 0]),
        "gt_final_y_m": float(gt_xyz[-1, 1]),
        "gt_yaw_range_rad": gt_yaw_range,
    }


def select_straight_candidates(
    *,
    request_bank_roots: list[Path],
    dataset_root: Path,
    future_len: int,
    history_len: int,
    dt_s: float,
    min_forward_m: float,
    max_abs_final_y_m: float,
    max_yaw_range_rad: float,
    max_samples: int,
    max_per_chunk: int,
) -> list[dict[str, Any]]:
    request_bank = load_request_bank()
    gnss_arrays = load_gnss_arrays(dataset_root)
    selected: list[dict[str, Any]] = []

    for request_bank_root in request_bank_roots:
        manifest_path = request_bank_root / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest_rows = load_json(manifest_path)
        chunk_candidates: list[dict[str, Any]] = []
        for row in manifest_rows:
            gt_xyz, gt_rot = build_gt_future_local_pose_fast(
                request_bank,
                gnss_arrays,
                t0_utc_ns=int(row["t0_utc_ns"]),
                future_len=future_len,
                dt_s=dt_s,
                history_len=history_len,
            )
            metrics = compute_gt_straight_metrics(gt_xyz, gt_rot)
            if metrics["gt_final_x_m"] < min_forward_m:
                continue
            if abs(metrics["gt_final_y_m"]) > max_abs_final_y_m:
                continue
            if metrics["gt_yaw_range_rad"] > max_yaw_range_rad:
                continue

            selected_frames = row.get("selected_frames", {})
            front_ids = selected_frames.get("front") or []
            front_frame_id = int(front_ids[-1]) if front_ids else int(row["sample_id"])
            chunk_candidates.append(
                {
                    "chunk_id": int(row["chunk_id"]),
                    "sample_id": int(row["sample_id"]),
                    "front_frame_id": front_frame_id,
                    "t0_utc_ns": int(row["t0_utc_ns"]),
                    "t0_us": int(row["t0_us"]),
                    "request_json": str(row["request_json"]),
                    "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
                    "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
                    "selected_frames": selected_frames,
                    **metrics,
                }
            )

        if not chunk_candidates:
            continue
        chunk_candidates.sort(key=lambda item: int(item["t0_utc_ns"]))
        take = min(max_per_chunk, len(chunk_candidates))
        pick_indices = np.linspace(0, len(chunk_candidates) - 1, num=take, dtype=int)
        for idx in pick_indices.tolist():
            selected.append(chunk_candidates[idx])

    selected.sort(key=lambda item: (int(item["chunk_id"]), int(item["t0_utc_ns"])))
    if len(selected) > max_samples:
        indices = np.linspace(0, len(selected) - 1, num=max_samples, dtype=int)
        selected = [selected[i] for i in indices.tolist()]
    return selected


def prepare_subset_request_root(selected: list[dict[str, Any]], request_root: Path) -> None:
    request_root.mkdir(parents=True, exist_ok=True)
    for stale in request_root.glob("request_*.json"):
        stale.unlink()
    for row in selected:
        src = Path(row["request_json"])
        dst = request_root / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def run_inference_on_subset(
    *,
    request_root: Path,
    output_root: Path,
    llm_inference_bin: Path,
    plugin_lib: Path,
    engine_dir: Path,
    multimodal_engine_dir: Path,
    fm_engine: Path,
    warmup: int,
    timeout_per_request: float,
    skip_existing: bool,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_request_bank_persistent.py"),
        "--request-root",
        str(request_root),
        "--output-root",
        str(output_root),
        "--llm-inference-bin",
        str(llm_inference_bin),
        "--plugin-lib",
        str(plugin_lib),
        "--engine-dir",
        str(engine_dir),
        "--multimodal-engine-dir",
        str(multimodal_engine_dir),
        "--fm-engine",
        str(fm_engine),
        "--warmup",
        str(warmup),
        "--timeout-per-request",
        str(timeout_per_request),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    subprocess.run(cmd, cwd=REPO_ROOT, env=build_env(plugin_lib), check=True)


def extract_front_image(request_json: Path) -> Path:
    request = load_json(request_json)
    current_camera = ""
    front_image: Path | None = None
    for item in request["requests"][0]["messages"][1]["content"]:
        if item["type"] == "text":
            text = item["text"].strip()
            if text.endswith("camera:"):
                current_camera = text.rstrip(":")
        elif item["type"] == "image" and current_camera == "Front camera":
            front_image = Path(item["image"])
    if front_image is None:
        raise FileNotFoundError(f"Could not find front image in {request_json}")
    return front_image


def load_history_yaw(ego_history_rot_npy: str) -> np.ndarray:
    hist_rot = np.load(ego_history_rot_npy).astype(np.float32)[0, 0]
    return wrap_angles(yaw_from_rot(hist_rot))


def build_sample_artifact(
    *,
    row: dict[str, Any],
    output_json: Path,
    dataset_root: Path,
    history_len: int,
    artifact_root: Path,
) -> dict[str, Any]:
    normalized_metadata = {
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "front_frame_id": int(row["front_frame_id"]),
        "t0_utc_ns": int(row["t0_utc_ns"]),
        "t0_us": int(row["t0_us"]),
        "target_offset_s": 0.0,
        "actual_offset_s": 0.0,
        "request_json": str(row["request_json"]),
        "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
        "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
        "selected_frames": row["selected_frames"],
    }
    final_summary, ac_summary, gt_summary, _ = build_result_artifacts(
        output_path=output_json,
        metadata=normalized_metadata,
        dataset_root=dataset_root,
        history_len=history_len,
        artifact_root=artifact_root,
    )
    hist_yaw = load_history_yaw(str(row["ego_history_rot_npy"]))
    hist_yaw_rel = unwrap_angles(hist_yaw) - float(unwrap_angles(hist_yaw)[-1])
    pred_curvature = np.asarray(final_summary["pred_curvature"], dtype=np.float32)
    avg_curvature = float(pred_curvature.mean()) if len(pred_curvature) else 0.0
    final_y = float(final_summary["pred_xyz"][-1][1])
    gt_y = float(gt_summary["pred_xyz"][-1][1])
    drift_y = final_y - gt_y
    return {
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "front_frame_id": int(row["front_frame_id"]),
        "request_json": str(row["request_json"]),
        "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
        "front_image": str(extract_front_image(Path(row["request_json"]))),
        "history_yaw_rel": hist_yaw_rel.astype(np.float32).tolist(),
        "gt_final_x_m": float(row["gt_final_x_m"]),
        "gt_final_y_m": float(row["gt_final_y_m"]),
        "gt_yaw_range_rad": float(row["gt_yaw_range_rad"]),
        "final_y_m": final_y,
        "gt_y_m": gt_y,
        "drift_y_m": drift_y,
        "avg_pred_curvature": avg_curvature,
        "avg_pred_curvature_sign": 1 if avg_curvature > 0 else (-1 if avg_curvature < 0 else 0),
        "output_text": final_summary.get("final_output", ""),
        "timing": final_summary.get("timing", {}),
        "final_summary": final_summary,
        "gt_summary": gt_summary,
    }


def plot_topdown(ax: plt.Axes, row: dict[str, Any]) -> None:
    hist_xyz = np.load(row["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    final_xyz = np.asarray(row["final_summary"]["pred_xyz"], dtype=np.float32)
    gt_xyz = np.asarray(row["gt_summary"]["pred_xyz"], dtype=np.float32)
    ax.set_facecolor("#0f1117")
    ax.plot(-hist_xyz[:, 1], hist_xyz[:, 0], color="#9ca3af", linewidth=1.5, label="ego history")
    ax.plot(-final_xyz[:, 1], final_xyz[:, 0], color="#60a5fa", linewidth=2.1, label="generated")
    ax.plot(-gt_xyz[:, 1], gt_xyz[:, 0], color="#34d399", linewidth=1.8, linestyle="--", label="GT")
    ax.scatter([0], [0], color="white", s=24, marker="x")

    all_x = np.concatenate([-hist_xyz[:, 1], -final_xyz[:, 1], -gt_xyz[:, 1]])
    all_y = np.concatenate([hist_xyz[:, 0], final_xyz[:, 0], gt_xyz[:, 0]])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad_x = max(0.8, 0.15 * (x_max - x_min))
    pad_y = max(1.0, 0.15 * (y_max - y_min))
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


def build_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 4.6 * len(rows)), dpi=180)
    fig.patch.set_facecolor("#0b1117")
    gs = GridSpec(len(rows), 3, figure=fig, width_ratios=[1.15, 1.0, 1.2], hspace=0.26, wspace=0.18)

    for row_idx, row in enumerate(rows):
        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(Image.open(row["front_image"]).convert("RGB"))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_facecolor("#111827")
        for spine in ax_img.spines.values():
            spine.set_color("#374151")
        ax_img.set_title(
            f"chunk {row['chunk_id']:04d} sample {row['sample_id']} | front image",
            fontsize=10,
            color="white",
            pad=8,
        )

        ax_traj = fig.add_subplot(gs[row_idx, 1])
        plot_topdown(ax_traj, row)
        if row_idx == 0:
            ax_traj.legend(facecolor="#111827", edgecolor="#374151", framealpha=0.95, fontsize=7, loc="upper left")
        ax_traj.set_title("Top-down path", fontsize=10, color="white", pad=8)

        ax_text = fig.add_subplot(gs[row_idx, 2])
        ax_text.set_facecolor("#111827")
        ax_text.axis("off")
        timing = row["timing"]
        drift_side = "left" if row["drift_y_m"] > 0 else ("right" if row["drift_y_m"] < 0 else "neutral")
        text = "\n".join(
            [
                f"GT final y: {row['gt_y_m']:.3f} m",
                f"Pred final y: {row['final_y_m']:.3f} m",
                f"Lateral drift: {row['drift_y_m']:+.3f} m ({drift_side})",
                f"Avg pred curvature: {row['avg_pred_curvature']:+.6f}",
                f"GT yaw range: {row['gt_yaw_range_rad']:.4f} rad",
                f"Latency: {timing.get('total_post_vlm_ms', float('nan')):.1f} ms",
                "",
                row["output_text"] or "(no final output)",
            ]
        )
        ax_text.text(0.04, 0.96, text, va="top", ha="left", color="white", fontsize=9, linespacing=1.5)

    fig.suptitle("Straight-Sample Bias Report", color="white", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    request_bank_roots = args.request_bank_root or [p for p in DEFAULT_REQUEST_BANK_ROOTS if p.exists()]
    if not request_bank_roots:
        raise RuntimeError("No request bank roots found")

    for path, description in [
        (args.dataset_root, "dataset root"),
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engineDir"),
        (args.multimodal_engine_dir, "multimodalEngineDir"),
        (args.fm_engine, "fmEngine"),
    ]:
        ensure_exists(path, description)

    args.work_root.mkdir(parents=True, exist_ok=True)
    subset_request_root = args.work_root / "subset_requests"
    output_root = args.work_root / "outputs"
    artifacts_root = args.work_root / "artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    selected = select_straight_candidates(
        request_bank_roots=request_bank_roots,
        dataset_root=args.dataset_root,
        future_len=args.future_len,
        history_len=args.history_len,
        dt_s=args.dt_s,
        min_forward_m=args.min_forward_m,
        max_abs_final_y_m=args.max_abs_final_y_m,
        max_yaw_range_rad=args.max_yaw_range_rad,
        max_samples=args.max_samples,
        max_per_chunk=args.max_per_chunk,
    )
    if not selected:
        raise RuntimeError("No straight-ish candidates found under current thresholds")

    prepare_subset_request_root(selected, subset_request_root)
    run_inference_on_subset(
        request_root=subset_request_root,
        output_root=output_root,
        llm_inference_bin=args.llm_inference_bin,
        plugin_lib=args.plugin_lib,
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        warmup=args.warmup,
        timeout_per_request=args.timeout_per_request,
        skip_existing=args.skip_existing,
    )

    rows: list[dict[str, Any]] = []
    for row in selected:
        output_name = Path(row["request_json"]).name.replace("request_", "output_")
        output_json = output_root / output_name
        ensure_exists(output_json, f"model output for sample {row['sample_id']}")
        sample_artifact_root = artifacts_root / Path(row["request_json"]).stem
        sample_artifact_root.mkdir(parents=True, exist_ok=True)
        rows.append(
            build_sample_artifact(
                row=row,
                output_json=output_json,
                dataset_root=args.dataset_root,
                history_len=args.history_len,
                artifact_root=sample_artifact_root,
            )
        )

    drift_values = np.asarray([row["drift_y_m"] for row in rows], dtype=np.float64)
    curv_values = np.asarray([row["avg_pred_curvature"] for row in rows], dtype=np.float64)
    left_count = int(np.sum(drift_values > 0))
    right_count = int(np.sum(drift_values < 0))
    contact_sheet = args.work_root / "straight_bias_contact_sheet.png"
    build_contact_sheet(rows, contact_sheet)

    summary = {
        "selected_request_bank_roots": [str(p) for p in request_bank_roots],
        "thresholds": {
            "min_forward_m": args.min_forward_m,
            "max_abs_final_y_m": args.max_abs_final_y_m,
            "max_yaw_range_rad": args.max_yaw_range_rad,
        },
        "num_samples": len(rows),
        "drift_stats": {
            "mean_drift_y_m": float(drift_values.mean()),
            "median_drift_y_m": float(np.median(drift_values)),
            "std_drift_y_m": float(drift_values.std()),
            "left_count": left_count,
            "right_count": right_count,
        },
        "curvature_stats": {
            "mean_avg_pred_curvature": float(curv_values.mean()),
            "median_avg_pred_curvature": float(np.median(curv_values)),
            "positive_count": int(np.sum(curv_values > 0)),
            "negative_count": int(np.sum(curv_values < 0)),
        },
        "samples": [
            {
                "chunk_id": row["chunk_id"],
                "sample_id": row["sample_id"],
                "front_frame_id": row["front_frame_id"],
                "request_json": row["request_json"],
                "front_image": row["front_image"],
                "gt_final_x_m": row["gt_final_x_m"],
                "gt_final_y_m": row["gt_final_y_m"],
                "gt_yaw_range_rad": row["gt_yaw_range_rad"],
                "final_y_m": row["final_y_m"],
                "gt_y_m": row["gt_y_m"],
                "drift_y_m": row["drift_y_m"],
                "avg_pred_curvature": row["avg_pred_curvature"],
                "avg_pred_curvature_sign": row["avg_pred_curvature_sign"],
                "output_text": row["output_text"],
                "timing": row["timing"],
            }
            for row in rows
        ],
        "contact_sheet_png": str(contact_sheet),
    }
    (args.work_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
