#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path("/root/TensorRT-Edge-LLM-v060")


@dataclass(frozen=True)
class RunPair:
    chunk: str
    sample_tag: str
    left_output: Path
    left_profile: Path
    right_output: Path
    right_profile: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FM compare visuals and summaries")
    parser.add_argument(
        "--left-run-dir",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks" / "real_rutuime" / "fm_fp16_vs_fp8_14chunk" / "runs" / "fm_fp16",
    )
    parser.add_argument(
        "--right-run-dir",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks" / "real_rutuime" / "fm_fp16_vs_fp8_14chunk" / "runs" / "fm_fp8",
    )
    parser.add_argument("--left-label", default="FM FP16")
    parser.add_argument("--right-label", default="FM FP8")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "dashboards" / "real_rutuime" / "fm_fp16_vs_fp8_14chunk_visual_compare",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "output" / "dashboards" / "real_rutuime" / "fm_fp16_vs_fp8_14chunk",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=[f"chunk{i:04d}" for i in range(14)],
    )
    parser.add_argument(
        "--skip-contact-sheet",
        action="store_true",
        help="Generate only per-sample compare PNGs",
    )
    return parser.parse_args()


def parse_sample_tag(path: Path) -> str:
    match = re.search(r"(chunk\d{4}_sid\d+_t0_\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse sample tag from {path}")
    return match.group(1)


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    data = np.asarray(field["data"], dtype=np.float32)
    shape = tuple(int(x) for x in field["shape"])
    return data.reshape(shape)


def load_stage_times(profile_file: Path | None) -> dict[str, float]:
    if profile_file is None or not profile_file.exists():
        return {}
    obj = json.loads(profile_file.read_text())
    out: dict[str, float] = {}
    for stage in obj.get("stages", []):
        out[str(stage["stage_id"])] = float(stage["average_time_per_run_ms"])
    return out


def extract_cot_text(output_text: str | None) -> str:
    if not output_text:
        return ""
    text = str(output_text)
    if "<|cot_end|>" in text:
        text = text.split("<|cot_end|>", 1)[0]
    return text.strip()


def summarize_dashboard_timing(*, response: dict[str, Any], profile_file: Path | None) -> dict[str, float]:
    post_vlm = response.get("alpamayo_post_vlm", {})
    fm = post_vlm.get("fm", {})
    fm_timing = fm.get("timing", {})
    post_timing = post_vlm.get("timing", {})
    stages = load_stage_times(profile_file)

    vision_ms = stages.get("vision_encoder")
    prefill_ms = stages.get("llm_prefill")
    generation_ms = stages.get("llm_generation")
    guided_pass_ms = post_timing.get("guided_pass_ms")
    preprocess_est = None
    if guided_pass_ms is not None and vision_ms is not None and prefill_ms is not None and generation_ms is not None:
        preprocess_est = max(float(guided_pass_ms) - vision_ms - prefill_ms - generation_ms, 0.0)

    fields: dict[str, float] = {}
    if preprocess_est is not None:
        fields["preprocess_est_ms"] = preprocess_est
    if vision_ms is not None:
        fields["vision_encoder_ms"] = vision_ms
    if prefill_ms is not None:
        fields["llm_prefill_ms"] = prefill_ms
    if generation_ms is not None:
        fields["llm_generation_ms"] = generation_ms
    if "ego_history_load_ms" in post_timing:
        fields["ego_history_load_ms"] = float(post_timing["ego_history_load_ms"])
    if "fm_wall_ms" in post_timing:
        fields["fm_wall_ms"] = float(post_timing["fm_wall_ms"])
    if "guided_pass_ms" in post_timing:
        fields["guided_pass_ms"] = float(post_timing["guided_pass_ms"])
    if "total_post_vlm_ms" in post_timing:
        fields["total_post_vlm_ms"] = float(post_timing["total_post_vlm_ms"])
    if "branch_prepare_ms" in fm_timing:
        fields["fm_branch_prepare_ms"] = float(fm_timing["branch_prepare_ms"])
    if "engine_step_total_ms" in fm_timing:
        fields["fm_engine_step_total_ms"] = float(fm_timing["engine_step_total_ms"])
    if "engine_step_avg_ms" in fm_timing:
        fields["fm_engine_step_avg_ms"] = float(fm_timing["engine_step_avg_ms"])
    if "decode_postprocess_ms" in fm_timing:
        fields["fm_decode_post_ms"] = float(fm_timing["decode_postprocess_ms"])
    if "total_ms" in fm_timing:
        fields["fm_total_ms"] = float(fm_timing["total_ms"])
    if "num_steps" in fm_timing:
        fields["fm_num_steps"] = float(fm_timing["num_steps"])
    return fields


def format_timing_block(timing: dict[str, float]) -> str:
    order = [
        ("preprocess_est_ms", "preprocess est"),
        ("vision_encoder_ms", "vision_encoder"),
        ("llm_prefill_ms", "llm_prefill"),
        ("llm_generation_ms", "llm_generation"),
        ("ego_history_load_ms", "ego_history_load"),
        ("fm_wall_ms", "fm_wall"),
        ("fm_branch_prepare_ms", "fm_branch_prepare"),
        ("fm_engine_step_total_ms", "fm_engine_step_total"),
        ("fm_engine_step_avg_ms", "fm_engine_step_avg"),
        ("fm_decode_post_ms", "fm_decode_post"),
        ("total_post_vlm_ms", "total_post_vlm"),
        ("fm_num_steps", "fm_num_steps"),
    ]
    lines = []
    for key, label in order:
        if key in timing:
            value = timing[key]
            if key == "fm_num_steps":
                lines.append(f"{label}: {int(round(value))}")
            else:
                lines.append(f"{label}: {value:.2f} ms")
    return "\n".join(lines) if lines else "timing unavailable"


def load_request_context(request_file: Path) -> dict[str, Any]:
    obj = json.loads(request_file.read_text())
    request = obj["requests"][0]
    user_message = next(msg for msg in request["messages"] if msg.get("role") == "user")

    camera_titles: list[str] = []
    camera_images: list[list[Path]] = []
    current_camera_idx = -1
    for item in user_message["content"]:
        if item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text.lower().endswith("camera:"):
                current_camera_idx += 1
                camera_titles.append(text[:-1])
                camera_images.append([])
        elif item.get("type") == "image" and current_camera_idx >= 0:
            camera_images[current_camera_idx].append(Path(item["image"]))

    latest_images = [paths[-1] for paths in camera_images if paths]
    hist_xyz = np.load(request["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    hist_rot = np.load(request["ego_history_rot_npy"]).astype(np.float32)[0, 0]
    return {
        "camera_titles": camera_titles,
        "latest_images": latest_images,
        "hist_xyz": hist_xyz,
        "hist_rot": hist_rot,
    }


def build_run_pairs(left_run_dir: Path, right_run_dir: Path, chunks: list[str]) -> list[RunPair]:
    pairs: list[RunPair] = []
    for chunk in chunks:
        left_outputs = sorted(left_run_dir.glob(f"output_*_{chunk}_*.json"))
        right_outputs = sorted(right_run_dir.glob(f"output_*_{chunk}_*.json"))
        if len(left_outputs) != 1 or len(right_outputs) != 1:
            raise FileNotFoundError(f"Expected exactly one output in each run dir for {chunk}")
        left_output = left_outputs[0]
        right_output = right_outputs[0]
        sample_tag = parse_sample_tag(left_output)
        pairs.append(
            RunPair(
                chunk=chunk,
                sample_tag=sample_tag,
                left_output=left_output,
                left_profile=left_output.with_name(left_output.name.replace("output_", "profile_")),
                right_output=right_output,
                right_profile=right_output.with_name(right_output.name.replace("output_", "profile_")),
            )
        )
    return pairs


def format_text_block(*, model_label: str, response: dict[str, Any], timing: dict[str, float], end_xyz: np.ndarray) -> str:
    output_text = extract_cot_text(response.get("output_text")) or str(response.get("output_text") or "(empty)")
    wrapped_text = textwrap.fill(output_text, width=58)
    return (
        f"{model_label}\n"
        f"end_xyz: [{end_xyz[0]:.3f}, {end_xyz[1]:.3f}, {end_xyz[2]:.3f}]\n\n"
        f"CoC / response:\n{wrapped_text}\n\n"
        f"Timing:\n{format_timing_block(timing)}"
    )


def compute_row(pair: RunPair) -> dict[str, Any]:
    obj_left = json.loads(pair.left_output.read_text())
    obj_right = json.loads(pair.right_output.read_text())
    if obj_left["input_file"] != obj_right["input_file"]:
        raise ValueError(f"Input mismatch for {pair.sample_tag}")

    resp_left = obj_left["responses"][0]
    resp_right = obj_right["responses"][0]
    pred_xyz_left = reshape_tensor_field(resp_left["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_xyz_right = reshape_tensor_field(resp_right["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_rot_left = reshape_tensor_field(resp_left["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]
    pred_rot_right = reshape_tensor_field(resp_right["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]
    delta_xyz = pred_xyz_right - pred_xyz_left
    delta_rot = pred_rot_right - pred_rot_left
    delta_xy_norm = np.linalg.norm(delta_xyz[:, :2], axis=1)
    timing_left = summarize_dashboard_timing(response=resp_left, profile_file=pair.left_profile)
    timing_right = summarize_dashboard_timing(response=resp_right, profile_file=pair.right_profile)

    row: dict[str, Any] = {
        "chunk": pair.chunk,
        "sample_tag": pair.sample_tag,
        "input_file": obj_left["input_file"],
        "text_same": str(resp_left.get("output_text") or "") == str(resp_right.get("output_text") or ""),
        "pred_xyz_mae": float(np.mean(np.abs(delta_xyz))),
        "pred_xyz_rmse": float(np.sqrt(np.mean(delta_xyz**2))),
        "pred_xyz_max_abs": float(np.max(np.abs(delta_xyz))),
        "pred_rot_mae": float(np.mean(np.abs(delta_rot))),
        "pred_rot_rmse": float(np.sqrt(np.mean(delta_rot**2))),
        "max_xy_delta_m": float(np.max(delta_xy_norm)),
        "end_dx": float(delta_xyz[-1, 0]),
        "end_dy": float(delta_xyz[-1, 1]),
        "end_dz": float(delta_xyz[-1, 2]),
        "left_end_x": float(pred_xyz_left[-1, 0]),
        "left_end_y": float(pred_xyz_left[-1, 1]),
        "left_end_z": float(pred_xyz_left[-1, 2]),
        "right_end_x": float(pred_xyz_right[-1, 0]),
        "right_end_y": float(pred_xyz_right[-1, 1]),
        "right_end_z": float(pred_xyz_right[-1, 2]),
        "left_output_text": str(resp_left.get("output_text") or ""),
        "right_output_text": str(resp_right.get("output_text") or ""),
    }
    for prefix, timing in [("left", timing_left), ("right", timing_right)]:
        for key, value in timing.items():
            row[f"{prefix}_{key}"] = value
    for metric in [
        "vision_encoder_ms",
        "llm_prefill_ms",
        "llm_generation_ms",
        "guided_pass_ms",
        "fm_wall_ms",
        "fm_branch_prepare_ms",
        "fm_engine_step_total_ms",
        "fm_engine_step_avg_ms",
        "fm_decode_post_ms",
        "total_post_vlm_ms",
        "fm_num_steps",
    ]:
        left = row.get(f"left_{metric}")
        right = row.get(f"right_{metric}")
        if left is not None and right is not None:
            row[f"delta_{metric}"] = float(right) - float(left)
    return row


def draw_compare_figure(pair: RunPair, out_dir: Path, *, left_label: str, right_label: str) -> Path:
    obj_left = json.loads(pair.left_output.read_text())
    obj_right = json.loads(pair.right_output.read_text())
    if obj_left["input_file"] != obj_right["input_file"]:
        raise ValueError(f"Input mismatch for {pair.sample_tag}")

    request_file = Path(obj_left["input_file"])
    sample = load_request_context(request_file)
    hist_xyz = sample["hist_xyz"]
    hist_rot = sample["hist_rot"]

    resp_left = obj_left["responses"][0]
    resp_right = obj_right["responses"][0]
    pred_xyz_left = reshape_tensor_field(resp_left["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_xyz_right = reshape_tensor_field(resp_right["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_rot_left = reshape_tensor_field(resp_left["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]
    pred_rot_right = reshape_tensor_field(resp_right["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]
    timing_left = summarize_dashboard_timing(response=resp_left, profile_file=pair.left_profile)
    timing_right = summarize_dashboard_timing(response=resp_right, profile_file=pair.right_profile)

    pred_path_left = np.vstack([hist_xyz[-1:], pred_xyz_left])
    pred_path_right = np.vstack([hist_xyz[-1:], pred_xyz_right])
    delta = pred_xyz_right - pred_xyz_left
    delta_norm = np.linalg.norm(delta[:, :2], axis=1)
    pred_mae = float(np.mean(np.abs(delta)))
    pred_rmse = float(np.sqrt(np.mean(delta**2)))
    pred_max_abs = float(np.max(np.abs(delta)))
    text_same = str(resp_left.get("output_text") or "") == str(resp_right.get("output_text") or "")

    all_xy = np.vstack([hist_xyz[:, :2], pred_xyz_left[:, :2], pred_xyz_right[:, :2]])
    x_min = float(np.min(all_xy[:, 0]))
    x_max = float(np.max(all_xy[:, 0]))
    y_min = float(np.min(all_xy[:, 1]))
    y_max = float(np.max(all_xy[:, 1]))
    pad_x = max(1.0, (x_max - x_min) * 0.10)
    pad_y = max(1.0, (y_max - y_min) * 0.10)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.15, 1.3])

    for idx, (title, image_path) in enumerate(zip(sample["camera_titles"], sample["latest_images"], strict=False)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(Image.open(image_path))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    ax_text16 = fig.add_subplot(gs[1, 0:2])
    ax_text8 = fig.add_subplot(gs[1, 2:4])
    for ax in (ax_text16, ax_text8):
        ax.axis("off")

    ax_text16.text(
        0.0,
        1.0,
        format_text_block(model_label=left_label, response=resp_left, timing=timing_left, end_xyz=pred_xyz_left[-1]),
        va="top",
        fontsize=10.5,
        family="monospace",
    )
    ax_text8.text(
        0.0,
        1.0,
        format_text_block(model_label=right_label, response=resp_right, timing=timing_right, end_xyz=pred_xyz_right[-1]),
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    ax_traj = fig.add_subplot(gs[2, :])
    ax_traj.plot(hist_xyz[:, 0], hist_xyz[:, 1], color="#1f77b4", linewidth=2.2, alpha=0.9, label="ego history")
    hist_step = 3
    ax_traj.quiver(
        hist_xyz[::hist_step, 0],
        hist_xyz[::hist_step, 1],
        hist_rot[::hist_step, 0, 0],
        hist_rot[::hist_step, 1, 0],
        color="#1f77b4",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0026,
        alpha=0.55,
    )
    ax_traj.plot(
        pred_path_left[:, 0],
        pred_path_left[:, 1],
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label=f"pred trajectory ({left_label})",
    )
    ax_traj.plot(
        pred_path_right[:, 0],
        pred_path_right[:, 1],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label=f"pred trajectory ({right_label})",
    )
    pred_step = 4
    ax_traj.quiver(
        pred_xyz_left[::pred_step, 0],
        pred_xyz_left[::pred_step, 1],
        pred_rot_left[::pred_step, 0, 0],
        pred_rot_left[::pred_step, 1, 0],
        color="#2ca02c",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0026,
        alpha=0.55,
    )
    ax_traj.quiver(
        pred_xyz_right[::pred_step, 0],
        pred_xyz_right[::pred_step, 1],
        pred_rot_right[::pred_step, 0, 0],
        pred_rot_right[::pred_step, 1, 0],
        color="#d62728",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0026,
        alpha=0.50,
    )
    ax_traj.scatter([0.0], [0.0], color="black", marker="x", s=85, label="current t0")
    ax_traj.scatter([pred_xyz_left[-1, 0]], [pred_xyz_left[-1, 1]], color="#2ca02c", s=65)
    ax_traj.scatter([pred_xyz_right[-1, 0]], [pred_xyz_right[-1, 1]], color="#d62728", s=65)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_traj.set_ylim(y_min - pad_y, y_max + pad_y)
    ax_traj.set_xlabel("Local X (Forward) [m]")
    ax_traj.set_ylabel("Local Y [m]")
    ax_traj.set_title("Ego history and predicted trajectory", fontsize=13, fontweight="bold")
    ax_traj.grid(True, alpha=0.3, linestyle=":")
    ax_traj.legend(loc="upper left")

    stats_text = (
        f"text_same: {text_same}\n"
        f"pred_xyz_mae: {pred_mae:.4f}\n"
        f"pred_xyz_rmse: {pred_rmse:.4f}\n"
        f"pred_xyz_max_abs: {pred_max_abs:.4f}\n"
        f"end_delta: [{pred_xyz_right[-1, 0] - pred_xyz_left[-1, 0]:.3f}, "
        f"{pred_xyz_right[-1, 1] - pred_xyz_left[-1, 1]:.3f}, {pred_xyz_right[-1, 2] - pred_xyz_left[-1, 2]:.3f}]\n"
        f"max_xy_delta: {float(np.max(delta_norm)):.3f} m"
    )
    ax_traj.text(
        0.985,
        0.02,
        stats_text,
        transform=ax_traj.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90, "edgecolor": "#bdbdbd"},
    )

    fig.suptitle(
        f"{left_label} vs {right_label} | {pair.sample_tag}",
        fontsize=16,
        fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{pair.sample_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_contact_sheet(images: list[Path], out_path: Path) -> Path:
    if not images:
        raise ValueError("No images to tile")

    opened = [Image.open(path).convert("RGB") for path in images]
    thumb_size = (1800, 1080)
    thumbs: list[Image.Image] = []
    for img in opened:
        thumb = img.copy()
        thumb.thumbnail(thumb_size)
        thumbs.append(thumb)

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs)
    margin = 40
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin), (248, 248, 248))

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = margin + col * (cell_w + margin) + (cell_w - thumb.width) // 2
        y = margin + row * (cell_h + margin) + (cell_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    for img in opened:
        img.close()
    return out_path


def write_summary(rows: list[dict[str, Any]], summary_dir: Path) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "summary_rows.csv"
    json_path = summary_dir / "summary.json"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def mean(key: str) -> float | None:
        vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    aggregates = {
        "num_samples": len(rows),
        "text_same_count": int(sum(bool(row["text_same"]) for row in rows)),
        "pred_xyz_mae_mean": mean("pred_xyz_mae"),
        "pred_xyz_rmse_mean": mean("pred_xyz_rmse"),
        "pred_xyz_max_abs_mean": mean("pred_xyz_max_abs"),
        "max_xy_delta_m_mean": mean("max_xy_delta_m"),
        "left_total_post_vlm_ms_mean": mean("left_total_post_vlm_ms"),
        "right_total_post_vlm_ms_mean": mean("right_total_post_vlm_ms"),
        "delta_total_post_vlm_ms_mean": mean("delta_total_post_vlm_ms"),
        "left_fm_wall_ms_mean": mean("left_fm_wall_ms"),
        "right_fm_wall_ms_mean": mean("right_fm_wall_ms"),
        "delta_fm_wall_ms_mean": mean("delta_fm_wall_ms"),
        "left_fm_engine_step_total_ms_mean": mean("left_fm_engine_step_total_ms"),
        "right_fm_engine_step_total_ms_mean": mean("right_fm_engine_step_total_ms"),
        "delta_fm_engine_step_total_ms_mean": mean("delta_fm_engine_step_total_ms"),
        "left_fm_num_steps_mean": mean("left_fm_num_steps"),
        "right_fm_num_steps_mean": mean("right_fm_num_steps"),
    }

    worst_by_rmse = sorted(rows, key=lambda row: float(row["pred_xyz_rmse"]), reverse=True)[:5]
    worst_by_total_delta = sorted(rows, key=lambda row: float(row.get("delta_total_post_vlm_ms", 0.0)), reverse=True)[:5]

    payload = {
        "aggregates": aggregates,
        "worst_pred_xyz_rmse": worst_by_rmse,
        "slowest_fp8_total_post_vlm_delta": worst_by_total_delta,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    pairs = build_run_pairs(args.left_run_dir, args.right_run_dir, args.chunks)
    compare_paths = [
        draw_compare_figure(pair, args.out_dir, left_label=args.left_label, right_label=args.right_label)
        for pair in pairs
    ]
    rows = [compute_row(pair) for pair in pairs]
    csv_path, json_path = write_summary(rows, args.summary_dir)

    print(f"generated {len(compare_paths)} compare images")
    for path in compare_paths:
        print(path)
    print(csv_path)
    print(json_path)
    if not args.skip_contact_sheet:
        contact_path = make_contact_sheet(compare_paths, args.out_dir / "compare_contact_sheet.png")
        print(contact_path)


if __name__ == "__main__":
    main()
