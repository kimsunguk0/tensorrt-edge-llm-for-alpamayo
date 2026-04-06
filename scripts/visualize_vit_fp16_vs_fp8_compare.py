#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    fp16_output: Path
    fp16_profile: Path
    fp8_output: Path
    fp8_profile: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create side-by-side FP16 vs FP8 ViT visual comparisons")
    p.add_argument(
        "--fp16-run-dir",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks" / "real_rutuime" / "vit_fp16_vs_fp8_4chunk" / "runs" / "vit_fp16",
    )
    p.add_argument(
        "--fp8-run-dir",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks" / "real_rutuime" / "vit_fp16_vs_fp8_4chunk" / "runs" / "vit_fp8",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "dashboards" / "real_rutuime" / "vit_fp16_vs_fp8_4chunk_visual_compare",
    )
    p.add_argument(
        "--chunks",
        nargs="+",
        default=["chunk0000", "chunk0005", "chunk0010", "chunk0015"],
    )
    p.add_argument(
        "--skip-contact-sheet",
        action="store_true",
        help="Generate only per-sample compare PNGs",
    )
    return p.parse_args()


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


def summarize_dashboard_timing(*, response: dict[str, Any], profile_file: Path | None) -> list[str]:
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

    lines: list[str] = []
    if preprocess_est is not None:
        lines.append(f"preprocess est: {preprocess_est:.2f} ms")
    if vision_ms is not None:
        lines.append(f"vision_encoder: {vision_ms:.2f} ms")
    if prefill_ms is not None:
        lines.append(f"llm_prefill: {prefill_ms:.2f} ms")
    if generation_ms is not None:
        lines.append(f"llm_generation: {generation_ms:.2f} ms")
    if "ego_history_load_ms" in post_timing:
        lines.append(f"ego_history_load: {float(post_timing['ego_history_load_ms']):.2f} ms")
    if "fm_wall_ms" in post_timing:
        lines.append(f"fm_wall: {float(post_timing['fm_wall_ms']):.2f} ms")
    if "branch_prepare_ms" in fm_timing:
        lines.append(f"fm_branch_prepare: {float(fm_timing['branch_prepare_ms']):.2f} ms")
    if "engine_step_total_ms" in fm_timing:
        lines.append(f"fm_engine_step_total: {float(fm_timing['engine_step_total_ms']):.2f} ms")
    if "decode_postprocess_ms" in fm_timing:
        lines.append(f"fm_decode_post: {float(fm_timing['decode_postprocess_ms']):.2f} ms")
    if "total_post_vlm_ms" in post_timing:
        lines.append(f"total_post_vlm: {float(post_timing['total_post_vlm_ms']):.2f} ms")
    return lines


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


def parse_sample_tag(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(chunk\d{4}_sid\d+_t0_\d+)", stem)
    if not match:
        raise ValueError(f"Could not parse sample tag from {path}")
    return match.group(1)


def build_run_pairs(fp16_run_dir: Path, fp8_run_dir: Path, chunks: list[str]) -> list[RunPair]:
    pairs: list[RunPair] = []
    for chunk in chunks:
        fp16_outputs = sorted(fp16_run_dir.glob(f"output_vit_fp16_fp16_{chunk}_*.json"))
        fp8_outputs = sorted(fp8_run_dir.glob(f"output_vit_fp8_fp16_{chunk}_*.json"))
        if len(fp16_outputs) != 1 or len(fp8_outputs) != 1:
            raise FileNotFoundError(f"Expected exactly one FP16 and one FP8 output for {chunk}")
        fp16_output = fp16_outputs[0]
        fp8_output = fp8_outputs[0]
        sample_tag = parse_sample_tag(fp16_output)
        pairs.append(
            RunPair(
                chunk=chunk,
                sample_tag=sample_tag,
                fp16_output=fp16_output,
                fp16_profile=fp16_output.with_name(fp16_output.name.replace("output_", "profile_")),
                fp8_output=fp8_output,
                fp8_profile=fp8_output.with_name(fp8_output.name.replace("output_", "profile_")),
            )
        )
    return pairs


def format_text_block(
    *,
    model_label: str,
    response: dict[str, Any],
    profile_file: Path,
    end_xyz: np.ndarray,
) -> str:
    output_text = extract_cot_text(response.get("output_text")) or str(response.get("output_text") or "(empty)")
    timing_lines = summarize_dashboard_timing(response=response, profile_file=profile_file)
    wrapped_text = textwrap.fill(output_text, width=58)
    timing_block = "\n".join(timing_lines) if timing_lines else "timing unavailable"
    return (
        f"{model_label}\n"
        f"end_xyz: [{end_xyz[0]:.3f}, {end_xyz[1]:.3f}, {end_xyz[2]:.3f}]\n\n"
        f"CoC / response:\n{wrapped_text}\n\n"
        f"Timing:\n{timing_block}"
    )


def draw_compare_figure(pair: RunPair, out_dir: Path) -> Path:
    fp16_obj = json.loads(pair.fp16_output.read_text())
    fp8_obj = json.loads(pair.fp8_output.read_text())
    if fp16_obj["input_file"] != fp8_obj["input_file"]:
        raise ValueError(f"Input mismatch for {pair.sample_tag}")

    request_file = Path(fp16_obj["input_file"])
    sample = load_request_context(request_file)
    hist_xyz = sample["hist_xyz"]
    hist_rot = sample["hist_rot"]

    resp16 = fp16_obj["responses"][0]
    resp8 = fp8_obj["responses"][0]
    pred_xyz16 = reshape_tensor_field(resp16["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_xyz8 = reshape_tensor_field(resp8["alpamayo_post_vlm"]["fm"]["pred_xyz"])[0]
    pred_rot16 = reshape_tensor_field(resp16["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]
    pred_rot8 = reshape_tensor_field(resp8["alpamayo_post_vlm"]["fm"]["pred_rot"])[0]

    pred_path16 = np.vstack([hist_xyz[-1:], pred_xyz16])
    pred_path8 = np.vstack([hist_xyz[-1:], pred_xyz8])
    delta = pred_xyz8 - pred_xyz16
    delta_norm = np.linalg.norm(delta[:, :2], axis=1)
    pred_mae = float(np.mean(np.abs(delta)))
    pred_rmse = float(np.sqrt(np.mean(delta**2)))
    text16 = str(resp16.get("output_text") or "")
    text8 = str(resp8.get("output_text") or "")
    text_same = text16 == text8

    all_xy = np.vstack([hist_xyz[:, :2], pred_xyz16[:, :2], pred_xyz8[:, :2]])
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
        format_text_block(model_label="ViT FP16", response=resp16, profile_file=pair.fp16_profile, end_xyz=pred_xyz16[-1]),
        va="top",
        fontsize=10.5,
        family="monospace",
    )
    ax_text8.text(
        0.0,
        1.0,
        format_text_block(model_label="ViT FP8", response=resp8, profile_file=pair.fp8_profile, end_xyz=pred_xyz8[-1]),
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
        pred_path16[:, 0],
        pred_path16[:, 1],
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="pred trajectory (FP16 ViT)",
    )
    ax_traj.plot(
        pred_path8[:, 0],
        pred_path8[:, 1],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="pred trajectory (FP8 ViT)",
    )
    pred_step = 4
    ax_traj.quiver(
        pred_xyz16[::pred_step, 0],
        pred_xyz16[::pred_step, 1],
        pred_rot16[::pred_step, 0, 0],
        pred_rot16[::pred_step, 1, 0],
        color="#2ca02c",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0026,
        alpha=0.55,
    )
    ax_traj.quiver(
        pred_xyz8[::pred_step, 0],
        pred_xyz8[::pred_step, 1],
        pred_rot8[::pred_step, 0, 0],
        pred_rot8[::pred_step, 1, 0],
        color="#d62728",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0026,
        alpha=0.50,
    )
    ax_traj.scatter([0.0], [0.0], color="black", marker="x", s=85, label="current t0")
    ax_traj.scatter([pred_xyz16[-1, 0]], [pred_xyz16[-1, 1]], color="#2ca02c", s=65)
    ax_traj.scatter([pred_xyz8[-1, 0]], [pred_xyz8[-1, 1]], color="#d62728", s=65)
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
        f"end_delta: [{pred_xyz8[-1, 0] - pred_xyz16[-1, 0]:.3f}, "
        f"{pred_xyz8[-1, 1] - pred_xyz16[-1, 1]:.3f}, {pred_xyz8[-1, 2] - pred_xyz16[-1, 2]:.3f}]\n"
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
        f"ViT FP16 vs FP8 Comparison | {pair.sample_tag}",
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


def main() -> None:
    args = parse_args()
    pairs = build_run_pairs(args.fp16_run_dir, args.fp8_run_dir, args.chunks)
    compare_paths = [draw_compare_figure(pair, args.out_dir) for pair in pairs]
    print(f"generated {len(compare_paths)} compare images")
    for path in compare_paths:
        print(path)
    if not args.skip_contact_sheet:
        contact_path = make_contact_sheet(compare_paths, args.out_dir / "compare_contact_sheet.png")
        print(contact_path)


if __name__ == "__main__":
    main()
