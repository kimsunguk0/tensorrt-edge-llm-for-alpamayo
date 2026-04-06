#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path("/root/TensorRT-Edge-LLM-v060")
STEP_COLORS = {
    10: "#1f77b4",
    8: "#2ca02c",
    6: "#ff7f0e",
    4: "#d62728",
    2: "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FM step sweep trajectories on one figure")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "output" / "benchmarks" / "real_rutuime" / "fm_mxfp8_step_sweep_4chunk",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "dashboards" / "real_rutuime" / "fm_mxfp8_step_sweep_paths_5way",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--chunks", nargs="+", default=["chunk0000", "chunk0005", "chunk0010", "chunk0015"])
    return parser.parse_args()


def parse_sample_tag(path: Path) -> str:
    stem = path.stem
    if stem.startswith("output_"):
        return stem[len("output_") :]
    raise ValueError(f"Unexpected output filename: {path}")


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    data = np.asarray(field["data"], dtype=np.float32)
    shape = tuple(int(x) for x in field["shape"])
    return data.reshape(shape)


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


def extract_text(output_text: str | None) -> str:
    if not output_text:
        return ""
    text = str(output_text)
    if "<|cot_end|>" in text:
        text = text.split("<|cot_end|>", 1)[0]
    return text.strip()


def build_sample_map(run_root: Path, steps: list[int], chunks: list[str]) -> dict[str, dict[int, Path]]:
    sample_map: dict[str, dict[int, Path]] = {}
    for step in steps:
        step_dir = run_root / f"step{step:02d}"
        for path in sorted(step_dir.glob("output_chunk*.json")):
            sample_tag = parse_sample_tag(path)
            if not any(sample_tag.startswith(chunk) for chunk in chunks):
                continue
            sample_map.setdefault(sample_tag, {})[step] = path
    for sample_tag, mapping in sample_map.items():
        missing = sorted(set(steps) - set(mapping))
        if missing:
            raise FileNotFoundError(f"Missing steps {missing} for sample {sample_tag}")
    return dict(sorted(sample_map.items()))


def draw_sample(sample_tag: str, outputs: dict[int, Path], out_dir: Path) -> Path:
    objs = {step: json.loads(path.read_text()) for step, path in outputs.items()}
    request_file = Path(next(iter(objs.values()))["input_file"])
    sample = load_request_context(request_file)
    hist_xyz = sample["hist_xyz"]
    hist_rot = sample["hist_rot"]

    trajectories: dict[int, np.ndarray] = {}
    rotations: dict[int, np.ndarray] = {}
    texts: dict[int, str] = {}
    fm_wall: dict[int, float] = {}
    total_post: dict[int, float] = {}
    end_xy = []

    for step, obj in sorted(objs.items()):
        resp = obj["responses"][0]
        post = resp["alpamayo_post_vlm"]
        pred_xyz = reshape_tensor_field(post["fm"]["pred_xyz"])[0]
        pred_rot = reshape_tensor_field(post["fm"]["pred_rot"])[0]
        trajectories[step] = pred_xyz
        rotations[step] = pred_rot
        texts[step] = extract_text(resp.get("output_text"))
        fm_wall[step] = float(post["timing"]["fm_wall_ms"])
        total_post[step] = float(post["timing"]["total_post_vlm_ms"])
        end_xy.append(pred_xyz[:, :2])

    all_xy = np.vstack([hist_xyz[:, :2], *end_xy])
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    pad_x = max(1.0, (x_max - x_min) * 0.12)
    pad_y = max(1.0, (y_max - y_min) * 0.12)

    fig = plt.figure(figsize=(22, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.2, 1.2], width_ratios=[1.0, 1.0, 1.0, 1.15])

    for idx, (title, image_path) in enumerate(zip(sample["camera_titles"], sample["latest_images"], strict=False)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(Image.open(image_path))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    ax_traj = fig.add_subplot(gs[1:, 0:3])
    ax_info = fig.add_subplot(gs[1:, 3])
    ax_info.axis("off")

    ax_traj.plot(hist_xyz[:, 0], hist_xyz[:, 1], color="black", linewidth=2.4, alpha=0.85, label="ego history")
    hist_step = 3
    ax_traj.quiver(
        hist_xyz[::hist_step, 0],
        hist_xyz[::hist_step, 1],
        hist_rot[::hist_step, 0, 0],
        hist_rot[::hist_step, 1, 0],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.0024,
        alpha=0.45,
    )

    pred_step = 4
    for step in sorted(trajectories, reverse=True):
        pred_xyz = trajectories[step]
        pred_rot = rotations[step]
        color = STEP_COLORS.get(step, None)
        path = np.vstack([hist_xyz[-1:], pred_xyz])
        ax_traj.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.95,
            label=f"step {step}",
        )
        ax_traj.quiver(
            pred_xyz[::pred_step, 0],
            pred_xyz[::pred_step, 1],
            pred_rot[::pred_step, 0, 0],
            pred_rot[::pred_step, 1, 0],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=10,
            width=0.0021,
            alpha=0.45,
        )
        ax_traj.scatter([pred_xyz[-1, 0]], [pred_xyz[-1, 1]], color=color, s=70, zorder=5)

    ax_traj.scatter([0.0], [0.0], color="black", marker="x", s=90, label="current t0")
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_traj.set_ylim(y_min - pad_y, y_max + pad_y)
    ax_traj.set_xlabel("Local X (Forward) [m]")
    ax_traj.set_ylabel("Local Y [m]")
    ax_traj.set_title("Trajectory by diffusion step count", fontsize=14, fontweight="bold")
    ax_traj.grid(True, alpha=0.3, linestyle=":")
    ax_traj.legend(loc="upper left")

    info_lines = []
    for step in sorted(trajectories, reverse=True):
        pred_xyz = trajectories[step]
        end_x, end_y = pred_xyz[-1, 0], pred_xyz[-1, 1]
        text = texts[step] or "(empty)"
        info_lines.append(
            f"step {step}\n"
            f"end_xy: [{end_x:.2f}, {end_y:.2f}]\n"
            f"fm_wall: {fm_wall[step]:.2f} ms\n"
            f"total_post_vlm: {total_post[step]:.2f} ms\n"
            f"text: {textwrap.fill(text, width=28)}"
        )
    ax_info.text(
        0.0,
        1.0,
        "\n\n".join(info_lines),
        va="top",
        fontsize=10.3,
        family="monospace",
    )

    fig.suptitle(
        f"FM MXFP8 Step Sweep | sample={sample_tag}",
        fontsize=17,
        fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_steps_{sample_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    sample_map = build_sample_map(args.run_root, args.steps, args.chunks)
    generated = [draw_sample(sample_tag, outputs, args.out_dir) for sample_tag, outputs in sample_map.items()]
    print(f"generated {len(generated)} compare images")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
