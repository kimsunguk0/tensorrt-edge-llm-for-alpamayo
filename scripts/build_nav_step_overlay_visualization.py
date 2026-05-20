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


COLOR_BY_VARIANT = {
    "no_nav": "#94a3b8",
    "nav_text_only": "#60a5fa",
    "nav_cfg_on": "#f97316",
    "nav_text_step2": "#34d399",
    "nav_text_step4": "#f43f5e",
}

LABEL_BY_VARIANT = {
    "no_nav": "No Nav",
    "nav_text_only": "Nav Text",
    "nav_cfg_on": "Nav CFG",
    "nav_text_step2": "2-Step",
    "nav_text_step4": "4-Step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact overlay visualization for nav/step comparison summaries."
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to a nav-step summary.json. May be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_latest_camera_images(request_json: Path) -> list[tuple[str, Path]]:
    request = load_json(request_json)
    items = request["requests"][0]["messages"][1]["content"]
    current_camera = "Camera"
    image_map: dict[str, list[Path]] = {}
    for item in items:
        if item["type"] == "text":
            text = item["text"].strip()
            if text.endswith("camera:"):
                current_camera = text[:-1]
        elif item["type"] == "image":
            image_map.setdefault(current_camera, []).append(Path(item["image"]))

    selected: list[tuple[str, Path]] = []
    for label, paths in image_map.items():
        if not paths:
            continue
        selected.append((label, paths[-1]))
    return selected[:4]


def packet_xy(path_json: Path) -> tuple[np.ndarray, np.ndarray]:
    summary = load_json(path_json)
    pts = summary["packet_points"]
    x = np.asarray([p["x_m"] for p in pts], dtype=np.float32)
    y = np.asarray([p["y_m"] for p in pts], dtype=np.float32)
    return x, y


def configure_axes(ax: plt.Axes, h_values: np.ndarray, v_values: np.ndarray) -> None:
    h_min, h_max = float(h_values.min()), float(h_values.max())
    v_min, v_max = float(v_values.min()), float(v_values.max())
    pad_h = max(1.0, (h_max - h_min) * 0.12)
    pad_v = max(1.5, (v_max - v_min) * 0.12)

    ax.set_xlim(h_max + pad_h, h_min - pad_h)
    ax.set_ylim(max(0.0, v_min - 1.0), v_max + pad_v)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0 if (v_max + pad_v) <= 40 else 10.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0 if (v_max + pad_v) <= 40 else 2.0))
    ax.grid(which="major", color="#334155", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", color="#1f2937", linewidth=0.35, alpha=0.4)
    ax.tick_params(colors="#e5e7eb", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.set_xlabel("Left +y [m]", color="#e5e7eb", fontsize=8)
    ax.set_ylabel("Forward +x [m]", color="#e5e7eb", fontsize=8)


def build_overlay_figure(summary_paths: list[Path], out_path: Path) -> None:
    summaries = [load_json(path) for path in summary_paths]
    nrows = len(summaries)
    fig = plt.figure(figsize=(19, 4.6 * nrows), dpi=170)
    fig.patch.set_facecolor("#0b1117")
    gs = GridSpec(nrows, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1.7, 1.2], hspace=0.28, wspace=0.14)

    for row_idx, summary in enumerate(summaries):
        first_case = summary["cases"][0]
        request_json = Path(first_case["run_root"]) / "request_bank" / "requests"
        request_json = sorted(request_json.glob("request_chunk*.json"))[0]
        camera_images = extract_latest_camera_images(request_json)

        for cam_idx, (label, image_path) in enumerate(camera_images):
            ax = fig.add_subplot(gs[row_idx, cam_idx])
            ax.imshow(Image.open(image_path))
            ax.set_title(label, fontsize=9, color="white", pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#111827")
            for spine in ax.spines.values():
                spine.set_color("#374151")

        path_ax = fig.add_subplot(gs[row_idx, 4])
        path_ax.set_facecolor("#0f1117")

        gt_path = Path(first_case["gt_path_json"])
        gx, gy = packet_xy(gt_path)
        path_ax.plot(-gy, gx, color="#fde68a", linewidth=2.0, linestyle="--", label="GT")
        path_ax.scatter([0], [0], color="white", s=24, marker="x", zorder=5)

        all_h = [-gy]
        all_v = [gx]
        for case in summary["cases"]:
            fx, fy = packet_xy(Path(case["final_path_json"]))
            all_h.append(-fy)
            all_v.append(fx)
            path_ax.plot(
                -fy,
                fx,
                color=COLOR_BY_VARIANT.get(case["variant_name"], "#cbd5e1"),
                linewidth=1.9,
                label=LABEL_BY_VARIANT.get(case["variant_name"], case["variant_name"]),
            )

        configure_axes(path_ax, np.concatenate(all_h), np.concatenate(all_v))
        if row_idx == 0:
            path_ax.legend(facecolor="#111827", edgecolor="#374151", framealpha=0.95, fontsize=7, loc="upper left")

        text_ax = fig.add_subplot(gs[row_idx, 5])
        text_ax.set_facecolor("#111827")
        text_ax.axis("off")
        lines = [
            f"chunk {summary['chunk_id']:04d} @ {summary['target_offset_s']:.1f}s",
            "",
        ]
        for case in summary["cases"]:
            lines.append(
                f"{LABEL_BY_VARIANT.get(case['variant_name'], case['variant_name'])}: "
                f"{case['timing']['total_post_vlm_ms']:.0f} ms | "
                f"ADE {case['ade_m']:.2f} | FDE {case['fde_m']:.2f}"
            )
        lines += [
            "",
            "Output:",
            str(first_case["output_text"]),
        ]
        text_ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", color="white", fontsize=9, linespacing=1.45)

    fig.suptitle("Nav / Step Overlay Comparison", color="white", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    build_overlay_figure([Path(p) for p in args.summary], args.output)
    print(str(args.output))


if __name__ == "__main__":
    main()
