#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ONE_SHOT = REPO_ROOT / "scripts" / "run_raw_dataset_one_shot_udp.py"
DEFAULT_NAV_TEXT = (
    "At the upcoming intersection, make a left turn. Follow the left-turn lane markings "
    "through the intersection. This is an intersection left turn, not a lane change."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare no-nav, nav-text, nav_cfg, 2-step, and 4-step on the same sample."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int, default=1)
    parser.add_argument("--target-offset-s", type=float, default=55.0)
    parser.add_argument("--nav-text", type=str, default=DEFAULT_NAV_TEXT)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "output" / "reports" / "nav_step_comparison_chunk0001_55",
    )
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=None, help="Override FM engine path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_variant(
    *,
    dataset_root: Path,
    chunk_id: int,
    offset_s: float,
    variant_name: str,
    nav_text: str | None,
    diffusion_num_steps: int,
    alpamayo_nav_cfg: bool,
    work_root: Path,
    engine_dir: Path,
    multimodal_engine_dir: Path,
    fm_engine: Path | None,
) -> Path:
    variant_root = work_root / "runs" / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)
    run_tag = f"chunk{chunk_id:04d}_offset_{str(offset_s).replace('.', 'p')}"
    run_root = variant_root / run_tag
    final_path = run_root / "artifacts" / "final_path.json"
    if final_path.exists():
        return run_root

    cmd = [
        sys.executable,
        str(RUN_ONE_SHOT),
        "--dataset-root",
        str(dataset_root),
        "--chunk-id",
        str(chunk_id),
        "--target-offset-s",
        str(offset_s),
        "--work-root",
        str(variant_root),
        "--engine-dir",
        str(engine_dir),
        "--multimodal-engine-dir",
        str(multimodal_engine_dir),
        "--diffusion-num-steps",
        str(diffusion_num_steps),
        "--skip-udp",
    ]
    if nav_text:
        cmd.extend(["--nav-text", nav_text])
    if alpamayo_nav_cfg:
        cmd.append("--alpamayo-nav-cfg")
    if fm_engine is not None:
        cmd.extend(["--fm-engine", str(fm_engine)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return run_root


def discover_run_outputs(run_root: Path) -> dict[str, Path]:
    request_paths = sorted((run_root / "request_bank" / "requests").glob("request_chunk*.json"))
    output_paths = sorted((run_root / "outputs").glob("output_chunk*.json"))
    paths = {
        "request_json": request_paths[0],
        "model_output_json": output_paths[0],
        "final_path_json": run_root / "artifacts" / "final_path.json",
        "ac_decoded_path_json": run_root / "artifacts" / "ac_decoded_path.json",
        "gt_path_json": run_root / "artifacts" / "gt_path.json",
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")
    return paths


def extract_latest_camera_images(request_path: Path) -> list[tuple[str, Path]]:
    request = load_json(request_path)
    items = request["requests"][0]["messages"][1]["content"]
    current_camera = "Camera"
    by_camera: dict[str, Path] = {}
    for item in items:
        if item["type"] == "text":
            text = item["text"].strip()
            if text.endswith("camera:"):
                current_camera = text[:-1]
        elif item["type"] == "image":
            by_camera[current_camera] = Path(item["image"])
    return list(by_camera.items())


def packet_xy(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pts = summary["packet_points"]
    x = np.asarray([p["x_m"] for p in pts], dtype=np.float32)
    y = np.asarray([p["y_m"] for p in pts], dtype=np.float32)
    return x, y


def compute_path_errors(final_summary: dict[str, Any], gt_summary: dict[str, Any]) -> tuple[float, float]:
    fx, fy = packet_xy(final_summary)
    gx, gy = packet_xy(gt_summary)
    n = min(len(fx), len(gx))
    if n == 0:
        return math.nan, math.nan
    err = np.sqrt((fx[:n] - gx[:n]) ** 2 + (fy[:n] - gy[:n]) ** 2)
    return float(err.mean()), float(err[-1])


def plot_path(ax: plt.Axes, final_summary: dict[str, Any], gt_summary: dict[str, Any]) -> None:
    fx, fy = packet_xy(final_summary)
    gx, gy = packet_xy(gt_summary)
    ax.set_facecolor("#0f1117")
    ax.plot(-fy, fx, color="#6ee7ff", linewidth=2.2, label="Final")
    ax.plot(-gy, gx, color="#72e39c", linewidth=1.7, linestyle="--", label="GT")
    ax.scatter([0], [0], color="white", s=24, marker="x", zorder=5)

    all_h = np.concatenate([-fy, -gy])
    all_v = np.concatenate([fx, gx])
    h_min, h_max = float(all_h.min()), float(all_h.max())
    v_min, v_max = float(all_v.min()), float(all_v.max())
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


def build_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    nrows = len(rows)
    fig = plt.figure(figsize=(18, 4.5 * nrows), dpi=170)
    fig.patch.set_facecolor("#0b1117")
    gs = GridSpec(nrows, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1.25, 1.2], hspace=0.28, wspace=0.14)

    for row_idx, row in enumerate(rows):
        for cam_idx, (label, image_path) in enumerate(row["camera_images"][:4]):
            ax = fig.add_subplot(gs[row_idx, cam_idx])
            ax.imshow(Image.open(image_path))
            ax.set_title(label, fontsize=9, color="white", pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#111827")
            for spine in ax.spines.values():
                spine.set_color("#374151")

        path_ax = fig.add_subplot(gs[row_idx, 4])
        plot_path(path_ax, row["final"], row["gt"])
        if row_idx == 0:
            path_ax.legend(facecolor="#111827", edgecolor="#374151", framealpha=0.95, fontsize=7, loc="upper left")

        text_ax = fig.add_subplot(gs[row_idx, 5])
        text_ax.set_facecolor("#111827")
        text_ax.axis("off")
        timing = row["timing"]
        output_text = row["output_text"] or "(no output text)"
        text = "\n".join(
            [
                row["title"],
                f"actual offset: {row['actual_offset_s']:.3f}s",
                f"latency: {timing.get('total_post_vlm_ms', math.nan):.1f} ms",
                f"guided: {timing.get('guided_pass_ms', math.nan):.1f} ms",
                f"fm: {timing.get('fm_wall_ms', math.nan):.1f} ms",
                f"unguided replay: {timing.get('unguided_replay_ms', math.nan):.1f} ms",
                f"fm_mode: {row['fm_mode']}",
                f"steps: {row['fm_num_steps']}",
                f"nav enabled: {row['nav_enabled']}",
                f"ADE/FDE: {row['ade_m']:.3f} / {row['fde_m']:.3f} m",
                "",
                "Final Output:",
                output_text,
            ]
        )
        text_ax.text(0.04, 0.96, text, va="top", ha="left", color="white", fontsize=9, linespacing=1.45)

    fig.suptitle("Nav / Step Comparison On One Sample", color="white", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_root = args.work_root
    report_root.mkdir(parents=True, exist_ok=True)

    variant_specs = [
        {
            "name": "no_nav",
            "title": "No Nav",
            "nav_text": None,
            "diffusion_num_steps": 1,
            "alpamayo_nav_cfg": False,
        },
        {
            "name": "nav_text_only",
            "title": "Nav Text Only",
            "nav_text": args.nav_text,
            "diffusion_num_steps": 1,
            "alpamayo_nav_cfg": False,
        },
        {
            "name": "nav_cfg_on",
            "title": "Nav CFG On",
            "nav_text": args.nav_text,
            "diffusion_num_steps": 1,
            "alpamayo_nav_cfg": True,
        },
        {
            "name": "nav_text_step2",
            "title": "Nav Text + 2-Step",
            "nav_text": args.nav_text,
            "diffusion_num_steps": 2,
            "alpamayo_nav_cfg": False,
        },
        {
            "name": "nav_text_step4",
            "title": "Nav Text + 4-Step",
            "nav_text": args.nav_text,
            "diffusion_num_steps": 4,
            "alpamayo_nav_cfg": False,
        },
    ]

    rows: list[dict[str, Any]] = []
    summary_cases: list[dict[str, Any]] = []
    for spec in variant_specs:
        run_root = run_variant(
            dataset_root=args.dataset_root,
            chunk_id=args.chunk_id,
            offset_s=args.target_offset_s,
            variant_name=spec["name"],
            nav_text=spec["nav_text"],
            diffusion_num_steps=spec["diffusion_num_steps"],
            alpamayo_nav_cfg=spec["alpamayo_nav_cfg"],
            work_root=report_root,
            engine_dir=args.engine_dir,
            multimodal_engine_dir=args.multimodal_engine_dir,
            fm_engine=args.fm_engine,
        )
        discovered = discover_run_outputs(run_root)
        request_json = load_json(discovered["request_json"])
        output_json = load_json(discovered["model_output_json"])
        final_summary = load_json(discovered["final_path_json"])
        gt_summary = load_json(discovered["gt_path_json"])
        post = output_json["responses"][0]["alpamayo_post_vlm"]
        ade_m, fde_m = compute_path_errors(final_summary, gt_summary)

        row = {
            "title": spec["title"],
            "variant_name": spec["name"],
            "chunk_id": args.chunk_id,
            "target_offset_s": args.target_offset_s,
            "actual_offset_s": float(final_summary.get("actual_offset_s", args.target_offset_s)),
            "camera_images": extract_latest_camera_images(discovered["request_json"]),
            "final": final_summary,
            "gt": gt_summary,
            "output_text": output_json["responses"][0].get("output_text"),
            "timing": final_summary.get("timing", {}),
            "fm_mode": post.get("fm_mode"),
            "fm_num_steps": post.get("fm", {}).get("num_steps"),
            "nav_enabled": post.get("nav", {}).get("enabled"),
            "ade_m": ade_m,
            "fde_m": fde_m,
            "run_root": str(run_root),
        }
        rows.append(row)
        summary_cases.append(
            {
                "title": spec["title"],
                "variant_name": spec["name"],
                "chunk_id": args.chunk_id,
                "target_offset_s": args.target_offset_s,
                "actual_offset_s": row["actual_offset_s"],
                "nav_text": spec["nav_text"],
                "alpamayo_nav_cfg": spec["alpamayo_nav_cfg"],
                "requested_diffusion_num_steps": spec["diffusion_num_steps"],
                "reported_fm_num_steps": row["fm_num_steps"],
                "fm_mode": row["fm_mode"],
                "nav_enabled": row["nav_enabled"],
                "output_text": row["output_text"],
                "timing": row["timing"],
                "ade_m": ade_m,
                "fde_m": fde_m,
                "run_root": str(run_root),
                "final_path_json": str(discovered["final_path_json"]),
                "gt_path_json": str(discovered["gt_path_json"]),
                "model_output_json": str(discovered["model_output_json"]),
            }
        )

    latencies = [
        float(case["timing"].get("total_post_vlm_ms", math.nan))
        for case in summary_cases
        if not math.isnan(float(case["timing"].get("total_post_vlm_ms", math.nan)))
    ]
    avg_latency_ms = float(np.mean(latencies)) if latencies else math.nan

    contact_sheet_path = report_root / "nav_step_comparison_contact_sheet.png"
    build_contact_sheet(rows, contact_sheet_path)

    summary_payload = {
        "chunk_id": args.chunk_id,
        "target_offset_s": args.target_offset_s,
        "nav_text": args.nav_text,
        "cases": summary_cases,
        "average_total_post_vlm_ms": avg_latency_ms,
        "contact_sheet_png": str(contact_sheet_path),
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
