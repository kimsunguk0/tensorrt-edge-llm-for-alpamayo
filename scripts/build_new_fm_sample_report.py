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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several representative new-FM one-shot cases and build a summary contact sheet."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec in chunk:offset form, for example 1:20 or 2:55. May be repeated.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "output" / "reports" / "new_fm_sample_report",
        help="Root directory for generated report artifacts.",
    )
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=None, help="Override FM engine path.")
    return parser.parse_args()


def parse_case_spec(spec: str) -> tuple[int, float]:
    chunk_text, offset_text = spec.split(":", 1)
    return int(chunk_text), float(offset_text)


def run_case(
    *,
    dataset_root: Path,
    chunk_id: int,
    offset_s: float,
    one_shot_root: Path,
    engine_dir: Path,
    multimodal_engine_dir: Path,
    fm_engine: Path | None,
) -> Path:
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
        str(one_shot_root),
        "--engine-dir",
        str(engine_dir),
        "--multimodal-engine-dir",
        str(multimodal_engine_dir),
        "--skip-udp",
    ]
    if fm_engine is not None:
        cmd.extend(["--fm-engine", str(fm_engine)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    run_tag = f"chunk{chunk_id:04d}_offset_{str(offset_s).replace('.', 'p')}"
    return one_shot_root / run_tag


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _plot_path(ax: plt.Axes, final_summary: dict[str, Any], ac_summary: dict[str, Any], gt_summary: dict[str, Any]) -> None:
    def xy(summary: dict[str, Any]) -> tuple[list[float], list[float]]:
        pts = summary["packet_points"]
        return [p["x_m"] for p in pts], [p["y_m"] for p in pts]

    fx, fy = xy(final_summary)
    axx, ayy = xy(ac_summary)
    gx, gy = xy(gt_summary)

    ax.set_facecolor("#0f1117")
    ax.plot([-v for v in fy], fx, color="#6ee7ff", linewidth=2.3, label="Final")
    ax.plot([-v for v in ayy], axx, color="#ffb86b", linewidth=1.6, linestyle="--", label="AC")
    ax.plot([-v for v in gy], gx, color="#72e39c", linewidth=1.6, label="GT")
    ax.scatter([0], [0], color="white", s=28, marker="x", zorder=5)

    all_h = [-v for v in fy + ayy + gy]
    all_v = fx + axx + gx
    h_min, h_max = min(all_h), max(all_h)
    v_min, v_max = min(all_v), max(all_v)
    pad_h = max(1.0, (h_max - h_min) * 0.12)
    pad_v = max(1.5, (v_max - v_min) * 0.12)
    ax.set_xlim(h_max + pad_h, h_min - pad_h)
    ax.set_ylim(max(0.0, v_min - 1.0), v_max + pad_v)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(10.0 if (v_max + pad_v) > 40 else 5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(2.0 if (v_max + pad_v) > 40 else 1.0))
    ax.grid(which="major", color="#334155", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", color="#1f2937", linewidth=0.35, alpha=0.4)
    ax.tick_params(colors="#e5e7eb", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.set_xlabel("Left +y [m]", color="#e5e7eb", fontsize=8)
    ax.set_ylabel("Forward +x [m]", color="#e5e7eb", fontsize=8)


def build_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    nrows = len(rows)
    fig = plt.figure(figsize=(18, 4.4 * nrows), dpi=170)
    fig.patch.set_facecolor("#0b1117")
    gs = GridSpec(nrows, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1.25, 1.15], hspace=0.28, wspace=0.14)

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
        _plot_path(path_ax, row["final"], row["ac"], row["gt"])
        if row_idx == 0:
            path_ax.legend(facecolor="#111827", edgecolor="#374151", framealpha=0.95, fontsize=7, loc="upper left")

        text_ax = fig.add_subplot(gs[row_idx, 5])
        text_ax.set_facecolor("#111827")
        text_ax.axis("off")
        output_text = row["output_text"] or "(no output text)"
        timing = row["timing"]
        text = "\n".join(
            [
                f"chunk {row['chunk_id']:04d} @ {row['actual_offset_s']:.3f}s",
                f"latency: {timing.get('total_post_vlm_ms', float('nan')):.1f} ms",
                f"guided: {timing.get('guided_pass_ms', float('nan')):.1f} ms",
                f"fm: {timing.get('fm_wall_ms', float('nan')):.1f} ms",
                "",
                "Final Output:",
                output_text,
            ]
        )
        text_ax.text(0.04, 0.96, text, va="top", ha="left", color="white", fontsize=9, linespacing=1.5)

    fig.suptitle("New FM Sample Sweep: 4 Cameras + Path + Latency", color="white", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases = args.case or ["1:20", "1:55", "2:20", "2:55"]
    parsed_cases = [parse_case_spec(spec) for spec in cases]

    report_root = args.work_root
    report_root.mkdir(parents=True, exist_ok=True)
    one_shot_root = report_root / "runs"
    one_shot_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary_cases: list[dict[str, Any]] = []
    for chunk_id, offset_s in parsed_cases:
        run_tag = f"chunk{chunk_id:04d}_offset_{str(offset_s).replace('.', 'p')}"
        run_root = one_shot_root / run_tag
        if not (run_root / "artifacts" / "final_path.json").exists():
            run_root = run_case(
                dataset_root=args.dataset_root,
                chunk_id=chunk_id,
                offset_s=offset_s,
                one_shot_root=one_shot_root,
                engine_dir=args.engine_dir,
                multimodal_engine_dir=args.multimodal_engine_dir,
                fm_engine=args.fm_engine,
            )
        discovered = discover_run_outputs(run_root)
        final_summary = load_json(discovered["final_path_json"])
        ac_summary = load_json(discovered["ac_decoded_path_json"])
        gt_summary = load_json(discovered["gt_path_json"])
        request_path = discovered["request_json"]
        output_json = load_json(discovered["model_output_json"])
        post = output_json["responses"][0]["alpamayo_post_vlm"]
        output_text = output_json["responses"][0].get("output_text")

        row = {
            "chunk_id": chunk_id,
            "target_offset_s": offset_s,
            "actual_offset_s": float(final_summary.get("actual_offset_s", offset_s)),
            "camera_images": extract_latest_camera_images(request_path),
            "final": final_summary,
            "ac": ac_summary,
            "gt": gt_summary,
            "output_text": output_text,
            "timing": final_summary.get("timing", {}),
            "fm_engine": post.get("fm_engine"),
            "run_root": str(run_root),
        }
        rows.append(row)
        summary_cases.append(
            {
                "chunk_id": chunk_id,
                "target_offset_s": offset_s,
                "actual_offset_s": float(final_summary.get("actual_offset_s", offset_s)),
                "output_text": output_text,
                "timing": final_summary.get("timing", {}),
                "run_root": str(run_root),
                "fm_engine": post.get("fm_engine"),
                "final_path_json": str(discovered["final_path_json"]),
                "ac_decoded_path_json": str(discovered["ac_decoded_path_json"]),
                "gt_path_json": str(discovered["gt_path_json"]),
            }
        )

    latencies = [float(row["timing"].get("total_post_vlm_ms", math.nan)) for row in rows]
    latencies = [x for x in latencies if not math.isnan(x)]
    avg_latency_ms = float(np.mean(latencies)) if latencies else math.nan

    contact_sheet_path = report_root / "new_fm_contact_sheet.png"
    build_contact_sheet(rows, contact_sheet_path)

    summary_payload = {
        "cases": summary_cases,
        "average_total_post_vlm_ms": avg_latency_ms,
        "num_cases": len(summary_cases),
        "contact_sheet_png": str(contact_sheet_path),
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
