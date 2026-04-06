#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_live_chunk_model_timeline as single


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-chunk model timeline viewers into one combined viewer.")
    parser.add_argument("--viewer-dirs", nargs="+", required=True)
    parser.add_argument("--request-bank-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-name", default="chunks_merged")
    args = parser.parse_args()

    if len(args.viewer_dirs) != len(args.request_bank_roots):
        raise ValueError("--viewer-dirs and --request-bank-roots must have the same length")

    ensure_dir(args.output_dir)
    images_bank = args.output_dir / "images_bank"
    ensure_dir(images_bank)

    all_samples = []
    world_points = []
    world_t0_xy = []
    local_extent = 1.0
    control_dt_s = None
    control_points = None
    chunk_ids = []

    for viewer_dir_str, request_bank_root_str in zip(args.viewer_dirs, args.request_bank_roots, strict=True):
        viewer_dir = Path(viewer_dir_str)
        request_bank_root = Path(request_bank_root_str)
        summary = json.loads((viewer_dir / "summary.json").read_text())
        data_json = Path(summary["viewer_data_json"])
        data = json.loads(data_json.read_text())

        if control_dt_s is None:
            control_dt_s = float(data["control_dt_s"])
            control_points = int(data["control_points"])
        else:
            if abs(control_dt_s - float(data["control_dt_s"])) > 1e-6 or control_points != int(data["control_points"]):
                raise RuntimeError("Inconsistent control packet settings between viewers")

        chunk_ids.append(str(data["chunk_id"]))
        local_extent = max(local_extent, float(data["local_extent_m"]))

        for sample in data["samples"]:
            all_samples.append(sample)
            world_t0_xy.append(sample["history_world_xy"][-1])
            world_points.extend(sample["history_world_xy"])
            world_points.extend(sample["plan_world_xy"])
            world_points.extend(sample["packet_world_xy"])

            image_stem = Path(sample["images"]["0"]).parts[1]
            safe_symlink(request_bank_root / "images" / image_stem, images_bank / image_stem)

    if not all_samples:
        raise RuntimeError("No samples loaded")

    all_samples.sort(key=lambda item: int(item["t0_us"]))
    t0_us0 = int(all_samples[0]["t0_us"])
    for sample in all_samples:
        sample["t_rel_s"] = round((int(sample["t0_us"]) - t0_us0) / 1e6, 3)

    world_points_arr = np.asarray(world_points, dtype=np.float32)
    world_t0_arr = np.asarray(world_t0_xy, dtype=np.float32)
    world_plan_stack = np.stack([np.asarray(sample["plan_world_xy"], dtype=np.float32) for sample in all_samples], axis=0)
    world_bounds = {
        "min_x": float(np.min(world_points_arr[:, 0])),
        "max_x": float(np.max(world_points_arr[:, 0])),
        "min_y": float(np.min(world_points_arr[:, 1])),
        "max_y": float(np.max(world_points_arr[:, 1])),
    }

    combined_data = {
        "chunk_id": args.combined_name,
        "num_samples": len(all_samples),
        "control_dt_s": control_dt_s,
        "control_points": control_points,
        "local_extent_m": local_extent,
        "world_bounds": world_bounds,
        "world_path_t0_xy": world_t0_arr.round(5).tolist(),
        "samples": all_samples,
    }

    data_json_out = args.output_dir / f"{args.combined_name}_model_viewer_data.json"
    data_json_out.write_text(json.dumps(combined_data), encoding="utf-8")

    html_out = args.output_dir / f"{args.combined_name}_model_timeline_viewer.html"
    single.write_viewer_html(html_out, data_json_out.name)

    overview_out = args.output_dir / f"{args.combined_name}_model_timeline_overview.png"
    single.write_overview_png(overview_out, world_t0_arr, world_plan_stack)

    summary_out = {
        "combined_name": args.combined_name,
        "source_chunk_ids": chunk_ids,
        "num_samples_in_viewer": len(all_samples),
        "viewer_html": str(html_out),
        "viewer_data_json": str(data_json_out),
        "overview_png": str(overview_out),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
