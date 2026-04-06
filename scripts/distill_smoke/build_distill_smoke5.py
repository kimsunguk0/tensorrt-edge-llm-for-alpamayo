#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build distillation smoke dataset (5 samples) from existing live_runtime outputs")
    p.add_argument("--runs-dir", default="/root/TensorRT-Edge-LLM-v060/output/runs/live_runtime")
    p.add_argument("--traj-dir", default="/root/TensorRT-Edge-LLM-v060/output/trajectories/live_runtime")
    p.add_argument("--image-dir", default="/root/TensorRT-Edge-LLM-v060/input/images/live_runtime")
    p.add_argument("--ego-dir", default="/root/TensorRT-Edge-LLM-v060/input/ego/live_runtime")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--output-pt", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/distill_smoke_5.pt")
    p.add_argument("--output-json", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/distill_smoke_5.summary.json")
    return p.parse_args()


def run_name_from_path(path: Path, prefix: str) -> str:
    name = path.stem
    if not name.startswith(prefix):
        raise ValueError(f"unexpected file name: {path}")
    return name[len(prefix):]


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    traj_dir = Path(args.traj_dir)
    image_dir = Path(args.image_dir)
    ego_dir = Path(args.ego_dir)
    output_pt = Path(args.output_pt)
    output_json = Path(args.output_json)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    output_files = sorted(Path(p) for p in glob.glob(str(runs_dir / "output_*.json")))
    traj_files = sorted(Path(p) for p in glob.glob(str(traj_dir / "trajectory_*.json")))

    out_map = {run_name_from_path(p, "output_"): p for p in output_files}
    traj_map = {run_name_from_path(p, "trajectory_"): p for p in traj_files}
    common_runs = sorted(set(out_map.keys()) & set(traj_map.keys()))
    if not common_runs:
        raise RuntimeError("No matching output_*.json and trajectory_*.json pairs found")

    selected = common_runs[-args.num_samples :]

    image_files = sorted(str(p) for p in image_dir.glob("cam*_f*.png"))
    ego_xyz_path = ego_dir / "ego_history_xyz.npy"
    ego_rot_path = ego_dir / "ego_history_rot.npy"

    samples: list[dict[str, Any]] = []
    for run_name in selected:
        output_obj = json.loads(out_map[run_name].read_text())
        traj_obj = json.loads(traj_map[run_name].read_text())

        response = output_obj["responses"][0]
        post = response.get("alpamayo_post_vlm", {})
        fm = post.get("fm", {})

        sample = {
            "run_name": run_name,
            "paths": {
                "output_json": str(out_map[run_name]),
                "trajectory_json": str(traj_map[run_name]),
                "image_dir": str(image_dir),
                "ego_xyz_npy": str(ego_xyz_path),
                "ego_rot_npy": str(ego_rot_path),
            },
            "meta": {
                "sequence": int(traj_obj.get("sequence", -1)),
                "t0_us": int(traj_obj.get("t0_us", -1)),
                "clip_id": str(traj_obj.get("clip_id", "")),
                "camera_indices": traj_obj.get("camera_indices", []),
                "camera_order": traj_obj.get("camera_order", []),
                "fm_mode": traj_obj.get("fm_mode"),
                "fm_status": traj_obj.get("fm_status"),
                "nav": traj_obj.get("nav"),
            },
            "images": {
                "files": image_files,
            },
            "teacher": {
                "engine_path": fm.get("engine_path"),
                "num_steps": int(fm.get("num_steps", -1)),
                "seed": int(fm.get("seed", -1)),
                "action_space_constants": fm.get("action_space_constants", {}),
                "x0": torch.tensor(traj_obj["x0"], dtype=torch.float32),
                "x_final": torch.tensor(traj_obj["x_final"], dtype=torch.float32),
                "pred_xyz": torch.tensor(traj_obj["pred_xyz"], dtype=torch.float32),
                "pred_rot": torch.tensor(traj_obj["pred_rot"], dtype=torch.float32),
            },
            "text": {
                "output_text": traj_obj.get("output_text", ""),
                "formatted_system_prompt": response.get("formatted_system_prompt", ""),
                "formatted_complete_request": response.get("formatted_complete_request", ""),
            },
            "timing": {
                "post_vlm_timing": post.get("timing", {}),
                "fm_timing": fm.get("timing", {}),
            },
        }
        samples.append(sample)

    dataset = {
        "meta": {
            "dataset_name": "distill_smoke_5",
            "num_samples": len(samples),
            "source_runs_dir": str(runs_dir),
            "source_traj_dir": str(traj_dir),
            "selection": selected,
        },
        "samples": samples,
    }

    torch.save(dataset, output_pt)

    summary = {
        "output_pt": str(output_pt),
        "num_samples": len(samples),
        "selected_runs": selected,
        "has_ego_history_files": ego_xyz_path.exists() and ego_rot_path.exists(),
        "num_image_files": len(image_files),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(output_pt)
    print(output_json)


if __name__ == "__main__":
    main()
