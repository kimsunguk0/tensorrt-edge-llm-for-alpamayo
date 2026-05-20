#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_raw_dataset_one_shot_udp import build_gt_future_local_pose, yaw_from_rot


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def symlink_dir(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=True)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x)).astype(np.float32)


def cumulative_planar_distance(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    xy = xyz[:, :2]
    origin = np.zeros((1, 2), dtype=np.float32)
    seg = np.diff(np.vstack([origin, xy]), axis=0)
    seg_dist = np.linalg.norm(seg, axis=1)
    return np.cumsum(seg_dist, dtype=np.float32)


def round_distance_m(distance_m: float, round_to_m: float) -> int:
    if round_to_m <= 0:
        return int(round(distance_m))
    return max(int(round(distance_m / round_to_m) * round_to_m), int(round_to_m))


def derive_oracle_nav(
    gt_xyz: np.ndarray,
    gt_rot: np.ndarray,
    *,
    turn_threshold_deg: float,
    onset_threshold_deg: float,
    ahead_distance_m: float,
    round_to_m: float,
    straight_text: str,
) -> dict[str, Any]:
    # `gt_rot` is already in the current ego-local frame, so each future yaw is
    # an absolute heading relative to "now". Re-unwrapping it can fabricate
    # multi-turn rotations from small sign flips near pi/-pi.
    yaw = wrap_to_pi(yaw_from_rot(np.asarray(gt_rot, dtype=np.float32)))
    cumdist = cumulative_planar_distance(gt_xyz)
    final_yaw_rad = float(yaw[-1]) if len(yaw) else 0.0
    final_yaw_deg = math.degrees(final_yaw_rad)

    onset_threshold_rad = math.radians(onset_threshold_deg)
    turn_threshold_rad = math.radians(turn_threshold_deg)
    turn_idx = None
    for idx, yaw_rad in enumerate(yaw):
        if abs(float(yaw_rad)) >= onset_threshold_rad:
            turn_idx = idx
            break
    onset_distance_m = float(cumdist[turn_idx]) if turn_idx is not None else None
    total_distance_m = float(cumdist[-1]) if len(cumdist) else 0.0

    if abs(final_yaw_rad) < turn_threshold_rad:
        return {
            "maneuver": "straight",
            "nav_text": straight_text,
            "final_yaw_deg": final_yaw_deg,
            "turn_onset_distance_m": onset_distance_m,
            "total_distance_m": total_distance_m,
        }

    direction = "left" if final_yaw_rad > 0.0 else "right"
    if onset_distance_m is None or onset_distance_m <= ahead_distance_m:
        nav_text = f"turn {direction} ahead"
    else:
        nav_text = f"turn {direction} in {round_distance_m(onset_distance_m, round_to_m)}m"

    return {
        "maneuver": f"turn_{direction}",
        "nav_text": nav_text,
        "final_yaw_deg": final_yaw_deg,
        "turn_onset_distance_m": onset_distance_m,
        "total_distance_m": total_distance_m,
    }


def rewrite_request_paths(request_obj: dict[str, Any], input_root: Path, output_root: Path) -> None:
    req = request_obj["requests"][0]
    for key in ("ego_history_xyz_npy", "ego_history_rot_npy"):
        old_path = Path(req[key])
        try:
            rel = old_path.relative_to(input_root)
        except ValueError:
            continue
        req[key] = str(output_root / rel)

    for msg in req.get("messages", []):
        for content in msg.get("content", []):
            if content.get("type") != "image":
                continue
            old_path = Path(content["image"])
            try:
                rel = old_path.relative_to(input_root)
            except ValueError:
                continue
            content["image"] = str(output_root / rel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a live chunk request bank and inject per-request oracle nav text derived from GT future path."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-request-bank-root", type=Path, required=True)
    parser.add_argument("--output-request-bank-root", type=Path, required=True)
    parser.add_argument("--future-len", type=int, default=64)
    parser.add_argument("--turn-threshold-deg", type=float, default=30.0)
    parser.add_argument("--turn-onset-threshold-deg", type=float, default=10.0)
    parser.add_argument("--ahead-distance-m", type=float, default=12.0)
    parser.add_argument("--distance-round-to-m", type=float, default=5.0)
    parser.add_argument("--straight-text", type=str, default="continue straight")
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_request_bank_root.resolve()
    output_root = args.output_request_bank_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input request bank root not found: {input_root}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output request bank root already exists: {output_root} (pass --force to overwrite)")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_manifest = load_json(input_root / "manifest.json")
    input_summary = load_json(input_root / "summary.json")
    history_len = int(input_summary["history_len"])
    dt_s = float(input_summary["dt_s"])

    symlink_dir(input_root / "images", output_root / "images")
    symlink_dir(input_root / "ego", output_root / "ego")
    if (input_root / "frame_cache").exists():
        symlink_dir(input_root / "frame_cache", output_root / "frame_cache")

    request_out_root = output_root / "requests"
    request_out_root.mkdir(parents=True, exist_ok=True)

    output_manifest: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    maneuver_counts: Counter[str] = Counter()

    for row in input_manifest:
        t0_utc_ns = int(row["t0_utc_ns"])
        request_name = Path(row["request_json"]).name
        input_request_path = input_root / "requests" / request_name
        output_request_path = request_out_root / request_name

        gt_xyz, gt_rot = build_gt_future_local_pose(
            dataset_root=args.dataset_root,
            t0_utc_ns=t0_utc_ns,
            future_len=int(args.future_len),
            dt_s=dt_s,
            history_len=history_len,
        )
        oracle_nav = derive_oracle_nav(
            gt_xyz,
            gt_rot,
            turn_threshold_deg=float(args.turn_threshold_deg),
            onset_threshold_deg=float(args.turn_onset_threshold_deg),
            ahead_distance_m=float(args.ahead_distance_m),
            round_to_m=float(args.distance_round_to_m),
            straight_text=str(args.straight_text),
        )

        request_obj = load_json(input_request_path)
        rewrite_request_paths(request_obj, input_root, output_root)
        req0 = request_obj["requests"][0]
        req0["nav_text"] = oracle_nav["nav_text"]
        req0["nav_guidance_weight"] = float(args.nav_guidance_weight)
        req0["oracle_nav_source"] = "gt_future_path"
        write_json(output_request_path, request_obj)

        output_row = dict(row)
        output_row["request_json"] = str(output_request_path)
        for key in ("ego_history_xyz_npy", "ego_history_rot_npy"):
            old_path = Path(str(output_row[key]))
            try:
                rel = old_path.relative_to(input_root)
            except ValueError:
                continue
            output_row[key] = str(output_root / rel)
        output_manifest.append(output_row)

        maneuver_counts[oracle_nav["maneuver"]] += 1
        oracle_rows.append(
            {
                "request_json": str(output_request_path),
                "sample_id": int(row["sample_id"]),
                "chunk_id": int(row["chunk_id"]),
                "t0_utc_ns": t0_utc_ns,
                "nav_text": oracle_nav["nav_text"],
                "maneuver": oracle_nav["maneuver"],
                "final_yaw_deg": oracle_nav["final_yaw_deg"],
                "turn_onset_distance_m": oracle_nav["turn_onset_distance_m"],
                "total_distance_m": oracle_nav["total_distance_m"],
            }
        )

    output_summary = dict(input_summary)
    output_summary["output_root"] = str(output_root)
    output_summary["request_root"] = str(output_root / "requests")
    output_summary["image_root"] = str(output_root / "images")
    output_summary["ego_root"] = str(output_root / "ego")
    if (output_root / "frame_cache").exists():
        output_summary["frame_cache_root"] = str(output_root / "frame_cache")
    output_summary["oracle_nav"] = {
        "source": "gt_future_path",
        "future_len": int(args.future_len),
        "turn_threshold_deg": float(args.turn_threshold_deg),
        "turn_onset_threshold_deg": float(args.turn_onset_threshold_deg),
        "ahead_distance_m": float(args.ahead_distance_m),
        "distance_round_to_m": float(args.distance_round_to_m),
        "straight_text": str(args.straight_text),
        "nav_guidance_weight": float(args.nav_guidance_weight),
        "maneuver_counts": dict(maneuver_counts),
    }

    write_json(output_root / "manifest.json", output_manifest)
    write_json(output_root / "summary.json", output_summary)
    write_json(output_root / "oracle_nav_manifest.json", oracle_rows)

    preview = oracle_rows[:10]
    print(
        json.dumps(
            {
                "output_request_bank_root": str(output_root),
                "num_requests": len(oracle_rows),
                "maneuver_counts": dict(maneuver_counts),
                "preview": preview,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
