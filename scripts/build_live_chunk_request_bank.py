#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

CAMERA_RUNTIME_ORDER = [
    ("left", "camera_left", "left_frame_id", 0),
    ("front", "camera_front", "front_frame_id", 1),
    ("right", "camera_right", "right_frame_id", 2),
    ("front_tele", "camera_front_tele", "front_tele_frame_id", 6),
]

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    6: "Front telephoto camera",
}

CAMERA_SEMANTIC_NAMES = [item[0] for item in CAMERA_RUNTIME_ORDER]

DEFAULT_ACTION_SPACE_CONSTANTS = {
    "accel_mean": 0.029052734375,
    "accel_std": 0.6796875,
    "curvature_mean": 0.0002689361572265625,
    "curvature_std": 0.026123046875,
    "dt_value": 0.1,
    "v_lambda": 0.000001,
    "v_ridge": 0.0001,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * slat * slat)
    x = (N + alt_m) * clat * clon
    y = (N + alt_m) * clat * slon
    z = (N * (1.0 - E2_WGS84) + alt_m) * slat
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu(xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref_xyz = geodetic_to_ecef(
        np.asarray([ref_lat_deg], dtype=np.float64),
        np.asarray([ref_lon_deg], dtype=np.float64),
        np.asarray([ref_alt_m], dtype=np.float64),
    )[0]
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)
    rot = np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )
    return (xyz - ref_xyz) @ rot.T


def yaw_to_rot(yaw_rad: np.ndarray) -> np.ndarray:
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    rot = np.zeros((len(yaw_rad), 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 1] = -sin_y
    rot[:, 1, 0] = sin_y
    rot[:, 1, 1] = cos_y
    rot[:, 2, 2] = 1.0
    return rot


def build_user_content(image_dir: Path, camera_indices: list[int], nav_text: str | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for cam_id in camera_indices:
        cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
        content.append({"type": "text", "text": f"{cam_name}: "})
        for frame_idx in range(4):
            image_path = image_dir / f"cam{cam_id}_f{frame_idx}.png"
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image", "image": str(image_path)})

    hist_placeholder = "<|traj_history_start|>" + ("<|traj_history|>" * 48) + "<|traj_history_end|>"
    route_section = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else ""
    prompt_text = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    content.append({"type": "text", "text": f"{hist_placeholder}{route_section}{prompt_text}"})
    return content


def nearest_frame_indices(frame_ts_ns: np.ndarray, target_ts_ns: np.ndarray) -> np.ndarray:
    right = np.searchsorted(frame_ts_ns, target_ts_ns, side="left")
    right = np.clip(right, 0, len(frame_ts_ns) - 1)
    left = np.clip(right - 1, 0, len(frame_ts_ns) - 1)
    choose_right = np.abs(frame_ts_ns[right] - target_ts_ns) < np.abs(frame_ts_ns[left] - target_ts_ns)
    return np.where(choose_right, right, left).astype(np.int64)


def build_pose_history(
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    t0_utc_ns: int,
    history_len: int,
    dt_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    hist_times = np.asarray([t0_utc_ns - (history_len - 1 - i) * dt_ns for i in range(history_len)], dtype=np.int64)
    all_times = hist_times

    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    interp_lat = np.interp(all_times, utc, lat)
    interp_lon = np.interp(all_times, utc, lon)
    interp_alt = np.interp(all_times, utc, alt)

    world_xyz = ecef_to_enu(geodetic_to_ecef(interp_lat, interp_lon, interp_alt), *ref_lla).astype(np.float32)

    vel_xy = np.zeros((len(world_xyz), 2), dtype=np.float32)
    vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_ns / 1e9)
    vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_ns / 1e9)
    vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_ns / 1e9)
    yaw = np.arctan2(vel_xy[:, 1], vel_xy[:, 0]).astype(np.float32)
    world_rot = yaw_to_rot(yaw)

    p0 = world_xyz[-1]
    R0 = world_rot[-1]
    R0_t = R0.T
    hist_local_xyz = ((world_xyz - p0) @ R0).astype(np.float32)
    hist_local_rot = np.einsum("ij,tjk->tik", R0_t, world_rot).astype(np.float32)
    return hist_local_xyz[None, None, ...], hist_local_rot[None, None, ...]


def extract_unique_frames(
    video_path: Path,
    frame_index_to_out_path: dict[int, Path],
    width: int,
    height: int,
) -> None:
    if not frame_index_to_out_path:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    wanted = sorted(frame_index_to_out_path.keys())
    wanted_pos = 0
    next_idx = wanted[wanted_pos]
    frame_idx = 0
    while wanted_pos < len(wanted):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Unexpected EOF in {video_path} at frame {frame_idx}")
        if frame_idx == next_idx:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            out_path = frame_index_to_out_path[frame_idx]
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), resized)
            wanted_pos += 1
            if wanted_pos < len(wanted):
                next_idx = wanted[wanted_pos]
        frame_idx += 1
    cap.release()


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    ensure_dir(link_path.parent)
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sample runtime requests for one raw live chunk.")
    parser.add_argument("--dataset-root", default="/root/live_dataset/2025-03-31-test2")
    parser.add_argument("--chunk-id", type=int, default=11)
    parser.add_argument(
        "--output-root",
        default="/root/TensorRT-Edge-LLM-v060/output/benchmarks/real_rutuime/chunk0011_model_request_bank",
    )
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=10)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--camera-semantics",
        nargs="+",
        default=None,
        choices=CAMERA_SEMANTIC_NAMES,
        help="Subset of camera semantic names to include. Default uses all cameras.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    request_root = out_root / "requests"
    image_root = out_root / "images"
    ego_root = out_root / "ego"
    cache_root = out_root / "frame_cache"
    ensure_dir(request_root)
    ensure_dir(image_root)
    ensure_dir(ego_root)
    ensure_dir(cache_root)

    selected_semantics = set(args.camera_semantics or CAMERA_SEMANTIC_NAMES)
    selected_camera_order = [item for item in CAMERA_RUNTIME_ORDER if item[0] in selected_semantics]
    if not selected_camera_order:
        raise RuntimeError("No cameras selected for request bank generation")

    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[["frame_id", "chunk_id"]]
    sample_index = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    samples = sample_index[sample_index["chunk_id"] == args.chunk_id].copy().sort_values("t0_utc_ns").reset_index(drop=True)
    if args.limit > 0:
        samples = samples.iloc[: args.limit].copy().reset_index(drop=True)
    if samples.empty:
        raise RuntimeError(f"No samples found for chunk {args.chunk_id}")

    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)
    first_t0 = int(samples["t0_utc_ns"].iloc[0])
    ref_lla = (
        float(np.interp([first_t0], utc, lat)[0]),
        float(np.interp([first_t0], utc, lon)[0]),
        float(np.interp([first_t0], utc, alt)[0]),
    )

    offsets_ns = np.asarray([-300_000_000, -200_000_000, -100_000_000, 0], dtype=np.int64)
    t0s = samples["t0_utc_ns"].to_numpy(dtype=np.int64)
    per_camera_selected: dict[str, np.ndarray] = {}

    for semantic_name, sensor_name, _, _ in selected_camera_order:
        frame_df = pd.read_parquet(dataset_root / "sensors" / sensor_name / "frames.parquet")
        frame_df = frame_df[frame_df["chunk_id"] == args.chunk_id].copy().sort_values("timestamp_utc_ns")
        frame_ts = frame_df["timestamp_utc_ns"].to_numpy(dtype=np.int64)
        frame_ids = frame_df["frame_id"].to_numpy(dtype=np.int64)
        frame_idx_in_chunk = frame_df["frame_index_in_chunk"].to_numpy(dtype=np.int64)
        chosen = np.zeros((len(samples), len(offsets_ns)), dtype=np.int64)
        for j, offset_ns in enumerate(offsets_ns):
            target = t0s + offset_ns
            nearest_idx = nearest_frame_indices(frame_ts, target)
            chosen[:, j] = frame_ids[nearest_idx]

        unique_frame_ids = np.unique(chosen.reshape(-1))
        id_to_chunk_idx = dict(zip(frame_ids.tolist(), frame_idx_in_chunk.tolist(), strict=True))
        frame_index_to_out_path: dict[int, Path] = {}
        for frame_id in unique_frame_ids.tolist():
            cache_path = cache_root / semantic_name / f"frame_{frame_id}.png"
            if cache_path.exists():
                continue
            frame_index_to_out_path[id_to_chunk_idx[int(frame_id)]] = cache_path

        extract_unique_frames(
            video_path=dataset_root / "sensors" / sensor_name / "chunks" / f"chunk_{args.chunk_id:04d}.mkv",
            frame_index_to_out_path=frame_index_to_out_path,
            width=args.width,
            height=args.height,
        )
        per_camera_selected[semantic_name] = chosen

    request_paths: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    dt_ns = int(round(args.dt_s * 1e9))
    camera_indices = [item[3] for item in selected_camera_order]

    for idx, row in samples.iterrows():
        sample_id = int(row["sample_id"])
        t0_utc_ns = int(row["t0_utc_ns"])
        t0_us = t0_utc_ns // 1000
        stem = f"chunk{args.chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}"

        hist_xyz, hist_rot = build_pose_history(
            gnss_valid=gnss_valid,
            ref_lla=ref_lla,
            t0_utc_ns=t0_utc_ns,
            history_len=args.history_len,
            dt_ns=dt_ns,
        )

        sample_ego_dir = ego_root / stem
        ensure_dir(sample_ego_dir)
        xyz_path = sample_ego_dir / "ego_history_xyz.npy"
        rot_path = sample_ego_dir / "ego_history_rot.npy"
        np.save(xyz_path, hist_xyz.astype(np.float32))
        np.save(rot_path, hist_rot.astype(np.float32))

        sample_image_dir = image_root / stem
        ensure_dir(sample_image_dir)
        selected_frame_record: dict[str, list[int]] = {}
        for semantic_name, _, _, cam_id in selected_camera_order:
            chosen_ids = per_camera_selected[semantic_name][idx].tolist()
            selected_frame_record[semantic_name] = [int(x) for x in chosen_ids]
            for step_idx, frame_id in enumerate(chosen_ids):
                cache_path = cache_root / semantic_name / f"frame_{int(frame_id)}.png"
                link_path = sample_image_dir / f"cam{cam_id}_f{step_idx}.png"
                safe_symlink(cache_path, link_path)

        request = {
            "batch_size": 1,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_generate_length": int(args.max_generate_length),
            "apply_chat_template": True,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "enable_thinking": False,
            "requests": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a driving assistant that generates safe and accurate actions.",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": build_user_content(sample_image_dir, camera_indices, args.nav_text),
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "<|cot_start|>"}],
                        },
                    ],
                    "ego_history_xyz_npy": str(xyz_path),
                    "ego_history_rot_npy": str(rot_path),
                    "traj_token_offset": int(args.traj_token_offset),
                    "action_space_constants": dict(DEFAULT_ACTION_SPACE_CONSTANTS),
                    "diffusion_seed": int(args.diffusion_seed),
                    "diffusion_num_steps": int(args.diffusion_num_steps),
                }
            ],
        }
        if args.nav_text:
            request["requests"][0]["nav_text"] = args.nav_text
            request["requests"][0]["nav_guidance_weight"] = 3.0

        request_path = request_root / f"request_{stem}.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False))
        request_paths.append(str(request_path))

        manifest_rows.append(
            {
                "chunk_id": args.chunk_id,
                "sample_id": sample_id,
                "t0_utc_ns": t0_utc_ns,
                "t0_us": t0_us,
                "request_json": str(request_path),
                "ego_history_xyz_npy": str(xyz_path),
                "ego_history_rot_npy": str(rot_path),
                "selected_frames": selected_frame_record,
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "chunk_id": args.chunk_id,
        "num_requests": len(request_paths),
        "sample_id_min": int(samples["sample_id"].min()),
        "sample_id_max": int(samples["sample_id"].max()),
        "t0_start_utc_ns": int(samples["t0_utc_ns"].min()),
        "t0_end_utc_ns": int(samples["t0_utc_ns"].max()),
        "width": args.width,
        "height": args.height,
        "history_len": args.history_len,
        "dt_s": args.dt_s,
        "camera_indices": camera_indices,
        "camera_runtime_order": [
            {"semantic_name": semantic_name, "camera_id": cam_id}
            for semantic_name, _, _, cam_id in selected_camera_order
        ],
        "chunk_ref_lla": [float(x) for x in ref_lla],
        "output_root": str(out_root),
        "request_root": str(request_root),
        "image_root": str(image_root),
        "ego_root": str(ego_root),
        "frame_cache_root": str(cache_root),
    }

    (out_root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
