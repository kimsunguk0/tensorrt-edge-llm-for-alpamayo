#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_team_replay_common import (
    COORD_MODE_LOCAL,
    build_packet_dict,
    pack_packet,
)
from fm_model_defaults import first_existing_fm_engine


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.float32).reshape(tuple(field["shape"]))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def wrap_angle_scalar(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script requires pandas + parquet support in the active Python environment. "
            "Install pandas and pyarrow first."
        ) from exc
    return pd


def load_request_bank():
    try:
        import build_live_chunk_request_bank as request_bank
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import build_live_chunk_request_bank dependencies. "
            "Make sure pandas, pyarrow, and opencv-python are installed in the active Python environment."
        ) from exc
    return request_bank


def load_chunk_frame_table(dataset_root: Path, sensor_name: str, chunk_id: int) -> Any:
    pd = load_pandas()
    frame_path = dataset_root / "sensors" / sensor_name / "frames.parquet"
    ensure_exists(frame_path, f"{sensor_name} frames parquet")
    frame_df = pd.read_parquet(frame_path)
    frame_df = frame_df[frame_df["chunk_id"] == chunk_id].copy().sort_values("timestamp_utc_ns").reset_index(drop=True)
    if frame_df.empty:
        raise RuntimeError(f"No frames found for {sensor_name} chunk {chunk_id}")
    return frame_df


def select_target_front_frame(dataset_root: Path, chunk_id: int, target_offset_s: float) -> dict[str, Any]:
    request_bank = load_request_bank()
    front_df = load_chunk_frame_table(dataset_root, "camera_front", chunk_id)
    frame_ts = front_df["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    target_ts = int(frame_ts[0] + round(target_offset_s * 1e9))
    nearest_idx = int(request_bank.nearest_frame_indices(frame_ts, np.asarray([target_ts], dtype=np.int64))[0])
    row = front_df.iloc[nearest_idx]
    return {
        "frame_id": int(row["frame_id"]),
        "frame_index_in_chunk": int(row["frame_index_in_chunk"]),
        "timestamp_utc_ns": int(row["timestamp_utc_ns"]),
        "chunk_start_utc_ns": int(frame_ts[0]),
        "target_offset_s": float(target_offset_s),
        "actual_offset_s": float((int(row["timestamp_utc_ns"]) - int(frame_ts[0])) / 1e9),
    }


def build_single_request(
    *,
    dataset_root: Path,
    chunk_id: int,
    anchor: dict[str, Any],
    out_root: Path,
    history_len: int,
    dt_s: float,
    width: int,
    height: int,
    nav_text: str | None,
    traj_token_offset: int,
    diffusion_seed: int,
    diffusion_num_steps: int,
    max_generate_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[Path, dict[str, Any]]:
    request_bank = load_request_bank()
    pd = load_pandas()
    request_root = out_root / "requests"
    image_root = out_root / "images"
    ego_root = out_root / "ego"
    cache_root = out_root / "frame_cache"
    for path in (request_root, image_root, ego_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)

    t0_utc_ns = int(anchor["timestamp_utc_ns"])
    t0_us = t0_utc_ns // 1000
    front_frame_id = int(anchor["frame_id"])
    stem = f"chunk{chunk_id:04d}_fid{front_frame_id:06d}_t0_{t0_us}"

    gnss_path = dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
    ensure_exists(gnss_path, "gnss_ins parquet")
    gnss = pd.read_parquet(gnss_path)
    # Match request-bank pose generation: the GNSS table contains multiple row
    # types at duplicated timestamps, and only the quality-tagged pose rows are
    # stable enough for GT trajectory interpolation.
    gnss_valid = request_bank.select_pose_gnss_rows(gnss)
    if gnss_valid.empty:
        raise RuntimeError("No valid GNSS rows found for pose history generation")

    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)
    ref_lla = (
        float(np.interp([t0_utc_ns], utc, lat)[0]),
        float(np.interp([t0_utc_ns], utc, lon)[0]),
        float(np.interp([t0_utc_ns], utc, alt)[0]),
    )

    offsets_ns = np.asarray([-300_000_000, -200_000_000, -100_000_000, 0], dtype=np.int64)
    selected_frame_record: dict[str, list[int]] = {}
    camera_indices = [item[3] for item in request_bank.CAMERA_RUNTIME_ORDER]

    sample_image_dir = image_root / stem
    sample_ego_dir = ego_root / stem
    sample_image_dir.mkdir(parents=True, exist_ok=True)
    sample_ego_dir.mkdir(parents=True, exist_ok=True)

    for semantic_name, sensor_name, _, cam_id in request_bank.CAMERA_RUNTIME_ORDER:
        frame_df = load_chunk_frame_table(dataset_root, sensor_name, chunk_id)
        frame_ts = frame_df["timestamp_utc_ns"].to_numpy(dtype=np.int64)
        frame_ids = frame_df["frame_id"].to_numpy(dtype=np.int64)
        frame_idx_in_chunk = frame_df["frame_index_in_chunk"].to_numpy(dtype=np.int64)
        chosen_idx = request_bank.nearest_frame_indices(frame_ts, t0_utc_ns + offsets_ns)
        chosen_ids = frame_ids[chosen_idx]
        selected_frame_record[semantic_name] = [int(x) for x in chosen_ids.tolist()]

        id_to_chunk_idx = dict(zip(frame_ids.tolist(), frame_idx_in_chunk.tolist(), strict=True))
        frame_index_to_out_path: dict[int, Path] = {}
        for frame_id in np.unique(chosen_ids).tolist():
            cache_path = cache_root / semantic_name / f"frame_{int(frame_id)}.png"
            if not cache_path.exists():
                frame_index_to_out_path[id_to_chunk_idx[int(frame_id)]] = cache_path

        request_bank.extract_unique_frames(
            video_path=dataset_root / "sensors" / sensor_name / "chunks" / f"chunk_{chunk_id:04d}.mkv",
            frame_index_to_out_path=frame_index_to_out_path,
            width=width,
            height=height,
        )

        for step_idx, frame_id in enumerate(chosen_ids.tolist()):
            cache_path = cache_root / semantic_name / f"frame_{int(frame_id)}.png"
            link_path = sample_image_dir / f"cam{cam_id}_f{step_idx}.png"
            request_bank.safe_symlink(cache_path, link_path)

    dt_ns = int(round(dt_s * 1e9))
    hist_xyz, hist_rot = request_bank.build_pose_history(
        gnss_valid=gnss_valid,
        ref_lla=ref_lla,
        t0_utc_ns=t0_utc_ns,
        history_len=history_len,
        dt_ns=dt_ns,
    )
    xyz_path = sample_ego_dir / "ego_history_xyz.npy"
    rot_path = sample_ego_dir / "ego_history_rot.npy"
    np.save(xyz_path, hist_xyz.astype(np.float32))
    np.save(rot_path, hist_rot.astype(np.float32))

    request = {
        "batch_size": 1,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_generate_length": int(max_generate_length),
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
                        "content": request_bank.build_user_content(sample_image_dir, camera_indices, nav_text),
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<|cot_start|>"}],
                    },
                ],
                "ego_history_xyz_npy": str(xyz_path),
                "ego_history_rot_npy": str(rot_path),
                "traj_token_offset": int(traj_token_offset),
                "action_space_constants": dict(request_bank.DEFAULT_ACTION_SPACE_CONSTANTS),
                "diffusion_seed": int(diffusion_seed),
                "diffusion_num_steps": int(diffusion_num_steps),
            }
        ],
    }
    if nav_text:
        request["requests"][0]["nav_text"] = nav_text
        request["requests"][0]["nav_guidance_weight"] = 3.0

    request_path = request_root / f"request_{stem}.json"
    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "chunk_id": int(chunk_id),
        "front_frame_id": int(front_frame_id),
        "t0_utc_ns": int(t0_utc_ns),
        "t0_us": int(t0_us),
        "chunk_start_utc_ns": int(anchor["chunk_start_utc_ns"]),
        "target_offset_s": float(anchor["target_offset_s"]),
        "actual_offset_s": float(anchor["actual_offset_s"]),
        "camera_indices": camera_indices,
        "camera_order": [item[0] for item in request_bank.CAMERA_RUNTIME_ORDER],
        "selected_frames": selected_frame_record,
        "request_json": str(request_path),
        "ego_history_xyz_npy": str(xyz_path),
        "ego_history_rot_npy": str(rot_path),
        "action_space_constants": dict(request_bank.DEFAULT_ACTION_SPACE_CONSTANTS),
    }
    (out_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return request_path, metadata


def run_single_request(
    *,
    request_root: Path,
    output_root: Path,
    llm_inference_bin: Path,
    plugin_lib: Path,
    engine_dir: Path,
    multimodal_engine_dir: Path,
    fm_engine: Path,
    warmup: int,
    timeout_per_request: float,
    alpamayo_nav_cfg: bool,
    alpamayo_fm_use_prefill_kv: bool,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_request_bank_persistent.py"),
        "--request-root",
        str(request_root),
        "--output-root",
        str(output_root),
        "--llm-inference-bin",
        str(llm_inference_bin),
        "--plugin-lib",
        str(plugin_lib),
        "--engine-dir",
        str(engine_dir),
        "--multimodal-engine-dir",
        str(multimodal_engine_dir),
        "--fm-engine",
        str(fm_engine),
        "--warmup",
        str(warmup),
        "--timeout-per-request",
        str(timeout_per_request),
        "--limit",
        "1",
    ]
    if alpamayo_nav_cfg:
        cmd.append("--alpamayo-nav-cfg")
    if alpamayo_fm_use_prefill_kv:
        cmd.append("--alpamayo-fm-use-prefill-kv")
    subprocess.run(cmd, check=True)
    outputs = sorted(output_root.glob("output_chunk*.json"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected exactly one output file under {output_root}, found {len(outputs)}")
    return outputs[0]


def compute_traj_kinematics(pred_xyz: np.ndarray, pred_rot: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    yaw = yaw_from_rot(np.asarray(pred_rot, dtype=np.float32))
    x = np.concatenate([np.asarray([0.0], dtype=np.float32), pred_xyz[:, 0].astype(np.float32)])
    y = np.concatenate([np.asarray([0.0], dtype=np.float32), pred_xyz[:, 1].astype(np.float32)])
    yaw_full = wrap_angles(np.concatenate([np.asarray([0.0], dtype=np.float32), yaw]))

    v = np.zeros_like(x, dtype=np.float32)
    if len(x) > 1:
        deltas = np.stack([np.diff(x), np.diff(y)], axis=1)
        seg_speed = np.linalg.norm(deltas, axis=1).astype(np.float32) / max(float(dt_s), 1e-6)
        v[0] = seg_speed[0]
        v[1:] = seg_speed

    curvature = np.zeros_like(x, dtype=np.float32)
    if len(x) > 2:
        deltas = np.stack([np.diff(x), np.diff(y)], axis=1)
        ds = np.linalg.norm(deltas, axis=1).astype(np.float32)
        yaw_unwrapped = np.unwrap(yaw_full.astype(np.float64)).astype(np.float32)
        dyaw = np.diff(yaw_unwrapped)
        curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
        curvature[0] = curvature[1]
    return x, y, yaw_full, v, curvature


def compute_planar_history_travel(hist_xyz: np.ndarray) -> float:
    hist_xyz = np.asarray(hist_xyz, dtype=np.float64)
    if len(hist_xyz) < 2:
        return 0.0
    diffs = hist_xyz[1:, :2] - hist_xyz[:-1, :2]
    return float(np.linalg.norm(diffs, axis=1).sum())


def sanitize_initial_forward_speed(v0: float, hist_xyz: np.ndarray) -> float:
    clamped_v0 = max(0.0, float(v0))
    history_travel = compute_planar_history_travel(hist_xyz)
    if history_travel <= 0.20 or clamped_v0 <= 0.25:
        return 0.0
    return clamped_v0


def add_third_order_dtd(lhs: np.ndarray, scale: float) -> None:
    n = lhs.shape[0]
    if n < 4:
        return
    coeffs = np.asarray([-1.0, 3.0, -3.0, 1.0], dtype=np.float64)
    for row in range(n - 3):
        lhs[row : row + 4, row : row + 4] += scale * np.outer(coeffs, coeffs)


def estimate_v0_one(hist_xyz: np.ndarray, hist_rot: np.ndarray, dt_s: float, v_lambda: float, v_ridge: float) -> float:
    hist_xyz = np.asarray(hist_xyz, dtype=np.float64)
    hist_rot = np.asarray(hist_rot, dtype=np.float64)
    history_len = hist_xyz.shape[0]
    if history_len < 2:
        raise RuntimeError("History length must be at least 2")

    n = history_len - 1
    theta = np.zeros(history_len, dtype=np.float64)
    prev_raw = 0.0
    for t in range(history_len):
        raw = float(np.arctan2(hist_rot[t, 1, 0], hist_rot[t, 0, 0]))
        if t == 0:
            theta[t] = raw
        else:
            theta[t] = theta[t - 1] + wrap_angle_scalar(raw - prev_raw)
        prev_raw = raw

    vdim = n + 1
    lhs = np.zeros((vdim, vdim), dtype=np.float64)
    rhs = np.zeros(vdim, dtype=np.float64)

    for t in range(n):
        dx = float(hist_xyz[t + 1, 0] - hist_xyz[t, 0])
        dy = float(hist_xyz[t + 1, 1] - hist_xyz[t, 1])
        gx = (2.0 / dt_s) * dx
        gy = (2.0 / dt_s) * dy

        c0 = float(np.cos(theta[t]))
        c1 = float(np.cos(theta[t + 1]))
        s0 = float(np.sin(theta[t]))
        s1 = float(np.sin(theta[t + 1]))

        lhs[t, t] += c0 * c0 + s0 * s0
        lhs[t, t + 1] += c0 * c1 + s0 * s1
        lhs[t + 1, t] += c1 * c0 + s1 * s0
        lhs[t + 1, t + 1] += c1 * c1 + s1 * s1

        rhs[t] += c0 * gx + s0 * gy
        rhs[t + 1] += c1 * gx + s1 * gy

    smooth_scale = float(v_lambda) / float(dt_s**6)
    add_third_order_dtd(lhs, smooth_scale)
    lhs[np.diag_indices_from(lhs)] += float(v_ridge)

    try:
        lower = np.linalg.cholesky(lhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Failed Cholesky factorization in Python FM exact decode") from exc
    y = np.linalg.solve(lower, rhs)
    x = np.linalg.solve(lower.T, y)
    return float(x[-1])


def decode_actions_exact(x_final: np.ndarray, hist_xyz: np.ndarray, hist_rot: np.ndarray, constants: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_final = np.asarray(x_final, dtype=np.float64)
    hist_xyz = np.asarray(hist_xyz, dtype=np.float64)
    hist_rot = np.asarray(hist_rot, dtype=np.float64)
    horizon = int(x_final.shape[0])
    dt_s = float(constants["dt_value"])
    v0 = sanitize_initial_forward_speed(
        estimate_v0_one(hist_xyz, hist_rot, dt_s, float(constants["v_lambda"]), float(constants["v_ridge"])),
        hist_xyz,
    )

    accel = x_final[:, 0] * float(constants["accel_std"]) + float(constants["accel_mean"])
    curvature = x_final[:, 1] * float(constants["curvature_std"]) + float(constants["curvature_mean"])

    velocity = np.zeros(horizon + 1, dtype=np.float64)
    theta = np.zeros(horizon + 1, dtype=np.float64)
    velocity[0] = v0
    for t in range(horizon):
        v_prev = float(velocity[t])
        v_next = max(0.0, v_prev + float(accel[t]) * dt_s)
        velocity[t + 1] = v_next
        theta[t + 1] = theta[t] + float(curvature[t]) * 0.5 * (v_prev + v_next) * dt_s

    pred_xyz = np.zeros((horizon, 3), dtype=np.float32)
    pred_rot = np.zeros((horizon, 3, 3), dtype=np.float32)
    x_cum = 0.0
    y_cum = 0.0
    z0 = float(hist_xyz[-1, 2])
    half_dt = 0.5 * dt_s
    for t in range(horizon):
        x_cum += (velocity[t] * np.cos(theta[t]) + velocity[t + 1] * np.cos(theta[t + 1])) * half_dt
        y_cum += (velocity[t] * np.sin(theta[t]) + velocity[t + 1] * np.sin(theta[t + 1])) * half_dt
        pred_xyz[t, 0] = float(x_cum)
        pred_xyz[t, 1] = float(y_cum)
        pred_xyz[t, 2] = float(z0)
        c = float(np.cos(theta[t + 1]))
        s = float(np.sin(theta[t + 1]))
        pred_rot[t] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return pred_xyz, pred_rot, accel.astype(np.float32), curvature.astype(np.float32)


def build_gt_future_local_pose(
    dataset_root: Path,
    t0_utc_ns: int,
    future_len: int,
    dt_s: float,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    pd = load_pandas()
    request_bank = load_request_bank()
    gnss_path = dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
    ensure_exists(gnss_path, "gnss_ins parquet")
    gnss = pd.read_parquet(gnss_path)
    # The GNSS parquet has duplicated timestamps for different message/quality
    # rows. Use the same pose-row selection as request-bank generation so GT
    # future does not jump between incompatible GNSS rows.
    gnss_valid = request_bank.select_pose_gnss_rows(gnss)
    if gnss_valid.empty:
        raise RuntimeError("No valid GNSS rows found for GT future path generation")

    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)
    ref_lla = (
        float(np.interp([t0_utc_ns], utc, lat)[0]),
        float(np.interp([t0_utc_ns], utc, lon)[0]),
        float(np.interp([t0_utc_ns], utc, alt)[0]),
    )

    dt_ns = int(round(dt_s * 1e9))
    hist_times = np.asarray([t0_utc_ns - (history_len - 1 - i) * dt_ns for i in range(history_len)], dtype=np.int64)
    fut_times = np.asarray([t0_utc_ns + (i + 1) * dt_ns for i in range(future_len)], dtype=np.int64)
    all_times = np.concatenate([hist_times, fut_times], axis=0)

    interp_lat = np.interp(all_times, utc, lat)
    interp_lon = np.interp(all_times, utc, lon)
    interp_alt = np.interp(all_times, utc, alt)
    world_xyz = request_bank.ecef_to_enu(
        request_bank.geodetic_to_ecef(interp_lat, interp_lon, interp_alt),
        *ref_lla,
    ).astype(np.float32)

    world_vel_xy = np.zeros((len(all_times), 2), dtype=np.float32)
    world_vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_ns / 1e9)
    world_vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_ns / 1e9)
    world_vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_ns / 1e9)
    world_yaw = np.arctan2(world_vel_xy[:, 1], world_vel_xy[:, 0]).astype(np.float32)
    world_rot = request_bank.yaw_to_rot(world_yaw)

    hist_world_xyz = world_xyz[:history_len]
    fut_world_xyz = world_xyz[history_len:]
    hist_world_rot = world_rot[:history_len]
    fut_world_rot = world_rot[history_len:]
    p0 = hist_world_xyz[-1]
    r0 = hist_world_rot[-1]
    r0_t = r0.T
    future_local_xyz = ((fut_world_xyz - p0) @ r0).astype(np.float32)
    future_local_rot = np.einsum("ij,tjk->tik", r0_t, fut_world_rot).astype(np.float32)
    return future_local_xyz, future_local_rot


def build_path_summary(
    *,
    label: str,
    metadata: dict[str, Any],
    dt_s: float,
    pred_xyz: np.ndarray,
    pred_rot: np.ndarray,
    final_output: str,
    timing: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    x, y, yaw, v, curvature = compute_traj_kinematics(pred_xyz, pred_rot, dt_s)
    packet = build_packet_dict(
        tx_seq=0,
        plan_seq=int(metadata["front_frame_id"]),
        sample_id=int(metadata["front_frame_id"]),
        source_t0_us=int(metadata["t0_us"]),
        tx_time_us=int(time.time_ns() // 1000),
        coord_mode=COORD_MODE_LOCAL,
        dt_s=dt_s,
        x=x,
        y=y,
        yaw=yaw,
        v=v,
        curvature=curvature,
    )
    summary = {
        "label": label,
        "chunk_id": int(metadata["chunk_id"]),
        "front_frame_id": int(metadata["front_frame_id"]),
        "t0_utc_ns": int(metadata["t0_utc_ns"]),
        "t0_us": int(metadata["t0_us"]),
        "target_offset_s": float(metadata["target_offset_s"]),
        "actual_offset_s": float(metadata["actual_offset_s"]),
        "final_output": final_output,
        "timing": timing or {},
        "plan_dt_s": dt_s,
        "plan_points_no_origin": int(pred_xyz.shape[0]),
        "traj_points_with_origin": int(x.shape[0]),
        "pred_xyz": np.asarray(pred_xyz, dtype=np.float32).tolist(),
        "pred_yaw_rad": yaw[1:].tolist(),
        "pred_v_mps": v[1:].tolist(),
        "pred_curvature": curvature[1:].tolist(),
    }
    return summary, packet, pack_packet(packet)


def build_result_artifacts(
    output_path: Path,
    metadata: dict[str, Any],
    dataset_root: Path,
    history_len: int,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    response = json.loads(output_path.read_text(encoding="utf-8"))["responses"][0]
    post = response["alpamayo_post_vlm"]
    fm = post["fm"]
    pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
    pred_rot = reshape_tensor_field(fm["pred_rot"])[0]
    x_final = reshape_tensor_field(fm["x_final"])[0]
    dt_s = float(fm["action_space_constants"]["dt_value"])
    final_text = str(response.get("output_text") or "")

    hist_xyz = np.load(metadata["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    hist_rot = np.load(metadata["ego_history_rot_npy"]).astype(np.float32)[0, 0]
    ac_pred_xyz, ac_pred_rot, accel, raw_curvature = decode_actions_exact(
        x_final,
        hist_xyz,
        hist_rot,
        fm["action_space_constants"],
    )
    gt_pred_xyz, gt_pred_rot = build_gt_future_local_pose(
        dataset_root=dataset_root,
        t0_utc_ns=int(metadata["t0_utc_ns"]),
        future_len=int(pred_xyz.shape[0]),
        dt_s=dt_s,
        history_len=history_len,
    )

    final_summary, final_packet, final_packet_bytes = build_path_summary(
        label="final_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        final_output=final_text,
        timing=post.get("timing", {}),
    )
    ac_summary, ac_packet, ac_packet_bytes = build_path_summary(
        label="ac_decoded_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=ac_pred_xyz,
        pred_rot=ac_pred_rot,
        final_output=final_text,
    )
    gt_summary, gt_packet, gt_packet_bytes = build_path_summary(
        label="gt_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=gt_pred_xyz,
        pred_rot=gt_pred_rot,
        final_output="GNSS ground-truth future path",
    )

    final_summary["path_type"] = "final_path"
    final_summary["packet_header"] = final_packet["header"]
    final_summary["packet_points"] = final_packet["points"]

    ac_summary["path_type"] = "ac_decoded_path"
    ac_summary["packet_header"] = ac_packet["header"]
    ac_summary["packet_points"] = ac_packet["points"]
    ac_summary["raw_action"] = {
        "num_points": int(x_final.shape[0]),
        "normalized_x_final": x_final.tolist(),
        "accel_mps2": accel.astype(np.float32).tolist(),
        "curvature": raw_curvature.astype(np.float32).tolist(),
        "action_space_constants": fm["action_space_constants"],
    }

    gt_summary["path_type"] = "gt_path"
    gt_summary["packet_header"] = gt_packet["header"]
    gt_summary["packet_points"] = gt_packet["points"]

    (artifact_root / "final_path.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (artifact_root / "ac_decoded_path.json").write_text(json.dumps(ac_summary, indent=2), encoding="utf-8")
    (artifact_root / "gt_path.json").write_text(json.dumps(gt_summary, indent=2), encoding="utf-8")
    return final_summary, ac_summary, gt_summary, final_packet_bytes


def send_udp_packet(host: str, port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (host, port))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build one raw-dataset sample at a chunk time offset, run Alpamayo once, save final/ac-decode/GT paths, and optionally send the final trajectory via the original UDP packet format."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int, default=1)
    parser.add_argument("--target-offset-s", type=float, required=True, help="Offset from chunk start time in seconds.")
    parser.add_argument("--work-root", type=Path, default=repo_root / "output" / "raw_dataset_one_shot")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--alpamayo-nav-cfg", action="store_true")
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true")
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--llm-inference-bin", type=Path, default=repo_root / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=repo_root / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=first_existing_fm_engine(),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--traj-udp-port", type=int, default=5001)
    parser.add_argument("--skip-udp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_exists(args.dataset_root, "dataset root")
    for path, description in [
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engineDir"),
        (args.multimodal_engine_dir, "multimodalEngineDir"),
        (args.fm_engine, "fmEngine"),
    ]:
        ensure_exists(path, description)

    run_tag = f"chunk{args.chunk_id:04d}_offset_{str(args.target_offset_s).replace('.', 'p')}"
    work_root = args.work_root / run_tag
    request_bank_root = work_root / "request_bank"
    output_root = work_root / "outputs"
    artifact_root = work_root / "artifacts"
    for path in (request_bank_root, output_root, artifact_root):
        path.mkdir(parents=True, exist_ok=True)
    for stale_path in output_root.iterdir():
        if stale_path.is_file():
            stale_path.unlink()
    for stale_path in artifact_root.iterdir():
        if stale_path.is_file():
            stale_path.unlink()

    anchor = select_target_front_frame(args.dataset_root, args.chunk_id, args.target_offset_s)
    request_path, metadata = build_single_request(
        dataset_root=args.dataset_root,
        chunk_id=args.chunk_id,
        anchor=anchor,
        out_root=request_bank_root,
        history_len=args.history_len,
        dt_s=args.dt_s,
        width=args.width,
        height=args.height,
        nav_text=args.nav_text,
        traj_token_offset=args.traj_token_offset,
        diffusion_seed=args.diffusion_seed,
        diffusion_num_steps=args.diffusion_num_steps,
        max_generate_length=args.max_generate_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    output_path = run_single_request(
        request_root=request_path.parent,
        output_root=output_root,
        llm_inference_bin=args.llm_inference_bin,
        plugin_lib=args.plugin_lib,
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        warmup=args.warmup,
        timeout_per_request=args.timeout_per_request,
        alpamayo_nav_cfg=args.alpamayo_nav_cfg,
        alpamayo_fm_use_prefill_kv=args.alpamayo_fm_use_prefill_kv,
    )
    traj_summary, ac_path_summary, gt_path_summary, traj_packet_bytes = build_result_artifacts(
        output_path=output_path,
        metadata=metadata,
        dataset_root=args.dataset_root,
        history_len=args.history_len,
        artifact_root=artifact_root,
    )

    if not args.skip_udp:
        send_udp_packet(args.udp_host, args.traj_udp_port, traj_packet_bytes)

    result = {
        "dataset_root": str(args.dataset_root),
        "chunk_id": int(args.chunk_id),
        "target_offset_s": float(args.target_offset_s),
        "actual_offset_s": float(metadata["actual_offset_s"]),
        "request_json": str(request_path),
        "model_output_json": str(output_path),
        "final_path_json": str(artifact_root / "final_path.json"),
        "ac_decoded_path_json": str(artifact_root / "ac_decoded_path.json"),
        "gt_path_json": str(artifact_root / "gt_path.json"),
        "udp_sent": bool(not args.skip_udp),
        "udp_host": args.udp_host,
        "traj_udp_port": int(args.traj_udp_port),
        "udp_format": "original_final_trajectory_only",
        "final_output": traj_summary["final_output"],
        "traj_points_with_origin": traj_summary["traj_points_with_origin"],
        "ac_decoded_traj_points_with_origin": ac_path_summary["traj_points_with_origin"],
        "gt_traj_points_with_origin": gt_path_summary["traj_points_with_origin"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
