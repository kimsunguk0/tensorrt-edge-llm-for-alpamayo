#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine
from run_raw_dataset_one_shot_udp import (
    build_path_summary,
    decode_actions_exact,
    reshape_tensor_field,
)
from run_request_bank_persistent import build_env, read_status


DEFAULT_ACTION_SPACE_CONSTANTS = {
    "accel_mean": 0.029052734375,
    "accel_std": 0.6796875,
    "curvature_mean": 0.0002689361572265625,
    "curvature_std": 0.026123046875,
    "dt_value": 0.1,
    "v_lambda": 0.000001,
    "v_ridge": 0.0001,
}

CAMERA_DISPLAY_NAMES = {
    "front": "Front camera",
    "left": "Front left camera",
    "right": "Front right camera",
    "front_tele": "Front telephoto camera",
}


@dataclass(frozen=True)
class RequestEntry:
    request_path: Path
    sample_id: int
    t0_us: int
    t0_utc_ns: int
    actual_offset_s: float
    ego_history_xyz_npy: str
    ego_history_rot_npy: str
    saved_images: tuple[str, ...]
    yaw_deg: float | None
    utm: tuple[float, float, float] | None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_samples(capture_root: Path) -> list[dict[str, Any]]:
    samples_path = capture_root / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")
    rows: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["t0_us"]))
    return rows


def scalar_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def first_utm(row: dict[str, Any]) -> tuple[float, float, float] | None:
    values = row.get("gnss_utm")
    if not values:
        return None
    item = values[0] if isinstance(values[0], list) else values
    if len(item) < 3:
        return None
    try:
        e, n, z = (float(item[0]), float(item[1]), float(item[2]))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(e) and np.isfinite(n) and np.isfinite(z)):
        return None
    valid = row.get("gnss_utm_valid", [1])
    if isinstance(valid, list) and valid and int(valid[0]) == 0:
        return None
    return e, n, z


def build_user_content(row: dict[str, Any], nav_text: str | None) -> list[dict[str, Any]]:
    camera_order = [str(item) for item in row.get("camera_order", ["front"])]
    saved_images = [str(path) for path in row.get("saved_images", [])]
    num_frames = int(row.get("num_frames_per_camera", 4))
    content: list[dict[str, Any]] = []
    cursor = 0
    for camera_name in camera_order:
        display = CAMERA_DISPLAY_NAMES.get(camera_name, f"{camera_name} camera")
        content.append({"type": "text", "text": f"{display}: "})
        for frame_idx in range(num_frames):
            if cursor >= len(saved_images):
                raise RuntimeError(f"Not enough saved_images for sample {row.get('seq')}")
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image", "image": saved_images[cursor]})
            cursor += 1

    hist_placeholder = "<|traj_history_start|>" + ("<|traj_history|>" * 48) + "<|traj_history_end|>"
    route_section = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else ""
    prompt = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    content.append({"type": "text", "text": f"{hist_placeholder}{route_section}{prompt}"})
    return content


def build_request_object(
    row: dict[str, Any],
    *,
    nav_text: str | None,
    traj_token_offset: int,
    diffusion_seed: int,
    diffusion_num_steps: int,
    max_generate_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    request = {
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
            {"role": "user", "content": build_user_content(row, nav_text)},
            {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
        ],
        "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
        "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
        "traj_token_offset": int(traj_token_offset),
        "diffusion_seed": int(diffusion_seed),
        "diffusion_num_steps": int(diffusion_num_steps),
        "action_space_constants": DEFAULT_ACTION_SPACE_CONSTANTS,
    }
    return {
        "batch_size": 1,
        "apply_chat_template": True,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "enable_thinking": False,
        "max_generate_length": int(max_generate_length),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "requests": [request],
    }


def build_capture_request_bank(capture_root: Path, request_bank_root: Path, args: argparse.Namespace) -> tuple[list[RequestEntry], dict[str, Any]]:
    request_root = request_bank_root / "requests"
    ensure_dir(request_root)
    rows = read_samples(capture_root)
    if not rows:
        raise RuntimeError(f"No samples in {capture_root}")

    manifest_rows: list[dict[str, Any]] = []
    t0_start_us = int(rows[0]["t0_us"])
    run_name = capture_root.name
    for row in rows:
        sample_id = int(row["seq"])
        t0_us = int(row["t0_us"])
        stem = f"request_{run_name}_sid{sample_id:06d}_t0_{t0_us}"
        request_path = request_root / f"{stem}.json"
        request_obj = build_request_object(
            row,
            nav_text=args.nav_text,
            traj_token_offset=args.traj_token_offset,
            diffusion_seed=args.diffusion_seed,
            diffusion_num_steps=args.diffusion_num_steps,
            max_generate_length=args.max_generate_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        write_json(request_path, request_obj)
        manifest_rows.append(
            {
                "run_name": run_name,
                "sample_id": sample_id,
                "request_json": str(request_path),
                "t0_us": t0_us,
                "t0_utc_ns": t0_us * 1000,
                "actual_offset_s": float((t0_us - t0_start_us) / 1e6),
                "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
                "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
                "saved_images": [str(path) for path in row.get("saved_images", [])],
                "camera_order": row.get("camera_order", []),
                "camera_indices": row.get("camera_indices", []),
                "utm": list(first_utm(row) or []),
                "yaw_deg": scalar_float(row.get("yaw_deg")),
            }
        )

    summary = {
        "capture_root": str(capture_root),
        "run_name": run_name,
        "num_requests": len(manifest_rows),
        "sample_id_min": int(min(row["sample_id"] for row in manifest_rows)),
        "sample_id_max": int(max(row["sample_id"] for row in manifest_rows)),
        "t0_start_us": int(min(row["t0_us"] for row in manifest_rows)),
        "t0_end_us": int(max(row["t0_us"] for row in manifest_rows)),
        "request_root": str(request_root),
        "output_root": str(request_bank_root),
        "nav_text": args.nav_text,
    }
    write_json(request_bank_root / "manifest.json", manifest_rows)
    write_json(request_bank_root / "summary.json", summary)
    return load_request_entries(request_bank_root)


def load_request_entries(request_bank_root: Path) -> tuple[list[RequestEntry], dict[str, Any]]:
    manifest_rows = json.loads((request_bank_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((request_bank_root / "summary.json").read_text(encoding="utf-8"))
    entries: list[RequestEntry] = []
    for row in sorted(manifest_rows, key=lambda item: int(item["t0_us"])):
        utm = row.get("utm") or None
        entries.append(
            RequestEntry(
                request_path=Path(row["request_json"]),
                sample_id=int(row["sample_id"]),
                t0_us=int(row["t0_us"]),
                t0_utc_ns=int(row["t0_utc_ns"]),
                actual_offset_s=float(row["actual_offset_s"]),
                ego_history_xyz_npy=str(row["ego_history_xyz_npy"]),
                ego_history_rot_npy=str(row["ego_history_rot_npy"]),
                saved_images=tuple(str(path) for path in row.get("saved_images", [])),
                yaw_deg=scalar_float(row.get("yaw_deg")),
                utm=tuple(float(v) for v in utm) if utm else None,
            )
        )
    return entries, summary


class CapturePoseSeries:
    def __init__(self, rows: list[dict[str, Any]], *, yaw_source: str) -> None:
        items: list[tuple[int, float, float, float, float | None]] = []
        for row in rows:
            utm = first_utm(row)
            if utm is None:
                continue
            yaw_deg = scalar_float(row.get("yaw_deg"))
            items.append((int(row["t0_us"]), float(utm[0]), float(utm[1]), float(utm[2]), yaw_deg))
        if len(items) < 2:
            raise RuntimeError("Need at least two valid UTM samples for GT interpolation")
        items.sort(key=lambda item: item[0])
        self.t_us = np.asarray([item[0] for item in items], dtype=np.float64)
        self.utm = np.asarray([[item[1], item[2], item[3]] for item in items], dtype=np.float64)
        self.yaw_source = yaw_source
        yaw_values = np.asarray([np.nan if item[4] is None else item[4] for item in items], dtype=np.float64)
        self.yaw_rad = self._build_yaw_rad(yaw_values)

    def _build_yaw_rad(self, yaw_deg: np.ndarray) -> np.ndarray:
        if self.yaw_source == "gnss" and int(np.isfinite(yaw_deg).sum()) >= 2:
            yaw = np.deg2rad(yaw_deg)
            good = np.isfinite(yaw)
            filled = np.interp(self.t_us, self.t_us[good], np.unwrap(yaw[good]))
            return filled.astype(np.float64)

        dxy = np.gradient(self.utm[:, :2], self.t_us / 1e6, axis=0)
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        yaw = np.unwrap(yaw)
        return yaw.astype(np.float64)

    @property
    def min_t_us(self) -> int:
        return int(self.t_us[0])

    @property
    def max_t_us(self) -> int:
        return int(self.t_us[-1])

    def interp_world(self, t_us: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t_us, dtype=np.float64)
        e = np.interp(t, self.t_us, self.utm[:, 0])
        n = np.interp(t, self.t_us, self.utm[:, 1])
        z = np.interp(t, self.t_us, self.utm[:, 2])
        return np.stack([e, n, z], axis=-1)

    def interp_yaw(self, t_us: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(t_us, dtype=np.float64), self.t_us, self.yaw_rad)

    @staticmethod
    def yaw_rot(yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def future_local_pose(self, t0_us: int, future_len: int, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        future_times = float(t0_us) + np.arange(1, future_len + 1, dtype=np.float64) * float(dt_s) * 1e6
        world = self.interp_world(future_times)
        p0 = self.interp_world(float(t0_us)).reshape(3)
        yaw0 = float(self.interp_yaw(float(t0_us)))
        r0 = self.yaw_rot(yaw0)
        local_xyz = ((world - p0) @ r0).astype(np.float32)

        future_yaw = self.interp_yaw(future_times)
        rel_yaw = np.unwrap(future_yaw) - yaw0
        local_rot = np.zeros((future_len, 3, 3), dtype=np.float32)
        local_rot[:, 2, 2] = 1.0
        local_rot[:, 0, 0] = np.cos(rel_yaw).astype(np.float32)
        local_rot[:, 0, 1] = -np.sin(rel_yaw).astype(np.float32)
        local_rot[:, 1, 0] = np.sin(rel_yaw).astype(np.float32)
        local_rot[:, 1, 1] = np.cos(rel_yaw).astype(np.float32)
        return local_xyz, local_rot

    def local_to_world(self, t0_us: int, local_xyz: np.ndarray) -> np.ndarray:
        p0 = self.interp_world(float(t0_us)).reshape(3)
        yaw0 = float(self.interp_yaw(float(t0_us)))
        r0 = self.yaw_rot(yaw0)
        return (np.asarray(local_xyz, dtype=np.float64) @ r0.T) + p0

    def full_gt_world(self) -> np.ndarray:
        return self.utm.copy()


def output_path_for_request(output_root: Path, request_path: Path) -> Path:
    return output_root / request_path.name.replace("request_", "output_", 1)


def runtime_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.llm_inference_bin),
        "--engineDir",
        str(args.engine_dir),
        "--multimodalEngineDir",
        str(args.multimodal_engine_dir),
        "--fmEngine",
        str(args.fm_engine),
        "--alpamayoPostVlmRuntime",
        "--persistentServer",
        "--warmup",
        str(args.warmup),
    ]
    if args.alpamayo_fm_use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")
    if args.alpamayo_nav_cfg:
        cmd.append("--alpamayoNavCfg")
    return cmd


def start_runtime(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd = runtime_cmd(args)
    print("[capture-dynamic] starting persistent llm_inference", flush=True)
    print("[capture-dynamic] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )
    ready = read_status(proc, timeout_s=180.0)
    if ready.get("status") != "ready":
        raise RuntimeError(f"Unexpected ready state: {ready}")
    print("[capture-dynamic] persistent llm_inference ready", flush=True)
    return proc


def shutdown_runtime(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            proc.stdin.flush()
        read_status(proc, timeout_s=10.0)
    except Exception:
        pass
    try:
        proc.wait(timeout=20.0)
    except Exception:
        proc.kill()


def run_one_inference(
    proc: subprocess.Popen[str] | None,
    args: argparse.Namespace,
    entry: RequestEntry,
    output_path: Path,
) -> tuple[subprocess.Popen[str] | None, float, bool]:
    if args.reuse_existing_outputs and output_path.exists():
        meta_path = output_path.with_suffix(".meta.json")
        inference_time_s = 0.0
        if meta_path.exists():
            try:
                inference_time_s = float(json.loads(meta_path.read_text(encoding="utf-8")).get("inference_time_s", 0.0))
            except Exception:
                inference_time_s = 0.0
        return proc, inference_time_s, True

    if proc is None:
        proc = start_runtime(args)
    payload = {"input_file": str(entry.request_path), "output_file": str(output_path)}
    assert proc.stdin is not None
    t0 = time.time()
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    status = read_status(proc, timeout_s=args.timeout_per_request)
    if status.get("status") != "ok":
        raise RuntimeError(f"Request failed for {entry.request_path.name}: {status}")
    inference_time_s = time.time() - t0
    write_json(output_path.with_suffix(".meta.json"), {"inference_time_s": inference_time_s, "request_json": str(entry.request_path)})
    return proc, inference_time_s, False


def build_capture_artifacts(
    *,
    output_path: Path,
    metadata: dict[str, Any],
    pose_series: CapturePoseSeries,
    history_len: int,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ensure_dir(artifact_root)
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
    gt_pred_xyz, gt_pred_rot = pose_series.future_local_pose(
        int(metadata["t0_us"]),
        future_len=int(pred_xyz.shape[0]),
        dt_s=dt_s,
    )

    final_summary, final_packet, _ = build_path_summary(
        label="final_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        final_output=final_text,
        timing=post.get("timing", {}),
    )
    ac_summary, ac_packet, _ = build_path_summary(
        label="ac_decoded_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=ac_pred_xyz,
        pred_rot=ac_pred_rot,
        final_output=final_text,
    )
    gt_summary, gt_packet, _ = build_path_summary(
        label="gt_path",
        metadata=metadata,
        dt_s=dt_s,
        pred_xyz=gt_pred_xyz,
        pred_rot=gt_pred_rot,
        final_output="GNSS/UTM ground-truth future path from live capture",
    )

    for summary, packet, path_type in (
        (final_summary, final_packet, "final_path"),
        (ac_summary, ac_packet, "ac_decoded_path"),
        (gt_summary, gt_packet, "gt_path"),
    ):
        summary["path_type"] = path_type
        summary["packet_header"] = packet["header"]
        summary["packet_points"] = packet["points"]
        summary["global_utm_xyz"] = pose_series.local_to_world(int(metadata["t0_us"]), np.asarray(summary["pred_xyz"], dtype=np.float32)).astype(float).tolist()

    ac_summary["raw_action"] = {
        "num_points": int(x_final.shape[0]),
        "normalized_x_final": x_final.tolist(),
        "accel_mps2": accel.astype(np.float32).tolist(),
        "curvature": raw_curvature.astype(np.float32).tolist(),
        "action_space_constants": fm["action_space_constants"],
    }

    write_json(artifact_root / "final_path.json", final_summary)
    write_json(artifact_root / "ac_decoded_path.json", ac_summary)
    write_json(artifact_root / "gt_path.json", gt_summary)
    return final_summary, ac_summary, gt_summary


def append_metric_rows(
    *,
    run_name: str,
    entry: RequestEntry,
    summaries: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    point_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> None:
    final_summary, ac_summary, gt_summary = summaries
    gt_xyz = np.asarray(gt_summary["pred_xyz"], dtype=np.float64)
    gt_world = np.asarray(gt_summary["global_utm_xyz"], dtype=np.float64)
    for summary in (final_summary, ac_summary):
        pred_xyz = np.asarray(summary["pred_xyz"], dtype=np.float64)
        pred_world = np.asarray(summary["global_utm_xyz"], dtype=np.float64)
        errors = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1)
        point_count = int(len(errors))
        sample_rows.append(
            {
                "run_name": run_name,
                "sample_id": entry.sample_id,
                "t0_us": entry.t0_us,
                "global_elapsed_s": entry.actual_offset_s,
                "path_type": summary["path_type"],
                "point_count": point_count,
                "ate_m": float(np.sqrt(np.mean(errors**2))) if point_count else float("nan"),
                "mean_error_m": float(np.mean(errors)) if point_count else float("nan"),
                "max_error_m": float(np.max(errors)) if point_count else float("nan"),
                "fde_m": float(errors[-1]) if point_count else float("nan"),
                "inference_time_s": float(summary.get("inference_time_s", 0.0)),
            }
        )
        for idx, err in enumerate(errors, start=1):
            point_rows.append(
                {
                    "run_name": run_name,
                    "sample_id": entry.sample_id,
                    "t0_us": entry.t0_us,
                    "global_elapsed_s": entry.actual_offset_s,
                    "path_type": summary["path_type"],
                    "horizon_index": idx,
                    "horizon_s": float(idx * float(summary["plan_dt_s"])),
                    "pred_x_m": float(pred_xyz[idx - 1, 0]),
                    "pred_y_m": float(pred_xyz[idx - 1, 1]),
                    "pred_z_m": float(pred_xyz[idx - 1, 2]),
                    "gt_x_m": float(gt_xyz[idx - 1, 0]),
                    "gt_y_m": float(gt_xyz[idx - 1, 1]),
                    "gt_z_m": float(gt_xyz[idx - 1, 2]),
                    "error_m": float(err),
                    "global_pred_e_m": float(pred_world[idx - 1, 0]),
                    "global_pred_n_m": float(pred_world[idx - 1, 1]),
                    "global_gt_e_m": float(gt_world[idx - 1, 0]),
                    "global_gt_n_m": float(gt_world[idx - 1, 1]),
                }
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    if not sample_rows:
        return {"summary": summary_rows}
    run_names = sorted({str(row["run_name"]) for row in sample_rows})
    path_types = sorted({str(row["path_type"]) for row in sample_rows})
    for run_name in run_names + ["overall"]:
        for path_type in path_types:
            rows = [
                row
                for row in sample_rows
                if row["path_type"] == path_type and (run_name == "overall" or row["run_name"] == run_name)
            ]
            if not rows:
                continue
            point_count = int(sum(int(row["point_count"]) for row in rows))
            summary_rows.append(
                {
                    "scope": run_name,
                    "path_type": path_type,
                    "sample_count": len(rows),
                    "point_count": point_count,
                    "ate_m": float(np.mean([float(row["ate_m"]) for row in rows])),
                    "mean_error_m": float(np.mean([float(row["mean_error_m"]) for row in rows])),
                    "max_error_m": float(np.max([float(row["max_error_m"]) for row in rows])),
                    "fde_mean_m": float(np.mean([float(row["fde_m"]) for row in rows])),
                    "fde_max_m": float(np.max([float(row["fde_m"]) for row in rows])),
                    "inference_time_mean_s": float(np.mean([float(row["inference_time_s"]) for row in rows])),
                }
            )
    return {"summary": summary_rows}


def select_next_index(entries: list[RequestEntry], current_idx: int, inference_time_s: float) -> int:
    next_t = entries[current_idx].t0_us + int(round(max(float(inference_time_s), 1e-6) * 1e6))
    t_values = [entry.t0_us for entry in entries]
    return max(current_idx + 1, bisect.bisect_left(t_values, next_t, lo=current_idx + 1))


def plot_run_map(
    *,
    run_name: str,
    pose_series: CapturePoseSeries,
    artifacts: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_world = pose_series.full_gt_world()
    ref = gt_world[0, :2]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(gt_world[:, 0] - ref[0], gt_world[:, 1] - ref[1], color="#111827", linewidth=2.2, label="GT driven path")

    if artifacts:
        cmap = plt.get_cmap("viridis")
        denom = max(len(artifacts) - 1, 1)
        for idx, item in enumerate(artifacts):
            color = cmap(idx / denom)
            ac = np.asarray(item["ac_summary"]["global_utm_xyz"], dtype=np.float64)
            gt = np.asarray(item["gt_summary"]["global_utm_xyz"], dtype=np.float64)
            ax.plot(gt[:, 0] - ref[0], gt[:, 1] - ref[1], color="#22c55e", alpha=0.20, linewidth=1.0)
            ax.plot(ac[:, 0] - ref[0], ac[:, 1] - ref[1], color=color, alpha=0.78, linewidth=1.4)
            p0 = pose_series.interp_world(float(item["entry"].t0_us)).reshape(3)
            ax.scatter([p0[0] - ref[0]], [p0[1] - ref[1]], color=color, s=10, alpha=0.85)

    ax.set_title(f"{run_name}: dynamic inference paths vs GT")
    ax.set_xlabel("UTM East offset from run start (m)")
    ax.set_ylabel("UTM North offset from run start (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_capture_run(
    *,
    capture_root: Path,
    run_index: int,
    args: argparse.Namespace,
    proc: subprocess.Popen[str] | None,
) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    run_name = capture_root.name
    run_root = args.output_root / run_name
    request_bank_root = run_root / "request_bank"
    output_root = run_root / "outputs"
    artifact_root = run_root / "artifacts"
    evaluation_root = run_root / "evaluation"
    for path in (request_bank_root, output_root, artifact_root, evaluation_root):
        ensure_dir(path)

    if args.rebuild_request_bank or not (request_bank_root / "manifest.json").exists():
        entries, bank_summary = build_capture_request_bank(capture_root, request_bank_root, args)
    else:
        entries, bank_summary = load_request_entries(request_bank_root)

    rows = read_samples(capture_root)
    pose_series = CapturePoseSeries(rows, yaw_source=args.gt_yaw_source)
    horizon_s = float(args.future_len) * float(args.plan_dt_s)
    latest_eval_t_us = pose_series.max_t_us - int(round(horizon_s * 1e6))

    if args.require_full_gt:
        candidate_entries = [entry for entry in entries if entry.t0_us <= latest_eval_t_us]
    else:
        candidate_entries = entries
    if args.max_requests > 0:
        candidate_entries = candidate_entries[: args.max_requests]
    if not candidate_entries:
        raise RuntimeError(f"No candidate requests with full GT horizon for {run_name}")

    print(
        f"[capture-dynamic] {run_name}: candidates={len(candidate_entries)} "
        f"full_gt_until_offset={(latest_eval_t_us - candidate_entries[0].t0_us) / 1e6:.2f}s",
        flush=True,
    )

    idx = 0
    selected: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    plot_artifacts: list[dict[str, Any]] = []
    run_start = time.time()
    while idx < len(candidate_entries):
        entry = candidate_entries[idx]
        output_path = output_path_for_request(output_root, entry.request_path)
        proc, inference_time_s, reused = run_one_inference(proc, args, entry, output_path)
        normalized_metadata = {
            "chunk_id": int(run_index),
            "sample_id": int(entry.sample_id),
            "front_frame_id": int(entry.sample_id),
            "t0_utc_ns": int(entry.t0_utc_ns),
            "t0_us": int(entry.t0_us),
            "target_offset_s": float(entry.actual_offset_s),
            "actual_offset_s": float(entry.actual_offset_s),
            "request_json": str(entry.request_path),
            "ego_history_xyz_npy": entry.ego_history_xyz_npy,
            "ego_history_rot_npy": entry.ego_history_rot_npy,
        }
        sample_artifact_root = artifact_root / entry.request_path.stem
        summaries = build_capture_artifacts(
            output_path=output_path,
            metadata=normalized_metadata,
            pose_series=pose_series,
            history_len=args.history_len,
            artifact_root=sample_artifact_root,
        )
        for summary in summaries:
            summary["inference_time_s"] = float(inference_time_s)
        append_metric_rows(
            run_name=run_name,
            entry=entry,
            summaries=summaries,
            point_rows=point_rows,
            sample_rows=sample_rows,
        )
        final_summary, ac_summary, gt_summary = summaries
        selected_row = {
            "run_name": run_name,
            "selected_index": len(selected),
            "candidate_index": idx,
            "sample_id": entry.sample_id,
            "t0_us": entry.t0_us,
            "actual_offset_s": entry.actual_offset_s,
            "request_json": str(entry.request_path),
            "output_json": str(output_path),
            "artifact_root": str(sample_artifact_root),
            "inference_time_s": float(inference_time_s),
            "inference_reused": bool(reused),
        }
        selected.append(selected_row)
        plot_artifacts.append({"entry": entry, "final_summary": final_summary, "ac_summary": ac_summary, "gt_summary": gt_summary})
        elapsed = time.time() - run_start
        print(
            f"[capture-dynamic] {run_name} #{len(selected)} sid={entry.sample_id:06d} "
            f"offset={entry.actual_offset_s:.2f}s infer={inference_time_s:.3f}s "
            f"reused={reused} elapsed={elapsed/60:.1f}m",
            flush=True,
        )
        next_idx = select_next_index(candidate_entries, idx, inference_time_s)
        if next_idx <= idx:
            next_idx = idx + 1
        idx = next_idx

    write_json(evaluation_root / "selected_schedule.json", selected)
    write_csv(evaluation_root / "sample_metrics_dynamic.csv", sample_rows)
    write_csv(evaluation_root / "path_points_dynamic.csv", point_rows)
    metrics_summary = summarize_metrics(sample_rows)
    write_json(evaluation_root / "summary_metrics.json", metrics_summary)
    paths_dynamic = {
        "run_name": run_name,
        "capture_root": str(capture_root),
        "request_bank_summary": bank_summary,
        "selection_policy": "next sample t0 >= previous sample t0 + measured inference_time_s",
        "require_full_gt": bool(args.require_full_gt),
        "future_len": int(args.future_len),
        "plan_dt_s": float(args.plan_dt_s),
        "sample_count": len(selected),
        "selected_schedule": selected,
    }
    write_json(evaluation_root / "paths_dynamic_meta.json", paths_dynamic)
    plot_run_map(
        run_name=run_name,
        pose_series=pose_series,
        artifacts=plot_artifacts,
        output_path=evaluation_root / "map_overview.png",
    )

    return proc, {
        "run_name": run_name,
        "run_root": str(run_root),
        "request_bank_root": str(request_bank_root),
        "output_root": str(output_root),
        "artifact_root": str(artifact_root),
        "evaluation_root": str(evaluation_root),
        "selected_count": len(selected),
        "metrics_summary": metrics_summary,
        "map_png": str(evaluation_root / "map_overview.png"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Alpamayo inference on saved live_camera_ego_history_capture runs. "
            "The next request is selected after the previous request's measured inference time."
        )
    )
    parser.add_argument("capture_roots", type=Path, nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "live_capture_dynamic_inference_20260526",
    )
    parser.add_argument("--rebuild-request-bank", action="store_true")
    parser.add_argument("--reuse-existing-outputs", action="store_true")
    parser.add_argument("--require-full-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-yaw-source", choices=["gnss", "motion"], default="gnss")
    parser.add_argument("--future-len", type=int, default=64)
    parser.add_argument("--plan-dt-s", type=float, default=0.1)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-requests", type=int, default=-1)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=first_existing_fm_engine())
    parser.add_argument("--alpamayo-nav-cfg", action="store_true")
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    for path, description in (
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engineDir"),
        (args.multimodal_engine_dir, "multimodalEngineDir"),
        (args.fm_engine, "fmEngine"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    proc: subprocess.Popen[str] | None = None
    results: list[dict[str, Any]] = []
    try:
        for run_index, capture_root in enumerate(args.capture_roots):
            proc, result = process_capture_run(
                capture_root=capture_root,
                run_index=run_index,
                args=args,
                proc=proc,
            )
            results.append(result)
    finally:
        shutdown_runtime(proc)

    combined = {
        "output_root": str(args.output_root),
        "capture_roots": [str(path) for path in args.capture_roots],
        "runs": results,
    }
    write_json(args.output_root / "summary.json", combined)
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
