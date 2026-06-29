#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
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

from build_live_chunk_request_bank import (  # noqa: E402
    ecef_to_enu,
    geodetic_to_ecef,
    select_pose_gnss_rows,
)
from fm_model_defaults import first_existing_fm_engine  # noqa: E402
from run_raw_dataset_one_shot_udp import build_result_artifacts, ensure_exists, load_pandas  # noqa: E402
from run_request_bank_persistent import build_env, read_status  # noqa: E402


BUILD_REQUEST_BANK = SCRIPT_DIR / "build_live_chunk_request_bank.py"

CAMERA_MODES: dict[str, tuple[str, ...]] = {
    "front": ("front",),
    "front_front_tele": ("front", "front_tele"),
}


@dataclass(frozen=True)
class RawRequestEntry:
    camera_mode: str
    chunk_id: int
    request_path: Path
    sample_id: int
    front_frame_id: int
    t0_utc_ns: int
    t0_us: int
    actual_offset_s: float
    ego_history_xyz_npy: str
    ego_history_rot_npy: str
    selected_frames: dict[str, list[int]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mode_semantics(mode_name: str) -> tuple[str, ...]:
    if mode_name not in CAMERA_MODES:
        raise ValueError(f"Unknown camera mode {mode_name!r}; choices={sorted(CAMERA_MODES)}")
    return CAMERA_MODES[mode_name]


def detect_chunks(dataset_root: Path, modes: list[str]) -> list[int]:
    needed_sensors: set[str] = set()
    sensor_by_semantic = {
        "front": "camera_front",
        "front_tele": "camera_front_tele",
        "left": "camera_left",
        "right": "camera_right",
    }
    for mode in modes:
        for semantic in mode_semantics(mode):
            needed_sensors.add(sensor_by_semantic[semantic])

    chunk_sets: list[set[int]] = []
    for sensor in sorted(needed_sensors):
        chunk_dir = dataset_root / "sensors" / sensor / "chunks"
        ensure_exists(chunk_dir, f"{sensor} chunks")
        chunks = set()
        for path in chunk_dir.glob("chunk_*.mkv"):
            try:
                chunks.add(int(path.stem.split("_")[-1]))
            except ValueError:
                pass
        if not chunks:
            raise RuntimeError(f"No chunks found for {sensor}: {chunk_dir}")
        chunk_sets.append(chunks)
    return sorted(set.intersection(*chunk_sets))


class RawPoseSeries:
    def __init__(self, dataset_root: Path) -> None:
        pd = load_pandas()
        gnss_path = dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
        ensure_exists(gnss_path, "gnss_ins parquet")
        gnss = pd.read_parquet(gnss_path)
        valid = select_pose_gnss_rows(gnss)
        if len(valid) < 2:
            raise RuntimeError("Need at least two valid GNSS pose rows")

        self.t_us = (valid["timestamp_utc_ns"].to_numpy(dtype=np.int64).astype(np.float64) / 1000.0)
        lat = valid["lat"].to_numpy(dtype=np.float64)
        lon = valid["lon"].to_numpy(dtype=np.float64)
        alt = valid["alt"].to_numpy(dtype=np.float64)
        self.ref_lla = (float(lat[0]), float(lon[0]), float(alt[0]))
        self.world = ecef_to_enu(geodetic_to_ecef(lat, lon, alt), *self.ref_lla).astype(np.float64)

        dt = self.t_us / 1e6
        dxy = np.gradient(self.world[:, :2], dt, axis=0)
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        self.yaw_rad = np.unwrap(yaw).astype(np.float64)

    @property
    def min_t_us(self) -> int:
        return int(self.t_us[0])

    @property
    def max_t_us(self) -> int:
        return int(self.t_us[-1])

    def interp_world(self, t_us: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t_us, dtype=np.float64)
        e = np.interp(t, self.t_us, self.world[:, 0])
        n = np.interp(t, self.t_us, self.world[:, 1])
        z = np.interp(t, self.t_us, self.world[:, 2])
        return np.stack([e, n, z], axis=-1)

    def interp_yaw(self, t_us: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(t_us, dtype=np.float64), self.t_us, self.yaw_rad)

    @staticmethod
    def yaw_rot(yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def local_to_world(self, t0_us: int, local_xyz: np.ndarray) -> np.ndarray:
        p0 = self.interp_world(float(t0_us)).reshape(3)
        yaw0 = float(self.interp_yaw(float(t0_us)))
        r0 = self.yaw_rot(yaw0)
        return (np.asarray(local_xyz, dtype=np.float64) @ r0.T) + p0

    def full_gt_world(self) -> np.ndarray:
        return self.world.copy()


def build_request_bank(
    *,
    dataset_root: Path,
    chunk_id: int,
    mode_name: str,
    request_bank_root: Path,
    args: argparse.Namespace,
) -> None:
    summary_path = request_bank_root / "summary.json"
    expected_semantics = list(mode_semantics(mode_name))
    if summary_path.exists() and not args.rebuild_request_bank:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            actual = [item["semantic_name"] for item in summary.get("camera_runtime_order", [])]
            if actual == expected_semantics:
                print(f"[raw-camera-compare] reuse request bank {request_bank_root}", flush=True)
                return
            print(
                f"[raw-camera-compare] rebuilding {request_bank_root}: camera order {actual} != {expected_semantics}",
                flush=True,
            )
        except Exception:
            print(f"[raw-camera-compare] rebuilding unreadable request bank {request_bank_root}", flush=True)

    ensure_dir(request_bank_root)
    cmd = [
        sys.executable,
        str(BUILD_REQUEST_BANK),
        "--dataset-root",
        str(dataset_root),
        "--chunk-id",
        str(chunk_id),
        "--output-root",
        str(request_bank_root),
        "--history-len",
        str(args.history_len),
        "--dt-s",
        str(args.plan_dt_s),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--traj-token-offset",
        str(args.traj_token_offset),
        "--diffusion-seed",
        str(args.diffusion_seed),
        "--diffusion-num-steps",
        str(args.diffusion_num_steps),
        "--max-generate-length",
        str(args.max_generate_length),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--camera-semantics",
        *expected_semantics,
    ]
    if args.max_bank_requests > 0:
        cmd.extend(["--limit", str(args.max_bank_requests)])
    if args.nav_text:
        cmd.extend(["--nav-text", args.nav_text])
    print("[raw-camera-compare] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_entries(request_bank_root: Path, camera_mode: str, chunk_id: int) -> tuple[list[RawRequestEntry], dict[str, Any]]:
    manifest_path = request_bank_root / "manifest.json"
    summary_path = request_bank_root / "summary.json"
    ensure_exists(manifest_path, "request bank manifest")
    ensure_exists(summary_path, "request bank summary")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No manifest rows: {manifest_path}")

    t0_start_ns = min(int(row["t0_utc_ns"]) for row in rows)
    entries: list[RawRequestEntry] = []
    for row in sorted(rows, key=lambda item: int(item["t0_utc_ns"])):
        selected_frames = {str(k): [int(x) for x in v] for k, v in row.get("selected_frames", {}).items()}
        front_ids = selected_frames.get("front") or []
        front_frame_id = int(front_ids[-1]) if front_ids else int(row["sample_id"])
        t0_utc_ns = int(row["t0_utc_ns"])
        entries.append(
            RawRequestEntry(
                camera_mode=camera_mode,
                chunk_id=chunk_id,
                request_path=Path(row["request_json"]),
                sample_id=int(row["sample_id"]),
                front_frame_id=front_frame_id,
                t0_utc_ns=t0_utc_ns,
                t0_us=int(row["t0_us"]),
                actual_offset_s=float((t0_utc_ns - t0_start_ns) / 1e9),
                ego_history_xyz_npy=str(row["ego_history_xyz_npy"]),
                ego_history_rot_npy=str(row["ego_history_rot_npy"]),
                selected_frames=selected_frames,
            )
        )
    return entries, summary


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
    print("[raw-camera-compare] starting persistent llm_inference", flush=True)
    print("[raw-camera-compare] " + " ".join(cmd), flush=True)
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
    print("[raw-camera-compare] persistent llm_inference ready", flush=True)
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


def output_path_for_request(output_root: Path, request_path: Path) -> Path:
    return output_root / request_path.name.replace("request_", "output_", 1)


def run_one_inference(
    proc: subprocess.Popen[str] | None,
    args: argparse.Namespace,
    entry: RawRequestEntry,
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
        if inference_time_s <= 0.0:
            inference_time_s = max(float(args.default_reused_inference_time_s), 1e-3)
        return proc, inference_time_s, True

    if proc is None:
        proc = start_runtime(args)
    ensure_dir(output_path.parent)
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


def add_global_paths(
    summaries: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    pose_series: RawPoseSeries,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for summary in summaries:
        local_xyz = np.asarray(summary["pred_xyz"], dtype=np.float32)
        summary["global_enu_xyz"] = pose_series.local_to_world(int(summary["t0_us"]), local_xyz).astype(float).tolist()
    write_json(artifact_root / "final_path.json", summaries[0])
    write_json(artifact_root / "ac_decoded_path.json", summaries[1])
    write_json(artifact_root / "gt_path.json", summaries[2])
    return summaries


def build_artifacts_for_entry(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    pose_series: RawPoseSeries,
    entry: RawRequestEntry,
    output_path: Path,
    artifact_root: Path,
    history_len: int,
    inference_time_s: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = {
        "chunk_id": int(entry.chunk_id),
        "sample_id": int(entry.sample_id),
        "front_frame_id": int(entry.front_frame_id),
        "t0_utc_ns": int(entry.t0_utc_ns),
        "t0_us": int(entry.t0_us),
        "target_offset_s": float(entry.actual_offset_s),
        "actual_offset_s": float(entry.actual_offset_s),
        "request_json": str(entry.request_path),
        "ego_history_xyz_npy": entry.ego_history_xyz_npy,
        "ego_history_rot_npy": entry.ego_history_rot_npy,
        "selected_frames": entry.selected_frames,
    }
    final_summary, ac_summary, gt_summary, _packet_bytes = build_result_artifacts(
        output_path=output_path,
        metadata=metadata,
        dataset_root=dataset_root,
        history_len=history_len,
        artifact_root=artifact_root,
    )
    summaries = (final_summary, ac_summary, gt_summary)
    for summary in summaries:
        summary["camera_mode"] = entry.camera_mode
        summary["inference_time_s"] = float(inference_time_s)
    return add_global_paths(summaries, pose_series, artifact_root)


def append_metric_rows(
    *,
    entry: RawRequestEntry,
    summaries: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    point_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> None:
    final_summary, ac_summary, gt_summary = summaries
    gt_xyz = np.asarray(gt_summary["pred_xyz"], dtype=np.float64)
    gt_world = np.asarray(gt_summary["global_enu_xyz"], dtype=np.float64)
    for summary in (final_summary, ac_summary):
        pred_xyz = np.asarray(summary["pred_xyz"], dtype=np.float64)
        pred_world = np.asarray(summary["global_enu_xyz"], dtype=np.float64)
        errors = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[:, :2], axis=1)
        point_count = int(len(errors))
        sample_rows.append(
            {
                "camera_mode": entry.camera_mode,
                "chunk_id": entry.chunk_id,
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
                    "camera_mode": entry.camera_mode,
                    "chunk_id": entry.chunk_id,
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


def summarize_metrics(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    if not sample_rows:
        return {"summary": summary_rows}
    modes = sorted({str(row["camera_mode"]) for row in sample_rows})
    path_types = sorted({str(row["path_type"]) for row in sample_rows})
    chunks = sorted({int(row["chunk_id"]) for row in sample_rows})
    scopes: list[tuple[str, int | None]] = [(f"chunk{chunk_id:04d}", chunk_id) for chunk_id in chunks] + [("overall", None)]
    for mode in modes:
        for scope_name, chunk_id in scopes:
            for path_type in path_types:
                rows = [
                    row
                    for row in sample_rows
                    if row["camera_mode"] == mode
                    and row["path_type"] == path_type
                    and (chunk_id is None or int(row["chunk_id"]) == chunk_id)
                ]
                if not rows:
                    continue
                summary_rows.append(
                    {
                        "camera_mode": mode,
                        "scope": scope_name,
                        "path_type": path_type,
                        "sample_count": len(rows),
                        "point_count": int(sum(int(row["point_count"]) for row in rows)),
                        "ate_m": float(np.mean([float(row["ate_m"]) for row in rows])),
                        "mean_error_m": float(np.mean([float(row["mean_error_m"]) for row in rows])),
                        "max_error_m": float(np.max([float(row["max_error_m"]) for row in rows])),
                        "fde_mean_m": float(np.mean([float(row["fde_m"]) for row in rows])),
                        "fde_max_m": float(np.max([float(row["fde_m"]) for row in rows])),
                        "inference_time_mean_s": float(np.mean([float(row["inference_time_s"]) for row in rows])),
                    }
                )
    return {"summary": summary_rows}


def select_next_index(entries: list[RawRequestEntry], current_idx: int, inference_time_s: float) -> int:
    next_t = entries[current_idx].t0_us + int(round(max(float(inference_time_s), 1e-6) * 1e6))
    t_values = [entry.t0_us for entry in entries]
    return max(current_idx + 1, bisect.bisect_left(t_values, next_t, lo=current_idx + 1))


def filter_full_gt(entries: list[RawRequestEntry], pose_series: RawPoseSeries, args: argparse.Namespace) -> list[RawRequestEntry]:
    if not args.require_full_gt:
        return entries
    horizon_us = int(round(float(args.future_len) * float(args.plan_dt_s) * 1e6))
    latest_t0_us = pose_series.max_t_us - horizon_us
    return [entry for entry in entries if entry.t0_us <= latest_t0_us]


def process_entries(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    pose_series: RawPoseSeries,
    proc: subprocess.Popen[str] | None,
    mode_name: str,
    chunk_id: int,
    entries: list[RawRequestEntry],
    run_root: Path,
    selection_kind: str,
) -> tuple[subprocess.Popen[str] | None, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = run_root / "outputs"
    artifact_root = run_root / "artifacts"
    evaluation_root = run_root / "evaluation"
    ensure_dir(output_root)
    ensure_dir(artifact_root)
    ensure_dir(evaluation_root)

    selected_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    plot_items: list[dict[str, Any]] = []
    run_start = time.time()
    history_len = int(args.history_len)

    for selected_index, entry in enumerate(entries):
        output_path = output_path_for_request(output_root, entry.request_path)
        proc, inference_time_s, reused = run_one_inference(proc, args, entry, output_path)
        sample_artifact_root = artifact_root / entry.request_path.stem
        summaries = build_artifacts_for_entry(
            args=args,
            dataset_root=dataset_root,
            pose_series=pose_series,
            entry=entry,
            output_path=output_path,
            artifact_root=sample_artifact_root,
            history_len=history_len,
            inference_time_s=inference_time_s,
        )
        append_metric_rows(entry=entry, summaries=summaries, point_rows=point_rows, sample_rows=sample_rows)
        final_summary, ac_summary, gt_summary = summaries
        selected_rows.append(
            {
                "camera_mode": mode_name,
                "chunk_id": chunk_id,
                "selected_index": selected_index,
                "sample_id": entry.sample_id,
                "t0_us": entry.t0_us,
                "actual_offset_s": entry.actual_offset_s,
                "selection_kind": selection_kind,
                "request_json": str(entry.request_path),
                "output_json": str(output_path),
                "artifact_root": str(sample_artifact_root),
                "inference_time_s": float(inference_time_s),
                "inference_reused": bool(reused),
            }
        )
        plot_items.append(
            {
                "entry": entry,
                "final_summary": final_summary,
                "ac_summary": ac_summary,
                "gt_summary": gt_summary,
            }
        )
        elapsed = time.time() - run_start
        print(
            f"[raw-camera-compare] {mode_name} chunk{chunk_id:04d} "
            f"#{selected_index + 1}/{len(entries)} sid={entry.sample_id:05d} "
            f"offset={entry.actual_offset_s:.2f}s infer={inference_time_s:.3f}s "
            f"reused={reused} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    write_json(evaluation_root / "selected_schedule.json", selected_rows)
    write_csv(evaluation_root / "sample_metrics_dynamic.csv", sample_rows)
    write_csv(evaluation_root / "path_points_dynamic.csv", point_rows)
    metrics_summary = summarize_metrics(sample_rows)
    write_json(evaluation_root / "summary_metrics.json", metrics_summary)
    return proc, metrics_summary, selected_rows, sample_rows, point_rows


def build_primary_schedule(entries: list[RawRequestEntry]) -> list[RawRequestEntry]:
    idx = 0
    selected: list[RawRequestEntry] = []
    # Placeholder schedule; actual stepping is handled online after measured inference.
    while idx < len(entries):
        selected.append(entries[idx])
        idx += 1
    return selected


def plot_mode_map(
    *,
    title: str,
    pose_series: RawPoseSeries,
    point_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_world = pose_series.full_gt_world()
    ref = gt_world[0, :2]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(gt_world[:, 0] - ref[0], gt_world[:, 1] - ref[1], color="#111827", linewidth=2.2, label="GT driven path")

    by_sample: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in point_rows:
        if row["path_type"] not in {"final_path"}:
            continue
        key = (int(row["chunk_id"]), int(row["sample_id"]), str(row["path_type"]))
        by_sample.setdefault(key, []).append(row)
    cmap = plt.get_cmap("viridis")
    denom = max(len(by_sample) - 1, 1)
    for idx, (_key, rows) in enumerate(sorted(by_sample.items())):
        rows = sorted(rows, key=lambda row: int(row["horizon_index"]))
        color = cmap(idx / denom)
        pred = np.asarray([[float(row["global_pred_e_m"]), float(row["global_pred_n_m"])] for row in rows])
        gt = np.asarray([[float(row["global_gt_e_m"]), float(row["global_gt_n_m"])] for row in rows])
        ax.plot(gt[:, 0] - ref[0], gt[:, 1] - ref[1], color="#22c55e", alpha=0.18, linewidth=1.0)
        ax.plot(pred[:, 0] - ref[0], pred[:, 1] - ref[1], color=color, alpha=0.78, linewidth=1.35)
        ax.scatter([pred[0, 0] - ref[0]], [pred[0, 1] - ref[1]], color=color, s=8, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("ENU East offset from dataset start (m)")
    ax.set_ylabel("ENU North offset from dataset start (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_comparison_map(
    *,
    pose_series: RawPoseSeries,
    point_rows: list[dict[str, Any]],
    modes: list[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "front": "#2563eb",
        "front_front_tele": "#dc2626",
    }
    gt_world = pose_series.full_gt_world()
    ref = gt_world[0, :2]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(gt_world[:, 0] - ref[0], gt_world[:, 1] - ref[1], color="#111827", linewidth=2.2, label="GT driven path")

    for mode in modes:
        rows_for_mode = [row for row in point_rows if row["camera_mode"] == mode and row["path_type"] == "final_path"]
        by_sample: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows_for_mode:
            by_sample.setdefault((int(row["chunk_id"]), int(row["sample_id"])), []).append(row)
        for _key, rows in sorted(by_sample.items()):
            rows = sorted(rows, key=lambda row: int(row["horizon_index"]))
            pred = np.asarray([[float(row["global_pred_e_m"]), float(row["global_pred_n_m"])] for row in rows])
            ax.plot(
                pred[:, 0] - ref[0],
                pred[:, 1] - ref[1],
                color=colors.get(mode, "#7c3aed"),
                alpha=0.52,
                linewidth=1.2,
            )
    handles = [
        plt.Line2D([0], [0], color="#111827", lw=2.2, label="GT driven path"),
        plt.Line2D([0], [0], color=colors.get("front", "#2563eb"), lw=2, label="front prediction"),
        plt.Line2D([0], [0], color=colors.get("front_front_tele", "#dc2626"), lw=2, label="front+tele prediction"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_title("front-only vs front+tele dynamic paths")
    ax.set_xlabel("ENU East offset from dataset start (m)")
    ax.set_ylabel("ENU North offset from dataset start (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def paired_comparison_rows(sample_rows: list[dict[str, Any]], modes: list[str]) -> list[dict[str, Any]]:
    if len(modes) < 2:
        return []
    left, right = modes[0], modes[1]
    rows_by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in sample_rows:
        rows_by_key[(str(row["camera_mode"]), int(row["chunk_id"]), int(row["sample_id"]), str(row["path_type"]))] = row

    out: list[dict[str, Any]] = []
    all_keys = sorted({(chunk, sample, path_type) for (_mode, chunk, sample, path_type) in rows_by_key})
    for chunk_id, sample_id, path_type in all_keys:
        a = rows_by_key.get((left, chunk_id, sample_id, path_type))
        b = rows_by_key.get((right, chunk_id, sample_id, path_type))
        if not a or not b:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "sample_id": sample_id,
                "path_type": path_type,
                "left_mode": left,
                "right_mode": right,
                "left_ade_m": float(a["mean_error_m"]),
                "right_ade_m": float(b["mean_error_m"]),
                "delta_ade_right_minus_left_m": float(b["mean_error_m"]) - float(a["mean_error_m"]),
                "left_fde_m": float(a["fde_m"]),
                "right_fde_m": float(b["fde_m"]),
                "delta_fde_right_minus_left_m": float(b["fde_m"]) - float(a["fde_m"]),
                "left_ate_m": float(a["ate_m"]),
                "right_ate_m": float(b["ate_m"]),
                "delta_ate_right_minus_left_m": float(b["ate_m"]) - float(a["ate_m"]),
                "left_inference_time_s": float(a["inference_time_s"]),
                "right_inference_time_s": float(b["inference_time_s"]),
            }
        )
    return out


def summarize_paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    if not rows:
        return {"summary": out}
    for path_type in sorted({str(row["path_type"]) for row in rows}):
        selected = [row for row in rows if row["path_type"] == path_type]
        if not selected:
            continue
        out.append(
            {
                "path_type": path_type,
                "paired_sample_count": len(selected),
                "delta_ade_right_minus_left_mean_m": float(np.mean([float(row["delta_ade_right_minus_left_m"]) for row in selected])),
                "delta_fde_right_minus_left_mean_m": float(np.mean([float(row["delta_fde_right_minus_left_m"]) for row in selected])),
                "delta_ate_right_minus_left_mean_m": float(np.mean([float(row["delta_ate_right_minus_left_m"]) for row in selected])),
                "right_better_ade_count": int(sum(float(row["delta_ade_right_minus_left_m"]) < 0.0 for row in selected)),
                "right_better_fde_count": int(sum(float(row["delta_fde_right_minus_left_m"]) < 0.0 for row in selected)),
            }
        )
    return {"summary": out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dynamic Alpamayo inference for raw live datasets and compare camera modes "
            "on the same primary-mode selected timestamps."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "raw_dataset_camera_compare_dynamic_20260527_test1",
    )
    parser.add_argument("--chunks", type=int, nargs="*", default=None)
    parser.add_argument("--camera-modes", nargs="+", default=["front", "front_front_tele"], choices=sorted(CAMERA_MODES))
    parser.add_argument("--primary-mode", default="front_front_tele", choices=sorted(CAMERA_MODES))
    parser.add_argument("--rebuild-request-bank", action="store_true")
    parser.add_argument("--reuse-existing-outputs", action="store_true")
    parser.add_argument("--require-full-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--future-len", type=int, default=64)
    parser.add_argument("--plan-dt-s", type=float, default=0.1)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--max-bank-requests", type=int, default=-1)
    parser.add_argument("--max-primary-requests", type=int, default=-1)
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
    parser.add_argument("--default-reused-inference-time-s", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_exists(args.dataset_root, "dataset root")
    for path, description in (
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engineDir"),
        (args.multimodal_engine_dir, "multimodalEngineDir"),
        (args.fm_engine, "fmEngine"),
    ):
        ensure_exists(Path(path), description)

    modes = list(dict.fromkeys(args.camera_modes))
    if args.primary_mode not in modes:
        modes.append(args.primary_mode)
    ordered_modes = [args.primary_mode] + [mode for mode in modes if mode != args.primary_mode]
    chunks = args.chunks if args.chunks is not None and len(args.chunks) else detect_chunks(args.dataset_root, modes)
    ensure_dir(args.output_root)

    pose_series = RawPoseSeries(args.dataset_root)
    request_banks: dict[tuple[str, int], Path] = {}
    all_entries: dict[tuple[str, int], list[RawRequestEntry]] = {}
    bank_summaries: dict[tuple[str, int], dict[str, Any]] = {}

    for mode in modes:
        for chunk_id in chunks:
            mode_chunk_root = args.output_root / mode / f"chunk{chunk_id:04d}"
            request_bank_root = mode_chunk_root / "request_bank"
            build_request_bank(
                dataset_root=args.dataset_root,
                chunk_id=chunk_id,
                mode_name=mode,
                request_bank_root=request_bank_root,
                args=args,
            )
            entries, bank_summary = load_entries(request_bank_root, mode, chunk_id)
            entries = filter_full_gt(entries, pose_series, args)
            if not entries:
                raise RuntimeError(f"No entries left after full-GT filter for {mode} chunk{chunk_id:04d}")
            request_banks[(mode, chunk_id)] = request_bank_root
            all_entries[(mode, chunk_id)] = entries
            bank_summaries[(mode, chunk_id)] = bank_summary
            print(
                f"[raw-camera-compare] {mode} chunk{chunk_id:04d}: candidates={len(entries)}",
                flush=True,
            )

    proc: subprocess.Popen[str] | None = None
    combined_sample_rows: list[dict[str, Any]] = []
    combined_point_rows: list[dict[str, Any]] = []
    combined_selected_rows: list[dict[str, Any]] = []
    selected_keys_by_chunk: dict[int, list[tuple[int, int]]] = {}
    mode_results: list[dict[str, Any]] = []

    try:
        for mode in ordered_modes:
            for chunk_id in chunks:
                entries = all_entries[(mode, chunk_id)]
                if mode == args.primary_mode:
                    candidate_entries = entries
                    if args.max_primary_requests > 0:
                        candidate_entries = candidate_entries[: args.max_primary_requests]
                    dynamic_entries: list[RawRequestEntry] = []
                    idx = 0
                    while idx < len(candidate_entries):
                        dynamic_entries.append(candidate_entries[idx])
                        # The actual stepping is applied after each inference below, so run one at a time.
                        break
                    selected_entries: list[RawRequestEntry] = []
                    # Process primary online because the next index depends on measured latency.
                    output_root = args.output_root / mode / f"chunk{chunk_id:04d}" / "outputs"
                    artifact_root = args.output_root / mode / f"chunk{chunk_id:04d}" / "artifacts"
                    evaluation_root = args.output_root / mode / f"chunk{chunk_id:04d}" / "evaluation"
                    ensure_dir(output_root)
                    ensure_dir(artifact_root)
                    ensure_dir(evaluation_root)
                    selected_rows: list[dict[str, Any]] = []
                    sample_rows: list[dict[str, Any]] = []
                    point_rows: list[dict[str, Any]] = []
                    idx = 0
                    run_start = time.time()
                    while idx < len(candidate_entries):
                        entry = candidate_entries[idx]
                        output_path = output_path_for_request(output_root, entry.request_path)
                        proc, inference_time_s, reused = run_one_inference(proc, args, entry, output_path)
                        sample_artifact_root = artifact_root / entry.request_path.stem
                        summaries = build_artifacts_for_entry(
                            args=args,
                            dataset_root=args.dataset_root,
                            pose_series=pose_series,
                            entry=entry,
                            output_path=output_path,
                            artifact_root=sample_artifact_root,
                            history_len=int(args.history_len),
                            inference_time_s=inference_time_s,
                        )
                        append_metric_rows(entry=entry, summaries=summaries, point_rows=point_rows, sample_rows=sample_rows)
                        selected_entries.append(entry)
                        selected_row = {
                            "camera_mode": mode,
                            "chunk_id": chunk_id,
                            "selected_index": len(selected_entries) - 1,
                            "candidate_index": idx,
                            "sample_id": entry.sample_id,
                            "t0_us": entry.t0_us,
                            "actual_offset_s": entry.actual_offset_s,
                            "selection_kind": "primary_dynamic_measured_inference_time",
                            "request_json": str(entry.request_path),
                            "output_json": str(output_path),
                            "artifact_root": str(sample_artifact_root),
                            "inference_time_s": float(inference_time_s),
                            "inference_reused": bool(reused),
                        }
                        selected_rows.append(selected_row)
                        elapsed = time.time() - run_start
                        print(
                            f"[raw-camera-compare] {mode} chunk{chunk_id:04d} "
                            f"#{len(selected_entries)} sid={entry.sample_id:05d} "
                            f"offset={entry.actual_offset_s:.2f}s infer={inference_time_s:.3f}s "
                            f"reused={reused} elapsed={elapsed/60:.1f}m",
                            flush=True,
                        )
                        next_idx = select_next_index(candidate_entries, idx, inference_time_s)
                        if next_idx <= idx:
                            next_idx = idx + 1
                        idx = next_idx

                    selected_keys_by_chunk[chunk_id] = [(entry.chunk_id, entry.sample_id) for entry in selected_entries]
                    write_json(evaluation_root / "selected_schedule.json", selected_rows)
                    write_csv(evaluation_root / "sample_metrics_dynamic.csv", sample_rows)
                    write_csv(evaluation_root / "path_points_dynamic.csv", point_rows)
                    metrics_summary = summarize_metrics(sample_rows)
                    write_json(evaluation_root / "summary_metrics.json", metrics_summary)
                    combined_selected_rows.extend(selected_rows)
                    combined_sample_rows.extend(sample_rows)
                    combined_point_rows.extend(point_rows)
                    mode_results.append(
                        {
                            "camera_mode": mode,
                            "chunk_id": chunk_id,
                            "selection_kind": "primary_dynamic_measured_inference_time",
                            "selected_count": len(selected_rows),
                            "evaluation_root": str(evaluation_root),
                            "metrics_summary": metrics_summary,
                        }
                    )
                else:
                    selected_ids = {sample_id for _chunk, sample_id in selected_keys_by_chunk[chunk_id]}
                    paired_entries = [entry for entry in entries if entry.sample_id in selected_ids]
                    proc, metrics_summary, selected_rows, sample_rows, point_rows = process_entries(
                        args=args,
                        dataset_root=args.dataset_root,
                        pose_series=pose_series,
                        proc=proc,
                        mode_name=mode,
                        chunk_id=chunk_id,
                        entries=paired_entries,
                        run_root=args.output_root / mode / f"chunk{chunk_id:04d}",
                        selection_kind=f"paired_to_{args.primary_mode}_dynamic_schedule",
                    )
                    combined_selected_rows.extend(selected_rows)
                    combined_sample_rows.extend(sample_rows)
                    combined_point_rows.extend(point_rows)
                    mode_results.append(
                        {
                            "camera_mode": mode,
                            "chunk_id": chunk_id,
                            "selection_kind": f"paired_to_{args.primary_mode}_dynamic_schedule",
                            "selected_count": len(selected_rows),
                            "evaluation_root": str(args.output_root / mode / f"chunk{chunk_id:04d}" / "evaluation"),
                            "metrics_summary": metrics_summary,
                        }
                    )
    finally:
        shutdown_runtime(proc)

    evaluation_root = args.output_root / "evaluation"
    write_csv(evaluation_root / "selected_schedule_all.csv", combined_selected_rows)
    write_csv(evaluation_root / "sample_metrics_all.csv", combined_sample_rows)
    write_csv(evaluation_root / "path_points_all.csv", combined_point_rows)
    overall_summary = summarize_metrics(combined_sample_rows)
    write_json(evaluation_root / "summary_metrics.json", overall_summary)

    for mode in modes:
        mode_points = [row for row in combined_point_rows if row["camera_mode"] == mode]
        plot_mode_map(
            title=f"{mode}: dynamic paths vs GT",
            pose_series=pose_series,
            point_rows=mode_points,
            output_path=evaluation_root / f"map_overview_{mode}.png",
        )
    plot_comparison_map(
        pose_series=pose_series,
        point_rows=combined_point_rows,
        modes=modes,
        output_path=evaluation_root / "map_compare_front_vs_front_tele.png",
    )

    paired_rows = paired_comparison_rows(combined_sample_rows, ["front", "front_front_tele"])
    write_csv(evaluation_root / "paired_camera_comparison.csv", paired_rows)
    paired_summary = summarize_paired(paired_rows)
    write_json(evaluation_root / "paired_camera_comparison_summary.json", paired_summary)

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "chunks": [int(chunk) for chunk in chunks],
        "camera_modes": modes,
        "primary_mode": args.primary_mode,
        "selection_policy": (
            f"{args.primary_mode}: next sample t0 >= previous sample t0 + measured inference_time_s; "
            f"other modes: paired to {args.primary_mode} selected sample ids"
        ),
        "require_full_gt": bool(args.require_full_gt),
        "request_banks": {f"{mode}/chunk{chunk_id:04d}": str(path) for (mode, chunk_id), path in request_banks.items()},
        "bank_summaries": {f"{mode}/chunk{chunk_id:04d}": summary for (mode, chunk_id), summary in bank_summaries.items()},
        "runs": mode_results,
        "overall_metrics_summary": overall_summary,
        "paired_comparison_summary": paired_summary,
        "evaluation_root": str(evaluation_root),
        "maps": {
            "front": str(evaluation_root / "map_overview_front.png"),
            "front_front_tele": str(evaluation_root / "map_overview_front_front_tele.png"),
            "comparison": str(evaluation_root / "map_compare_front_vs_front_tele.png"),
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
