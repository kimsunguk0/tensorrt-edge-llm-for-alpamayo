#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_live_chunk_request_bank as request_bank
from scripts.run_raw_dataset_one_shot_udp import build_result_artifacts
from scripts.run_request_bank_persistent import build_env, read_status


DEFAULT_LEGACY_FM = Path(
    "/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/"
    "teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan"
)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    label: str
    color: str
    engine_dir: Path
    multimodal_engine_dir: Path
    fm_engine: Path
    use_prefill_kv: bool
    diffusion_seed: int
    diffusion_num_steps: int
    max_generate_length: int


@dataclass(frozen=True)
class OverlayVariant:
    name: str
    label: str
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 20-sample request banks for the 2026-06-12 raw datasets, run "
            "legacy/FLEX prefill/decode variants, and render 3-way trajectory overlays."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "data" / "2026-06-12-test1",
            REPO_ROOT / "data" / "2026-06-12-test2",
        ],
    )
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "flex_legacy_threeway_20260612")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--legacy-engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--legacy-multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--legacy-fm-engine", type=Path, default=DEFAULT_LEGACY_FM)
    parser.add_argument("--legacy-seed", type=int, default=42)
    parser.add_argument("--legacy-steps", type=int, default=2)
    parser.add_argument(
        "--flex-engine-dir",
        type=Path,
        default=Path("/workspace/models/student_weights/engines/flex_k512_fp16/llm"),
    )
    parser.add_argument(
        "--flex-multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/student_weights/engines/flex_k512_fp16"),
    )
    parser.add_argument(
        "--flex-fm-engine",
        type=Path,
        default=Path("/workspace/models/student_weights/engines/flex_k512_fp16/ae28/ae28_single_step.plan"),
    )
    parser.add_argument("--flex-seed", type=int, default=42)
    parser.add_argument("--flex-steps", type=int, default=4)
    parser.add_argument("--flex-decode-max-generate-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=900.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_evenly_spaced_rows(samples: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if sample_count <= 0 or len(samples) <= sample_count:
        return samples.copy().reset_index(drop=True)
    indices = np.linspace(0, len(samples) - 1, sample_count, dtype=np.int64)
    indices = np.unique(indices)
    return samples.iloc[indices].copy().reset_index(drop=True)


def build_dataset_request_bank(
    *,
    dataset_root: Path,
    out_root: Path,
    chunk_id: int,
    sample_count: int,
    width: int,
    height: int,
    history_len: int,
    dt_s: float,
    traj_token_offset: int,
    temperature: float,
    top_p: float,
    top_k: int,
    camera_semantics: list[str] | tuple[str, ...] | None = None,
    sample_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_root = out_root / "requests"
    image_root = out_root / "images"
    ego_root = out_root / "ego"
    cache_root = out_root / "frame_cache"
    for path in (request_root, image_root, ego_root, cache_root):
        path.mkdir(parents=True, exist_ok=True)

    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[
        ["frame_id", "chunk_id"]
    ]
    sample_index = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    chunk_samples = (
        sample_index[sample_index["chunk_id"] == chunk_id]
        .copy()
        .sort_values("t0_utc_ns")
        .reset_index(drop=True)
    )
    if chunk_samples.empty:
        raise RuntimeError(f"No samples found for {dataset_root} chunk {chunk_id}")
    if sample_ids:
        wanted = set(int(x) for x in sample_ids)
        samples = chunk_samples[chunk_samples["sample_id"].isin(wanted)].copy().reset_index(drop=True)
        if samples.empty:
            raise RuntimeError(f"No requested sample ids found for {dataset_root} chunk {chunk_id}: {sorted(wanted)}")
    else:
        samples = select_evenly_spaced_rows(chunk_samples, sample_count)

    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = request_bank.select_pose_gnss_rows(gnss)
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

    selected_semantics = set(camera_semantics or request_bank.CAMERA_SEMANTIC_NAMES)
    selected_camera_order = [item for item in request_bank.CAMERA_RUNTIME_ORDER if item[0] in selected_semantics]
    if not selected_camera_order:
        raise RuntimeError("No cameras selected for request bank")

    for semantic_name, sensor_name, _, _ in selected_camera_order:
        frame_df = pd.read_parquet(dataset_root / "sensors" / sensor_name / "frames.parquet")
        frame_df = frame_df[frame_df["chunk_id"] == chunk_id].copy().sort_values("timestamp_utc_ns")
        if frame_df.empty:
            raise RuntimeError(f"No frames found for {sensor_name} chunk {chunk_id}")
        frame_ts = frame_df["timestamp_utc_ns"].to_numpy(dtype=np.int64)
        frame_ids = frame_df["frame_id"].to_numpy(dtype=np.int64)
        frame_idx_in_chunk = frame_df["frame_index_in_chunk"].to_numpy(dtype=np.int64)

        chosen = np.zeros((len(samples), len(offsets_ns)), dtype=np.int64)
        for j, offset_ns in enumerate(offsets_ns):
            nearest_idx = request_bank.nearest_frame_indices(frame_ts, t0s + offset_ns)
            chosen[:, j] = frame_ids[nearest_idx]

        id_to_chunk_idx = dict(zip(frame_ids.tolist(), frame_idx_in_chunk.tolist(), strict=True))
        frame_index_to_out_path: dict[int, Path] = {}
        for frame_id in np.unique(chosen.reshape(-1)).tolist():
            cache_path = cache_root / semantic_name / f"frame_{int(frame_id)}.png"
            if not cache_path.exists():
                frame_index_to_out_path[id_to_chunk_idx[int(frame_id)]] = cache_path

        request_bank.extract_unique_frames(
            video_path=dataset_root / "sensors" / sensor_name / "chunks" / f"chunk_{chunk_id:04d}.mkv",
            frame_index_to_out_path=frame_index_to_out_path,
            width=width,
            height=height,
        )
        per_camera_selected[semantic_name] = chosen

    manifest_rows: list[dict[str, Any]] = []
    dt_ns = int(round(dt_s * 1e9))
    camera_indices = [item[3] for item in selected_camera_order]

    for idx, row in samples.iterrows():
        sample_id = int(row["sample_id"])
        front_frame_id = int(row["front_frame_id"])
        t0_utc_ns = int(row["t0_utc_ns"])
        t0_us = t0_utc_ns // 1000
        stem = f"chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}"

        hist_xyz, hist_rot = request_bank.build_pose_history(
            gnss_valid=gnss_valid,
            ref_lla=ref_lla,
            t0_utc_ns=t0_utc_ns,
            history_len=history_len,
            dt_ns=dt_ns,
        )

        sample_ego_dir = ego_root / stem
        sample_image_dir = image_root / stem
        sample_ego_dir.mkdir(parents=True, exist_ok=True)
        sample_image_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = sample_ego_dir / "ego_history_xyz.npy"
        rot_path = sample_ego_dir / "ego_history_rot.npy"
        np.save(xyz_path, hist_xyz.astype(np.float32))
        np.save(rot_path, hist_rot.astype(np.float32))

        selected_frame_record: dict[str, list[int]] = {}
        for semantic_name, _, _, cam_id in selected_camera_order:
            chosen_ids = per_camera_selected[semantic_name][idx].tolist()
            selected_frame_record[semantic_name] = [int(x) for x in chosen_ids]
            for step_idx, frame_id in enumerate(chosen_ids):
                cache_path = cache_root / semantic_name / f"frame_{int(frame_id)}.png"
                link_path = sample_image_dir / f"cam{cam_id}_f{step_idx}.png"
                request_bank.safe_symlink(cache_path, link_path)

        request = {
            "batch_size": 1,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_generate_length": 20,
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
                            "content": request_bank.build_user_content(sample_image_dir, camera_indices, None),
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
                    ],
                    "ego_history_xyz_npy": str(xyz_path),
                    "ego_history_rot_npy": str(rot_path),
                    "traj_token_offset": int(traj_token_offset),
                    "action_space_constants": dict(request_bank.DEFAULT_ACTION_SPACE_CONSTANTS),
                    "diffusion_seed": 42,
                    "diffusion_num_steps": 4,
                }
            ],
        }
        request_path = request_root / f"request_{stem}.json"
        write_json(request_path, request)

        manifest_rows.append(
            {
                "dataset_root": str(dataset_root),
                "chunk_id": int(chunk_id),
                "sample_id": sample_id,
                "front_frame_id": front_frame_id,
                "t0_utc_ns": t0_utc_ns,
                "t0_us": t0_us,
                "chunk_start_utc_ns": int(chunk_samples["t0_utc_ns"].iloc[0]),
                "target_offset_s": float(t0_utc_ns - int(chunk_samples["t0_utc_ns"].iloc[0])) / 1e9,
                "actual_offset_s": float(t0_utc_ns - int(chunk_samples["t0_utc_ns"].iloc[0])) / 1e9,
                "request_json": str(request_path),
                "image_dir": str(sample_image_dir),
                "ego_history_xyz_npy": str(xyz_path),
                "ego_history_rot_npy": str(rot_path),
                "selected_frames": selected_frame_record,
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "chunk_id": int(chunk_id),
        "num_requests": len(manifest_rows),
        "sample_count_requested": int(sample_count),
        "sample_ids": [int(row["sample_id"]) for row in manifest_rows],
        "width": int(width),
        "height": int(height),
        "history_len": int(history_len),
        "dt_s": float(dt_s),
        "camera_indices": camera_indices,
        "camera_runtime_order": [
            {"semantic_name": semantic_name, "camera_id": int(cam_id)}
            for semantic_name, _, _, cam_id in selected_camera_order
        ],
        "chunk_ref_lla": [float(x) for x in ref_lla],
        "request_root": str(request_root),
        "image_root": str(image_root),
        "ego_root": str(ego_root),
    }
    write_json(out_root / "manifest.json", manifest_rows)
    write_json(out_root / "summary.json", summary)
    return manifest_rows, summary


def prepare_variant_requests(
    *,
    manifest_rows: list[dict[str, Any]],
    variant: VariantConfig,
    out_root: Path,
) -> list[Path]:
    request_root = out_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    request_paths: list[Path] = []
    for row in manifest_rows:
        src = Path(row["request_json"])
        request = load_json(src)
        request["max_generate_length"] = int(variant.max_generate_length)
        request["requests"][0]["diffusion_seed"] = int(variant.diffusion_seed)
        request["requests"][0]["diffusion_num_steps"] = int(variant.diffusion_num_steps)
        dst = request_root / src.name
        write_json(dst, request)
        request_paths.append(dst)
    return request_paths


def run_variant_outputs(
    *,
    variant: VariantConfig,
    request_paths: list[Path],
    output_root: Path,
    llm_inference_bin: Path,
    plugin_lib: Path,
    warmup: int,
    timeout_per_request: float,
    skip_existing: bool,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [output_root / p.name.replace("request_", "output_") for p in request_paths]
    if skip_existing and output_paths and all(path.exists() for path in output_paths):
        print(f"[threeway] reuse all outputs for {variant.name}", flush=True)
        return output_paths

    cmd = [
        str(llm_inference_bin),
        "--engineDir",
        str(variant.engine_dir),
        "--multimodalEngineDir",
        str(variant.multimodal_engine_dir),
        "--fmEngine",
        str(variant.fm_engine),
        "--alpamayoPostVlmRuntime",
        "--persistentServer",
        "--warmup",
        str(warmup),
    ]
    if variant.use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")

    print(f"[threeway] starting {variant.name}", flush=True)
    print("[threeway] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(plugin_lib),
    )

    try:
        ready = read_status(proc, timeout_s=180.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected llm_inference ready state for {variant.name}: {ready}")
        print(f"[threeway] {variant.name} ready", flush=True)

        completed = 0
        start = time.time()
        for request_path, output_path in zip(request_paths, output_paths, strict=True):
            if skip_existing and output_path.exists():
                completed += 1
                print(f"[threeway] skip {variant.name} {completed}/{len(request_paths)} {output_path.name}", flush=True)
                continue

            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            t0 = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"{variant.name} failed for {request_path.name}: {status}")
            completed += 1
            elapsed = time.time() - start
            avg = elapsed / max(completed, 1)
            eta = avg * (len(request_paths) - completed)
            print(
                f"[threeway] {variant.name} done {completed}/{len(request_paths)} "
                f"{request_path.name} in {time.time() - t0:.2f}s | eta {eta/60:.1f}m",
                flush=True,
            )
    finally:
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

    return output_paths


def build_variant_artifacts(
    *,
    variant: VariantConfig,
    output_paths: list[Path],
    request_paths: list[Path],
    manifest_rows: list[dict[str, Any]],
    dataset_root: Path,
    history_len: int,
    artifact_root: Path,
) -> dict[str, dict[str, Any]]:
    by_request_stem: dict[str, dict[str, Any]] = {}
    rows_by_request_name = {Path(row["request_json"]).name: row for row in manifest_rows}
    for request_path, output_path in zip(request_paths, output_paths, strict=True):
        row = rows_by_request_name[request_path.name]
        per_sample_root = artifact_root / request_path.stem
        final_summary, ac_summary, gt_summary, _ = build_result_artifacts(
            output_path=output_path,
            metadata=row,
            dataset_root=dataset_root,
            history_len=history_len,
            artifact_root=per_sample_root,
        )
        by_request_stem[request_path.stem] = {
            "variant": variant.name,
            "request_json": str(request_path),
            "output_json": str(output_path),
            "artifact_root": str(per_sample_root),
            "final": final_summary,
            "ac_decoded": ac_summary,
            "gt": gt_summary,
        }
    return by_request_stem


def path_xy(summary: dict[str, Any]) -> np.ndarray:
    pred_xyz = np.asarray(summary["pred_xyz"], dtype=np.float32)
    points = np.zeros((pred_xyz.shape[0] + 1, 2), dtype=np.float32)
    points[1:, 0] = pred_xyz[:, 0]
    points[1:, 1] = pred_xyz[:, 1]
    return points


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def timing_value(summary: dict[str, Any], key: str) -> float | None:
    timing = summary.get("timing", {})
    value = timing.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def short_text(text: str, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def draw_overlay_png(
    *,
    out_path: Path,
    dataset_name: str,
    row: dict[str, Any],
    variants: list[OverlayVariant],
    summaries_by_variant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fig = plt.figure(figsize=(13.5, 10.8), dpi=160)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 2.1], hspace=0.22, wspace=0.08)
    fig.patch.set_facecolor("#f8fafc")

    image_dir = Path(row["image_dir"])
    camera_items = [
        ("cam0 f3", image_dir / "cam0_f3.png"),
        ("cam1 f3", image_dir / "cam1_f3.png"),
        ("cam2 f3", image_dir / "cam2_f3.png"),
        ("cam6 f3", image_dir / "cam6_f3.png"),
    ]
    for idx, (title, image_path) in enumerate(camera_items):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.set_title(title, fontsize=9)
        ax_img.axis("off")
        if image_path.exists():
            ax_img.imshow(plt.imread(image_path))
        else:
            ax_img.text(0.5, 0.5, "missing", ha="center", va="center")

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    header = (
        f"{dataset_name} | chunk {row['chunk_id']:04d} | sample {row['sample_id']} | "
        f"t0_us {row['t0_us']}"
    )
    lines = [header]
    for variant in variants:
        summary = summaries_by_variant[variant.name]["final"]
        pts = path_xy(summary)
        total_ms = timing_value(summary, "total_post_vlm_ms")
        fm_ms = timing_value(summary, "fm_wall_ms")
        text = short_text(summary.get("final_output", ""))
        if not text:
            text = "(no decode text)"
        fm_text = f"{fm_ms:.1f}ms" if fm_ms is not None else "n/a"
        lines.append(
            f"{variant.label}: end=({pts[-1, 0]:.2f},{pts[-1, 1]:.2f})m "
            f"len={path_length(pts):.2f}m total={total_ms if total_ms is not None else float('nan'):.1f}ms "
            f"fm={fm_text} | {text}"
        )
    ax_text.text(
        0.01,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        transform=ax_text.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94, "pad": 6},
    )

    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor("#ffffff")
    hist_xyz = np.load(row["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    ax.plot(hist_xyz[:, 1], hist_xyz[:, 0], color="#0f172a", linewidth=1.6, alpha=0.72, label="ego history")

    gt_summary = next(iter(summaries_by_variant.values()))["gt"]
    gt_pts = path_xy(gt_summary)
    ax.plot(gt_pts[:, 1], gt_pts[:, 0], color="#64748b", linewidth=2.0, alpha=0.72, linestyle="--", label="GNSS GT")

    all_pts = [hist_xyz[:, :2], gt_pts]
    for variant in variants:
        summary = summaries_by_variant[variant.name]["final"]
        pts = path_xy(summary)
        all_pts.append(pts)
        ax.plot(
            pts[:, 1],
            pts[:, 0],
            color=variant.color,
            linewidth=2.4,
            alpha=0.95,
            label=variant.label,
        )
        ax.scatter(pts[-1, 1], pts[-1, 0], color=variant.color, s=22, zorder=5)

    ax.scatter([0.0], [0.0], color="#111827", s=28, zorder=6, label="t0")
    stack = np.concatenate([np.asarray(p, dtype=np.float32)[:, :2] for p in all_pts], axis=0)
    x_forward_min, y_lat_min = stack.min(axis=0)
    x_forward_max, y_lat_max = stack.max(axis=0)
    forward_pad = max(5.0, float(x_forward_max - x_forward_min) * 0.15)
    lat_pad = max(3.0, float(y_lat_max - y_lat_min) * 0.20)
    ax.set_ylim(float(x_forward_min - forward_pad), float(x_forward_max + forward_pad))
    ax.set_xlim(float(y_lat_min - lat_pad), float(y_lat_max + lat_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    ax.set_title("Local trajectory overlay")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    entry: dict[str, Any] = {
        "dataset": dataset_name,
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "t0_us": int(row["t0_us"]),
        "png": str(out_path),
        "variants": {},
    }
    for variant in variants:
        summary = summaries_by_variant[variant.name]["final"]
        pts = path_xy(summary)
        entry["variants"][variant.name] = {
            "label": variant.label,
            "output_json": summaries_by_variant[variant.name]["output_json"],
            "end_xy": [float(pts[-1, 0]), float(pts[-1, 1])],
            "path_len_m": path_length(pts),
            "total_post_vlm_ms": timing_value(summary, "total_post_vlm_ms"),
            "fm_wall_ms": timing_value(summary, "fm_wall_ms"),
            "final_output": summary.get("final_output", ""),
        }
    return entry


OFFICIAL_VARIANT_CONFIGS = {
    "official_10b_ae": OverlayVariant(
        name="official_10b_ae",
        label="Official 10B AE",
        color="#ea580c",
    ),
    "official_10b_discrete128": OverlayVariant(
        name="official_10b_discrete128",
        label="Official 10B VLM discrete128",
        color="#0891b2",
    ),
}


def runtime_overlay_variants(variants: list[VariantConfig]) -> list[OverlayVariant]:
    return [OverlayVariant(name=v.name, label=v.label, color=v.color) for v in variants]


def load_official_10b_artifacts(
    *,
    dataset_work: Path,
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[OverlayVariant], dict[str, dict[str, dict[str, Any]]]]:
    jsonl_path = dataset_work / "official_10b" / "predictions.jsonl"
    if not jsonl_path.exists():
        return [], {}

    stem_by_key = {
        (int(row["chunk_id"]), int(row["sample_id"])): Path(row["request_json"]).stem
        for row in manifest_rows
    }
    artifacts: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in OFFICIAL_VARIANT_CONFIGS
    }
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            name = str(item.get("label", ""))
            if name not in OFFICIAL_VARIANT_CONFIGS:
                continue
            key = (int(item["chunk_id"]), int(item["sample_id"]))
            stem = stem_by_key.get(key)
            if stem is None:
                continue
            runtime_s = item.get("runtime_s")
            total_ms = None if runtime_s is None else float(runtime_s) * 1000.0
            final_summary = {
                "label": name,
                "chunk_id": int(item["chunk_id"]),
                "front_frame_id": int(item.get("front_frame_id", -1)),
                "t0_utc_ns": int(item["t0_utc_ns"]),
                "t0_us": int(item["t0_us"]),
                "final_output": item.get("cot") or "",
                "timing": {
                    "total_post_vlm_ms": total_ms,
                    "runtime_s": runtime_s,
                },
                "plan_dt_s": float(item.get("plan_dt_s", 0.1)),
                "plan_points_no_origin": int(item.get("plan_points_no_origin", len(item["pred_xyz"]))),
                "pred_xyz": item["pred_xyz"],
                "path_type": name,
                "discrete_token_count_between_markers": item.get(
                    "discrete_token_count_between_markers"
                ),
                "expected_discrete_tokens": item.get("expected_discrete_tokens"),
            }
            artifacts[name][stem] = {
                "variant": name,
                "request_json": "",
                "output_json": str(jsonl_path),
                "artifact_root": str(dataset_work / "official_10b"),
                "final": final_summary,
                "ac_decoded": final_summary,
                "gt": {},
            }

    active = [
        cfg
        for name, cfg in OFFICIAL_VARIANT_CONFIGS.items()
        if artifacts.get(name)
    ]
    artifacts = {name: by_stem for name, by_stem in artifacts.items() if by_stem}
    return active, artifacts


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 4, thumb_width: int = 680) -> Path:
    if not image_paths:
        raise RuntimeError("No images for contact sheet")
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        aspect = img.height / max(img.width, 1)
        thumb = img.resize((thumb_width, int(thumb_width * aspect)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
        labels.append(path.stem)

    label_h = 22
    cell_w = thumb_width
    cell_h = max(img.height for img in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 8, y + 4), labels[idx], fill="#0f172a")
        sheet.paste(thumb, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def variants_from_args(args: argparse.Namespace) -> list[VariantConfig]:
    return [
        VariantConfig(
            name="legacy_prefill",
            label="Legacy prefill KV",
            color="#2563eb",
            engine_dir=args.legacy_engine_dir,
            multimodal_engine_dir=args.legacy_multimodal_engine_dir,
            fm_engine=args.legacy_fm_engine,
            use_prefill_kv=True,
            diffusion_seed=args.legacy_seed,
            diffusion_num_steps=args.legacy_steps,
            max_generate_length=20,
        ),
        VariantConfig(
            name="legacy_decode128",
            label="Legacy decode 128 KV",
            color="#7c3aed",
            engine_dir=args.legacy_engine_dir,
            multimodal_engine_dir=args.legacy_multimodal_engine_dir,
            fm_engine=args.legacy_fm_engine,
            use_prefill_kv=False,
            diffusion_seed=args.legacy_seed,
            diffusion_num_steps=args.legacy_steps,
            max_generate_length=128,
        ),
        VariantConfig(
            name="flex_decode",
            label="FLEX decode KV",
            color="#dc2626",
            engine_dir=args.flex_engine_dir,
            multimodal_engine_dir=args.flex_multimodal_engine_dir,
            fm_engine=args.flex_fm_engine,
            use_prefill_kv=False,
            diffusion_seed=args.flex_seed,
            diffusion_num_steps=args.flex_steps,
            max_generate_length=args.flex_decode_max_generate_length,
        ),
        VariantConfig(
            name="flex_prefill",
            label="FLEX prefill KV",
            color="#16a34a",
            engine_dir=args.flex_engine_dir,
            multimodal_engine_dir=args.flex_multimodal_engine_dir,
            fm_engine=args.flex_fm_engine,
            use_prefill_kv=True,
            diffusion_seed=args.flex_seed,
            diffusion_num_steps=args.flex_steps,
            max_generate_length=20,
        ),
    ]


def main() -> None:
    args = parse_args()
    ensure_exists(args.llm_inference_bin, "llm_inference")
    ensure_exists(args.plugin_lib, "plugin lib")
    for variant in variants_from_args(args):
        ensure_exists(variant.engine_dir, f"{variant.name} engine dir")
        ensure_exists(variant.multimodal_engine_dir, f"{variant.name} multimodal engine dir")
        ensure_exists(variant.fm_engine, f"{variant.name} fm engine")

    args.work_root.mkdir(parents=True, exist_ok=True)
    variants = variants_from_args(args)
    top_summary: dict[str, Any] = {
        "work_root": str(args.work_root),
        "chunk_id": int(args.chunk_id),
        "sample_count": int(args.sample_count),
        "variants": [
            {
                "name": v.name,
                "label": v.label,
                "engine_dir": str(v.engine_dir),
                "multimodal_engine_dir": str(v.multimodal_engine_dir),
                "fm_engine": str(v.fm_engine),
                "use_prefill_kv": bool(v.use_prefill_kv),
                "diffusion_seed": int(v.diffusion_seed),
                "diffusion_num_steps": int(v.diffusion_num_steps),
                "max_generate_length": int(v.max_generate_length),
            }
            for v in variants
        ],
        "datasets": {},
    }

    for dataset_root in args.dataset_roots:
        dataset_root = dataset_root.resolve()
        ensure_exists(dataset_root, "dataset root")
        dataset_name = dataset_root.name
        dataset_work = args.work_root / dataset_name
        request_bank_root = dataset_work / "request_bank"
        print(f"[threeway] building request bank for {dataset_name}", flush=True)
        manifest_rows, request_summary = build_dataset_request_bank(
            dataset_root=dataset_root,
            out_root=request_bank_root,
            chunk_id=args.chunk_id,
            sample_count=args.sample_count,
            width=args.width,
            height=args.height,
            history_len=args.history_len,
            dt_s=args.dt_s,
            traj_token_offset=args.traj_token_offset,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        all_variant_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
        for variant in variants:
            variant_root = dataset_work / "variants" / variant.name
            request_paths = prepare_variant_requests(
                manifest_rows=manifest_rows,
                variant=variant,
                out_root=variant_root / "request_bank",
            )
            output_paths = run_variant_outputs(
                variant=variant,
                request_paths=request_paths,
                output_root=variant_root / "outputs",
                llm_inference_bin=args.llm_inference_bin,
                plugin_lib=args.plugin_lib,
                warmup=args.warmup,
                timeout_per_request=args.timeout_per_request,
                skip_existing=args.skip_existing,
            )
            all_variant_artifacts[variant.name] = build_variant_artifacts(
                variant=variant,
                output_paths=output_paths,
                request_paths=request_paths,
                manifest_rows=manifest_rows,
                dataset_root=dataset_root,
                history_len=args.history_len,
                artifact_root=variant_root / "artifacts",
            )

        base_overlay_variants = runtime_overlay_variants(variants)
        official_overlay_variants, official_artifacts = load_official_10b_artifacts(
            dataset_work=dataset_work,
            manifest_rows=manifest_rows,
        )
        if official_overlay_variants:
            print(
                "[threeway] including official 10B overlays: "
                + ", ".join(v.name for v in official_overlay_variants),
                flush=True,
            )

        overlay_dir = dataset_work / "overlays"
        overlay_entries: list[dict[str, Any]] = []
        overlay_paths: list[Path] = []
        for row in manifest_rows:
            stem = Path(row["request_json"]).stem
            summaries_by_variant = {
                variant.name: all_variant_artifacts[variant.name][stem] for variant in variants
            }
            sample_overlay_variants = list(base_overlay_variants)
            for official_variant in official_overlay_variants:
                item = official_artifacts.get(official_variant.name, {}).get(stem)
                if item is not None:
                    summaries_by_variant[official_variant.name] = item
                    sample_overlay_variants.append(official_variant)
            out_png = overlay_dir / f"overlay_{dataset_name}_{stem.removeprefix('request_')}.png"
            entry = draw_overlay_png(
                out_path=out_png,
                dataset_name=dataset_name,
                row=row,
                variants=sample_overlay_variants,
                summaries_by_variant=summaries_by_variant,
            )
            overlay_entries.append(entry)
            overlay_paths.append(out_png)

        contact_sheet = make_contact_sheet(
            overlay_paths,
            dataset_work / f"contact_sheet_{dataset_name}_threeway.png",
            cols=4,
        )
        dataset_summary = {
            "request_bank": request_summary,
            "contact_sheet": str(contact_sheet),
            "overlays": overlay_entries,
        }
        write_json(dataset_work / "threeway_summary.json", dataset_summary)
        top_summary["datasets"][dataset_name] = {
            "dataset_root": str(dataset_root),
            "summary_json": str(dataset_work / "threeway_summary.json"),
            "contact_sheet": str(contact_sheet),
            "overlay_count": len(overlay_entries),
        }

    write_json(args.work_root / "summary.json", top_summary)
    print(json.dumps(top_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
