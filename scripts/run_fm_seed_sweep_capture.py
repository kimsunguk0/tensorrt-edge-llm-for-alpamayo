#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_capture_dynamic_inference as dyn
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "output"
    / "live_capture_dynamic_inference_20260526_063851_064421_latest_model"
    / "run_20260526_063851"
)
DEFAULT_CAPTURE_ROOT = Path("/workspace/live_camera_ego_history_capture/0526_1")
STALE_CAPTURE_ROOT = Path("/workspace/live_camera_ego_history_capture/run_20260526_063851")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FM seed sweep on a fixed set of saved live-capture samples.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "fm_seed_sweep_20260526_063851",
    )
    parser.add_argument("--seeds", default="0,1,2,42,100")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-future-distance-m", type=float, default=10.0)
    parser.add_argument("--min-first-step-speed-mps", type=float, default=0.5)
    parser.add_argument("--reuse-existing-outputs", action="store_true")
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=Path(
            "/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/"
            "teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan"
        ),
    )
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def replace_path_roots(value: Any, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, str):
        old = str(old_root)
        if value.startswith(old):
            return str(new_root) + value[len(old) :]
        return value
    if isinstance(value, list):
        return [replace_path_roots(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: replace_path_roots(item, old_root, new_root) for key, item in value.items()}
    return value


def with_origin(xyz: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1, 3), dtype=np.float64), xyz], axis=0)


def path_xyz(path: Path) -> np.ndarray:
    data = load_json(path)
    assert isinstance(data, dict)
    xyz = np.asarray(data.get("pred_xyz", []), dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    if xyz.shape[1] == 2:
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return xyz[:, :3]


def path_distance_m(xyz: np.ndarray) -> float:
    pts = with_origin(xyz)
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def first_step_speed_mps(xyz: np.ndarray, dt_s: float = 0.1) -> float:
    pts = with_origin(xyz)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[1, :2] - pts[0, :2]) / dt_s)


def interp_y_at_x(xyz: np.ndarray, x_m: float) -> float | None:
    pts = with_origin(xyz)
    x = pts[:, 0]
    y = pts[:, 1]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    if unique_x.size < 2 or x_m < float(unique_x[0]) or x_m > float(unique_x[-1]):
        return None
    return float(np.interp(float(x_m), unique_x, y_sorted[unique_idx]))


def compute_ade_fde(final_xyz: np.ndarray, gt_xyz: np.ndarray) -> tuple[float, float]:
    n = min(len(final_xyz), len(gt_xyz))
    if n == 0:
        return math.nan, math.nan
    err = np.linalg.norm(final_xyz[:n, :2] - gt_xyz[:n, :2], axis=1)
    return float(np.mean(err)), float(err[-1])


def choose_schedule_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    schedule = load_json(args.run_root / "evaluation" / "selected_schedule.json")
    assert isinstance(schedule, list)
    candidates: list[dict[str, Any]] = []
    for row in schedule:
        artifact_root = Path(row["artifact_root"])
        gt_path = artifact_root / "gt_path.json"
        if not gt_path.exists():
            continue
        gt_xyz = path_xyz(gt_path)
        distance = path_distance_m(gt_xyz)
        speed0 = first_step_speed_mps(gt_xyz)
        if distance < args.min_future_distance_m or speed0 < args.min_first_step_speed_mps:
            continue
        row = dict(row)
        row["gt_distance_m"] = distance
        row["gt_first_step_speed_mps"] = speed0
        candidates.append(row)
    if len(candidates) <= args.num_samples:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, args.num_samples, dtype=int)
    return [candidates[int(idx)] for idx in indices]


def fixed_ego_paths(capture_root: Path, sample_id: int, t0_us: int) -> tuple[Path, Path]:
    xyz = capture_root / "ego_history" / f"sample_{sample_id:06d}_t0_{t0_us}_ego_history_xyz.npy"
    rot = capture_root / "ego_history" / f"sample_{sample_id:06d}_t0_{t0_us}_ego_history_rot.npy"
    if xyz.exists() and rot.exists():
        return xyz, rot
    matches = sorted((capture_root / "ego_history").glob(f"sample_{sample_id:06d}_t0_*_ego_history_xyz.npy"))
    if not matches:
        raise FileNotFoundError(f"missing ego history sample={sample_id}")
    xyz = matches[0]
    rot = Path(str(xyz).replace("_xyz.npy", "_rot.npy"))
    return xyz, rot


def write_seed_request(row: dict[str, Any], *, seed: int, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source_request = Path(row["request_json"])
    req_obj = load_json(source_request)
    req_obj = replace_path_roots(req_obj, STALE_CAPTURE_ROOT, args.capture_root)
    assert isinstance(req_obj, dict)
    req_obj["requests"][0]["diffusion_seed"] = int(seed)

    sample_id = int(row["sample_id"])
    t0_us = int(row["t0_us"])
    xyz_path, rot_path = fixed_ego_paths(args.capture_root, sample_id, t0_us)
    req_obj["requests"][0]["ego_history_xyz_npy"] = str(xyz_path)
    req_obj["requests"][0]["ego_history_rot_npy"] = str(rot_path)

    request_dir = args.output_root / "requests" / f"seed_{seed}"
    request_path = request_dir / f"request_seed{seed}_sid{sample_id:06d}_t0_{t0_us}.json"
    write_json(request_path, req_obj)
    return request_path, xyz_path, rot_path


def runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        llm_inference_bin=args.llm_inference_bin,
        plugin_lib=args.plugin_lib,
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        warmup=args.warmup,
        alpamayo_fm_use_prefill_kv=args.alpamayo_fm_use_prefill_kv,
        alpamayo_nav_cfg=False,
        timeout_per_request=args.timeout_per_request,
        reuse_existing_outputs=args.reuse_existing_outputs,
    )


def summarize_by_seed(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for seed in seeds:
        subset = [row for row in rows if int(row["seed"]) == int(seed)]
        out: dict[str, Any] = {"seed": int(seed), "n": len(subset)}
        for key in (
            "final_minus_gt_y_at_5m",
            "final_minus_gt_y_at_10m",
            "final_y_at_10m",
            "gt_y_at_10m",
            "ade_m",
            "fde_m",
            "inference_time_s",
            "x0_std",
        ):
            vals = [
                float(row[key])
                for row in subset
                if row.get(key) not in (None, "") and math.isfinite(float(row[key]))
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[f"{key}_mean"] = float(np.mean(arr))
                out[f"{key}_median"] = float(np.median(arr))
                out[f"{key}_min"] = float(np.min(arr))
                out[f"{key}_max"] = float(np.max(arr))
                out[f"{key}_positive_pct"] = float(np.mean(arr > 0.0) * 100.0)
        summary.append(out)
    return summary


def make_plots(rows: list[dict[str, Any]], summary: list[dict[str, Any]], output_root: Path) -> None:
    seeds = [int(item["seed"]) for item in summary]
    data = [
        [
            float(row["final_minus_gt_y_at_10m"])
            for row in rows
            if int(row["seed"]) == seed and row.get("final_minus_gt_y_at_10m") not in (None, "")
        ]
        for seed in seeds
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=[str(seed) for seed in seeds], showmeans=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("diffusion seed")
    ax.set_ylabel("model - GT y at x=10m [m], +left")
    ax.set_title("FM seed sweep lateral bias")
    ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(output_root / "seed_sweep_y10_boxplot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    selected_rows = choose_schedule_rows(args)
    write_json(args.output_root / "selected_samples.json", selected_rows)

    rows = dyn.read_samples(args.capture_root)
    pose_series = dyn.CapturePoseSeries(rows, yaw_source="gnss")
    rt_args = runtime_args(args)
    proc = None
    result_rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for idx, row in enumerate(selected_rows):
                request_path, xyz_path, rot_path = write_seed_request(row, seed=seed, args=args)
                sample_id = int(row["sample_id"])
                t0_us = int(row["t0_us"])
                entry = dyn.RequestEntry(
                    request_path=request_path,
                    sample_id=sample_id,
                    t0_us=t0_us,
                    t0_utc_ns=t0_us * 1000,
                    actual_offset_s=float(row["actual_offset_s"]),
                    ego_history_xyz_npy=str(xyz_path),
                    ego_history_rot_npy=str(rot_path),
                    saved_images=(),
                    yaw_deg=None,
                    utm=None,
                )
                output_path = args.output_root / "outputs" / f"seed_{seed}" / f"output_seed{seed}_sid{sample_id:06d}_t0_{t0_us}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                proc, inference_time_s, reused = dyn.run_one_inference(proc, rt_args, entry, output_path)
                artifact_root = args.output_root / "artifacts" / f"seed_{seed}" / f"sid{sample_id:06d}_t0_{t0_us}"
                metadata = {
                    "chunk_id": 0,
                    "sample_id": sample_id,
                    "front_frame_id": sample_id,
                    "t0_utc_ns": t0_us * 1000,
                    "t0_us": t0_us,
                    "target_offset_s": float(row["actual_offset_s"]),
                    "actual_offset_s": float(row["actual_offset_s"]),
                    "request_json": str(request_path),
                    "ego_history_xyz_npy": str(xyz_path),
                    "ego_history_rot_npy": str(rot_path),
                }
                final_summary, ac_summary, gt_summary = dyn.build_capture_artifacts(
                    output_path=output_path,
                    metadata=metadata,
                    pose_series=pose_series,
                    history_len=16,
                    artifact_root=artifact_root,
                )
                final_xyz = np.asarray(final_summary["pred_xyz"], dtype=np.float64)
                gt_xyz = np.asarray(gt_summary["pred_xyz"], dtype=np.float64)
                ade_m, fde_m = compute_ade_fde(final_xyz, gt_xyz)
                output_obj = load_json(output_path)
                assert isinstance(output_obj, dict)
                fm = output_obj["responses"][0]["alpamayo_post_vlm"]["fm"]
                x0 = np.asarray(fm["x0"]["data"], dtype=np.float64)
                row_out = {
                    "seed": int(seed),
                    "sample_order": idx,
                    "sample_id": sample_id,
                    "t0_us": t0_us,
                    "actual_offset_s": float(row["actual_offset_s"]),
                    "inference_time_s": float(inference_time_s),
                    "reused": bool(reused),
                    "gt_distance_m": float(row["gt_distance_m"]),
                    "gt_first_step_speed_mps": float(row["gt_first_step_speed_mps"]),
                    "gt_y_at_5m": interp_y_at_x(gt_xyz, 5.0),
                    "final_y_at_5m": interp_y_at_x(final_xyz, 5.0),
                    "final_minus_gt_y_at_5m": None,
                    "gt_y_at_10m": interp_y_at_x(gt_xyz, 10.0),
                    "final_y_at_10m": interp_y_at_x(final_xyz, 10.0),
                    "final_minus_gt_y_at_10m": None,
                    "ade_m": ade_m,
                    "fde_m": fde_m,
                    "x0_mean": float(np.mean(x0)),
                    "x0_std": float(np.std(x0)),
                    "num_steps": int(fm.get("num_steps", -1)),
                    "guidance_weight": float(fm.get("guidance_weight", math.nan)),
                    "output_json": str(output_path),
                    "final_path_json": str(artifact_root / "final_path.json"),
                    "gt_path_json": str(artifact_root / "gt_path.json"),
                }
                if row_out["gt_y_at_5m"] is not None and row_out["final_y_at_5m"] is not None:
                    row_out["final_minus_gt_y_at_5m"] = float(row_out["final_y_at_5m"] - row_out["gt_y_at_5m"])
                if row_out["gt_y_at_10m"] is not None and row_out["final_y_at_10m"] is not None:
                    row_out["final_minus_gt_y_at_10m"] = float(row_out["final_y_at_10m"] - row_out["gt_y_at_10m"])
                result_rows.append(row_out)
                print(
                    f"[seed-sweep] seed={seed} sample={sample_id} "
                    f"dY10={row_out['final_minus_gt_y_at_10m']} infer={inference_time_s:.3f}s "
                    f"{len(result_rows)}/{len(seeds) * len(selected_rows)}",
                    flush=True,
                )
    finally:
        dyn.shutdown_runtime(proc)

    csv_path = args.output_root / "seed_sweep_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = summarize_by_seed(result_rows, seeds)
    write_json(args.output_root / "seed_sweep_summary.json", summary)
    make_plots(result_rows, summary, args.output_root)
    combined = {
        "output_root": str(args.output_root),
        "seeds": seeds,
        "num_samples": len(selected_rows),
        "metrics_csv": str(csv_path),
        "summary_json": str(args.output_root / "seed_sweep_summary.json"),
        "boxplot_png": str(args.output_root / "seed_sweep_y10_boxplot.png"),
        "summary": summary,
    }
    write_json(args.output_root / "summary.json", combined)
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
