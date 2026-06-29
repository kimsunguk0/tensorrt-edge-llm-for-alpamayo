#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_live_chunk_request_bank as request_bank
from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    VariantConfig,
    build_dataset_request_bank,
    build_variant_artifacts,
    ensure_exists,
    path_xy,
    prepare_variant_requests,
    run_variant_outputs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy prefill-KV seed 23 at latency-like cadence and stitch the "
            "published plan segments into one continuous executed path."
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
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "legacy_seed23_latency_rollout_20260612")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=DEFAULT_LEGACY_FM)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=900.0)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_pose_rows(dataset_root: Path) -> pd.DataFrame:
    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = request_bank.select_pose_gnss_rows(gnss)
    if gnss_valid.empty:
        raise RuntimeError(f"No valid GNSS rows in {dataset_root}")
    return gnss_valid.sort_values("timestamp_utc_ns").reset_index(drop=True)


def interp_world_xyz(
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    times_ns: np.ndarray,
) -> np.ndarray:
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)
    interp_lat = np.interp(times_ns, utc, lat)
    interp_lon = np.interp(times_ns, utc, lon)
    interp_alt = np.interp(times_ns, utc, alt)
    return request_bank.ecef_to_enu(
        request_bank.geodetic_to_ecef(interp_lat, interp_lon, interp_alt),
        *ref_lla,
    ).astype(np.float64)


def pose_at(
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    t_ns: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt_ns = int(round(dt_s * 1e9))
    pts = interp_world_xyz(
        gnss_valid,
        ref_lla,
        np.asarray([int(t_ns) - dt_ns, int(t_ns)], dtype=np.int64),
    )
    p0 = pts[-1, :2]
    delta = pts[-1, :2] - pts[0, :2]
    yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else 0.0
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return p0, rot


def local_to_world(local_xy: np.ndarray, p0: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return np.asarray(local_xy, dtype=np.float64) @ rot.T + p0[None, :]


def interp_plan(times: np.ndarray, points: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    q = np.asarray(query_times, dtype=np.float64)
    out = np.empty((len(q), 2), dtype=np.float64)
    out[:, 0] = np.interp(q, times, points[:, 0])
    out[:, 1] = np.interp(q, times, points[:, 1])
    return out


def stitch_latency_rollout(
    *,
    dataset_name: str,
    dataset_root: Path,
    request_summary: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    dt_s: float,
    out_root: Path,
) -> dict[str, Any]:
    rows = sorted(manifest_rows, key=lambda r: int(r["t0_utc_ns"]))
    gnss_valid = select_pose_rows(dataset_root)
    ref_lla = tuple(float(x) for x in request_summary["chunk_ref_lla"])

    stitched_world: list[np.ndarray] = []
    stitched_times_ns: list[np.ndarray] = []
    segment_records: list[dict[str, Any]] = []
    thin_plan_segments: list[np.ndarray] = []

    latencies_s: list[float] = []
    publish_times_ns: list[int] = []
    sample_times_ns = [int(row["t0_utc_ns"]) for row in rows]

    for row in rows:
        stem = Path(row["request_json"]).stem
        summary = artifacts[stem]["final"]
        timing = summary.get("timing", {})
        latency_s = float(timing.get("request_wall_ms", timing.get("total_post_vlm_ms", 0.0))) / 1000.0
        latencies_s.append(latency_s)
        publish_times_ns.append(int(row["t0_utc_ns"]) + int(round(latency_s * 1e9)))

    publish_intervals_s = np.diff(np.asarray(publish_times_ns, dtype=np.float64)) / 1e9
    fallback_interval_s = float(np.median(publish_intervals_s)) if publish_intervals_s.size else float(np.median(latencies_s))

    for idx, row in enumerate(rows):
        stem = Path(row["request_json"]).stem
        summary = artifacts[stem]["final"]
        points_local = path_xy(summary).astype(np.float64)
        plan_dt_s = float(summary.get("plan_dt_s", dt_s))
        plan_times = np.arange(len(points_local), dtype=np.float64) * plan_dt_s
        horizon_s = float(plan_times[-1])

        t0_ns = int(row["t0_utc_ns"])
        p0, rot = pose_at(gnss_valid, ref_lla, t0_ns, dt_s)
        full_world = local_to_world(points_local, p0, rot)
        thin_plan_segments.append(full_world)

        start_rel_s = min(max(latencies_s[idx], 0.0), horizon_s)
        if idx + 1 < len(rows):
            end_abs_ns = publish_times_ns[idx + 1]
            end_rel_s = (float(end_abs_ns - t0_ns) / 1e9)
        else:
            end_rel_s = start_rel_s + fallback_interval_s
        end_rel_s = min(max(end_rel_s, start_rel_s + plan_dt_s), horizon_s)

        inner = plan_times[(plan_times > start_rel_s) & (plan_times < end_rel_s)]
        query = np.concatenate(
            [
                np.asarray([start_rel_s], dtype=np.float64),
                inner,
                np.asarray([end_rel_s], dtype=np.float64),
            ]
        )
        local_segment = interp_plan(plan_times, points_local, query)
        world_segment = local_to_world(local_segment, p0, rot)
        abs_times = t0_ns + np.round(query * 1e9).astype(np.int64)

        if stitched_world and len(world_segment) > 1:
            world_segment = world_segment[1:]
            abs_times = abs_times[1:]
        stitched_world.append(world_segment)
        stitched_times_ns.append(abs_times)
        segment_records.append(
            {
                "index": int(idx),
                "sample_id": int(row["sample_id"]),
                "request_stem": stem,
                "t0_utc_ns": int(t0_ns),
                "publish_time_ns": int(publish_times_ns[idx]),
                "latency_s": float(latencies_s[idx]),
                "start_rel_s": float(start_rel_s),
                "end_rel_s": float(end_rel_s),
                "duration_s": float(end_rel_s - start_rel_s),
                "plan_horizon_s": horizon_s,
                "points": int(len(world_segment)),
            }
        )

    stitched = np.concatenate(stitched_world, axis=0)
    stitched_t = np.concatenate(stitched_times_ns, axis=0)
    gt = interp_world_xyz(gnss_valid, ref_lla, stitched_t)[:, :2]
    errors = np.linalg.norm(stitched - gt, axis=1)

    gt_dense_t = np.arange(stitched_t[0], stitched_t[-1] + int(round(dt_s * 1e9)), int(round(dt_s * 1e9)), dtype=np.int64)
    gt_dense = interp_world_xyz(gnss_valid, ref_lla, gt_dense_t)[:, :2]
    sample_pose = interp_world_xyz(gnss_valid, ref_lla, np.asarray(sample_times_ns, dtype=np.int64))[:, :2]
    publish_pose = interp_world_xyz(gnss_valid, ref_lla, np.asarray(publish_times_ns, dtype=np.int64))[:, :2]

    out_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_root / "stitched_latency_rollout.npz",
        stitched_world_xy=stitched,
        stitched_times_ns=stitched_t,
        gt_world_xy=gt,
        gt_dense_world_xy=gt_dense,
        gt_dense_times_ns=gt_dense_t,
        sample_world_xy=sample_pose,
        publish_world_xy=publish_pose,
        latencies_s=np.asarray(latencies_s, dtype=np.float64),
        publish_intervals_s=np.asarray(publish_intervals_s, dtype=np.float64),
        errors_m=errors,
    )

    summary = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "sample_count": int(len(rows)),
        "seed": int(segment_records[0].get("seed", 23)) if segment_records else 23,
        "mean_latency_ms": float(np.mean(latencies_s) * 1000.0),
        "p50_latency_ms": float(np.median(latencies_s) * 1000.0),
        "p90_latency_ms": float(np.quantile(latencies_s, 0.90) * 1000.0),
        "mean_publish_interval_s": float(np.mean(publish_intervals_s)) if publish_intervals_s.size else 0.0,
        "p50_publish_interval_s": float(np.median(publish_intervals_s)) if publish_intervals_s.size else 0.0,
        "stitched_points": int(len(stitched)),
        "time_start_ns": int(stitched_t[0]),
        "time_end_ns": int(stitched_t[-1]),
        "duration_s": float((stitched_t[-1] - stitched_t[0]) / 1e9),
        "ade_m": float(np.mean(errors)),
        "p50_error_m": float(np.median(errors)),
        "p90_error_m": float(np.quantile(errors, 0.90)),
        "max_error_m": float(np.max(errors)),
        "fde_m": float(errors[-1]),
        "segments": segment_records,
    }
    write_json(out_root / "stitched_latency_rollout_summary.json", summary)
    draw_rollout_png(
        out_path=out_root / f"{dataset_name}_seed23_latency_stitched_vs_gt.png",
        dataset_name=dataset_name,
        stitched=stitched,
        gt=gt,
        gt_dense=gt_dense,
        sample_pose=sample_pose,
        publish_pose=publish_pose,
        errors=errors,
        times_ns=stitched_t,
        latencies_s=np.asarray(latencies_s, dtype=np.float64),
        publish_intervals_s=np.asarray(publish_intervals_s, dtype=np.float64),
        thin_plan_segments=thin_plan_segments,
        summary=summary,
    )
    return summary


def draw_rollout_png(
    *,
    out_path: Path,
    dataset_name: str,
    stitched: np.ndarray,
    gt: np.ndarray,
    gt_dense: np.ndarray,
    sample_pose: np.ndarray,
    publish_pose: np.ndarray,
    errors: np.ndarray,
    times_ns: np.ndarray,
    latencies_s: np.ndarray,
    publish_intervals_s: np.ndarray,
    thin_plan_segments: list[np.ndarray],
    summary: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(15, 9), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0], width_ratios=[2.1, 1.0, 1.0], hspace=0.28, wspace=0.22)
    fig.patch.set_facecolor("#f8fafc")

    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("white")
    for seg in thin_plan_segments[:: max(1, len(thin_plan_segments) // 20)]:
        ax.plot(seg[:, 0], seg[:, 1], color="#93c5fd", linewidth=0.8, alpha=0.25)
    ax.plot(gt_dense[:, 0], gt_dense[:, 1], color="#475569", linewidth=2.0, linestyle="--", label="GNSS GT")
    ax.plot(stitched[:, 0], stitched[:, 1], color="#dc2626", linewidth=2.2, label="seed 23 latency-stitched")
    ax.scatter(sample_pose[:, 0], sample_pose[:, 1], s=10, color="#64748b", alpha=0.45, label="request t0")
    ax.scatter(publish_pose[:, 0], publish_pose[:, 1], s=12, color="#f59e0b", alpha=0.65, label="publish time")
    ax.scatter(stitched[0, 0], stitched[0, 1], s=42, color="#111827", zorder=5, label="start")
    ax.scatter(stitched[-1, 0], stitched[-1, 1], s=42, color="#dc2626", zorder=5, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("ENU east [m]")
    ax.set_ylabel("ENU north [m]")
    ax.set_title(f"{dataset_name}: stitched executed path vs GT")
    ax.legend(loc="best", fontsize=8)

    rel_t = (times_ns.astype(np.float64) - float(times_ns[0])) / 1e9
    ax_err = fig.add_subplot(gs[0, 1:])
    ax_err.set_facecolor("white")
    ax_err.plot(rel_t, errors, color="#dc2626", linewidth=1.8)
    ax_err.axhline(float(np.mean(errors)), color="#0f766e", linewidth=1.2, linestyle="--", label=f"mean {np.mean(errors):.2f}m")
    ax_err.axhline(float(np.quantile(errors, 0.90)), color="#7c3aed", linewidth=1.2, linestyle=":", label=f"p90 {np.quantile(errors, 0.90):.2f}m")
    ax_err.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax_err.set_xlabel("rollout time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.set_title("GT error over stitched rollout")
    ax_err.legend(fontsize=8)

    ax_lat = fig.add_subplot(gs[1, 1])
    ax_lat.set_facecolor("white")
    ax_lat.hist(latencies_s * 1000.0, bins=12, color="#2563eb", alpha=0.82)
    ax_lat.axvline(np.mean(latencies_s) * 1000.0, color="#111827", linewidth=1.2)
    ax_lat.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax_lat.set_xlabel("latency [ms]")
    ax_lat.set_ylabel("count")
    ax_lat.set_title("Plan compute delay")

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")
    pub_text = "n/a" if publish_intervals_s.size == 0 else f"{np.mean(publish_intervals_s):.2f}s mean"
    text = "\n".join(
        [
            f"samples: {summary['sample_count']}",
            f"duration: {summary['duration_s']:.1f}s",
            f"latency: {summary['mean_latency_ms']:.1f}ms mean / {summary['p90_latency_ms']:.1f}ms p90",
            f"publish interval: {pub_text}",
            f"ADE: {summary['ade_m']:.2f}m",
            f"p90 error: {summary['p90_error_m']:.2f}m",
            f"max error: {summary['max_error_m']:.2f}m",
            f"FDE: {summary['fde_m']:.2f}m",
        ]
    )
    ax_txt.text(
        0.02,
        0.95,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax_txt.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95, "pad": 8},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Legacy Seed 23 Latency Rollout",
        "",
        "이 결과는 legacy prefill-KV, FM seed 23, diffusion 2-step을 chunk 전체에서 약 1초 간격으로 실행한 뒤, "
        "각 plan의 `total_post_vlm_ms` 만큼 plan 앞부분을 건너뛰고 다음 plan publish 시점까지의 구간을 이어 붙인 것이다.",
        "",
        "즉 synchronous runtime에서 새 plan이 나올 때까지 직전 plan을 추종한다고 보는 latency-aware stitched rollout이다. "
        "실제 서비스가 frame drop, queueing, async pipeline을 쓰면 publish cadence는 달라질 수 있다.",
        "",
        "| dataset | samples | mean latency ms | publish interval s | ADE m | p90 error m | max error m | FDE m | PNG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        png = f"{item['dataset']}/{item['dataset']}_seed23_latency_stitched_vs_gt.png"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["dataset"]),
                    str(item["sample_count"]),
                    f"{item['mean_latency_ms']:.1f}",
                    f"{item['mean_publish_interval_s']:.2f}",
                    f"{item['ade_m']:.2f}",
                    f"{item['p90_error_m']:.2f}",
                    f"{item['max_error_m']:.2f}",
                    f"{item['fde_m']:.2f}",
                    png,
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path, desc in [
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "legacy engine dir"),
        (args.multimodal_engine_dir, "legacy multimodal engine dir"),
        (args.fm_engine, "legacy fm engine"),
    ]:
        ensure_exists(path, desc)

    variant = VariantConfig(
        name=f"legacy_prefill_seed_{args.seed}",
        label=f"Legacy prefill KV seed {args.seed}",
        color="#dc2626",
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        use_prefill_kv=True,
        diffusion_seed=int(args.seed),
        diffusion_num_steps=int(args.diffusion_num_steps),
        max_generate_length=int(args.max_generate_length),
    )

    top_summary: list[dict[str, Any]] = []
    for dataset_root in args.dataset_roots:
        dataset_root = dataset_root.resolve()
        ensure_exists(dataset_root, "dataset root")
        dataset_name = dataset_root.name
        dataset_work = args.work_root / dataset_name
        request_bank_root = dataset_work / "request_bank"
        print(f"[rollout] building request bank for {dataset_name}", flush=True)
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
        variant_root = dataset_work / "variant" / variant.name
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
            skip_existing=not args.rerun,
        )
        artifacts = build_variant_artifacts(
            variant=variant,
            output_paths=output_paths,
            request_paths=request_paths,
            manifest_rows=manifest_rows,
            dataset_root=dataset_root,
            history_len=args.history_len,
            artifact_root=variant_root / "artifacts",
        )
        summary = stitch_latency_rollout(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            request_summary=request_summary,
            manifest_rows=manifest_rows,
            artifacts=artifacts,
            dt_s=args.dt_s,
            out_root=dataset_work,
        )
        top_summary.append(summary)

    write_json(args.work_root / "summary.json", {"datasets": top_summary})
    write_report(args.work_root / "latency_rollout_report.md", top_summary)
    print(json.dumps({"work_root": str(args.work_root), "datasets": top_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
