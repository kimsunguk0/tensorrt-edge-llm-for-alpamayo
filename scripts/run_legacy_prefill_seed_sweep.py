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

from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    OverlayVariant,
    VariantConfig,
    build_variant_artifacts,
    ensure_exists,
    make_contact_sheet,
    path_length,
    path_xy,
    prepare_variant_requests,
    run_variant_outputs,
    write_json,
)


DEFAULT_SEEDS = [42, 2, 0, 1, 7, 11, 17, 23, 99, 1234]
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#f59e0b",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#9333ea",
    "#0f766e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep FM diffusion seeds for the legacy Alpamayo prefill-KV runtime, "
            "score against GNSS GT with driving-shape penalties, and render overlays."
        )
    )
    parser.add_argument(
        "--source-work-root",
        type=Path,
        default=REPO_ROOT / "output" / "flex_legacy_threeway_20260612",
        help="Existing work root that contains per-dataset request_bank/manifest.json.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=["2026-06-12-test1", "2026-06-12-test2"],
        help="Dataset subdirectories under --source-work-root to reuse.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "output" / "legacy_prefill_seed_sweep_20260612",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--sample-limit-per-dataset", type=int, default=20)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--legacy-steps", type=int, default=2)
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
    parser.add_argument("--rerun", action="store_true", help="Ignore existing output JSON files.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(source_work_root: Path, dataset_name: str, limit: int) -> list[dict[str, Any]]:
    manifest_path = source_work_root / dataset_name / "request_bank" / "manifest.json"
    ensure_exists(manifest_path, "request bank manifest")
    rows = load_json(manifest_path)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Invalid or empty manifest: {manifest_path}")
    if limit > 0:
        rows = rows[:limit]
    return rows


def history_speed_mps(row: dict[str, Any], dt_s: float) -> float:
    hist = np.load(row["ego_history_xyz_npy"]).astype(np.float32)
    hist = hist.reshape(-1, hist.shape[-1])
    if len(hist) < 2:
        return 0.0
    tail = hist[-min(len(hist), 5) :, :2]
    speeds = np.linalg.norm(np.diff(tail, axis=0), axis=1) / max(float(dt_s), 1e-6)
    if len(speeds) == 0:
        return 0.0
    return float(np.median(speeds))


def aligned_xy(summary: dict[str, Any], gt_summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pred = path_xy(summary).astype(np.float64)
    gt = path_xy(gt_summary).astype(np.float64)
    n = min(len(pred), len(gt))
    if n < 2:
        raise RuntimeError("Need at least origin + one future point for scoring")
    return pred[:n], gt[:n]


def score_path(summary: dict[str, Any], gt_summary: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    pred, gt = aligned_xy(summary, gt_summary)
    dt_s = float(summary.get("plan_dt_s", 0.1))
    err = np.linalg.norm(pred[1:] - gt[1:], axis=1)
    ade = float(np.mean(err))
    fde = float(err[-1])

    progress_error = float(abs(pred[-1, 0] - gt[-1, 0]))
    lateral_fde = float(abs(pred[-1, 1] - gt[-1, 1]))
    reverse_m = float(np.maximum(0.0, -np.diff(pred[:, 0])).sum())
    gt_progress = float(gt[-1, 0])
    pred_progress = float(pred[-1, 0])
    under_progress = float(max(0.0, gt_progress - pred_progress - 1.0))
    over_progress = float(max(0.0, pred_progress - gt_progress - 2.0))
    lateral_margin = float(max(0.0, np.max(np.abs(pred[:, 1])) - np.max(np.abs(gt[:, 1])) - 1.5))

    speeds = np.asarray(summary.get("pred_v_mps") or [], dtype=np.float64)
    if speeds.size == 0:
        speeds = np.linalg.norm(np.diff(pred, axis=0), axis=1) / max(dt_s, 1e-6)
    hist_speed = history_speed_mps(row, dt_s)
    initial_speed_jump = float(abs(float(speeds[0]) - hist_speed)) if speeds.size else 0.0
    accel = np.diff(np.concatenate([[hist_speed], speeds])) / max(dt_s, 1e-6)
    mean_abs_accel = float(np.mean(np.abs(accel))) if accel.size else 0.0
    max_abs_accel = float(np.max(np.abs(accel))) if accel.size else 0.0
    jerk = np.diff(accel) / max(dt_s, 1e-6)
    rms_jerk = float(np.sqrt(np.mean(np.square(np.clip(jerk, -80.0, 80.0))))) if jerk.size else 0.0
    hard_accel_excess = float(max(0.0, max_abs_accel - 4.0))

    curvature = np.asarray(summary.get("pred_curvature") or [], dtype=np.float64)
    max_abs_curvature = float(np.max(np.abs(curvature))) if curvature.size else 0.0
    mean_abs_curvature = float(np.mean(np.abs(curvature))) if curvature.size else 0.0
    curv_excess = float(max(0.0, max_abs_curvature - 0.25))
    if curvature.size > 1:
        curv_rate_rms = float(np.sqrt(np.mean(np.square(np.diff(curvature) / max(dt_s, 1e-6)))))
    else:
        curv_rate_rms = 0.0

    tracking_cost = ade + 0.35 * fde + 0.20 * progress_error + 0.15 * lateral_fde
    comfort_cost = 0.25 * initial_speed_jump + 0.08 * mean_abs_accel + 0.01 * rms_jerk
    shape_cost = (
        3.0 * reverse_m
        + 0.70 * under_progress
        + 0.45 * over_progress
        + 1.50 * curv_excess
        + 0.80 * lateral_margin
        + 0.20 * hard_accel_excess
    )
    drive_cost = float(tracking_cost + comfort_cost + shape_cost)
    drive_score = float(100.0 / (1.0 + drive_cost))
    bad_case = float(fde > 5.0 or reverse_m > 0.5 or lateral_margin > 2.0 or max_abs_curvature > 0.40)

    return {
        "ade_m": ade,
        "fde_m": fde,
        "progress_error_m": progress_error,
        "lateral_fde_m": lateral_fde,
        "reverse_m": reverse_m,
        "under_progress_m": under_progress,
        "over_progress_m": over_progress,
        "lateral_margin_m": lateral_margin,
        "hist_speed_mps": hist_speed,
        "initial_speed_jump_mps": initial_speed_jump,
        "mean_abs_accel_mps2": mean_abs_accel,
        "max_abs_accel_mps2": max_abs_accel,
        "rms_jerk_mps3": rms_jerk,
        "mean_abs_curvature_1pm": mean_abs_curvature,
        "max_abs_curvature_1pm": max_abs_curvature,
        "curvature_rate_rms_1pms": curv_rate_rms,
        "path_len_m": path_length(pred),
        "gt_path_len_m": path_length(gt),
        "drive_cost": drive_cost,
        "drive_score": drive_score,
        "bad_case": bad_case,
    }


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    best_cost_idx = metrics_df.groupby(["dataset", "request_stem"])["drive_cost"].idxmin()
    best_ade_idx = metrics_df.groupby(["dataset", "request_stem"])["ade_m"].idxmin()
    drive_wins = metrics_df.loc[best_cost_idx].groupby("seed").size().to_dict()
    ade_wins = metrics_df.loc[best_ade_idx].groupby("seed").size().to_dict()
    rows: list[dict[str, Any]] = []
    for seed, group in metrics_df.groupby("seed", sort=False):
        mean_cost = float(group["drive_cost"].mean())
        p90_cost = float(group["drive_cost"].quantile(0.90))
        bad_rate = float(group["bad_case"].mean())
        robust_cost = mean_cost + 0.25 * p90_cost + 2.0 * bad_rate
        row = {
            "seed": int(seed),
            "samples": int(len(group)),
            "mean_drive_cost": mean_cost,
            "p90_drive_cost": p90_cost,
            "bad_case_rate": bad_rate,
            "robust_drive_cost": robust_cost,
            "robust_drive_score": float(100.0 / (1.0 + robust_cost)),
            "mean_ade_m": float(group["ade_m"].mean()),
            "mean_fde_m": float(group["fde_m"].mean()),
            "p90_fde_m": float(group["fde_m"].quantile(0.90)),
            "mean_progress_error_m": float(group["progress_error_m"].mean()),
            "mean_lateral_fde_m": float(group["lateral_fde_m"].mean()),
            "mean_reverse_m": float(group["reverse_m"].mean()),
            "mean_initial_speed_jump_mps": float(group["initial_speed_jump_mps"].mean()),
            "mean_abs_accel_mps2": float(group["mean_abs_accel_mps2"].mean()),
            "mean_rms_jerk_mps3": float(group["rms_jerk_mps3"].mean()),
            "mean_max_abs_curvature_1pm": float(group["max_abs_curvature_1pm"].mean()),
            "wins_drive_cost": int(drive_wins.get(seed, 0)),
            "wins_ade": int(ade_wins.get(seed, 0)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["robust_drive_cost", "mean_drive_cost"]).reset_index(drop=True)


def draw_seed_overlay(
    *,
    out_path: Path,
    dataset_name: str,
    row: dict[str, Any],
    variants: list[OverlayVariant],
    artifacts_by_seed: dict[str, dict[str, dict[str, Any]]],
    metrics_by_seed: dict[str, dict[str, float]],
) -> None:
    fig = plt.figure(figsize=(13.5, 10.8), dpi=150)
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

    stem = Path(row["request_json"]).stem
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    lines = [
        f"{dataset_name} | sample {row['sample_id']} | t0_us {row['t0_us']}",
        "score: lower drive_cost is better; score also penalizes progress, speed discontinuity, jerk, curvature, reverse motion.",
    ]
    for variant in variants:
        metrics = metrics_by_seed[variant.name]
        final_summary = artifacts_by_seed[variant.name][stem]["final"]
        pts = path_xy(final_summary)
        timing = final_summary.get("timing", {})
        total_ms = timing.get("total_post_vlm_ms")
        total_text = "n/a" if total_ms is None else f"{float(total_ms):.1f}ms"
        lines.append(
            f"{variant.label}: cost={metrics['drive_cost']:.2f} ADE={metrics['ade_m']:.2f} "
            f"FDE={metrics['fde_m']:.2f} end=({pts[-1,0]:.2f},{pts[-1,1]:.2f}) total={total_text}"
        )
    ax_text.text(
        0.01,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        transform=ax_text.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95, "pad": 6},
    )

    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor("#ffffff")
    hist_xyz = np.load(row["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    ax.plot(hist_xyz[:, 1], hist_xyz[:, 0], color="#0f172a", linewidth=1.6, alpha=0.72, label="ego history")

    gt_summary = next(iter(artifacts_by_seed.values()))[stem]["gt"]
    gt_pts = path_xy(gt_summary)
    ax.plot(gt_pts[:, 1], gt_pts[:, 0], color="#64748b", linewidth=2.0, alpha=0.72, linestyle="--", label="GNSS GT")

    all_pts = [hist_xyz[:, :2], gt_pts]
    for variant in variants:
        summary = artifacts_by_seed[variant.name][stem]["final"]
        pts = path_xy(summary)
        all_pts.append(pts)
        ax.plot(pts[:, 1], pts[:, 0], color=variant.color, linewidth=2.4, alpha=0.92, label=variant.label)
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8, frameon=True)
    ax.set_title("Legacy prefill-KV FM seed overlay")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(path: Path, summary_df: pd.DataFrame, best_seed: int, selected_seeds: list[int]) -> None:
    top = summary_df.head(10).copy()
    cols = [
        "seed",
        "robust_drive_cost",
        "robust_drive_score",
        "mean_ade_m",
        "mean_fde_m",
        "p90_fde_m",
        "bad_case_rate",
        "wins_drive_cost",
    ]
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    table_lines = [header, divider]
    for _, row in top[cols].iterrows():
        values: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        "# Legacy Prefill-KV FM Seed Sweep",
        "",
        f"Recommended fixed seed: **{best_seed}**",
        "",
        "Seed 변경은 inference-time FM 초기 노이즈만 바꾸는 것이므로 재학습이 필요 없다. "
        "단, 이 결과는 현재 legacy prefill-KV 엔진/AE/FM/데이터 분포 기준의 calibration이다.",
        "",
        "## Scoring",
        "",
        "Offline 기준에서는 GT를 쓰되 ADE만 최적화하지 않았다. `drive_cost`는 ADE/FDE에 더해 "
        "종단 진행거리 오차, lateral 종단 오차, history 대비 초기 속도 점프, 평균 가속, 저크, "
        "후진성, 과소/과대 진행, lateral drift, 과한 곡률을 같이 패널티로 준다. "
        "`robust_drive_cost = mean + 0.25 * p90 + 2.0 * bad_case_rate`라서 평균만 좋은 seed보다 "
        "큰 실패가 적은 seed를 우선한다.",
        "",
        "## Ranking",
        "",
        "\n".join(table_lines),
        "",
        "## Overlay Seeds",
        "",
        "시각화에는 추천 seed, 기존 기준 seed 42, 기존 command seed 2, 그리고 ranking runner-up들을 중복 없이 포함했다.",
        "",
        ", ".join(str(x) for x in selected_seeds),
        "",
    ]
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

    args.work_root.mkdir(parents=True, exist_ok=True)
    variants = [
        VariantConfig(
            name=f"seed_{seed}",
            label=f"seed {seed}",
            color=COLORS[idx % len(COLORS)],
            engine_dir=args.engine_dir,
            multimodal_engine_dir=args.multimodal_engine_dir,
            fm_engine=args.fm_engine,
            use_prefill_kv=True,
            diffusion_seed=int(seed),
            diffusion_num_steps=int(args.legacy_steps),
            max_generate_length=int(args.max_generate_length),
        )
        for idx, seed in enumerate(args.seeds)
    ]

    all_metrics: list[dict[str, Any]] = []
    all_dataset_artifacts: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    all_dataset_rows: dict[str, list[dict[str, Any]]] = {}

    for dataset_name in args.dataset_names:
        manifest_rows = load_manifest(args.source_work_root, dataset_name, args.sample_limit_per_dataset)
        dataset_root = Path(manifest_rows[0]["dataset_root"])
        ensure_exists(dataset_root, "dataset root")
        dataset_work = args.work_root / dataset_name
        all_dataset_rows[dataset_name] = manifest_rows
        all_dataset_artifacts[dataset_name] = {}

        for variant in variants:
            variant_root = dataset_work / "seeds" / variant.name
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
            all_dataset_artifacts[dataset_name][variant.name] = artifacts

        for row in manifest_rows:
            stem = Path(row["request_json"]).stem
            for variant in variants:
                item = all_dataset_artifacts[dataset_name][variant.name][stem]
                metrics = score_path(item["final"], item["gt"], row)
                all_metrics.append(
                    {
                        "dataset": dataset_name,
                        "request_stem": stem,
                        "sample_id": int(row["sample_id"]),
                        "seed": int(variant.diffusion_seed),
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(all_metrics)
    summary_df = aggregate_metrics(metrics_df)
    best_seed = int(summary_df.iloc[0]["seed"])
    selected_seeds: list[int] = []
    for seed in [best_seed, 42, 2] + [int(x) for x in summary_df["seed"].head(4).tolist()]:
        if seed in args.seeds and seed not in selected_seeds:
            selected_seeds.append(seed)
    selected_variants = [
        OverlayVariant(name=f"seed_{seed}", label=f"seed {seed}", color=COLORS[args.seeds.index(seed) % len(COLORS)])
        for seed in selected_seeds
    ]

    metrics_csv = args.work_root / "per_sample_metrics.csv"
    summary_csv = args.work_root / "seed_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    write_json(
        args.work_root / "summary.json",
        {
            "best_seed": best_seed,
            "selected_overlay_seeds": selected_seeds,
            "seeds": [int(x) for x in args.seeds],
            "legacy_steps": int(args.legacy_steps),
            "source_work_root": str(args.source_work_root),
            "work_root": str(args.work_root),
            "metrics_csv": str(metrics_csv),
            "seed_summary_csv": str(summary_csv),
        },
    )
    write_markdown_report(args.work_root / "seed_sweep_report.md", summary_df, best_seed, selected_seeds)

    for dataset_name, manifest_rows in all_dataset_rows.items():
        overlay_paths: list[Path] = []
        overlay_dir = args.work_root / dataset_name / "overlays_selected"
        for row in manifest_rows:
            stem = Path(row["request_json"]).stem
            sample_metrics = {
                f"seed_{int(m['seed'])}": m
                for m in all_metrics
                if m["dataset"] == dataset_name and m["request_stem"] == stem and int(m["seed"]) in selected_seeds
            }
            out_png = overlay_dir / f"overlay_{dataset_name}_{stem.removeprefix('request_')}.png"
            draw_seed_overlay(
                out_path=out_png,
                dataset_name=dataset_name,
                row=row,
                variants=selected_variants,
                artifacts_by_seed=all_dataset_artifacts[dataset_name],
                metrics_by_seed=sample_metrics,
            )
            overlay_paths.append(out_png)
        contact_sheet = make_contact_sheet(
            overlay_paths,
            args.work_root / dataset_name / f"contact_sheet_{dataset_name}_selected_seed_overlay.png",
            cols=4,
        )
        print(f"[seed-sweep] {dataset_name} contact sheet: {contact_sheet}", flush=True)

    print(summary_df.head(10).to_string(index=False))
    print(f"[seed-sweep] recommended seed: {best_seed}")
    print(f"[seed-sweep] report: {args.work_root / 'seed_sweep_report.md'}")


if __name__ == "__main__":
    main()
