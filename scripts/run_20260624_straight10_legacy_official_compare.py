#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_live_chunk_request_bank as request_bank
from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    VariantConfig,
    build_dataset_request_bank,
    build_variant_artifacts,
    path_xy,
    prepare_variant_requests,
    run_variant_outputs,
    write_json,
)


@dataclass(frozen=True)
class SelectedSample:
    sample_id: int
    offset_s: float
    score: float
    gt_y4: float
    gt_y8: float
    gt_end_y: float
    gt_len_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick straight-scene legacy seed23 vs official 10B compare on 2026-06-24-test1."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "2026-06-24-test1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "straight10_20260624_test1_legacy_official")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--min-sample-gap", type=int, default=80)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--future-len", type=int, default=64)
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
    parser.add_argument("--official-model-dir", type=Path, default=Path("/workspace/alpamayo1.5/model"))
    parser.add_argument("--official-diffusion-steps", type=int, default=10)
    parser.add_argument("--official-seed", type=int, default=42)
    parser.add_argument("--official-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run-official", action="store_true")
    parser.add_argument("--no-run-legacy", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def points_from_pred_xyz(pred_xyz: Any) -> np.ndarray:
    pred = np.asarray(pred_xyz, dtype=np.float64)
    pts = np.zeros((pred.shape[0] + 1, 2), dtype=np.float64)
    pts[1:, 0] = pred[:, 0]
    pts[1:, 1] = pred[:, 1]
    return pts


def point_at_arc(points: np.ndarray, arc_m: float) -> np.ndarray | None:
    if len(points) < 2:
        return None
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < arc_m:
        return None
    idx = int(np.searchsorted(cum, arc_m, side="right") - 1)
    idx = min(max(idx, 0), len(seg) - 1)
    ratio = (arc_m - float(cum[idx])) / max(float(seg[idx]), 1e-9)
    return points[idx] + ratio * (points[idx + 1] - points[idx])


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def interpolate_world(gnss_valid: pd.DataFrame, ref_lla: tuple[float, float, float], times_ns: np.ndarray) -> np.ndarray:
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


def gt_future_local(
    *,
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    t0_ns: int,
    dt_s: float,
    future_len: int,
) -> np.ndarray:
    dt_ns = int(round(dt_s * 1e9))
    times = np.asarray([t0_ns - dt_ns] + [t0_ns + (i + 1) * dt_ns for i in range(future_len)], dtype=np.int64)
    world = interpolate_world(gnss_valid, ref_lla, times)[:, :2]
    p_prev = world[0]
    p0 = interpolate_world(gnss_valid, ref_lla, np.asarray([t0_ns], dtype=np.int64))[0, :2]
    delta = p0 - p_prev
    yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else 0.0
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    future = world[1:]
    pts = np.zeros((future.shape[0] + 1, 2), dtype=np.float64)
    pts[1:] = (future - p0[None, :]) @ rot
    return pts


def select_straight_samples(args: argparse.Namespace) -> list[SelectedSample]:
    sample_index = pd.read_parquet(args.dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(args.dataset_root / "sensors" / "camera_front" / "frames.parquet")[
        ["frame_id", "chunk_id"]
    ]
    sample_index = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    rows = (
        sample_index[sample_index["chunk_id"] == args.chunk_id]
        .copy()
        .sort_values("t0_utc_ns")
        .reset_index(drop=True)
    )
    gnss = pd.read_parquet(args.dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = request_bank.select_pose_gnss_rows(gnss).sort_values("timestamp_utc_ns").reset_index(drop=True)
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    ref_t = int(rows["t0_utc_ns"].iloc[0])
    ref_lla = (
        float(np.interp([ref_t], utc, gnss_valid["lat"].to_numpy(dtype=np.float64))[0]),
        float(np.interp([ref_t], utc, gnss_valid["lon"].to_numpy(dtype=np.float64))[0]),
        float(np.interp([ref_t], utc, gnss_valid["alt"].to_numpy(dtype=np.float64))[0]),
    )
    dt_ns = int(round(args.dt_s * 1e9))
    min_t = int(utc[0]) + (args.history_len + 1) * dt_ns
    max_t = int(utc[-1]) - (args.future_len + 1) * dt_ns
    candidates: list[SelectedSample] = []
    chunk_start = int(rows["t0_utc_ns"].iloc[0])
    for row in rows.itertuples(index=False):
        t0_ns = int(row.t0_utc_ns)
        if t0_ns < min_t or t0_ns > max_t:
            continue
        pts = gt_future_local(
            gnss_valid=gnss_valid,
            ref_lla=ref_lla,
            t0_ns=t0_ns,
            dt_s=args.dt_s,
            future_len=args.future_len,
        )
        length = path_length(pts)
        if length < 8.5:
            continue
        p4 = point_at_arc(pts, 4.0)
        p8 = point_at_arc(pts, 8.0)
        if p4 is None or p8 is None:
            continue
        # Small near/mid-horizon lateral movement and adequate length indicate straight driving.
        end_idx = min(len(pts) - 1, 40)
        end_y = float(pts[end_idx, 1])
        score = abs(float(p8[1])) + 0.5 * abs(float(p4[1])) + 0.15 * abs(end_y)
        candidates.append(
            SelectedSample(
                sample_id=int(row.sample_id),
                offset_s=float(t0_ns - chunk_start) / 1e9,
                score=float(score),
                gt_y4=float(p4[1]),
                gt_y8=float(p8[1]),
                gt_end_y=end_y,
                gt_len_m=length,
            )
        )
    candidates.sort(key=lambda item: item.score)
    selected: list[SelectedSample] = []
    min_gap = int(args.min_sample_gap)
    while len(selected) < args.sample_count and min_gap >= 0:
        selected.clear()
        for item in candidates:
            if all(abs(item.sample_id - other.sample_id) >= min_gap for other in selected):
                selected.append(item)
                if len(selected) >= args.sample_count:
                    break
        if len(selected) >= args.sample_count:
            break
        min_gap = int(min_gap * 0.7)
    selected = sorted(selected[: args.sample_count], key=lambda item: item.sample_id)
    if len(selected) < args.sample_count:
        raise RuntimeError(f"Only selected {len(selected)} straight samples")
    return selected


def run_official(args: argparse.Namespace, sample_ids: list[int], request_bank_root: Path, official_root: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_official_10b_hf_request_bank.py"),
        "--model-dir",
        str(args.official_model_dir),
        "--request-bank-root",
        str(request_bank_root),
        "--output-root",
        str(official_root),
        "--sample-ids",
        *[str(x) for x in sample_ids],
        "--mode",
        "ae",
        "--diffusion-steps",
        str(args.official_diffusion_steps),
        "--seed",
        str(args.official_seed),
        "--dtype",
        args.official_dtype,
        "--attn-implementation",
        "sdpa",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    print("[straight10] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_official_points(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("label") != "official_10b_ae":
                continue
            out[int(item["sample_id"])] = points_from_pred_xyz(item["pred_xyz"])
    return out


def finite_metric(points: np.ndarray | None, arc_m: float) -> float | None:
    if points is None:
        return None
    p = point_at_arc(points, arc_m)
    return None if p is None else float(p[1])


def draw_panels(
    *,
    out_path: Path,
    manifest_rows: list[dict[str, Any]],
    selected: list[SelectedSample],
    legacy_points: dict[int, np.ndarray],
    official_points: dict[int, np.ndarray],
    gt_points: dict[int, np.ndarray],
) -> None:
    rows_by_id = {int(row["sample_id"]): row for row in manifest_rows}
    fig, axes = plt.subplots(5, 4, figsize=(18, 22), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    colors = {
        "gt": "#64748b",
        "legacy": "#111827",
        "official": "#ea580c",
    }
    for idx, item in enumerate(selected):
        r = idx // 2
        c = (idx % 2) * 2
        sid = item.sample_id
        row = rows_by_id[sid]
        image_path = Path(row["image_dir"]) / "cam1_f3.png"
        if not image_path.exists():
            image_path = Path(row["image_dir"]) / "cam1_f0.png"
        ax_img = axes[r, c]
        ax_img.imshow(Image.open(image_path).convert("RGB"))
        ax_img.set_title(f"sid {sid} / {item.offset_s:.1f}s / front", fontsize=9, fontweight="bold")
        ax_img.axis("off")

        ax = axes[r, c + 1]
        ax.set_facecolor("white")
        series = [
            ("GT", gt_points.get(sid), colors["gt"], "--", 2.0),
            ("Legacy seed23", legacy_points.get(sid), colors["legacy"], "-", 1.8),
            ("Official 10B", official_points.get(sid), colors["official"], "-", 1.8),
        ]
        for label, pts, color, linestyle, linewidth in series:
            if pts is None:
                continue
            ax.plot(pts[:, 1], pts[:, 0], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            p4 = point_at_arc(pts, 4.0)
            p8 = point_at_arc(pts, 8.0)
            if p4 is not None:
                ax.scatter([p4[1]], [p4[0]], color=color, s=16, marker="o")
            if p8 is not None:
                ax.scatter([p8[1]], [p8[0]], color=color, s=18, marker="^")
        ax.scatter([0.0], [0.0], marker="x", color="#020617", s=26)
        ax.axvline(0.0, color="#0f172a", linewidth=0.8, alpha=0.75)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(4.0))
        ax.grid(True, axis="x", which="major", color="#94a3b8", linewidth=0.7, alpha=0.75)
        ax.grid(True, axis="x", which="minor", color="#cbd5e1", linewidth=0.5, alpha=0.75)
        ax.grid(True, axis="y", which="major", color="#cbd5e1", linewidth=0.6, alpha=0.7)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.25, 24.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("local y [m], minor grid = 0.1m")
        ax.set_ylabel("local x [m]")
        ax.set_title(f"GT straight score {item.score:.3f}", fontsize=9)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("2026-06-24-test1 straight 10: current legacy seed23 vs official 10B vs GT", fontsize=15, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_metrics(out_path: Path, metrics: list[dict[str, Any]]) -> None:
    sample_ids = sorted({int(row["sample_id"]) for row in metrics})
    labels = ["gt", "legacy_seed23", "official_10b"]
    colors = {"gt": "#64748b", "legacy_seed23": "#111827", "official_10b": "#ea580c"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), dpi=170, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, key, title in zip(axes, ("y_at_4m", "y_at_8m"), ("local y at arc 4m", "local y at arc 8m")):
        ax.set_facecolor("white")
        x = np.arange(len(sample_ids))
        width = 0.24
        for j, label in enumerate(labels):
            vals: list[float] = []
            for sid in sample_ids:
                match = next((row for row in metrics if row["sample_id"] == sid and row["model"] == label), None)
                val = float("nan") if match is None or match[key] is None else float(match[key])
                vals.append(val)
            ax.bar(x + (j - 1) * width, vals, width=width, label=label, color=colors[label], alpha=0.82)
        ax.axhline(0.0, color="#0f172a", linewidth=0.8)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.75)
        ax.set_ylabel("local y [m]")
        ax.set_title(title)
    axes[-1].set_xticks(np.arange(len(sample_ids)))
    axes[-1].set_xticklabels([str(sid) for sid in sample_ids], rotation=45, ha="right")
    axes[0].legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selected = select_straight_samples(args)
    sample_ids = [item.sample_id for item in selected]
    run_root = args.output_root / args.dataset_root.name
    request_bank_root = run_root / "request_bank"
    legacy_root = run_root / "legacy_seed23"
    official_root = run_root / "official_10b"
    report_root = run_root / "report"

    manifest_rows, request_summary = build_dataset_request_bank(
        dataset_root=args.dataset_root,
        out_root=request_bank_root,
        chunk_id=args.chunk_id,
        sample_count=len(sample_ids),
        sample_ids=sample_ids,
        width=args.width,
        height=args.height,
        history_len=args.history_len,
        dt_s=args.dt_s,
        traj_token_offset=args.traj_token_offset,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    legacy_variant = VariantConfig(
        name="legacy_seed23",
        label="Legacy seed23 AE",
        color="#111827",
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        use_prefill_kv=True,
        diffusion_seed=args.seed,
        diffusion_num_steps=args.diffusion_num_steps,
        max_generate_length=args.max_generate_length,
    )
    request_paths = prepare_variant_requests(
        manifest_rows=manifest_rows,
        variant=legacy_variant,
        out_root=legacy_root / "requests",
    )
    output_paths = [legacy_root / "outputs" / p.name.replace("request_", "output_") for p in request_paths]
    if not args.no_run_legacy:
        output_paths = run_variant_outputs(
            variant=legacy_variant,
            request_paths=request_paths,
            output_root=legacy_root / "outputs",
            llm_inference_bin=args.llm_inference_bin,
            plugin_lib=args.plugin_lib,
            warmup=args.warmup,
            timeout_per_request=args.timeout_per_request,
            skip_existing=args.skip_existing,
        )
    artifacts = build_variant_artifacts(
        variant=legacy_variant,
        output_paths=output_paths,
        request_paths=request_paths,
        manifest_rows=manifest_rows,
        dataset_root=args.dataset_root,
        history_len=args.history_len,
        artifact_root=legacy_root / "artifacts",
    )
    if not args.no_run_official:
        run_official(args, sample_ids, request_bank_root, official_root)
    official_points = load_official_points(official_root / "predictions.jsonl")

    legacy_points: dict[int, np.ndarray] = {}
    gt_points: dict[int, np.ndarray] = {}
    rows_by_stem = {Path(row["request_json"]).stem: row for row in manifest_rows}
    for stem, item in artifacts.items():
        sid = int(rows_by_stem[stem]["sample_id"])
        legacy_points[sid] = path_xy(item["ac_decoded"]).astype(np.float64)
        gt_points[sid] = path_xy(item["gt"]).astype(np.float64)

    metrics: list[dict[str, Any]] = []
    selected_by_id = {item.sample_id: item for item in selected}
    for sid in sample_ids:
        for model_name, pts in (
            ("gt", gt_points.get(sid)),
            ("legacy_seed23", legacy_points.get(sid)),
            ("official_10b", official_points.get(sid)),
        ):
            if pts is None:
                continue
            metrics.append(
                {
                    "sample_id": sid,
                    "offset_s": selected_by_id[sid].offset_s,
                    "model": model_name,
                    "y_at_4m": finite_metric(pts, 4.0),
                    "y_at_8m": finite_metric(pts, 8.0),
                    "end_y_m": float(pts[-1, 1]),
                    "path_len_m": path_length(pts),
                }
            )

    panel_png = report_root / "straight10_legacy_official10b_gt_panels.png"
    metrics_png = report_root / "straight10_lateral_metrics.png"
    draw_panels(
        out_path=panel_png,
        manifest_rows=manifest_rows,
        selected=selected,
        legacy_points=legacy_points,
        official_points=official_points,
        gt_points=gt_points,
    )
    draw_metrics(metrics_png, metrics)
    summary = {
        "dataset_root": str(args.dataset_root),
        "sample_ids": sample_ids,
        "selected_samples": [item.__dict__ for item in selected],
        "request_summary": request_summary,
        "outputs": {
            "panel_png": str(panel_png),
            "metrics_png": str(metrics_png),
            "summary_json": str(report_root / "straight10_summary.json"),
            "report_md": str(report_root / "straight10_report.md"),
        },
        "metrics": metrics,
    }
    write_json(report_root / "straight10_summary.json", summary)
    lines = [
        "# 2026-06-24 Test1 Straight 10 Compare",
        "",
        f"- panel: `{panel_png}`",
        f"- metrics: `{metrics_png}`",
        "",
        "## Selected Samples",
        "",
        "| sample | offset_s | straight_score | gt y@4 | gt y@8 | gt end_y | gt len |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in selected:
        lines.append(
            f"| {item.sample_id} | {item.offset_s:.1f} | {item.score:.4f} | "
            f"{item.gt_y4:.3f} | {item.gt_y8:.3f} | {item.gt_end_y:.3f} | {item.gt_len_m:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Model Metrics",
            "",
            "| sample | offset_s | model | y@4 | y@8 | end_y | path_len |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics:
        def fmt(value: Any) -> str:
            return "nan" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {row['sample_id']} | {row['offset_s']:.1f} | {row['model']} | "
            f"{fmt(row['y_at_4m'])} | {fmt(row['y_at_8m'])} | "
            f"{fmt(row['end_y_m'])} | {fmt(row['path_len_m'])} |"
        )
    (report_root / "straight10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
