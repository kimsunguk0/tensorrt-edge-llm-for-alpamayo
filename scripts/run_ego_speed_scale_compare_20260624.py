#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    VariantConfig,
    build_dataset_request_bank,
    build_variant_artifacts,
    path_xy,
    path_length,
    run_variant_outputs,
    write_json,
)


@dataclass(frozen=True)
class SpeedVariant:
    name: str
    label: str
    scale: float
    color: str
    linestyle: str = "-"


VARIANTS = [
    SpeedVariant("actual_10kmh", "actual ego (~10km/h)", 1.0, "#111827", "-"),
    SpeedVariant("fake_1kmh", "fake 1km/h", 0.1, "#7c3aed", "-"),
    SpeedVariant("fake_5kmh", "fake 5km/h", 0.5, "#2563eb", "-"),
    SpeedVariant("fake_20kmh", "fake 20km/h", 2.0, "#16a34a", "-"),
    SpeedVariant("fake_50kmh", "fake 50km/h", 5.0, "#f97316", "-"),
    SpeedVariant("fake_100kmh", "fake 100km/h", 10.0, "#dc2626", "-"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale ego_history_xyz for fake speed sensitivity tests and compare current "
            "legacy seed23 paths on 2026-06-24-test1."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "2026-06-24-test1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "ego_speed_scale_20260624_test1")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument(
        "--selection-bank-count",
        type=int,
        default=None,
        help="Number of candidate samples to build before optional speed filtering.",
    )
    parser.add_argument(
        "--min-history-speed-kmh",
        type=float,
        default=None,
        help="If set, keep only candidate rows whose last ego-history speed is at least this value.",
    )
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
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def finite_stats(values: list[float]) -> dict[str, float | int | None]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "mean_abs": None, "p90_abs": None, "max_abs": None}
    abs_vals = np.abs(vals)
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "mean_abs": float(abs_vals.mean()),
        "p90_abs": float(np.quantile(abs_vals, 0.9)),
        "max_abs": float(abs_vals.max()),
    }


def select_evenly_spaced_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(rows) <= count:
        return list(rows)
    idxs = np.linspace(0, len(rows) - 1, count).round().astype(int)
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for idx in idxs:
        idx_i = int(idx)
        if idx_i in seen:
            continue
        seen.add(idx_i)
        selected.append(rows[idx_i])
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        if cursor not in seen:
            seen.add(cursor)
            selected.append(rows[cursor])
        cursor += 1
    return selected[:count]


def infer_last_history_speed_kmh(xyz_path: Path, dt_s: float) -> float:
    xyz = np.load(xyz_path)
    hist = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if hist.shape[0] < 2:
        return float("nan")
    # Local history ends at zero. Last delta magnitude approximates current speed.
    dist_m = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]))
    return dist_m / max(dt_s, 1e-9) * 3.6


def prepare_scaled_requests(
    *,
    manifest_rows: list[dict[str, Any]],
    variant: SpeedVariant,
    request_root: Path,
    ego_root: Path,
    seed: int,
    steps: int,
    max_generate_length: int,
) -> list[Path]:
    request_root.mkdir(parents=True, exist_ok=True)
    ego_root.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for row in manifest_rows:
        src = Path(row["request_json"])
        request = load_json(src)
        request["max_generate_length"] = int(max_generate_length)
        inner = request["requests"][0]
        inner["diffusion_seed"] = int(seed)
        inner["diffusion_num_steps"] = int(steps)

        src_xyz = Path(row["ego_history_xyz_npy"])
        src_rot = Path(row["ego_history_rot_npy"])
        stem = src.parent.name
        dst_dir = ego_root / stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_xyz = dst_dir / "ego_history_xyz.npy"
        dst_rot = dst_dir / "ego_history_rot.npy"
        if not dst_xyz.exists():
            xyz = np.load(src_xyz).astype(np.float32)
            xyz_scaled = xyz.copy()
            xyz_scaled[..., :2] *= np.float32(variant.scale)
            np.save(dst_xyz, xyz_scaled)
        if not dst_rot.exists():
            rot = np.load(src_rot).astype(np.float32)
            np.save(dst_rot, rot)

        inner["ego_history_xyz_npy"] = str(dst_xyz)
        inner["ego_history_rot_npy"] = str(dst_rot)
        dst_request = request_root / src.name
        write_json(dst_request, request)
        out_paths.append(dst_request)
    return out_paths


def draw_sample_panels(
    *,
    out_path: Path,
    manifest_rows: list[dict[str, Any]],
    points_by_variant: dict[str, dict[int, np.ndarray]],
    gt_by_sample: dict[int, np.ndarray],
    sample_count: int = 20,
) -> None:
    rows = sorted(manifest_rows, key=lambda row: int(row["sample_id"]))[:sample_count]
    cols = 4
    rows_n = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, max(3.5, rows_n * 3.6)), dpi=170)
    axes_arr = np.asarray(axes).reshape(-1)
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes_arr:
        ax.axis("off")
    for ax, row in zip(axes_arr, rows):
        ax.axis("on")
        ax.set_facecolor("white")
        sample_id = int(row["sample_id"])
        gt = gt_by_sample.get(sample_id)
        if gt is not None:
            ax.plot(gt[:, 1], gt[:, 0], color="#64748b", linewidth=2.2, linestyle="--", label="GT")
        for variant in VARIANTS:
            pts = points_by_variant.get(variant.name, {}).get(sample_id)
            if pts is None:
                continue
            ax.plot(
                pts[:, 1],
                pts[:, 0],
                color=variant.color,
                linestyle=variant.linestyle,
                linewidth=1.45,
                label=variant.label if ax is axes_arr[0] else None,
            )
        ax.scatter([0.0], [0.0], marker="x", color="#020617", s=20)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.75)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.grid(True, axis="x", which="minor", color="#e2e8f0", linewidth=0.45, alpha=0.8)
        ax.grid(True, axis="x", which="major", color="#cbd5e1", linewidth=0.6, alpha=0.8)
        ax.grid(True, axis="y", which="major", color="#e2e8f0", linewidth=0.55, alpha=0.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-0.5, 28.0)
        ax.set_title(f"sid {sample_id} / {float(row['actual_offset_s']):.1f}s", fontsize=8)
        ax.set_xlabel("local y [m]", fontsize=8)
        ax.set_ylabel("local x [m]", fontsize=8)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Ego history speed scaling sensitivity: current legacy seed23", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_metrics(out_path: Path, metrics: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), dpi=170, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, key, title in zip(
        axes,
        ("y_at_4m", "y_at_8m", "end_y_m", "path_len_m"),
        ("local y at 4m", "local y at 8m", "endpoint local y", "path length"),
    ):
        ax.set_facecolor("white")
        for variant in VARIANTS:
            rows = metrics.get(variant.name, [])
            if not rows:
                continue
            xs = [row["offset_s"] for row in rows]
            ys = [row[key] for row in rows]
            ax.plot(xs, ys, color=variant.color, linewidth=1.6, marker="o", markersize=2.6, label=variant.label)
        if key != "path_len_m":
            ax.axhline(0.0, color="#0f172a", linewidth=0.8)
        ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.75)
        ax.set_ylabel(title)
    axes[0].legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("dataset offset [s]")
    fig.suptitle("Ego history speed scaling lateral/path-length metrics", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.output_root / args.dataset_root.name
    request_bank_root = run_root / "request_bank"
    report_root = run_root / "report"
    candidate_count = int(args.selection_bank_count or args.sample_count)
    manifest_rows, request_summary = build_dataset_request_bank(
        dataset_root=args.dataset_root,
        out_root=request_bank_root,
        chunk_id=args.chunk_id,
        sample_count=candidate_count,
        width=args.width,
        height=args.height,
        history_len=args.history_len,
        dt_s=args.dt_s,
        traj_token_offset=args.traj_token_offset,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if args.min_history_speed_kmh is not None:
        filtered_rows = [
            row
            for row in manifest_rows
            if infer_last_history_speed_kmh(Path(row["ego_history_xyz_npy"]), args.dt_s)
            >= float(args.min_history_speed_kmh)
        ]
        if len(filtered_rows) < args.sample_count:
            raise RuntimeError(
                f"Only {len(filtered_rows)} candidate rows satisfy "
                f"--min-history-speed-kmh {args.min_history_speed_kmh}; need {args.sample_count}."
            )
        manifest_rows = select_evenly_spaced_rows(filtered_rows, args.sample_count)
        request_summary = dict(request_summary)
        request_summary["candidate_sample_count"] = candidate_count
        request_summary["min_history_speed_kmh"] = float(args.min_history_speed_kmh)
        request_summary["filtered_candidate_count"] = len(filtered_rows)
        request_summary["selected_sample_ids_after_filter"] = [
            int(row["sample_id"]) for row in manifest_rows
        ]

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
    points_by_variant: dict[str, dict[int, np.ndarray]] = {}
    gt_by_sample: dict[int, np.ndarray] = {}
    metrics: dict[str, list[dict[str, float]]] = {}
    per_variant_summary: dict[str, Any] = {}

    for speed_variant in VARIANTS:
        variant_root = run_root / "variants" / speed_variant.name
        request_paths = prepare_scaled_requests(
            manifest_rows=manifest_rows,
            variant=speed_variant,
            request_root=variant_root / "requests",
            ego_root=variant_root / "ego",
            seed=args.seed,
            steps=args.diffusion_num_steps,
            max_generate_length=args.max_generate_length,
        )
        output_paths = run_variant_outputs(
            variant=legacy_variant,
            request_paths=request_paths,
            output_root=variant_root / "outputs",
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
            artifact_root=variant_root / "artifacts",
        )
        rows_by_request = {Path(row["request_json"]).name: row for row in manifest_rows}
        points_by_variant[speed_variant.name] = {}
        metrics[speed_variant.name] = []
        for request_path in request_paths:
            stem = request_path.stem
            row = rows_by_request[request_path.name]
            sample_id = int(row["sample_id"])
            artifact = artifacts[stem]
            pts = path_xy(artifact["ac_decoded"]).astype(np.float64)
            gt = path_xy(artifact["gt"]).astype(np.float64)
            points_by_variant[speed_variant.name][sample_id] = pts
            gt_by_sample.setdefault(sample_id, gt)
            p4 = point_at_arc(pts, 4.0)
            p8 = point_at_arc(pts, 8.0)
            metrics[speed_variant.name].append(
                {
                    "sample_id": float(sample_id),
                    "offset_s": float(row["actual_offset_s"]),
                    "history_speed_kmh_before_scale": infer_last_history_speed_kmh(Path(row["ego_history_xyz_npy"]), args.dt_s),
                    "scale": float(speed_variant.scale),
                    "y_at_4m": float(p4[1]) if p4 is not None else float("nan"),
                    "y_at_8m": float(p8[1]) if p8 is not None else float("nan"),
                    "end_y_m": float(pts[-1, 1]),
                    "path_len_m": path_length(pts),
                }
            )
        per_variant_summary[speed_variant.name] = {
            "label": speed_variant.label,
            "scale": speed_variant.scale,
            "count": len(metrics[speed_variant.name]),
            "y_at_4m": finite_stats([row["y_at_4m"] for row in metrics[speed_variant.name]]),
            "y_at_8m": finite_stats([row["y_at_8m"] for row in metrics[speed_variant.name]]),
            "end_y_m": finite_stats([row["end_y_m"] for row in metrics[speed_variant.name]]),
            "path_len_m": finite_stats([row["path_len_m"] for row in metrics[speed_variant.name]]),
        }

    panel_png = report_root / "ego_speed_scale_20sample_panels.png"
    metrics_png = report_root / "ego_speed_scale_metrics.png"
    draw_sample_panels(
        out_path=panel_png,
        manifest_rows=manifest_rows,
        points_by_variant=points_by_variant,
        gt_by_sample=gt_by_sample,
        sample_count=args.sample_count,
    )
    draw_metrics(metrics_png, metrics)
    summary = {
        "dataset_root": str(args.dataset_root),
        "request_summary": request_summary,
        "speed_variants": [variant.__dict__ for variant in VARIANTS],
        "outputs": {
            "panel_png": str(panel_png),
            "metrics_png": str(metrics_png),
            "summary_json": str(report_root / "ego_speed_scale_summary.json"),
            "report_md": str(report_root / "ego_speed_scale_report.md"),
        },
        "summary": per_variant_summary,
        "metrics": metrics,
    }
    write_json(report_root / "ego_speed_scale_summary.json", summary)

    lines = [
        "# Ego History Speed Scale Compare",
        "",
        f"- dataset: `{args.dataset_root}`",
        f"- panel: `{panel_png}`",
        f"- metrics: `{metrics_png}`",
        "",
        "Scale convention: actual ~10km/h is 1.0x, so fake 5km/h is 0.5x and fake 100km/h is 10.0x. Only ego_history_xyz x/y is scaled; ego rotations are unchanged.",
        "",
        "## Summary",
        "",
        "| variant | scale | n | y@4 mean | y@4 mean_abs | y@8 mean | y@8 mean_abs | end_y mean | path_len mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if args.min_history_speed_kmh is not None:
        lines[8:8] = [
            "",
            (
                f"Sample selection: built {candidate_count} candidates, kept rows with raw "
                f"ego-history speed >= {args.min_history_speed_kmh:.2f}km/h, then selected "
                f"{args.sample_count} evenly spaced rows."
            ),
            f"Selected sample ids: `{request_summary['selected_sample_ids_after_filter']}`",
        ]
    for variant in VARIANTS:
        item = per_variant_summary[variant.name]

        def fmt(section: str, key: str) -> str:
            value = item[section][key]
            return "nan" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {item['label']} | {item['scale']:.1f} | {item['count']} | "
            f"{fmt('y_at_4m', 'mean')} | {fmt('y_at_4m', 'mean_abs')} | "
            f"{fmt('y_at_8m', 'mean')} | {fmt('y_at_8m', 'mean_abs')} | "
            f"{fmt('end_y_m', 'mean')} | {fmt('path_len_m', 'mean')} |"
        )
    lines.extend(["", "## Per-Sample Metrics", ""])
    lines.append("| variant | sample | offset_s | raw_speed_kmh | y@4 | y@8 | end_y | path_len |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant in VARIANTS:
        for row in metrics[variant.name]:
            lines.append(
                f"| {variant.label} | {int(row['sample_id'])} | {row['offset_s']:.1f} | "
                f"{row['history_speed_kmh_before_scale']:.2f} | {row['y_at_4m']:.3f} | "
                f"{row['y_at_8m']:.3f} | {row['end_y_m']:.3f} | {row['path_len_m']:.3f} |"
            )
    (report_root / "ego_speed_scale_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
