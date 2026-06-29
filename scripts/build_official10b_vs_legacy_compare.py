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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_flex_legacy_threeway_dataset_compare import path_length, path_xy
from scripts.run_legacy_seed23_latency_rollout import (
    interp_world_xyz,
    local_to_world,
    pose_at,
    select_pose_rows,
)


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    color: str
    linestyle: str = "-"


VARIANTS = [
    Variant("gt_future", "GNSS GT future", "#94a3b8", "--"),
    Variant("legacy_seed23", "Legacy seed23 AE", "#111827"),
    Variant("official_10b_ae", "Official 10B AE", "#ea580c"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local/world overlays comparing legacy seed23 and official 10B predictions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/data/2026-06-12-test1"),
    )
    parser.add_argument(
        "--request-bank-root",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/lane_intent_compare_20260612_test1/"
            "2026-06-12-test1/request_bank"
        ),
    )
    parser.add_argument(
        "--legacy-artifact-root",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/lane_intent_compare_20260612_test1/"
            "2026-06-12-test1/variants/no_intent/artifacts"
        ),
    )
    parser.add_argument(
        "--official-jsonl",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/official10b_lane_center_compare_20260612_test1/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/output/official10b_vs_legacy_20260612_test1"),
    )
    parser.add_argument("--dt-s", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def points_from_pred_xyz(pred_xyz: Any) -> np.ndarray:
    pred = np.asarray(pred_xyz, dtype=np.float64)
    pts = np.zeros((pred.shape[0] + 1, 2), dtype=np.float64)
    pts[1:, 0] = pred[:, 0]
    pts[1:, 1] = pred[:, 1]
    return pts


def point_at_arc(points: np.ndarray, arc_m: float) -> np.ndarray | None:
    if len(points) < 2:
        return None
    diffs = np.diff(points[:, :2], axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < arc_m:
        return None
    idx = int(np.searchsorted(cum, arc_m, side="right") - 1)
    idx = min(max(idx, 0), len(seg) - 1)
    denom = max(float(seg[idx]), 1e-9)
    ratio = (arc_m - float(cum[idx])) / denom
    return points[idx] + ratio * (points[idx + 1] - points[idx])


def finite_stats(values: list[float]) -> dict[str, float | int | None]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90_abs": None, "max_abs": None}
    abs_vals = np.abs(vals)
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90_abs": float(np.quantile(abs_vals, 0.9)),
        "max_abs": float(abs_vals.max()),
    }


def load_manifest(request_bank_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_json(request_bank_root / "manifest.json")
    rows = sorted(rows, key=lambda row: int(row["t0_utc_ns"]))
    summary = load_json(request_bank_root / "summary.json")
    return rows, summary


def load_legacy_points(artifact_root: Path, rows: list[dict[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        stem = Path(row["request_json"]).stem
        sample_id = int(row["sample_id"])
        root = artifact_root / stem
        legacy = load_json(root / "ac_decoded_path.json")
        gt = load_json(root / "gt_path.json")
        out[sample_id] = {
            "legacy_seed23": path_xy(legacy).astype(np.float64),
            "gt_future": path_xy(gt).astype(np.float64),
        }
    return out


def load_official_points(jsonl_path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            label = str(item.get("label", ""))
            if label not in {"official_10b_ae", "official_10b_discrete128"}:
                continue
            sample_id = int(item["sample_id"])
            out.setdefault(sample_id, {})[label] = points_from_pred_xyz(item["pred_xyz"])
    return out


def metric_rows(
    rows: list[dict[str, Any]],
    points_by_sample: dict[int, dict[str, np.ndarray]],
) -> dict[str, list[dict[str, float]]]:
    metrics: dict[str, list[dict[str, float]]] = {variant.name: [] for variant in VARIANTS}
    for row in rows:
        sample_id = int(row["sample_id"])
        by_variant = points_by_sample.get(sample_id, {})
        for variant in VARIANTS:
            pts = by_variant.get(variant.name)
            if pts is None:
                continue
            p4 = point_at_arc(pts, 4.0)
            p8 = point_at_arc(pts, 8.0)
            pend = pts[-1]
            metrics[variant.name].append(
                {
                    "sample_id": float(sample_id),
                    "offset_s": float(row["actual_offset_s"]),
                    "end_x_m": float(pend[0]),
                    "end_y_m": float(pend[1]),
                    "y_at_4m": float(p4[1]) if p4 is not None else float("nan"),
                    "y_at_8m": float(p8[1]) if p8 is not None else float("nan"),
                    "path_len_m": float(path_length(pts)),
                }
            )
    return metrics


def summarize(metrics: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = metrics.get(variant.name, [])
        if not rows:
            continue
        item: dict[str, Any] = {"label": variant.label, "count": len(rows)}
        for key in ("y_at_4m", "y_at_8m", "end_y_m", "path_len_m"):
            item[key] = finite_stats([float(row[key]) for row in rows])
        summary[variant.name] = item

    base = {int(row["sample_id"]): row for row in metrics.get("legacy_seed23", [])}
    for name in ("official_10b_ae", "official_10b_discrete128"):
        if name not in summary:
            continue
        rows = metrics.get(name, [])
        for key in ("y_at_4m", "y_at_8m", "end_y_m"):
            deltas: list[float] = []
            for row in rows:
                b = base.get(int(row["sample_id"]))
                if b is None:
                    continue
                v0 = float(b[key])
                v1 = float(row[key])
                if math.isfinite(v0) and math.isfinite(v1):
                    deltas.append(v1 - v0)
            summary[name][f"delta_vs_legacy_{key}"] = finite_stats(deltas)
    return summary


def draw_metrics(out_path: Path, metrics: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=170, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, key, title in zip(
        axes,
        ("y_at_4m", "y_at_8m", "end_y_m"),
        ("local y at arc 4m [m]", "local y at arc 8m [m]", "endpoint local y [m]"),
    ):
        ax.set_facecolor("white")
        for variant in VARIANTS:
            rows = metrics.get(variant.name, [])
            if not rows:
                continue
            xs = [row["offset_s"] for row in rows]
            ys = [row[key] for row in rows]
            ax.plot(
                xs,
                ys,
                color=variant.color,
                linestyle=variant.linestyle,
                marker="o",
                markersize=2.3,
                linewidth=1.45,
                label=variant.label,
            )
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
        ax.set_ylabel(title)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("dataset offset [s]")
    fig.suptitle("Legacy seed23 vs official 10B path lateral response", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_world_tracks(
    *,
    dataset_root: Path,
    request_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    points_by_sample: dict[int, dict[str, np.ndarray]],
    dt_s: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    gnss_valid = select_pose_rows(dataset_root)
    ref_lla = tuple(float(x) for x in request_summary["chunk_ref_lla"])
    t0s = np.asarray([int(row["t0_utc_ns"]) for row in rows], dtype=np.int64)
    dense_step_ns = int(round(dt_s * 1e9))
    dense_times = np.arange(int(t0s[0]), int(t0s[-1]) + dense_step_ns, dense_step_ns, dtype=np.int64)
    gt_dense = interp_world_xyz(gnss_valid, ref_lla, dense_times)[:, :2]

    tracks: dict[str, list[np.ndarray]] = {variant.name: [] for variant in VARIANTS if variant.name != "gt_future"}
    for row_idx, row in enumerate(rows):
        sample_id = int(row["sample_id"])
        t0_ns = int(row["t0_utc_ns"])
        next_t0_ns = int(rows[row_idx + 1]["t0_utc_ns"]) if row_idx + 1 < len(rows) else t0_ns + int(round(1e9))
        duration_s = max(dt_s, min(1.0, float(next_t0_ns - t0_ns) / 1e9))
        p0, rot = pose_at(gnss_valid, ref_lla, t0_ns, dt_s)
        by_variant = points_by_sample.get(sample_id, {})
        for name in list(tracks):
            pts = by_variant.get(name)
            if pts is None or len(pts) < 2:
                continue
            plan_t = np.arange(len(pts), dtype=np.float64) * 0.1
            query_t = np.arange(0.0, min(float(plan_t[-1]), duration_s) + 1e-9, dt_s)
            if query_t.size < 2:
                query_t = np.asarray([0.0, min(float(plan_t[-1]), duration_s)], dtype=np.float64)
            local_seg = np.column_stack(
                [
                    np.interp(query_t, plan_t, pts[:, 0]),
                    np.interp(query_t, plan_t, pts[:, 1]),
                ]
            )
            world_seg = local_to_world(local_seg, p0, rot)
            if tracks[name]:
                world_seg = world_seg[1:]
            tracks[name].append(world_seg)
    final_tracks = {name: np.concatenate(parts, axis=0) for name, parts in tracks.items() if parts}
    return final_tracks, gt_dense


def draw_world(out_path: Path, tracks: dict[str, np.ndarray], gt_dense: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    ax.plot(gt_dense[:, 0], gt_dense[:, 1], color="#64748b", linewidth=3.0, alpha=0.45, label="GNSS GT")
    for variant in VARIANTS:
        if variant.name == "gt_future" or variant.name not in tracks:
            continue
        track = tracks[variant.name]
        ax.plot(
            track[:, 0],
            track[:, 1],
            color=variant.color,
            linestyle=variant.linestyle,
            linewidth=2.0,
            label=variant.label,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.75)
    ax.set_xlabel("ENU east [m]")
    ax.set_ylabel("ENU north [m]")
    ax.set_title("2026-06-12-test1 legacy vs official 10B stitched tracks")
    ax.legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_sample_panels(
    out_path: Path,
    rows: list[dict[str, Any]],
    points_by_sample: dict[int, dict[str, np.ndarray]],
    cols: int = 4,
) -> None:
    if not rows:
        return
    if len(rows) <= 12:
        selected = rows
    else:
        idxs = np.linspace(0, len(rows) - 1, 12).round().astype(int).tolist()
        selected = [rows[i] for i in idxs]
    rows_n = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, max(3.0, rows_n * 3.1)), dpi=170)
    axes_arr = np.asarray(axes).reshape(-1)
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes_arr:
        ax.axis("off")
    for ax, row in zip(axes_arr, selected):
        ax.axis("on")
        ax.set_facecolor("white")
        sample_id = int(row["sample_id"])
        by_variant = points_by_sample.get(sample_id, {})
        for variant in VARIANTS:
            pts = by_variant.get(variant.name)
            if pts is None:
                continue
            ax.plot(
                pts[:, 1],
                pts[:, 0],
                color=variant.color,
                linestyle=variant.linestyle,
                linewidth=1.6,
                label=variant.label if ax is axes_arr[0] else None,
            )
        ax.scatter([0], [0], marker="x", color="#111827", s=20)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.7)
        ax.set_title(f"sid {sample_id} / {float(row['actual_offset_s']):.1f}s", fontsize=8)
        ax.set_xlabel("local y [m]", fontsize=8)
        ax.set_ylabel("local x [m]", fontsize=8)
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Local trajectory samples: legacy vs official 10B", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, outputs: dict[str, str], summary: dict[str, Any]) -> None:
    lines = [
        "# Official 10B vs Legacy Path Compare",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Lateral Metrics", ""])
    lines.append("| variant | n | y@4 mean | y@4 p90abs | y@8 mean | y@8 p90abs | endpoint y mean | path len mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant in VARIANTS:
        item = summary.get(variant.name)
        if not item:
            continue
        def fmt(section: str, key: str) -> str:
            value = item[section][key]
            return "nan" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {item['label']} | {item['count']} | "
            f"{fmt('y_at_4m', 'mean')} | {fmt('y_at_4m', 'p90_abs')} | "
            f"{fmt('y_at_8m', 'mean')} | {fmt('y_at_8m', 'p90_abs')} | "
            f"{fmt('end_y_m', 'mean')} | {fmt('path_len_m', 'mean')} |"
        )
    lines.extend(["", "## Official Delta Vs Legacy", ""])
    for name in ("official_10b_ae", "official_10b_discrete128"):
        item = summary.get(name)
        if not item:
            continue
        lines.append(f"### {item['label']}")
        for key in ("delta_vs_legacy_y_at_4m", "delta_vs_legacy_y_at_8m", "delta_vs_legacy_end_y_m"):
            stats = item.get(key)
            if stats:
                lines.append(
                    f"- {key}: mean={stats['mean']}, median={stats['median']}, "
                    f"p90_abs={stats['p90_abs']}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, request_summary = load_manifest(args.request_bank_root)
    legacy = load_legacy_points(args.legacy_artifact_root, rows)
    official = load_official_points(args.official_jsonl)
    points_by_sample: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        points_by_sample[sample_id] = {}
        points_by_sample[sample_id].update(legacy.get(sample_id, {}))
        points_by_sample[sample_id].update(official.get(sample_id, {}))

    metrics = metric_rows(rows, points_by_sample)
    summary = summarize(metrics)
    report_root = args.output_root / args.dataset_root.name / "report"
    world_png = report_root / "official10b_vs_legacy_world_overlay.png"
    metrics_png = report_root / "official10b_vs_legacy_lateral_metrics.png"
    panels_png = report_root / "official10b_vs_legacy_sample_panels.png"
    summary_json = report_root / "official10b_vs_legacy_summary.json"
    report_md = report_root / "official10b_vs_legacy_report.md"

    tracks, gt_dense = build_world_tracks(
        dataset_root=args.dataset_root,
        request_summary=request_summary,
        rows=rows,
        points_by_sample=points_by_sample,
        dt_s=args.dt_s,
    )
    draw_world(world_png, tracks, gt_dense)
    draw_metrics(metrics_png, metrics)
    draw_sample_panels(panels_png, rows, points_by_sample)
    outputs = {
        "world_overlay": str(world_png),
        "lateral_metrics": str(metrics_png),
        "sample_panels": str(panels_png),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    write_json(
        summary_json,
        {
            "dataset_root": str(args.dataset_root),
            "request_bank_root": str(args.request_bank_root),
            "legacy_artifact_root": str(args.legacy_artifact_root),
            "official_jsonl": str(args.official_jsonl),
            "outputs": outputs,
            "metrics": summary,
            "per_sample_metrics": metrics,
        },
    )
    write_report(report_md, outputs, summary)
    print(json.dumps({"outputs": outputs, "metrics": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
