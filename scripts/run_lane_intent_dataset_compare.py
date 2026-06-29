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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    build_dataset_request_bank,
    build_variant_artifacts,
    ensure_exists,
    load_json,
    path_xy,
    write_json,
)
from scripts.run_legacy_seed23_latency_rollout import (
    interp_world_xyz,
    local_to_world,
    pose_at,
    select_pose_rows,
)
from scripts.run_request_bank_persistent import build_env, read_status


@dataclass(frozen=True)
class LaneIntentVariant:
    name: str
    label: str
    nav_text: str | None
    color: str
    linestyle: str = "-"


VARIANTS = [
    LaneIntentVariant("no_intent", "No intent", None, "#111827", "-"),
    LaneIntentVariant(
        "keep_current_lane",
        "Keep current lane",
        "Keep the current lane. Follow the center of the current lane and do not change lanes.",
        "#2563eb",
        "-",
    ),
    LaneIntentVariant(
        "left_lane_center",
        "Left lane center",
        "Follow the center of the left lane. Move toward and stay centered in the left lane if it is safe.",
        "#dc2626",
        "-",
    ),
    LaneIntentVariant(
        "right_lane_center",
        "Right lane center",
        "Follow the center of the right lane. Move toward and stay centered in the right lane if it is safe.",
        "#16a34a",
        "-",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy Alpamayo on a raw dataset with lane-level route text variants "
            "and render whole-dataset path comparison overlays."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "2026-06-12-test1")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "lane_intent_compare_20260612_test1")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--diffusion-seed", type=int, default=23)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0)
    parser.add_argument("--alpamayo-nav-cfg", action="store_true", default=True)
    parser.add_argument("--no-alpamayo-nav-cfg", dest="alpamayo_nav_cfg", action="store_false")
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true", default=True)
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
    parser.add_argument(
        "--path-source",
        choices=["ac_decoded", "final"],
        default="ac_decoded",
        help="ac_decoded matches the UDP/control path packet convention used by replay tools.",
    )
    return parser.parse_args()


def remove_route_span(text: str) -> str:
    start_token = "<|route_start|>"
    end_token = "<|route_end|>"
    start = text.find(start_token)
    if start < 0:
        return text
    end = text.find(end_token, start + len(start_token))
    if end < 0:
        return text
    return text[:start] + text[end + len(end_token) :]


def insert_route_span(text: str, nav_text: str) -> str:
    text = remove_route_span(text)
    history_end_token = "<|traj_history_end|>"
    history_end = text.find(history_end_token)
    if history_end < 0:
        return text
    insert_pos = history_end + len(history_end_token)
    route_span = f"<|route_start|>{nav_text}<|route_end|>"
    return text[:insert_pos] + route_span + text[insert_pos:]


def prepare_lane_intent_requests(
    *,
    manifest_rows: list[dict[str, Any]],
    variant: LaneIntentVariant,
    out_root: Path,
    seed: int,
    steps: int,
    max_generate_length: int,
    nav_guidance_weight: float,
) -> list[Path]:
    request_root = out_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for row in manifest_rows:
        src = Path(row["request_json"])
        request_obj = load_json(src)
        request_obj["max_generate_length"] = int(max_generate_length)
        request = request_obj["requests"][0]
        request["diffusion_seed"] = int(seed)
        request["diffusion_num_steps"] = int(steps)
        request.pop("nav_text", None)
        request.pop("nav_guidance_weight", None)

        for message in request.get("messages", []):
            if message.get("role") != "user":
                continue
            for content in message.get("content", []):
                if content.get("type") != "text":
                    continue
                text = str(content.get("text", ""))
                if "<|traj_history_end|>" not in text and "<|route_start|>" not in text:
                    continue
                content["text"] = (
                    insert_route_span(text, variant.nav_text)
                    if variant.nav_text
                    else remove_route_span(text)
                )

        if variant.nav_text:
            request["nav_text"] = variant.nav_text
            request["nav_guidance_weight"] = float(nav_guidance_weight)

        dst = request_root / src.name
        write_json(dst, request_obj)
        out_paths.append(dst)
    return out_paths


def run_outputs(
    *,
    args: argparse.Namespace,
    variant: LaneIntentVariant,
    request_paths: list[Path],
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [output_root / path.name.replace("request_", "output_") for path in request_paths]
    if args.skip_existing and output_paths and all(path.exists() for path in output_paths):
        print(f"[lane-intent] reuse all outputs for {variant.name}", flush=True)
        return output_paths

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

    print(f"[lane-intent] starting {variant.name}", flush=True)
    print("[lane-intent] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )

    try:
        ready = read_status(proc, timeout_s=180.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected llm_inference ready state for {variant.name}: {ready}")
        print(f"[lane-intent] {variant.name} ready", flush=True)

        start = time.time()
        for idx, (request_path, output_path) in enumerate(zip(request_paths, output_paths, strict=True), start=1):
            if args.skip_existing and output_path.exists():
                print(f"[lane-intent] skip {variant.name} {idx}/{len(request_paths)}", flush=True)
                continue
            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            t0 = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=args.timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"{variant.name} failed for {request_path.name}: {status}")
            elapsed = time.time() - start
            avg = elapsed / max(idx, 1)
            eta = avg * (len(request_paths) - idx)
            print(
                f"[lane-intent] {variant.name} done {idx}/{len(request_paths)} "
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


def point_at_arc(points: np.ndarray, arc_m: float) -> np.ndarray | None:
    if len(points) < 2:
        return None
    diffs = np.diff(points[:, :2], axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < arc_m:
        return None
    idx = int(np.searchsorted(cum, arc_m, side="right") - 1)
    idx = max(0, min(idx, len(seg) - 1))
    denom = max(float(seg[idx]), 1e-6)
    alpha = (float(arc_m) - float(cum[idx])) / denom
    return points[idx] + alpha * (points[idx + 1] - points[idx])


def build_dataset_tracks(
    *,
    dataset_root: Path,
    request_summary: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    artifacts_by_variant: dict[str, dict[str, dict[str, Any]]],
    path_source: str,
    dt_s: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, list[dict[str, float]]]]:
    gnss_valid = select_pose_rows(dataset_root)
    ref_lla = tuple(float(x) for x in request_summary["chunk_ref_lla"])
    rows = sorted(manifest_rows, key=lambda row: int(row["t0_utc_ns"]))

    t0s = np.asarray([int(row["t0_utc_ns"]) for row in rows], dtype=np.int64)
    dense_step_ns = int(round(dt_s * 1e9))
    dense_times = np.arange(int(t0s[0]), int(t0s[-1]) + dense_step_ns, dense_step_ns, dtype=np.int64)
    gt_dense = interp_world_xyz(gnss_valid, ref_lla, dense_times)[:, :2]

    tracks: dict[str, list[np.ndarray]] = {variant.name: [] for variant in VARIANTS}
    metric_rows: dict[str, list[dict[str, float]]] = {variant.name: [] for variant in VARIANTS}

    for row_idx, row in enumerate(rows):
        t0_ns = int(row["t0_utc_ns"])
        next_t0_ns = int(rows[row_idx + 1]["t0_utc_ns"]) if row_idx + 1 < len(rows) else t0_ns + int(round(1e9))
        duration_s = max(dt_s, min(1.0, float(next_t0_ns - t0_ns) / 1e9))
        stem = Path(row["request_json"]).stem
        p0, rot = pose_at(gnss_valid, ref_lla, t0_ns, dt_s)

        for variant in VARIANTS:
            source = artifacts_by_variant[variant.name][stem][path_source]
            pts = path_xy(source).astype(np.float64)
            plan_dt = float(source.get("plan_dt_s", dt_s))
            plan_t = np.arange(len(pts), dtype=np.float64) * plan_dt
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
            if tracks[variant.name]:
                world_seg = world_seg[1:]
            tracks[variant.name].append(world_seg)

            p4 = point_at_arc(pts, 4.0)
            p8 = point_at_arc(pts, 8.0)
            pend = pts[-1]
            metric_rows[variant.name].append(
                {
                    "sample_id": float(row["sample_id"]),
                    "offset_s": float(row["actual_offset_s"]),
                    "end_x_m": float(pend[0]),
                    "end_y_m": float(pend[1]),
                    "y_at_4m": float(p4[1]) if p4 is not None else float("nan"),
                    "y_at_8m": float(p8[1]) if p8 is not None else float("nan"),
                    "path_len_m": float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum()),
                }
            )

    final_tracks = {name: np.concatenate(parts, axis=0) for name, parts in tracks.items() if parts}
    return final_tracks, gt_dense, metric_rows


def draw_world_overlay(
    out_path: Path,
    tracks: dict[str, np.ndarray],
    gt_dense: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    ax.plot(gt_dense[:, 0], gt_dense[:, 1], color="#64748b", linewidth=3.0, alpha=0.55, label="GNSS GT")
    for variant in VARIANTS:
        track = tracks[variant.name]
        ax.plot(track[:, 0], track[:, 1], color=variant.color, linestyle=variant.linestyle, linewidth=2.0, label=variant.label)
        ax.scatter(track[0, 0], track[0, 1], color=variant.color, s=18)
        ax.scatter(track[-1, 0], track[-1, 1], color=variant.color, s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.75)
    ax.set_xlabel("ENU east [m]")
    ax.set_ylabel("ENU north [m]")
    ax.set_title("2026-06-12-test1 lane intent comparison: stitched 1Hz tracks")
    ax.legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_metrics(out_path: Path, metric_rows: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), dpi=160, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    metric_defs = [("y_at_4m", "local y at arc 4m [m]"), ("y_at_8m", "local y at arc 8m [m]"), ("end_y_m", "endpoint local y [m]")]
    for ax, (key, ylabel) in zip(axes, metric_defs, strict=True):
        ax.set_facecolor("white")
        ax.axhline(0.0, color="#94a3b8", linewidth=1.0)
        for variant in VARIANTS:
            rows = metric_rows[variant.name]
            x = np.asarray([r["offset_s"] for r in rows], dtype=np.float64)
            y = np.asarray([r[key] for r in rows], dtype=np.float64)
            ax.plot(x, y, color=variant.color, linestyle=variant.linestyle, linewidth=1.7, marker="o", markersize=2.4, label=variant.label)
        ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.75)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("dataset offset [s]")
    fig.suptitle("Lane intent lateral response over full test1 sample span", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_sample_panels(
    *,
    out_path: Path,
    manifest_rows: list[dict[str, Any]],
    artifacts_by_variant: dict[str, dict[str, dict[str, Any]]],
    path_source: str,
    max_panels: int = 12,
) -> None:
    rows = sorted(manifest_rows, key=lambda row: int(row["t0_utc_ns"]))
    if len(rows) > max_panels:
        indices = np.linspace(0, len(rows) - 1, max_panels, dtype=np.int64)
        rows = [rows[int(i)] for i in indices]
    cols = 4
    panel_rows = int(np.ceil(len(rows) / cols))
    fig = plt.figure(figsize=(18, 4.8 * panel_rows), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(panel_rows, cols, hspace=0.28, wspace=0.18)
    for idx, row in enumerate(rows):
        ax = fig.add_subplot(grid[idx // cols, idx % cols])
        ax.set_facecolor("white")
        stem = Path(row["request_json"]).stem
        gt = path_xy(next(iter(artifacts_by_variant.values()))[stem]["gt"])
        ax.plot(gt[:, 1], gt[:, 0], color="#64748b", linewidth=2.2, alpha=0.55, label="GT")
        for variant in VARIANTS:
            pts = path_xy(artifacts_by_variant[variant.name][stem][path_source])
            ax.plot(pts[:, 1], pts[:, 0], color=variant.color, linewidth=1.7, label=variant.label)
        ax.scatter([0.0], [0.0], marker="x", color="#111827", s=36)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.75)
        ax.set_title(f"sid {row['sample_id']} / {row['actual_offset_s']:.1f}s", fontsize=9)
        ax.set_xlabel("local y [m]")
        ax.set_ylabel("local x [m]")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(metric_rows: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    base = metric_rows["no_intent"]
    base_by_sid = {int(row["sample_id"]): row for row in base}
    for variant in VARIANTS:
        rows = metric_rows[variant.name]
        item: dict[str, Any] = {"label": variant.label, "nav_text": variant.nav_text, "count": len(rows)}
        for key in ("y_at_4m", "y_at_8m", "end_y_m", "path_len_m"):
            arr = np.asarray([r[key] for r in rows], dtype=np.float64)
            item[key] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p90_abs": float(np.nanquantile(np.abs(arr), 0.90)),
                "max_abs": float(np.nanmax(np.abs(arr))),
            }
        if variant.name != "no_intent":
            for key in ("y_at_4m", "y_at_8m", "end_y_m"):
                diffs = []
                for row in rows:
                    base_row = base_by_sid.get(int(row["sample_id"]))
                    if base_row is not None:
                        diffs.append(float(row[key]) - float(base_row[key]))
                arr = np.asarray(diffs, dtype=np.float64)
                item[f"delta_vs_no_intent_{key}"] = {
                    "mean": float(np.nanmean(arr)),
                    "median": float(np.nanmedian(arr)),
                    "p90_abs": float(np.nanquantile(np.abs(arr), 0.90)),
                }
        summary[variant.name] = item
    return summary


def write_report(path: Path, args: argparse.Namespace, outputs: dict[str, str], metric_summary: dict[str, Any]) -> None:
    lines = [
        "# Lane Intent Dataset Compare",
        "",
        f"- dataset: `{args.dataset_root}`",
        f"- chunk: `{args.chunk_id}`",
        f"- samples: `{args.sample_count}` evenly spaced over full chunk",
        f"- model: legacy seed `{args.diffusion_seed}`, FM steps `{args.diffusion_num_steps}`, prefill-KV `{args.alpamayo_fm_use_prefill_kv}`",
        f"- nav CFG: `{args.alpamayo_nav_cfg}`, nav guidance weight `{args.nav_guidance_weight}`",
        f"- path source: `{args.path_source}`",
        "",
        "## Outputs",
        "",
    ]
    for name, out in outputs.items():
        lines.append(f"- {name}: `{out}`")
    lines += [
        "",
        "## Lateral Metrics",
        "",
        "| variant | y@4 mean | y@4 p90abs | y@8 mean | y@8 p90abs | endpoint y mean | path len mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        item = metric_summary[variant.name]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["label"],
                    f"{item['y_at_4m']['mean']:.3f}",
                    f"{item['y_at_4m']['p90_abs']:.3f}",
                    f"{item['y_at_8m']['mean']:.3f}",
                    f"{item['y_at_8m']['p90_abs']:.3f}",
                    f"{item['end_y_m']['mean']:.3f}",
                    f"{item['path_len_m']['mean']:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_exists(args.dataset_root, "dataset root")
    for path, description in [
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engine dir"),
        (args.multimodal_engine_dir, "multimodal engine dir"),
        (args.fm_engine, "fm engine"),
    ]:
        ensure_exists(path, description)

    args.work_root.mkdir(parents=True, exist_ok=True)
    request_bank_root = args.work_root / args.dataset_root.name / "request_bank"
    manifest_rows, request_summary = build_dataset_request_bank(
        dataset_root=args.dataset_root,
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

    artifacts_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        variant_root = args.work_root / args.dataset_root.name / "variants" / variant.name
        request_paths = prepare_lane_intent_requests(
            manifest_rows=manifest_rows,
            variant=variant,
            out_root=variant_root / "request_bank",
            seed=args.diffusion_seed,
            steps=args.diffusion_num_steps,
            max_generate_length=args.max_generate_length,
            nav_guidance_weight=args.nav_guidance_weight,
        )
        output_paths = run_outputs(args=args, variant=variant, request_paths=request_paths, output_root=variant_root / "outputs")
        fake_variant = type(
            "Variant",
            (),
            {
                "name": variant.name,
                "label": variant.label,
                "color": variant.color,
            },
        )()
        artifacts_by_variant[variant.name] = build_variant_artifacts(
            variant=fake_variant,
            output_paths=output_paths,
            request_paths=request_paths,
            manifest_rows=manifest_rows,
            dataset_root=args.dataset_root,
            history_len=args.history_len,
            artifact_root=variant_root / "artifacts",
        )

    tracks, gt_dense, metric_rows = build_dataset_tracks(
        dataset_root=args.dataset_root,
        request_summary=request_summary,
        manifest_rows=manifest_rows,
        artifacts_by_variant=artifacts_by_variant,
        path_source=args.path_source,
        dt_s=args.dt_s,
    )
    report_root = args.work_root / args.dataset_root.name / "report"
    world_png = report_root / "lane_intent_world_overlay.png"
    metrics_png = report_root / "lane_intent_lateral_metrics.png"
    panels_png = report_root / "lane_intent_sample_panels.png"
    draw_world_overlay(world_png, tracks, gt_dense)
    draw_metrics(metrics_png, metric_rows)
    draw_sample_panels(
        out_path=panels_png,
        manifest_rows=manifest_rows,
        artifacts_by_variant=artifacts_by_variant,
        path_source=args.path_source,
    )
    metric_summary = summarize_metrics(metric_rows)
    outputs = {
        "world_overlay": str(world_png),
        "lateral_metrics": str(metrics_png),
        "sample_panels": str(panels_png),
        "summary_json": str(report_root / "lane_intent_summary.json"),
        "report_md": str(report_root / "lane_intent_report.md"),
    }
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(
        report_root / "lane_intent_summary.json",
        {
            "args": serializable_args,
            "variants": [
                {"name": v.name, "label": v.label, "nav_text": v.nav_text, "color": v.color}
                for v in VARIANTS
            ],
            "outputs": outputs,
            "metrics": metric_summary,
            "per_sample_metrics": metric_rows,
        },
    )
    write_report(report_root / "lane_intent_report.md", args, outputs, metric_summary)
    print(json.dumps({"outputs": outputs, "metrics": metric_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
