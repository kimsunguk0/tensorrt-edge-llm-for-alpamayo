#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine
from scripts.run_raw_dataset_one_shot_udp import build_result_artifacts
from scripts.run_request_bank_persistent import build_env, read_status


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    nav_text: str | None
    color: str
    linestyle: str
    linewidth: float = 2.2


VARIANTS = [
    Variant("no_nav", "No Nav", None, "#111827", "-", 2.6),
    Variant(
        "left_mild",
        "Left Mild",
        "At the upcoming intersection, gently prefer the left path if it is safe and available.",
        "#93c5fd",
        "-",
    ),
    Variant(
        "left_clear",
        "Left Clear",
        "Turn left at the upcoming intersection. Follow the left-turn route smoothly.",
        "#2563eb",
        "-",
    ),
    Variant(
        "left_strong",
        "Left Strong",
        "The intended route is left. Strongly prioritize the left turn and do not continue straight or turn right.",
        "#1d4ed8",
        "-",
        2.6,
    ),
    Variant(
        "left_must",
        "Left Must",
        "Critical route instruction: you must make a decisive left turn now. Reject any straight or right-turn path.",
        "#1e3a8a",
        "-",
        2.8,
    ),
    Variant(
        "right_mild",
        "Right Mild",
        "At the upcoming intersection, gently prefer the right path if it is safe and available.",
        "#fca5a5",
        (0, (5, 3)),
    ),
    Variant(
        "right_clear",
        "Right Clear",
        "Turn right at the upcoming intersection. Follow the right-turn route smoothly.",
        "#ef4444",
        (0, (5, 3)),
    ),
    Variant(
        "right_strong",
        "Right Strong",
        "The intended route is right. Strongly prioritize the right turn and do not continue straight or turn left.",
        "#dc2626",
        (0, (5, 3)),
        2.6,
    ),
    Variant(
        "right_must",
        "Right Must",
        "Critical route instruction: you must make a decisive right turn now. Reject any straight or left-turn path.",
        "#991b1b",
        (0, (5, 3)),
        2.8,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed sample with multiple nav-strength prompts and plot path changes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2"),
    )
    parser.add_argument("--chunk-id", type=int, default=4)
    parser.add_argument("--target-offset-s", type=float, default=23.0)
    parser.add_argument(
        "--request-bank-root",
        type=Path,
        default=None,
        help="Existing request bank for this chunk. Defaults to output/live_chunk_udp_replay_2026_04_17_test2/chunkXXXX/request_bank.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "output" / "reports" / "nav_strength_single_sample",
    )
    parser.add_argument(
        "--path-source",
        choices=["ac_decoded", "final"],
        default="ac_decoded",
        help="Path family to compare. ac_decoded matches the control-team UDP path.",
    )
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0)
    parser.add_argument("--alpamayo-nav-cfg", action="store_true")
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=first_existing_fm_engine())
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--reuse-existing-outputs", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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
    history_end_token = "<|traj_history_end|>"
    history_end = text.find(history_end_token)
    if history_end < 0:
        return text
    text = remove_route_span(text)
    route_span = f"<|route_start|>{nav_text}<|route_end|>"
    insert_pos = history_end + len(history_end_token)
    return text[:insert_pos] + route_span + text[insert_pos:]


def default_request_bank_root(chunk_id: int) -> Path:
    return (
        REPO_ROOT
        / "output"
        / "live_chunk_udp_replay_2026_04_17_test2"
        / f"chunk{chunk_id:04d}"
        / "request_bank"
    )


def select_request_row(manifest_rows: list[dict[str, Any]], target_offset_s: float) -> dict[str, Any]:
    if not manifest_rows:
        raise RuntimeError("Request manifest is empty")
    rows = sorted(manifest_rows, key=lambda row: int(row["t0_utc_ns"]))
    start_ns = int(rows[0]["t0_utc_ns"])
    target_ns = start_ns + int(round(target_offset_s * 1e9))
    return min(rows, key=lambda row: abs(int(row["t0_utc_ns"]) - target_ns))


def build_metadata(row: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(manifest_rows, key=lambda item: int(item["t0_utc_ns"]))
    chunk_start_ns = int(rows[0]["t0_utc_ns"])
    selected_frames = row.get("selected_frames", {})
    front_ids = selected_frames.get("front") or []
    front_frame_id = int(front_ids[-1]) if front_ids else int(row["sample_id"])
    offset_s = float((int(row["t0_utc_ns"]) - chunk_start_ns) / 1e9)
    return {
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "front_frame_id": front_frame_id,
        "t0_utc_ns": int(row["t0_utc_ns"]),
        "t0_us": int(row["t0_us"]),
        "target_offset_s": offset_s,
        "actual_offset_s": offset_s,
        "request_json": str(row["request_json"]),
        "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
        "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
        "selected_frames": selected_frames,
    }


def prepare_request(base_request_path: Path, variant: Variant, out_path: Path, nav_guidance_weight: float) -> None:
    request_obj = load_json(base_request_path)
    assert isinstance(request_obj, dict)
    request = request_obj["requests"][0]
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
            content["text"] = insert_route_span(text, variant.nav_text) if variant.nav_text else remove_route_span(text)

    if variant.nav_text:
        request["nav_text"] = variant.nav_text
        request["nav_guidance_weight"] = float(nav_guidance_weight)
    write_json(out_path, request_obj)


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
    if args.alpamayo_nav_cfg:
        cmd.append("--alpamayoNavCfg")
    return cmd


def run_requests(args: argparse.Namespace, request_paths: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / request_path.name.replace("request_", "output_")
        for name, request_path in request_paths.items()
    }
    if args.reuse_existing_outputs and all(path.exists() for path in output_paths.values()):
        return output_paths

    cmd = runtime_cmd(args)
    print("[nav-strength] " + " ".join(cmd), flush=True)
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
        ready = read_status(proc, timeout_s=120.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected llm_inference ready state: {ready}")
        print("[nav-strength] runtime ready", flush=True)

        for idx, variant in enumerate(VARIANTS, start=1):
            request_path = request_paths[variant.name]
            output_path = output_paths[variant.name]
            if args.reuse_existing_outputs and output_path.exists():
                print(f"[nav-strength] reuse {variant.name} {idx}/{len(VARIANTS)}", flush=True)
                continue
            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            start = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=args.timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"llm_inference failed for {variant.name}: {status}")
            print(f"[nav-strength] {variant.name} {idx}/{len(VARIANTS)} done in {time.time() - start:.2f}s", flush=True)
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


def packet_points_xy(summary: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [[float(point["x_m"]), float(point["y_m"])] for point in summary["packet_points"]],
        dtype=np.float32,
    )


def plot_report(
    *,
    out_path: Path,
    image_paths: dict[str, Path],
    summaries: dict[str, dict[str, Any]],
    gt_summary: dict[str, Any],
    path_source: str,
    chunk_id: int,
    offset_s: float,
) -> None:
    fig = plt.figure(figsize=(16, 12), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.16, 2.05], hspace=0.25, wspace=0.08)

    for col, (name, image_path) in enumerate(image_paths.items()):
        ax_img = fig.add_subplot(grid[0, col])
        ax_img.imshow(mpimg.imread(image_path))
        ax_img.set_title(name, fontsize=11, fontweight="bold")
        ax_img.axis("off")

    ax_text = fig.add_subplot(grid[1, :])
    ax_text.axis("off")

    ax = fig.add_subplot(grid[2, :])
    ax.set_facecolor("#ffffff")

    gt_xy = packet_points_xy(gt_summary)
    ax.plot(
        -gt_xy[:, 1],
        gt_xy[:, 0],
        color="#94a3b8",
        linewidth=8.0,
        alpha=0.32,
        solid_capstyle="round",
        label="GT future",
        zorder=1,
    )
    ax.plot(-gt_xy[:, 1], gt_xy[:, 0], color="#475569", linewidth=1.4, alpha=0.8, zorder=2)

    for variant in VARIANTS:
        pts = packet_points_xy(summaries[variant.name])
        ax.plot(
            -pts[:, 1],
            pts[:, 0],
            color=variant.color,
            linestyle=variant.linestyle,
            linewidth=variant.linewidth,
            alpha=0.95,
            solid_capstyle="round",
            label=variant.label,
            zorder=4 if variant.name != "no_nav" else 5,
        )
        ax.scatter(-pts[-1, 1], pts[-1, 0], color=variant.color, s=28, alpha=0.95, zorder=6)

    ax.scatter([0.0], [0.0], marker="x", s=90, linewidths=2.0, color="#111827", zorder=8)
    ax.text(0.2, 0.8, "ego (0,0)", color="#111827", fontsize=9)

    all_pts = [np.column_stack([-gt_xy[:, 1], gt_xy[:, 0]])]
    all_pts.extend(np.column_stack([-packet_points_xy(summary)[:, 1], packet_points_xy(summary)[:, 0]]) for summary in summaries.values())
    stack = np.concatenate(all_pts, axis=0)
    x_min, y_min = stack.min(axis=0)
    x_max, y_max = stack.max(axis=0)
    pad_x = max(4.0, float(x_max - x_min) * 0.18)
    pad_y = max(4.0, float(y_max - y_min) * 0.12)
    ax.set_xlim(float(x_min - pad_x), float(x_max + pad_x))
    ax.set_ylim(min(-2.0, float(y_min - pad_y)), float(y_max + pad_y))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.72)
    ax.set_xlabel("local right [m]  (-local y; left turn goes left on the plot)")
    ax.set_ylabel("local forward [m]")
    ax.set_title(
        f"Nav Strength Test: chunk{chunk_id:04d} @ {offset_s:.1f}s, path={path_source}",
        fontsize=15,
        fontweight="bold",
    )

    handles = [Line2D([0], [0], color="#94a3b8", linewidth=8.0, alpha=0.45, label="GT future")]
    handles.extend(
        Line2D([0], [0], color=v.color, linestyle=v.linestyle, linewidth=v.linewidth, label=v.label)
        for v in VARIANTS
    )
    ax.legend(handles=handles, ncol=2, fontsize=9, loc="upper left", frameon=True)

    final_lines = []
    for variant in VARIANTS:
        text = str(summaries[variant.name].get("final_output", "")).strip()
        if text and text not in final_lines:
            final_lines.append(text)
    final_text = " / ".join(final_lines[:3]) if final_lines else "(no final output)"
    final_text = textwrap.fill(f"Final output examples: {final_text}", width=150)
    ax_text.text(
        0.5,
        0.52,
        final_text,
        transform=ax_text.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_variant(summary: dict[str, Any], no_nav_summary: dict[str, Any], gt_summary: dict[str, Any]) -> dict[str, Any]:
    pts = packet_points_xy(summary)
    no_nav = packet_points_xy(no_nav_summary)
    gt = packet_points_xy(gt_summary)
    n = min(len(pts), len(no_nav), len(gt))
    diff_no_nav = np.linalg.norm(pts[:n] - no_nav[:n], axis=1)
    diff_gt = np.linalg.norm(pts[:n] - gt[:n], axis=1)
    curvature = np.asarray(summary.get("pred_curvature", []), dtype=np.float32)
    speed = np.asarray(summary.get("pred_v_mps", []), dtype=np.float32)
    endpoint = pts[-1]
    return {
        "final_output": summary.get("final_output", ""),
        "timing": summary.get("timing", {}),
        "endpoint_x_forward_m": float(endpoint[0]),
        "endpoint_y_left_m": float(endpoint[1]),
        "endpoint_right_m": float(-endpoint[1]),
        "mean_curvature": float(np.nanmean(curvature)) if curvature.size else None,
        "final_speed_kph": float(speed[-1] * 3.6) if speed.size else None,
        "vs_no_nav_mean_m": float(np.mean(diff_no_nav)),
        "vs_no_nav_endpoint_m": float(diff_no_nav[n - 1]),
        "vs_gt_mean_m": float(np.mean(diff_gt)),
        "vs_gt_endpoint_m": float(diff_gt[n - 1]),
    }


def main() -> None:
    args = parse_args()
    request_bank_root = args.request_bank_root or default_request_bank_root(args.chunk_id)
    manifest_rows = load_json(request_bank_root / "manifest.json")
    request_summary = load_json(request_bank_root / "summary.json")
    assert isinstance(manifest_rows, list)
    assert isinstance(request_summary, dict)

    selected_row = select_request_row(manifest_rows, args.target_offset_s)
    metadata = build_metadata(selected_row, manifest_rows)
    stem = f"chunk{args.chunk_id:04d}_sid{int(metadata['sample_id']):05d}_offset{metadata['actual_offset_s']:.1f}".replace(".", "p")
    work_root = args.work_root / stem
    request_dir = work_root / "requests"
    output_dir = work_root / "outputs"
    artifact_dir = work_root / "artifacts"
    request_dir.mkdir(parents=True, exist_ok=True)

    request_paths: dict[str, Path] = {}
    base_request = Path(selected_row["request_json"])
    for variant in VARIANTS:
        request_path = request_dir / f"request_{variant.name}_{base_request.name.removeprefix('request_')}"
        prepare_request(base_request, variant, request_path, args.nav_guidance_weight)
        request_paths[variant.name] = request_path

    output_paths = run_requests(args, request_paths, output_dir)

    summaries: dict[str, dict[str, Any]] = {}
    gt_summary: dict[str, Any] | None = None
    for variant in VARIANTS:
        variant_artifact_dir = artifact_dir / variant.name
        final_summary, ac_summary, this_gt_summary, _ = build_result_artifacts(
            output_path=output_paths[variant.name],
            metadata=metadata,
            dataset_root=args.dataset_root,
            history_len=int(request_summary["history_len"]),
            artifact_root=variant_artifact_dir,
        )
        summaries[variant.name] = ac_summary if args.path_source == "ac_decoded" else final_summary
        if gt_summary is None:
            gt_summary = this_gt_summary
    assert gt_summary is not None

    selected_frames = metadata["selected_frames"]
    image_root = Path(request_summary["image_root"]) / base_request.stem.removeprefix("request_")
    image_paths = {
        "Left": image_root / "cam0_f3.png",
        "Front": image_root / "cam1_f3.png",
        "Right": image_root / "cam2_f3.png",
        "Front Tele": image_root / "cam6_f3.png",
    }
    for image_path in image_paths.values():
        if not image_path.exists():
            raise RuntimeError(f"Missing input image: {image_path}")

    out_png = work_root / f"{stem}_nav_strength_overlay.png"
    plot_report(
        out_path=out_png,
        image_paths=image_paths,
        summaries=summaries,
        gt_summary=gt_summary,
        path_source=args.path_source,
        chunk_id=args.chunk_id,
        offset_s=float(metadata["actual_offset_s"]),
    )

    no_nav_summary = summaries["no_nav"]
    stats = {
        "dataset_root": str(args.dataset_root),
        "chunk_id": int(args.chunk_id),
        "target_offset_s": float(args.target_offset_s),
        "actual_offset_s": float(metadata["actual_offset_s"]),
        "sample_id": int(metadata["sample_id"]),
        "path_source": args.path_source,
        "coordinate_note": "plot uses horizontal=local_right_positive=-local_y, vertical=local_x_forward_positive",
        "fm_engine": str(args.fm_engine),
        "alpamayo_nav_cfg": bool(args.alpamayo_nav_cfg),
        "nav_guidance_weight": float(args.nav_guidance_weight),
        "output_png": str(out_png),
        "input_images": {name: str(path) for name, path in image_paths.items()},
        "variants": {
            variant.name: {
                "label": variant.label,
                "nav_text": variant.nav_text,
                **summarize_variant(summaries[variant.name], no_nav_summary, gt_summary),
                "artifact_dir": str(artifact_dir / variant.name),
                "request_path": str(request_paths[variant.name]),
                "output_path": str(output_paths[variant.name]),
            }
            for variant in VARIANTS
        },
        "gt": summarize_variant(gt_summary, no_nav_summary, gt_summary),
    }
    write_json(work_root / f"{stem}_nav_strength_stats.json", stats)
    write_json(work_root / "run_summary.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
