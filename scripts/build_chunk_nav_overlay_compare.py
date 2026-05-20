#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine
from scripts.build_live_chunk_request_bank import (
    ecef_to_enu,
    geodetic_to_ecef,
    select_pose_gnss_rows,
    yaw_to_rot,
)
from scripts.run_raw_dataset_one_shot_udp import build_result_artifacts
from scripts.run_request_bank_persistent import build_env, read_status


DEFAULT_NAV_TEXT = (
    "Continue straight in the current lane. Keep lane and do not change lanes."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one live chunk with and without nav text at a fixed sample interval, "
            "then overlay GNSS GT and predicted world trajectories into one PNG."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2"),
    )
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument(
        "--request-bank-root",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2/chunk0000/request_bank"
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "output" / "reports" / "chunk0000_nav_overlay_compare",
    )
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--nav-text", type=str, default=DEFAULT_NAV_TEXT)
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0)
    parser.add_argument(
        "--alpamayo-nav-cfg",
        action="store_true",
        help="Enable dual-branch nav CFG when nav text is present.",
    )
    parser.add_argument(
        "--llm-inference-bin",
        type=Path,
        default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference",
    )
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"),
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=first_existing_fm_engine())
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument(
        "--reuse-existing-outputs",
        action="store_true",
        help="Skip requests whose output JSON already exists in the work root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def chunk_request_summary(request_bank_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(request_bank_root / "manifest.json")
    summary = load_json(request_bank_root / "summary.json")
    return manifest, summary


def chunk_samples(dataset_root: Path, chunk_id: int) -> pd.DataFrame:
    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[
        ["frame_id", "chunk_id"]
    ]
    sample_index = sample_index.merge(
        front_frames, left_on="front_frame_id", right_on="frame_id", how="left"
    )
    samples = sample_index[sample_index["chunk_id"] == chunk_id].copy().sort_values("t0_utc_ns").reset_index(drop=True)
    if samples.empty:
        raise RuntimeError(f"No sample rows found for chunk {chunk_id}")
    return samples


def select_manifest_rows(
    manifest_rows: list[dict[str, Any]], dt_s: float, sample_interval_s: float
) -> list[dict[str, Any]]:
    stride = max(1, int(round(sample_interval_s / dt_s)))
    return manifest_rows[::stride]


def variant_request_root(work_root: Path, variant_name: str) -> Path:
    return work_root / "variants" / variant_name / "request_bank"


def variant_output_root(work_root: Path, variant_name: str) -> Path:
    return work_root / "variants" / variant_name / "outputs"


def variant_artifact_root(work_root: Path, variant_name: str) -> Path:
    return work_root / "variants" / variant_name / "artifacts"


def prepare_variant_requests(
    *,
    selected_rows: list[dict[str, Any]],
    out_root: Path,
    nav_text: str | None,
    nav_guidance_weight: float,
) -> list[Path]:
    request_root = out_root / "requests"
    request_root.mkdir(parents=True, exist_ok=True)
    request_paths: list[Path] = []
    for row in selected_rows:
        src = Path(row["request_json"])
        request_obj = load_json(src)
        request = request_obj["requests"][0]
        request.pop("nav_text", None)
        request.pop("nav_guidance_weight", None)
        if nav_text:
            request["nav_text"] = nav_text
            request["nav_guidance_weight"] = float(nav_guidance_weight)
        dst = request_root / src.name
        write_json(dst, request_obj)
        request_paths.append(dst)
    return request_paths


def llm_runtime_cmd(args: argparse.Namespace) -> list[str]:
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


def run_variant_outputs(
    *,
    args: argparse.Namespace,
    variant_name: str,
    request_paths: list[Path],
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = llm_runtime_cmd(args)
    print(f"[nav-overlay] starting runtime for {variant_name}", flush=True)
    print("[nav-overlay] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )
    output_paths: list[Path] = []
    try:
        ready = read_status(proc, timeout_s=120.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected llm_inference ready state: {ready}")
        print(f"[nav-overlay] runtime ready for {variant_name}", flush=True)

        for idx, request_path in enumerate(request_paths, start=1):
            output_path = output_root / request_path.name.replace("request_", "output_")
            output_paths.append(output_path)
            if args.reuse_existing_outputs and output_path.exists():
                print(
                    f"[nav-overlay] reuse {variant_name} {idx}/{len(request_paths)} {output_path.name}",
                    flush=True,
                )
                continue

            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            t0 = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=args.timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"llm_inference request failed for {request_path.name}: {status}")
            print(
                f"[nav-overlay] {variant_name} {idx}/{len(request_paths)} "
                f"{request_path.name} done in {time.time() - t0:.2f}s",
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


def build_metadata_lookup(
    *,
    manifest_rows: list[dict[str, Any]],
    samples: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    samples_by_sample_id = {
        int(row.sample_id): {
            "front_frame_id": int(row.front_frame_id),
            "t0_utc_ns": int(row.t0_utc_ns),
        }
        for row in samples.itertuples(index=False)
    }
    chunk_start_utc_ns = int(samples["t0_utc_ns"].iloc[0])
    lookup: dict[int, dict[str, Any]] = {}
    for row in manifest_rows:
        sample_id = int(row["sample_id"])
        sample_meta = samples_by_sample_id[sample_id]
        t0_utc_ns = int(row["t0_utc_ns"])
        lookup[sample_id] = {
            "chunk_id": int(row["chunk_id"]),
            "sample_id": sample_id,
            "front_frame_id": int(sample_meta["front_frame_id"]),
            "t0_utc_ns": t0_utc_ns,
            "t0_us": int(row["t0_us"]),
            "chunk_start_utc_ns": chunk_start_utc_ns,
            "target_offset_s": float(t0_utc_ns - chunk_start_utc_ns) / 1e9,
            "actual_offset_s": float(t0_utc_ns - chunk_start_utc_ns) / 1e9,
            "request_json": str(row["request_json"]),
            "ego_history_xyz_npy": str(row["ego_history_xyz_npy"]),
            "ego_history_rot_npy": str(row["ego_history_rot_npy"]),
            "selected_frames": row["selected_frames"],
        }
    return lookup


def parse_sample_id_from_request_path(path: Path) -> int:
    name = path.stem
    sid_part = next(part for part in name.split("_") if part.startswith("sid"))
    return int(sid_part.replace("sid", ""))


def load_gnss_world(
    dataset_root: Path, ref_lla: tuple[float, float, float]
) -> pd.DataFrame:
    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = select_pose_gnss_rows(gnss)
    xyz = ecef_to_enu(
        geodetic_to_ecef(
            gnss_valid["lat"].to_numpy(dtype=np.float64),
            gnss_valid["lon"].to_numpy(dtype=np.float64),
            gnss_valid["alt"].to_numpy(dtype=np.float64),
        ),
        *ref_lla,
    ).astype(np.float32)
    gnss_valid = gnss_valid.copy()
    gnss_valid["enu_x_m"] = xyz[:, 0]
    gnss_valid["enu_y_m"] = xyz[:, 1]
    return gnss_valid


def build_world_pose_at_t0(
    *,
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    t0_utc_ns: int,
    history_len: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt_ns = int(round(dt_s * 1e9))
    hist_times = np.asarray(
        [t0_utc_ns - (history_len - 1 - i) * dt_ns for i in range(history_len)],
        dtype=np.int64,
    )
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    interp_lat = np.interp(hist_times, utc, lat)
    interp_lon = np.interp(hist_times, utc, lon)
    interp_alt = np.interp(hist_times, utc, alt)
    world_xyz = ecef_to_enu(
        geodetic_to_ecef(interp_lat, interp_lon, interp_alt),
        *ref_lla,
    ).astype(np.float32)

    vel_xy = np.zeros((len(world_xyz), 2), dtype=np.float32)
    if len(world_xyz) > 2:
        vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_ns / 1e9)
    if len(world_xyz) > 1:
        vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_ns / 1e9)
        vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_ns / 1e9)
    yaw = np.arctan2(vel_xy[:, 1], vel_xy[:, 0]).astype(np.float32)
    world_rot = yaw_to_rot(yaw)
    return world_xyz[-1], world_rot[-1]


def local_packet_points_to_world(
    packet_points: list[dict[str, Any]], world_pos: np.ndarray, world_rot: np.ndarray
) -> np.ndarray:
    local_xyz = np.asarray(
        [[float(pt["x_m"]), float(pt["y_m"]), 0.0] for pt in packet_points],
        dtype=np.float32,
    )
    return world_pos[None, :] + local_xyz @ world_rot.T


def gt_chunk_path(
    gnss_valid: pd.DataFrame,
    *,
    chunk_start_utc_ns: int,
    chunk_end_utc_ns: int,
    horizon_dt_s: float,
    horizon_points: int,
) -> np.ndarray:
    end_ns = chunk_end_utc_ns + int(round(horizon_dt_s * horizon_points * 1e9))
    mask = (
        (gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64) >= chunk_start_utc_ns)
        & (gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64) <= end_ns)
    )
    rows = gnss_valid.loc[mask, ["enu_x_m", "enu_y_m"]].to_numpy(dtype=np.float32)
    if len(rows) < 2:
        raise RuntimeError("Not enough GNSS rows for GT chunk path overlay")
    return rows


def build_variant_world_runs(
    *,
    args: argparse.Namespace,
    variant_name: str,
    request_paths: list[Path],
    metadata_lookup: dict[int, dict[str, Any]],
    dataset_root: Path,
    history_len: int,
    dt_s: float,
    gnss_valid: pd.DataFrame,
    ref_lla: tuple[float, float, float],
    work_root: Path,
) -> list[dict[str, Any]]:
    output_root = variant_output_root(work_root, variant_name)
    artifact_root = variant_artifact_root(work_root, variant_name)
    outputs = run_variant_outputs(
        args=args,
        variant_name=variant_name,
        request_paths=request_paths,
        output_root=output_root,
    )
    runs: list[dict[str, Any]] = []
    for request_path, output_path in zip(request_paths, outputs, strict=True):
        sample_id = parse_sample_id_from_request_path(request_path)
        metadata = dict(metadata_lookup[sample_id])
        per_request_artifact_root = artifact_root / request_path.stem
        final_summary, _, _, _ = build_result_artifacts(
            output_path=output_path,
            metadata=metadata,
            dataset_root=dataset_root,
            history_len=history_len,
            artifact_root=per_request_artifact_root,
        )
        world_pos, world_rot = build_world_pose_at_t0(
            gnss_valid=gnss_valid,
            ref_lla=ref_lla,
            t0_utc_ns=int(metadata["t0_utc_ns"]),
            history_len=history_len,
            dt_s=dt_s,
        )
        world_pts = local_packet_points_to_world(
            final_summary["packet_points"], world_pos, world_rot
        )
        runs.append(
            {
                "sample_id": sample_id,
                "t0_utc_ns": int(metadata["t0_utc_ns"]),
                "offset_s": float(metadata["actual_offset_s"]),
                "final_output": str(final_summary.get("final_output", "")),
                "timing": final_summary.get("timing", {}),
                "world_pts": world_pts,
            }
        )
    return runs


def plot_overlay(
    *,
    out_path: Path,
    gt_world_xy: np.ndarray,
    no_nav_runs: list[dict[str, Any]],
    nav_runs: list[dict[str, Any]],
    nav_text: str,
    nav_cfg_enabled: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 11), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    # Keep GNSS as context in the background so the prediction families stay readable.
    ax.plot(
        gt_world_xy[:, 0],
        gt_world_xy[:, 1],
        color="#64748b",
        alpha=0.34,
        linewidth=8.0,
        solid_capstyle="round",
        label="GT GNSS",
        zorder=1,
    )
    ax.plot(
        gt_world_xy[:, 0],
        gt_world_xy[:, 1],
        color="#334155",
        alpha=0.42,
        linewidth=1.2,
        solid_capstyle="round",
        zorder=2,
    )

    for run in no_nav_runs:
        pts = run["world_pts"]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#2563eb",
            alpha=0.58,
            linewidth=1.65,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(pts[0, 0], pts[0, 1], color="#1d4ed8", alpha=0.55, s=10, zorder=5)

    for run in nav_runs:
        pts = run["world_pts"]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#dc2626",
            alpha=0.62,
            linewidth=1.65,
            linestyle=(0, (4.5, 2.5)),
            solid_capstyle="round",
            zorder=6,
        )
        ax.scatter(pts[0, 0], pts[0, 1], color="#b91c1c", alpha=0.62, s=10, zorder=7)

    handles = [
        Line2D([0], [0], color="#64748b", linewidth=7.0, alpha=0.45, label="GT GNSS"),
        Line2D([0], [0], color="#2563eb", linewidth=2.4, alpha=0.95, label="No Nav predictions"),
        Line2D(
            [0],
            [0],
            color="#dc2626",
            linewidth=2.4,
            linestyle=(0, (4.5, 2.5)),
            alpha=0.95,
            label="Nav predictions",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    all_points = [gt_world_xy]
    all_points.extend([run["world_pts"][:, :2] for run in no_nav_runs])
    all_points.extend([run["world_pts"][:, :2] for run in nav_runs])
    stack = np.concatenate(all_points, axis=0)
    x_min, y_min = stack.min(axis=0)
    x_max, y_max = stack.max(axis=0)
    pad_x = max(2.0, float(x_max - x_min) * 0.06)
    pad_y = max(2.0, float(y_max - y_min) * 0.06)
    ax.set_xlim(float(x_min - pad_x), float(x_max + pad_x))
    ax.set_ylim(float(y_min - pad_y), float(y_max + pad_y))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("ENU X / East [m]")
    ax.set_ylabel("ENU Y / North [m]")

    mode_text = "nav_cfg" if nav_cfg_enabled else "nav_text_only"
    title = (
        "Chunk Trajectory Overlay: GT vs No-Nav vs Nav\n"
        f"nav mode={mode_text}, nav text=\"{nav_text}\""
    )
    ax.set_title(title, fontsize=12)

    text_lines = [
        f"no_nav runs: {len(no_nav_runs)}",
        f"nav runs: {len(nav_runs)}",
    ]
    if no_nav_runs:
        no_nav_ms = [float(run["timing"].get("total_post_vlm_ms", np.nan)) for run in no_nav_runs]
        text_lines.append(f"no_nav avg total_post_vlm_ms: {np.nanmean(no_nav_ms):.1f}")
    if nav_runs:
        nav_ms = [float(run["timing"].get("total_post_vlm_ms", np.nan)) for run in nav_runs]
        text_lines.append(f"nav avg total_post_vlm_ms: {np.nanmean(nav_ms):.1f}")
    ax.text(
        0.015,
        0.985,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)

    manifest_rows, request_bank_summary = chunk_request_summary(args.request_bank_root)
    samples = chunk_samples(args.dataset_root, args.chunk_id)
    dt_s = float(request_bank_summary["dt_s"])
    history_len = int(request_bank_summary["history_len"])
    ref_lla = tuple(float(x) for x in request_bank_summary["chunk_ref_lla"])

    selected_rows = select_manifest_rows(manifest_rows, dt_s, args.sample_interval_s)
    metadata_lookup = build_metadata_lookup(manifest_rows=manifest_rows, samples=samples)

    no_nav_request_root = variant_request_root(args.work_root, "no_nav")
    nav_request_root = variant_request_root(args.work_root, "nav")
    no_nav_request_paths = prepare_variant_requests(
        selected_rows=selected_rows,
        out_root=no_nav_request_root,
        nav_text=None,
        nav_guidance_weight=args.nav_guidance_weight,
    )
    nav_request_paths = prepare_variant_requests(
        selected_rows=selected_rows,
        out_root=nav_request_root,
        nav_text=args.nav_text,
        nav_guidance_weight=args.nav_guidance_weight,
    )

    gnss_valid = load_gnss_world(args.dataset_root, ref_lla)
    no_nav_runs = build_variant_world_runs(
        args=args,
        variant_name="no_nav",
        request_paths=no_nav_request_paths,
        metadata_lookup=metadata_lookup,
        dataset_root=args.dataset_root,
        history_len=history_len,
        dt_s=dt_s,
        gnss_valid=gnss_valid,
        ref_lla=ref_lla,
        work_root=args.work_root,
    )
    nav_runs = build_variant_world_runs(
        args=args,
        variant_name="nav",
        request_paths=nav_request_paths,
        metadata_lookup=metadata_lookup,
        dataset_root=args.dataset_root,
        history_len=history_len,
        dt_s=dt_s,
        gnss_valid=gnss_valid,
        ref_lla=ref_lla,
        work_root=args.work_root,
    )

    chunk_start_utc_ns = int(samples["t0_utc_ns"].iloc[0])
    chunk_end_utc_ns = int(samples["t0_utc_ns"].iloc[-1])
    gt_world_xy = gt_chunk_path(
        gnss_valid,
        chunk_start_utc_ns=chunk_start_utc_ns,
        chunk_end_utc_ns=chunk_end_utc_ns,
        horizon_dt_s=dt_s,
        horizon_points=64,
    )

    out_png = args.work_root / "nav_vs_no_nav_overlay.png"
    plot_overlay(
        out_path=out_png,
        gt_world_xy=gt_world_xy,
        no_nav_runs=no_nav_runs,
        nav_runs=nav_runs,
        nav_text=args.nav_text,
        nav_cfg_enabled=bool(args.alpamayo_nav_cfg),
    )

    summary = {
        "dataset_root": str(args.dataset_root),
        "chunk_id": int(args.chunk_id),
        "request_bank_root": str(args.request_bank_root),
        "fm_engine": str(args.fm_engine),
        "engine_dir": str(args.engine_dir),
        "multimodal_engine_dir": str(args.multimodal_engine_dir),
        "sample_interval_s": float(args.sample_interval_s),
        "selected_request_count": int(len(selected_rows)),
        "nav_text": args.nav_text,
        "alpamayo_nav_cfg": bool(args.alpamayo_nav_cfg),
        "output_png": str(out_png),
        "no_nav_output_root": str(variant_output_root(args.work_root, "no_nav")),
        "nav_output_root": str(variant_output_root(args.work_root, "nav")),
    }
    write_json(args.work_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
