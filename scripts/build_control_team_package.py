#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_live_chunk_model_timeline as timeline
from control_team_replay_common import (
    COORD_MODE_LOCAL,
    build_packet_dict,
    pack_packet,
    sample_plan_points,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.float32).reshape(tuple(field["shape"]))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build control-team UDP replay artifacts from chunk model outputs.")
    parser.add_argument("--dataset-root", default="/root/live_dataset/2025-03-31-test2")
    parser.add_argument("--chunk-id", type=int, default=11)
    parser.add_argument("--request-bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--viewer-dir",
        type=Path,
        default=Path("/root/TensorRT-Edge-LLM-v060/output/dashboards/real_rutuime/chunk0011_model_timeline_replay"),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("/root/TensorRT-Edge-LLM-v060/output/deliverables/control_team/chunk0011_udp_replay_package"),
    )
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    parser.add_argument("--preview-packets", type=int, default=8)
    parser.add_argument("--default-host", default="127.0.0.1")
    parser.add_argument("--default-port", type=int, default=5001)
    args = parser.parse_args()

    ensure_dir(args.package_dir)

    manifest = json.loads((args.request_bank_root / "manifest.json").read_text())
    manifest_by_sample = {int(item["sample_id"]): item for item in manifest}
    summary = json.loads((args.request_bank_root / "summary.json").read_text())
    ref_lat, ref_lon, ref_alt = summary["chunk_ref_lla"]

    gnss = pd.read_parquet(Path(args.dataset_root) / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    output_paths = sorted(args.output_root.glob(f"output_chunk{args.chunk_id:04d}_sid*_t0_*.json"))
    if not output_paths:
        raise RuntimeError(f"No output files found in {args.output_root}")

    sample_ids: list[int] = []
    t0_us_list: list[int] = []
    t_rel_s_list: list[float] = []
    ref_x_world: list[float] = []
    ref_y_world: list[float] = []
    ref_yaw_world: list[float] = []
    traj_x_local: list[np.ndarray] = []
    traj_y_local: list[np.ndarray] = []
    traj_yaw_local: list[np.ndarray] = []
    traj_v_mps: list[np.ndarray] = []
    traj_curvature: list[np.ndarray] = []
    history_x_local: list[np.ndarray] = []
    history_y_local: list[np.ndarray] = []
    output_texts: list[str] = []
    image_stems: list[str] = []

    first_t0_ns = int(manifest[0]["t0_utc_ns"])
    plan_dt_s: float | None = None
    plan_points: int | None = None

    for output_path in output_paths:
        chunk_id, sample_id, t0_us = timeline.parse_identity(output_path)
        meta = manifest_by_sample.get(sample_id)
        if meta is None:
            continue

        t0_utc_ns = int(meta["t0_utc_ns"])
        t0_lat = float(np.interp([t0_utc_ns], utc, lat)[0])
        t0_lon = float(np.interp([t0_utc_ns], utc, lon)[0])
        t0_alt = float(np.interp([t0_utc_ns], utc, alt)[0])
        t0_world = timeline.ecef_to_enu(
            timeline.geodetic_to_ecef(np.asarray([t0_lat]), np.asarray([t0_lon]), np.asarray([t0_alt])),
            ref_lat,
            ref_lon,
            ref_alt,
        )[0].astype(np.float32)

        yaw_times = np.asarray([t0_utc_ns - 100_000_000, t0_utc_ns, t0_utc_ns + 100_000_000], dtype=np.int64)
        yaw_lat = np.interp(yaw_times, utc, lat)
        yaw_lon = np.interp(yaw_times, utc, lon)
        yaw_alt = np.interp(yaw_times, utc, alt)
        yaw_world_pts = timeline.ecef_to_enu(
            timeline.geodetic_to_ecef(yaw_lat, yaw_lon, yaw_alt),
            ref_lat,
            ref_lon,
            ref_alt,
        )
        yaw_vec = yaw_world_pts[2, :2] - yaw_world_pts[0, :2]
        t0_yaw = float(np.arctan2(yaw_vec[1], yaw_vec[0]))

        req_path = Path(meta["request_json"])
        req_obj = json.loads(req_path.read_text())
        req_item = req_obj["requests"][0]
        hist_xyz = np.load(req_item["ego_history_xyz_npy"]).astype(np.float32)[0, 0]

        output = json.loads(output_path.read_text())["responses"][0]
        fm = output["alpamayo_post_vlm"]["fm"]
        pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
        pred_rot = reshape_tensor_field(fm["pred_rot"])[0]
        this_plan_dt_s = float(fm["action_space_constants"]["dt_value"])
        this_plan_points = int(pred_xyz.shape[0])
        if plan_dt_s is None:
            plan_dt_s = this_plan_dt_s
            plan_points = this_plan_points
        elif abs(plan_dt_s - this_plan_dt_s) > 1e-6 or plan_points != this_plan_points:
            raise RuntimeError("Inconsistent plan horizon across outputs")

        yaw_local = yaw_from_rot(pred_rot)
        yaw_unwrapped = np.unwrap(yaw_local.astype(np.float64)).astype(np.float32)

        v_mps = np.zeros(pred_xyz.shape[0], dtype=np.float32)
        v_mps[0] = float(np.linalg.norm(pred_xyz[0, :2]) / max(this_plan_dt_s, 1e-6))
        if pred_xyz.shape[0] > 1:
            deltas = pred_xyz[1:, :2] - pred_xyz[:-1, :2]
            v_mps[1:] = np.linalg.norm(deltas, axis=1) / max(this_plan_dt_s, 1e-6)

        curvature = np.zeros(pred_xyz.shape[0], dtype=np.float32)
        if pred_xyz.shape[0] > 2:
            ds = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1)
            dyaw = np.diff(yaw_unwrapped)
            curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
            curvature[0] = curvature[1]

        sample_ids.append(sample_id)
        t0_us_list.append(int(t0_us))
        t_rel_s_list.append(float((t0_utc_ns - first_t0_ns) / 1e9))
        ref_x_world.append(float(t0_world[0]))
        ref_y_world.append(float(t0_world[1]))
        ref_yaw_world.append(t0_yaw)
        traj_x_local.append(pred_xyz[:, 0].astype(np.float32))
        traj_y_local.append(pred_xyz[:, 1].astype(np.float32))
        traj_yaw_local.append(wrap_angles(yaw_unwrapped))
        traj_v_mps.append(v_mps)
        traj_curvature.append(curvature)
        history_x_local.append(hist_xyz[:, 0].astype(np.float32))
        history_y_local.append(hist_xyz[:, 1].astype(np.float32))
        output_texts.append(output.get("output_text") or "")
        image_stems.append(req_path.stem.replace("request_", ""))

    if plan_dt_s is None or plan_points is None:
        raise RuntimeError("No valid outputs were converted")

    max_text_len = max(max((len(x) for x in output_texts), default=1), 1)
    max_image_stem_len = max(max((len(x) for x in image_stems), default=1), 1)

    plan_bank_path = args.package_dir / f"chunk{args.chunk_id:04d}_model_plan_bank.npz"
    np.savez_compressed(
        plan_bank_path,
        chunk_id=np.int32(args.chunk_id),
        sample_id=np.asarray(sample_ids, dtype=np.int32),
        t0_us=np.asarray(t0_us_list, dtype=np.int64),
        t_rel_s=np.asarray(t_rel_s_list, dtype=np.float32),
        plan_dt_s=np.float32(plan_dt_s),
        plan_points=np.int32(plan_points),
        coord_mode_default=np.int32(COORD_MODE_LOCAL),
        control_dt_default_s=np.float32(args.control_dt),
        control_points_default=np.int32(args.control_points),
        traj_x_local=np.stack(traj_x_local, axis=0).astype(np.float32),
        traj_y_local=np.stack(traj_y_local, axis=0).astype(np.float32),
        traj_yaw_local=np.stack(traj_yaw_local, axis=0).astype(np.float32),
        traj_v_mps=np.stack(traj_v_mps, axis=0).astype(np.float32),
        traj_curvature=np.stack(traj_curvature, axis=0).astype(np.float32),
        history_x_local=np.stack(history_x_local, axis=0).astype(np.float32),
        history_y_local=np.stack(history_y_local, axis=0).astype(np.float32),
        ref_x_world_enu=np.asarray(ref_x_world, dtype=np.float32),
        ref_y_world_enu=np.asarray(ref_y_world, dtype=np.float32),
        ref_yaw_world_rad=np.asarray(ref_yaw_world, dtype=np.float32),
        output_text=np.asarray(output_texts, dtype=f"<U{max_text_len}"),
        image_stem=np.asarray(image_stems, dtype=f"<U{max_image_stem_len}"),
    )

    sys.path.insert(0, str(SCRIPT_DIR))
    from control_team_replay_common import load_plan_bank  # local import to reuse saved file

    bank = load_plan_bank(str(plan_bank_path))
    preview_packets = []
    preview_dir = args.package_dir / "packet_previews"
    ensure_dir(preview_dir)
    preview_count = min(args.preview_packets, len(bank.sample_id))
    base_tx_time_us = int(time.time_ns() // 1000)
    for tx_seq in range(preview_count):
        plan_idx = tx_seq
        plan = sample_plan_points(
            bank=bank,
            plan_idx=plan_idx,
            age_s=0.0,
            control_dt_s=args.control_dt,
            control_points=args.control_points,
            coord_mode=COORD_MODE_LOCAL,
        )
        packet = build_packet_dict(
            tx_seq=tx_seq,
            plan_seq=plan_idx,
            sample_id=int(bank.sample_id[plan_idx]),
            source_t0_us=int(bank.t0_us[plan_idx]),
            tx_time_us=base_tx_time_us + int(tx_seq * args.control_dt * 1e6),
            coord_mode=COORD_MODE_LOCAL,
            dt_s=args.control_dt,
            x=plan["x"],
            y=plan["y"],
            yaw=plan["yaw"],
            v=plan["v"],
            curvature=plan["curvature"],
        )
        packet_bytes = pack_packet(packet)
        (preview_dir / f"packet_{tx_seq:04d}.bin").write_bytes(packet_bytes)
        preview_packets.append(packet)

    preview_json_path = args.package_dir / "packet_preview_first_packets.json"
    preview_json_path.write_text(json.dumps(preview_packets, indent=2))

    summary_out = {
        "chunk_id": args.chunk_id,
        "num_samples": len(sample_ids),
        "plan_dt_s": plan_dt_s,
        "plan_points": plan_points,
        "plan_horizon_s": float(plan_dt_s * max(plan_points - 1, 0)),
        "control_dt_default_s": args.control_dt,
        "control_points_default": args.control_points,
        "control_horizon_s": float(args.control_dt * args.control_points),
        "coord_mode_default": "local",
        "plan_bank_npz": str(plan_bank_path),
        "preview_json": str(preview_json_path),
        "viewer_dir": str(args.viewer_dir),
        "viewer_html": str(args.viewer_dir / f"chunk{args.chunk_id:04d}_model_timeline_viewer.html"),
    }
    (args.package_dir / "summary.json").write_text(json.dumps(summary_out, indent=2))

    tools_dir = args.package_dir / "tools"
    ensure_dir(tools_dir)
    shutil.copy2(SCRIPT_DIR / "udp_replay_player.py", tools_dir / "udp_replay_player.py")
    shutil.copy2(SCRIPT_DIR / "control_team_replay_common.py", tools_dir / "control_team_replay_common.py")

    package_viewer_dir = args.package_dir / "viewer"
    ensure_dir(package_viewer_dir)
    for name in [
        f"chunk{args.chunk_id:04d}_model_timeline_viewer.html",
        f"chunk{args.chunk_id:04d}_model_viewer_data.json",
        f"chunk{args.chunk_id:04d}_model_timeline_overview.png",
        "summary.json",
    ]:
        shutil.copy2(args.viewer_dir / name, package_viewer_dir / name)
    safe_symlink(args.request_bank_root / "images", package_viewer_dir / "images_bank")

    run_script = args.package_dir / "run_localhost_replay.sh"
    write_text(
        run_script,
        f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

python "${{SCRIPT_DIR}}/tools/udp_replay_player.py" \\
  --plan-bank "${{SCRIPT_DIR}}/{plan_bank_path.name}" \\
  --host "{args.default_host}" \\
  --port {args.default_port} \\
  --control-dt {args.control_dt} \\
  --control-points {args.control_points} \\
  --coord-mode local \\
  "$@"
""",
    )
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    write_text(
        args.package_dir / "PACKET_SPEC.md",
        f"""# UDP Packet Spec

This package replays the model output as a compact UDP packet intended for control integration.

## Byte order
- Little-endian

## Header layout
- `magic`: `4s` = `ALPA`
- `version`: `uint16`
- `flags`: `uint16`
- `tx_seq`: `uint32`
- `plan_seq`: `uint32`
- `sample_id`: `uint32`
- `source_t0_us`: `uint64`
- `tx_time_us`: `uint64`
- `coord_mode`: `uint16`
  - `0 = local`
  - `1 = world`
- `num_points`: `uint16`
- `dt_s`: `float32`

Header size: `44 bytes`

## Point layout
Repeated `num_points` times:
- `x_m`: `float32`
- `y_m`: `float32`
- `yaw_rad`: `float32`
- `v_mps`: `float32`
- `curvature`: `float32`

Point size: `20 bytes`

## Trailer
- `crc32`: `uint32`
  - computed over `header + point payload`

## Current default replay settings
- `coord_mode = local`
- `dt_s = {args.control_dt:.3f}`
- `num_points = {args.control_points}`
- `control_horizon = {args.control_dt * args.control_points:.3f} s`

## C reference struct
```c
#pragma pack(push, 1)
typedef struct {{
  char magic[4];
  uint16_t version;
  uint16_t flags;
  uint32_t tx_seq;
  uint32_t plan_seq;
  uint32_t sample_id;
  uint64_t source_t0_us;
  uint64_t tx_time_us;
  uint16_t coord_mode;
  uint16_t num_points;
  float dt_s;
}} ControlPacketHeader;

typedef struct {{
  float x_m;
  float y_m;
  float yaw_rad;
  float v_mps;
  float curvature;
}} ControlPoint;
#pragma pack(pop)
```
""",
    )

    write_text(
        args.package_dir / "README.md",
        f"""# Control Team UDP Replay Package

This package contains the artifacts needed to replay the `chunk_{args.chunk_id:04d}` model output toward the control stack without requiring live Thor-to-control-PC UDP.

## What is included
- `chunk{args.chunk_id:04d}_model_plan_bank.npz`
  - planner output bank derived from actual model inference over all `{len(sample_ids)}` samples
- `packet_previews/`
  - first few packets encoded as binary `.bin`
- `packet_preview_first_packets.json`
  - decoded preview of those packets
- `PACKET_SPEC.md`
  - byte-level UDP format
- `run_localhost_replay.sh`
  - quick launcher for localhost replay
- `viewer/`
  - HTML replay viewer for visual inspection
  - note: `viewer/images_bank` is a symlink to the request-bank images to avoid duplicating the cached PNG set
- `tools/udp_replay_player.py`
  - configurable UDP replay sender

## Model/runtime used
- LLM: `/alpamayo_vlm_engines/alpa1.5/llm.engine`
- ViT: `/alpamayo_vlm_engines/alpa1.5_visual_fp8_rebuild`
- FM: `/root/test/output/alpamayo15_fm_one_step_mxfp8/alpamayo15_fm_one_step_mxfp8_thor.plan`

## Coordinate frame
- Default replay packet uses `local` ego frame
- The underlying NPZ also contains `ref_x_world_enu`, `ref_y_world_enu`, `ref_yaw_world_rad`, so the replay player can also emit `world` packets if needed

## NPZ schema
- `sample_id [N]`
- `t0_us [N]`
- `t_rel_s [N]`
- `plan_dt_s`
- `plan_points`
- `traj_x_local [N, T]`
- `traj_y_local [N, T]`
- `traj_yaw_local [N, T]`
- `traj_v_mps [N, T]`
- `traj_curvature [N, T]`
- `history_x_local [N, H]`
- `history_y_local [N, H]`
- `ref_x_world_enu [N]`
- `ref_y_world_enu [N]`
- `ref_yaw_world_rad [N]`

## Quick start
```bash
cd "{args.package_dir}"
./run_localhost_replay.sh
```

Equivalent direct command:
```bash
python "./tools/udp_replay_player.py" \\
  --plan-bank "./{plan_bank_path.name}" \\
  --host {args.default_host} \\
  --port {args.default_port} \\
  --control-dt {args.control_dt} \\
  --control-points {args.control_points} \\
  --coord-mode local
```

## Tuning knobs
The replay packet shape is not fixed by the model. These values can be changed at replay time:
- `--control-dt`
- `--control-points`
- `--coord-mode local|world`
- `--loop`
- `--playback-rate`

For example, `25 points @ 0.02s` gives a `0.5s` control horizon, while `50 points @ 0.02s` gives a `1.0s` control horizon.

## Visual inspection
If the viewer is being served already:
- `http://localhost:8769/chunk{args.chunk_id:04d}_model_timeline_viewer.html`

Or open the files under `viewer/`.
""",
    )

    print(
        json.dumps(
            {
                "package_dir": str(args.package_dir),
                "plan_bank_npz": str(plan_bank_path),
                "packet_preview_json": str(preview_json_path),
                "viewer_dir": str(args.package_dir / "viewer"),
                "summary_json": str(args.package_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
