#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tarfile
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
    load_plan_bank,
    pack_packet,
    sample_plan_points,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.float32).reshape(tuple(field["shape"]))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a control-team UDP replay package from multiple chunks.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chunk-ids", nargs="+", type=int, required=True)
    parser.add_argument("--request-bank-roots", nargs="+", required=True)
    parser.add_argument("--output-roots", nargs="+", required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    parser.add_argument("--preview-packets", type=int, default=8)
    parser.add_argument("--default-host", default="127.0.0.1")
    parser.add_argument("--default-port", type=int, default=5001)
    parser.add_argument("--package-name", default="control_team_multichunk")
    args = parser.parse_args()

    if not (len(args.chunk_ids) == len(args.request_bank_roots) == len(args.output_roots)):
        raise ValueError("--chunk-ids, --request-bank-roots, --output-roots must have same length")

    dataset_root = Path(args.dataset_root)
    request_bank_roots = [Path(x) for x in args.request_bank_roots]
    output_roots = [Path(x) for x in args.output_roots]

    if args.package_dir.exists():
        shutil.rmtree(args.package_dir)
    ensure_dir(args.package_dir)

    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    manifests: list[list[dict[str, Any]]] = []
    first_t0_ns: int | None = None
    for rb in request_bank_roots:
        manifest = json.loads((rb / "manifest.json").read_text())
        manifests.append(manifest)
        local_first = int(manifest[0]["t0_utc_ns"])
        first_t0_ns = local_first if first_t0_ns is None else min(first_t0_ns, local_first)
    assert first_t0_ns is not None
    global_ref_lla = (
        float(np.interp([first_t0_ns], utc, lat)[0]),
        float(np.interp([first_t0_ns], utc, lon)[0]),
        float(np.interp([first_t0_ns], utc, alt)[0]),
    )

    sample_ids: list[int] = []
    chunk_ids_per_sample: list[int] = []
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

    plan_dt_s: float | None = None
    plan_points: int | None = None

    counts: dict[str, int] = {}

    for chunk_id, rb_root, out_root, manifest in zip(args.chunk_ids, request_bank_roots, output_roots, manifests, strict=True):
        manifest_by_sample = {int(item["sample_id"]): item for item in manifest}
        output_paths = sorted(out_root.glob(f"output_chunk{chunk_id:04d}_sid*_t0_*.json"))
        counts[f"chunk{chunk_id:04d}"] = len(output_paths)
        if not output_paths:
            raise RuntimeError(f"No outputs found for chunk {chunk_id} in {out_root}")

        for output_path in output_paths:
            parsed_chunk_id, sample_id, t0_us = timeline.parse_identity(output_path)
            meta = manifest_by_sample.get(sample_id)
            if meta is None:
                continue
            t0_utc_ns = int(meta["t0_utc_ns"])

            t0_lat = float(np.interp([t0_utc_ns], utc, lat)[0])
            t0_lon = float(np.interp([t0_utc_ns], utc, lon)[0])
            t0_alt = float(np.interp([t0_utc_ns], utc, alt)[0])
            t0_world = timeline.ecef_to_enu(
                timeline.geodetic_to_ecef(np.asarray([t0_lat]), np.asarray([t0_lon]), np.asarray([t0_alt])),
                *global_ref_lla,
            )[0].astype(np.float32)

            yaw_times = np.asarray([t0_utc_ns - 100_000_000, t0_utc_ns, t0_utc_ns + 100_000_000], dtype=np.int64)
            yaw_lat = np.interp(yaw_times, utc, lat)
            yaw_lon = np.interp(yaw_times, utc, lon)
            yaw_alt = np.interp(yaw_times, utc, alt)
            yaw_world_pts = timeline.ecef_to_enu(
                timeline.geodetic_to_ecef(yaw_lat, yaw_lon, yaw_alt),
                *global_ref_lla,
            )
            yaw_vec = yaw_world_pts[2, :2] - yaw_world_pts[0, :2]
            t0_yaw = float(np.arctan2(yaw_vec[1], yaw_vec[0]))

            req_path = Path(meta["request_json"])
            req = json.loads(req_path.read_text())["requests"][0]
            hist_xyz = np.load(req["ego_history_xyz_npy"]).astype(np.float32)[0, 0]

            response = json.loads(output_path.read_text())["responses"][0]
            fm = response["alpamayo_post_vlm"]["fm"]
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
                v_mps[1:] = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1) / max(this_plan_dt_s, 1e-6)

            curvature = np.zeros(pred_xyz.shape[0], dtype=np.float32)
            if pred_xyz.shape[0] > 2:
                ds = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1)
                dyaw = np.diff(yaw_unwrapped)
                curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
                curvature[0] = curvature[1]

            sample_ids.append(sample_id)
            chunk_ids_per_sample.append(parsed_chunk_id)
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
            output_texts.append(response.get("output_text") or "")

    if plan_dt_s is None or plan_points is None:
        raise RuntimeError("No valid outputs were converted")

    max_text_len = max(max((len(x) for x in output_texts), default=1), 1)
    plan_bank_name = f"{args.package_name}_plan_bank.npz"
    plan_bank_path = args.package_dir / plan_bank_name
    np.savez_compressed(
        plan_bank_path,
        package_name=np.asarray(args.package_name),
        chunk_id=np.int32(-1),
        sample_id=np.asarray(sample_ids, dtype=np.int32),
        sample_chunk_id=np.asarray(chunk_ids_per_sample, dtype=np.int32),
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
    )

    tools_dir = args.package_dir / "tools"
    ensure_dir(tools_dir)
    shutil.copy2(SCRIPT_DIR / "udp_replay_player.py", tools_dir / "udp_replay_player.py")
    shutil.copy2(SCRIPT_DIR / "control_team_replay_common.py", tools_dir / "control_team_replay_common.py")

    bank = load_plan_bank(str(plan_bank_path))
    preview_dir = args.package_dir / "packet_previews"
    ensure_dir(preview_dir)
    preview_packets = []
    preview_count = min(args.preview_packets, len(bank.sample_id))
    base_tx_time_us = int(time.time_ns() // 1000)
    for tx_seq in range(preview_count):
        sampled = sample_plan_points(
            bank=bank,
            plan_idx=tx_seq,
            age_s=0.0,
            control_dt_s=args.control_dt,
            control_points=args.control_points,
            coord_mode=COORD_MODE_LOCAL,
        )
        packet = build_packet_dict(
            tx_seq=tx_seq,
            plan_seq=tx_seq,
            sample_id=int(bank.sample_id[tx_seq]),
            source_t0_us=int(bank.t0_us[tx_seq]),
            tx_time_us=base_tx_time_us + int(tx_seq * args.control_dt * 1e6),
            coord_mode=COORD_MODE_LOCAL,
            dt_s=args.control_dt,
            x=sampled["x"],
            y=sampled["y"],
            yaw=sampled["yaw"],
            v=sampled["v"],
            curvature=sampled["curvature"],
        )
        (preview_dir / f"packet_{tx_seq:04d}.bin").write_bytes(pack_packet(packet))
        preview_packets.append(packet)
    preview_json_path = args.package_dir / "packet_preview_first_packets.json"
    preview_json_path.write_text(json.dumps(preview_packets, indent=2), encoding="utf-8")

    run_script = args.package_dir / "run_localhost_replay.sh"
    write_text(
        run_script,
        f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

python "${{SCRIPT_DIR}}/tools/udp_replay_player.py" \\
  --plan-bank "${{SCRIPT_DIR}}/{plan_bank_name}" \\
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
""",
    )

    write_text(
        args.package_dir / "README.md",
        f"""# Control Team UDP Replay Package

This package is intended for the control team. It excludes HTML viewers and image assets and focuses on replaying the actual model output as UDP packets.

## Included chunks
- {", ".join(f"chunk{cid:04d}" for cid in args.chunk_ids)}

## What is included
- `{plan_bank_name}`
  - combined planner output bank derived from actual model inference over all included chunks
- `packet_previews/`
  - first few packets encoded as binary `.bin`
- `packet_preview_first_packets.json`
  - decoded preview of those packets
- `PACKET_SPEC.md`
  - byte-level UDP format
- `run_localhost_replay.sh`
  - quick launcher for localhost replay
- `tools/udp_replay_player.py`
  - configurable UDP replay sender
- `tools/control_team_replay_common.py`
  - pack/unpack utilities and plan-bank helpers

## Counts
{os.linesep.join(f"- chunk{cid:04d}: {counts[f'chunk{cid:04d}']}" for cid in args.chunk_ids)}
- total: {sum(counts.values())}

## Model/runtime used
- LLM: `/alpamayo_vlm_engines/alpa1.5/llm.engine`
- ViT: `/alpamayo_vlm_engines/alpa1.5_visual_fp8_rebuild`
- FM: `/root/test/output/alpamayo15_fm_one_step_mxfp8/alpamayo15_fm_one_step_mxfp8_thor.plan`

## Coordinate frame
- Default replay packet uses `local` ego frame
- The NPZ also contains `ref_x_world_enu`, `ref_y_world_enu`, `ref_yaw_world_rad`, so the replay player can emit `world` packets if needed

## Quick start
```bash
cd "{args.package_dir}"
./run_localhost_replay.sh
```

Equivalent direct command:
```bash
python "./tools/udp_replay_player.py" \\
  --plan-bank "./{plan_bank_name}" \\
  --host {args.default_host} \\
  --port {args.default_port} \\
  --control-dt {args.control_dt} \\
  --control-points {args.control_points} \\
  --coord-mode local
```

## Tuning knobs
- `--control-dt`
- `--control-points`
- `--coord-mode local|world`
- `--loop`
- `--playback-rate`
""",
    )

    summary = {
        "package_name": args.package_name,
        "chunk_ids": args.chunk_ids,
        "counts": counts,
        "total": sum(counts.values()),
        "plan_bank_npz": str(plan_bank_path),
        "packet_preview_json": str(preview_json_path),
        "control_dt_default_s": args.control_dt,
        "control_points_default": args.control_points,
        "tar_path": str(args.tar_path),
    }
    (args.package_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.tar_path.exists():
        args.tar_path.unlink()
    ensure_dir(args.tar_path.parent)
    with tarfile.open(args.tar_path, "w:gz") as tar:
        tar.add(args.package_dir, arcname=args.package_dir.name, recursive=True)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
