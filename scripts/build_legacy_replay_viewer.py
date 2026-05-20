#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_live_chunk_udp_replay import (  # noqa: E402
    ReplayViewerState,
    _viewer_html,
    allocate_viewer_server,
    persist_viewer_state,
    resolve_viewer_advertise_host,
    update_latest_viewer_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static legacy chunk replay viewer from a one-shot run directory."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="One-shot run root containing request_bank/, outputs/, and artifacts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Viewer artifact directory. Defaults to <run-root>/artifacts/legacy_viewer.",
    )
    parser.add_argument(
        "--chosen-source",
        type=str,
        default="final_path",
        help="Path source label shown in the viewer.",
    )
    parser.add_argument("--serve", action="store_true", help="Start the legacy viewer server and keep it running.")
    parser.add_argument("--viewer-host", type=str, default="0.0.0.0")
    parser.add_argument("--viewer-port", type=int, default=8765)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_outputs(run_root: Path) -> dict[str, Path]:
    request_paths = sorted((run_root / "request_bank" / "requests").glob("request_chunk*.json"))
    output_paths = sorted((run_root / "outputs").glob("output_chunk*.json"))
    paths = {
        "request_json": request_paths[0] if request_paths else None,
        "model_output_json": output_paths[0] if output_paths else None,
        "final_path_json": run_root / "artifacts" / "final_path.json",
        "ac_decoded_path_json": run_root / "artifacts" / "ac_decoded_path.json",
        "gt_path_json": run_root / "artifacts" / "gt_path.json",
    }
    missing = [key for key, path in paths.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run artifacts under {run_root}: {', '.join(missing)}")
    return paths  # type: ignore[return-value]


def infer_sample_label(request_path: Path, final_summary: dict[str, Any]) -> str:
    stem = request_path.stem
    if stem.startswith("request_"):
        stem = stem[len("request_") :]
    for pattern in [r"(sid\d+)", r"(fid\d+)"]:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    frame_id = final_summary.get("front_frame_id")
    if frame_id is not None:
        return f"fid{int(frame_id):06d}"
    return stem


def build_tx_row(
    *,
    request_path: Path,
    output_json_path: Path,
    run_root: Path,
    final_summary: dict[str, Any],
    output_json: dict[str, Any],
    chosen_source: str,
) -> dict[str, Any]:
    response = (output_json.get("responses") or [{}])[0]
    post = response.get("alpamayo_post_vlm") or {}
    guided = post.get("guided") or {}
    timing = final_summary.get("timing") or post.get("timing") or {}
    packet_points = final_summary.get("packet_points") or []
    initial_speed_mps = None
    if packet_points:
        initial_speed_mps = packet_points[0].get("v_mps")

    total_post_vlm_ms = timing.get("total_post_vlm_ms")
    request_elapsed_s = (float(total_post_vlm_ms) / 1000.0) if total_post_vlm_ms is not None else None
    actual_offset_s = final_summary.get("actual_offset_s")
    sample_label = infer_sample_label(request_path, final_summary)

    return {
        "request_index": 1,
        "request_count": 1,
        "selected_entry_index": 0,
        "sample_id": sample_label,
        "chunk_id": final_summary.get("chunk_id"),
        "front_frame_id": final_summary.get("front_frame_id"),
        "t0_us": final_summary.get("t0_us"),
        "request_json": str(request_path),
        "output_json": str(output_json_path),
        "artifact_root": str(run_root / "artifacts"),
        "replay_mode": "one_shot",
        "desired_offset_s": final_summary.get("target_offset_s"),
        "actual_offset_s": actual_offset_s,
        "skipped_since_last": 0,
        "skipped_total": 0,
        "udp_path_source": chosen_source,
        "udp_host": None,
        "udp_port": None,
        "udp_sent": False,
        "udp_mode": "not_sent",
        "udp_num_points": len(packet_points),
        "initial_speed_mps": initial_speed_mps,
        "sensor_speed_mps": initial_speed_mps,
        "output_text": final_summary.get("final_output") or guided.get("output_text") or response.get("output_text"),
        "traj_points_with_origin": final_summary.get("traj_points_with_origin"),
        "inference_time_s": request_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "total_elapsed_s": request_elapsed_s,
    }


def write_viewer_artifacts(
    *,
    output_dir: Path,
    request_bank_root: Path,
    request_path: Path,
    final_summary: dict[str, Any],
    ac_summary: dict[str, Any],
    gt_summary: dict[str, Any],
    tx_row: dict[str, Any],
    chosen_source: str,
) -> ReplayViewerState:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(_viewer_html(), encoding="utf-8")
    update_latest_viewer_images(output_dir, request_bank_root, request_path)
    (output_dir / "latest_final_path.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (output_dir / "latest_ac_decoded_path.json").write_text(json.dumps(ac_summary, indent=2), encoding="utf-8")
    (output_dir / "latest_gt_path.json").write_text(json.dumps(gt_summary, indent=2), encoding="utf-8")
    (output_dir / "latest_tx_row.json").write_text(json.dumps(tx_row, indent=2), encoding="utf-8")

    viewer_state = ReplayViewerState()
    viewer_state.update(
        phase="finished",
        message="one-shot viewer snapshot ready",
        current_request_index=1,
        total_requests=1,
        udp_sent_count=0,
        replay_mode="one_shot",
        skipped_request_count=0,
        chunk_id=final_summary.get("chunk_id"),
        udp_host=None,
        udp_port=None,
        udp_path_source=chosen_source,
    )
    viewer_state.set_latest(
        final_summary=final_summary,
        ac_summary=ac_summary,
        gt_summary=gt_summary,
        chosen_source=chosen_source,
        tx_row=tx_row,
    )
    persist_viewer_state(output_dir, viewer_state)
    return viewer_state


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"run root not found: {run_root}")
    output_dir = (args.output_dir or (run_root / "artifacts" / "legacy_viewer")).resolve()

    discovered = discover_run_outputs(run_root)
    request_path = discovered["request_json"]
    output_json_path = discovered["model_output_json"]
    final_summary = load_json(discovered["final_path_json"])
    ac_summary = load_json(discovered["ac_decoded_path_json"])
    gt_summary = load_json(discovered["gt_path_json"])
    output_json = load_json(output_json_path)
    tx_row = build_tx_row(
        request_path=request_path,
        output_json_path=output_json_path,
        run_root=run_root,
        final_summary=final_summary,
        output_json=output_json,
        chosen_source=args.chosen_source,
    )

    viewer_state = write_viewer_artifacts(
        output_dir=output_dir,
        request_bank_root=run_root / "request_bank",
        request_path=request_path,
        final_summary=final_summary,
        ac_summary=ac_summary,
        gt_summary=gt_summary,
        tx_row=tx_row,
        chosen_source=args.chosen_source,
    )

    summary = {
        "run_root": str(run_root),
        "viewer_dir": str(output_dir),
        "viewer_index_html": str(output_dir / "index.html"),
        "latest_view_state_json": str(output_dir / "latest_view_state.json"),
        "sample_id": tx_row["sample_id"],
        "output_text": tx_row["output_text"],
        "actual_offset_s": tx_row["actual_offset_s"],
    }

    if not args.serve:
        print(json.dumps(summary, indent=2), flush=True)
        return

    server, actual_port = allocate_viewer_server(args.viewer_host, args.viewer_port, output_dir, viewer_state)
    advertise_host = resolve_viewer_advertise_host(args.viewer_host)
    viewer_url = f"http://{advertise_host}:{actual_port}/"
    viewer_state.update(viewer_url=viewer_url, message="legacy viewer serving one-shot snapshot")
    persist_viewer_state(output_dir, viewer_state)
    summary["viewer_url"] = viewer_url
    print(json.dumps(summary, indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
