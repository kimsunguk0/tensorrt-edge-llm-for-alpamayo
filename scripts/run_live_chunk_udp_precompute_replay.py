#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine
from run_live_chunk_udp_replay import (
    ReplayViewerState,
    _viewer_html,
    allocate_viewer_server,
    build_request_bank,
    build_text_udp_payload,
    choose_summary,
    load_entries,
    load_sensor_speed_lookup,
    lookup_nearest_sensor_speed,
    persist_viewer_state,
    resolve_viewer_advertise_host,
    select_entries_from_target_offset,
    try_open_viewer_url,
    update_latest_viewer_images,
)
from run_raw_dataset_one_shot_udp import build_result_artifacts, ensure_exists
from run_request_bank_persistent import build_env, read_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute every selected 0.1s raw live-dataset request, then replay the "
            "computed paths to control over UDP at a fixed interval."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "live_chunk_udp_replay")
    parser.add_argument("--request-bank-root", type=Path, default=None)
    parser.add_argument("--request-limit", type=int, default=-1)
    parser.add_argument("--request-stride", type=int, default=1)
    parser.add_argument(
        "--target-offset-s",
        type=float,
        default=None,
        help="Start from the request nearest to this chunk-relative offset in seconds.",
    )
    parser.add_argument("--rebuild-request-bank", action="store_true")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=first_existing_fm_engine())
    parser.add_argument("--alpamayo-nav-cfg", action="store_true")
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--udp-host", type=str, default="192.168.0.29")
    parser.add_argument("--udp-port", type=int, default=5001)
    parser.add_argument(
        "--udp-path-source",
        choices=["ac_decoded", "final", "gt"],
        default="ac_decoded",
        help="Which path variant to packetize and send.",
    )
    parser.add_argument("--send-interval-s", type=float, default=0.1)
    parser.add_argument("--send-start-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--send-only",
        action="store_true",
        help="Skip model/artifact work and replay an existing precomputed_udp_payloads.jsonl over UDP.",
    )
    parser.add_argument(
        "--payloads-jsonl",
        type=Path,
        default=None,
        help="Payload JSONL to replay with --send-only. Defaults to the chunk precompute artifact path.",
    )
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--skip-udp", action="store_true")
    parser.add_argument(
        "--reuse-existing-outputs",
        action="store_true",
        help="Use existing output JSON files when present instead of rerunning inference for those requests.",
    )
    parser.add_argument("--viewer-host", type=str, default="127.0.0.1")
    parser.add_argument("--viewer-port", type=int, default=8780)
    parser.add_argument("--open-viewer", action="store_true")
    parser.add_argument("--disable-viewer", action="store_true")
    return parser.parse_args()


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
    if args.alpamayo_fm_use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")
    if args.alpamayo_nav_cfg:
        cmd.append("--alpamayoNavCfg")
    return cmd


def start_runtime(args: argparse.Namespace, viewer_state: ReplayViewerState, artifact_root: Path) -> subprocess.Popen[str]:
    cmd = runtime_cmd(args)
    print("[precompute-replay] starting persistent llm_inference", flush=True)
    print("[precompute-replay] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )
    viewer_state.update(phase="starting_runtime", message="loading persistent llm_inference")
    persist_viewer_state(artifact_root, viewer_state)
    ready = read_status(proc, timeout_s=120.0)
    if ready.get("status") != "ready":
        raise RuntimeError(f"Unexpected ready state: {ready}")
    print("[precompute-replay] persistent llm_inference ready", flush=True)
    viewer_state.update(phase="runtime_ready", message="persistent llm_inference ready")
    persist_viewer_state(artifact_root, viewer_state)
    return proc


def shutdown_runtime(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def enrich_udp_payload_for_source(
    payload: dict[str, Any],
    *,
    chunk_id: int,
    sample_id: int,
    t0_utc_ns: int,
    t0_us: int,
    actual_offset_s: float,
    precompute_index: int,
    precompute_count: int,
    inference_reused: bool,
) -> dict[str, Any]:
    enriched = copy.deepcopy(payload)
    enriched.update(
        {
            "udp_replay_mode": "precompute_fixed_interval",
            "source_chunk_id": int(chunk_id),
            "source_sample_id": int(sample_id),
            "source_t0_utc_ns": int(t0_utc_ns),
            "source_t0_us": int(t0_us),
            "source_actual_offset_s": float(actual_offset_s),
            "precompute_index": int(precompute_index),
            "precompute_count": int(precompute_count),
            "precompute_inference_reused": bool(inference_reused),
        }
    )
    return enriched


def enrich_udp_payload_for_send(
    payload: dict[str, Any],
    *,
    replay_index: int,
    replay_count: int,
    send_interval_s: float,
    scheduled_send_offset_s: float,
    replay_start_unix_s: float,
) -> dict[str, Any]:
    enriched = copy.deepcopy(payload)
    send_unix_s = time.time()
    enriched.update(
        {
            "udp_replay_mode": "precompute_fixed_interval",
            "replay_index": int(replay_index),
            "replay_count": int(replay_count),
            "send_interval_s": float(send_interval_s),
            "scheduled_send_offset_s": float(scheduled_send_offset_s),
            "replay_start_unix_s": float(replay_start_unix_s),
            "replay_send_unix_s": float(send_unix_s),
            "replay_elapsed_s": float(send_unix_s - replay_start_unix_s),
        }
    )
    return enriched


def load_precomputed_payloads(payloads_jsonl: Path) -> list[dict[str, Any]]:
    ensure_exists(payloads_jsonl, "precomputed UDP payloads JSONL")
    payloads: list[dict[str, Any]] = []
    for line_no, line in enumerate(payloads_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {payloads_jsonl}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected object payload in {payloads_jsonl}:{line_no}")
        payloads.append(payload)
    if not payloads:
        raise RuntimeError(f"No payloads found in {payloads_jsonl}")
    return payloads


def select_payloads_for_send(args: argparse.Namespace, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = list(payloads)
    if args.target_offset_s is not None:
        target = float(args.target_offset_s)

        def offset_for(idx: int) -> float:
            payload = selected[idx]
            value = payload.get("source_actual_offset_s")
            if value is None:
                value = payload.get("scheduled_send_offset_s")
            if value is None:
                value = idx * float(args.send_interval_s)
            return float(value)

        start_idx = min(range(len(selected)), key=lambda idx: abs(offset_for(idx) - target))
        selected = selected[start_idx:]
        print(
            "[precompute-replay] send-only target offset "
            f"{target:.3f}s -> selected replay start index={start_idx + 1}",
            flush=True,
        )
    if args.request_limit > 0:
        selected = selected[: args.request_limit]
    if not selected:
        raise RuntimeError("No payloads selected for send-only replay")
    return selected


def send_precomputed_payloads(
    *,
    args: argparse.Namespace,
    payloads: list[dict[str, Any]],
    artifact_root: Path,
    viewer_state: ReplayViewerState,
) -> tuple[list[dict[str, Any]], int]:
    if args.skip_udp:
        raise RuntimeError("--send-only cannot be combined with --skip-udp")

    target = (args.udp_host, int(args.udp_port))
    tx_log: list[dict[str, Any]] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        print(
            "[precompute-replay] send-only UDP playback enabled "
            f"target={args.udp_host}:{args.udp_port} interval={float(args.send_interval_s):.3f}s "
            f"count={len(payloads)}",
            flush=True,
        )
        if args.send_start_delay_s > 0.0:
            viewer_state.update(
                phase="send_only_waiting",
                message=f"waiting {float(args.send_start_delay_s):.3f}s before UDP playback",
            )
            persist_viewer_state(artifact_root, viewer_state)
            time.sleep(float(args.send_start_delay_s))

        playback_start_monotonic = time.monotonic()
        playback_start_unix_s = time.time()
        for replay_index, payload in enumerate(payloads, start=1):
            scheduled_offset_s = (replay_index - 1) * float(args.send_interval_s)
            sleep_s = playback_start_monotonic + scheduled_offset_s - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

            send_payload = enrich_udp_payload_for_send(
                payload,
                replay_index=replay_index,
                replay_count=len(payloads),
                send_interval_s=float(args.send_interval_s),
                scheduled_send_offset_s=scheduled_offset_s,
                replay_start_unix_s=playback_start_unix_s,
            )
            payload_bytes = json.dumps(send_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload_bytes, target)

            sample_id = send_payload.get("source_sample_id", send_payload.get("sample_id"))
            tx_row = {
                "request_index": int(replay_index),
                "request_count": int(len(payloads)),
                "sample_id": int(sample_id) if sample_id is not None else None,
                "t0_us": send_payload.get("source_t0_us"),
                "t0_utc_ns": send_payload.get("source_t0_utc_ns"),
                "replay_mode": "precompute_fixed_interval_send_only",
                "actual_offset_s": send_payload.get("source_actual_offset_s"),
                "udp_path_source": args.udp_path_source,
                "udp_host": args.udp_host,
                "udp_port": int(args.udp_port),
                "udp_sent": True,
                "udp_mode": "precompute_fixed_interval_send_only",
                "udp_num_points": int(len(send_payload.get("packet_points") or [])),
                "initial_speed_mps": send_payload.get("initial_speed_mps"),
                "sensor_speed_mps": send_payload.get("sensor_speed_mps"),
                "inference_time_s": send_payload.get("inference_time_s"),
                "replay_index": int(replay_index),
                "replay_count": int(len(payloads)),
                "scheduled_send_offset_s": float(scheduled_offset_s),
                "replay_start_unix_s": float(playback_start_unix_s),
                "replay_send_unix_s": float(send_payload["replay_send_unix_s"]),
                "replay_elapsed_s": float(send_payload["replay_elapsed_s"]),
                "send_interval_s": float(args.send_interval_s),
            }
            tx_log.append(tx_row)
            write_json(artifact_root / "latest_udp_payload.json", send_payload)
            write_json(artifact_root / "latest_tx_row.json", tx_row)
            write_json(artifact_root / "send_only_tx_log.json", tx_log)
            viewer_state.update(
                phase="send_only_udp_playback",
                message=f"sent precomputed path {replay_index}/{len(payloads)}",
                current_request_index=replay_index,
                total_requests=len(payloads),
                udp_sent_count=replay_index,
                latest={
                    "chosen_source": args.udp_path_source,
                    "tx_row": copy.deepcopy(tx_row),
                    "udp_payload": copy.deepcopy(send_payload),
                },
            )
            persist_viewer_state(artifact_root, viewer_state)
            print(
                f"[precompute-replay] send-only sent {replay_index}/{len(payloads)} "
                f"sample_id={tx_row['sample_id']} scheduled_offset_s={scheduled_offset_s:.3f}",
                flush=True,
            )
    finally:
        sock.close()
    return tx_log, len(payloads)


def main() -> None:
    args = parse_args()
    if args.send_interval_s <= 0.0:
        raise ValueError("--send-interval-s must be positive")
    ensure_exists(args.dataset_root, "dataset root")
    if not args.send_only:
        for path, description in [
            (args.llm_inference_bin, "llm_inference"),
            (args.plugin_lib, "plugin lib"),
            (args.engine_dir, "engineDir"),
            (args.multimodal_engine_dir, "multimodalEngineDir"),
            (args.fm_engine, "fmEngine"),
        ]:
            ensure_exists(path, description)

    chunk_tag = f"chunk{args.chunk_id:04d}"
    run_root = args.work_root / chunk_tag / "precompute_udp_replay"
    request_bank_root = args.request_bank_root or (args.work_root / chunk_tag / "request_bank")
    output_root = run_root / "outputs"
    artifact_root = run_root / "artifacts"
    payload_root = artifact_root / "udp_payloads"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload_root.mkdir(parents=True, exist_ok=True)

    viewer_state = ReplayViewerState()
    viewer_state.update(
        phase="starting",
        message="preparing request bank and precompute runtime",
        udp_host=args.udp_host,
        udp_port=int(args.udp_port),
        udp_path_source=args.udp_path_source,
        replay_mode="precompute_fixed_interval",
        chunk_id=int(args.chunk_id),
        dataset_root=str(args.dataset_root),
        send_interval_s=float(args.send_interval_s),
    )
    persist_viewer_state(artifact_root, viewer_state)

    viewer_server = None
    if not args.disable_viewer:
        try:
            (artifact_root / "index.html").write_text(_viewer_html(), encoding="utf-8")
            viewer_server, actual_viewer_port = allocate_viewer_server(
                args.viewer_host,
                int(args.viewer_port),
                artifact_root,
                viewer_state,
            )
            viewer_url = f"http://{resolve_viewer_advertise_host(args.viewer_host)}:{actual_viewer_port}/viewer"
            viewer_state.update(
                viewer_host=args.viewer_host,
                viewer_port=int(actual_viewer_port),
                viewer_url=viewer_url,
            )
            persist_viewer_state(artifact_root, viewer_state)
            print(f"[precompute-replay] viewer: {viewer_url}", flush=True)
            if int(actual_viewer_port) != int(args.viewer_port):
                print(
                    "[precompute-replay] viewer requested port "
                    f"{int(args.viewer_port)} was busy; using {int(actual_viewer_port)} instead. "
                    "If you are accessing from outside Docker, make sure that port is published too.",
                    flush=True,
                )
            if args.open_viewer:
                opened = try_open_viewer_url(viewer_url)
                print(f"[precompute-replay] viewer auto-open: {'requested' if opened else 'failed'}", flush=True)
        except Exception as exc:
            print(f"[precompute-replay] viewer disabled automatically: {exc}", flush=True)

    proc: subprocess.Popen[str] | None = None
    tx_log: list[dict[str, Any]] = []
    precomputed_rows: list[dict[str, Any]] = []
    try:
        if args.send_only:
            payloads_jsonl = args.payloads_jsonl or (artifact_root / "precomputed_udp_payloads.jsonl")
            payloads = select_payloads_for_send(args, load_precomputed_payloads(payloads_jsonl))
            viewer_state.update(
                phase="send_only_ready",
                message="loaded precomputed payloads for UDP replay",
                total_requests=len(payloads),
                payloads_jsonl_path=str(payloads_jsonl),
            )
            persist_viewer_state(artifact_root, viewer_state)
            _, sent_count = send_precomputed_payloads(
                args=args,
                payloads=payloads,
                artifact_root=artifact_root,
                viewer_state=viewer_state,
            )
            summary = {
                "dataset_root": str(args.dataset_root),
                "chunk_id": int(args.chunk_id),
                "replay_mode": "precompute_fixed_interval_send_only",
                "artifact_root": str(artifact_root),
                "payloads_jsonl_path": str(payloads_jsonl),
                "selected_request_count": len(payloads),
                "precomputed_request_count": len(payloads),
                "udp_sent_count": int(sent_count),
                "udp_host": args.udp_host,
                "udp_port": int(args.udp_port),
                "udp_path_source": args.udp_path_source,
                "udp_enabled": True,
                "send_interval_s": float(args.send_interval_s),
                "send_start_delay_s": float(args.send_start_delay_s),
                "tx_log_path": str(artifact_root / "send_only_tx_log.json"),
            }
            write_json(artifact_root / "send_only_summary.json", summary)
            viewer_state.update(phase="finished", message="send-only UDP replay completed")
            persist_viewer_state(artifact_root, viewer_state)
            print(json.dumps(summary, indent=2), flush=True)
            return

        if args.rebuild_request_bank or not (request_bank_root / "summary.json").exists():
            build_request_bank(args, request_bank_root)
        else:
            print(f"[precompute-replay] reusing request bank: {request_bank_root}", flush=True)

        entries, request_bank_summary = load_entries(request_bank_root, max(1, int(args.request_stride)))
        entries, start_entry_index = select_entries_from_target_offset(entries, args.target_offset_s)
        if args.request_limit > 0:
            entries = entries[: args.request_limit]
        if not entries:
            raise RuntimeError("No request entries selected after applying stride/limit")
        if args.target_offset_s is not None and start_entry_index is not None:
            first = entries[0]
            print(
                "[precompute-replay] target offset "
                f"{float(args.target_offset_s):.3f}s -> selected sample_id={first.sample_id} "
                f"actual_offset_s={first.actual_offset_s:.6f}",
                flush=True,
            )

        sensor_speed_times_utc_ns, sensor_speeds_mps, sensor_speed_source = load_sensor_speed_lookup(args.dataset_root)
        history_len = int(request_bank_summary["history_len"])
        total = len(entries)
        viewer_state.update(
            phase="request_bank_ready",
            message="request bank loaded",
            total_requests=total,
            request_bank_root=str(request_bank_root),
            artifact_root=str(artifact_root),
            output_root=str(output_root),
            target_offset_s=float(args.target_offset_s) if args.target_offset_s is not None else None,
            selected_start_sample_id=int(entries[0].sample_id),
            selected_start_actual_offset_s=float(entries[0].actual_offset_s),
        )
        persist_viewer_state(artifact_root, viewer_state)

        precompute_start = time.time()
        for idx, entry in enumerate(entries, start=1):
            output_name = entry.request_path.name.replace("request_", "output_")
            output_path = output_root / output_name
            sample_artifact_root = artifact_root / entry.request_path.stem
            inference_reused = bool(args.reuse_existing_outputs and output_path.exists())
            viewer_state.update(
                phase="precomputing",
                message=f"precomputing request {idx}/{total}",
                current_request_index=idx,
                total_requests=total,
                current_sample_id=entry.sample_id,
                current_t0_us=entry.t0_us,
                current_request_json=str(entry.request_path),
                desired_offset_s=float(entry.actual_offset_s),
            )
            persist_viewer_state(artifact_root, viewer_state)

            if inference_reused:
                inference_time_s = 0.0
                existing_payload_path = sample_artifact_root / "udp_payload.json"
                if existing_payload_path.exists():
                    try:
                        existing_payload = json.loads(existing_payload_path.read_text(encoding="utf-8"))
                        inference_time_s = float(existing_payload.get("inference_time_s", 0.0))
                    except Exception:
                        inference_time_s = 0.0
                print(
                    f"[precompute-replay] {idx}/{total} reusing output {output_path.name}",
                    flush=True,
                )
            else:
                if proc is None:
                    proc = start_runtime(args, viewer_state, artifact_root)
                payload = {"input_file": str(entry.request_path), "output_file": str(output_path)}
                assert proc.stdin is not None
                request_t0 = time.time()
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                status = read_status(proc, timeout_s=args.timeout_per_request)
                if status.get("status") != "ok":
                    raise RuntimeError(f"Request failed for {entry.request_path.name}: {status}")
                inference_time_s = time.time() - request_t0

            normalized_metadata = {
                "chunk_id": args.chunk_id,
                "sample_id": entry.sample_id,
                "front_frame_id": entry.front_frame_id,
                "t0_utc_ns": entry.t0_utc_ns,
                "t0_us": entry.t0_us,
                "target_offset_s": entry.actual_offset_s,
                "actual_offset_s": entry.actual_offset_s,
                "request_json": str(entry.request_path),
                "ego_history_xyz_npy": entry.ego_history_xyz_npy,
                "ego_history_rot_npy": entry.ego_history_rot_npy,
                "selected_frames": entry.selected_frames,
            }
            sample_artifact_root.mkdir(parents=True, exist_ok=True)
            final_summary, ac_summary, gt_summary, _ = build_result_artifacts(
                output_path=output_path,
                metadata=normalized_metadata,
                dataset_root=args.dataset_root,
                history_len=history_len,
                artifact_root=sample_artifact_root,
            )
            chosen_summary = choose_summary(args.udp_path_source, final_summary, ac_summary, gt_summary)
            sensor_speed_mps, sensor_speed_t0_utc_ns = lookup_nearest_sensor_speed(
                sensor_speed_times_utc_ns,
                sensor_speeds_mps,
                entry.t0_utc_ns,
            )
            udp_payload = build_text_udp_payload(
                summary=chosen_summary,
                sensor_speed_mps=sensor_speed_mps,
                sensor_speed_t0_utc_ns=sensor_speed_t0_utc_ns,
                sensor_speed_source=sensor_speed_source,
                inference_time_s=inference_time_s,
            )
            udp_payload = enrich_udp_payload_for_source(
                udp_payload,
                chunk_id=args.chunk_id,
                sample_id=entry.sample_id,
                t0_utc_ns=entry.t0_utc_ns,
                t0_us=entry.t0_us,
                actual_offset_s=entry.actual_offset_s,
                precompute_index=idx,
                precompute_count=total,
                inference_reused=inference_reused,
            )

            elapsed = time.time() - precompute_start
            tx_row = {
                "request_index": idx,
                "request_count": total,
                "selected_entry_index": idx - 1,
                "sample_id": entry.sample_id,
                "t0_us": entry.t0_us,
                "t0_utc_ns": entry.t0_utc_ns,
                "request_json": str(entry.request_path),
                "output_json": str(output_path),
                "artifact_root": str(sample_artifact_root),
                "replay_mode": "precompute_fixed_interval",
                "desired_offset_s": float(entry.actual_offset_s),
                "actual_offset_s": float(entry.actual_offset_s),
                "udp_path_source": args.udp_path_source,
                "udp_host": args.udp_host,
                "udp_port": int(args.udp_port),
                "udp_sent": False,
                "udp_mode": "pending_playback" if not args.skip_udp else "disabled",
                "udp_num_points": int(len(udp_payload.get("packet_points") or [])),
                "initial_speed_mps": udp_payload.get("initial_speed_mps"),
                "sensor_speed_mps": udp_payload.get("sensor_speed_mps"),
                "output_text": chosen_summary.get("final_output"),
                "traj_points_with_origin": int(chosen_summary.get("traj_points_with_origin", 0)),
                "inference_time_s": float(inference_time_s),
                "precompute_inference_reused": inference_reused,
                "precompute_elapsed_s": float(elapsed),
                "send_interval_s": float(args.send_interval_s),
            }
            tx_log.append(tx_row)
            precomputed_rows.append(
                {
                    "tx_row": tx_row,
                    "udp_payload": udp_payload,
                    "final_summary": final_summary,
                    "ac_summary": ac_summary,
                    "gt_summary": gt_summary,
                    "request_path": entry.request_path,
                }
            )

            update_latest_viewer_images(artifact_root, request_bank_root, entry.request_path)
            write_json(artifact_root / "latest_final_path.json", final_summary)
            write_json(artifact_root / "latest_ac_decoded_path.json", ac_summary)
            write_json(artifact_root / "latest_gt_path.json", gt_summary)
            write_json(artifact_root / "latest_tx_row.json", tx_row)
            write_json(sample_artifact_root / "udp_payload.json", udp_payload)
            write_json(payload_root / f"udp_payload_{idx:06d}_sid{entry.sample_id}.json", udp_payload)
            write_json(artifact_root / "latest_udp_payload.json", udp_payload)
            write_json(artifact_root / "tx_log.json", tx_log)
            viewer_state.set_latest(
                final_summary=final_summary,
                ac_summary=ac_summary,
                gt_summary=gt_summary,
                chosen_source=args.udp_path_source,
                tx_row=tx_row,
            )
            viewer_state.update(
                phase="precomputed_sample",
                message=f"precomputed sample {entry.sample_id}",
                udp_sent_count=sum(1 for row in tx_log if row.get("udp_sent")),
                skipped_request_count=0,
            )
            persist_viewer_state(artifact_root, viewer_state)
            print(
                f"[precompute-replay] precomputed {idx}/{total} sample_id={entry.sample_id} "
                f"path={args.udp_path_source} inference_time_s={inference_time_s:.3f} reused={inference_reused}",
                flush=True,
            )

        jsonl_path = artifact_root / "precomputed_udp_payloads.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(row["udp_payload"], ensure_ascii=False, separators=(",", ":")) for row in precomputed_rows) + "\n",
            encoding="utf-8",
        )
        viewer_state.update(
            phase="precompute_finished",
            message="all paths precomputed",
            current_request_index=total,
            total_requests=total,
            udp_sent_count=0,
        )
        persist_viewer_state(artifact_root, viewer_state)
        shutdown_runtime(proc)
        proc = None

        if args.precompute_only or args.skip_udp:
            print(
                "[precompute-replay] UDP playback skipped "
                f"precompute_only={bool(args.precompute_only)} skip_udp={bool(args.skip_udp)}",
                flush=True,
            )
        else:
            target = (args.udp_host, int(args.udp_port))
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                print(
                    "[precompute-replay] udp fixed-interval playback enabled "
                    f"target={args.udp_host}:{args.udp_port} interval={float(args.send_interval_s):.3f}s "
                    f"count={len(precomputed_rows)}",
                    flush=True,
                )
                if args.send_start_delay_s > 0.0:
                    viewer_state.update(
                        phase="playback_waiting",
                        message=f"waiting {float(args.send_start_delay_s):.3f}s before UDP playback",
                    )
                    persist_viewer_state(artifact_root, viewer_state)
                    time.sleep(float(args.send_start_delay_s))

                playback_start_monotonic = time.monotonic()
                playback_start_unix_s = time.time()
                for replay_index, row in enumerate(precomputed_rows, start=1):
                    scheduled_offset_s = (replay_index - 1) * float(args.send_interval_s)
                    sleep_s = playback_start_monotonic + scheduled_offset_s - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

                    send_payload = enrich_udp_payload_for_send(
                        row["udp_payload"],
                        replay_index=replay_index,
                        replay_count=len(precomputed_rows),
                        send_interval_s=float(args.send_interval_s),
                        scheduled_send_offset_s=scheduled_offset_s,
                        replay_start_unix_s=playback_start_unix_s,
                    )
                    payload_bytes = json.dumps(send_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    sock.sendto(payload_bytes, target)

                    tx_row = row["tx_row"]
                    update_latest_viewer_images(artifact_root, request_bank_root, row["request_path"])
                    tx_row.update(
                        {
                            "udp_sent": True,
                            "udp_mode": "precompute_fixed_interval",
                            "replay_index": int(replay_index),
                            "replay_count": int(len(precomputed_rows)),
                            "scheduled_send_offset_s": float(scheduled_offset_s),
                            "replay_start_unix_s": float(playback_start_unix_s),
                            "replay_send_unix_s": float(send_payload["replay_send_unix_s"]),
                            "replay_elapsed_s": float(send_payload["replay_elapsed_s"]),
                        }
                    )
                    write_json(artifact_root / "latest_udp_payload.json", send_payload)
                    write_json(artifact_root / "latest_tx_row.json", tx_row)
                    write_json(artifact_root / "tx_log.json", tx_log)
                    viewer_state.set_latest(
                        final_summary=row["final_summary"],
                        ac_summary=row["ac_summary"],
                        gt_summary=row["gt_summary"],
                        chosen_source=args.udp_path_source,
                        tx_row=tx_row,
                    )
                    viewer_state.update(
                        phase="udp_playback",
                        message=f"sent precomputed path {replay_index}/{len(precomputed_rows)}",
                        current_request_index=replay_index,
                        total_requests=len(precomputed_rows),
                        udp_sent_count=replay_index,
                    )
                    persist_viewer_state(artifact_root, viewer_state)
                    print(
                        f"[precompute-replay] sent {replay_index}/{len(precomputed_rows)} "
                        f"sample_id={tx_row['sample_id']} scheduled_offset_s={scheduled_offset_s:.3f}",
                        flush=True,
                    )
            finally:
                sock.close()

        summary = {
            "dataset_root": str(args.dataset_root),
            "chunk_id": int(args.chunk_id),
            "replay_mode": "precompute_fixed_interval",
            "request_bank_root": str(request_bank_root),
            "output_root": str(output_root),
            "artifact_root": str(artifact_root),
            "payload_root": str(payload_root),
            "payloads_jsonl_path": str(artifact_root / "precomputed_udp_payloads.jsonl"),
            "selected_request_count": len(entries),
            "precomputed_request_count": len(tx_log),
            "udp_sent_count": sum(1 for row in tx_log if row.get("udp_sent")),
            "udp_host": args.udp_host,
            "udp_port": int(args.udp_port),
            "udp_path_source": args.udp_path_source,
            "udp_enabled": bool(not args.skip_udp and not args.precompute_only),
            "send_interval_s": float(args.send_interval_s),
            "send_start_delay_s": float(args.send_start_delay_s),
            "alpamayo_nav_cfg": bool(args.alpamayo_nav_cfg),
            "fm_engine": str(args.fm_engine),
            "tx_log_path": str(artifact_root / "tx_log.json"),
        }
        write_json(artifact_root / "summary.json", summary)
        viewer_state.update(phase="finished", message="precompute UDP replay completed")
        persist_viewer_state(artifact_root, viewer_state)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        shutdown_runtime(proc)
        if viewer_server is not None:
            viewer_server.shutdown()
            viewer_server.server_close()


if __name__ == "__main__":
    main()
