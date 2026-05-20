#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Watch raw dataset chunk video sizes until stable, then run one-shot inference. Retries on failure."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int, default=1)
    parser.add_argument("--target-offset-s", type=float, required=True)
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=first_existing_fm_engine(),
    )
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--traj-udp-port", type=int, default=5001)
    parser.add_argument("--action-udp-port", type=int, default=5002)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--stable-count", type=int, default=2)
    parser.add_argument("--retry-s", type=float, default=10.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--work-root", type=Path, default=repo_root / "output" / "raw_dataset_one_shot")
    return parser.parse_args()


def chunk_paths(dataset_root: Path, chunk_id: int) -> list[Path]:
    chunk_name = f"chunk_{chunk_id:04d}.mkv"
    return [
        dataset_root / "sensors" / "camera_left" / "chunks" / chunk_name,
        dataset_root / "sensors" / "camera_front" / "chunks" / chunk_name,
        dataset_root / "sensors" / "camera_right" / "chunks" / chunk_name,
        dataset_root / "sensors" / "camera_front_tele" / "chunks" / chunk_name,
    ]


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} {msg}", flush=True)


def wait_until_stable(paths: list[Path], poll_s: float, stable_count: int) -> tuple[int, ...]:
    prev: tuple[int, ...] | None = None
    stable = 0
    while True:
        sizes = tuple(path.stat().st_size for path in paths)
        log(f"sizes={sizes}")
        if sizes == prev:
            stable += 1
        else:
            stable = 0
        prev = sizes
        if stable >= stable_count:
            return sizes
        time.sleep(poll_s)


def run_one_shot(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        args.python,
        str(repo_root / "scripts" / "run_raw_dataset_one_shot_udp.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--chunk-id",
        str(args.chunk_id),
        "--target-offset-s",
        str(args.target_offset_s),
        "--work-root",
        str(args.work_root),
        "--engine-dir",
        str(args.engine_dir),
        "--multimodal-engine-dir",
        str(args.multimodal_engine_dir),
        "--fm-engine",
        str(args.fm_engine),
        "--udp-host",
        args.udp_host,
        "--traj-udp-port",
        str(args.traj_udp_port),
        "--action-udp-port",
        str(args.action_udp_port),
    ]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(repo_root) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    log("starting one-shot run")
    log("cmd=" + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, env=env)
    log(f"one-shot exited code={proc.returncode}")
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    paths = chunk_paths(args.dataset_root, args.chunk_id)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing chunk files: " + ", ".join(missing))

    while True:
        sizes = wait_until_stable(paths, poll_s=args.poll_s, stable_count=args.stable_count)
        log(f"sizes_stable={sizes}")
        code = run_one_shot(args)
        if code == 0:
            return
        log(f"run failed; sleeping {args.retry_s:.1f}s before monitoring again")
        time.sleep(args.retry_s)


if __name__ == "__main__":
    main()
