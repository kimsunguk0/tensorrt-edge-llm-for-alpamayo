#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine


def build_env(plugin_lib: Path) -> dict[str, str]:
    env = dict(os.environ)
    if plugin_lib.exists():
        env["EDGELLM_PLUGIN_PATH"] = str(plugin_lib)
        build_dir = str(plugin_lib.parent)
        env["LD_LIBRARY_PATH"] = build_dir + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")
    return env


def read_status(proc: subprocess.Popen[str], timeout_s: float) -> dict[str, Any]:
    assert proc.stdout is not None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"Persistent llm_inference exited early with code {proc.returncode}")
            time.sleep(0.02)
            continue
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            print(f"[llm_inference] {text}", flush=True)
            continue
        if isinstance(obj, dict) and "status" in obj:
            return obj
    raise TimeoutError("Timed out waiting for llm_inference status")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a request bank through llm_inference persistent server.")
    parser.add_argument("--request-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--llm-inference-bin", type=Path, default=Path("/root/TensorRT-Edge-LLM-v060/build/examples/llm/llm_inference"))
    parser.add_argument("--plugin-lib", type=Path, default=Path("/root/TensorRT-Edge-LLM-v060/build/libNvInfer_edgellm_plugin.so"))
    parser.add_argument("--engine-dir", type=Path, default=Path("/alpamayo_vlm_engines/alpa1.5"))
    parser.add_argument("--multimodal-engine-dir", type=Path, default=Path("/alpamayo_vlm_engines/alpa1.5_visual_fp8_rebuild"))
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=first_existing_fm_engine(),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--alpamayo-nav-cfg",
        action="store_true",
        help="Pass --alpamayoNavCfg through to llm_inference so nav-guided dual-cache FM is enabled.",
    )
    parser.add_argument(
        "--alpamayo-fm-use-prefill-kv",
        action="store_true",
        help="Pass --alpamayoFmUsePrefillKv through to llm_inference.",
    )
    parser.add_argument(
        "--dump-nav-dual-cache",
        action="store_true",
        help="Pass --dumpNavDualCache through to llm_inference to export guided/unguided KV snapshot artifacts.",
    )
    parser.add_argument(
        "--nav-cache-output-dir",
        type=Path,
        default=Path("./output/nav_dual_cache"),
        help="Directory for --dumpNavDualCache artifacts.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    request_paths = sorted(args.request_root.glob("request_chunk*.json"))
    if args.limit > 0:
        request_paths = request_paths[: args.limit]
    if not request_paths:
        raise RuntimeError(f"No request JSON files found under {args.request_root}")

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
    if args.dump_nav_dual_cache:
        cmd.extend(["--dumpNavDualCache", "--navCacheOutputDir", str(args.nav_cache_output_dir)])

    print("[runner] starting persistent llm_inference")
    print("[runner] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )

    completed = 0
    start_time = time.time()
    try:
        ready = read_status(proc, timeout_s=120.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected ready state: {ready}")
        print("[runner] persistent llm_inference ready", flush=True)

        for idx, request_path in enumerate(request_paths, start=1):
            output_name = request_path.name.replace("request_", "output_")
            output_path = args.output_root / output_name
            if args.skip_existing and output_path.exists():
                completed += 1
                print(f"[runner] skip existing {completed}/{len(request_paths)} {output_path.name}", flush=True)
                continue

            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            t0 = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            status = read_status(proc, timeout_s=args.timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"Request failed for {request_path.name}: {status}")

            if args.dump_nav_dual_cache:
                src_nav_dir = args.nav_cache_output_dir / "request_0"
                if src_nav_dir.exists():
                    dst_nav_dir = args.nav_cache_output_dir / request_path.stem
                    if dst_nav_dir.exists():
                        shutil.rmtree(dst_nav_dir)
                    shutil.copytree(src_nav_dir, dst_nav_dir)

            completed += 1
            dt = time.time() - t0
            elapsed = time.time() - start_time
            avg = elapsed / max(completed, 1)
            remaining = avg * (len(request_paths) - completed)
            print(
                f"[runner] done {completed}/{len(request_paths)} {request_path.name} "
                f"in {dt:.2f}s | elapsed {elapsed/60:.1f}m | eta {remaining/60:.1f}m",
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

    print("[runner] complete", flush=True)


if __name__ == "__main__":
    main()
