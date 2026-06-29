#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_flex_legacy_threeway_dataset_compare import (
    DEFAULT_LEGACY_FM,
    VariantConfig,
    build_dataset_request_bank,
    build_variant_artifacts,
    prepare_variant_requests,
    run_variant_outputs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build chunk-wide portfolio inference outputs: legacy path at 10Hz and "
            "official 10B path/CoT at a lower keyframe rate."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data" / "2026-06-24-test1")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "portfolio_chunk_20260624_test1")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--legacy-stride", type=int, default=1, help="Legacy sample stride in 10Hz samples.")
    parser.add_argument("--official-stride", type=int, default=10, help="Official 10B sample stride in 10Hz samples.")
    parser.add_argument("--include-last-official", action="store_true", default=True)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
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
    parser.add_argument("--fm-engine", type=Path, default=DEFAULT_LEGACY_FM)
    parser.add_argument("--legacy-seed", type=int, default=23)
    parser.add_argument("--legacy-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=900.0)
    parser.add_argument("--official-model-dir", type=Path, default=Path("/workspace/alpamayo1.5/model"))
    parser.add_argument("--official-diffusion-steps", type=int, default=10)
    parser.add_argument("--official-seed", type=int, default=42)
    parser.add_argument("--official-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run-legacy", action="store_true")
    parser.add_argument("--no-run-official", action="store_true")
    return parser.parse_args()


def chunk_sample_ids(dataset_root: Path, chunk_id: int) -> list[int]:
    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[
        ["frame_id", "chunk_id"]
    ]
    sample_index = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    rows = sample_index[sample_index["chunk_id"] == int(chunk_id)].copy().sort_values("t0_utc_ns")
    if rows.empty:
        raise RuntimeError(f"No samples found for {dataset_root} chunk {chunk_id}")
    return [int(x) for x in rows["sample_id"].tolist()]


def stride_ids(ids: list[int], stride: int, include_last: bool = False) -> list[int]:
    stride = max(1, int(stride))
    out = ids[::stride]
    if include_last and ids and ids[-1] not in out:
        out.append(ids[-1])
    return out


def run_official(args: argparse.Namespace, sample_ids: list[int], request_bank_root: Path, official_root: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_official_10b_hf_request_bank.py"),
        "--model-dir",
        str(args.official_model_dir),
        "--request-bank-root",
        str(request_bank_root),
        "--output-root",
        str(official_root),
        "--sample-ids",
        *[str(x) for x in sample_ids],
        "--mode",
        "ae",
        "--diffusion-steps",
        str(args.official_diffusion_steps),
        "--seed",
        str(args.official_seed),
        "--dtype",
        args.official_dtype,
        "--attn-implementation",
        "sdpa",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    print("[portfolio-chunk] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    all_chunk_ids = chunk_sample_ids(args.dataset_root, args.chunk_id)
    legacy_ids = stride_ids(all_chunk_ids, args.legacy_stride, include_last=True)
    official_ids = stride_ids(all_chunk_ids, args.official_stride, include_last=args.include_last_official)
    request_ids = sorted(set(legacy_ids) | set(official_ids))

    run_root = args.output_root / args.dataset_root.name / f"chunk{args.chunk_id:04d}"
    request_bank_root = run_root / "request_bank"
    legacy_root = run_root / "legacy_seed23"
    official_root = run_root / "official_10b"

    manifest_rows, request_summary = build_dataset_request_bank(
        dataset_root=args.dataset_root,
        out_root=request_bank_root,
        chunk_id=args.chunk_id,
        sample_count=len(request_ids),
        sample_ids=request_ids,
        width=args.width,
        height=args.height,
        history_len=args.history_len,
        dt_s=args.dt_s,
        traj_token_offset=args.traj_token_offset,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    rows_by_sid = {int(row["sample_id"]): row for row in manifest_rows}
    legacy_rows = [rows_by_sid[sid] for sid in legacy_ids if sid in rows_by_sid]

    legacy_variant = VariantConfig(
        name="legacy_seed23",
        label="Legacy seed23 AE",
        color="#111827",
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        use_prefill_kv=True,
        diffusion_seed=args.legacy_seed,
        diffusion_num_steps=args.legacy_steps,
        max_generate_length=args.max_generate_length,
    )
    request_paths = prepare_variant_requests(
        manifest_rows=legacy_rows,
        variant=legacy_variant,
        out_root=legacy_root / "requests",
    )
    output_paths = [legacy_root / "outputs" / p.name.replace("request_", "output_") for p in request_paths]
    if not args.no_run_legacy:
        output_paths = run_variant_outputs(
            variant=legacy_variant,
            request_paths=request_paths,
            output_root=legacy_root / "outputs",
            llm_inference_bin=args.llm_inference_bin,
            plugin_lib=args.plugin_lib,
            warmup=args.warmup,
            timeout_per_request=args.timeout_per_request,
            skip_existing=args.skip_existing,
        )
    build_variant_artifacts(
        variant=legacy_variant,
        output_paths=output_paths,
        request_paths=request_paths,
        manifest_rows=legacy_rows,
        dataset_root=args.dataset_root,
        history_len=args.history_len,
        artifact_root=legacy_root / "artifacts",
    )
    if not args.no_run_official:
        run_official(args, official_ids, request_bank_root, official_root)

    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "chunk_id": int(args.chunk_id),
        "run_root": str(run_root),
        "request_bank_root": str(request_bank_root),
        "legacy_root": str(legacy_root),
        "official_root": str(official_root),
        "all_chunk_sample_ids": [int(all_chunk_ids[0]), int(all_chunk_ids[-1])],
        "all_chunk_sample_count": len(all_chunk_ids),
        "legacy_sample_count": len(legacy_ids),
        "legacy_sample_ids": legacy_ids,
        "official_sample_count": len(official_ids),
        "official_sample_ids": official_ids,
        "request_summary": request_summary,
    }
    write_json(run_root / "portfolio_chunk_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
