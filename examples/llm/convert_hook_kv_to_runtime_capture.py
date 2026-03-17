#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


KV_LAYOUT = "[numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a full hook KV-cache dump into a runtime-style cropped capture."
    )
    parser.add_argument(
        "--hook-meta",
        type=Path,
        required=True,
        help="Path to hook kv_cache_request_<idx>.json metadata file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to llm_inference output JSON that contains output_token_ids",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write runtime-style cropped artifacts",
    )
    parser.add_argument(
        "--request-idx",
        type=int,
        default=0,
        help="Request index to extract from output JSON",
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="Batch index inside the selected request",
    )
    parser.add_argument(
        "--future-token-count",
        type=int,
        default=64,
        help="Number of future diffusion tokens",
    )
    parser.add_argument(
        "--traj-future-start-token-id",
        type=int,
        default=None,
        help="Override token id for <|traj_future_start|>. If omitted, reads output_json.token_metadata.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_raw_dtype_for_itemsize(itemsize: int):
    mapping = {
        1: np.uint8,
        2: np.uint16,
        4: np.uint32,
        8: np.uint64,
    }
    if itemsize not in mapping:
        raise ValueError(f"Unsupported itemsize for raw copy: {itemsize}")
    return mapping[itemsize]


def get_dtype_itemsize(dtype_name: str) -> int:
    mapping = {
        "FP8": 1,
        "FLOAT16": 2,
        "BFLOAT16": 2,
        "INT16": 2,
        "FLOAT32": 4,
        "INT32": 4,
        "INT64": 8,
        "UINT8": 1,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype in hook metadata: {dtype_name}")
    return mapping[dtype_name]


def find_response(output_json, request_idx: int, batch_idx: int):
    responses = output_json.get("responses", [])
    for response in responses:
        if response.get("request_idx") == request_idx and response.get("batch_idx") == batch_idx:
            return response
    raise KeyError(f"Could not find response for request_idx={request_idx}, batch_idx={batch_idx}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    hook_meta = load_json(args.hook_meta)
    output_json = load_json(args.output_json)
    response = find_response(output_json, args.request_idx, args.batch_idx)

    token_metadata = output_json.get("token_metadata", {})
    traj_future_start_token_id = args.traj_future_start_token_id
    if traj_future_start_token_id is None:
        traj_future_start_token_id = token_metadata.get("traj_future_start_token_id")
    if traj_future_start_token_id is None:
        raise ValueError(
            "traj_future_start_token_id is missing. Pass --traj-future-start-token-id or rerun llm_inference with the updated JSON export."
        )

    output_token_ids = response.get("output_token_ids", [])
    if not output_token_ids:
        raise ValueError("output_token_ids is empty. Rerun llm_inference with the updated JSON export.")

    hook_kv = hook_meta["kv_cache"]
    hook_lengths = hook_meta["kv_cache_lengths"]
    hook_shape = tuple(hook_kv["shape"])
    max_batch_size = hook_shape[1]
    if args.batch_idx < 0 or args.batch_idx >= max_batch_size:
        raise ValueError(f"batch_idx={args.batch_idx} is outside hook KV batch dimension {max_batch_size}")

    active_values = hook_lengths.get("active_values") or hook_lengths.get("all_values")
    if active_values is None or args.batch_idx >= len(active_values):
        raise ValueError("Hook metadata does not contain a usable active KV length for the requested batch.")
    hook_active_len = int(active_values[args.batch_idx])
    generated_token_count = len(output_token_ids)
    # The last sampled token is emitted in output_token_ids, but its KV state is not
    # materialized until the next decode step. The hook KV therefore only reflects
    # generated tokens up to output_token_count - 1.
    cached_generated_token_count = max(generated_token_count - 1, 0)
    input_token_count = hook_active_len - cached_generated_token_count
    if input_token_count < 0:
        raise ValueError(
            "Invalid input token count: "
            f"hook_active_len={hook_active_len}, "
            f"cached_generated={cached_generated_token_count}, "
            f"output_token_count={generated_token_count}"
        )

    if traj_future_start_token_id in output_token_ids:
        traj_future_start_output_index = output_token_ids.index(traj_future_start_token_id)
    else:
        traj_future_start_output_index = generated_token_count - 1

    offset = input_token_count + traj_future_start_output_index + 1
    runtime_seq_len = min(offset, hook_active_len)

    rope_values = hook_meta.get("rope_deltas", {}).get("values")
    if rope_values is None:
        rope_bin_path = args.hook_meta.parent / hook_meta["rope_deltas"]["file"]
        rope_values = np.fromfile(rope_bin_path, dtype=np.int64).reshape(hook_meta["rope_deltas"]["shape"]).tolist()
    rope_array = np.asarray(rope_values, dtype=np.int64).reshape(hook_meta["rope_deltas"]["shape"])
    if rope_array.size == 0:
        raise ValueError("rope_deltas is empty")
    rope_delta = int(rope_array[min(args.batch_idx, rope_array.shape[0] - 1), 0])

    ensure_dir(args.output_dir)
    request_suffix = f"request_{args.request_idx}" if args.batch_idx == 0 else f"request_{args.request_idx}_batch_{args.batch_idx}"

    kv_in_path = args.hook_meta.parent / hook_kv["file"]
    kv_out_path = args.output_dir / f"kv_cache_{request_suffix}.bin"
    lengths_out_path = args.output_dir / f"kv_cache_lengths_{request_suffix}.bin"
    pos_out_path = args.output_dir / f"position_ids_{request_suffix}.bin"
    rope_out_path = args.output_dir / f"rope_deltas_{request_suffix}.bin"
    attn_out_path = args.output_dir / f"attention_mask_{request_suffix}.bin"
    offset_out_path = args.output_dir / f"offset_{request_suffix}.bin"
    meta_out_path = args.output_dir / f"runtime_capture_{request_suffix}.json"

    kv_itemsize = get_dtype_itemsize(hook_kv["dtype"])
    raw_dtype = get_raw_dtype_for_itemsize(kv_itemsize)
    kv_memmap = np.memmap(kv_in_path, dtype=raw_dtype, mode="r", shape=hook_shape)
    kv_cropped = np.asarray(kv_memmap[:, args.batch_idx : args.batch_idx + 1, :, :, :runtime_seq_len, :])
    kv_cropped.tofile(kv_out_path)

    kv_lengths = np.asarray([runtime_seq_len], dtype=np.int32)
    kv_lengths.tofile(lengths_out_path)

    position_ids = np.broadcast_to(
        np.arange(args.future_token_count, dtype=np.int64).reshape(1, 1, args.future_token_count),
        (3, 1, args.future_token_count),
    ).copy()
    position_ids += np.int64(rope_delta + offset)
    position_ids.tofile(pos_out_path)

    rope_out = np.asarray([[rope_delta]], dtype=np.int64)
    rope_out.tofile(rope_out_path)

    offset_out = np.asarray([offset], dtype=np.int64)
    offset_out.tofile(offset_out_path)

    attention_mask = np.zeros((1, 1, args.future_token_count, runtime_seq_len + args.future_token_count), dtype=np.float16)
    if offset < runtime_seq_len:
        attention_mask[:, :, :, offset:runtime_seq_len] = np.finfo(np.float16).min
    attention_mask.tofile(attn_out_path)

    metadata = {
        "request_idx": args.request_idx,
        "batch_idx": args.batch_idx,
        "source": {
            "hook_meta": str(args.hook_meta),
            "output_json": str(args.output_json),
        },
        "derived": {
            "input_token_count": input_token_count,
            "cached_generated_token_count": cached_generated_token_count,
            "output_token_count": generated_token_count,
            "traj_future_start_token_id": traj_future_start_token_id,
            "traj_future_start_output_index": traj_future_start_output_index if traj_future_start_token_id in output_token_ids else None,
            "offset": offset,
            "runtime_seq_len": runtime_seq_len,
            "future_token_count": args.future_token_count,
            "hook_active_len": hook_active_len,
        },
        "kv_cache": {
            "file": kv_out_path.name,
            "shape": [hook_shape[0], 1, hook_shape[2], hook_shape[3], runtime_seq_len, hook_shape[5]],
            "dtype": hook_kv["dtype"],
            "num_bytes": int(kv_cropped.size * kv_itemsize),
        },
        "kv_cache_lengths": {
            "file": lengths_out_path.name,
            "shape": [1],
            "dtype": "INT32",
            "num_bytes": int(kv_lengths.nbytes),
            "active_batch_size": 1,
            "active_values": [runtime_seq_len],
            "all_values": [runtime_seq_len],
        },
        "position_ids": {
            "file": pos_out_path.name,
            "shape": [3, 1, args.future_token_count],
            "dtype": "INT64",
            "num_bytes": int(position_ids.nbytes),
        },
        "attention_mask": {
            "file": attn_out_path.name,
            "shape": [1, 1, args.future_token_count, runtime_seq_len + args.future_token_count],
            "dtype": "FLOAT16",
            "num_bytes": int(attention_mask.nbytes),
        },
        "rope_deltas": {
            "file": rope_out_path.name,
            "shape": [1, 1],
            "dtype": "INT64",
            "num_bytes": int(rope_out.nbytes),
            "values": [[rope_delta]],
        },
        "offset": {
            "file": offset_out_path.name,
            "shape": [1],
            "dtype": "INT64",
            "num_bytes": int(offset_out.nbytes),
            "values": [offset],
        },
        "layout": KV_LAYOUT,
        "notes": {
            "missing_fields": ["future_token_embeds", "x", "t"],
            "description": "Derived from hook KV + generated token ids. This reproduces runtime-style cache, position_ids, attention_mask, rope_deltas, and offset only.",
        },
    }

    with meta_out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote runtime-style capture to: {args.output_dir}")
    print(f"  hook_active_len={hook_active_len}")
    print(f"  cached_generated_token_count={cached_generated_token_count}")
    print(f"  output_token_count={generated_token_count}")
    print(f"  input_token_count={input_token_count}")
    print(f"  offset={offset}")
    print(f"  runtime_seq_len={runtime_seq_len}")


if __name__ == "__main__":
    main()
