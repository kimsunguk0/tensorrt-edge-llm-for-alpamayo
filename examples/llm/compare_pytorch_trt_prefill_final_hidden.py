#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch Alpamayo prefill final post-norm hidden states against TRT prefill hidden dump."
    )
    parser.add_argument(
        "--pytorchDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prefill_only_cache",
    )
    parser.add_argument(
        "--trtDir",
        default="/root/TensorRT-Edge-LLM/output/prefill_hidden_states/llm_vlm_nq_prefill_final_hidden_debug/request_0",
    )
    parser.add_argument(
        "--imageDir",
        default="/root/TensorRT-Edge-LLM/input/images",
    )
    parser.add_argument(
        "--outputJson",
        default="/root/TensorRT-Edge-LLM/output/prefill_hidden_states/llm_vlm_nq_prefill_final_hidden_debug/compare_against_pytorch_post_norm_hidden_detailed.json",
    )
    parser.add_argument("--blockThreshold", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "exact_match": bool(np.array_equal(a, b)),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "cosine": cosine_similarity(a, b),
    }


def contiguous_spans(categories: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    current = categories[0]
    for idx in range(1, len(categories)):
        if categories[idx] != current:
            spans.append((start, idx, current))
            start = idx
            current = categories[idx]
    spans.append((start, len(categories), current))
    return spans


def summarize_id_span(ids: np.ndarray, start: int, end: int) -> dict:
    preview_len = min(8, end - start)
    return {
        "head_ids": ids[start : start + preview_len].tolist(),
        "tail_ids": ids[max(start, end - preview_len) : end].tolist(),
    }


def build_image_span_annotations(image_dir: Path) -> dict[tuple[int, int], dict]:
    annotations: dict[tuple[int, int], dict] = {}
    image_paths = sorted(p.name for p in image_dir.glob("*.png"))
    start = 21
    for idx, image_name in enumerate(image_paths):
        annotations[(start, start + 180)] = {"image_index": idx, "image_file": image_name}
        if idx + 1 < len(image_paths):
            annotations[(start + 180, start + 182)] = {
                "between_images": [image_name, image_paths[idx + 1]],
                "token_meaning": ["<|vision_end|>", "<|vision_start|>"],
            }
        start += 182
    return annotations


def main() -> None:
    args = parse_args()
    pt_dir = Path(args.pytorchDir)
    trt_dir = Path(args.trtDir)
    out_json = Path(args.outputJson)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pt_meta = load_json(pt_dir / "pytorch_prefill_only_cache_request_0.json")
    trt_meta = load_json(trt_dir / "prefill_hidden_states_request_0.json")

    pt_input_ids = np.fromfile(pt_dir / pt_meta["input_ids"]["file"], dtype=np.int64).reshape(pt_meta["input_ids"]["shape"])
    pt_full = np.fromfile(
        pt_dir / pt_meta["prefill_post_norm_hidden_state_float16"]["file"], dtype=np.float16
    ).reshape(pt_meta["prefill_post_norm_hidden_state_float16"]["shape"])
    trt = np.fromfile(
        trt_dir / trt_meta["prefill_hidden_states"]["file"], dtype=np.float16
    ).reshape(trt_meta["prefill_hidden_states"]["shape"])

    # TRT dump is one token shorter; choose the better-aligned crop from PyTorch.
    pt_head = pt_full[:, : trt.shape[1], :]
    pt_tail = pt_full[:, -trt.shape[1] :, :]
    head_cmp = compare_arrays(pt_head, trt)
    tail_cmp = compare_arrays(pt_tail, trt)
    use_head = head_cmp["mean_abs_diff"] <= tail_cmp["mean_abs_diff"]
    pt = pt_head if use_head else pt_tail
    alignment = {
        "strategy": "drop_last_token_from_pytorch" if use_head else "drop_first_token_from_pytorch",
        "head_comparison": head_cmp,
        "tail_comparison": tail_cmp,
    }

    report = {
        "metadata": {
            "pytorch_dir": str(pt_dir.resolve()),
            "trt_dir": str(trt_dir.resolve()),
        },
        "alignment": alignment,
        "overall": compare_arrays(pt, trt),
    }

    diff = np.abs(pt.astype(np.float64) - trt.astype(np.float64))
    report["overall"]["first256_mean_abs_diff"] = float(diff[:, :256, :].mean())
    mid = pt.shape[1] // 2
    report["overall"]["mid256_mean_abs_diff"] = float(diff[:, mid - 128 : mid + 128, :].mean())
    report["overall"]["last256_mean_abs_diff"] = float(diff[:, -256:, :].mean())

    image_token_id = int(pt_meta["token_ids"]["image_token_id"])
    traj_token_start_idx = int(pt_meta["token_ids"]["traj_token_start_idx"])
    traj_vocab_size = int(pt_meta["token_ids"]["traj_vocab_size"])

    categories = []
    for token_id in pt_input_ids[0, : pt.shape[1]]:
        if token_id == image_token_id:
            categories.append("image_tokens")
        elif traj_token_start_idx <= token_id < traj_token_start_idx + traj_vocab_size:
            categories.append("traj_value_tokens")
        else:
            categories.append("text_or_special")

    annotations = build_image_span_annotations(Path(args.imageDir))
    spans = contiguous_spans(categories)
    threshold = float(args.blockThreshold)
    span_reports = []
    first_divergent = None
    for start, end, category in spans:
        block_diff = np.abs(
            pt[:, start:end, :].astype(np.float64) - trt[:, start:end, :].astype(np.float64)
        )
        per_token = block_diff.mean(axis=(0, 2))
        report_entry = {
            "kind": "input_prompt",
            "category": category,
            "start": start,
            "end": end,
            "length": end - start,
            "mean_abs_diff": float(block_diff.mean()),
            "max_abs_diff": float(block_diff.max()),
            "first_token_over_threshold": next((int(start + i) for i, v in enumerate(per_token) if v > threshold), None),
            "worst_token": int(start + np.argmax(per_token)),
            "worst_token_mean_abs_diff": float(np.max(per_token)),
        }
        report_entry.update(summarize_id_span(pt_input_ids[0], start, end))
        report_entry.update(annotations.get((start, end), {}))
        span_reports.append(report_entry)
        if first_divergent is None and report_entry["mean_abs_diff"] > threshold:
            first_divergent = report_entry

    span_reports_sorted = sorted(span_reports, key=lambda x: x["mean_abs_diff"], reverse=True)
    report["token_block_analysis"] = {
        "threshold": threshold,
        "first_material_divergence": first_divergent,
        "top_blocks_by_mean_abs_diff": span_reports_sorted[:16],
    }

    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
