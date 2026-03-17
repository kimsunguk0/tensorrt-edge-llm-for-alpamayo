#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch Alpamayo prompt-only prefill KV against TRT prompt-only hook KV."
    )
    parser.add_argument(
        "--pytorchDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prefill_only_cache",
    )
    parser.add_argument(
        "--trtHookDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/llm_vlm_nq_fp16_prefill_only_hook",
    )
    parser.add_argument(
        "--imageDir",
        default="/root/TensorRT-Edge-LLM/input/images",
    )
    parser.add_argument(
        "--outputJson",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prefill_only_cache/compare_against_trt_prefill_only_hook.json",
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
    trt_dir = Path(args.trtHookDir)
    output_json = Path(args.outputJson)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pt_meta = load_json(pt_dir / "pytorch_prefill_only_cache_request_0.json")
    trt_meta = load_json(trt_dir / "kv_cache_request_0.json")

    pt_input_ids = np.fromfile(pt_dir / pt_meta["input_ids"]["file"], dtype=np.int64).reshape(pt_meta["input_ids"]["shape"])
    pt_kv = np.fromfile(pt_dir / pt_meta["kv_cache_float16"]["file"], dtype=np.float16).reshape(pt_meta["kv_cache_float16"]["shape"])
    pt_pos = np.fromfile(pt_dir / pt_meta["position_ids"]["file"], dtype=np.int64).reshape(pt_meta["position_ids"]["shape"])
    pt_rope = np.fromfile(pt_dir / pt_meta["rope_deltas"]["file"], dtype=np.int64).reshape(pt_meta["rope_deltas"]["shape"])

    trt_kv_full = np.fromfile(trt_dir / trt_meta["kv_cache"]["file"], dtype=np.float16).reshape(trt_meta["kv_cache"]["shape"])
    trt_active_len = int(trt_meta["kv_cache_lengths"]["active_values"][0])
    trt_kv = trt_kv_full[..., :trt_active_len, :]
    trt_pos_full = np.fromfile(trt_dir / trt_meta["position_ids"]["file"], dtype=np.int64).reshape(trt_meta["position_ids"]["shape"])
    trt_pos = np.transpose(trt_pos_full[:, :, :trt_active_len], (1, 0, 2))
    trt_rope = np.fromfile(trt_dir / trt_meta["rope_deltas"]["file"], dtype=np.int64).reshape(trt_meta["rope_deltas"]["shape"])

    image_token_id = int(pt_meta["token_ids"]["image_token_id"])
    traj_token_start_idx = int(pt_meta["token_ids"]["traj_token_start_idx"])
    traj_vocab_size = int(pt_meta["token_ids"]["traj_vocab_size"])

    kv_report = {
        "overall": compare_arrays(pt_kv, trt_kv),
        "position_ids": compare_arrays(pt_pos, trt_pos),
        "rope_deltas": compare_arrays(pt_rope, trt_rope),
        "trt_active_len": trt_active_len,
        "pt_seq_len": int(pt_meta["derived"]["prompt_cache_seq_len"]),
    }

    diff = np.abs(pt_kv.astype(np.float64) - trt_kv.astype(np.float64))
    kv_report["k_mean_abs_diff"] = float(diff[:, :, 0].mean())
    kv_report["v_mean_abs_diff"] = float(diff[:, :, 1].mean())
    if pt_kv.shape[-2] >= 256:
        kv_report["first256_mean_abs_diff"] = float(diff[..., :256, :].mean())
        mid = pt_kv.shape[-2] // 2
        kv_report["mid256_mean_abs_diff"] = float(diff[..., mid - 128 : mid + 128, :].mean())
        kv_report["last256_mean_abs_diff"] = float(diff[..., -256:, :].mean())
    per_layer = diff.mean(axis=(1, 2, 3, 4, 5))
    kv_report["top_layers"] = [
        {"layer": int(i), "mean_abs_diff": float(per_layer[i])}
        for i in np.argsort(per_layer)[::-1][:8]
    ]

    categories = []
    for token_id in pt_input_ids[0]:
        if token_id == image_token_id:
            categories.append("image_tokens")
        elif traj_token_start_idx <= token_id < traj_token_start_idx + traj_vocab_size:
            categories.append("traj_value_tokens")
        else:
            categories.append("text_or_special")
    spans = contiguous_spans(categories)
    annotations = build_image_span_annotations(Path(args.imageDir))
    threshold = float(args.blockThreshold)
    span_reports = []
    first_divergent = None
    for start, end, category in spans:
        block_diff = np.abs(
            pt_kv[:, 0, :, :, start:end, :].astype(np.float64)
            - trt_kv[:, 0, :, :, start:end, :].astype(np.float64)
        )
        layer_means = block_diff.mean(axis=(1, 2, 3, 4))
        report = {
            "kind": "input_prompt",
            "category": category,
            "start": start,
            "end": end,
            "length": end - start,
            "mean_abs_diff": float(block_diff.mean()),
            "max_abs_diff": float(block_diff.max()),
            "first_layer_over_threshold": next((int(i) for i, v in enumerate(layer_means) if v > threshold), None),
            "worst_layer": int(np.argmax(layer_means)),
            "worst_layer_mean_abs_diff": float(np.max(layer_means)),
        }
        report.update(summarize_id_span(pt_input_ids[0], start, end))
        report.update(annotations.get((start, end), {}))
        span_reports.append(report)
        if first_divergent is None and report["mean_abs_diff"] > threshold:
            first_divergent = report

    span_reports_sorted = sorted(span_reports, key=lambda x: x["mean_abs_diff"], reverse=True)

    report = {
        "metadata": {
            "pytorch_dir": str(pt_dir.resolve()),
            "trt_hook_dir": str(trt_dir.resolve()),
        },
        "kv_alignment": kv_report,
        "token_block_analysis": {
            "threshold": threshold,
            "first_material_divergence": first_divergent,
            "top_blocks_by_mean_abs_diff": span_reports_sorted[:16],
        },
    }

    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
