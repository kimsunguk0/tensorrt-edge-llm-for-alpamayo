#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch Alpamayo prefill inputs / prompt cache against TRT prefill inputs / runtime capture."
    )
    parser.add_argument(
        "--pytorchDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prompt_cache_prefill_debug",
    )
    parser.add_argument(
        "--trtPrefillDir",
        default="/root/TensorRT-Edge-LLM/output/prefill_inputs/llm_vlm_nq_fp16_prefill_debug/request_0",
    )
    parser.add_argument(
        "--trtRuntimeCaptureDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/llm_vlm_nq_fp16_prefill_debug_runtime_capture",
    )
    parser.add_argument(
        "--outputJson",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prompt_cache_prefill_debug/compare_prefill_and_kv_against_trt_fp16_prefill_debug.json",
    )
    parser.add_argument(
        "--imageDir",
        default="/root/TensorRT-Edge-LLM/input/images",
    )
    parser.add_argument(
        "--blockThreshold",
        type=float,
        default=0.01,
        help="Threshold used to flag the first materially divergent block.",
    )
    return parser.parse_args()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def contiguous_spans(values: np.ndarray, categories: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
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
    head = ids[start : start + preview_len].tolist()
    tail = ids[max(start, end - preview_len) : end].tolist()
    return {"head_ids": head, "tail_ids": tail}


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
    trt_prefill_dir = Path(args.trtPrefillDir)
    trt_runtime_dir = Path(args.trtRuntimeCaptureDir)
    output_json = Path(args.outputJson)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_span_annotations = build_image_span_annotations(Path(args.imageDir))

    pt_meta = load_json(pt_dir / "pytorch_prompt_cache_request_0.json")
    trt_prefill_meta = load_json(trt_prefill_dir / "prefill_inputs_request_0.json")
    trt_runtime_meta = load_json(trt_runtime_dir / "runtime_capture_request_0.json")

    pt_input_ids = np.fromfile(pt_dir / pt_meta["input_ids"]["file"], dtype=np.int64).reshape(pt_meta["input_ids"]["shape"])
    trt_input_ids = np.fromfile(
        trt_prefill_dir / trt_prefill_meta["input_ids"]["file"], dtype=np.int32
    ).astype(np.int64).reshape(trt_prefill_meta["input_ids"]["shape"])

    image_token_id = int(pt_meta["token_ids"]["image_token_id"])
    traj_token_start_idx = int(pt_meta["token_ids"]["traj_token_start_idx"])
    traj_vocab_size = int(pt_meta["token_ids"]["traj_vocab_size"])
    # TRT encodes image-token positions as vocabSize + running_index rather than repeated image_token_id.
    # Infer the cut by locating the first token larger than the largest normal token id in the PyTorch prompt.
    normal_token_upper_bound = max(image_token_id, traj_token_start_idx + traj_vocab_size - 1)
    trt_image_mask = trt_input_ids[0] > normal_token_upper_bound
    pt_image_mask = pt_input_ids[0] == image_token_id

    non_image_mask = ~pt_image_mask
    prefill_input_report = {
        "input_ids_raw": compare_arrays(pt_input_ids, trt_input_ids),
        "input_ids_non_image_positions": compare_arrays(pt_input_ids[:, non_image_mask], trt_input_ids[:, non_image_mask]),
        "image_position_count": int(pt_image_mask.sum()),
        "trt_image_position_count_on_pytorch_image_positions": int((trt_image_mask & pt_image_mask).sum()),
        "trt_extra_gt_vocab_positions_outside_image_block": int((trt_image_mask & ~pt_image_mask).sum()),
        "image_token_scheme": {
            "pytorch": f"repeated image_token_id={image_token_id}",
            "trt": "vocabSize + running_image_token_index",
            "trt_first_image_token_ids": trt_input_ids[0, trt_image_mask][:8].tolist(),
        },
    }

    pt_inputs_embeds = np.fromfile(
        pt_dir / pt_meta["inputs_embeds_prefill_float16"]["file"], dtype=np.float16
    ).reshape(pt_meta["inputs_embeds_prefill_float16"]["shape"])
    trt_inputs_embeds = np.fromfile(
        trt_prefill_dir / trt_prefill_meta["inputs_embeds"]["file"], dtype=np.float16
    ).reshape(trt_prefill_meta["inputs_embeds"]["shape"])
    prefill_input_report["inputs_embeds"] = compare_arrays(pt_inputs_embeds, trt_inputs_embeds)

    prefill_input_report["deepstack_embeds"] = []
    for idx in range(len(pt_meta["deepstack_embeds_aligned"])):
        pt_deep = np.fromfile(
            pt_dir / pt_meta["deepstack_embeds_aligned"][idx]["float16_file"], dtype=np.float16
        ).reshape(pt_meta["deepstack_embeds_aligned"][idx]["shape"])
        trt_deep = np.fromfile(
            trt_prefill_dir / trt_prefill_meta["deepstack_embeds"][idx]["file"], dtype=np.float16
        ).reshape(trt_prefill_meta["deepstack_embeds"][idx]["shape"])
        item = compare_arrays(pt_deep, trt_deep)
        item["index"] = idx
        prefill_input_report["deepstack_embeds"].append(item)

    pt_kv = np.fromfile(pt_dir / pt_meta["kv_cache_float16"]["file"], dtype=np.float16).reshape(
        pt_meta["kv_cache_float16"]["shape"]
    )
    trt_kv = np.fromfile(trt_runtime_dir / trt_runtime_meta["kv_cache"]["file"], dtype=np.float16).reshape(
        trt_runtime_meta["kv_cache"]["shape"]
    )
    kv_report = {
        "overall": compare_arrays(pt_kv, trt_kv),
        "position_ids_exact_match": bool(
            np.array_equal(
                np.fromfile(pt_dir / pt_meta["position_ids"]["file"], dtype=np.int64).reshape(pt_meta["position_ids"]["shape"]),
                np.fromfile(trt_runtime_dir / trt_runtime_meta["position_ids"]["file"], dtype=np.int64).reshape(
                    trt_runtime_meta["position_ids"]["shape"]
                ),
            )
        ),
        "attention_mask_exact_match": bool(
            np.array_equal(
                np.fromfile(pt_dir / pt_meta["attention_mask"]["file"], dtype=np.float16).reshape(
                    pt_meta["attention_mask"]["shape"]
                ),
                np.fromfile(trt_runtime_dir / trt_runtime_meta["attention_mask"]["file"], dtype=np.float16).reshape(
                    trt_runtime_meta["attention_mask"]["shape"]
                ),
            )
        ),
    }

    input_token_count = int(pt_meta["derived"]["input_token_count"])
    offset = int(pt_meta["derived"]["offset"])
    generated_prefix_len = offset - input_token_count
    generated_prefix_ids = np.fromfile(
        pt_dir / pt_meta["output_token_ids"]["file"], dtype=np.int64
    ).reshape(pt_meta["output_token_ids"]["shape"])[0, :generated_prefix_len]

    categories: list[str] = []
    for token_id in pt_input_ids[0]:
        if token_id == image_token_id:
            categories.append("image_tokens")
        elif traj_token_start_idx <= token_id < traj_token_start_idx + traj_vocab_size:
            categories.append("traj_value_tokens")
        else:
            categories.append("text_or_special")
    spans = contiguous_spans(pt_input_ids[0], categories)

    span_reports = []
    threshold = float(args.blockThreshold)
    first_divergent_span = None
    for start, end, category in spans:
        diff = np.abs(pt_kv[:, 0, :, :, start:end, :].astype(np.float64) - trt_kv[:, 0, :, :, start:end, :].astype(np.float64))
        per_layer = diff.mean(axis=(1, 2, 3, 4))
        report = {
            "kind": "input_prompt",
            "category": category,
            "start": start,
            "end": end,
            "length": end - start,
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "first_layer_over_threshold": next((int(i) for i, v in enumerate(per_layer) if v > threshold), None),
            "worst_layer": int(np.argmax(per_layer)),
            "worst_layer_mean_abs_diff": float(np.max(per_layer)),
        }
        report.update(summarize_id_span(pt_input_ids[0], start, end))
        report.update(image_span_annotations.get((start, end), {}))
        span_reports.append(report)
        if first_divergent_span is None and report["mean_abs_diff"] > threshold:
            first_divergent_span = report

    if generated_prefix_len > 0:
        start = input_token_count
        end = offset
        diff = np.abs(pt_kv[:, 0, :, :, start:end, :].astype(np.float64) - trt_kv[:, 0, :, :, start:end, :].astype(np.float64))
        per_layer = diff.mean(axis=(1, 2, 3, 4))
        gen_report = {
            "kind": "generated_prefix",
            "category": "generated_prefix_to_traj_future_start",
            "start": start,
            "end": end,
            "length": end - start,
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "first_layer_over_threshold": next((int(i) for i, v in enumerate(per_layer) if v > threshold), None),
            "worst_layer": int(np.argmax(per_layer)),
            "worst_layer_mean_abs_diff": float(np.max(per_layer)),
            "head_ids": generated_prefix_ids[:8].tolist(),
            "tail_ids": generated_prefix_ids[-8:].tolist(),
        }
        span_reports.append(gen_report)
        if first_divergent_span is None and gen_report["mean_abs_diff"] > threshold:
            first_divergent_span = gen_report

    span_reports_sorted = sorted(span_reports, key=lambda x: x["mean_abs_diff"], reverse=True)
    kv_report["token_block_analysis"] = {
        "threshold": threshold,
        "first_divergent_span": first_divergent_span,
        "top_spans_by_mean_abs_diff": span_reports_sorted[:8],
        "all_spans": span_reports,
    }

    report = {
        "paths": {
            "pytorch_dir": str(pt_dir.resolve()),
            "trt_prefill_dir": str(trt_prefill_dir.resolve()),
            "trt_runtime_capture_dir": str(trt_runtime_dir.resolve()),
        },
        "meta": {
            "input_token_count": input_token_count,
            "offset": offset,
            "generated_prefix_len": generated_prefix_len,
            "traj_future_start_output_index": int(pt_meta["derived"]["traj_future_start_output_index"]),
        },
        "prefill_input_alignment": prefill_input_report,
        "kv_alignment": kv_report,
    }
    output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
