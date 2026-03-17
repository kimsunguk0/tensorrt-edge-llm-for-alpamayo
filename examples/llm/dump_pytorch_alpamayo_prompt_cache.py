#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import einops
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/alpamayo/src")

from alpamayo_r1 import helper  # noqa: E402
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor  # noqa: E402
from alpamayo_r1.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token  # noqa: E402
from transformers import StoppingCriteriaList  # noqa: E402
from transformers.generation.logits_process import LogitsProcessorList  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Alpamayo PyTorch prompt_cache and compare it against a TRT runtime-capture."
    )
    parser.add_argument(
        "--imageDir",
        default="/root/TensorRT-Edge-LLM/input/images",
        help="Directory containing the 16 PNG images.",
    )
    parser.add_argument(
        "--egoHistoryXYZNpy",
        default="/root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy",
        help="Path to ego_history_xyz.npy.",
    )
    parser.add_argument(
        "--egoHistoryRotNpy",
        default="/root/TensorRT-Edge-LLM/input/ego/ego_history_rot.npy",
        help="Path to ego_history_rot.npy.",
    )
    parser.add_argument(
        "--hfModel",
        default="nvidia/Alpamayo-R1-10B",
        help="HF model name for Alpamayo.",
    )
    parser.add_argument(
        "--outputDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prompt_cache",
        help="Directory to write PyTorch prompt_cache outputs.",
    )
    parser.add_argument(
        "--compareRuntimeCaptureDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/llm_vlm_nq_fp16_rerun_runtime_capture",
        help="Optional TRT runtime-capture directory to compare against.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="CUDA seed for VLM sampling.",
    )
    parser.add_argument(
        "--topP",
        type=float,
        default=0.98,
        help="top_p for VLM rollout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="temperature for VLM rollout.",
    )
    parser.add_argument(
        "--maxGenerationLength",
        type=int,
        default=256,
        help="max_new_tokens for VLM rollout.",
    )
    return parser.parse_args()


def load_frames(image_dir: Path) -> torch.Tensor:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if len(paths) != 16:
        raise ValueError(f"Expected 16 PNG images in {image_dir}, found {len(paths)}")
    frames = []
    for path in paths:
        arr = np.array(Image.open(path).convert("RGB"), copy=True)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(frames, dim=0)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


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


def save_raw(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def assemble_deepstack_embeds(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    deepstack_image_embeds: list[torch.Tensor],
    image_token_id: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    image_mask = input_ids == image_token_id
    expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        expanded_image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    )

    image_mask_flat = image_mask
    aligned = []
    for feat in deepstack_image_embeds:
        full = feat.new_zeros(inputs_embeds.shape)
        full[image_mask_flat, :] = feat.to(full.device, full.dtype)
        aligned.append(full)
    return inputs_embeds, aligned


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(Path(args.imageDir))
    ego_history_xyz = torch.from_numpy(np.load(args.egoHistoryXYZNpy))
    ego_history_rot = torch.from_numpy(np.load(args.egoHistoryRotNpy))

    model = AlpamayoR1.from_pretrained(args.hfModel, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    messages = helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    tokenized_data = model_inputs["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_embeds_prefill = model.vlm.get_input_embeddings()(input_ids)
        image_embeds_list, deepstack_image_embeds = model.vlm.get_image_features(
            tokenized_data["pixel_values"], tokenized_data["image_grid_thw"]
        )
        image_embeds = torch.cat(image_embeds_list, dim=0).to(
            inputs_embeds_prefill.device, inputs_embeds_prefill.dtype
        )
        inputs_embeds_prefill, deepstack_embeds_aligned = assemble_deepstack_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds_prefill,
            image_embeds=image_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            image_token_id=model.vlm.config.image_token_id,
        )

    generation_config = model.vlm.generation_config
    generation_config.top_p = args.topP
    generation_config.temperature = args.temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = args.maxGenerationLength
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    torch.cuda.manual_seed_all(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **model_inputs["tokenized_data"],
        )

    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    prompt_cache = vlm_outputs.past_key_values

    sequences = vlm_outputs.sequences
    b_star = sequences.shape[0]
    traj_future_start_mask = sequences == eos_token_id
    has_traj_future_start = traj_future_start_mask.any(dim=1)
    traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
    last_token_positions = torch.full((b_star,), sequences.shape[1] - 1, device=sequences.device)
    valid_token_pos_id = torch.where(has_traj_future_start, traj_future_start_positions, last_token_positions)
    offset = valid_token_pos_id + 1

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    position_ids = torch.arange(n_diffusion_tokens, device=sequences.device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    delta = vlm_outputs.rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
        dtype=torch.float32,
        device=sequences.device,
    )
    for i in range(b_star):
        attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(attention_mask.dtype).min

    legacy_cache = prompt_cache.to_legacy_cache()
    key_values = []
    for key, value in legacy_cache:
        key_values.append(torch.stack([key, value], dim=1))
    kv_tensor = torch.stack(key_values, dim=0).detach().cpu()

    kv_bf16 = kv_tensor.to(dtype=torch.bfloat16)
    kv_fp16 = kv_tensor.to(dtype=torch.float16)
    kv_shape = list(kv_fp16.shape)

    output_token_ids = sequences[:, input_ids.shape[1] :].detach().cpu().numpy()
    output_token_ids_int64 = output_token_ids.astype(np.int64, copy=False)
    output_token_count = int(output_token_ids_int64.shape[1])

    kv_bf16_u16 = kv_bf16.view(torch.uint16).numpy()
    kv_fp16_np = kv_fp16.numpy()
    input_ids_np = tensor_to_numpy(input_ids.to(dtype=torch.int64))
    inputs_embeds_prefill_fp16 = tensor_to_numpy(inputs_embeds_prefill.to(dtype=torch.float16))
    inputs_embeds_prefill_bf16_u16 = (
        inputs_embeds_prefill.to(dtype=torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    )
    deepstack_embeds_aligned_fp16 = [
        tensor_to_numpy(t.to(dtype=torch.float16)) for t in deepstack_embeds_aligned
    ]
    deepstack_embeds_aligned_bf16_u16 = [
        t.to(dtype=torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
        for t in deepstack_embeds_aligned
    ]
    position_ids_np = tensor_to_numpy(position_ids.to(dtype=torch.int64))
    attention_mask_fp16 = tensor_to_numpy(attention_mask.to(dtype=torch.float16))
    rope_deltas_np = tensor_to_numpy(vlm_outputs.rope_deltas.to(dtype=torch.int64))
    offset_np = tensor_to_numpy(offset.to(dtype=torch.int64))
    kv_cache_lengths_np = np.array([prompt_cache.get_seq_length()], dtype=np.int32)

    save_raw(output_dir / "kv_cache_bfloat16_request_0.bin", kv_bf16_u16)
    save_raw(output_dir / "kv_cache_float16_request_0.bin", kv_fp16_np)
    save_raw(output_dir / "input_ids_request_0.bin", input_ids_np)
    save_raw(output_dir / "inputs_embeds_prefill_float16_request_0.bin", inputs_embeds_prefill_fp16)
    save_raw(output_dir / "inputs_embeds_prefill_bfloat16_request_0.bin", inputs_embeds_prefill_bf16_u16)
    save_raw(output_dir / "kv_cache_lengths_request_0.bin", kv_cache_lengths_np)
    save_raw(output_dir / "position_ids_request_0.bin", position_ids_np)
    save_raw(output_dir / "attention_mask_request_0.bin", attention_mask_fp16)
    save_raw(output_dir / "rope_deltas_request_0.bin", rope_deltas_np)
    save_raw(output_dir / "offset_request_0.bin", offset_np)
    save_raw(output_dir / "output_token_ids_request_0.bin", output_token_ids_int64)
    for idx, (fp16_arr, bf16_arr) in enumerate(
        zip(deepstack_embeds_aligned_fp16, deepstack_embeds_aligned_bf16_u16)
    ):
        save_raw(output_dir / f"deepstack_embeds_aligned_{idx}_float16_request_0.bin", fp16_arr)
        save_raw(output_dir / f"deepstack_embeds_aligned_{idx}_bfloat16_request_0.bin", bf16_arr)

    meta = {
        "request_idx": 0,
        "hf_model": args.hfModel,
        "seed": args.seed,
        "input": {
            "image_dir": str(Path(args.imageDir).resolve()),
            "ego_history_xyz_npy": str(Path(args.egoHistoryXYZNpy).resolve()),
            "ego_history_rot_npy": str(Path(args.egoHistoryRotNpy).resolve()),
        },
        "derived": {
            "prompt_cache_seq_len": int(prompt_cache.get_seq_length()),
            "input_token_count": int(input_ids.shape[1]),
            "output_token_count": output_token_count,
            "traj_future_start_token_id": int(eos_token_id),
            "traj_future_start_output_index": int(valid_token_pos_id[0].item() - input_ids.shape[1]) if has_traj_future_start[0] else -1,
            "offset": int(offset_np[0]),
            "rope_delta": int(rope_deltas_np[0, 0]),
        },
        "token_ids": {
            "image_token_id": int(model.vlm.config.image_token_id),
            "traj_token_start_idx": int(model.config.traj_token_start_idx),
            "traj_vocab_size": int(model.config.traj_vocab_size),
        },
        "input_ids": {
            "file": "input_ids_request_0.bin",
            "shape": list(input_ids_np.shape),
            "dtype": "INT64",
        },
        "inputs_embeds_prefill_float16": {
            "file": "inputs_embeds_prefill_float16_request_0.bin",
            "shape": list(inputs_embeds_prefill_fp16.shape),
            "dtype": "FLOAT16",
            "num_bytes": int(inputs_embeds_prefill_fp16.nbytes),
        },
        "inputs_embeds_prefill_bfloat16": {
            "file": "inputs_embeds_prefill_bfloat16_request_0.bin",
            "shape": list(inputs_embeds_prefill.shape),
            "dtype": "BFLOAT16",
            "num_bytes": int(inputs_embeds_prefill_bf16_u16.nbytes),
        },
        "deepstack_embeds_aligned": [
            {
                "index": idx,
                "float16_file": f"deepstack_embeds_aligned_{idx}_float16_request_0.bin",
                "bfloat16_file": f"deepstack_embeds_aligned_{idx}_bfloat16_request_0.bin",
                "shape": list(deepstack_embeds_aligned_fp16[idx].shape),
                "dtype_float16": "FLOAT16",
                "dtype_bfloat16": "BFLOAT16",
            }
            for idx in range(len(deepstack_embeds_aligned_fp16))
        ],
        "kv_cache_bfloat16": {
            "file": "kv_cache_bfloat16_request_0.bin",
            "shape": kv_shape,
            "dtype": "BFLOAT16",
            "num_bytes": int(kv_bf16_u16.nbytes),
        },
        "kv_cache_float16": {
            "file": "kv_cache_float16_request_0.bin",
            "shape": kv_shape,
            "dtype": "FLOAT16",
            "num_bytes": int(kv_fp16_np.nbytes),
        },
        "kv_cache_lengths": {
            "file": "kv_cache_lengths_request_0.bin",
            "shape": [1],
            "dtype": "INT32",
            "active_values": kv_cache_lengths_np.tolist(),
        },
        "position_ids": {
            "file": "position_ids_request_0.bin",
            "shape": list(position_ids_np.shape),
            "dtype": "INT64",
        },
        "attention_mask": {
            "file": "attention_mask_request_0.bin",
            "shape": list(attention_mask_fp16.shape),
            "dtype": "FLOAT16",
        },
        "rope_deltas": {
            "file": "rope_deltas_request_0.bin",
            "shape": list(rope_deltas_np.shape),
            "dtype": "INT64",
            "values": rope_deltas_np.tolist(),
        },
        "offset_file": {
            "file": "offset_request_0.bin",
            "shape": list(offset_np.shape),
            "dtype": "INT64",
            "values": offset_np.tolist(),
        },
        "output_token_ids": {
            "file": "output_token_ids_request_0.bin",
            "shape": list(output_token_ids_int64.shape),
            "dtype": "INT64",
        },
        "layout": "[numDecoderLayers, batchSize, 2, numKVHeads, seqLen, headDim]",
    }

    compare_dir = Path(args.compareRuntimeCaptureDir)
    if compare_dir.exists():
        trt_meta = json.loads((compare_dir / "runtime_capture_request_0.json").read_text())
        trt_kv = np.fromfile(compare_dir / trt_meta["kv_cache"]["file"], dtype=np.float16).reshape(
            trt_meta["kv_cache"]["shape"]
        )
        trt_position_ids = np.fromfile(compare_dir / trt_meta["position_ids"]["file"], dtype=np.int64).reshape(
            trt_meta["position_ids"]["shape"]
        )
        trt_attention_mask = np.fromfile(compare_dir / trt_meta["attention_mask"]["file"], dtype=np.float16).reshape(
            trt_meta["attention_mask"]["shape"]
        )

        meta["comparison_against_trt_runtime_capture"] = {
            "runtime_capture_dir": str(compare_dir.resolve()),
            "kv_cache_float16": compare_arrays(kv_fp16_np, trt_kv),
            "position_ids": compare_arrays(position_ids_np, trt_position_ids),
            "attention_mask": compare_arrays(attention_mask_fp16, trt_attention_mask),
            "offset_match": int(offset_np[0]) == int(trt_meta["offset"]["values"][0]),
            "rope_delta_match": int(rope_deltas_np[0, 0]) == int(trt_meta["rope_deltas"]["values"][0][0]),
        }

    (output_dir / "pytorch_prompt_cache_request_0.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
