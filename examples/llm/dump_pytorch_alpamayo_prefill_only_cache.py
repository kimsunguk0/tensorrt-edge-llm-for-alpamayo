#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/alpamayo/src")

from alpamayo_r1 import helper  # noqa: E402
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Alpamayo PyTorch prompt-only prefill cache and inputs."
    )
    parser.add_argument("--imageDir", default="/root/TensorRT-Edge-LLM/input/images")
    parser.add_argument(
        "--egoHistoryXYZNpy",
        default="/root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy",
    )
    parser.add_argument(
        "--egoHistoryRotNpy",
        default="/root/TensorRT-Edge-LLM/input/ego/ego_history_rot.npy",
    )
    parser.add_argument("--hfModel", default="nvidia/Alpamayo-R1-10B")
    parser.add_argument(
        "--outputDir",
        default="/root/TensorRT-Edge-LLM/output/kv_cache/pytorch_alpamayo_prefill_only_cache",
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


def save_raw(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


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

    aligned = []
    for feat in deepstack_image_embeds:
        full = feat.new_zeros(inputs_embeds.shape)
        full[image_mask, :] = feat.to(full.device, full.dtype)
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
    input_ids = tokenized_data["input_ids"]
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )

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

        outputs = model.vlm(
            input_ids=input_ids,
            attention_mask=tokenized_data.get("attention_mask"),
            pixel_values=tokenized_data.get("pixel_values"),
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    prompt_cache = outputs.past_key_values
    rope_deltas = outputs.rope_deltas if getattr(outputs, "rope_deltas", None) is not None else model.vlm.model.rope_deltas
    position_ids, rope_deltas_check = model.vlm.model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=tokenized_data.get("image_grid_thw"),
        video_grid_thw=tokenized_data.get("video_grid_thw"),
        attention_mask=tokenized_data.get("attention_mask"),
    )
    if not torch.equal(rope_deltas, rope_deltas_check):
        raise RuntimeError("rope_deltas mismatch between forward output and get_rope_index()")

    hidden_states = outputs.hidden_states[-1].detach().to(dtype=torch.float16).cpu().numpy()

    legacy_cache = prompt_cache.to_legacy_cache()
    key_values = []
    for key, value in legacy_cache:
        key_values.append(torch.stack([key, value], dim=1))
    kv_tensor = torch.stack(key_values, dim=0).detach().cpu()

    kv_bf16 = kv_tensor.to(dtype=torch.bfloat16)
    kv_fp16 = kv_tensor.to(dtype=torch.float16)
    kv_shape = list(kv_fp16.shape)

    input_ids_np = tensor_to_numpy(input_ids.to(dtype=torch.int64))
    inputs_embeds_prefill_fp16 = tensor_to_numpy(inputs_embeds_prefill.to(dtype=torch.float16))
    deepstack_embeds_aligned_fp16 = [
        tensor_to_numpy(t.to(dtype=torch.float16)) for t in deepstack_embeds_aligned
    ]
    position_ids_np = tensor_to_numpy(position_ids.to(dtype=torch.int64))
    rope_deltas_np = tensor_to_numpy(rope_deltas.to(dtype=torch.int64))
    kv_cache_lengths_np = np.array([prompt_cache.get_seq_length()], dtype=np.int32)
    kv_bf16_u16 = kv_bf16.view(torch.uint16).numpy()
    kv_fp16_np = kv_fp16.numpy()

    save_raw(output_dir / "kv_cache_bfloat16_request_0.bin", kv_bf16_u16)
    save_raw(output_dir / "kv_cache_float16_request_0.bin", kv_fp16_np)
    save_raw(output_dir / "kv_cache_lengths_request_0.bin", kv_cache_lengths_np)
    save_raw(output_dir / "input_ids_request_0.bin", input_ids_np)
    save_raw(output_dir / "inputs_embeds_prefill_float16_request_0.bin", inputs_embeds_prefill_fp16)
    save_raw(output_dir / "position_ids_request_0.bin", position_ids_np)
    save_raw(output_dir / "rope_deltas_request_0.bin", rope_deltas_np)
    save_raw(output_dir / "prefill_last_hidden_state_float16_request_0.bin", hidden_states)
    for idx, fp16_arr in enumerate(deepstack_embeds_aligned_fp16):
        save_raw(output_dir / f"deepstack_embeds_aligned_{idx}_float16_request_0.bin", fp16_arr)

    meta = {
        "request_idx": 0,
        "hf_model": args.hfModel,
        "derived": {
            "prompt_cache_seq_len": int(prompt_cache.get_seq_length()),
            "input_token_count": int(input_ids.shape[1]),
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
        "deepstack_embeds_aligned": [
            {
                "index": idx,
                "float16_file": f"deepstack_embeds_aligned_{idx}_float16_request_0.bin",
                "shape": list(deepstack_embeds_aligned_fp16[idx].shape),
                "dtype_float16": "FLOAT16",
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
        "rope_deltas": {
            "file": "rope_deltas_request_0.bin",
            "shape": list(rope_deltas_np.shape),
            "dtype": "INT64",
            "values": rope_deltas_np.tolist(),
        },
        "prefill_last_hidden_state_float16": {
            "file": "prefill_last_hidden_state_float16_request_0.bin",
            "shape": list(hidden_states.shape),
            "dtype": "FLOAT16",
            "num_bytes": int(hidden_states.nbytes),
        },
        "layout": "[numDecoderLayers, batchSize, 2, numKVHeads, seqLen, headDim]",
    }

    with open(output_dir / "pytorch_prefill_only_cache_request_0.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
