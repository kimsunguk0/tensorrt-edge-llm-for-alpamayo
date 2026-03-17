#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Alpamayo HF visual tensors for the current 16-image input set."
    )
    parser.add_argument(
        "--imageDir",
        required=True,
        help="Directory containing 16 PNG images ordered as cam0_f0 .. cam3_f3.",
    )
    parser.add_argument(
        "--outputDir",
        required=True,
        help="Directory to write .npy dumps.",
    )
    parser.add_argument(
        "--hfModel",
        default="nvidia/Alpamayo-R1-10B",
        help="HF model repo for Alpamayo visual weights.",
    )
    parser.add_argument(
        "--processorName",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF processor repo used for image preprocessing.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for visual forward.",
    )
    parser.add_argument(
        "--compareDir",
        default="",
        help="Optional directory containing TRT visual dump files to compare against.",
    )
    return parser.parse_args()


def create_message(frames: list[Image.Image]) -> list[dict]:
    hist_placeholder = (
        "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": (
                        f"{hist_placeholder}output the chain-of-thought reasoning of the "
                        "driving process, then output the future trajectory."
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def load_images(image_dir: str) -> list[Image.Image]:
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )
    if len(paths) != 16:
        raise ValueError(f"Expected 16 PNG images in {image_dir}, found {len(paths)}")
    return [Image.open(path).convert("RGB") for path in paths]


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def compare_arrays(ref: np.ndarray, cand: np.ndarray) -> dict:
    diff = ref.astype(np.float64) - cand.astype(np.float64)
    return {
        "shape_match": list(ref.shape) == list(cand.shape),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "max_abs_diff": float(np.abs(diff).max()),
        "cosine": cosine_similarity(ref, cand),
        "exact_match": bool(np.array_equal(ref, cand)),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outputDir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = load_images(args.imageDir)
    processor = AutoProcessor.from_pretrained(
        args.processorName,
        min_pixels=163840,
        max_pixels=196608,
    )
    text = processor.apply_chat_template(
        create_message(frames),
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=[text],
        images=frames,
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].reshape(-1, inputs["pixel_values"].shape[-1]).float()
    image_grid_thw = inputs["image_grid_thw"].cpu()

    shard_path = hf_hub_download(args.hfModel, "model-00001-of-00005.safetensors")
    visual_state = {}
    for key, value in load_file(shard_path).items():
        if key.startswith("vlm.model.visual."):
            visual_state[key.removeprefix("vlm.model.visual.")] = value

    config = AutoConfig.from_pretrained(args.processorName)
    vision_model = Qwen3VLVisionModel(config.vision_config)
    missing, unexpected = vision_model.load_state_dict(visual_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected state mismatch. missing={missing}, unexpected={unexpected}")

    device = torch.device(args.device)
    vision_model = vision_model.to(device=device, dtype=torch.float16).eval()

    with torch.inference_mode():
        visual_out = vision_model(
            hidden_states=pixel_values.to(device=device, dtype=torch.float16),
            grid_thw=image_grid_thw.to(device=device),
        )

    last_hidden_state = visual_out.last_hidden_state.detach().cpu().numpy()
    pooler_output = visual_out.pooler_output.detach().cpu().numpy()
    deepstack_features = torch.cat(
        [feat.detach().cpu() for feat in visual_out.deepstack_features], dim=0
    ).numpy()

    save_npy(outdir / "pixel_values.npy", pixel_values.cpu().numpy())
    save_npy(outdir / "image_grid_thw.npy", image_grid_thw.numpy())
    save_npy(outdir / "last_hidden_state.npy", last_hidden_state)
    save_npy(outdir / "pooler_output_flat.npy", pooler_output)
    save_npy(outdir / "deepstack_features.npy", deepstack_features)

    manifest = {
        "hf_model": args.hfModel,
        "processor_name": args.processorName,
        "image_dir": os.path.abspath(args.imageDir),
        "pixel_values_shape": list(pixel_values.shape),
        "image_grid_thw_shape": list(image_grid_thw.shape),
        "last_hidden_state_shape": list(last_hidden_state.shape),
        "pooler_output_flat_shape": list(pooler_output.shape),
        "deepstack_features_shape": list(deepstack_features.shape),
        "device": str(device),
        "dtype": "float16",
    }

    if args.compareDir:
        compare_dir = Path(args.compareDir)
        comparisons = {}
        for name in [
            "pixel_values.npy",
            "image_grid_thw.npy",
            "last_hidden_state.npy",
            "pooler_output_flat.npy",
            "deepstack_features.npy",
        ]:
            ref_path = outdir / name
            cand_path = compare_dir / name
            if ref_path.exists() and cand_path.exists():
                comparisons[name] = compare_arrays(np.load(ref_path), np.load(cand_path))
        manifest["comparisons_against_candidate"] = comparisons

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved Alpamayo visual dump to {outdir}")
    for key in [
        "pixel_values_shape",
        "image_grid_thw_shape",
        "last_hidden_state_shape",
        "pooler_output_flat_shape",
        "deepstack_features_shape",
    ]:
        print(f"{key}: {manifest[key]}")
    if args.compareDir:
        print("compareDir:", os.path.abspath(args.compareDir))
        print(json.dumps(manifest["comparisons_against_candidate"], indent=2))


if __name__ == "__main__":
    main()
