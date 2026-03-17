#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import onnx
from onnx import helper


DEFAULT_SOURCE = Path("/models/onnx/llm_kv")
DEFAULT_TARGET = Path("/root/TensorRT-Edge-LLM/output/onnx/llm_prefill_final_hidden_debug")
SOURCE_VALUE_NAME = "/model/norm/Mul_1_output_0"
OUTPUT_NAME = "hidden_states"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a debug LLM ONNX directory that exposes final prefill hidden states."
    )
    parser.add_argument("--srcDir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dstDir", default=str(DEFAULT_TARGET))
    return parser.parse_args()


def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.name in {"onnx_model.data", "embedding.safetensors"}:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    src_dir = Path(args.srcDir)
    dst_dir = Path(args.dstDir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_model = src_dir / "model.onnx"
    dst_model = dst_dir / "model.onnx"

    for name in (
        "model.onnx",
        "onnx_model.data",
        "config.json",
        "embedding.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "processed_chat_template.json",
        "chat_template.jinja",
    ):
        src = src_dir / name
        if src.exists():
            copy_or_symlink(src, dst_dir / name)

    model = onnx.load(src_model, load_external_data=True)
    existing_outputs = {out.name for out in model.graph.output}
    if OUTPUT_NAME in existing_outputs:
        print(f"{OUTPUT_NAME} already exists in {src_model}")
        return

    value_info_lookup = {vi.name: vi for vi in model.graph.value_info}
    if SOURCE_VALUE_NAME not in value_info_lookup:
        raise KeyError(f"Could not find value_info for {SOURCE_VALUE_NAME}")

    src_vi = value_info_lookup[SOURCE_VALUE_NAME]
    elem_type = src_vi.type.tensor_type.elem_type
    shape = []
    for dim in src_vi.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)

    model.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[SOURCE_VALUE_NAME],
            outputs=[OUTPUT_NAME],
            name="ExposeFinalHiddenStates",
        )
    )
    model.graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, elem_type, shape))

    onnx.save(
        model,
        dst_model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="onnx_model.data",
    )
    print(f"Wrote debug ONNX dir to {dst_dir}")


if __name__ == "__main__":
    main()
