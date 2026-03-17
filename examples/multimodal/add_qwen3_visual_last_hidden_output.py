#!/usr/bin/env python3
"""
Create a debug Qwen3-VL visual ONNX by exposing the pre-merger hidden state.
"""

import argparse
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


DEFAULT_SOURCE_TENSOR = "/blocks.26/Add_1_output_0"
DEFAULT_OUTPUT_NAME = "last_hidden_state"


def find_value_info(model: onnx.ModelProto, tensor_name: str):
    for coll in (model.graph.value_info, model.graph.output, model.graph.input):
        for value in coll:
            if value.name == tensor_name:
                return value
    return None


def clone_shape(value_info, output_name: str):
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return helper.make_tensor_value_info(output_name, tensor_type.elem_type, dims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--source-tensor", default=DEFAULT_SOURCE_TENSOR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_model = src_dir / "model.onnx"
    dst_model = dst_dir / "model.onnx"
    for name in ("model.onnx", "onnx_model.data", "config.json", "preprocessor_config.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)

    model = onnx.load(dst_model.as_posix())
    model = shape_inference.infer_shapes(model)

    source_value = find_value_info(model, args.source_tensor)
    if source_value is None:
        raise RuntimeError(f"Could not find value_info for tensor: {args.source_tensor}")

    identity_node = helper.make_node(
        "Identity",
        inputs=[args.source_tensor],
        outputs=[args.output_name],
        name="ExposeLastHiddenState",
    )
    model.graph.node.append(identity_node)

    output_value = clone_shape(source_value, args.output_name)
    output_value.type.tensor_type.elem_type = TensorProto.FLOAT16
    model.graph.output.append(output_value)

    onnx.save(model, dst_model.as_posix())
    print(dst_model)


if __name__ == "__main__":
    main()
