#!/usr/bin/env python3
"""Convert a static Alpamayo FM ONNX into a Thor-friendly MXFP8 ONNX."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import onnx
from modelopt.onnx.quantization.qdq_utils import quantize_weights_to_mxfp8
from onnx import TensorProto, helper, numpy_helper

BLOCK_SIZE = 32
TRT_DOMAIN = "trt"
TRT_OPSET = 1
MIN_ONNX_OPSET = 23
MIN_IR_VERSION = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", type=Path, required=True, help="Path to the source FP16 ONNX")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model.onnx and onnx_model.data will be written",
    )
    return parser.parse_args()


def ensure_opsets(model: onnx.ModelProto) -> None:
    found_default = False
    found_trt = False
    for opset in model.opset_import:
        if opset.domain == "":
            found_default = True
            opset.version = max(int(opset.version), MIN_ONNX_OPSET)
        elif opset.domain == TRT_DOMAIN:
            found_trt = True
            opset.version = max(int(opset.version), TRT_OPSET)
    if not found_default:
        model.opset_import.append(helper.make_operatorsetid("", MIN_ONNX_OPSET))
    if not found_trt:
        model.opset_import.append(helper.make_operatorsetid(TRT_DOMAIN, TRT_OPSET))
    model.ir_version = max(int(model.ir_version), MIN_IR_VERSION)


def initializer_arrays(model: onnx.ModelProto) -> dict[str, Any]:
    return {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}


def initializer_objects(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def make_constant_node(name: str, output_name: str, value: float) -> onnx.NodeProto:
    tensor = helper.make_tensor(
        name=f"{name}_value",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[float(value)],
    )
    return helper.make_node("Constant", [], [output_name], name=name, value=tensor)


def derive_quant_weight_name(node_name: str) -> str:
    base = node_name.rsplit("/", 1)[0].lstrip("/")
    return base.replace("/", ".") + ".weight"


def node_base(node_name: str) -> str:
    return node_name.rsplit("/", 1)[0]


def is_quantizable_linear(node: onnx.NodeProto, init_arrays: dict[str, Any]) -> tuple[bool, str | None]:
    if node.op_type not in {"MatMul", "Gemm"}:
        return False, "unsupported_op"
    if len(node.input) < 2 or node.input[1] not in init_arrays:
        return False, "non_constant_weight"
    weight = init_arrays[node.input[1]]
    if weight.ndim != 2:
        return False, f"weight_rank_{weight.ndim}"
    if int(weight.shape[-1]) % BLOCK_SIZE != 0:
        return False, f"last_dim_{int(weight.shape[-1])}_not_divisible_by_{BLOCK_SIZE}"
    return True, None


def transform_to_mxfp8(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, Any]], list[dict[str, Any]]]:
    init_arrays = initializer_arrays(model)
    init_objects = initializer_objects(model)
    new_nodes: list[onnx.NodeProto] = []
    quantized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for original_node in model.graph.node:
        node = copy.deepcopy(original_node)
        if node.op_type == "LayerNormalization":
            base = node_base(node.name)
            cast_name = f"{base}/Cast"
            cast_out = f"{cast_name}_output_0"
            cast_node = helper.make_node(
                "Cast",
                [node.input[0]],
                [cast_out],
                name=cast_name,
                to=TensorProto.FLOAT,
            )
            node.input[0] = cast_out
            new_nodes.extend([cast_node, node])
            continue

        quantizable, reason = is_quantizable_linear(node, init_arrays)
        if not quantizable:
            if node.op_type in {"MatMul", "Gemm"}:
                skipped.append({"node": node.name, "reason": reason})
            new_nodes.append(node)
            continue

        weight_name = node.input[1]
        weight = init_arrays[weight_name]
        if ".weight" not in weight_name:
            new_weight_name = derive_quant_weight_name(node.name)
            init_tensor = init_objects[weight_name]
            init_tensor.name = new_weight_name
            del init_objects[weight_name]
            init_objects[new_weight_name] = init_tensor
            init_arrays[new_weight_name] = init_arrays.pop(weight_name)
            weight_name = new_weight_name
            node.input[1] = new_weight_name

        if node.op_type == "Gemm" and len(node.input) > 2 and node.input[2] in init_arrays:
            bias_name = node.input[2]
            bias_array = init_arrays[bias_name]
            if str(bias_array.dtype) != "float16":
                init_arrays[bias_name] = bias_array.astype("float16")
                init_objects[bias_name].CopyFrom(
                    numpy_helper.from_array(init_arrays[bias_name], name=bias_name)
                )

        base = node_base(node.name)

        act_dyn_q_name = f"{base}/input_quantizer/TRT_MXFP8DynamicQuantize"
        act_dyn_q_out = f"{act_dyn_q_name}_output_0"
        act_dyn_scale_out = f"{act_dyn_q_name}_output_1"
        act_dyn_q = helper.make_node(
            "TRT_MXFP8DynamicQuantize",
            [node.input[0]],
            [act_dyn_q_out, act_dyn_scale_out],
            name=act_dyn_q_name,
            domain=TRT_DOMAIN,
            axis=-1,
            block_size=BLOCK_SIZE,
            output_dtype=TensorProto.FLOAT8E4M3FN,
        )

        act_dq_name = f"{base}/input_quantizer/TRT_MXFP8DequantizeLinear"
        act_dq_out = f"{act_dq_name}_output_0"
        act_dq = helper.make_node(
            "TRT_MXFP8DequantizeLinear",
            [act_dyn_q_out, act_dyn_scale_out],
            [act_dq_out],
            name=act_dq_name,
            domain=TRT_DOMAIN,
            axis=-1,
            block_size=BLOCK_SIZE,
            output_dtype=TensorProto.FLOAT16,
        )

        weight_const_name = f"{base}/weight_quantizer/Constant"
        weight_const_out = f"{weight_const_name}_output_0"
        weight_const = make_constant_node(weight_const_name, weight_const_out, 1.0)

        weight_dq_name = f"{base}/weight_quantizer/TRT_MXFP8DequantizeLinear"
        weight_dq_out = f"{weight_dq_name}_output_0"
        weight_dq = helper.make_node(
            "TRT_MXFP8DequantizeLinear",
            [weight_name, weight_const_out],
            [weight_dq_out],
            name=weight_dq_name,
            domain=TRT_DOMAIN,
            axis=-1,
            block_size=BLOCK_SIZE,
            output_dtype=TensorProto.FLOAT16,
        )

        node.input[0] = act_dq_out
        node.input[1] = weight_dq_out

        new_nodes.extend([act_dyn_q, act_dq, weight_const, weight_dq, node])
        quantized.append(
            {
                "node": node.name,
                "op_type": node.op_type,
                "weight_name": weight_name,
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
            }
        )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    ensure_opsets(model)
    model = quantize_weights_to_mxfp8(model)
    return model, quantized, skipped


def save_model(model: onnx.ModelProto, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    data_path = output_dir / "onnx_model.data"
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        convert_attribute=True,
    )
    return onnx_path


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.input_onnx), load_external_data=True)
    model, quantized, skipped = transform_to_mxfp8(model)
    onnx_path = save_model(model, args.output_dir)
    summary = {
        "input_onnx": str(args.input_onnx),
        "output_onnx": str(onnx_path),
        "quantized_linear_nodes": len(quantized),
        "skipped_linear_nodes": len(skipped),
        "quantized": quantized,
        "skipped": skipped,
    }
    (args.output_dir / "quant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
