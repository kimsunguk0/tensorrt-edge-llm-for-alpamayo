#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import onnx
from onnx import helper

DEFAULT_SOURCE = Path('/models/onnx/llm_kv')
DEFAULT_TARGET = Path('/root/TensorRT-Edge-LLM/output/onnx/llm_prefill_layer_debug')
DEFAULT_LAYERS = [0, 1, 20, 23, 33, 35]
DEFAULT_DETAIL_LAYERS = [0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a debug LLM ONNX directory that exposes selected layer outputs during prefill.'
    )
    parser.add_argument('--srcDir', default=str(DEFAULT_SOURCE))
    parser.add_argument('--dstDir', default=str(DEFAULT_TARGET))
    parser.add_argument('--layers', nargs='*', type=int, default=DEFAULT_LAYERS)
    parser.add_argument('--detail-layers', nargs='*', type=int, default=DEFAULT_DETAIL_LAYERS)
    parser.add_argument(
        '--include-final-hidden', action='store_true', help='Also expose final hidden_states output if not present.'
    )
    return parser.parse_args()


def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.name in {'onnx_model.data', 'embedding.safetensors'}:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def add_identity_output(model: onnx.ModelProto, src_name: str, out_name: str) -> None:
    existing_outputs = {out.name for out in model.graph.output}
    if out_name in existing_outputs:
        return
    value_info_lookup = {vi.name: vi for vi in model.graph.value_info}
    if src_name not in value_info_lookup:
        raise KeyError(f'Could not find value_info for {src_name}')
    src_vi = value_info_lookup[src_name]
    elem_type = src_vi.type.tensor_type.elem_type
    shape = []
    for dim in src_vi.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    model.graph.node.append(helper.make_node('Identity', inputs=[src_name], outputs=[out_name], name=f'Expose_{out_name}'))
    model.graph.output.append(helper.make_tensor_value_info(out_name, elem_type, shape))


def main() -> None:
    args = parse_args()
    src_dir = Path(args.srcDir)
    dst_dir = Path(args.dstDir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        'model.onnx',
        'onnx_model.data',
        'config.json',
        'embedding.safetensors',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
        'added_tokens.json',
        'special_tokens_map.json',
        'processed_chat_template.json',
        'chat_template.jinja',
    ):
        src = src_dir / name
        if src.exists():
            copy_or_symlink(src, dst_dir / name)

    model_path = src_dir / 'model.onnx'
    model = onnx.load(model_path, load_external_data=True)

    for layer in args.layers:
        add_identity_output(model, f'/model/layers.{layer}/Add_output_0', f'debug_layer_{layer:02d}_post_attn_hidden')
        add_identity_output(model, f'/model/layers.{layer}/Add_1_output_0', f'debug_layer_{layer:02d}_hidden')

    for layer in args.detail_layers:
        add_identity_output(
            model, f'/model/layers.{layer}/input_layernorm/Mul_1_output_0', f'debug_layer_{layer:02d}_input_ln'
        )
        add_identity_output(
            model, f'/model/layers.{layer}/self_attn/q_proj/MatMul_output_0', f'debug_layer_{layer:02d}_q_proj'
        )
        add_identity_output(
            model, f'/model/layers.{layer}/self_attn/k_proj/MatMul_output_0', f'debug_layer_{layer:02d}_k_proj'
        )
        add_identity_output(
            model, f'/model/layers.{layer}/self_attn/v_proj/MatMul_output_0', f'debug_layer_{layer:02d}_v_proj'
        )
        add_identity_output(model, f'/model/layers.{layer}/self_attn/q_norm/Mul_1_output_0', f'debug_layer_{layer:02d}_q_norm')
        add_identity_output(model, f'/model/layers.{layer}/self_attn/k_norm/Mul_1_output_0', f'debug_layer_{layer:02d}_k_norm')
        add_identity_output(
            model, f'/model/layers.{layer}/self_attn/Reshape_4_output_0', f'debug_layer_{layer:02d}_attn_reshape'
        )
        add_identity_output(
            model, f'/model/layers.{layer}/self_attn/o_proj/MatMul_output_0', f'debug_layer_{layer:02d}_o_proj'
        )
        add_identity_output(
            model,
            f'/model/layers.{layer}/post_attention_layernorm/Mul_1_output_0',
            f'debug_layer_{layer:02d}_post_attn_ln',
        )
        add_identity_output(model, f'/model/layers.{layer}/mlp/Mul_output_0', f'debug_layer_{layer:02d}_mlp_mul')
        add_identity_output(
            model, f'/model/layers.{layer}/mlp/down_proj/MatMul_output_0', f'debug_layer_{layer:02d}_mlp_down'
        )

    if args.include_final_hidden and 'hidden_states' not in {o.name for o in model.graph.output}:
        add_identity_output(model, '/model/norm/Mul_1_output_0', 'hidden_states')

    onnx.save(
        model,
        dst_dir / 'model.onnx',
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location='onnx_model.data',
    )
    print(f'Wrote debug ONNX dir to {dst_dir}')


if __name__ == '__main__':
    main()
