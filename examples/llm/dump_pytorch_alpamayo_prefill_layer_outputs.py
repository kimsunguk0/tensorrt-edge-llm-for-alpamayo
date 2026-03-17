#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/root/alpamayo/src')

from alpamayo_r1 import helper  # noqa: E402
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: E402

DEFAULT_LAYERS = [0, 1, 20, 23, 33, 35]
DEFAULT_DETAIL_LAYERS = [0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dump Alpamayo PyTorch prompt-only prefill layer outputs.')
    parser.add_argument('--imageDir', default='/root/TensorRT-Edge-LLM/input/images')
    parser.add_argument('--egoHistoryXYZNpy', default='/root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy')
    parser.add_argument('--egoHistoryRotNpy', default='/root/TensorRT-Edge-LLM/input/ego/ego_history_rot.npy')
    parser.add_argument('--hfModel', default='nvidia/Alpamayo-R1-10B')
    parser.add_argument('--outputDir', default='/root/TensorRT-Edge-LLM/output/prefill_layer_outputs/pytorch_alpamayo_prefill_layers')
    parser.add_argument('--layers', nargs='*', type=int, default=DEFAULT_LAYERS)
    parser.add_argument('--detailLayers', nargs='*', type=int, default=DEFAULT_DETAIL_LAYERS)
    return parser.parse_args()


def load_frames(image_dir: Path) -> torch.Tensor:
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    if len(paths) != 16:
        raise ValueError(f'Expected 16 PNG images in {image_dir}, found {len(paths)}')
    frames = []
    for path in paths:
        arr = np.array(Image.open(path).convert('RGB'), copy=True)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(frames, dim=0)


def save_raw(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(Path(args.imageDir))
    ego_history_xyz = torch.from_numpy(np.load(args.egoHistoryXYZNpy))
    ego_history_rot = torch.from_numpy(np.load(args.egoHistoryRotNpy))

    model = AlpamayoR1.from_pretrained(args.hfModel, dtype=torch.bfloat16).to('cuda')
    processor = helper.get_processor(model.tokenizer)

    messages = helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors='pt',
    )
    model_inputs = {'tokenized_data': inputs, 'ego_history_xyz': ego_history_xyz, 'ego_history_rot': ego_history_rot}
    model_inputs = helper.to_device(model_inputs, 'cuda')

    tokenized_data = model_inputs['tokenized_data']
    input_ids = model.fuse_traj_tokens(
        tokenized_data['input_ids'],
        {'ego_history_xyz': model_inputs['ego_history_xyz'], 'ego_history_rot': model_inputs['ego_history_rot']},
    )

    language_model = model.vlm.language_model
    layer_inputs = {}
    layer_outputs = {}
    post_attn_hidden = {}
    post_attn_ln = {}
    input_ln = {}
    q_proj = {}
    k_proj = {}
    v_proj = {}
    q_norm = {}
    k_norm = {}
    attn_reshape = {}
    o_proj = {}
    mlp_mul = {}
    mlp_down = {}
    hooks = []

    def make_layer_input_hook(layer_idx: int):
        def hook(_module, args_):
            layer_inputs[layer_idx] = args_[0].detach().clone()
        return hook

    def make_post_attn_pre_hook(layer_idx: int):
        def hook(_module, args_):
            post_attn_hidden[layer_idx] = args_[0].detach().clone()
        return hook

    def make_layer_output_hook(layer_idx: int):
        def hook(_module, _args, output):
            layer_outputs[layer_idx] = output.detach().clone()
        return hook

    def make_post_attn_ln_hook(layer_idx: int):
        def hook(_module, _args, output):
            post_attn_ln[layer_idx] = output.detach().clone()
        return hook

    def make_input_ln_hook(layer_idx: int):
        def hook(_module, _args, output):
            input_ln[layer_idx] = output.detach().clone()
        return hook

    def make_q_norm_hook(layer_idx: int):
        def hook(_module, _args, output):
            q_norm[layer_idx] = output.detach().clone()
        return hook

    def make_q_proj_hook(layer_idx: int):
        def hook(_module, _args, output):
            q_proj[layer_idx] = output.detach().clone()
        return hook

    def make_k_proj_hook(layer_idx: int):
        def hook(_module, _args, output):
            k_proj[layer_idx] = output.detach().clone()
        return hook

    def make_v_proj_hook(layer_idx: int):
        def hook(_module, _args, output):
            v_proj[layer_idx] = output.detach().clone()
        return hook

    def make_k_norm_hook(layer_idx: int):
        def hook(_module, _args, output):
            k_norm[layer_idx] = output.detach().clone()
        return hook

    def make_attn_reshape_pre_hook(layer_idx: int):
        def hook(_module, args_):
            attn_reshape[layer_idx] = args_[0].detach().clone()
        return hook

    def make_o_proj_hook(layer_idx: int):
        def hook(_module, _args, output):
            o_proj[layer_idx] = output.detach().clone()
        return hook

    def make_mlp_mul_pre_hook(layer_idx: int):
        def hook(_module, args_):
            mlp_mul[layer_idx] = args_[0].detach().clone()
        return hook

    def make_mlp_down_hook(layer_idx: int):
        def hook(_module, _args, output):
            mlp_down[layer_idx] = output.detach().clone()
        return hook

    for layer_idx in args.layers:
        hooks.append(language_model.layers[layer_idx].register_forward_pre_hook(make_layer_input_hook(layer_idx)))
        hooks.append(language_model.layers[layer_idx].register_forward_hook(make_layer_output_hook(layer_idx)))
        hooks.append(
            language_model.layers[layer_idx].post_attention_layernorm.register_forward_pre_hook(
                make_post_attn_pre_hook(layer_idx)
            )
        )
        if layer_idx in args.detailLayers:
            hooks.append(
                language_model.layers[layer_idx].input_layernorm.register_forward_hook(make_input_ln_hook(layer_idx))
            )
            hooks.append(language_model.layers[layer_idx].self_attn.q_proj.register_forward_hook(make_q_proj_hook(layer_idx)))
            hooks.append(language_model.layers[layer_idx].self_attn.k_proj.register_forward_hook(make_k_proj_hook(layer_idx)))
            hooks.append(language_model.layers[layer_idx].self_attn.v_proj.register_forward_hook(make_v_proj_hook(layer_idx)))
            hooks.append(language_model.layers[layer_idx].self_attn.q_norm.register_forward_hook(make_q_norm_hook(layer_idx)))
            hooks.append(language_model.layers[layer_idx].self_attn.k_norm.register_forward_hook(make_k_norm_hook(layer_idx)))
            hooks.append(
                language_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_attn_reshape_pre_hook(layer_idx)
                )
            )
            hooks.append(language_model.layers[layer_idx].self_attn.o_proj.register_forward_hook(make_o_proj_hook(layer_idx)))
            hooks.append(
                language_model.layers[layer_idx].post_attention_layernorm.register_forward_hook(
                    make_post_attn_ln_hook(layer_idx)
                )
            )
            hooks.append(language_model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_mlp_mul_pre_hook(layer_idx)))
            hooks.append(language_model.layers[layer_idx].mlp.down_proj.register_forward_hook(make_mlp_down_hook(layer_idx)))

    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model.vlm(
            input_ids=input_ids,
            attention_mask=tokenized_data.get('attention_mask'),
            pixel_values=tokenized_data.get('pixel_values'),
            image_grid_thw=tokenized_data.get('image_grid_thw'),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    for hook in hooks:
        hook.remove()

    metadata = {
        'request_idx': 0,
        'hf_model': args.hfModel,
        'layers': [],
        'input_shape': list(input_ids.shape),
        'named_outputs': [],
    }

    def append_named_output(name: str, tensor: np.ndarray) -> dict:
        file_name = f'{name}_request_0.bin'
        save_raw(output_dir / file_name, tensor)
        record = {
            'name': name,
            'file': file_name,
            'shape': list(tensor.shape),
            'dtype': 'FLOAT16',
            'num_bytes': int(tensor.nbytes),
        }
        metadata['named_outputs'].append(record)
        return record

    for layer_idx in args.layers:
        layer_hidden = layer_outputs[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
        post_attn = post_attn_hidden[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
        layer_in = layer_inputs[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()

        layer_hidden_rec = append_named_output(f'debug_layer_{layer_idx:02d}_hidden', layer_hidden)
        post_attn_rec = append_named_output(f'debug_layer_{layer_idx:02d}_post_attn_hidden', post_attn)
        layer_input_rec = append_named_output(f'debug_layer_{layer_idx:02d}_input', layer_in)

        detail_records = {}
        if layer_idx in args.detailLayers:
            input_ln_arr = input_ln[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            q_proj_arr = q_proj[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            k_proj_arr = k_proj[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            v_proj_arr = v_proj[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            q_norm_arr = q_norm[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            k_norm_arr = k_norm[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            attn_reshape_arr = attn_reshape[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            o_proj_arr = o_proj[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            post_attn_ln_arr = post_attn_ln[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            mlp_mul_arr = mlp_mul[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            mlp_down_arr = mlp_down[layer_idx].detach().to(dtype=torch.float16).cpu().numpy()
            detail_records['input_ln'] = append_named_output(f'debug_layer_{layer_idx:02d}_input_ln', input_ln_arr)
            detail_records['q_proj'] = append_named_output(f'debug_layer_{layer_idx:02d}_q_proj', q_proj_arr)
            detail_records['k_proj'] = append_named_output(f'debug_layer_{layer_idx:02d}_k_proj', k_proj_arr)
            detail_records['v_proj'] = append_named_output(f'debug_layer_{layer_idx:02d}_v_proj', v_proj_arr)
            detail_records['q_norm'] = append_named_output(f'debug_layer_{layer_idx:02d}_q_norm', q_norm_arr)
            detail_records['k_norm'] = append_named_output(f'debug_layer_{layer_idx:02d}_k_norm', k_norm_arr)
            detail_records['attn_reshape'] = append_named_output(
                f'debug_layer_{layer_idx:02d}_attn_reshape', attn_reshape_arr
            )
            detail_records['o_proj'] = append_named_output(f'debug_layer_{layer_idx:02d}_o_proj', o_proj_arr)
            detail_records['post_attn_ln'] = append_named_output(
                f'debug_layer_{layer_idx:02d}_post_attn_ln', post_attn_ln_arr
            )
            detail_records['mlp_mul'] = append_named_output(f'debug_layer_{layer_idx:02d}_mlp_mul', mlp_mul_arr)
            detail_records['mlp_down'] = append_named_output(f'debug_layer_{layer_idx:02d}_mlp_down', mlp_down_arr)

        metadata['layers'].append(
            {
                'layer': layer_idx,
                'layer_input': layer_input_rec,
                'post_attn_hidden': post_attn_rec,
                'layer_hidden': layer_hidden_rec,
                **detail_records,
            }
        )

    with open(output_dir / 'pytorch_prefill_layer_outputs_request_0.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Wrote PyTorch prefill layer outputs to {output_dir}')


if __name__ == '__main__':
    main()
