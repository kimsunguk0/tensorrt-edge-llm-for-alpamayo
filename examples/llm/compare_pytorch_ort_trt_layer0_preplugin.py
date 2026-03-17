#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compare PyTorch / ORT / TRT layer-0 pre-plugin tensors.')
    p.add_argument('--ortModel', default='/root/TensorRT-Edge-LLM/output/onnx/llm_raw_layer0_preplugin_ort/model.onnx')
    p.add_argument('--prefillInputDir', default='/root/TensorRT-Edge-LLM/output/prefill_inputs/llm_vlm_nq_fp16_prefill_debug/request_0')
    p.add_argument('--trtLayerDir', default='/root/TensorRT-Edge-LLM/output/prefill_layer_outputs/llm_vlm_nq_prefill_layer0_8_detail_debug/request_0')
    p.add_argument('--pytorchLayerDir', default='/root/TensorRT-Edge-LLM/output/prefill_layer_outputs/pytorch_alpamayo_prefill_layer0_8_detail')
    p.add_argument('--onnxConfig', default='/models/onnx/llm_raw/config.json')
    p.add_argument('--chunkSize', type=int, default=64)
    p.add_argument('--output', default='/root/TensorRT-Edge-LLM/output/prefill_layer_outputs/compare_pytorch_ort_trt_layer0_preplugin.json')
    return p.parse_args()


def load_raw(path: Path, shape, dtype):
    return np.fromfile(path, dtype=dtype).reshape(shape)


def tensor_stats(a: np.ndarray, b: np.ndarray) -> dict:
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    diff = np.abs(af - bf)
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    cosine = float(np.dot(af, bf) / denom) if denom else 0.0
    return {
        'mean_abs_diff': float(diff.mean()),
        'max_abs_diff': float(diff.max()),
        'cosine': cosine,
    }


def build_ort_outputs(model_path: Path, prefill_dir: Path) -> dict:
    meta = json.load(open(prefill_dir / 'prefill_inputs_request_0.json'))
    feeds = {
        'inputs_embeds': load_raw(prefill_dir / meta['inputs_embeds']['file'], meta['inputs_embeds']['shape'], np.float16),
    }
    for i, entry in enumerate(meta['deepstack_embeds']):
        feeds[f'deepstack_embeds_{i}'] = load_raw(prefill_dir / entry['file'], entry['shape'], np.float16)
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    out_names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(out_names, sess.run(None, feeds)))
    position_ids_full = load_raw(prefill_dir / meta['position_ids']['file'], meta['position_ids']['shape'], np.int64)
    return {
        'input_ln': outs['/model/layers.0/input_layernorm/Mul_1_output_0'],
        'q_proj': outs['/model/layers.0/self_attn/q_proj/MatMul_output_0'],
        'k_proj': outs['/model/layers.0/self_attn/k_proj/MatMul_output_0'],
        'v_proj': outs['/model/layers.0/self_attn/v_proj/MatMul_output_0'],
        'q_norm': outs['/model/layers.0/self_attn/q_norm/Mul_1_output_0'],
        'k_norm': outs['/model/layers.0/self_attn/k_norm/Mul_1_output_0'],
        'position_ids_full': position_ids_full,
    }


def load_pytorch_outputs(layer_dir: Path) -> dict:
    meta = json.load(open(layer_dir / 'pytorch_prefill_layer_outputs_request_0.json'))
    layer0 = next(x for x in meta['layers'] if x['layer'] == 0)
    out = {}
    for name in ['input_ln', 'q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'attn_reshape']:
        entry = layer0[name]
        out[name] = load_raw(layer_dir / entry['file'], entry['shape'], np.float16)
    return out


def load_trt_outputs(layer_dir: Path) -> dict:
    meta = json.load(open(layer_dir / 'prefill_layer_outputs_request_0.json'))
    entries = {e['name']: e for e in meta['layer_outputs']}
    out = {}
    for name in ['input_ln', 'q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'attn_reshape']:
        entry = entries[f'debug_layer_00_{name}']
        out[name] = load_raw(layer_dir / entry['file'], entry['shape'], np.float16)
    return out


def aligned_first_3005(arr: np.ndarray) -> np.ndarray:
    return arr[:, :3005, ...]


def build_rotary_config(config_path: Path) -> Qwen3VLTextConfig:
    cfg = json.load(open(config_path))
    return Qwen3VLTextConfig(
        vocab_size=cfg['vocab_size'],
        max_position_embeddings=cfg['max_position_embeddings'],
        hidden_size=cfg['hidden_size'],
        intermediate_size=cfg['intermediate_size'],
        num_hidden_layers=cfg['num_hidden_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        num_key_value_heads=cfg['num_key_value_heads'],
        rope_theta=cfg['rope_theta'],
        rope_scaling=cfg['rope_scaling'],
        head_dim=cfg['head_dim'],
    )


def to_rope_inputs(q_norm: np.ndarray, k_norm: np.ndarray, position_ids_b3s: np.ndarray, rotary: Qwen3VLTextRotaryEmbedding):
    q = torch.from_numpy(q_norm.astype(np.float32)).transpose(1, 2).contiguous()
    k = torch.from_numpy(k_norm.astype(np.float32)).transpose(1, 2).contiguous()
    pos = torch.from_numpy(position_ids_b3s[:, :, : q.shape[2]].transpose(1, 0, 2)).long()
    cos, sin = rotary(q, pos)
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    return q_rope.numpy(), k_rope.numpy()


def repeat_kv_np(hidden_states: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return hidden_states
    bs, n_kv_heads, slen, head_dim = hidden_states.shape
    expanded = np.repeat(hidden_states[:, :, None, :, :], n_rep, axis=2)
    return expanded.reshape(bs, n_kv_heads * n_rep, slen, head_dim)


def causal_attention_output(q: np.ndarray, k: np.ndarray, v: np.ndarray, num_heads: int, num_kv_heads: int, chunk_size: int) -> np.ndarray:
    q_t = torch.from_numpy(q.astype(np.float32))
    k_t = torch.from_numpy(k.astype(np.float32))
    v_t = torch.from_numpy(v.astype(np.float32))
    n_rep = num_heads // num_kv_heads
    if n_rep != 1:
        k_t = k_t[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(q_t.shape[0], num_heads, q_t.shape[2], q_t.shape[3])
        v_t = v_t[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(q_t.shape[0], num_heads, q_t.shape[2], q_t.shape[3])
    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0 / math.sqrt(q_t.shape[-1]))
    out = out.transpose(1, 2).contiguous().reshape(out.shape[0], out.shape[2], -1)
    return out.to(torch.float16).cpu().numpy()


def main() -> None:
    args = parse_args()
    ort_out = build_ort_outputs(Path(args.ortModel), Path(args.prefillInputDir))
    pt_out = load_pytorch_outputs(Path(args.pytorchLayerDir))
    trt_out = load_trt_outputs(Path(args.trtLayerDir))
    rotary_cfg = build_rotary_config(Path(args.onnxConfig))
    rotary = Qwen3VLTextRotaryEmbedding(rotary_cfg)

    report = {
        'summary': {},
        'pre_plugin': {},
        'post_rope': {},
        'derived_attention_output': {},
        'derived_vs_dumped_attention_output': {},
    }

    # Pre-plugin comparisons.
    for name in ['input_ln', 'q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']:
        report['pre_plugin'][name] = {
            'ort_vs_pytorch_exact': tensor_stats(ort_out[name], pt_out[name]),
            'ort_vs_trt_aligned_3005': tensor_stats(aligned_first_3005(ort_out[name]), trt_out[name]),
            'pytorch_vs_trt_aligned_3005': tensor_stats(aligned_first_3005(pt_out[name]), trt_out[name]),
        }

    # RoPE-applied q/k on aligned 3005 tokens.
    ort_q_rope, ort_k_rope = to_rope_inputs(aligned_first_3005(ort_out['q_norm']), aligned_first_3005(ort_out['k_norm']), ort_out['position_ids_full'], rotary)
    pt_q_rope, pt_k_rope = to_rope_inputs(aligned_first_3005(pt_out['q_norm']), aligned_first_3005(pt_out['k_norm']), ort_out['position_ids_full'], rotary)
    trt_q_rope, trt_k_rope = to_rope_inputs(trt_out['q_norm'], trt_out['k_norm'], ort_out['position_ids_full'], rotary)

    report['post_rope']['q'] = {
        'ort_vs_pytorch': tensor_stats(ort_q_rope, pt_q_rope),
        'ort_vs_trt': tensor_stats(ort_q_rope, trt_q_rope),
        'pytorch_vs_trt': tensor_stats(pt_q_rope, trt_q_rope),
    }
    report['post_rope']['k'] = {
        'ort_vs_pytorch': tensor_stats(ort_k_rope, pt_k_rope),
        'ort_vs_trt': tensor_stats(ort_k_rope, trt_k_rope),
        'pytorch_vs_trt': tensor_stats(pt_k_rope, trt_k_rope),
    }

    # Derived attention output on aligned 3005 tokens.
    ort_v = aligned_first_3005(ort_out['v_proj']).reshape(1, 3005, 8, 128).transpose(0, 2, 1, 3)
    pt_v = aligned_first_3005(pt_out['v_proj']).reshape(1, 3005, 8, 128).transpose(0, 2, 1, 3)
    trt_v = trt_out['v_proj'].reshape(1, 3005, 8, 128).transpose(0, 2, 1, 3)

    ort_attn = causal_attention_output(ort_q_rope, ort_k_rope, ort_v, rotary_cfg.num_attention_heads, rotary_cfg.num_key_value_heads, args.chunkSize)
    pt_attn = causal_attention_output(pt_q_rope, pt_k_rope, pt_v, rotary_cfg.num_attention_heads, rotary_cfg.num_key_value_heads, args.chunkSize)
    trt_attn = causal_attention_output(trt_q_rope, trt_k_rope, trt_v, rotary_cfg.num_attention_heads, rotary_cfg.num_key_value_heads, args.chunkSize)

    report['derived_attention_output'] = {
        'ort_vs_pytorch': tensor_stats(ort_attn, pt_attn),
        'ort_vs_trt': tensor_stats(ort_attn, trt_attn),
        'pytorch_vs_trt': tensor_stats(pt_attn, trt_attn),
    }

    report['derived_vs_dumped_attention_output'] = {
        'pytorch_derived_vs_dumped': tensor_stats(pt_attn, aligned_first_3005(pt_out['attn_reshape'])),
        'trt_derived_vs_dumped': tensor_stats(trt_attn, trt_out['attn_reshape']),
    }

    report['summary'] = {
        'interpretation': [
            'If ORT ~= PyTorch on q_proj/k_proj/q_norm and TRT differs, export graph before AttentionPlugin is likely fine and TRT runtime/path is introducing the drift.',
            'If derived TRT attention output ~= dumped TRT attn_reshape, the attention plugin is likely not the main source of divergence for layer 0.',
            'If ORT and PyTorch are already apart before attention, the mismatch originates before TRT runtime.'
        ]
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
