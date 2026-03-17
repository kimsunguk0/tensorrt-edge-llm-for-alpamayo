#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_raw(base: Path, rel: str, shape, dtype):
    return np.fromfile(base / rel, dtype=dtype).reshape(shape)


def stats(a: np.ndarray, b: np.ndarray):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf))
    cosine = float(np.dot(af, bf) / denom) if denom else 0.0
    return {
        'mean_abs_diff': float(diff.mean()),
        'max_abs_diff': float(diff.max()),
        'cosine': cosine,
    }


def align_tensors(pt: np.ndarray, trt: np.ndarray):
    if pt.shape == trt.shape:
        return pt, trt, 'exact'
    if (
        pt.ndim == trt.ndim
        and pt.ndim >= 3
        and pt.shape[0] == trt.shape[0]
        and pt.shape[1] == trt.shape[1] + 1
        and pt.shape[2:] == trt.shape[2:]
    ):
        return pt[:, :-1, ...], trt, 'drop_last_token_from_pytorch'
    raise ValueError(f'Unable to align PyTorch shape {pt.shape} with TRT shape {trt.shape}')


def parse_args():
    p = argparse.ArgumentParser(description='Compare PyTorch and TRT prefill layer outputs.')
    p.add_argument('--trtDir', required=True)
    p.add_argument('--pytorchDir', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    trt_dir = Path(args.trtDir)
    pt_dir = Path(args.pytorchDir)
    with open(trt_dir / 'prefill_layer_outputs_request_0.json') as f:
        trt_meta = json.load(f)
    with open(pt_dir / 'pytorch_prefill_layer_outputs_request_0.json') as f:
        pt_meta = json.load(f)

    pt_named_outputs = {}
    if 'named_outputs' in pt_meta:
        named_outputs = pt_meta['named_outputs']
        if isinstance(named_outputs, list):
            pt_named_outputs = {entry['name']: entry for entry in named_outputs}
        else:
            pt_named_outputs = named_outputs
    else:
        for entry in pt_meta['layers']:
            pt_named_outputs[entry['layer_input']['name']] = entry['layer_input']
            pt_named_outputs[entry['post_attn_hidden']['name']] = entry['post_attn_hidden']
            pt_named_outputs[entry['layer_hidden']['name']] = entry['layer_hidden']

    results = {'layers': []}

    for trt_entry in trt_meta['layer_outputs']:
        name = trt_entry['name']
        parts = name.split('_')
        layer = int(parts[2])
        if name not in pt_named_outputs:
            continue
        pt_entry = pt_named_outputs[name]
        trt = load_raw(trt_dir, trt_entry['file'], trt_entry['shape'], np.float16)
        pt = load_raw(pt_dir, pt_entry['file'], pt_entry['shape'], np.float16)
        pt_aligned, trt_aligned, alignment = align_tensors(pt, trt)
        results['layers'].append(
            {
                'name': name,
                'layer': layer,
                'kind': '_'.join(parts[3:]),
                'trt_shape': trt_entry['shape'],
                'pytorch_shape': pt_entry['shape'],
                'alignment': alignment,
                'stats': stats(pt_aligned, trt_aligned),
            }
        )

    layer_summary = {}
    for row in results['layers']:
        layer_summary.setdefault(row['layer'], {})[row['kind']] = row['stats']
    results['layer_summary'] = layer_summary

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote comparison to {args.output}')


if __name__ == '__main__':
    main()
