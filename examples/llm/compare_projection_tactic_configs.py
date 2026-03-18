#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch


@dataclass
class ConfigSpec:
    name: str
    disable_tf32: bool = False
    disable_timing_cache: bool = False
    builder_optimization_level: int | None = None
    tactic_sources_mask: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compare projection GEMM drift across TRT builder/tactic configs.')
    p.add_argument('--onnxModel', default='/root/tmp_git_blockers/output/onnx/llm_raw_layer0_preplugin_ort/model.onnx')
    p.add_argument('--prefillInputDir', default='/root/tmp_git_blockers/output/prefill_inputs/llm_vlm_nq_fp16_prefill_debug/request_0')
    p.add_argument('--pytorchLayerDir', default='/root/tmp_git_blockers/output/prefill_layer_outputs/pytorch_alpamayo_prefill_layer0_8_detail')
    p.add_argument('--output', default='/root/TensorRT-Edge-LLM/output/projection_tactic_compare.json')
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


def tactic_mask(*sources: trt.TacticSource) -> int:
    mask = 0
    for source in sources:
        mask |= 1 << int(source)
    return mask


def load_prefill_feeds(prefill_dir: Path) -> dict[str, np.ndarray]:
    meta = json.load(open(prefill_dir / 'prefill_inputs_request_0.json'))
    feeds = {
        'inputs_embeds': load_raw(prefill_dir / meta['inputs_embeds']['file'], meta['inputs_embeds']['shape'], np.float16),
    }
    for i, entry in enumerate(meta['deepstack_embeds']):
        feeds[f'deepstack_embeds_{i}'] = load_raw(prefill_dir / entry['file'], entry['shape'], np.float16)
    return feeds


def load_pytorch_outputs(layer_dir: Path) -> dict[str, np.ndarray]:
    meta = json.load(open(layer_dir / 'pytorch_prefill_layer_outputs_request_0.json'))
    layer0 = next(x for x in meta['layers'] if x['layer'] == 0)
    outputs = {}
    for name in ['input_ln', 'q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']:
        entry = layer0[name]
        outputs[name] = load_raw(layer_dir / entry['file'], entry['shape'], np.float16)
    return outputs


def build_ort_outputs(model_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    return {
        'input_ln': outs['/model/layers.0/input_layernorm/Mul_1_output_0'],
        'q_proj': outs['/model/layers.0/self_attn/q_proj/MatMul_output_0'],
        'k_proj': outs['/model/layers.0/self_attn/k_proj/MatMul_output_0'],
        'v_proj': outs['/model/layers.0/self_attn/v_proj/MatMul_output_0'],
        'q_norm': outs['/model/layers.0/self_attn/q_norm/Mul_1_output_0'],
        'k_norm': outs['/model/layers.0/self_attn/k_norm/Mul_1_output_0'],
    }


def run_trt(onnx_path: Path, feeds: dict[str, np.ndarray], spec: ConfigSpec) -> tuple[dict[str, np.ndarray], dict]:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        ok = parser.parse(f.read())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f'Failed to parse ONNX for {spec.name}: {errors}')

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(feeds[tensor.name].shape)
        profile.set_shape(tensor.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if spec.disable_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if spec.disable_timing_cache:
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    if spec.builder_optimization_level is not None:
        config.builder_optimization_level = spec.builder_optimization_level
    if spec.tactic_sources_mask is not None:
        if not config.set_tactic_sources(spec.tactic_sources_mask):
            raise RuntimeError(f'Failed to set tactic sources for {spec.name}: mask={spec.tactic_sources_mask}')

    build_start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    if plan is None:
        raise RuntimeError(f'Failed to build TRT engine for {spec.name}')

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError(f'Failed to deserialize TRT engine for {spec.name}')
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f'Failed to create execution context for {spec.name}')

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, tuple(feeds[name].shape))

    stream = torch.cuda.Stream()
    buffers: dict[str, torch.Tensor] = {}
    infer_start = time.perf_counter()
    with torch.cuda.stream(stream):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = tuple(context.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                tensor = torch.from_numpy(feeds[name]).to('cuda')
            else:
                if np_dtype == np.float16:
                    torch_dtype = torch.float16
                elif np_dtype == np.float32:
                    torch_dtype = torch.float32
                else:
                    raise TypeError(f'Unsupported output dtype for {name}: {np_dtype}')
                tensor = torch.empty(shape, device='cuda', dtype=torch_dtype)
            buffers[name] = tensor
            context.set_tensor_address(name, int(tensor.data_ptr()))
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError(f'Failed to execute TRT engine for {spec.name}')
    stream.synchronize()
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    outputs = {
        'input_ln': buffers['/model/layers.0/input_layernorm/Mul_1_output_0'].cpu().numpy(),
        'q_proj': buffers['/model/layers.0/self_attn/q_proj/MatMul_output_0'].cpu().numpy(),
        'k_proj': buffers['/model/layers.0/self_attn/k_proj/MatMul_output_0'].cpu().numpy(),
        'v_proj': buffers['/model/layers.0/self_attn/v_proj/MatMul_output_0'].cpu().numpy(),
        'q_norm': buffers['/model/layers.0/self_attn/q_norm/Mul_1_output_0'].cpu().numpy(),
        'k_norm': buffers['/model/layers.0/self_attn/k_norm/Mul_1_output_0'].cpu().numpy(),
    }
    meta = {
        'build_ms': build_ms,
        'infer_ms': infer_ms,
        'disable_tf32': spec.disable_tf32,
        'disable_timing_cache': spec.disable_timing_cache,
        'builder_optimization_level': spec.builder_optimization_level,
        'tactic_sources_mask': spec.tactic_sources_mask,
    }
    return outputs, meta


def compare_all(trt_outputs: dict[str, np.ndarray], pt_outputs: dict[str, np.ndarray], ort_outputs: dict[str, np.ndarray]) -> dict:
    out = {}
    for name in ['input_ln', 'q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm']:
        out[name] = {
            'trt_vs_pytorch': tensor_stats(trt_outputs[name], pt_outputs[name]),
            'trt_vs_ort': tensor_stats(trt_outputs[name], ort_outputs[name]),
        }
    projection_mean = float(np.mean([
        out['q_proj']['trt_vs_pytorch']['mean_abs_diff'],
        out['k_proj']['trt_vs_pytorch']['mean_abs_diff'],
        out['v_proj']['trt_vs_pytorch']['mean_abs_diff'],
    ]))
    out['projection_mean_abs_diff_vs_pytorch'] = projection_mean
    return out


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnxModel)
    prefill_dir = Path(args.prefillInputDir)
    pytorch_dir = Path(args.pytorchLayerDir)

    feeds = load_prefill_feeds(prefill_dir)
    pt_outputs = load_pytorch_outputs(pytorch_dir)
    ort_outputs = build_ort_outputs(onnx_path, feeds)

    configs = [
        ConfigSpec(name='baseline_default'),
        ConfigSpec(name='disable_tf32', disable_tf32=True),
        ConfigSpec(name='disable_tf32_opt0', disable_tf32=True, builder_optimization_level=0),
        ConfigSpec(
            name='disable_tf32_opt0_cublas_lt',
            disable_tf32=True,
            builder_optimization_level=0,
            tactic_sources_mask=tactic_mask(trt.TacticSource.CUBLAS_LT),
        ),
        ConfigSpec(
            name='disable_tf32_opt0_cublas',
            disable_tf32=True,
            builder_optimization_level=0,
            tactic_sources_mask=tactic_mask(trt.TacticSource.CUBLAS),
        ),
        ConfigSpec(
            name='disable_tf32_opt0_no_timing_cache',
            disable_tf32=True,
            builder_optimization_level=0,
            disable_timing_cache=True,
        ),
    ]

    report = {'configs': {}, 'summary': {}}
    best_name = None
    best_projection = None

    for spec in configs:
        print(f'=== Running {spec.name} ===', flush=True)
        try:
            trt_outputs, meta = run_trt(onnx_path, feeds, spec)
            comparisons = compare_all(trt_outputs, pt_outputs, ort_outputs)
            report['configs'][spec.name] = {
                'build': meta,
                'comparison': comparisons,
            }
            projection = comparisons['projection_mean_abs_diff_vs_pytorch']
            if best_projection is None or projection < best_projection:
                best_projection = projection
                best_name = spec.name
            print(f"{spec.name}: projection_mean_abs_diff_vs_pytorch={projection:.9f}", flush=True)
        except Exception as exc:
            report['configs'][spec.name] = {'error': str(exc)}
            print(f'{spec.name}: ERROR {exc}', flush=True)

    baseline = report['configs'].get('baseline_default', {})
    baseline_projection = baseline.get('comparison', {}).get('projection_mean_abs_diff_vs_pytorch')
    report['summary'] = {
        'baseline_projection_mean_abs_diff_vs_pytorch': baseline_projection,
        'best_config': best_name,
        'best_projection_mean_abs_diff_vs_pytorch': best_projection,
        'reduced_vs_baseline': bool(
            baseline_projection is not None and best_projection is not None and best_projection < baseline_projection
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['summary'], indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
