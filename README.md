# Alpamayo VLM on TensorRT-Edge-LLM v0.6.0

This fork layers an Alpamayo 1.5 post-VLM trajectory stack on top of NVIDIA TensorRT-Edge-LLM v0.6.0.

The main additions are:
- native C++ post-VLM runtime for Alpamayo FM/action decoding
- FM one-step ONNX export entrypoint inside the repo
- live sample consumer for Alpamayo 1.5
- persistent `llm_inference` mode for reusing loaded engines across live requests

The original NVIDIA top-level README is preserved as [README_nvidia.md](README_nvidia.md).

## What This Fork Adds

Key custom files:
- `cpp/runtime/alpamayoFmRuntime.*`
- `cpp/runtime/alpamayoPostVlmRuntime.*`
- `cpp/common/deltaTrajectoryTokenizer.*`
- `cpp/common/npyUtils.*`
- `tensorrt_edgellm/onnx_export/fm_export.py`
- `tensorrt_edgellm/scripts/export_fm.py`
- `jetson_live_infer_alpamayo15.py`

## Repository Layout

Important paths used in this fork:
- `examples/llm/llm_inference.cpp`: main inference binary with Alpamayo post-VLM hooks
- `jetson_live_infer_alpamayo15.py`: live HTTP sample consumer
- `input/requests/`: request JSON inputs
- `input/images/`, `input/ego/`: image and ego-history assets used by requests
- `output/`: runtime outputs, profiles, trajectories, dashboards

## Prerequisites

This fork assumes the standard TensorRT-Edge-LLM build prerequisites plus:
- TensorRT installation available through `TRT_PACKAGE_DIR`
- CUDA toolkit available through `CUDA_DIR` or `CUDA_CTK_VERSION`
- Alpamayo 1.5 Python source tree available separately for FM export
- prebuilt Alpamayo VLM TensorRT engines for the VLM stage

Typical engine layout used during development:
- LLM/VLM engine dir: `/alpamayo_vlm_engines/alpa1.5`
- FM engine plan: `/root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan`

## Build

Standard CMake configure:

```bash
cd /root/TensorRT-Edge-LLM-v060
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DCUDA_CTK_VERSION=12.8
```

Build the inference binary:

```bash
cmake --build build --target llm_inference -j$(nproc)
```

Notes:
- the live Python runner expects the plugin library at `build/libNvInfer_edgellm_plugin.so`
- if you run the binary manually from outside the repo root, set `EDGELLM_PLUGIN_PATH` explicitly

Example:

```bash
export EDGELLM_PLUGIN_PATH=/root/TensorRT-Edge-LLM-v060/build/libNvInfer_edgellm_plugin.so
```

## Export FM ONNX

This fork includes an in-repo FM export CLI:

```bash
PYTHONPATH=/root/TensorRT-Edge-LLM-v060 python -m tensorrt_edgellm.scripts.export_fm \
  --model_dir /path/to/alpamayo/model \
  --alpamayo_src_dir /path/to/alpamayo1.5/src \
  --packet /path/to/replay_packet.pt \
  --output_dir /path/to/export_dir \
  --max_seq_len 8192 \
  --dtype fp16
```

Useful options:
- `--dtype {bf16,fp16}`
- `--check_only` to validate wrapper wiring without exporting ONNX

## Non-Live Inference

The integrated Alpamayo path is driven through `llm_inference`.

Minimal example:

```bash
/root/TensorRT-Edge-LLM-v060/build/examples/llm/llm_inference \
  --engineDir /alpamayo_vlm_engines/alpa1.5 \
  --multimodalEngineDir /alpamayo_vlm_engines/alpa1.5 \
  --fmEngine /root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan \
  --alpamayoPostVlmRuntime \
  --inputFile /root/test/input/alpamayo15_b2_native_no_nav_request_with_fm.json \
  --outputFile /tmp/alpamayo_output.json \
  --warmup 0
```

Optional flags:
- `--alpamayoNavCfg` to enable guided/unguided nav CFG path
- `--dumpProfile` and `--profileOutputFile <path>` for single-shot timing dumps
- `--dumpKVCache` for KV cache export

## Live Inference

The live consumer polls an HTTP sample server that exposes Alpamayo-formatted NPZ payloads at `/latest`.

Basic run:

```bash
python /root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py \
  --server-url http://<sample-server>:8765 \
  --engine-dir /alpamayo_vlm_engines/alpa1.5 \
  --multimodal-engine-dir /alpamayo_vlm_engines/alpa1.5 \
  --fm-engine /root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan \
  --once
```

Recommended live mode:

```bash
python /root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py \
  --server-url http://<sample-server>:8765 \
  --engine-dir /alpamayo_vlm_engines/alpa1.5 \
  --multimodal-engine-dir /alpamayo_vlm_engines/alpa1.5 \
  --fm-engine /root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan \
  --persistent-llm-inference
```

Useful live flags:
- `--once`
- `--dump-profile`
- `--dump-kv-cache`
- `--nav-text "..."`
- `--dump-nav-dual-cache`
- `--persistent-llm-inference`

The live consumer writes:
- run outputs: `output/runs/live_runtime/`
- trajectories: `output/trajectories/live_runtime/`
- dashboards: `output/dashboards/live_runtime/`
- optional nav cache dumps: `output/nav_cache/live_runtime/`

## Persistent `llm_inference` Mode

This fork adds `--persistentServer` to `llm_inference`.

In this mode the binary:
- loads engines once
- keeps runtime state alive
- reads newline-delimited JSON commands from `stdin`

Request command:

```json
{"input_file":"/path/request.json","output_file":"/path/output.json"}
```

Shutdown command:

```json
{"command":"shutdown"}
```

Current limitation:
- persistent mode disables per-request profile export through `--dumpProfile` / `--profileOutputFile`

## Input Expectations

The live consumer expects an NPZ sample containing:
- `image_frames`
- `camera_indices`
- `ego_history_xyz`
- `ego_history_rot`
- `relative_timestamps`
- `absolute_timestamps`
- `t0_us`
- `fixed_delta_seconds`
- `clip_id`
- `camera_order`

Shapes used by the live path:
- `image_frames`: `[Cam, T, C, H, W]`
- `ego_history_xyz`: `[1, 1, 16, 3]`
- `ego_history_rot`: `[1, 1, 16, 3, 3]`

## Known Limitations

- persistent server mode currently does not emit per-request profiler JSON
- this repo does not include the external Alpamayo 1.5 source tree; FM export expects `--alpamayo_src_dir`
- live mode is designed around an external sample server and does not include the producer implementation in this repo

## Upstream Documentation

For the original TensorRT-Edge-LLM overview, installation guide, and supported-platform matrix, see [README_nvidia.md](README_nvidia.md) and the upstream NVIDIA documentation linked there.
