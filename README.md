# Alpamayo Runtime Guide

This document summarizes the local build and run flow for the Alpamayo VLM setup integrated into `llm_inference`.

## What This Setup Does

- Uses `llm_inference` as the single entry point
- Loads image inputs from JSON
- Loads ego history from `.npy`
- Injects trajectory tokens inside `llm_inference`
- Supports profile dump and KV-cache dump

## Runtime Layout

All runtime assets are organized inside this repository:

```text
/root/TensorRT-Edge-LLM/
  engines/
    llm_kv/
    visual/
  input/
    requests/
      input_fixed_template.json
      input_llm_inference_integrated.json
    ego/
      ego_history_xyz.npy
      ego_history_rot.npy
    images/
      cam0_f0.png
      ...
      cam3_f3.png
  output/
    runs/
    kv_cache/
```

## Prerequisites

- TensorRT installed and visible through `TRT_PACKAGE_DIR`
- CUDA 13.0
- A configured CMake build directory

The current local build cache uses:

```bash
TRT_PACKAGE_DIR=/usr
CUDA_VERSION=13.0
```

## Build

### Configure From Scratch

```bash
cd /root/TensorRT-Edge-LLM

cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_VERSION=13.0
```

### Build `llm_inference`

```bash
cd /root/TensorRT-Edge-LLM

cmake --build build --target llm_inference -j$(nproc)
```

## Input Format

The runtime now supports request-level ego history fields:

```json
{
  "requests": [
    {
      "ego_history_xyz_npy": "/root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy",
      "ego_history_rot_npy": "/root/TensorRT-Edge-LLM/input/ego/ego_history_rot.npy",
      "messages": [
        {
          "role": "system",
          "content": "You are a driving assistant that generates safe and accurate actions."
        },
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam0_f0.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam0_f1.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam0_f2.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam0_f3.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam1_f0.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam1_f1.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam1_f2.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam1_f3.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam2_f0.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam2_f1.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam2_f2.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam2_f3.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam3_f0.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam3_f1.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam3_f2.png"},
            {"type": "image", "image": "/root/TensorRT-Edge-LLM/input/images/cam3_f3.png"},
            {
              "type": "text",
              "text": "<|traj_history_start|><|traj_history_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory."
            }
          ]
        }
      ]
    }
  ]
}
```

Notes:

- Keep the trajectory placeholder:
  - `<|traj_history_start|><|traj_history_end|>`
- `llm_inference` will replace that region with generated `<i*>` tokens
- `predict_yaw` defaults to `false`

## Run

### Standard Run

```bash
cd /root/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir /root/TensorRT-Edge-LLM/engines/llm_kv \
  --multimodalEngineDir /root/TensorRT-Edge-LLM/engines/visual \
  --inputFile /root/TensorRT-Edge-LLM/input/requests/input_fixed_template.json \
  --outputFile /root/TensorRT-Edge-LLM/output/runs/output.json
```

## Live CARLA Pipeline

The repository also contains a live producer/consumer flow for CARLA:

- PC-side CARLA producer:
  - `/root/TensorRT-Edge-LLM/auto_pilot.py`
- Jetson-side live consumer:
  - `/root/TensorRT-Edge-LLM/jetson_live_infer.py`

### Live Flow Summary

The live flow is:

1. A 3090 PC runs CARLA and `auto_pilot.py`.
2. `auto_pilot.py` collects the latest 4-camera, 4-frame window and ego-history tensors.
3. The producer resizes images to `576x320` before transmission.
4. The producer publishes the latest sample through HTTP.
5. Jetson pulls the latest sample with `jetson_live_infer.py`.
6. Jetson writes:
   - images to `/root/TensorRT-Edge-LLM/input/images/live_runtime`
   - ego history to `/root/TensorRT-Edge-LLM/input/ego/live_runtime`
   - request JSON to `/root/TensorRT-Edge-LLM/input/requests/input_live_runtime.json`
7. Jetson runs `llm_inference`.

This flow is latest-only. It does not build a backlog queue of pending samples.

### PC-Side Producer

`auto_pilot.py` now supports a live HTTP server mode.

What it does in server mode:

- Runs CARLA in sync mode
- Collects the latest camera frames and ego history
- Builds online samples without waiting for future GT
- Resizes transmitted images to `576x320`
- Stores only the newest sample in memory
- Serves:
  - `GET /health`
  - `GET /latest`

Run on the PC with CARLA SIM:

```bash

python3.12 ./auto_pilot.py \
  --mode server \
  --host 127.0.0.1 \
  --port 2000 \
  --server_host 0.0.0.0 \
  --server_port 8765 \
  --max_frames -1 \
  --send_width 576 \
  --send_height 320
```

If ego-history values need to be checked before transmission:

```bash

python3.12 ./auto_pilot.py \
  --mode server \
  --host 127.0.0.1 \
  --port 2000 \
  --server_host 0.0.0.0 \
  --server_port 8765 \
  --max_frames -1 \
  --send_width 576 \
  --send_height 320 \
  --log_online_history
```

### Jetson-Side Consumer

`jetson_live_infer.py` pulls samples from the PC producer and runs inference.

What it does:

- Pulls `/latest` from the PC-side HTTP server
- Writes 16 images as PNG files
- Writes:
  - `ego_history_xyz.npy`
  - `ego_history_rot.npy`
- Rewrites the template request JSON with live file paths
- Calls `llm_inference`
- Optionally dumps profile and KV cache

Run on Jetson:

```bash
cd /root/TensorRT-Edge-LLM

python3.12 ./jetson_live_infer.py \
  --server-url http://192.168.10.183:8765 \
  --poll-interval 0.5 \
  --warmup 0
```

Run once and exit:

```bash
cd /root/TensorRT-Edge-LLM

python3.12 ./jetson_live_infer.py \
  --server-url http://192.168.10.183:8765 \
  --once \
  --max-runs 1 \
  --warmup 0
```

Run with profile and KV-cache dump:

```bash
cd /root/TensorRT-Edge-LLM

python3.12 ./jetson_live_infer.py \
  --server-url http://192.168.10.183:8765 \
  --poll-interval 0.5 \
  --warmup 0 \
  --dump-profile \
  --dump-kv-cache
```

### Live Runtime Files

Live input files written by the Jetson consumer:

- `/root/TensorRT-Edge-LLM/input/images/live_runtime/cam0_f0.png`
- `/root/TensorRT-Edge-LLM/input/images/live_runtime/cam3_f3.png`
- `/root/TensorRT-Edge-LLM/input/ego/live_runtime/ego_history_xyz.npy`
- `/root/TensorRT-Edge-LLM/input/ego/live_runtime/ego_history_rot.npy`
- `/root/TensorRT-Edge-LLM/input/requests/input_live_runtime.json`

Live output files:

- `/root/TensorRT-Edge-LLM/output/runs/live_runtime/output_seq<SEQ>_t0_<T0>.json`
- `/root/TensorRT-Edge-LLM/output/runs/live_runtime/profile_seq<SEQ>_t0_<T0>.json`
- `/root/TensorRT-Edge-LLM/output/kv_cache/live_runtime/seq<SEQ>_t0_<T0>/kv_cache_request_0.bin`

### Network Notes

Current network setup:

- PC: `192.168.10.183`
- Jetson: `192.168.10.108`
- Producer port: `8765`

Quick checks from Jetson:

```bash
curl http://192.168.10.183:8765/health
curl http://192.168.10.183:8765/latest -o /tmp/sample_latest.npz
```

### Run With Profile + KV-Cache Dump

```bash
cd /root/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir /root/TensorRT-Edge-LLM/engines/llm_kv \
  --multimodalEngineDir /root/TensorRT-Edge-LLM/engines/visual \
  --inputFile /root/TensorRT-Edge-LLM/input/requests/input_fixed_template.json \
  --outputFile /root/TensorRT-Edge-LLM/output/runs/output_internal_layout.json \
  --dumpProfile \
  --profileOutputFile /root/TensorRT-Edge-LLM/output/runs/profile_internal_layout.json \
  --dumpKVCache \
  --kvCacheOutputDir /root/TensorRT-Edge-LLM/output/kv_cache/internal_layout \
  --warmup 0
```

### CLI Override For Ego History

Use this if the JSON does not include `ego_history_xyz_npy` and `ego_history_rot_npy`:

```bash
cd /root/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir /root/TensorRT-Edge-LLM/engines/llm_kv \
  --multimodalEngineDir /root/TensorRT-Edge-LLM/engines/visual \
  --inputFile /root/TensorRT-Edge-LLM/input/requests/input_fixed_template.json \
  --outputFile /root/TensorRT-Edge-LLM/output/runs/output_cli_override.json \
  --egoHistoryXYZNpy /root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy
```

If yaw tokenization is needed:

```bash
cd /root/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir /root/TensorRT-Edge-LLM/engines/llm_kv \
  --multimodalEngineDir /root/TensorRT-Edge-LLM/engines/visual \
  --inputFile /root/TensorRT-Edge-LLM/input/requests/input_fixed_template.json \
  --outputFile /root/TensorRT-Edge-LLM/output/runs/output_cli_override_yaw.json \
  --egoHistoryXYZNpy /root/TensorRT-Edge-LLM/input/ego/ego_history_xyz.npy \
  --egoHistoryRotNpy /root/TensorRT-Edge-LLM/input/ego/ego_history_rot.npy \
  --predictYaw
```

## Outputs

Typical output files:

- `/root/TensorRT-Edge-LLM/output/runs/output.json`
- `/root/TensorRT-Edge-LLM/output/runs/profile.json`
- `/root/TensorRT-Edge-LLM/output/kv_cache/<run_name>/kv_cache_request_0.bin`
- `/root/TensorRT-Edge-LLM/output/kv_cache/<run_name>/kv_cache_request_0.json`

## Notes

- The current runtime uses copied engines inside this repo
- The `models/` directory is reserved for future use and is currently not required for inference
- The separate `traj_tokenize` tool is no longer required for the normal single-binary flow

