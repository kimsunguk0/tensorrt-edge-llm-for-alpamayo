## Title

Planner live `/latest` samples appear to encode camera frames with a layout mismatch, causing corrupted staged PNGs, dashboard images, and likely corrupted VLM inputs

## Describe the bug

When the planner live service consumes real sensor-container `/latest` NPZ samples, the camera images rendered in the live dashboard are visibly corrupted with striping and color-channel mixing instead of showing a normal scene.

This is not a PNG encoding problem. The generated files are valid PNGs, but the underlying `image_frames` buffer appears to be interpreted with the wrong memory layout.

Impact:

- dashboard visualization is misleading
- staged camera PNGs are corrupted
- the same staged PNGs are used as multimodal runtime inputs, so live inference is likely running on corrupted images too

## Steps/Code to reproduce bug

**Build configuration:**

```bash
cmake -S /workspace/alpamayo_vlm \
  -B /workspace/alpamayo_vlm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=/workspace/alpamayo_vlm/cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor
cmake --build /workspace/alpamayo_vlm/build -j
```

**Runtime command used:**

```bash
cd /workspace/alpamayo_vlm
export PYTHONPATH=/workspace/alpamayo_vlm

python3 -m planner_live.planner_live_service \
  --server-url http://172.17.0.1:18080 \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/alpamayo15_fm_one_step_mxfp8_thor.plan \
  --output-root /workspace/alpamayo_vlm/output/planner_live \
  --enable-dashboard
```

**Observed output paths:**

- dashboard: `/workspace/alpamayo_vlm/output/planner_live/dashboards/latest_dashboard.png`
- staged image example: `/workspace/alpamayo_vlm/output/planner_live/staging/images/cam1_f0.png`
- latest parsed result: `/workspace/alpamayo_vlm/output/planner_live/results/latest_result.json`

**Current image staging code path:**

- frame staging assumes `image_frames[cam, t]` is true `C,H,W` and transposes it at [jetson_live_infer_alpamayo15.py:225](/workspace/alpamayo_vlm/jetson_live_infer_alpamayo15.py#L225)
- dashboard rendering makes the same assumption at [jetson_live_infer_alpamayo15.py:629](/workspace/alpamayo_vlm/jetson_live_infer_alpamayo15.py#L629)
- planner live uses those staged PNGs as the actual multimodal request inputs at [persistent_runtime.py:161](/workspace/alpamayo_vlm/planner_live/persistent_runtime.py#L161) and [jetson_live_infer_alpamayo15.py:259](/workspace/alpamayo_vlm/jetson_live_infer_alpamayo15.py#L259)

## Expected behavior

- staged camera PNGs should show normal camera scenes
- live dashboard should show the same normal images
- the runtime should receive correctly decoded camera frames

## Actual behavior

- staged PNGs are valid PNG files but visually corrupted
- dashboard panels show the same corruption
- reinterpreting the same raw bytes as `H,W,C` rather than transposing `C,H,W -> H,W,C` recovers plausible camera images

## Evidence

Comparison artifacts created during debugging:

- single-camera decode comparison: [debug_decode_candidates_cam1_f0.png](/workspace/alpamayo_vlm/output/planner_live/debug_decode_candidates_cam1_f0.png)
- all-camera decode comparison: [debug_decode_allcams.png](/workspace/alpamayo_vlm/output/planner_live/debug_decode_allcams.png)
- corrupted dashboard sample: [latest_dashboard.png](/workspace/alpamayo_vlm/output/planner_live/dashboards/latest_dashboard.png)

What the comparison shows:

- `current CHW->HWC` is corrupted
- `reinterpret raw as HWC` produces a plausible image for all tested cameras
- `reinterpret raw as HWC, BGR->RGB` only changes channel ordering slightly; the main issue is layout, not PNG encoding

## Suspected root cause

The producer and consumer disagree on the actual memory layout of `image_frames`.

The planner-side contract documents `image_frames` as `[Cam, T, C, H, W]`, but the real sensor `/latest` payload appears consistent with one of these cases:

- producer stores `H,W,C` image bytes and then reshapes or labels them as `C,H,W` without a real transpose
- producer emits a flattened byte layout that the consumer currently interprets as planar `C,H,W`, but it is actually interleaved `H,W,C`

Because the live consumer stages PNGs first and then references those PNGs in the request JSON, this issue affects both visualization and inference input quality.

## Proposed fix

Short term:

- add a planner-side compatibility decode path so image staging and dashboard rendering use the same corrected interpretation for this producer

Long term:

- fix the sensor container so `/latest` emits `image_frames` that truly match the documented `[Cam, T, C, H, W]` contract
- add an explicit layout field or a validation image sanity check in the producer/consumer contract to prevent silent regressions

## Notes

- This issue is independent of PNG metadata. The PNGs themselves are valid:
  - staged camera images are regular `RGB` PNG
  - dashboard output is regular `RGBA` PNG from Matplotlib
- Latest successful live result observed while reproducing:
  - sequence: `3424`
  - t0_us: `1775554293900000`
  - output text: `Yield due to a pedestrian blocking the lane ahead`

## System information (Edge Device)

- Platform: NVIDIA Jetson Thor
- Software release: Ubuntu 24.04.2 LTS container environment
- CPU architecture: `aarch64`
- GPU compute capability: not collected in this draft
- Total device memory: `122Gi`
- Build type: `Release`
- Library versions:
  - TensorRT Edge-LLM version or commit hash: `251cb0983c98b4443bdc8f9262eb5451e42f94a1`
  - CUDA: `13.0.48`
  - TensorRT: `10.13.2.6`
  - C++ compiler: `gcc 13.3.0`
- CMake options used:
  - `CMAKE_TOOLCHAIN_FILE=/workspace/alpamayo_vlm/cmake/aarch64_linux_toolchain.cmake`
  - `EMBEDDED_TARGET=jetson-thor`
  - `TRT_PACKAGE_DIR=/usr`
