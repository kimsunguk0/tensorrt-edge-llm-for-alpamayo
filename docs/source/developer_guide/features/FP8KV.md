# FP8 KV Cache

## Overview

FP8 KV cache reduces memory usage by quantizing the key-value cache from FP16 to FP8 (NVIDIA FP8 E4M3 format), achieving approximately 50% memory reduction. This feature is particularly beneficial in memory-constrained scenarios.

**Key Points**:
- Reduces KV cache memory usage by ~50% (FP8 vs FP16)
- Uses NVIDIA FP8 E4M3 format with per-tensor quantization scales
- Can be used independently or combined with weight quantization
- Automatically detected during engine build from ONNX model configuration
- Requires calibration data during quantization step

---

## Workflow

### Vanilla Decode

#### Step 1: Quantize KV Cache

Quantize the KV cache using `tensorrt-edgellm-quantize-llm` with the `--kv_cache_quantization fp8` flag. This can be done independently or combined with weight quantization.

```bash
tensorrt-edgellm-quantize-llm \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --output_dir quantized/llama-3.1-8b-nvfp4-fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8
```

#### Step 2: Export to ONNX

Export the quantized model to ONNX format with the `--fp8_kv_cache` flag:

```bash
tensorrt-edgellm-export-llm \
  --model_dir quantized/llama-3.1-8b-nvfp4-fp8kv \
  --output_dir onnx_models/llama-3.1-8b-nvfp4-fp8kv \
  --fp8_kv_cache
```

**Note**: 
1. The `--fp8_kv_cache` flag is required during export to enable FP8 KV cache support in the ONNX model. This flag configures the attention plugin nodes to use FP8 KV cache format.

#### Step 3: Build Engine

Build the TensorRT engine as usual. The build process automatically detects FP8 KV cache configuration from the ONNX model:

```bash
./build/examples/llm/llm_build \
  --onnxDir onnx_models/llama-3.1-8b-nvfp4-fp8kv \
  --engineDir engines/llama-3.1-8b-nvfp4-fp8kv \
  --maxBatchSize=1 
```

No special build flags are required — FP8 KV cache is automatically enabled based on the ONNX model configuration.

#### Step 4: Run Inference

Run inference with the built engine. No special flags are needed:

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/llama-3.1-8b-nvfp4-fp8kv \
  --inputFile tests/test_cases/llm_basic.json \
  --outputFile output.json
```

---

### Spec Decode EAGLE

FP8 KV cache is fully supported with EAGLE speculative decoding. Both base and draft models can use FP8 KV cache to reduce memory usage.

#### Step 1: Quantize Base Model KV Cache

```bash
export MODEL_NAME=Llama-3.1-8B-Instruct

tensorrt-edgellm-quantize-llm \
    --model_dir meta-llama/${MODEL_NAME} \
    --output_dir ${MODEL_NAME}/quantized-base-nvfp4-fp8kv \
    --quantization nvfp4 \
    --kv_cache_quantization fp8
```

#### Step 2: Export Base Model

Export base model with FP8 KV cache and EAGLE base flag:

```bash
tensorrt-edgellm-export-llm \
    --model_dir ${MODEL_NAME}/quantized-base-nvfp4-fp8kv \
    --output_dir ${MODEL_NAME}/onnx/base-nvfp4-fp8kv \
    --is_eagle_base \
    --fp8_kv_cache
```

#### Step 3: Export Draft Model

Export draft model (references quantized base model):

```bash
tensorrt-edgellm-export-draft \
    --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
    --output_dir ${MODEL_NAME}/onnx/draft \
    --base_model_dir ${MODEL_NAME}/quantized-base-nvfp4-fp8kv
```

**Note**: The draft model does not use FP8 KV cache. Since the KV cache used for the draft model is small, FP8 quantization is not applied to avoid accuracy loss. Only the base model uses FP8 KV cache.

#### Step 4: Build Base Engine

```bash
./build/examples/llm/llm_build \
    --onnxDir=${MODEL_NAME}/onnx/base-nvfp4-fp8kv \
    --engineDir=${MODEL_NAME}/engines/eagle-nvfp4-fp8kv/ \
    --maxBatchSize=1 \
    --eagleBase
```

#### Step 5: Build Draft Engine

```bash
./build/examples/llm/llm_build \
    --onnxDir=${MODEL_NAME}/onnx/draft \
    --engineDir=${MODEL_NAME}/engines/eagle-nvfp4-fp8kv/ \
    --maxBatchSize=1 \
    --eagleDraft
```

#### Step 6: Run Inference with EAGLE

```bash
./build/examples/llm/llm_inference \
    --inputFile=tests/test_cases/llm_basic.json \
    --engineDir=${MODEL_NAME}/engines/eagle-nvfp4-fp8kv/ \
    --outputFile=output.json \
    --eagle
```

---

## Technical Details

### Quantization Format

- **Format**: NVIDIA FP8 E4M3 (4 exponent bits, 3 mantissa bits)
- **Scale**: Per-tensor quantization scales for K and V caches separately
- **Calibration**: Uses calibration dataset to determine optimal quantization scales
- **Memory Reduction**: ~50% reduction compared to FP16 (8 bits vs 16 bits per element)

### Quantization Process

During quantization:
1. Calibration data is passed through the model
2. Maximum absolute values (amax) are collected for K and V caches per layer
3. Quantization scales are computed: `scale = amax / FP8_E4M3_MAX` (where `FP8_E4M3_MAX = 448.0`)
4. Scales are stored in the model for use during inference

### Runtime Behavior

The main objective of FP8 KV cache is to optimize memory usage. During inference:
- KV cache is stored in FP8 format to reduce memory footprint
- Dequantization scales are provided to attention kernels
- KV cache values are dequantized to FP16 before computation
- Attention computation is performed in FP16 (transparent to the user)
- No accuracy degradation expected for most use cases

---

## Warning

**Model-Specific Accuracy Issues**: Not all models work well with FP8 KV cache. Some models may experience significant accuracy loss due to model-specific characteristics.

**Known Issues**:
- **Qwen2.5-7B-Instruct** and **Qwen2.5-VL-7B-Instruct** exhibit significant accuracy degradation with FP8 KV cache
- The root cause is that the KV cache of the last layer has very large values, resulting in substantial quantization loss
- This is a model-specific issue that also occurs in other frameworks, not a limitation of this implementation
- See [GitHub issue #4218](https://github.com/NVIDIA/TensorRT-LLM/issues/4218) for detailed discussion and examples of the accuracy issues with Qwen2.5 models

**Recommendation**: Test accuracy on your specific model before deploying FP8 KV cache in production. If significant accuracy loss is observed, consider using FP16 KV cache instead.

---

## Notes

- **Calibration Required**: FP8 KV cache quantization requires calibration data. Use `--dataset_dir` to specify a calibration dataset (default: `cnn_dailymail`).
- **Independent of Weight Quantization**: FP8 KV cache can be enabled independently of weight quantization. You can use NVFP4 weights (or FP8 weights) with FP8 KV cache.
- **Automatic Detection**: Engine build automatically detects FP8 KV cache from ONNX model configuration — no special build flags needed.
- **Compatibility**: Works with all supported models and features including LoRA, EAGLE speculative decoding, and tree attention.
- **Memory Benefits**: Most beneficial for long-context models and high batch sizes where KV cache memory dominates.
- **Platform Requirements**: Requires CUDA 11.8+ for FP8 support (`cuda_fp8.h`). And FP8 KV cache also works on GPUs with compute capability <SM89.

