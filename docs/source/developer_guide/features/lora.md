# LoRA (Low-Rank Adaptation)

## Overview

TensorRT Edge-LLM supports two approaches for using LoRA adapters:

| Approach | Scripts | Use Case |
|----------|---------|----------|
| **Static Merge** | `tensorrt-edgellm-merge-lora` | Permanently merge LoRA into base model before export |
| **Dynamic Runtime** | `tensorrt-edgellm-insert-lora` + `tensorrt-edgellm-process-lora` | Switch between multiple adapters at runtime |

---

## Approach 1: Static LoRA Merge

Permanently merges LoRA weights into the base model. Use this when:
- The LoRA is always required (e.g., Phi-4-multimodal vision-lora)
- You don't need runtime adapter switching
- You want simpler deployment with a single merged model

### Workflow

```bash
# Step 1: Merge LoRA into base model
tensorrt-edgellm-merge-lora \
  --model_dir Phi-4-multimodal-instruct \
  --lora_dir Phi-4-multimodal-instruct/vision-lora \
  --output_dir merged_model

# Step 2: Continue with standard export pipeline
tensorrt-edgellm-quantize-llm \
  --model_dir merged_model \
  --output_dir quantized \
  --quantization fp8

tensorrt-edgellm-export-llm \
  --model_dir quantized \
  --output_dir llm_onnx

# Step 3-4: Build and run as usual (no LoRA flags needed)
```

---

## Approach 2: Dynamic Runtime LoRA

Enables switching between multiple LoRA adapters at runtime without rebuilding engines. Use this when:
- You have multiple domain-specific adapters
- You need multi-tenant serving with different fine-tuned models
- You want A/B testing between adapter variants

### Workflow

```bash
# Step 1: Export base model
tensorrt-edgellm-export-llm \
  --model_dir Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir llm_onnx

# Step 2: Insert LoRA support into ONNX (creates lora_model.onnx)
tensorrt-edgellm-insert-lora \
  --onnx_dir llm_onnx

# Step 3: Process each LoRA adapter
tensorrt-edgellm-process-lora \
  --input_dir /path/to/adapter1 \
  --output_dir llm_onnx/lora_weights/adapter1

tensorrt-edgellm-process-lora \
  --input_dir /path/to/adapter2 \
  --output_dir llm_onnx/lora_weights/adapter2

# Step 4: Build engine with LoRA support
./build/examples/llm/llm_build \
  --onnxDir llm_onnx \
  --engineDir engines \
  --maxBatchSize 1 \
  --maxLoraRank 64

# Step 5: Run inference with adapter selection (see Input Format below)
./build/examples/llm/llm_inference \
  --engineDir engines \
  --inputFile input.json \
  --outputFile output.json
```

---

## Input Format for Runtime LoRA

Specify available adapters and select per-request:

```json
{
    "available_lora_weights": {
        "french": "/path/to/engines/lora_weights/french/processed_adapter_model.safetensors",
        "medical": "/path/to/engines/lora_weights/medical/processed_adapter_model.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Translate to French: Hello world"}
            ],
            "lora_name": "french"
        },
        {
            "messages": [
                {"role": "user", "content": "What is aspirin?"}
            ],
            "lora_name": "medical"
        }
    ]
}
```

**Note**: All requests in the same batch must use the same LoRA adapter. To disable LoRA, omit `lora_name` or set it to empty string.

---

## Script Reference

### `tensorrt-edgellm-merge-lora`

Permanently merges LoRA weights into a base HuggingFace model.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | Yes | - | Base model directory |
| `--lora_dir` | Yes | - | LoRA checkpoint directory |
| `--output_dir` | Yes | - | Output directory for merged model |
| `--device` | No | `cuda` | Device for loading model |

### `tensorrt-edgellm-insert-lora`

Inserts LoRA patterns into an exported ONNX model, creating `lora_model.onnx`.

| Argument | Required | Description |
|----------|----------|-------------|
| `--onnx_dir` | Yes | Directory containing `model.onnx` |

**Output**: Creates `lora_model.onnx` in the same directory.

### `tensorrt-edgellm-process-lora`

Processes HuggingFace LoRA adapter weights for runtime use.

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_dir` | Yes | Directory with `adapter_config.json` and `adapter_model.safetensors` |
| `--output_dir` | Yes | Output directory for processed weights |

**Output**: Creates `processed_adapter_model.safetensors` and `config.json`.

**Processing**:
- Converts bf16 → fp16
- Applies scaling: `lora_B *= lora_alpha / r`
- Transposes tensors to correct shapes
- Filters out norm and lm_head layers

---

## Build Parameters

When building with dynamic LoRA support:

| Parameter | Description |
|-----------|-------------|
| `--maxLoraRank` | Maximum LoRA rank to support (e.g., 64). Set to 0 to disable LoRA. |

---

## Notes

- Static merge is simpler but doesn't allow runtime switching
- Dynamic LoRA adds slight overhead but enables multi-adapter deployments
- All adapters must have rank ≤ `--maxLoraRank` specified at build time
- CUDA graphs are captured separately for each LoRA configuration
