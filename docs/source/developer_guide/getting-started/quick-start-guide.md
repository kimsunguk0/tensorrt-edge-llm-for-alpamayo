# Quick Start Guide

> **Repository:** [github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

> For the NVIDIA DRIVE platform, please refer to the documentation shipped with the DriveOS release

This quick start guide will get you up and running with TensorRT Edge-LLM in ~15 minutes.

**Prerequisites:** Complete the [Installation Guide](installation.md) for both x86 host and edge device before proceeding.

---

## Part 1: Export and Quantize on x86 Host

### Standard LLM Export

Let's use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as a lightweight example:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Step 1: Quantize to FP8 (downloads model automatically)
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir $MODEL_NAME/quantized \
    --quantization fp8

# Step 2: Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir $MODEL_NAME/quantized \
    --output_dir $MODEL_NAME/onnx
```

> **⚠️ Troubleshooting Export Issues:** If you encounter issues during quantization or export, see the [Python Export Pipeline - Common Issues and Solutions](../software-design/python-export-pipeline.md#common-issues-and-solutions).

### Transfer to Device

Transfer the ONNX folder to your Thor device:

```bash
# From x86 host - transfer to device
scp -r $MODEL_NAME/onnx <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

> **Note:** Replace `<device_user>` and `<device_ip>` with your actual device credentials (e.g., `nvidia@192.168.1.100`). If the directory doesn't exist on the device, create it first: `ssh <device_user>@<device_ip> "mkdir -p ~/tensorrt-edgellm-workspace/$MODEL_NAME"`

---

## Part 2: Build and Run on Edge Device

### Build TensorRT Engine

On your Thor device:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
cd ~/TensorRT-Edge-LLM

# Build engine
./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096
```

Build time: ~2-5 minutes

### Run Inference

Create an input file with a sample question:

```bash
cat > $WORKSPACE_DIR/input.json << 'EOF'
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of United States?"
                }
            ]
        }
    ]
}
EOF
```

> **Tip:** You can also use example input files from `~/TensorRT-Edge-LLM/tests/test_cases/` (e.g., `llm_basic.json`) instead of creating your own.

Run inference:

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json
```

Verify the output:

```bash
# View the model response
cat $WORKSPACE_DIR/output.json
```

You should see a JSON response with the model's answer, similar to:

```json
{
  "responses": [
    {
      "text": "The capital of the United States is Washington, D.C.",
      "finish_reason": "stop"
    }
  ]
}
```

**Success!** 🎉 You've successfully run LLM inference on your edge device!

---

## Next Steps

**For more advanced workflows, see [Examples](examples.md):**
- **VLM Inference** - Vision-language models with image understanding
- **EAGLE Speculative Decoding** - Accelerated generation for LLM and VLM
- **LoRA Support** - Dynamic adapter loading at runtime

**Input Format:** Our format matches closely with the OpenAI API format. See [Input Format Guide](input-format.md) for detailed specifications. Example input files are available in `tests/test_cases/` (e.g., `llm_basic.json`, `vlm_basic.json`).
