# Installation

> For the NVIDIA DRIVE platform, please refer to the documentation shipped with the DriveOS release

TensorRT Edge-LLM has two separate components that need to be installed on different systems:

1. **Python Export Pipeline** (runs on x86 host with GPU)
2. **C++ Runtime** (builds and runs on Edge devices)

---

## Part 1: Python Export Pipeline (x86 Host with GPU)

The Python export pipeline converts and quantizes models. This must run on an x86 Linux system with an NVIDIA GPU.

### System Requirements

- **Platform**: x86-64 Linux system
- **Recommended OS**: Ubuntu 22.04, 24.04
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- **CUDA**: 12.x or 13.x
- **Python**: 3.10+

#### Memory Requirements

**GPU Memory (VRAM):**
- General rule: ~2-3x model size for most operations, ~5-6x model size for FP8 ONNX export
- Small models (0.6B-3B): 8-16GB
- Large models (7B-8B): 20-48GB
- Very large models (13B+): 48GB+

**CPU Memory (RAM):**
- General rule: ~2-3x model size for most operations, **~18-20x** model size for FP8 ONNX export
- Small models (0.6B-3B): 8-16GB (48GB+ for FP8 ONNX export)
- Large models (7B-8B): 20-48GB (128GB+ for FP8 ONNX export)
- Very large models (13B+): 48GB+

> **Note:** FP8 ONNX export currently requires significantly higher CPU (up to 20x model size) and GPU (up to 6x model size) memory due to internal processing. This is a known issue and is being actively optimized.

**Verify Your Prerequisites:**

```bash
# Check CUDA installation
nvcc --version
# Should show CUDA 12.x or 13.x

# Check GPU and available memory
nvidia-smi
# Look for GPU memory (e.g., "24576MiB" for 24GB)

# Check Python version
python3 --version
# Should show Python 3.10 or higher
```

**If CUDA is not installed:**

Download and install CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). Choose version 12.x or 13.x for your system.

After installation, verify with `nvcc --version` and `nvidia-smi`.

### Installing

For a containerized environment for clean installation, it is recommended to use the NVIDIA PyTorch Docker image:

```bash
# Pull the recommended Docker image
docker pull nvcr.io/nvidia/pytorch:25.12-py3

# Run the container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash
```

**1. Clone Repository**

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**2. Install Python Package**

If you are not using container, it is recommended to use a virtual environment:
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

Then just install the software:

```bash
# Install package with all dependencies
pip3 install .
```

This installs all required Python dependencies including:
- PyTorch
- Transformers
- NVIDIA Model Optimizer
- ONNX
- And all other required dependencies

> **Note:** For specific version requirements, please refer to `requirements.txt` and `pyproject.toml` in the repository root.

**3. Verify Installation**

```bash
# Test export tools
tensorrt-edgellm-export-llm --help
tensorrt-edgellm-quantize-llm --help
```

**4. Configure HuggingFace Access (Optional)**

Some models on HuggingFace require you to accept terms before downloading. This is **not required** for the quick start example (Qwen3-0.6B).

**Models that require HuggingFace login:**
- Llama family (Llama 3.x)
- Phi-4 and Phi-4-Multimodal
- Other models marked as "gated" on HuggingFace

**To configure access:**

```bash
# Install HuggingFace CLI and login
huggingface-cli login
# Enter your HuggingFace access token when prompted
```

> **How to get a token:** Visit [HuggingFace Settings - Tokens](https://huggingface.co/settings/tokens), create a new token (read access is sufficient), and copy it.

> **For the quick start guide:** You can skip this step and proceed to verification.

**You're done with export pipeline setup!** You can now export and quantize models. The ONNX files will be transferred to the Edge device for runtime deployment.

---

## Part 2: C++ Runtime (Edge Device)

The C++ runtime builds and executes models on the target Edge device. This must be built on or for the target platform.

### System Requirements

**Target Platform:**
- NVIDIA Jetson Thor
- JetPack 7.1
- CUDA 13.x (included in JetPack)
- TensorRT 10.x+ (included in JetPack)
- Disk Space: ~20-50GB for ONNX files and TensorRT engines

### Build Instructions

**1. Install System Dependencies (on Edge device)**

```bash
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git
```

**2. Verify CUDA and TensorRT Installation**

After JetPack is installed, TensorRT should be installed in /usr

```bash
# Check CUDA version
nvcc --version  # Should show CUDA 13.x

# Check TensorRT version
dpkg -l | grep tensorrt  # Should show TensorRT 10.x+
```

**3. Clone Repository (on Edge device)**

```bash
# Clone to home directory (used in all examples)
cd ~
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**4. Configure Build**

On your Jetson Thor device, configure the build with the following command:

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
```

**Alternative: Building on x86 GPU Systems (Optional for Developers)**

If you want to build and test on an x86 workstation with NVIDIA GPU (for development purposes before deploying to Edge devices), you can use this configuration instead:

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_VERSION=<YOUR_CUDA_VERSION>
```

> **Note:** Replace `/usr/local/TensorRT-10.x.x` with your actual TensorRT installation path. Use `dpkg -l | grep tensorrt` to find it, or download from [NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt). Replace `<YOUR_CUDA_VERSION>` with your actual CUDA version (e.g., `13.0`). Use `nvcc --version` to check your CUDA version.

**CMake Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `TRT_PACKAGE_DIR` | Path to TensorRT installation | Required |
| `CMAKE_TOOLCHAIN_FILE` | **Required for Edge devices**: Use `cmake/aarch64_linux_toolchain.cmake` for Edge device builds. **Not needed for GPU builds** | N/A |
| `EMBEDDED_TARGET` | **Required for Edge devices**: Target platform (`jetson-thor`). **Not needed for GPU builds** | N/A |
| `CUDA_VERSION` | CUDA version (such as 13.0). Important for matching target platform. | 13.0 |
| `BUILD_UNIT_TESTS` | Build unit tests | OFF |

> **For supported GPU architectures and compute capabilities**, see [Supported Models - Platform Compatibility](supported-models.md#platform-compatibility)

**5. Build Project**

```bash
make -j$(nproc)
```

Build time: ~1-2 minutes depending on hardware.

**6. Verify Build**

```bash
# Test C++ examples
./examples/llm/llm_build --help
./examples/llm/llm_inference --help
```

**You're done with C++ runtime setup!** You can now build engines and run inference on the Edge device.

---

## Next Steps

After installation, proceed to the [Quick Start Guide](quick-start-guide.md) for a complete end-to-end workflow, or see the [Examples Guide](examples.md) for detailed pipeline stages and advanced use cases.

---

## Troubleshooting

### Common Installation Issues

**Issue: Python package import errors**

Solution: Ensure virtual environment is activated and package is installed:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install .
```

**Issue: `nvcc: command not found`**

Solution: Ensure JetPack 7.1 is properly installed with CUDA support:
```bash
# Verify CUDA installation
nvcc --version
# Should show CUDA 13.x
```

**Issue: `TensorRT not found` during CMake**

Solution: Specify TensorRT package directory. This directory should contain `lib` and `include` directories, and we are looking for the `nvinfer` library and header:
```bash
cmake .. \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
```

**Issue: Thread issue during C++ build**

Solution: Reduce parallel jobs or even use sequential build:
```bash
make -j  # Instead of make -j$(nproc)
```

### Getting Help

- **Documentation**: Check the `docs/source/developer_guide` directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Community**: Join the NVIDIA Developer Forums

## Uninstalling

**Python Export Pipeline (x86 Host):**
- Deactivate and remove virtual environment: `deactivate && rm -rf venv`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`

**C++ Runtime (Edge Device):**
- Remove build directory: `rm -rf build`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`
