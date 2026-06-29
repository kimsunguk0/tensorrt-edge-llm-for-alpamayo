/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Edge-LLM MoE Marlin W4A16 GEMM Kernel Wrapper
 *
 * This file provides the rt::Tensor based interface for MoE W4A16 GEMM
 * using Marlin optimized kernels with AWQ (kU4) quantization format.
 *
 * AWQ format (kU4):
 * - Uses unsigned 4-bit integers with zero point = 8 baked in
 * - Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 */

#include "moeMarlin.h"
#include <cuda_runtime.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/stringUtils.h"
#include "marlin/scalar_type.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>

#include "marlin_moe_wna16/ops.cu"
#include "marlin_moe_wna16/sm80_kernel_float16_u4_float16.cu"

namespace trt_edgellm
{
namespace kernel
{

using namespace trt_edgellm::format;

// Internal constants for FP32 reduction (always enabled for accuracy)
constexpr bool kUseFp32Reduce = true;
constexpr bool kUseAtomicAdd = false;

namespace
{

__global__ void prepareDenseMarlinMetadataKernel(int32_t* sortedTokenIds, int32_t* expertIds,
    int32_t* numTokensPostPadded, float* topkWeights, int32_t numTokens, int32_t paddedTokens, int32_t blockSize)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < paddedTokens)
    {
        sortedTokenIds[idx] = idx < numTokens ? idx : numTokens;
        topkWeights[idx] = 1.0f;
    }
    if (idx < (paddedTokens + blockSize - 1) / blockSize)
    {
        expertIds[idx] = 0;
    }
    if (idx == 0)
    {
        numTokensPostPadded[0] = paddedTokens;
    }
}

int getDenseMarlinConfigMode() noexcept
{
    static int const modeValue = []() {
        char const* mode = std::getenv("TRT_EDGELLM_DENSE_MARLIN_CONFIG");
        if (mode == nullptr || mode[0] == '\0')
        {
            return 0;
        }
        return std::atoi(mode);
    }();
    return modeValue;
}

} // namespace

void prepareDenseMarlinMetadata(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedTokens, int32_t blockSize, cudaStream_t stream)
{
    int32_t const blocks = static_cast<int32_t>(divUp(paddedTokens, 256));
    prepareDenseMarlinMetadataKernel<<<blocks, 256, 0, stream>>>(
        sortedTokenIds, expertIds, numTokensPostPadded, topkWeights, numTokens, paddedTokens, blockSize);
}

void moeAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& sortedTokenIds, rt::Tensor const& expertIds,
    rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights, rt::Tensor& workspace, int64_t moeBlockSize,
    int64_t topK, bool mulTopkWeights, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = scales.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D tensor [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D tensor [numTokens*topK, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D tensor [numExperts, K/tile, N*tile/pack]");
    check::check(scalesShape.getNumDims() == 3, "Scales must be 3D tensor [numExperts, numGroups, outDim]");

    int64_t numTokens = inputShape[0];
    int64_t hiddenDim = inputShape[1];
    int64_t numExperts = weightsShape[0];
    int64_t numGroups = scalesShape[1];
    int64_t outDim = scalesShape[2];

    // Validate output shape
    // Note: For MoE, the input can be either:
    // 1. Original tokens [numTokens, K] → output [numTokens*topK, N] (first GEMM)
    // 2. Slot activations [numTokens*topK, K] → output [numTokens*topK, N] (second GEMM)
    // We accept both cases by checking if output matches input*topK OR just input
    bool validOutputShape = (outputShape[0] == numTokens * topK) || (outputShape[0] == numTokens);
    check::check(validOutputShape,
        fmtstr("Output shape[0] %ld must be either numTokens*topK=%ld or numTokens=%ld", outputShape[0],
            numTokens * topK, numTokens));
    check::check(outputShape[1] == outDim, fmtstr("Output shape[1] %ld != outDim %ld", outputShape[1], outDim));

    // Validate data types
    check::check(input.getDataType() == nvinfer1::DataType::kHALF, "Input must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output must be FP16");
    check::check(weights.getDataType() == nvinfer1::DataType::kINT32, "Weights must be INT32 (Marlin packed)");
    check::check(scales.getDataType() == nvinfer1::DataType::kHALF, "Scales must be FP16");
    check::check(sortedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "sortedTokenIds must be INT32");
    check::check(expertIds.getDataType() == nvinfer1::DataType::kINT32, "expertIds must be INT32");
    check::check(numTokensPostPadded.getDataType() == nvinfer1::DataType::kINT32, "numTokensPostPadded must be INT32");
    check::check(topkWeights.getDataType() == nvinfer1::DataType::kFLOAT, "topkWeights must be FP32");
    check::check(workspace.getDataType() == nvinfer1::DataType::kINT32, "workspace must be INT32");

    // Validate moe_block_size
    if (moeBlockSize != 8)
    {
        check::check(moeBlockSize % 16 == 0, fmtstr("moeBlockSize %ld must be divisible by 16", moeBlockSize));
        check::check(
            moeBlockSize >= 16 && moeBlockSize <= 64, fmtstr("moeBlockSize %ld must be in [16, 64]", moeBlockSize));
    }

    // Calculate group size
    int groupSize = (numGroups > 1) ? static_cast<int>(hiddenDim / numGroups) : -1;

    // Get device info
    int dev;
    int sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    // AWQ W4A16 fixed scalar types
    trt_edgellm::marlin_dtypes::ScalarType aType = trt_edgellm::marlin_dtypes::kFloat16; // A16 activation
    trt_edgellm::marlin_dtypes::ScalarType bType
        = trt_edgellm::marlin_dtypes::kU4; // AWQ INT4 with zero point = 8 baked in
    trt_edgellm::marlin_dtypes::ScalarType cType = trt_edgellm::marlin_dtypes::kFloat16; // Output FP16
    trt_edgellm::marlin_dtypes::ScalarType sType = trt_edgellm::marlin_dtypes::kFloat16; // Scales FP16

    // Extract raw pointers
    void* inputPtr = const_cast<void*>(input.rawPointer());
    void* outputPtr = output.rawPointer();
    void* weightsPtr = const_cast<void*>(weights.rawPointer());
    void* scalesPtr = const_cast<void*>(scales.rawPointer());
    void* sortedTokenIdsPtr = const_cast<void*>(sortedTokenIds.rawPointer());
    void* expertIdsPtr = const_cast<void*>(expertIds.rawPointer());
    void* numTokensPostPaddedPtr = const_cast<void*>(numTokensPostPadded.rawPointer());
    void* topkWeightsPtr = const_cast<void*>(topkWeights.rawPointer());

    // Workspace layout: [locks (INT32)] [c_tmp (FP32)]
    int64_t numTokensPadded = sortedTokenIds.getShape()[0];
    int64_t locksSize = std::min(
        (outDim / marlin_moe_wna16::min_thread_n) * static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize)),
        static_cast<int64_t>(sms * 4));

    // Round locksSize up to multiple of 4 to ensure cTmpPtr is 16-byte aligned
    // (since locks are INT32 = 4 bytes, we need 4 INT32s = 16 bytes alignment)
    locksSize = static_cast<int64_t>(divUp(locksSize, static_cast<int64_t>(4))) * 4;

    int32_t* workspaceBasePtr = static_cast<int32_t*>(workspace.rawPointer());
    void* locksPtr = workspaceBasePtr;

    // c_tmp buffer starts after locks (FP32 reduction buffer)
    void* cTmpPtr = workspaceBasePtr + locksSize;

    // Marlin uses locks for reductions; these must be zeroed each call.
    CUDA_CHECK(cudaMemsetAsync(locksPtr, 0, locksSize * sizeof(int32_t), stream));

    // Call the raw marlin_mm function with AWQ-specific parameters
    marlin_moe_wna16::marlin_mm(inputPtr, // A
        weightsPtr,                       // B
        outputPtr,                        // C
        cTmpPtr,                          // C_tmp (FP32 reduction buffer)
        nullptr,                          // b_bias (not used)
        nullptr,                          // a_scales (not used for A16)
        scalesPtr,                        // b_scales
        nullptr,                          // global_scale (not used)
        nullptr,                          // zp (not used - zero point = 8 baked in)
        nullptr,                          // g_idx (not used - no act_order)
        nullptr,                          // perm (not used - no act_order)
        nullptr,                          // a_tmp (not used - no act_order)
        sortedTokenIdsPtr, expertIdsPtr, numTokensPostPaddedPtr, topkWeightsPtr, static_cast<int>(moeBlockSize),
        static_cast<int>(numExperts), static_cast<int>(topK), mulTopkWeights,
        static_cast<int>(numTokens), // prob_m
        static_cast<int>(outDim),    // prob_n
        static_cast<int>(hiddenDim), // prob_k
        locksPtr, aType, bType, cType, sType,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full (always true for standard AWQ)
        false, // has_zp (zero point = 8 baked into dequant)
        static_cast<int>(numGroups), groupSize, dev, stream,
        -1, // thread_k (auto)
        -1, // thread_n (auto)
        sms,
        -1, // blocks_per_sm (auto)
        kUseAtomicAdd, kUseFp32Reduce,
        false // is_zp_float
    );
}

void denseAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor& workspace, int64_t blockSize, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = scales.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D tensor [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D tensor [numTokens, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D tensor [1, K/16, 2*N]");
    check::check(scalesShape.getNumDims() == 3, "Scales must be 3D tensor [1, numGroups, outDim]");

    int64_t numTokens = inputShape[0];
    int64_t hiddenDim = inputShape[1];
    int64_t outDim = outputShape[1];
    int64_t numGroups = scalesShape[1];

    check::check(outputShape[0] == numTokens, "Output token dimension must match input");
    check::check(weightsShape[0] == 1, "Dense Marlin weights must use one synthetic expert");
    check::check(weightsShape[1] == hiddenDim / 16, "Dense Marlin weight K dimension mismatch");
    check::check(weightsShape[2] == 2 * outDim, "Dense Marlin weight N dimension mismatch");
    check::check(scalesShape[0] == 1, "Dense Marlin scales must use one synthetic expert");
    check::check(scalesShape[2] == outDim, "Dense Marlin scale N dimension mismatch");

    // Validate data types
    check::check(input.getDataType() == nvinfer1::DataType::kHALF, "Input must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output must be FP16");
    check::check(weights.getDataType() == nvinfer1::DataType::kINT32, "Weights must be INT32 (Marlin packed)");
    check::check(scales.getDataType() == nvinfer1::DataType::kHALF, "Scales must be FP16");
    check::check(workspace.getDataType() == nvinfer1::DataType::kINT32, "workspace must be INT32");

    if (blockSize != 8)
    {
        check::check(blockSize % 16 == 0, fmtstr("blockSize %ld must be divisible by 16", blockSize));
        check::check(blockSize >= 16 && blockSize <= 64, fmtstr("blockSize %ld must be in [16, 64]", blockSize));
    }

    int groupSize = (numGroups > 1) ? static_cast<int>(hiddenDim / numGroups) : -1;

    int dev;
    int sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    trt_edgellm::marlin_dtypes::ScalarType aType = trt_edgellm::marlin_dtypes::kFloat16;
    trt_edgellm::marlin_dtypes::ScalarType bType = trt_edgellm::marlin_dtypes::kU4;
    trt_edgellm::marlin_dtypes::ScalarType cType = trt_edgellm::marlin_dtypes::kFloat16;
    trt_edgellm::marlin_dtypes::ScalarType sType = trt_edgellm::marlin_dtypes::kFloat16;

    int threadK = -1;
    int threadN = -1;
    int blocksPerSm = -1;
    switch (getDenseMarlinConfigMode())
    {
    case 1:
        threadK = 64;
        threadN = 128;
        blocksPerSm = 1;
        break;
    case 2:
        threadK = 64;
        threadN = 128;
        blocksPerSm = 2;
        break;
    case 3:
        threadK = 64;
        threadN = 256;
        blocksPerSm = 1;
        break;
    case 4:
        threadK = 64;
        threadN = 256;
        blocksPerSm = 2;
        break;
    default: break;
    }

    int32_t* workspaceBasePtr = static_cast<int32_t*>(workspace.rawPointer());
    int64_t paddedTokens = static_cast<int64_t>(divUp(static_cast<int32_t>(numTokens), static_cast<int32_t>(blockSize)))
        * blockSize;
    int64_t locksSize = std::min((outDim / marlin_moe_wna16::min_thread_n)
            * static_cast<int64_t>(divUp(paddedTokens, blockSize)),
        static_cast<int64_t>(sms * 4));
    locksSize = static_cast<int64_t>(divUp(locksSize, static_cast<int64_t>(4))) * 4;

    void* locksPtr = workspaceBasePtr;
    void* cTmpPtr = workspaceBasePtr + locksSize;
    CUDA_CHECK(cudaMemsetAsync(locksPtr, 0, locksSize * sizeof(int32_t), stream));

    marlin_moe_wna16::marlin_mm(const_cast<void*>(input.rawPointer()), const_cast<void*>(weights.rawPointer()),
        output.rawPointer(), cTmpPtr, nullptr, nullptr, const_cast<void*>(scales.rawPointer()), nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr, // sortedTokenIds: nullptr selects dense identity row mapping
        nullptr, // expertIds
        nullptr, // numTokensPostPadded
        nullptr, // topkWeights
        static_cast<int>(blockSize),
        1,     // num_experts
        1,     // top_k
        false, // mul_topk_weights
        static_cast<int>(numTokens), static_cast<int>(outDim), static_cast<int>(hiddenDim), locksPtr, aType, bType,
        cType, sType,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full
        false, // has_zp
        static_cast<int>(numGroups), groupSize, dev, stream,
        threadK,
        threadN,
        sms,
        blocksPerSm,
        kUseAtomicAdd, kUseFp32Reduce,
        false // is_zp_float
    );
}

int64_t getMoeMarlinWorkspaceSize(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize, int64_t numSMs)
{
    // Locks size (INT32)
    int64_t maxNTiles = outDim / marlin_moe_wna16::min_thread_n;
    int64_t numBlocks = static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize));
    int64_t locksSize = std::min(maxNTiles * numBlocks, numSMs * 4);
    // Round locksSize up to multiple of 4 to ensure cTmpPtr is 16-byte aligned.
    locksSize = divUp(locksSize, 4) * 4;

    // C_tmp size for FP32 reduction (same element size as INT32)
    int64_t cTmpSize = std::min(
        outDim * numTokensPadded, numSMs * 4 * moeBlockSize * static_cast<int64_t>(marlin_moe_wna16::max_thread_n));
    if (moeBlockSize == 8)
    {
        cTmpSize *= 2;
    }

    // Total workspace = locks + c_tmp (both are 4 bytes per element)
    return locksSize + cTmpSize;
}

} // namespace kernel
} // namespace trt_edgellm

#else // CUDA_VERSION < 11080

// Stub implementations when CUDA < 11.8 (Marlin deps unavailable).
// Throws at runtime if INT4 MoE is used; allows plugin to link on older CUDA.
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

void moeAwqW4A16MarlinGemm(rt::Tensor const&, rt::Tensor&, rt::Tensor const&, rt::Tensor const&, rt::Tensor const&,
    rt::Tensor const&, rt::Tensor const&, rt::Tensor const&, rt::Tensor&, int64_t, int64_t, bool, cudaStream_t)
{
    throw std::runtime_error(
        "INT4 MoE Marlin GEMM requires CUDA 11.8 or later. Please use CUDA 11.8+ for INT4 MoE support.");
}

void denseAwqW4A16MarlinGemm(
    rt::Tensor const&, rt::Tensor&, rt::Tensor const&, rt::Tensor const&, rt::Tensor&, int64_t, cudaStream_t)
{
    throw std::runtime_error(
        "INT4 dense Marlin GEMM requires CUDA 11.8 or later. Please use CUDA 11.8+ for INT4 support.");
}

void prepareDenseMarlinMetadata(
    int32_t*, int32_t*, int32_t*, float*, int32_t, int32_t, int32_t, cudaStream_t)
{
    throw std::runtime_error(
        "INT4 dense Marlin GEMM requires CUDA 11.8 or later. Please use CUDA 11.8+ for INT4 support.");
}

int64_t getMoeMarlinWorkspaceSize(int64_t, int64_t, int64_t, int64_t)
{
    throw std::runtime_error(
        "INT4 MoE Marlin GEMM requires CUDA 11.8 or later. Please use CUDA 11.8+ for INT4 MoE support.");
}

} // namespace kernel
} // namespace trt_edgellm

#endif // CUDA_VERSION >= 11080
