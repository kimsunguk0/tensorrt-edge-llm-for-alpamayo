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
 * Edge-LLM MoE Marlin W4A16 GEMM Kernel Interface
 *
 * This header provides the rt::Tensor based interface for MoE W4A16 GEMM
 * using Marlin optimized kernels with AWQ quantization format.
 *
 * AWQ (kU4) format:
 * - Uses unsigned 4-bit integers with zero point = 8 baked in
 * - Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 *
 * Note: Weight repacking (AWQ -> Marlin format) should be done in Python
 * using the awq_marlin_repack utility. This interface expects pre-swizzled
 * weights in Marlin format.
 */

#pragma once

#include "common/tensor.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief MoE W4A16 GEMM using Marlin kernel with AWQ format
 *
 * Performs MoE grouped GEMM with 4-bit AWQ quantized weights and 16-bit activations.
 * Uses kU4 format where zero point = 8 is baked into dequantization.
 *
 * Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 *
 * Note: Weights must be pre-swizzled into Marlin format using awq_marlin_repack.
 *
 * Internally uses FP32 reduction for numerical accuracy. The workspace buffer
 * must be sized using getMoeMarlinWorkspaceSize() which includes space for both
 * synchronization locks and the FP32 reduction buffer.
 *
 * @param input Input activations [numTokens, hiddenDim] (FP16)
 * @param output Output tensor [numTokens * topK, outDim] (FP16)
 * @param weights Marlin-repacked INT4 weights [numExperts, K/tile, N*tile/pack]
 * @param scales Per-group scales [numExperts, numGroups, outDim] (FP16)
 * @param sortedTokenIds Sorted token indices [numTokensPadded] (INT32)
 * @param expertIds Expert assignment per block [numBlocks] (INT32)
 * @param numTokensPostPadded Total padded token count [1] (INT32)
 * @param topkWeights Routing weights [numTokensPadded] (FP32)
 * @param workspace Workspace buffer sized by getMoeMarlinWorkspaceSize() (INT32)
 * @param moeBlockSize MoE processing block size (8, 16, 32, 48, or 64)
 * @param topK Number of experts per token
 * @param mulTopkWeights Whether to multiply output by topk weights
 * @param stream CUDA stream
 */
void moeAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& sortedTokenIds, rt::Tensor const& expertIds,
    rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights, rt::Tensor& workspace, int64_t moeBlockSize,
    int64_t topK, bool mulTopkWeights, cudaStream_t stream);

/*!
 * @brief Dense W4A16 GEMM using the Marlin AWQ kernel.
 *
 * This is a dense identity wrapper around the Marlin kernel. Decode can keep
 * using the existing GEMV path while prefill uses Marlin's prepacked GEMM
 * schedule without MoE token metadata.
 *
 * @param input Input activations [numTokens, hiddenDim] (FP16)
 * @param output Output tensor [numTokens, outDim] (FP16)
 * @param weights Marlin-repacked INT4 weights [1, K/16, 2*N] as INT32 view
 * @param scales Marlin-permuted scales [1, K/groupSize, N] (FP16)
 * @param workspace Workspace buffer sized by getMoeMarlinWorkspaceSize()
 * @param blockSize Dense processing block size, normally 32 for prefill
 * @param stream CUDA stream
 */
void denseAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor& workspace, int64_t blockSize, cudaStream_t stream);

/*!
 * @brief Fill identity metadata for dense Marlin GEMM.
 *
 * sortedTokenIds[i] = i for valid tokens and a sentinel >= M for padded slots;
 * expertIds are all zero; numTokensPostPadded stores the padded token count.
 */
void prepareDenseMarlinMetadata(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedTokens, int32_t blockSize, cudaStream_t stream);

/*!
 * @brief Get required workspace size for MoE Marlin GEMM
 *
 * Returns the total workspace size needed, which includes:
 * - Synchronization locks for thread blocks
 * - FP32 reduction buffer (c_tmp) for numerical accuracy
 *
 * @param numTokensPadded Maximum padded token count
 * @param outDim Output dimension N
 * @param moeBlockSize MoE block size
 * @param numSMs Number of SMs on the device
 * @return Required workspace size in number of int32_t elements
 */
int64_t getMoeMarlinWorkspaceSize(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize, int64_t numSMs);

} // namespace kernel
} // namespace trt_edgellm
