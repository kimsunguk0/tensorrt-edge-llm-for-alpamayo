/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "embeddingKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

// CUDA kernel for embedding lookup (FP16 only)
__global__ void embeddingLookupKernel(int32_t const* inputIds, half const* embeddingTable, half* output,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;

    // Get token ID and check bounds
    int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];
    bool const isValidToken = (tokenId >= 0 && tokenId < vocabSize);

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isValidToken)
        {
            // Load embedding data for valid token
            uint32_t const embeddingOffset = tokenId * hiddenSize + offset;
            embeddingVec.load(embeddingTable + embeddingOffset);
        }
        else
        {
            // Use zero embedding for out-of-bounds tokens
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// CUDA kernel for embedding lookup with image insertion (FP16 only)
__global__ void embeddingLookupWithImageInsertionKernel(int32_t const* inputIds, half const* embeddingTable,
    half const* imageEmbeds, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    int64_t imageTokenLen)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;

    // Get token ID
    int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];

    // Check if this is an image token (tokenId > vocabSize - 1)
    bool const isImageToken = tokenId > (vocabSize - 1);

    // Calculate base indices for this warp's work
    uint32_t baseEmbeddingOffset;
    half const* sourceTable;

    if (isImageToken)
    {
        // For image tokens, use imageEmbeds
        int32_t const visualTokenId = tokenId - vocabSize;

        // Validate that visualTokenId is within imageTokenLen
        if (visualTokenId >= 0 && visualTokenId < imageTokenLen)
        {
            baseEmbeddingOffset = visualTokenId * hiddenSize;
            sourceTable = imageEmbeds;
        }
        else
        {
            // Error case: visual token ID out of range, use zero embedding
            baseEmbeddingOffset = 0;
            sourceTable = nullptr;
        }
    }
    else
    {
        // For normal tokens, check bounds
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            baseEmbeddingOffset = tokenId * hiddenSize;
            sourceTable = embeddingTable;
        }
        else
        {
            // Out-of-bounds normal token, use zero embedding
            baseEmbeddingOffset = 0;
            sourceTable = nullptr;
        }
    }

    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (sourceTable != nullptr)
        {
            // Load embedding data from source table
            uint32_t const embeddingOffset = baseEmbeddingOffset + offset;
            embeddingVec.load(sourceTable + embeddingOffset);
        }
        else
        {
            // Use zero embedding for error cases
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// Helper function to launch vectorized embedding lookup kernel
void launchEmbeddingLookupKernel(int32_t const* inputIds, half const* embeddingTable, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, cudaStream_t stream)
{
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupKernel<<<gridSize, threadsPerBlock, 0, stream>>>(
        inputIds, embeddingTable, output, batchSize, seqLen, vocabSize, hiddenSize);
}

// Helper function to launch vectorized embedding lookup with image insertion kernel
void launchEmbeddingLookupWithImageInsertionKernel(int32_t const* inputIds, half const* embeddingTable,
    half const* imageEmbeds, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    int64_t imageTokenLen, cudaStream_t stream)
{
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupWithImageInsertionKernel<<<gridSize, threadsPerBlock, 0, stream>>>(
        inputIds, embeddingTable, imageEmbeds, output, batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen);
}

// CUDA kernel for assembling deepstack embeddings (FP16 only)
// Extracts image token embeddings from deepstack features based on token IDs
// Token IDs >= vocabSize are mapped to deepstack features, others get zero embeddings
__global__ void assembleDeepstackEmbeddingKernel(int32_t const* inputIds, half const* deepstackFeatures, half* output,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, int64_t numImageTokens)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;

    // Get token ID
    int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];

    // Determine if this is an image token (>= vocabSize)
    bool const isImageToken = tokenId >= vocabSize;

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isImageToken)
        {
            // Calculate the index into deepstackFeatures
            int32_t const deepstackIdx = tokenId - vocabSize;

            // Validate that deepstackIdx is within bounds
            if (deepstackIdx >= 0 && deepstackIdx < numImageTokens)
            {
                // Load embedding data from deepstack features
                uint32_t const embeddingOffset = deepstackIdx * hiddenSize + offset;
                embeddingVec.load(deepstackFeatures + embeddingOffset);
            }
            else
            {
                // Out-of-bounds image token, use zero embedding
#pragma unroll
                for (uint32_t i = 0; i < vecSize; ++i)
                {
                    embeddingVec[i] = __float2half(0.0f);
                }
            }
        }
        else
        {
            // Token ID < vocabSize, use zero embedding
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// CUDA kernel for Qwen3 multimodal embedding lookup (FP16 only)
// Uses pre-computed multimodalIndices to determine the index into audio/image embeds
// Handles special token IDs for audio and image modalities
__global__ void embeddingLookupMultimodalKernel(int32_t const* inputIds, half const* embeddingTable,
    int32_t const* multimodalIndices, int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen,
    int32_t audioTokenId, half const* audioEmbeds, int64_t audioTokenLen, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;
    uint32_t const linearIdx = batchIdx * seqLen + tokenIdx;

    // Get token ID
    int32_t const tokenId = inputIds[linearIdx];

    // Determine token type and source embedding table
    uint32_t baseEmbeddingOffset;
    half const* sourceTable;

    if (imageEmbeds != nullptr && tokenId == imageTokenId)
    {
        // Image token: use multimodalIndices to get the index into imageEmbeds
        int32_t const imageIdx = multimodalIndices[linearIdx];

        // Validate image index
        if (imageIdx >= 0 && imageIdx < imageTokenLen)
        {
            baseEmbeddingOffset = imageIdx * hiddenSize;
            sourceTable = imageEmbeds;
        }
        else
        {
            // Out-of-bounds image token, use zero embedding
            baseEmbeddingOffset = 0;
            sourceTable = nullptr;
        }
    }
    else if (audioEmbeds != nullptr && tokenId == audioTokenId)
    {
        // Audio token: use multimodalIndices to get the index into audioEmbeds
        int32_t const audioIdx = multimodalIndices[linearIdx];

        // Validate audio index
        if (audioIdx >= 0 && audioIdx < audioTokenLen)
        {
            baseEmbeddingOffset = audioIdx * hiddenSize;
            sourceTable = audioEmbeds;
        }
        else
        {
            // Out-of-bounds audio token, use zero embedding
            baseEmbeddingOffset = 0;
            sourceTable = nullptr;
        }
    }
    else
    {
        // Normal text token
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            baseEmbeddingOffset = tokenId * hiddenSize;
            sourceTable = embeddingTable;
        }
        else
        {
            // Out-of-bounds text token, use zero embedding
            baseEmbeddingOffset = 0;
            sourceTable = nullptr;
        }
    }

    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (sourceTable != nullptr)
        {
            // Load embedding data from source table
            uint32_t const embeddingOffset = baseEmbeddingOffset + offset;
            embeddingVec.load(sourceTable + embeddingOffset);
        }
        else
        {
            // Use zero embedding for error cases
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// Helper function to launch Qwen3-Omni multimodal embedding lookup kernel
void launchEmbeddingLookupMultimodalKernel(int32_t const* inputIds, half const* embeddingTable,
    int32_t const* multimodalIndices, int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen,
    int32_t audioTokenId, half const* audioEmbeds, int64_t audioTokenLen, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, cudaStream_t stream)
{
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupMultimodalKernel<<<gridSize, threadsPerBlock, 0, stream>>>(inputIds, embeddingTable,
        multimodalIndices, imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen, output,
        batchSize, seqLen, vocabSize, hiddenSize);
}

} // namespace

void embeddingLookup(
    rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int32_t const vocabSize = embeddingShape[0];
    int64_t const hiddenSize = embeddingShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* embeddingTablePtr = embeddingTable.dataPointer<half>();
    half* outputPtr = output.dataPointer<half>();

    // Launch optimized kernel with dynamic thread block sizing
    launchEmbeddingLookupKernel(
        inputIdsPtr, embeddingTablePtr, outputPtr, batchSize, seqLen, vocabSize, hiddenSize, stream);
}

void embeddingLookupWithImageInsertion(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::Tensor const& imageEmbeds, rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const imageShape = imageEmbeds.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const vocabSize = embeddingShape[0];
    int64_t const hiddenSize = embeddingShape[1];
    int64_t const imageTokenLen = imageShape[0];

    check::check(embeddingShape[1] == imageShape[1], "Hidden size mismatch between embeddingTable and imageEmbeds");
    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16");
    check::check(imageEmbeds.getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* embeddingTablePtr = embeddingTable.dataPointer<half>();
    half const* imageEmbedsPtr = imageEmbeds.dataPointer<half>();
    half* outputPtr = output.dataPointer<half>();

    // Launch optimized kernel with dynamic thread block sizing
    launchEmbeddingLookupWithImageInsertionKernel(inputIdsPtr, embeddingTablePtr, imageEmbedsPtr, outputPtr, batchSize,
        seqLen, vocabSize, hiddenSize, imageTokenLen, stream);
}

void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures, int32_t vocabSize,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const featuresShape = deepstackFeatures.getShape();
    auto const outputShape = deepstackEmbeds.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(featuresShape.getNumDims() == 2, "deepstackFeatures must be 2D tensor [numImageTokens, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "deepstackEmbeds must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const numImageTokens = featuresShape[0];
    int64_t const hiddenSize = featuresShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(deepstackFeatures.getDataType() == nvinfer1::DataType::kHALF, "deepstackFeatures must be FP16");
    check::check(deepstackEmbeds.getDataType() == nvinfer1::DataType::kHALF, "deepstackEmbeds must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* deepstackFeaturesPtr = deepstackFeatures.dataPointer<half>();
    half* outputPtr = deepstackEmbeds.dataPointer<half>();

    // Launch kernel
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    assembleDeepstackEmbeddingKernel<<<gridSize, threadsPerBlock, 0, stream>>>(
        inputIdsPtr, deepstackFeaturesPtr, outputPtr, batchSize, seqLen, vocabSize, hiddenSize, numImageTokens);
}

void embeddingLookupMultimodal(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor multimodalIndices, std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds,
    std::optional<int32_t> audioTokenId, rt::OptionalInputTensor audioEmbeds, rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const vocabSize = embeddingShape[0];
    int64_t const hiddenSize = embeddingShape[1];

    // Validate output shape
    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types for required inputs
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Handle optional image parameters
    bool const hasImage = imageTokenId.has_value() && imageEmbeds.has_value();
    if (hasImage)
    {
        auto const imageShape = imageEmbeds->get().getShape();
        check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
        check::check(imageShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and imageEmbeds");
        check::check(imageEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    }

    // Handle optional audio parameters
    bool const hasAudio = audioTokenId.has_value() && audioEmbeds.has_value();
    if (hasAudio)
    {
        auto const audioShape = audioEmbeds->get().getShape();
        check::check(audioShape.getNumDims() == 2, "audioEmbeds must be 2D tensor [audioTokenLen, hiddenSize]");
        check::check(audioShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and audioEmbeds");
        check::check(audioEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "audioEmbeds must be FP16");
    }

    // Validate that imageTokenId and audioTokenId are different when both are present
    if (hasImage && hasAudio)
    {
        check::check(*imageTokenId != *audioTokenId, "imageTokenId and audioTokenId must be different");
    }

    // Validate multimodalIndices if any multimodal input is present
    bool const hasMultimodal = hasImage || hasAudio;
    if (hasMultimodal)
    {
        check::check(
            multimodalIndices.has_value(), "multimodalIndices is required when image or audio inputs are provided");
        auto const multimodalIndicesShape = multimodalIndices->get().getShape();
        check::check(
            multimodalIndicesShape.getNumDims() == 2, "multimodalIndices must be 2D tensor [batchSize, seqLen]");
        check::check(multimodalIndicesShape[0] == batchSize, "multimodalIndices batch size mismatch");
        check::check(multimodalIndicesShape[1] == seqLen, "multimodalIndices sequence length mismatch");
        check::check(
            multimodalIndices->get().getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");
    }

    // Extract values for kernel - use safe defaults when modalities are absent
    int32_t const imageTokenIdValue = hasImage ? *imageTokenId : -1;
    half const* imageEmbedsPtr = hasImage ? imageEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const imageTokenLen = hasImage ? imageEmbeds->get().getShape()[0] : 0;

    int32_t const audioTokenIdValue = hasAudio ? *audioTokenId : -1;
    half const* audioEmbedsPtr = hasAudio ? audioEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const audioTokenLen = hasAudio ? audioEmbeds->get().getShape()[0] : 0;

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* embeddingTablePtr = embeddingTable.dataPointer<half>();
    int32_t const* multimodalIndicesPtr = hasMultimodal ? multimodalIndices->get().dataPointer<int32_t>() : nullptr;
    half* outputPtr = output.dataPointer<half>();

    // Launch kernel
    launchEmbeddingLookupMultimodalKernel(inputIdsPtr, embeddingTablePtr, multimodalIndicesPtr, imageTokenIdValue,
        imageEmbedsPtr, imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize, seqLen,
        vocabSize, hiddenSize, stream);
}

} // namespace kernel
} // namespace trt_edgellm
