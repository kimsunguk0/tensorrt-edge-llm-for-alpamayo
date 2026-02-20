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
#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace trt_edgellm;

// Debug flag for detailed error reporting
static constexpr bool DEBUG_MODE = false;

namespace
{

// Helper function to compare results using direct half comparison
bool compareResults(
    std::vector<half> const& ref, std::vector<half> const& test, std::string const& testName = "Embedding Lookup")
{
    if (ref.size() != test.size())
    {
        std::cout << testName << " validation failed: size mismatch (ref=" << ref.size() << ", test=" << test.size()
                  << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (!isclose(test[i], ref[i], 1e-2, 1e-2))
        {
            std::cout << testName << " validation failed at index " << i << ": expected=" << __half2float(ref[i])
                      << ", got=" << __half2float(test[i]) << std::endl;
            return false;
        }
    }

    return true;
}

} // namespace

class EmbeddingLookupTest : public ::testing::Test
{
protected:
    cudaStream_t stream;

    void SetUp() override
    {
        // Initialize CUDA device
        cudaSetDevice(0);

        // Create a non-default CUDA stream for testing
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        // Destroy the CUDA stream
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// Test standard embedding lookup accuracy
TEST_F(EmbeddingLookupTest, StandardEmbeddingLookupAccuracy)
{
    // Simple test cases for accuracy
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t>> testCases = {
        {1, 10, 10, 128},
        {2, 20, 50, 256},
        {4, 50, 100, 512},
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize));

        // Generate test data using testUtils
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult = embeddingLookupRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Standard Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize;
    }
}

// Test that kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, UnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 15; // Not a multiple of 8

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        { kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor, stream); }, std::runtime_error)
        << "Kernel should error out when hiddenSize is not a multiple of 8";
}

// Test that image insertion kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, UnevenHiddenSizeErrorWithImageInsertion)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const imageTokenLen = 8;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize + imageTokenLen - 1);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        {
            kernel::embeddingLookupWithImageInsertion(
                inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor, stream);
        },
        std::runtime_error)
        << "Image insertion kernel should error out when hiddenSize is not a multiple of 8";
}

// Test out-of-bounds token handling (should use zero embeddings)
TEST_F(EmbeddingLookupTest, OutOfBoundsTokenHandling)
{
    // Test case with out-of-bounds tokens
    int64_t const batchSize = 1;
    int64_t const seqLen = 4;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 16;

    // Generate test data with out-of-bounds tokens: [-1, 0, 9, 10]
    std::vector<int32_t> inputIds = {-1, 0, -1, 10}; // -1 and 10 are out-of-bounds

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::embeddingLookup(inputIdsTensor, embeddingTableTensor, outputTensor, stream);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Run CPU reference
    auto cpuResult = embeddingLookupRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Out-of-Bounds Token Handling Test"))
        << "GPU and CPU results don't match for out-of-bounds token handling";

    // Verify that out-of-bounds tokens produce zero embeddings
    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        int32_t const tokenId = inputIds[tokenIdx];
        bool const isOutOfBounds = (tokenId < 0 || tokenId >= vocabSize);

        if (isOutOfBounds)
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Out-of-bounds token " << tokenId << " should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test out-of-bounds token handling with image insertion
TEST_F(EmbeddingLookupTest, OutOfBoundsTokenHandlingWithImageInsertion)
{
    // Test case with out-of-bounds tokens and image tokens
    int64_t const batchSize = 1;
    int64_t const seqLen = 7;
    int32_t const vocabSize = 10;
    int64_t const hiddenSize = 16;
    int64_t const imageTokenLen = 8;

    // Generate test data with mixed tokens: [-1, 0, 9, 10, 15, 20]
    // -1: out-of-bounds normal token (should be zero)
    // 0, 9: valid normal tokens
    // 10: out-of-bounds normal token (should be zero)
    // 15: valid image token (10 + 5)
    // 20: out-of-bounds image token (10 + 10, but imageTokenLen = 8)
    std::vector<int32_t> inputIds = {0, 9, 10, -1, 15, 20, -1};

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::embeddingLookupWithImageInsertion(
        inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor, stream);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Run CPU reference
    auto cpuResult = embeddingLookupRef(
        inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize, imageEmbeds, imageTokenLen);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Out-of-Bounds Token Handling with Image Insertion Test"))
        << "GPU and CPU results don't match for out-of-bounds token handling with image insertion";

    // Verify specific token behaviors
    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        int32_t const tokenId = inputIds[tokenIdx];
        bool const isImageToken = tokenId > (vocabSize - 1);
        bool isOutOfBounds = false; // Will be determined per token type

        if (isImageToken)
        {
            int32_t const visualTokenId = tokenId - vocabSize;
            isOutOfBounds = (visualTokenId < 0 || visualTokenId >= imageTokenLen);
        }
        else
        {
            isOutOfBounds = (tokenId < 0 || tokenId >= vocabSize);
        }

        if (isOutOfBounds)
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Out-of-bounds token " << tokenId << " should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test embedding lookup with image insertion accuracy
TEST_F(EmbeddingLookupTest, EmbeddingLookupWithImageInsertionAccuracy)
{
    // Simple test cases for accuracy
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t>> testCases = {
        {1, 10, 10, 128, 64},  // Small test
        {2, 20, 50, 256, 128}, // Medium test
        {4, 50, 100, 128, 64}, // Large test
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", imageTokenLen=" + std::to_string(imageTokenLen));

        // Generate test data using testUtils
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, 0, vocabSize + imageTokenLen - 1);

        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half),
            cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::embeddingLookupWithImageInsertion(
            inputIdsTensor, embeddingTableTensor, imageEmbedsTensor, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult = embeddingLookupRef(
            inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize, imageEmbeds, imageTokenLen);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Embedding Lookup with Image Insertion Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize << ", imageTokenLen=" << imageTokenLen;
    }
}

// Test deepstack embedding lookup accuracy
TEST_F(EmbeddingLookupTest, DeepstackEmbeddingLookupAccuracy)
{
    // Simple test cases for accuracy
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t>> testCases = {
        {1, 10, 100, 128, 64},  // Small test
        {2, 20, 200, 256, 128}, // Medium test
        {4, 50, 500, 128, 256}, // Large test
    };

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, numImageTokens] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", numImageTokens=" + std::to_string(numImageTokens));

        // Generate test data - mix of tokens < vocabSize and >= vocabSize
        std::vector<int32_t> inputIds(batchSize * seqLen);
        uniformIntInitialization<int32_t>(inputIds, vocabSize, vocabSize + numImageTokens - 1);

        std::vector<half> deepstackFeatures(numImageTokens * hiddenSize);
        uniformFloatInitialization<half>(deepstackFeatures, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords featuresShape{numImageTokens, hiddenSize};
        rt::Tensor deepstackFeaturesTensor(featuresShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deepstackFeaturesTensor.rawPointer(), deepstackFeatures.data(),
            deepstackFeatures.size() * sizeof(half), cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::assembleDeepstackEmbedding(inputIdsTensor, deepstackFeaturesTensor, vocabSize, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult = assembleDeepstackEmbeddingRef(
            inputIds, deepstackFeatures, batchSize, seqLen, vocabSize, hiddenSize, numImageTokens);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Deepstack Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize << ", numImageTokens=" << numImageTokens;
    }
}

// Test deepstack embedding lookup with out-of-bounds handling
TEST_F(EmbeddingLookupTest, DeepstackEmbeddingLookupOutOfBounds)
{
    // Test case with mixed tokens
    int64_t const batchSize = 1;
    int64_t const seqLen = 6;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 128;
    int64_t const numImageTokens = 10;

    // Generate test data with specific tokens:
    // - Tokens < vocabSize (should be zero)
    // - Tokens >= vocabSize and < vocabSize + numImageTokens (should use deepstack features)
    // - Tokens >= vocabSize + numImageTokens (should be zero - out of bounds)
    std::vector<int32_t> inputIds = {50, 100, 105, 110, 115, 200};
    // 50: < vocabSize -> zero
    // 100: = vocabSize -> deepstack[0]
    // 105: = vocabSize + 5 -> deepstack[5]
    // 110: = vocabSize + 10 -> out of bounds -> zero
    // 115: = vocabSize + 15 -> out of bounds -> zero
    // 200: >> vocabSize + numImageTokens -> out of bounds -> zero

    std::vector<half> deepstackFeatures(numImageTokens * hiddenSize);
    uniformFloatInitialization<half>(deepstackFeatures, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords featuresShape{numImageTokens, hiddenSize};
    rt::Tensor deepstackFeaturesTensor(featuresShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deepstackFeaturesTensor.rawPointer(), deepstackFeatures.data(),
        deepstackFeatures.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::assembleDeepstackEmbedding(inputIdsTensor, deepstackFeaturesTensor, vocabSize, outputTensor, stream);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Run CPU reference
    auto cpuResult = assembleDeepstackEmbeddingRef(
        inputIds, deepstackFeatures, batchSize, seqLen, vocabSize, hiddenSize, numImageTokens);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Deepstack Embedding Lookup Out-of-Bounds Test"))
        << "GPU and CPU results don't match for deepstack out-of-bounds handling";

    // Verify specific token behaviors
    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        int32_t const tokenId = inputIds[tokenIdx];
        bool shouldBeZero = false;

        if (tokenId < vocabSize)
        {
            // Tokens below vocabSize should be zero
            shouldBeZero = true;
        }
        else
        {
            int32_t const deepstackIdx = tokenId - vocabSize;
            if (deepstackIdx < 0 || deepstackIdx >= numImageTokens)
            {
                // Out-of-bounds image tokens should be zero
                shouldBeZero = true;
            }
        }

        if (shouldBeZero)
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Token " << tokenId << " at position " << tokenIdx
                    << " should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test that deepstack kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, DeepstackUnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 5;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const numImageTokens = 10;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, vocabSize, vocabSize + numImageTokens - 1);

    std::vector<half> deepstackFeatures(numImageTokens * hiddenSize);
    uniformFloatInitialization<half>(deepstackFeatures, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords featuresShape{numImageTokens, hiddenSize};
    rt::Tensor deepstackFeaturesTensor(featuresShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deepstackFeaturesTensor.rawPointer(), deepstackFeatures.data(),
        deepstackFeatures.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        {
            kernel::assembleDeepstackEmbedding(
                inputIdsTensor, deepstackFeaturesTensor, vocabSize, outputTensor, stream);
        },
        std::runtime_error)
        << "Deepstack kernel should error out when hiddenSize is not a multiple of 8";
}

// Test Qwen3-Omni multimodal embedding lookup accuracy
TEST_F(EmbeddingLookupTest, MultimodalAccuracy)
{
    // Test cases with varying sizes
    std::vector<std::tuple<int64_t, int64_t, int32_t, int64_t, int64_t, int64_t>> testCases = {
        // {batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen, audioTokenLen}
        {1, 16, 100, 128, 8, 8},   // Small test
        {2, 32, 200, 128, 16, 9},  // Medium test
        {4, 64, 250, 128, 32, 10}, // Large test
    };

    // Special token IDs (similar to Qwen3-Omni)
    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    for (auto const& [batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen, audioTokenLen] : testCases)
    {
        SCOPED_TRACE("Testing: batchSize=" + std::to_string(batchSize) + ", seqLen=" + std::to_string(seqLen)
            + ", vocabSize=" + std::to_string(vocabSize) + ", hiddenSize=" + std::to_string(hiddenSize)
            + ", imageTokenLen=" + std::to_string(imageTokenLen) + ", audioTokenLen=" + std::to_string(audioTokenLen));

        // Generate test data
        // Create inputIds with a mix of text tokens, image tokens, and audio tokens
        std::vector<int32_t> inputIds(batchSize * seqLen);
        std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

        int32_t imageCounter = 0;
        int32_t audioCounter = 0;

        for (int64_t i = 0; i < batchSize * seqLen; ++i)
        {
            int32_t choice = i % 5; // Distribute tokens across types
            if (choice == 0 && imageCounter < imageTokenLen)
            {
                // Image token
                inputIds[i] = imageTokenId;
                multimodalIndices[i] = imageCounter++;
            }
            else if (choice == 1 && audioCounter < audioTokenLen)
            {
                // Audio token
                inputIds[i] = audioTokenId;
                multimodalIndices[i] = audioCounter++;
            }
            else
            {
                // Text token (random valid token ID)
                inputIds[i] = i % vocabSize;
            }
        }

        // Generate embedding tables
        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

        std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
        uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords audioShape{audioTokenLen, hiddenSize};
        rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(multimodalIndicesTensor.rawPointer(), multimodalIndices.data(),
            multimodalIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(audioEmbedsTensor.rawPointer(), audioEmbeds.data(), audioEmbeds.size() * sizeof(half),
            cudaMemcpyHostToDevice));

        // Run GPU kernel
        kernel::embeddingLookupMultimodal(inputIdsTensor, embeddingTableTensor, multimodalIndicesTensor, imageTokenId,
            imageEmbedsTensor, audioTokenId, audioEmbedsTensor, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult
            = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
                multimodalIndices, imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Qwen3-Omni Embedding Lookup Accuracy Test"))
            << "GPU and CPU results don't match for test case: batchSize=" << batchSize << ", seqLen=" << seqLen
            << ", vocabSize=" << vocabSize << ", hiddenSize=" << hiddenSize << ", imageTokenLen=" << imageTokenLen
            << ", audioTokenLen=" << audioTokenLen;
    }
}

// Test Qwen3-Omni embedding lookup with out-of-bounds handling
TEST_F(EmbeddingLookupTest, MultimodalOutOfBounds)
{
    // Test case with specific tokens to verify out-of-bounds handling
    int64_t const batchSize = 1;
    int64_t const seqLen = 10;
    int32_t const vocabSize = 1000;
    int64_t const hiddenSize = 128;
    int64_t const imageTokenLen = 8;
    int64_t const audioTokenLen = 4;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    // Create inputIds with specific patterns:
    // - Text tokens (valid and invalid)
    // - Image tokens (valid and invalid indices)
    // - Audio tokens (valid and invalid indices)
    std::vector<int32_t> inputIds = {
        100,          // Valid text token
        imageTokenId, // Valid image token (index 0)
        audioTokenId, // Valid audio token (index 0)
        imageTokenId, // Valid image token (index 1)
        audioTokenId, // Valid audio token (index 1)
        -1,           // Invalid text token (out of bounds)
        imageTokenId, // Image token with out-of-bounds index
        audioTokenId, // Audio token with out-of-bounds index
        2000,         // Invalid text token (> vocabSize)
        500,          // Valid text token
    };

    std::vector<int32_t> multimodalIndices = {
        0,  // Ignored (text token)
        0,  // Valid image index
        0,  // Valid audio index
        1,  // Valid image index
        1,  // Valid audio index
        0,  // Ignored (text token)
        10, // Invalid image index (>= imageTokenLen)
        -1, // Invalid audio index (< 0)
        0,  // Ignored (text token)
        0,  // Ignored (text token)
    };

    // Generate embedding tables
    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
    uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords audioShape{audioTokenLen, hiddenSize};
    rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(multimodalIndicesTensor.rawPointer(), multimodalIndices.data(),
        multimodalIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        audioEmbedsTensor.rawPointer(), audioEmbeds.data(), audioEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Run GPU kernel
    kernel::embeddingLookupMultimodal(inputIdsTensor, embeddingTableTensor, multimodalIndicesTensor, imageTokenId,
        imageEmbedsTensor, audioTokenId, audioEmbedsTensor, outputTensor, stream);

    // Get result from GPU
    std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
    CUDA_CHECK(cudaMemcpy(
        gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

    // Run CPU reference
    auto cpuResult = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
        multimodalIndices, imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen);

    // Compare results
    EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Qwen3-Omni Embedding Lookup Out-of-Bounds Test"))
        << "GPU and CPU results don't match for out-of-bounds handling";

    // Verify specific token behaviors
    std::vector<bool> shouldBeZero = {
        false, // Valid text token
        false, // Valid image token
        false, // Valid audio token
        false, // Valid image token
        false, // Valid audio token
        true,  // Invalid text token (-1)
        true,  // Image token with out-of-bounds index (10)
        true,  // Audio token with out-of-bounds index (-1)
        true,  // Invalid text token (2000 > vocabSize)
        false, // Valid text token
    };

    for (int64_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
    {
        if (shouldBeZero[tokenIdx])
        {
            // Check that all elements for this token are zero
            for (int64_t elementIdx = 0; elementIdx < hiddenSize; ++elementIdx)
            {
                int64_t const resultIdx = tokenIdx * hiddenSize + elementIdx;
                EXPECT_TRUE(isclose(gpuResult[resultIdx], __float2half(0.0f), 1e-6, 1e-6))
                    << "Token at position " << tokenIdx << " (tokenId=" << inputIds[tokenIdx]
                    << ", multimodalIndices=" << multimodalIndices[tokenIdx]
                    << ") should produce zero embedding at element " << elementIdx;
            }
        }
    }
}

// Test that Qwen3-Omni kernel properly errors out for uneven hiddenSize
TEST_F(EmbeddingLookupTest, MultimodalUnevenHiddenSizeError)
{
    // Test case with hiddenSize = 15 (not a multiple of 8)
    int64_t const batchSize = 1;
    int64_t const seqLen = 10;
    int32_t const vocabSize = 100;
    int64_t const hiddenSize = 15; // Not a multiple of 8
    int64_t const imageTokenLen = 4;
    int64_t const audioTokenLen = 4;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    // Generate test data
    std::vector<int32_t> inputIds(batchSize * seqLen);
    uniformIntInitialization<int32_t>(inputIds, 0, vocabSize - 1);

    std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

    std::vector<half> embeddingTable(vocabSize * hiddenSize);
    uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

    std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
    uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);

    std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
    uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);

    // Create tensors
    rt::Coords inputShape{batchSize, seqLen};
    rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Coords embeddingShape{vocabSize, hiddenSize};
    rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords imageShape{imageTokenLen, hiddenSize};
    rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords audioShape{audioTokenLen, hiddenSize};
    rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    rt::Coords outputShape{batchSize, seqLen, hiddenSize};
    rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(
        inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(multimodalIndicesTensor.rawPointer(), multimodalIndices.data(),
        multimodalIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
        embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        audioEmbedsTensor.rawPointer(), audioEmbeds.data(), audioEmbeds.size() * sizeof(half), cudaMemcpyHostToDevice));

    // Expect the kernel to throw an error due to uneven hiddenSize
    EXPECT_THROW(
        {
            kernel::embeddingLookupMultimodal(inputIdsTensor, embeddingTableTensor, multimodalIndicesTensor,
                imageTokenId, imageEmbedsTensor, audioTokenId, audioEmbedsTensor, outputTensor, stream);
        },
        std::runtime_error)
        << "Qwen3-Omni kernel should error out when hiddenSize is not a multiple of 8";
}

// Test multimodal embedding lookup with optional inputs (image only, audio only, text only)
TEST_F(EmbeddingLookupTest, MultimodalOptionalInputs)
{
    struct TestCase
    {
        bool hasImage;
        bool hasAudio;
        std::string name;
    };

    std::vector<TestCase> testCases = {
        {true, false, "TextImageOnly"},
        {false, true, "TextAudioOnly"},
        {false, false, "TextOnly"},
    };

    int64_t const batchSize = 2;
    int64_t const seqLen = 32;
    int32_t const vocabSize = 200;
    int64_t const hiddenSize = 128;
    int64_t const imageTokenLen = 16;
    int64_t const audioTokenLen = 12;

    int32_t const imageTokenId = 33;
    int32_t const audioTokenId = 44;

    for (auto const& tc : testCases)
    {
        SCOPED_TRACE("Testing: " + tc.name);

        // Create inputIds with appropriate token mix
        std::vector<int32_t> inputIds(batchSize * seqLen);
        std::vector<int32_t> multimodalIndices(batchSize * seqLen, 0);

        int32_t imageCounter = 0;
        int32_t audioCounter = 0;
        for (int64_t i = 0; i < batchSize * seqLen; ++i)
        {
            if (tc.hasImage && i % 4 == 0 && imageCounter < imageTokenLen)
            {
                inputIds[i] = imageTokenId;
                multimodalIndices[i] = imageCounter++;
            }
            else if (tc.hasAudio && i % 5 == 0 && audioCounter < audioTokenLen)
            {
                inputIds[i] = audioTokenId;
                multimodalIndices[i] = audioCounter++;
            }
            else
            {
                inputIds[i] = i % vocabSize;
            }
        }

        // Generate embedding tables
        std::vector<half> embeddingTable(vocabSize * hiddenSize);
        uniformFloatInitialization<half>(embeddingTable, -1.0f, 1.0f);

        std::vector<half> imageEmbeds(imageTokenLen * hiddenSize);
        std::vector<half> audioEmbeds(audioTokenLen * hiddenSize);
        if (tc.hasImage)
        {
            uniformFloatInitialization<half>(imageEmbeds, -1.0f, 1.0f);
        }
        if (tc.hasAudio)
        {
            uniformFloatInitialization<half>(audioEmbeds, -1.0f, 1.0f);
        }

        // Create tensors
        rt::Coords inputShape{batchSize, seqLen};
        rt::Tensor inputIdsTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor multimodalIndicesTensor(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        rt::Coords embeddingShape{vocabSize, hiddenSize};
        rt::Tensor embeddingTableTensor(embeddingShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords imageShape{imageTokenLen, hiddenSize};
        rt::Tensor imageEmbedsTensor(imageShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords audioShape{audioTokenLen, hiddenSize};
        rt::Tensor audioEmbedsTensor(audioShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        rt::Coords outputShape{batchSize, seqLen, hiddenSize};
        rt::Tensor outputTensor(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(
            inputIdsTensor.rawPointer(), inputIds.data(), inputIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(multimodalIndicesTensor.rawPointer(), multimodalIndices.data(),
            multimodalIndices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(embeddingTableTensor.rawPointer(), embeddingTable.data(),
            embeddingTable.size() * sizeof(half), cudaMemcpyHostToDevice));
        if (tc.hasImage)
        {
            CUDA_CHECK(cudaMemcpy(imageEmbedsTensor.rawPointer(), imageEmbeds.data(), imageEmbeds.size() * sizeof(half),
                cudaMemcpyHostToDevice));
        }
        if (tc.hasAudio)
        {
            CUDA_CHECK(cudaMemcpy(audioEmbedsTensor.rawPointer(), audioEmbeds.data(), audioEmbeds.size() * sizeof(half),
                cudaMemcpyHostToDevice));
        }

        // Set up optional parameters for kernel call
        bool const hasMultimodal = tc.hasImage || tc.hasAudio;
        rt::OptionalInputTensor multimodalIndicesOpt
            = hasMultimodal ? rt::OptionalInputTensor(multimodalIndicesTensor) : std::nullopt;
        std::optional<int32_t> imageTokenIdOpt = tc.hasImage ? std::optional(imageTokenId) : std::nullopt;
        rt::OptionalInputTensor imageEmbedsOpt
            = tc.hasImage ? rt::OptionalInputTensor(imageEmbedsTensor) : std::nullopt;
        std::optional<int32_t> audioTokenIdOpt = tc.hasAudio ? std::optional(audioTokenId) : std::nullopt;
        rt::OptionalInputTensor audioEmbedsOpt
            = tc.hasAudio ? rt::OptionalInputTensor(audioEmbedsTensor) : std::nullopt;

        // Run GPU kernel
        kernel::embeddingLookupMultimodal(inputIdsTensor, embeddingTableTensor, multimodalIndicesOpt, imageTokenIdOpt,
            imageEmbedsOpt, audioTokenIdOpt, audioEmbedsOpt, outputTensor, stream);

        // Get result from GPU
        std::vector<half> gpuResult(batchSize * seqLen * hiddenSize);
        CUDA_CHECK(cudaMemcpy(
            gpuResult.data(), outputTensor.rawPointer(), gpuResult.size() * sizeof(half), cudaMemcpyDeviceToHost));

        // Run CPU reference
        auto cpuResult
            = embeddingLookupMultimodalRef(inputIds, embeddingTable, batchSize, seqLen, vocabSize, hiddenSize,
                multimodalIndices, tc.hasImage ? imageTokenId : -1, tc.hasImage ? imageEmbeds : std::vector<half>{},
                tc.hasImage ? imageTokenLen : 0, tc.hasAudio ? audioTokenId : -1,
                tc.hasAudio ? audioEmbeds : std::vector<half>{}, tc.hasAudio ? audioTokenLen : 0);

        // Compare results
        EXPECT_TRUE(compareResults(cpuResult, gpuResult, "Multimodal Embedding Lookup " + tc.name + " Test"))
            << "GPU and CPU results don't match for " << tc.name << " multimodal lookup";
    }
}
