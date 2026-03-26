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

#include "llmInferenceRuntime.h"

#include <algorithm>

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{
std::string getDataTypeString(nvinfer1::DataType dataType);
bool copyTensorToHostBytes(rt::Tensor const& tensor, std::vector<uint8_t>& hostData, cudaStream_t stream);

bool isDebugQwenTextPreprocessEnabled()
{
    char const* env = std::getenv("EDGELLM_DEBUG_QWEN_TEXTPRE");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

std::optional<std::filesystem::path> getDebugLogitsDir()
{
    char const* env = std::getenv("EDGELLM_DEBUG_LOGITS_DIR");
    if (env == nullptr || *env == '\0')
    {
        return std::nullopt;
    }
    return std::filesystem::path(env);
}

void dumpTopKLogits(rt::Tensor const& logitsTensor, int32_t selectedTokenId, int32_t generationStep, cudaStream_t stream,
    std::filesystem::path const& outputDir, int32_t topK = 10)
{
    if (logitsTensor.getDataType() != DataType::kFLOAT)
    {
        LOG_WARNING("Skipping logits dump because mOutputLogits dtype is %s, expected FLOAT32",
            getDataTypeString(logitsTensor.getDataType()).c_str());
        return;
    }

    auto const shape = logitsTensor.getShape();
    check::check(shape.getNumDims() == 2, "Expected logits tensor to be 2D [B, vocab]");
    int64_t const batchSize = shape[0];
    int64_t const vocabSize = shape[1];
    if (batchSize <= 0 || vocabSize <= 0)
    {
        return;
    }

    std::vector<uint8_t> hostBytes;
    if (!copyTensorToHostBytes(logitsTensor, hostBytes, stream))
    {
        LOG_WARNING("Failed to copy logits tensor for debug dump");
        return;
    }

    float const* logitsPtr = reinterpret_cast<float const*>(hostBytes.data());
    std::vector<std::pair<float, int32_t>> top;
    top.reserve(static_cast<size_t>(topK));
    for (int64_t tokenId = 0; tokenId < vocabSize; ++tokenId)
    {
        float const score = logitsPtr[tokenId];
        if (static_cast<int32_t>(top.size()) < topK)
        {
            top.emplace_back(score, static_cast<int32_t>(tokenId));
            if (static_cast<int32_t>(top.size()) == topK)
            {
                std::sort(top.begin(), top.end(),
                    [](auto const& a, auto const& b) { return a.first > b.first; });
            }
        }
        else if (score > top.back().first)
        {
            top.back() = {score, static_cast<int32_t>(tokenId)};
            std::sort(top.begin(), top.end(), [](auto const& a, auto const& b) { return a.first > b.first; });
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);
    if (ec)
    {
        LOG_WARNING("Failed to create logits debug dir '%s': %s", outputDir.c_str(), ec.message().c_str());
        return;
    }

    nlohmann::json record;
    record["generation_step"] = generationStep;
    record["selected_token_id"] = selectedTokenId;
    record["selected_logit"] = logitsPtr[selectedTokenId];
    record["topk"] = nlohmann::json::array();
    for (auto const& [score, tokenId] : top)
    {
        record["topk"].push_back({{"token_id", tokenId}, {"logit", score}});
    }

    auto const filePath = outputDir / "step_logits.jsonl";
    std::ofstream out(filePath, std::ios::app);
    out << record.dump() << '\n';
}

bool copyTensorToHostBytes(rt::Tensor const& tensor, std::vector<uint8_t>& hostData, cudaStream_t stream)
{
    size_t const byteSize = tensor.getShape().volume() * rt::utils::getTypeSize(tensor.getDataType());
    hostData.resize(byteSize);
    if (byteSize == 0)
    {
        return true;
    }

    if (tensor.getDeviceType() == rt::DeviceType::kGPU)
    {
        CUDA_CHECK(cudaMemcpyAsync(hostData.data(), tensor.rawPointer(), byteSize, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else
    {
        std::memcpy(hostData.data(), tensor.rawPointer(), byteSize);
    }
    return true;
}

std::string getDataTypeString(nvinfer1::DataType dataType)
{
    using DataType = nvinfer1::DataType;
    switch (dataType)
    {
    case DataType::kFLOAT: return "FLOAT32";
    case DataType::kHALF: return "FLOAT16";
    case DataType::kBF16: return "BF16";
    case DataType::kINT8: return "INT8";
    case DataType::kINT32: return "INT32";
    case DataType::kINT64: return "INT64";
    case DataType::kBOOL: return "BOOL";
    case DataType::kUINT8: return "UINT8";
    case DataType::kFP8: return "FP8";
    default: return "UNKNOWN";
    }
}

std::vector<int64_t> coordsToVector(rt::Coords const& coords)
{
    std::vector<int64_t> values;
    values.reserve(coords.getNumDims());
    for (int32_t i = 0; i < coords.getNumDims(); ++i)
    {
        values.push_back(coords[i]);
    }
    return values;
}

std::tuple<std::string, std::string> keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

//! \brief Generate multimodal indices for embeddingLookupMultimodal kernel
//!
//! This function scans the input IDs and generates indices for audio/image embeddings.
//! For each multimodal token position, it assigns a sequential index starting from 0.
//!
//! \param inputIds Input token IDs on CPU [batchSize, seqLen]
//! \param audioTokenId Special token ID for audio (<|audio_pad|>), or std::nullopt if no audio
//! \param imageTokenId Special token ID for image (usually >= vocabSize), or std::nullopt if no image
//! \param vocabSize Vocabulary size (for detecting image tokens when imageTokenId is not explicitly set)
//! \return multimodalIndices tensor on CPU [batchSize, seqLen]
rt::Tensor generateMultimodalIndices(rt::Tensor const& inputIds, std::optional<int32_t> audioTokenId,
    std::optional<int32_t> imageTokenId, int32_t vocabSize)
{
    auto const shape = inputIds.getShape();
    check::check(shape.getNumDims() == 2, "inputIds must be 2D tensor");
    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];

    // Create output tensor on CPU
    rt::Tensor multimodalIndices({batchSize, seqLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t* indicesPtr = multimodalIndices.dataPointer<int32_t>();

    // Global indexing across the entire batch (not per-sample)
    int32_t audioIndex = 0;
    int32_t imageIndex = 0;

    // Process each sample in the batch
    for (int64_t b = 0; b < batchSize; ++b)
    {
        for (int64_t s = 0; s < seqLen; ++s)
        {
            int64_t const pos = b * seqLen + s;
            int32_t const tokenId = inputIdsPtr[pos];

            // Check if this is an audio token
            if (audioTokenId.has_value() && tokenId == *audioTokenId)
            {
                indicesPtr[pos] = audioIndex++;
            }
            // Check if this is an image token (either explicit imageTokenId or >= vocabSize)
            else if ((imageTokenId.has_value() && tokenId == *imageTokenId) || tokenId >= vocabSize)
            {
                indicesPtr[pos] = imageIndex++;
            }
            else
            {
                // Normal text token, index not used
                indicesPtr[pos] = 0;
            }
        }
    }

    return multimodalIndices;
}

} // namespace
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    // Find the first .engine file in engineDir
    // For Qwen3-Omni: export ensures only thinker.engine exists in this directory
    std::filesystem::path enginePath;
    for (auto const& entry : std::filesystem::directory_iterator(engineDir))
    {
        if (entry.path().extension() == ".engine")
        {
            enginePath = entry.path();
            break;
        }
    }
    if (enginePath.empty())
    {
        throw std::runtime_error("No .engine file found in directory: " + engineDir);
    }
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    // Load embedding table from embedding.safetensors
    std::filesystem::path const embeddingPath = std::filesystem::path(engineDir) / "embedding.safetensors";
    LOG_INFO("Loading embedding table from: %s", embeddingPath.string().c_str());
    std::vector<rt::Tensor> embeddingTensors;
    if (!safetensors::loadSafetensors(embeddingPath, embeddingTensors, stream))
    {
        LOG_ERROR("Failed to load embedding table from: %s", embeddingPath.string().c_str());
        throw std::runtime_error("Failed to load embedding table from: " + embeddingPath.string());
    }
    check::check(embeddingTensors.size() == 1, "embedding.safetensors should contain exactly one tensor");
    check::check(
        embeddingTensors[0].getShape().getNumDims() == 2, "embedding tensor should be 2D [vocabSize, hiddenSize]");
    mEmbeddingTable = std::move(embeddingTensors[0]);
    LOG_INFO("Embedding table loaded successfully with shape [%d, %d]", mEmbeddingTable.getShape()[0],
        mEmbeddingTable.getShape()[1]);

    try
    {
        mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize LLMEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("LLMEngineRunner successfully loaded and initialized llm engine.");

    mEngineConfig = mLLMEngineRunner->getEngineConfig();

    // Use TopP sampling parameter to reserve max possible workspace size for sampling.
    int32_t const defaultTopK{0};
    float const defaultTopP{0.9F};
    trt_edgellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(trt_edgellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, samplingParams));

    // Allocate workspace and activation tensors for LLM engine.
    try
    {
        // Use Int8 to indicate byte for workspace.
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mSamplingWorkspace");
        mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mInputIds");
        mInputsEmbeds = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
            rt::DeviceType::kGPU, DataType::kHALF, "LLMInferenceRuntime::mInputsEmbeds");
        // Allocate deepstack embeddings if needed (one tensor per feature)
        if (mEngineConfig.numDeepstackFeatures > 0)
        {
            mDeepstackEmbeds.resize(mEngineConfig.numDeepstackFeatures);
            for (int32_t i = 0; i < mEngineConfig.numDeepstackFeatures; ++i)
            {
                mDeepstackEmbeds[i] = rt::Tensor({mEngineConfig.maxSupportedBatchSize,
                                                     mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize},
                    rt::DeviceType::kGPU, DataType::kHALF,
                    format::fmtstr("LLMInferenceRuntime::mDeepstackEmbeds[%d]", i));
            }
            LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]",
                mEngineConfig.numDeepstackFeatures, mEngineConfig.maxSupportedBatchSize,
                mEngineConfig.maxSupportedInputLength, mEngineConfig.hiddenSize);
        }
        mHostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kCPU, DataType::kINT32, "LLMInferenceRuntime::mHostPackedInputIds");
        mOutputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMInferenceRuntime::mOutputLogits");
        mSelectedIndices = rt::Tensor({mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mSelectedIndices");
        mHostSelectedTokenIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostContextLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostContextLengths");
        mHostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostReuseKVCacheLengths");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate workspace and activation tensors for LLM Inference Runtime: %s", e.what());
        throw std::runtime_error(
            "Failed to allocate workspace and activation tensors for LLM Inference Runtime: " + std::string(e.what()));
    }

    // Setup tokenizer
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }

    // Optional: Load vocabulary mapping table if reduced vocabulary is used
    if (mEngineConfig.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for reduced vocab size: %d -> %d", mEngineConfig.reducedVocabSize,
            mEngineConfig.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        if (!safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream))
        {
            LOG_ERROR(
                "Failed to load %s from model directory: %s", binding_names::kVocabMapFileName, engineDir.c_str());
            throw std::runtime_error("Failed to load " + std::string(binding_names::kVocabMapFileName)
                + " from model directory: " + engineDir);
        }

        // Check we have exactly one tensor and use it
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mEngineConfig.reducedVocabSize,
            "vocab_map tensor length should match reduced vocab size");
        mVocabMappingTable = std::move(vocabMapTensors[0]);
        LOG_INFO("Vocabulary mapping table successfully loaded.");
    }

    // Optional: Setup multimodal engine runners
    if (!multimodalEngineDir.empty())
    {
        // Multimodal engine directory structure:
        //   multimodalEngineDir/audio/  - Audio encoder (audio_encoder.engine, config.json)
        //   multimodalEngineDir/visual/ - Visual encoder (visual.engine, config.json)
        //
        // Note: audio_build and visual_build automatically append /audio and /visual subdirectories.
        // Both builders should use the same base --engineDir path.

        // Helper lambda to try loading a runner from a directory
        auto tryLoadRunner = [&](std::string const& dir, std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            try
            {
                LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
                auto runner = MultimodalRunner::create(
                    dir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
                LOG_INFO("%s runner successfully initialized", name.c_str());
                return runner;
            }
            catch (std::exception const& e)
            {
                LOG_DEBUG("Failed to load %s runner from %s: %s", name.c_str(), dir.c_str(), e.what());
                return nullptr;
            }
        };

        // Try to load audio runner from multimodalEngineDir/audio
        mAudioRunner = tryLoadRunner(multimodalEngineDir + "/audio", "Audio");

        // Try to load visual runner from multimodalEngineDir/visual (with fallback to root for pure visual models)
        mVisionRunner = tryLoadRunner(multimodalEngineDir + "/visual", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = tryLoadRunner(multimodalEngineDir, "Vision");
        }

        // At least one runner must be available
        if (!mAudioRunner && !mVisionRunner)
        {
            throw std::runtime_error("No valid multimodal engine found in " + multimodalEngineDir);
        }
    }
}

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request) noexcept
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("LLMInferenceRuntime(): The request is empty with no requests supplied.");
        return false;
    }

    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("LLMInferenceRuntime(): The batched request size (%d) exceeds the max supported batch size (%d).",
            activeBatchSize, mEngineConfig.maxSupportedBatchSize);
        return false;
    }

    for (auto const& request : request.requests)
    {
        if (request.messages.empty())
        {
            LOG_ERROR(
                "There is an empty request in the batch. 'messages' must be provided. "
                "Skip this batch of requests. Please check the input data contents.");
            return false;
        }
    }

    return true;
}

bool LLMInferenceRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
    std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    // Record the length of the reused KVCache for each sequence using pre-allocated tensor
    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();

    // Search if the system prompt has been cached. If there are cached system prompts, insert
    // the pre-computed KVCache and remove the contents from inputIds.
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptKey];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            auto reuseLength = math::cast<size_t>(kvCacheContent.getShape()[3]);
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(math::cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = math::cast<int32_t>(reuseLength);
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            bool const matchIds = std::equal(precachedKVCache.tokenizedPrompt.begin(),
                precachedKVCache.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "LLMInferenceRuntime(): Though system prompt strings are matched, token_ids are not perfectly "
                    "aligned. "
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    // Pack inputIds, instantiate input data for prefill step, and reset the KVCache state.
    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The max input length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    // Reshape and fill the pre-allocated pinned host tensor with pad tokens
    int32_t const packedInputLength = maxInputLength;
    check::check(mHostPackedInputIds.reshape({activeBatchSize, packedInputLength}), "Tensor reshape failed");
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Pad each sequence to the max length of this batch.
        // TODO: Implement remove input padding for better efficiency until multi-batch.
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // For each Mamba layer and each batch element, either restore the cached
    // SSM/conv state if the cache hit, or zero the state.
    if (mEngineConfig.numMambaLayers > 0)
    {
        rt::LinearKVCache& kvCache = mLLMEngineRunner->getLinearKVCache();
        rt::LinearKVCache::CacheConfig const& cacheConfig = kvCache.getConfig();
        size_t const ssmElemSize = rt::utils::getTypeSize(cacheConfig.ssmStateType);
        size_t const convElemSize = rt::utils::getTypeSize(cacheConfig.convStateType);
        size_t const ssmBatchBytes
            = static_cast<size_t>(cacheConfig.mambaNumHeads * cacheConfig.mambaHeadDim * cacheConfig.ssmStateSize)
            * ssmElemSize;
        size_t const convBatchBytes = static_cast<size_t>(cacheConfig.convDim * cacheConfig.convKernel) * convElemSize;

        for (int32_t layer = 0; layer < mEngineConfig.numMambaLayers; ++layer)
        {
            rt::Tensor ssmLayer = kvCache.getSSMStateForLayer(layer);
            rt::Tensor convLayer = kvCache.getConvStateForLayer(layer);

            for (int32_t i = 0; i < activeBatchSize; ++i)
            {
                auto* ssmDst = static_cast<std::byte*>(ssmLayer.rawPointer()) + i * ssmBatchBytes;
                auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + i * convBatchBytes;

                auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
                auto it = mSystemPromptKVCache.find(promptKey);
                bool const hasCache = (it != mSystemPromptKVCache.end());

                if (hasCache && layer < static_cast<int32_t>(it->second.ssmStateContents.size()))
                {
                    CUDA_CHECK(cudaMemcpyAsync(ssmDst, it->second.ssmStateContents[layer].rawPointer(), ssmBatchBytes,
                        cudaMemcpyDeviceToDevice, stream));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(ssmDst, 0, ssmBatchBytes, stream));
                }

                if (hasCache && layer < static_cast<int32_t>(it->second.convStateContents.size()))
                {
                    CUDA_CHECK(cudaMemcpyAsync(convDst, it->second.convStateContents[layer].rawPointer(),
                        convBatchBytes, cudaMemcpyDeviceToDevice, stream));
                }
                else
                {
                    CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
                }
            }
        }
    }

    check::check(mInputIds.reshape({activeBatchSize, packedInputLength}), "Tensor reshape failed");
    check::check(mHostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize}), "Tensor reshape failed");

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", loraWeightsName.c_str());
        return false;
    }

    return true;
}

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;
    std::string loraWeightsName = request.loraWeightsName;

    if (!examineRequest(request))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request examination failed. This request cannot be handled.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    // Apply chat template, extract system prompts, and optionally save KVCache
    request.formattedRequests.resize(activeBatchSize);
    batchSystemPrompts.reserve(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.continueFinalMessage, request.enableThinking);

        // Extract system prompt
        batchSystemPrompts.emplace_back(request.formattedRequests[i].formattedSystemPrompt);

        // Save KVCache if requested
        if (request.saveSystemPromptKVCache)
        {
            if (mVisionRunner)
            {
                mVisionRunner->preprocessSystemPrompt(
                    batchSystemPrompts[i], mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }
            else if (mAudioRunner)
            {
                mAudioRunner->preprocessSystemPrompt(
                    batchSystemPrompts[i], mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(batchSystemPrompts[i], loraWeightsName, stream);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache. Continue to handle the request without saving the system "
                    "prompt KVCache.");
            }
        }
    }

    // Preprocess user prompts and encode them.
    // Check if request has audio or vision inputs
    bool hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });

    if ((hasAudio && mAudioRunner) || (hasVision && mVisionRunner))
    {
        // Mark multimodal preprocessing and inference for NVTX profiling
        NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);

        // Process audio inputs (if present)
        if (hasAudio && mAudioRunner)
        {
            LOG_INFO("Processing audio inputs");
            if (!mAudioRunner->preprocess(
                    request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Audio preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (!mAudioRunner->infer(stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Audio inference failed. This request cannot be handled.");
                return false;
            }
        }

        // Process vision inputs (if present)
        if (hasVision && mVisionRunner)
        {
            LOG_INFO("Processing vision inputs");
            if (mDumpPreparedPrefillInputsEnabled)
            {
                auto const debugRequestDir
                    = mDumpPreparedPrefillInputsDir / format::fmtstr("request_%zu", mPrefillInputDumpCounter);
                mVisionRunner->beginDebugInputsDump(debugRequestDir);
            }
            if (!mVisionRunner->preprocess(
                    request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Vision preprocessing failed. This request cannot be handled.");
                return false;
            }

            if (isDebugQwenTextPreprocessEnabled())
            {
                for (size_t reqIdx = 0; reqIdx < batchedInputIds.size(); ++reqIdx)
                {
                    LOG_INFO("LLMInferenceRuntime debug: batchedInputIds[%zu].size() after vision preprocess = %zu",
                        reqIdx, batchedInputIds[reqIdx].size());
                }
            }

            if (mDumpPreparedPrefillInputsEnabled)
            {
                auto const debugRequestDir
                    = mDumpPreparedPrefillInputsDir / format::fmtstr("request_%zu", mPrefillInputDumpCounter);
                if (!mVisionRunner->dumpDebugInputs(debugRequestDir, stream))
                {
                    LOG_WARNING("Failed to dump visual debug inputs. Continuing without visual debug artifacts.");
                }
            }

            if (!mVisionRunner->infer(stream))
            {
                LOG_ERROR("LLMInferenceRuntime(): Vision inference failed. This request cannot be handled.");
                return false;
            }
        }
    }
    else
    {
        // Pure text mode: directly tokenize
        batchedInputIds.reserve(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, true));
            if (batchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to encode input text for request %d in batch", i);
                return false;
            }
        }
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(batchedInputIds, batchSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR("LLMInferenceRuntime(): Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // Record context information for performance tracking
    auto tokenCount = calculateTokenCounts(batchedInputIds, batchSystemPrompts, loraWeightsName);

    int32_t const maxInputIdsLength = mInputIds.getShape()[1];
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (maxInputIdsLength + maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
    {
        maxGenerationLength = mEngineConfig.maxKVCacheCapacity - maxInputIdsLength;
        LOG_WARNING(
            "The requested input length (%d) + max generation length (%d) = %d exceeds the max KV "
            "cache capacity (%d). Reduce the generation length to %d to avoid the truncation of the generated tokens.",
            maxInputIdsLength, maxGenerationLength, maxInputIdsLength + maxGenerationLength,
            mEngineConfig.maxKVCacheCapacity, maxGenerationLength);
    }

    // Set up data structures to store the generated results during decoding.
    // Also set up sampling parameters and sampling lambda function.
    int32_t unFinishedBatchNum = activeBatchSize;
    int32_t generationIter{0};
    std::vector<std::vector<int32_t>> outputIds(activeBatchSize);
    std::vector<bool> finishedStates(activeBatchSize, false);
    check::check(mSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();

    SamplingParams params(
        activeBatchSize, mEngineConfig.outputVocabSize, request.temperature, request.topK, request.topP);
    auto const debugLogitsDir = getDebugLogitsDir();
    auto sampleTokens = [&]() {
        trt_edgellm::topKtopPSamplingFromLogits(mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
        // Apply vocabulary mapping if reduced vocabulary is used
        if (mEngineConfig.reducedVocabSize > 0)
        {
            trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (debugLogitsDir.has_value() && activeBatchSize > 0)
        {
            dumpTopKLogits(mOutputLogits, hostSelectedTokenIdsData[0], generationIter, stream, *debugLogitsDir);
        }
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!finishedStates[i])
            {
                outputIds[i].push_back(hostSelectedTokenIdsData[i]);
                finishedStates[i] = hostSelectedTokenIdsData[i] == mTokenizer->getEosId();
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
    };

    // Perform embedding lookup for prefill
    int32_t const prefillSequenceLength = mInputIds.getShape()[1];
    check::check(mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
        "Tensor reshape failed");

    // Get embeddings from independent runners
    rt::OptionalInputTensor visionEmbeddings
        = mVisionRunner ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = mAudioRunner ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor multimodalEmbeddings = visionEmbeddings.has_value() ? visionEmbeddings : audioEmbeddings;

    if (audioEmbeddings.has_value())
    {
        // Audio present: use embeddingLookupMultimodal (handles audio and/or vision)
        auto const inputShape = mInputIds.getShape();
        size_t const inputSizeBytes = inputShape.volume() * sizeof(int32_t);
        rt::Tensor inputIdsCPU(inputShape, rt::DeviceType::kCPU, mInputIds.getDataType());
        CUDA_CHECK(
            cudaMemcpy(inputIdsCPU.rawPointer(), mInputIds.rawPointer(), inputSizeBytes, cudaMemcpyDeviceToHost));

        std::optional<int32_t> audioTokenId
            = (mEngineConfig.audioTokenId != 0) ? std::optional{mEngineConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenId
            = (mEngineConfig.imageTokenId != 0) ? std::optional{mEngineConfig.imageTokenId} : std::nullopt;
        rt::Tensor multimodalIndicesCPU
            = generateMultimodalIndices(inputIdsCPU, audioTokenId, imageTokenId, mEngineConfig.vocabSize);

        auto const indicesShape = multimodalIndicesCPU.getShape();
        size_t const indicesSizeBytes = indicesShape.volume() * sizeof(int32_t);
        mMultimodalIndices = rt::Tensor(indicesShape, rt::DeviceType::kGPU, multimodalIndicesCPU.getDataType());
        CUDA_CHECK(cudaMemcpy(mMultimodalIndices.rawPointer(), multimodalIndicesCPU.rawPointer(), indicesSizeBytes,
            cudaMemcpyHostToDevice));

        kernel::embeddingLookupMultimodal(mInputIds, mEmbeddingTable, std::optional{std::ref(mMultimodalIndices)},
            imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings, mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Legacy vision path (Qwen2.5-VL, InternVL: imageTokenId >= vocabSize or not set)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding assembly if vision runner is available
    // Note: Deepstack features are only provided by VisionRunner, not Qwen3OmniAudioRunner
    rt::OptionalInputTensors deepstackEmbeds{};
    rt::OptionalInputTensors deepstackFeatures{};
    if (mEngineConfig.numDeepstackFeatures > 0 && mVisionRunner)
    {
        deepstackFeatures = mVisionRunner->getDeepstackFeatures();

        // Prepare multimodal indices for deepstack assembly (needed when imageTokenId < vocabSize)
        rt::OptionalInputTensor deepstackMultimodalIndices{std::nullopt};
        if (mMultimodalIndices.getShape().volume() > 0)
        {
            deepstackMultimodalIndices = std::ref(mMultimodalIndices);
        }

        for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
        {
            rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

            // Reshape and perform embedding assembly for this feature
            check::check(
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
                "Tensor reshape failed");
            kernel::assembleDeepstackEmbedding(mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx],
                stream, mEngineConfig.imageTokenId, deepstackMultimodalIndices);

            // Add to output vector (engine will bind by index)
            deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
        }
    }

    if (mDumpPreparedPrefillInputsEnabled)
    {
        if (!dumpPreparedPrefillInputs(
                activeBatchSize, prefillSequenceLength, multimodalEmbeddings, deepstackFeatures, deepstackEmbeds, stream))
        {
            LOG_WARNING("Failed to dump prepared prefill inputs");
        }
    }

    // Profile all sampling operations as one stage
    // Prefill profiling session
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);
        // Enhanced NVTX range with detailed information
        NVTX_SCOPED_RANGE(nvtx_prefill,
            ("LLM_PREFILL[BS=" + std::to_string(activeBatchSize)
                + ",Reused=" + std::to_string(tokenCount.totalReusedTokens)
                + ",Computed=" + std::to_string(tokenCount.totalComputedTokens) + "]")
                .c_str(),
            nvtx_colors::BLUE);

        bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputsEmbeds, mHostContextLengths, deepstackEmbeds,
            mOutputLogits, rt::OptionalOutputTensor{std::nullopt}, stream);
        if (!prefillStatus)
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Failed to execute prefill step. Cannot generate the KVCache for this prompt.");
            return false;
        }

        sampleTokens();
    }

    // Record prefill metrics
    mPrefillMetrics.recordRun(tokenCount.totalReusedTokens, tokenCount.totalComputedTokens);

    // Reshape for decoding step
    check::check(mInputsEmbeds.reshape({activeBatchSize, 1, mEngineConfig.hiddenSize}), "Tensor reshape failed");

    // Profile entire generation phase like benchmark profiler
    {
        TIME_STAGE(metrics::StageNames::kLLM_GENERATION, stream);
        // Enhanced NVTX range with batch size
        NVTX_SCOPED_RANGE(nvtx_generation,
            ("LLM_GENERATION[BS=" + std::to_string(activeBatchSize) + ",MaxLen=" + std::to_string(maxGenerationLength)
                + "]")
                .c_str(),
            nvtx_colors::GREEN);

        while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
        {
            // Mark each decoding iteration with detailed info
            NVTX_SCOPED_RANGE(iter_range,
                ("Decode_Iter[" + std::to_string(generationIter) + "/" + std::to_string(maxGenerationLength)
                    + ",Active=" + std::to_string(unFinishedBatchNum) + "]")
                    .c_str(),
                nvtx_colors::LIGHT_GREEN);

            // Perform embedding lookup for the selected token indices (decode only has text, no images)
            kernel::embeddingLookup(mSelectedIndices, mEmbeddingTable, mInputsEmbeds, stream);

            // Use the embedded tokens as input for the decoding step.
            // No hidden states output needed for standard LLM decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(
                mInputsEmbeds, mOutputLogits, outputHiddenStates, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            sampleTokens();
        }
    }

    // Record generation and sampling metrics
    int32_t totalGeneratedTokens = 0;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        totalGeneratedTokens += static_cast<int32_t>(outputIds[i].size() - 1);
    }

    if (totalGeneratedTokens > 0)
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
    }

    return true;
}

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    int32_t const maxSupportedBatchSize = mEngineConfig.maxSupportedBatchSize;
    int32_t const minSupportedBatchSize = 1;

    bool captureStatus{true};
    // Capture the CUDA graph for all available batch sizes.
    for (int32_t batchSize = minSupportedBatchSize; batchSize <= maxSupportedBatchSize; ++batchSize)
    {
        check::check(mSelectedIndices.reshape({batchSize, 1}), "Tensor reshape failed");
        check::check(mInputsEmbeds.reshape({batchSize, 1, mEngineConfig.hiddenSize}), "Tensor reshape failed");
        check::check(mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize}), "Tensor reshape failed");

        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
            mInputsEmbeds, mOutputLogits, mEmptyLoraWeightsName, stream);
        if (mEngineConfig.maxSupportedLoraRank > 0)
        {
            for (auto const& loraWeightsName : mLLMEngineRunner->getAvailableLoraWeights())
            {
                captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
                    mInputsEmbeds, mOutputLogits, loraWeightsName, stream);
            }
        }
    }

    if (captureStatus)
    {
        LOG_INFO(
            "LLMInferenceRuntime(): Successfully captured the decoding CUDA graph for all execution batch sizes and "
            "LoRA weights.");
    }
    else
    {
        LOG_WARNING(
            "LLMInferenceRuntime(): Failed to capture the decoding CUDA graph for some of execution batch sizes and "
            "LoRA weights.");
    }
    return captureStatus;
}

rt::Tensor LLMInferenceRuntime::getKVCacheBuffer()
{
    return mLLMEngineRunner->getLinearKVCache().getKVCacheBuffer();
}

rt::Tensor LLMInferenceRuntime::getKVCacheLengths()
{
    rt::Tensor& kvCacheLengths = mLLMEngineRunner->getLinearKVCache().getKVCacheLengths();
    return rt::Tensor(kvCacheLengths.rawPointer(), kvCacheLengths.getShape(), kvCacheLengths.getDeviceType(),
        kvCacheLengths.getDataType(), kvCacheLengths.getName());
}

rt::OptionalInputTensor LLMInferenceRuntime::getPositionIds() const
{
    return mVisionRunner ? mVisionRunner->getPositionIds() : std::nullopt;
}

rt::OptionalInputTensor LLMInferenceRuntime::getRopeDeltas() const
{
    return mVisionRunner ? mVisionRunner->getRopeDeltas() : std::nullopt;
}

std::string LLMInferenceRuntime::decodeTokenIds(std::vector<int32_t> const& tokenIds, bool skipSpecialTokens) const
{
    std::vector<tokenizer::Rank> ranks(tokenIds.begin(), tokenIds.end());
    return mTokenizer->decode(ranks, skipSpecialTokens);
}

LLMInferenceRuntime::TokenCountInfo LLMInferenceRuntime::calculateTokenCounts(
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<std::string> const& systemPrompts,
    std::string const& loraWeightsName) const noexcept
{
    TokenCountInfo tokenCount;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t contextLength = static_cast<int32_t>(batchedInputIds[i].size());
        // Calculate reused length from system prompt cache
        auto const promptKey = keySystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
        {
            int32_t reusedLength = static_cast<int32_t>(mSystemPromptKVCache.at(promptKey).tokenizedPrompt.size());
            tokenCount.totalReusedTokens += reusedLength;
            tokenCount.totalComputedTokens += (contextLength - reusedLength);
        }
        else
        {
            tokenCount.totalComputedTokens += contextLength;
        }
    }

    return tokenCount;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("LLMInferenceRuntime(): The prompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    // hash the prompt if check if the prompt cache already exists.
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptKey) != mSystemPromptKVCache.end())
    {
        LOG_DEBUG(
            "LLMInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());
    int32_t const activeBatchSize = 1;

    if (promptIdsLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The prompt length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            promptIdsLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(activeBatchSize, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(activeBatchSize, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): Prefill execution setup failed. Cannot generate the KVCache for this prompt.");
        return false;
    }

    // Execute prefill step to initialize the KVCache data.
    // Perform embedding lookup
    int32_t const prefillSequenceLength = mInputIds.getShape()[1];
    check::check(mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
        "Tensor reshape failed");

    // Get embeddings from independent runners
    rt::OptionalInputTensor visionEmbeddings
        = mVisionRunner ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = mAudioRunner ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;

    if (audioEmbeddings.has_value())
    {
        // Audio present: use embeddingLookupMultimodal (handles audio and/or vision)
        auto const inputShape = mInputIds.getShape();
        size_t const inputSizeBytes = inputShape.volume() * sizeof(int32_t);
        rt::Tensor inputIdsCPU(inputShape, rt::DeviceType::kCPU, mInputIds.getDataType());
        CUDA_CHECK(
            cudaMemcpy(inputIdsCPU.rawPointer(), mInputIds.rawPointer(), inputSizeBytes, cudaMemcpyDeviceToHost));

        std::optional<int32_t> audioTokenId
            = (mEngineConfig.audioTokenId != 0) ? std::optional{mEngineConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenId
            = (mEngineConfig.imageTokenId != 0) ? std::optional{mEngineConfig.imageTokenId} : std::nullopt;
        rt::Tensor multimodalIndicesCPU
            = generateMultimodalIndices(inputIdsCPU, audioTokenId, imageTokenId, mEngineConfig.vocabSize);

        auto const indicesShape = multimodalIndicesCPU.getShape();
        size_t const indicesSizeBytes = indicesShape.volume() * sizeof(int32_t);
        mMultimodalIndices = rt::Tensor(indicesShape, rt::DeviceType::kGPU, multimodalIndicesCPU.getDataType());
        CUDA_CHECK(cudaMemcpy(mMultimodalIndices.rawPointer(), multimodalIndicesCPU.rawPointer(), indicesSizeBytes,
            cudaMemcpyHostToDevice));

        kernel::embeddingLookupMultimodal(mInputIds, mEmbeddingTable, std::optional{std::ref(mMultimodalIndices)},
            imageTokenId, visionEmbeddings, audioTokenId, audioEmbeddings, mInputsEmbeds, stream);
    }
    else if (visionEmbeddings.has_value())
    {
        // Vision-only (Qwen2-VL, InternVL, etc.)
        rt::Tensor const& imageEmbedsTensor = visionEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup (pure text)
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding lookup if vision runner is available
    rt::OptionalInputTensors deepstackEmbeds{};
    if (mEngineConfig.numDeepstackFeatures > 0 && mVisionRunner)
    {
        rt::OptionalInputTensors deepstackFeatures = mVisionRunner->getDeepstackFeatures();

        rt::OptionalInputTensor deepstackMultimodalIndices{std::nullopt};
        if (mMultimodalIndices.getShape().volume() > 0)
        {
            deepstackMultimodalIndices = std::ref(mMultimodalIndices);
        }

        for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
        {
            rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

            check::check(
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize}),
                "Tensor reshape failed");
            kernel::assembleDeepstackEmbedding(mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx],
                stream, mEngineConfig.imageTokenId, deepstackMultimodalIndices);

            deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
        }
    }

    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    bool prefillStatus = mLLMEngineRunner->executePrefillStep(
        mInputsEmbeds, mHostContextLengths, deepstackEmbeds, mOutputLogits, outputHiddenStates, stream);
    if (!prefillStatus)
    {
        LOG_ERROR("LLMInferenceRuntime(): Failed to execute prefill step.");
        return false;
    }

    // Copy out the KVCache content from the prefill step.
    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numAttentionLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU,
        linearKVCache.getConfig().kvCacheTypeTRT, "LLMInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);

    // Save SSM and conv states for Mamba layers
    if (mEngineConfig.numMambaLayers > 0)
    {
        savedKVCache.ssmStateContents = mLLMEngineRunner->getLinearKVCache().captureSSMStates(CACHE_BATCH_IDX, stream);
        savedKVCache.convStateContents
            = mLLMEngineRunner->getLinearKVCache().captureConvStates(CACHE_BATCH_IDX, stream);
    }

    mSystemPromptKVCache.insert({promptKey, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

bool LLMInferenceRuntime::dumpPreparedPrefillInputs(int32_t activeBatchSize, int32_t prefillSequenceLength,
    rt::OptionalInputTensor const& multimodalEmbeddings, rt::OptionalInputTensors const& deepstackFeatures,
    rt::OptionalInputTensors const& deepstackEmbeds, cudaStream_t stream)
{
    namespace fs = std::filesystem;
    fs::path const requestDir = mDumpPreparedPrefillInputsDir / format::fmtstr("request_%zu", mPrefillInputDumpCounter++);
    std::error_code ec;
    fs::create_directories(requestDir, ec);
    if (ec)
    {
        LOG_ERROR("Failed to create prefill input dump directory '%s': %s", requestDir.c_str(), ec.message().c_str());
        return false;
    }

    auto dumpTensor = [&](rt::Tensor const& tensor, std::string const& fileName) -> nlohmann::json {
        std::vector<uint8_t> hostData;
        if (!copyTensorToHostBytes(tensor, hostData, stream))
        {
            throw std::runtime_error("Failed to copy tensor to host for debug dump");
        }
        std::ofstream out(requestDir / fileName, std::ios::binary);
        out.write(reinterpret_cast<char const*>(hostData.data()), static_cast<std::streamsize>(hostData.size()));
        return {{"file", fileName},
            {"shape", coordsToVector(tensor.getShape())},
            {"dtype", getDataTypeString(tensor.getDataType())},
            {"num_bytes", static_cast<int64_t>(hostData.size())}};
    };

    nlohmann::json meta;
    meta["input_ids"] = dumpTensor(mInputIds, "input_ids_request_0.bin");
    meta["inputs_embeds"] = dumpTensor(mInputsEmbeds, "inputs_embeds_request_0.bin");

    if (mMultimodalIndices.getShape().volume() > 0)
    {
        meta["multimodal_indices"] = dumpTensor(mMultimodalIndices, "multimodal_indices_request_0.bin");
    }

    if (multimodalEmbeddings.has_value())
    {
        meta["multimodal_output_embedding"]
            = dumpTensor(multimodalEmbeddings.value().get(), "multimodal_output_embedding_request_0.bin");
    }

    nlohmann::json deepstackFeatureEntries = nlohmann::json::array();
    for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
    {
        auto entry = dumpTensor(
            deepstackFeatures[idx].get(), format::fmtstr("deepstack_feature_raw_%d_request_0.bin", idx));
        entry["index"] = idx;
        deepstackFeatureEntries.push_back(entry);
    }
    meta["deepstack_features"] = deepstackFeatureEntries;

    nlohmann::json deepstackEmbedEntries = nlohmann::json::array();
    for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackEmbeds.size()); ++idx)
    {
        auto entry = dumpTensor(deepstackEmbeds[idx].get(), format::fmtstr("deepstack_embed_%d_request_0.bin", idx));
        entry["index"] = idx;
        deepstackEmbedEntries.push_back(entry);
    }
    meta["deepstack_embeds"] = deepstackEmbedEntries;

    meta["active_batch_size"] = activeBatchSize;
    meta["prefill_sequence_length"] = prefillSequenceLength;

    std::ofstream metaFile(requestDir / "prefill_inputs_request_0.json");
    metaFile << meta.dump(2);
    return true;
}

} // namespace rt
} // namespace trt_edgellm
