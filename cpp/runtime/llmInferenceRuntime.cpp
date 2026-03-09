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

#include "llmInferenceRuntime.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <chrono>
#include <fstream>
#include <functional>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{
std::tuple<std::string, std::string> keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

} // namespace
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
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
        mTrajectoryModeFlags = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mTrajectoryModeFlags");
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

    // Resolve token IDs used for trajectory-gated masking of index tokens (<i0..>).
    tokenizer::Rank indexTokenIdStart{-1};
    tokenizer::Rank indexTokenIdEnd{-1};
    tokenizer::Rank trajFutureStartTokenId{-1};
    tokenizer::Rank trajFutureEndTokenId{-1};
    bool const hasIndexTokenRange = mTokenizer->getIndexTokenIdRange(indexTokenIdStart, indexTokenIdEnd);
    bool const hasTrajFutureStart = mTokenizer->getSpecialTokenId("<|traj_future_start|>", trajFutureStartTokenId);
    bool const hasTrajFutureEnd = mTokenizer->getSpecialTokenId("<|traj_future_end|>", trajFutureEndTokenId);

    if (hasIndexTokenRange && hasTrajFutureStart && hasTrajFutureEnd && indexTokenIdStart >= 0
        && indexTokenIdEnd < mEngineConfig.outputVocabSize)
    {
        mEnableTrajectoryIndexTokenGating = true;
        mIndexTokenIdStart = indexTokenIdStart;
        mIndexTokenIdEnd = indexTokenIdEnd;
        mTrajFutureStartTokenId = trajFutureStartTokenId;
        mTrajFutureEndTokenId = trajFutureEndTokenId;
        LOG_INFO("Enabled trajectory-gated index-token mask: <i*> token IDs [%d, %d], start=%d, end=%d",
            mIndexTokenIdStart, mIndexTokenIdEnd, mTrajFutureStartTokenId, mTrajFutureEndTokenId);
    }
    else
    {
        LOG_WARNING(
            "Trajectory-gated index-token mask disabled (hasIndexRange=%s, hasStart=%s, hasEnd=%s, vocab=%d).",
            hasIndexTokenRange ? "true" : "false", hasTrajFutureStart ? "true" : "false",
            hasTrajFutureEnd ? "true" : "false", mEngineConfig.outputVocabSize);
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

    // Optional: Setup multimodal engine runner
    if (!multimodalEngineDir.empty())
    {
        try
        {
            mMultimodalRunner = MultimodalRunner::create(
                multimodalEngineDir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize MultimodalRunner: %s", e.what());
            throw std::runtime_error("Failed to initialize MultimodalRunner: " + std::string(e.what()));
        }
        LOG_INFO("MultimodalRunner successfully loaded and initialized multimodal engine.");
    }
}

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request)
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
    mHostReuseKVCacheLengths.reshape({activeBatchSize});
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
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
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
    mHostPackedInputIds.reshape({activeBatchSize, packedInputLength});
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Pad each sequence to the max length of this batch.
        // TODO: Implement remove input padding for better efficiency until multi-batch.
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    mInputIds.reshape({activeBatchSize, packedInputLength});
    mHostContextLengths.reshape({activeBatchSize});
    mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize});

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName, stream))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", loraWeightsName.c_str());
        return false;
    }

    return true;
}

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    mHasLastVlmPostVisionToOutputTiming = false;
    mLastVlmPostVisionToOutputMs = 0.0;

    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;
    std::string loraWeightsName = request.loraWeightsName;
    bool hasVlmPostVisionStart = false;
    std::chrono::steady_clock::time_point vlmPostVisionStart{};

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
            request.addGenerationPrompt, request.enableThinking);

        // Extract system prompt
        batchSystemPrompts.emplace_back(request.formattedRequests[i].formattedSystemPrompt);

        // Save KVCache if requested
        if (request.saveSystemPromptKVCache)
        {
            if (mMultimodalRunner)
            {
                mMultimodalRunner->preprocessSystemPrompt(
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
    if (!mMultimodalRunner)
    {
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
    else
    {
        // Mark multimodal preprocessing and inference for NVTX profiling
        NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);
        if (!mMultimodalRunner->preprocess(
                request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("LLMInferenceRuntime(): Multimodal inference failed. This request cannot be handled.");
            return false;
        }

        // Start VLM-only timing window after vision encoder is done.
        // This captures: LLM token preparation/prefill/generation + output text decode.
        vlmPostVisionStart = std::chrono::steady_clock::now();
        hasVlmPostVisionStart = true;
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
    // 0: natural language / CoT (mask <i*>), 1: trajectory mode (allow <i*>)
    std::vector<int8_t> trajectoryModeFlags(activeBatchSize, 0);
    mSelectedIndices.reshape({activeBatchSize, 1});
    mHostSelectedTokenIds.reshape({activeBatchSize});
    mTrajectoryModeFlags.reshape({activeBatchSize});
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();

    SamplingParams params(
        activeBatchSize, mEngineConfig.outputVocabSize, request.temperature, request.topK, request.topP);
    auto sampleTokens = [&]() -> bool {
        // In natural language / CoT mode, forbid all <i0..> action tokens.
        if (mEnableTrajectoryIndexTokenGating)
        {
            CUDA_CHECK(cudaMemcpyAsync(mTrajectoryModeFlags.rawPointer(), trajectoryModeFlags.data(),
                activeBatchSize * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            trt_edgellm::maskTokenRangeByMode(
                mOutputLogits, mTrajectoryModeFlags, mIndexTokenIdStart, mIndexTokenIdEnd, 1, stream);
        }

        trt_edgellm::topKtopPSamplingFromLogits(mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
        // Apply vocabulary mapping if reduced vocabulary is used
        if (mEngineConfig.reducedVocabSize > 0)
        {
            trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!finishedStates[i])
            {
                int32_t const selectedTokenId = hostSelectedTokenIdsData[i];
                outputIds[i].push_back(selectedTokenId);

                if (mEnableTrajectoryIndexTokenGating)
                {
                    if (selectedTokenId == mTrajFutureStartTokenId)
                    {
                        trajectoryModeFlags[i] = 1;
                    }
                    else if (selectedTokenId == mTrajFutureEndTokenId)
                    {
                        trajectoryModeFlags[i] = 0;
                    }
                }

                finishedStates[i] = selectedTokenId == mTokenizer->getEosId();
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
        return true;
    };

    // Perform embedding lookup for prefill
    int32_t const prefillSequenceLength = mInputIds.getShape()[1];
    mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize});

    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;

    if (multimodalEmbeddings.has_value())
    {
        // Use image insertion variant for multimodal models
        rt::Tensor const& imageEmbedsTensor = multimodalEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding assembly or error if not available
    rt::OptionalInputTensors deepstackEmbeds{};
    if (mEngineConfig.numDeepstackFeatures > 0)
    {
        if (mMultimodalRunner)
        {
            // Multimodal runner exists: perform deepstack embedding assembly
            rt::OptionalInputTensors deepstackFeatures = mMultimodalRunner->getDeepstackFeatures();
            for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
            {
                rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

                // Reshape and perform embedding assembly for this feature
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize});
                kernel::assembleDeepstackEmbedding(
                    mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx], stream);

                // Add to output vector (engine will bind by index)
                deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
            }
        }
        else
        {
            LOG_ERROR(
                "Deepstack features are required (numDeepstackFeatures=%d) but no multimodal runner is available to "
                "provide them.",
                mEngineConfig.numDeepstackFeatures);
            return false;
        }
    }

    // Profile all sampling operations as one stage
    // Prefill profiling session
    // For non-spec decode, we don't need to output hidden states.
    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);
        // Enhanced NVTX range with detailed information
        NVTX_SCOPED_RANGE(nvtx_prefill,
            ("LLM_PREFILL[BS=" + std::to_string(activeBatchSize)
                + ",Reused=" + std::to_string(tokenCount.totalReusedTokens)
                + ",Computed=" + std::to_string(tokenCount.totalComputedTokens) + "]")
                .c_str(),
            nvtx_colors::BLUE);

        bool prefillStatus = mLLMEngineRunner->executePrefillStep(
            mInputsEmbeds, mHostContextLengths, deepstackEmbeds, mOutputLogits, outputHiddenStates, stream);
        if (!prefillStatus)
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Failed to execute prefill step. Cannot generate the KVCache for this prompt.");
            return false;
        }
        if (!sampleTokens())
        {
            LOG_ERROR("LLMInferenceRuntime(): Failed to sample tokens during prefill.");
            return false;
        }
    }

    // Record prefill metrics
    mPrefillMetrics.recordRun(tokenCount.totalReusedTokens, tokenCount.totalComputedTokens);

    // Reshape for decoding step
    mInputsEmbeds.reshape({activeBatchSize, 1, mEngineConfig.hiddenSize});

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
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(mInputsEmbeds, mOutputLogits, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            if (!sampleTokens())
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to sample tokens during generation.");
                return false;
            }
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

    if (hasVlmPostVisionStart)
    {
        auto const vlmPostVisionEnd = std::chrono::steady_clock::now();
        mLastVlmPostVisionToOutputMs
            = std::chrono::duration<double, std::milli>(vlmPostVisionEnd - vlmPostVisionStart).count();
        mHasLastVlmPostVisionToOutputTiming = true;
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
        mSelectedIndices.reshape({batchSize, 1});
        mInputsEmbeds.reshape({batchSize, 1, mEngineConfig.hiddenSize});
        mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize});

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
    return mMultimodalRunner ? mMultimodalRunner->getPositionIds() : std::nullopt;
}

rt::OptionalInputTensor LLMInferenceRuntime::getRopeDeltas() const
{
    return mMultimodalRunner ? mMultimodalRunner->getRopeDeltas() : std::nullopt;
}

LLMInferenceRuntime::TokenCountInfo LLMInferenceRuntime::calculateTokenCounts(
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<std::string> const& systemPrompts,
    std::string const& loraWeightsName) const
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
    mInputsEmbeds.reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize});

    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;

    if (multimodalEmbeddings.has_value())
    {
        // Use image insertion variant for multimodal models
        rt::Tensor const& imageEmbedsTensor = multimodalEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mInputIds, mEmbeddingTable, imageEmbedsTensor, mInputsEmbeds, stream);
    }
    else
    {
        // Standard embedding lookup
        kernel::embeddingLookup(mInputIds, mEmbeddingTable, mInputsEmbeds, stream);
    }

    // Process deepstack features: perform embedding lookup or provide zero tensors
    rt::OptionalInputTensors deepstackEmbeds{};
    if (mEngineConfig.numDeepstackFeatures > 0)
    {
        if (mMultimodalRunner)
        {
            // Multimodal runner exists: perform deepstack embedding lookup
            rt::OptionalInputTensors deepstackFeatures = mMultimodalRunner->getDeepstackFeatures();
            for (int32_t idx = 0; idx < static_cast<int32_t>(deepstackFeatures.size()); ++idx)
            {
                rt::Tensor const& featureTensor = deepstackFeatures[idx].get();

                // Reshape and perform embedding lookup for this feature
                mDeepstackEmbeds[idx].reshape({activeBatchSize, prefillSequenceLength, mEngineConfig.hiddenSize});
                kernel::assembleDeepstackEmbedding(
                    mInputIds, featureTensor, mEngineConfig.vocabSize, mDeepstackEmbeds[idx], stream);

                // Add to output vector (engine will bind by index)
                deepstackEmbeds.push_back(std::ref(mDeepstackEmbeds[idx]));
            }
        }
        else
        {
            LOG_ERROR(
                "Deepstack features are required (numDeepstackFeatures=%d) but no multimodal runner is available to "
                "provide them.",
                mEngineConfig.numDeepstackFeatures);
            return false;
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
        cacheConfig.numDecoderLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU,
        linearKVCache.getConfig().kvCacheTypeTRT, "LLMInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);
    mSystemPromptKVCache.insert({promptKey, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

} // namespace rt
} // namespace trt_edgellm
