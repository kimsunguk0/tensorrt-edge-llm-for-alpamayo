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

#include "runtime/eagleDraftEngineRunner.h"

#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/version.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{
std::string formatEngineConfig(trt_edgellm::rt::EagleDraftEngineRunnerConfig const& config)
{
    std::stringstream ss;
    ss << std::boolalpha;
    ss << "EagleDraftEngineRunnerConfig:"
       << "  numDecoderLayers: " << config.numDecoderLayers << "  numKVHeads: " << config.numKVHeads
       << "  headDim: " << config.headDim << "  rotaryDim: " << config.rotaryDim
       << "  maxSupportedInputLength: " << config.maxSupportedInputLength
       << "  maxKVCacheCapacity: " << config.maxKVCacheCapacity
       << "  draftModelVocabSize: " << config.draftModelVocabSize << "  maxDraftTreeSize: " << config.maxDraftTreeSize
       << "  baseModelHiddenDim: " << config.baseModelHiddenDim
       << "  draftModelHiddenDim: " << config.draftModelHiddenDim;
    return ss.str();
}

trt_edgellm::rt::EagleDraftEngineRunner::DraftProposalKey draftProposalKey(rt::Tensor const& draftTreeInputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates)
{
    int64_t const activeBatchSize = draftTreeInputsEmbeds.getShape()[0];
    int64_t const paddedDraftTreeSize = draftTreeInputsEmbeds.getShape()[1];
    int64_t const selectTokenSize = outputLogits.getShape()[1];
    uintptr_t const inputsEmbedsAddr = reinterpret_cast<uintptr_t>(draftTreeInputsEmbeds.rawPointer());
    uintptr_t const baseModelHiddenStatesAddr = reinterpret_cast<uintptr_t>(baseModelHiddenStates.rawPointer());
    uintptr_t const draftModelHiddenStatesAddr = reinterpret_cast<uintptr_t>(draftModelHiddenStates.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());
    uintptr_t const outputHiddenStatesAddr = reinterpret_cast<uintptr_t>(outputHiddenStates.rawPointer());
    return std::make_tuple(activeBatchSize, paddedDraftTreeSize, selectTokenSize, inputsEmbedsAddr,
        baseModelHiddenStatesAddr, draftModelHiddenStatesAddr, outputLogitsAddr, outputHiddenStatesAddr);
}

trt_edgellm::rt::EagleDraftEngineRunner::AcceptDecodeTokenKey acceptDecodeTokenKey(
    rt::Tensor const& acceptedTokensEmbeds, rt::Tensor const& baseModelHiddenStates,
    rt::Tensor const& draftModelHiddenStates, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates)
{
    int64_t const activeBatchSize = acceptedTokensEmbeds.getShape()[0];
    int64_t const inputEmbedsLength = acceptedTokensEmbeds.getShape()[1];
    uintptr_t const acceptedTokensEmbedsAddr = reinterpret_cast<uintptr_t>(acceptedTokensEmbeds.rawPointer());
    uintptr_t const baseModelHiddenStatesAddr = reinterpret_cast<uintptr_t>(baseModelHiddenStates.rawPointer());
    uintptr_t const draftModelHiddenStatesAddr = reinterpret_cast<uintptr_t>(draftModelHiddenStates.rawPointer());
    uintptr_t const outputLogitsAddr = reinterpret_cast<uintptr_t>(outputLogits.rawPointer());
    uintptr_t const outputHiddenStatesAddr = reinterpret_cast<uintptr_t>(outputHiddenStates.rawPointer());
    return std::make_tuple(activeBatchSize, inputEmbedsLength, acceptedTokensEmbedsAddr, baseModelHiddenStatesAddr,
        draftModelHiddenStatesAddr, outputLogitsAddr, outputHiddenStatesAddr);
}

} // namespace

namespace trt_edgellm
{
namespace rt
{
static constexpr int32_t kDRAFT_MODEL_PREFILL_PROFILE_INDEX{0};
static constexpr int32_t kDRAFT_MODEL_GENERATION_PROFILE_INDEX{1};

EagleDraftEngineRunner::EagleDraftEngineRunner(
    std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream)
{

    LOG_INFO("Loading eagle draft config file: %s", configPath.string().c_str());

    // Parse and validate configuration from JSON file first to fail fast if config is invalid
    Json configJson;
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file with error: %s", e.what());
        throw std::runtime_error("Failed to parse config file: " + configPath.string());
    }

    if (!this->initializeConfigFromJson(configJson))
    {
        LOG_ERROR("Failed to initialize EagleDraftEngineRunner from config file: %s", configPath.string().c_str());
        throw std::runtime_error(
            "Failed to initialize EagleDraftEngineRunner from config file: " + configPath.string());
    }

    LOG_INFO("Loading eagle draft engine file: %s", enginePath.string().c_str());
    // Load the engine after config loading succeeds
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    if (mmapReader->getData() == nullptr)
    {
        LOG_ERROR("Failed to use MMap to read engine from file path: %s", enginePath.string());
        throw std::runtime_error("Failed to use MMap to read engine from file path: " + enginePath.string());
    }

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    int64_t const execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
    // Allocate device memory for the execution contexts. UINT8 is used to represent raw bytes.
    mExecContextMemory = rt::Tensor({execContextMemoryInBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "EagleDraftEngineRunner::mExecContextMemory");

    // Use single executionContext for both prefill and generation.
    mTRTExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));

    mTRTExecutionContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(), execContextMemoryInBytes);
    LOG_INFO("Allocated a shared device memory of %zu bytes for the prefill and generation contexts.",
        execContextMemoryInBytes);

    if (!this->validateConfigFromEngine())
    {
        LOG_ERROR("Failed to match config file %s with engine file: %s", configPath.string().c_str(),
            enginePath.string().c_str());
        throw std::runtime_error(
            "Failed to match config file " + configPath.string() + " with engine file: " + enginePath.string());
    }

    // Detect KV cache storage dtype from engine bindings.
    std::string const kvBindingName0 = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
    DataType kvCacheType = mEngine->getTensorDataType(kvBindingName0.c_str());

    // Sanity check: ensure KV-cache precision (dtype) is consistent across all layers (and both past/present).
    // We rely on a single dtype when allocating/owning the KV cache buffers.
    auto const checkKVCacheDType = [&](int32_t layerIdx, bool isPast) {
        std::string const kvBindingName = binding_names::formatKVCacheName(layerIdx, isPast);
        DataType const dt = mEngine->getTensorDataType(kvBindingName.c_str());
        if (dt != kvCacheType)
        {
            LOG_ERROR(
                "KV cache dtype mismatch detected. Expected all layers to use the same dtype as '%s' (dtype=%d), but "
                "binding '%s' (layer=%d, %s) has dtype=%d.",
                kvBindingName0.c_str(), static_cast<int32_t>(kvCacheType), kvBindingName.c_str(), layerIdx,
                (isPast ? "past" : "present"), static_cast<int32_t>(dt));
            throw std::runtime_error("KV cache dtype mismatch across layers");
        }
    };

    for (int32_t layerIdx = 0; layerIdx < mConfig.numDecoderLayers; ++layerIdx)
    {
        checkKVCacheDType(layerIdx, /*isPast=*/true);
        checkKVCacheDType(layerIdx, /*isPast=*/false);
    }

    // Instantiate the KVCache instance of the EngineRunner.
    this->mLinearKVCache
        = rt::LinearKVCache(rt::LinearKVCache::CacheConfig{mConfig.numDecoderLayers, mConfig.maxSupportedBatchSize,
                                mConfig.maxKVCacheCapacity, mConfig.numKVHeads, mConfig.headDim, kvCacheType},
            stream);

    // By design for tree attention kernel we use, the tree mask will be packed into in32_t values where each bit
    // represents the relationship between two
    int32_t const packedTreeMaskLen = static_cast<int64_t>(divUp(mConfig.maxDraftTreeSize, 32));
    // Instantiate other GPU memory input that needed by the Engine execution.
    // last_token_ids is 2D [batch_size, num_selected_tokens] to match the frontend model export (commit 106a3623d)
    this->mSelectTokenIndices = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxDraftTreeSize},
        rt::DeviceType::kGPU, DataType::kINT64, "EagleDraftEngineRunner::mSelectTokenIndices");
    this->mSequenceContextLengths = rt::Tensor({mConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
        "EagleDraftEngineRunner::mSequenceContextLengths");
    this->mDraftTreePositionIds = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxDraftTreeSize},
        rt::DeviceType::kGPU, DataType::kINT32, "EagleDraftEngineRunner::mDraftTreePositionIds");
    this->mPackedTreeMask = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxDraftTreeSize, packedTreeMaskLen},
        rt::DeviceType::kGPU, DataType::kINT32, "EagleDraftEngineRunner::mPackedTreeMask");
    this->mAcceptedTokenNums = rt::Tensor({mConfig.maxSupportedBatchSize}, rt::DeviceType::kGPU, DataType::kINT32,
        "EagleDraftEngineRunner::mAcceptedTokenNums");

    // Initialize the dummy tensor for unused input tensors as TensorRT does not support nullptr for binding.
    // Calculate maximum memory requirements across all use cases:
    // 1. Attention mask: {maxSupportedBatchSize, 1, 1}
    // 2. Attention position IDs: {maxSupportedBatchSize, 1}
    // 3. KV cache start index: {maxSupportedBatchSize}
    int64_t maxDummyElements = std::max({
        static_cast<int64_t>(mConfig.maxSupportedBatchSize * 1 * 1), // attention mask
        static_cast<int64_t>(mConfig.maxSupportedBatchSize * 1),     // attention position IDs
        static_cast<int64_t>(mConfig.maxSupportedBatchSize)          // KV cache start index
    });
    this->mDummyTensor
        = rt::Tensor({maxDummyElements}, rt::DeviceType::kGPU, DataType::kHALF, "EagleDraftEngineRunner::mDummyTensor");
    // Initialize dummy tensor memory to zero
    CUDA_CHECK(cudaMemsetAsync(mDummyTensor.rawPointer(), 0, mDummyTensor.getMemoryCapacity(), stream));

    mConfig.ropeConfig = collectRopeConfig(configJson);

    if (mConfig.ropeConfig.type != RopeType::kMRope)
    {
        // For non-MRope (Default Rope): allocate with batch_size=1
        // AttentionPlugin will handle broadcasting via the independent rope_batch_size axis
        LOG_DEBUG("Initialize 1D persistent Rope CosSinCache.");
        this->mPosEncCosSinCache = rt::Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "EagleDraftEngineRunner::mPosEncCosSinCache");
        bool const initRopeStatus
            = initializeRopeCosSinCache(mPosEncCosSinCache, mConfig.ropeConfig, configJson, stream);
        if (!initRopeStatus)
        {
            LOG_ERROR("Failed to initialize persistent Rope CosSinCache.");
            throw std::runtime_error("Failed to initialize persistent Rope CosSinCache.");
        }
    }
    else
    {
        this->mPosEncCosSinCache
            = rt::Tensor({mConfig.maxSupportedBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim},
                rt::DeviceType::kGPU, DataType::kFLOAT, "EagleDraftEngineRunner::mPosEncCosSinCache");
        CUDA_CHECK(cudaMemsetAsync(mPosEncCosSinCache.rawPointer(), 0, mPosEncCosSinCache.getMemoryCapacity(), stream));
    }

    // Multi-batch support: Initialize with max supported batch size from config
    // mPosEncCosSinCache shape will match the max batch size for MRope (multimodal)
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kRopeCosSin, mPosEncCosSinCache.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Initialize kKVCacheStartIndex to dummy tensor for both profiles to avoid "address not set" error
    // when switching optimization profiles. The actual address will be set during runtime execution.
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kKVCacheStartIndex, mDummyTensor.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kKVCacheStartIndex, rt::Coords{0}.getTRTDims());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input tensors.");
        throw std::runtime_error("Failed to bind engine input tensors.");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

EagleDraftEngineRunner::~EagleDraftEngineRunner()
{
    for (auto& [key, graphPair] : mDraftProposalCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
    for (auto& [key, graphPair] : mAcceptDecodeTokenCudaGraphs)
    {
        CUDA_CHECK(cudaGraphDestroy(graphPair.first));
        CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
    }
}

bool EagleDraftEngineRunner::initializeConfigFromJson(Json const& configJson)
{
    try
    {
        // Check model version
        std::string modelVersion = configJson.value(binding_names::kEdgellmVersion, "");
        version::checkVersion(modelVersion);

        // Define required fields for main config
        std::vector<std::string> const requiredConfigFields = {"num_hidden_layers", "num_key_value_heads", "head_dim",
            "hidden_size", "base_model_hidden_size", "draft_vocab_size", "builder_config"};

        // Validate required fields exist in main config
        for (auto const& field : requiredConfigFields)
        {
            if (!configJson.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in config", field.c_str());
                return false;
            }
        }

        auto const& builderConfig = configJson["builder_config"];

        // Define required fields for builder_config
        std::vector<std::string> const requiredBuilderConfigFields
            = {"max_batch_size", "max_input_len", "max_kv_cache_capacity", "eagle_draft", "max_draft_tree_size"};

        // Validate required fields exist in builder_config
        for (auto const& field : requiredBuilderConfigFields)
        {
            if (!builderConfig.contains(field))
            {
                LOG_ERROR("initializeConfigFromJson(): Missing required field '%s' in builder_config", field.c_str());
                return false;
            }
        }

        // Validate this is actually an Eagle draft model
        if (!builderConfig["eagle_draft"].get<bool>())
        {
            LOG_ERROR("initializeConfigFromJson(): Config indicates this is not an Eagle draft model");
            return false;
        }

        // Extract values with proper type checking
        mConfig.numDecoderLayers = configJson["num_hidden_layers"].get<int32_t>();
        mConfig.numKVHeads = configJson["num_key_value_heads"].get<int32_t>();
        mConfig.headDim = configJson["head_dim"].get<int32_t>();
        mConfig.rotaryDim = mConfig.headDim;
        mConfig.draftModelHiddenDim = configJson["hidden_size"].get<int32_t>();
        mConfig.baseModelHiddenDim = configJson["base_model_hidden_size"].get<int32_t>();
        mConfig.draftModelVocabSize = configJson["draft_vocab_size"].get<int32_t>();

        // Extract builder_config values
        mConfig.maxSupportedBatchSize = builderConfig["max_batch_size"].get<int32_t>();
        mConfig.maxSupportedInputLength = builderConfig["max_input_len"].get<int32_t>();
        mConfig.maxKVCacheCapacity = builderConfig["max_kv_cache_capacity"].get<int32_t>();
        mConfig.maxDraftTreeSize = builderConfig["max_draft_tree_size"].get<int32_t>();

        // Validate configuration values - all must be positive except numDecoderLayers (can be 1 for draft)
        if (mConfig.numDecoderLayers < 1)
        {
            LOG_ERROR("initializeConfigFromJson(): Invalid num_decoder_layers: %d (must be >= 1 for draft models)",
                mConfig.numDecoderLayers);
            return false;
        }

        std::vector<std::pair<std::string, int32_t>> positiveFields = {{"num_key_value_heads", mConfig.numKVHeads},
            {"head_dim", mConfig.headDim}, {"base_model_hidden_dim", mConfig.baseModelHiddenDim},
            {"draft_model_hidden_dim", mConfig.draftModelHiddenDim},
            {"draft_model_vocab_size", mConfig.draftModelVocabSize}, {"max_input_len", mConfig.maxSupportedInputLength},
            {"kv_cache_capacity_length", mConfig.maxKVCacheCapacity},
            {"max_draft_tree_size", mConfig.maxDraftTreeSize}};

        for (auto const& [fieldName, value] : positiveFields)
        {
            if (value <= 0)
            {
                LOG_ERROR("initializeConfigFromJson(): Invalid %s: %d (must be positive)", fieldName.c_str(), value);
                return false;
            }
        }

        if (mConfig.maxSupportedInputLength > mConfig.maxKVCacheCapacity)
        {
            LOG_ERROR(
                "initializeConfigFromJson(): Invalid configuration: max_input_len (%d) cannot be greater than "
                "max_kv_cache_capacity (%d)",
                mConfig.maxSupportedInputLength, mConfig.maxKVCacheCapacity);
            return false;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("initializeConfigFromJson(): Unexpected error while parsing config: %s", e.what());
        return false;
    }

    LOG_INFO("initializeConfigFromJson(): Loaded EagleDraftEngineRunner with config: %s",
        formatEngineConfig(mConfig).c_str());
    return true;
}

bool EagleDraftEngineRunner::validateConfigFromEngine()
{
    auto identifyKVCacheBinding = [](std::string const& bindingName, Dims const& tensorDim) {
        return tensorDim.nbDims == 5 && bindingName.find(binding_names::kPastKeyValuesTemplate) != std::string::npos;
    };

    int32_t nbKVCacheInputs{0};
    int32_t numIOBindings = mEngine->getNbIOTensors();

    // Lambda to validate KV cache dimensions against profile shape
    auto validateKVCacheProfile = [&](Dims const& maxKVCacheShape, std::string const& profileName) -> bool {
        if (mConfig.numKVHeads != maxKVCacheShape.d[2])
        {
            LOG_ERROR("numKVHeads is not consistent. From engine %s profile: %d, from config: %d", profileName.c_str(),
                maxKVCacheShape.d[2], mConfig.numKVHeads);
            return false;
        }
        if (mConfig.maxKVCacheCapacity != maxKVCacheShape.d[3])
        {
            LOG_ERROR("maxKVCacheCapacity is not consistent. From engine %s profile max: %d, from config: %d",
                profileName.c_str(), maxKVCacheShape.d[3], mConfig.maxKVCacheCapacity);
            return false;
        }
        if (mConfig.headDim != maxKVCacheShape.d[4])
        {
            LOG_ERROR("headDim is not consistent. From engine %s profile: %d, from config: %d", profileName.c_str(),
                maxKVCacheShape.d[4], mConfig.headDim);
            return false;
        }
        return true;
    };

    for (int32_t i = 0; i < numIOBindings; ++i)
    {
        std::string const bindingName = mEngine->getIOTensorName(i);
        Dims const tensorDim = mEngine->getTensorShape(bindingName.c_str());

        if (identifyKVCacheBinding(bindingName, tensorDim))
        {
            // Get max profile shapes for both prefill and generation profiles
            Dims const maxKVCacheShapePrefill = mEngine->getProfileShape(
                bindingName.c_str(), kDRAFT_MODEL_PREFILL_PROFILE_INDEX, OptProfileSelector::kMAX);
            Dims const maxKVCacheShapeGen = mEngine->getProfileShape(
                bindingName.c_str(), kDRAFT_MODEL_GENERATION_PROFILE_INDEX, OptProfileSelector::kMAX);

            // Validate both profiles
            if (!validateKVCacheProfile(maxKVCacheShapePrefill, "prefill"))
            {
                return false;
            }
            if (!validateKVCacheProfile(maxKVCacheShapeGen, "generation"))
            {
                return false;
            }
            ++nbKVCacheInputs;
        }
    }

    if (nbKVCacheInputs != mConfig.numDecoderLayers)
    {
        LOG_ERROR("numDecoderLayers is not consistent. From engine: %d, from config: %d", nbKVCacheInputs,
            mConfig.numDecoderLayers);
        return false;
    }

    // Validate input shapes from optimization profiles (inputs_embeds is 3D: [batch, seq, hidden])
    Dims const maxInputPrefillShape = mEngine->getProfileShape(
        binding_names::kInputsEmbeds, kDRAFT_MODEL_PREFILL_PROFILE_INDEX, OptProfileSelector::kMAX);
    Dims const maxInputGenShape = mEngine->getProfileShape(
        binding_names::kInputsEmbeds, kDRAFT_MODEL_GENERATION_PROFILE_INDEX, OptProfileSelector::kMAX);

    // Validate and potentially override maxSupportedBatchSize from engine's actual max profile
    int32_t const engineMaxBatchSize = maxInputPrefillShape.d[0];
    if (mConfig.maxSupportedBatchSize != engineMaxBatchSize)
    {
        LOG_ERROR("maxSupportedBatchSize mismatch! Config is %d, engine's max optimization profile is %d.",
            mConfig.maxSupportedBatchSize, engineMaxBatchSize);
        return false;
    }

    if (mConfig.maxSupportedInputLength != maxInputPrefillShape.d[1])
    {
        LOG_ERROR("maxSupportedInputLength is not consistent. From engine: %d, from config: %d",
            maxInputPrefillShape.d[1], mConfig.maxSupportedInputLength);
        return false;
    }
    if (mConfig.maxDraftTreeSize != maxInputGenShape.d[1])
    {
        LOG_ERROR("maxDraftTreeSize is not consistent. From engine: %d, from config: %d", maxInputGenShape.d[1],
            mConfig.maxDraftTreeSize);
        return false;
    }
    if (mConfig.draftModelHiddenDim != maxInputPrefillShape.d[2])
    {
        LOG_ERROR("draftModelHiddenDim is not consistent. From engine inputs_embeds: %d, from config: %d",
            maxInputPrefillShape.d[2], mConfig.draftModelHiddenDim);
        return false;
    }

    // Validate vocab size from the engine.
    // Logits shape is [batch_size, num_selected_tokens, vocab_size] for EAGLE draft model
    Dims const logitsDim = mEngine->getTensorShape(binding_names::kLogits);
    if (mConfig.draftModelVocabSize != logitsDim.d[2])
    {
        LOG_ERROR("draftModelVocabSize is not consistent. From engine: %d, from config: %d", logitsDim.d[2],
            mConfig.draftModelVocabSize);
        return false;
    }

    // Validate rotary dim from the engine.
    Dims const ropeCosSinCacheDim = mEngine->getTensorShape(binding_names::kRopeCosSin);
    if (mConfig.rotaryDim != ropeCosSinCacheDim.d[2])
    {
        LOG_ERROR("rotaryDim is not consistent. From engine: %d, from config: %d", ropeCosSinCacheDim.d[2],
            mConfig.rotaryDim);
        return false;
    }

    return true;
}

rt::EagleDraftEngineRunnerConfig EagleDraftEngineRunner::getDraftEngineConfig() const
{
    return mConfig;
}

rt::Tensor& EagleDraftEngineRunner::getRopeCosSinCacheTensor()
{
    return mPosEncCosSinCache;
}

rt::LinearKVCache& EagleDraftEngineRunner::getLinearKVCache()
{
    return mLinearKVCache;
}

bool EagleDraftEngineRunner::prefillStepInputValidation(rt::Tensor const& inputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor const& contextLengths,
    rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates)
{
    bool const checkInputsGPUTensor = inputsEmbeds.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && contextLengths.getDeviceType() == rt::DeviceType::kCPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. contextLengths should be on CPU, others on GPU.");
        return false;
    }

    bool const isInputTypeValid = inputsEmbeds.getDataType() == DataType::kHALF
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && contextLengths.getDataType() == DataType::kINT32
        && outputLogits.getDataType() == DataType::kFLOAT && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Invalid data type of I/O tensors. InputsEmbeds shall be FLOAT16, contextLengths shall be INT32, "
            "base model hidden states shall be FLOAT16, draft model hidden states shall be FLOAT16, "
            "output logits shall be FLOAT32, output hidden states shall be FLOAT16.");
        return false;
    }

    // Validate batch size consistency and bounds
    int32_t const batchSize = inputsEmbeds.getShape()[0];
    bool const isBatchValid = batchSize > 0 && batchSize <= mConfig.maxSupportedBatchSize
        && baseModelHiddenStates.getShape()[0] == batchSize && draftModelHiddenStates.getShape()[0] == batchSize
        && contextLengths.getShape()[0] == batchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be in range [1, %d] and consistent across "
            "tensors, "
            "current inputsEmbeds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            mConfig.maxSupportedBatchSize, inputsEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    int64_t const inputSequenceLength = inputsEmbeds.getShape()[1];
    bool const sequenceLengthValid = inputSequenceLength <= mConfig.maxSupportedInputLength
        && baseModelHiddenStates.getShape()[1] == inputSequenceLength
        && draftModelHiddenStates.getShape()[1] == inputSequenceLength;
    if (!sequenceLengthValid)
    {
        LOG_ERROR(
            "Invalid sequence length of the input tensors. Sequence length shall be consistent and smaller than max "
            "supported length %d, current inputsEmbeds shape: %s, baseModelHiddenStates shape: %s, "
            "draftModelHiddenStates "
            "shape: %s",
            mConfig.maxSupportedInputLength, inputsEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    bool const isInputHiddenSizeValid = inputsEmbeds.getShape()[2] == mConfig.draftModelHiddenDim
        && baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isInputHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current inputsEmbeds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            inputsEmbeds.getShape().formatString().c_str(), baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    bool const isOutputShapeValid = outputLogits.getShape()[0] == batchSize
        && outputLogits.getShape()[1] == mConfig.draftModelVocabSize && outputHiddenStates.getShape()[0] == batchSize
        && outputHiddenStates.getShape()[1] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [%d, %d], hidden states shape shall be [%d, "
            "%d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            batchSize, mConfig.draftModelVocabSize, batchSize, mConfig.draftModelHiddenDim,
            outputLogits.getShape().formatString().c_str(), outputHiddenStates.getShape().formatString().c_str());
        return false;
    }
    return true;
}

bool EagleDraftEngineRunner::executeEaglePrefillStep(rt::Tensor const& inputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates, rt::Tensor const& contextLengths,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, rt::Tensor const& baseRopeCosSinCache,
    cudaStream_t stream)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mTRTExecutionContext->setOptimizationProfileAsync(kDRAFT_MODEL_PREFILL_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    bool const validateInputStatus = this->prefillStepInputValidation(
        inputsEmbeds, baseModelHiddenStates, draftModelHiddenStates, contextLengths, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Prefill request not performed due to invalid input tensors.");
        return false;
    }

    // Prepare the input for the engine execution.
    int32_t const activeBatchSize = static_cast<int32_t>(inputsEmbeds.getShape()[0]);
    int32_t const inputSequenceLength = static_cast<int32_t>(inputsEmbeds.getShape()[1]);
    constexpr int32_t kCONTEXT_SELECT_TOKEN_LENGTH{1};
    mSequenceContextLengths.reshape({activeBatchSize});
    mSelectTokenIndices.reshape({activeBatchSize, kCONTEXT_SELECT_TOKEN_LENGTH}); // 2D tensor [batch, num_tokens]

    // Copy per-batch context lengths.
    CUDA_CHECK(cudaMemcpyAsync(mSequenceContextLengths.rawPointer(), contextLengths.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::prepareEaglePrefillInputs(mSequenceContextLengths, mSelectTokenIndices, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(inputsEmbeds.rawPointer()));
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kInputsEmbeds, inputsEmbeds.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kBaseModelHiddenStates, const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kBaseModelHiddenStates, baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kDraftModelHiddenStates, const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kDraftModelHiddenStates, draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());

    // Setup the KVCache start index tensor. If all KVCache are empty then we can supply zero tensor to the engine.
    // Otherwise, we shall supply the KVCache lengths tensor to the engine.
    if (!mLinearKVCache.getKVCacheAllEmpty())
    {
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().getShape().getTRTDims());
    }
    else
    {
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kKVCacheStartIndex, mDummyTensor.rawPointer());
        setEngineIOStatus
            &= mTRTExecutionContext->setInputShape(binding_names::kKVCacheStartIndex, rt::Coords{0}.getTRTDims());
    }

    // For MRope (ND-Rope, context-dependent), reshape to match activeBatchSize (per-batch values needed)
    // For non-MRope (Default Rope), keep batch_size=1 (TensorRT broadcasts via independent rope_batch_size axis)
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        // Copy MRoPE cosine/sine cache tensor from the base model
        CUDA_CHECK(cudaMemcpyAsync(mPosEncCosSinCache.rawPointer(), baseRopeCosSinCache.rawPointer(),
            baseRopeCosSinCache.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }

    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Update KV cache shapes to match activeBatchSize (critical for dynamic batching)
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    // attention-pos-id and attention-mask are unused during the execution. We set the dummy tensor with zero
    // shape.
    rt::Coords const emptyPosIdShape{activeBatchSize, 1};
    rt::Coords const emptyMaskShape{activeBatchSize, 1, 1};
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mDummyTensor.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kAttentionPosId, emptyPosIdShape.getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mDummyTensor.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kAttentionMask, emptyMaskShape.getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT prefill stage enqueueV3() call.");
        return false;
    }
    // In prefill step. commit all KVCache generated during the execution.
    mLinearKVCache.commitSequenceLength(mSequenceContextLengths, stream);

    LOG_DEBUG("Prefill stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool EagleDraftEngineRunner::draftProposalStepInputValidation(rt::Tensor const& draftTreeInputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask, rt::Tensor const& outputLogits,
    rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = draftTreeInputsEmbeds.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftTreeLength.getDeviceType() == rt::DeviceType::kGPU
        && draftTreeMask.getDeviceType() == rt::DeviceType::kGPU && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. All inputs and outputs shall reside on GPU.");
        return false;
    }
    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = draftTreeInputsEmbeds.getDataType() == DataType::kHALF
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && draftTreeLength.getDataType() == DataType::kINT32
        && draftTreeMask.getDataType() == DataType::kINT8 && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Input embeds shall be FLOAT16, hidden states I/O shall be FLOAT16, "
            "draft tree length shall be INT32, draft tree mask shall be INT8, output logits shall be FLOAT32.");
        return false;
    }
    // Validate batch size consistency and bounds
    int32_t const batchSize = draftTreeInputsEmbeds.getShape()[0];
    bool const isBatchValid = batchSize > 0 && batchSize <= mConfig.maxSupportedBatchSize
        && baseModelHiddenStates.getShape()[0] == batchSize && draftModelHiddenStates.getShape()[0] == batchSize
        && draftTreeLength.getShape()[0] == batchSize && draftTreeMask.getShape()[0] == batchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be in range [1, %d] and consistent across "
            "tensors, "
            "current draft tree input embeds shape: %s, base model hidden states shape: %s, draft model hidden states "
            "shape: %s, "
            "draft tree length shape: %s, draft tree mask shape: %s",
            mConfig.maxSupportedBatchSize, draftTreeInputsEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str(), draftTreeLength.getShape().formatString().c_str(),
            draftTreeMask.getShape().formatString().c_str());
        return false;
    }

    int64_t const paddedDraftTreeSize = draftTreeInputsEmbeds.getShape()[1];
    bool const isPaddedDraftTreeSizeValid = paddedDraftTreeSize <= mConfig.maxDraftTreeSize
        && draftTreeMask.getShape()[1] == paddedDraftTreeSize && draftTreeMask.getShape()[2] == paddedDraftTreeSize
        && baseModelHiddenStates.getShape()[1] == paddedDraftTreeSize
        && draftModelHiddenStates.getShape()[1] == paddedDraftTreeSize;
    if (!isPaddedDraftTreeSizeValid)
    {
        LOG_ERROR(
            "Invalid padded draft tree size of the input tensors. Padded draft tree size shall be smaller than max "
            "limit %d and be consistent among input tensors, current draft tree mask shape: %s, base model hidden "
            "states shape: %s, draft model hidden states shape: %s",
            mConfig.maxDraftTreeSize, draftTreeMask.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    bool const isHiddenSizeValid = draftTreeInputsEmbeds.getShape()[2] == mConfig.draftModelHiddenDim
        && baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current inputsEmbeds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            draftTreeInputsEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    // Output tensors must be 3D: [batch_size, num_tokens, vocab_size/hidden_dim]
    bool const isOutputShapeValid = outputLogits.getShape()[0] == batchSize
        && outputLogits.getShape()[0] == outputHiddenStates.getShape()[0]
        && outputLogits.getShape()[1] == outputHiddenStates.getShape()[1]
        && outputLogits.getShape()[2] == mConfig.draftModelVocabSize
        && outputHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [%d, num_tokens, %d], hidden states shape "
            "shall be [%d, num_tokens, %d], current outputLogits shape: %s, outputHiddenStates shape: %s",
            batchSize, mConfig.draftModelVocabSize, batchSize, mConfig.draftModelHiddenDim,
            outputLogits.getShape().formatString().c_str(), outputHiddenStates.getShape().formatString().c_str());
        return false;
    }

    return true;
}

bool EagleDraftEngineRunner::executeEagleDraftProposalStep(rt::Tensor const& draftTreeInputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus
        = this->draftProposalStepInputValidation(draftTreeInputsEmbeds, baseModelHiddenStates, draftModelHiddenStates,
            draftTreeLength, draftTreeMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Draft proposal request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(draftTreeInputsEmbeds.getShape()[0]);
    int32_t const paddedDraftTreeSize = static_cast<int32_t>(draftTreeInputsEmbeds.getShape()[1]);
    // outputLogits is now 3D: [batch_size, num_tokens, vocab_size]
    int32_t const selectTokenSize = static_cast<int32_t>(outputLogits.getShape()[1]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(paddedDraftTreeSize, 32));

    // Prepare extra input for engine execution. Assemble packed tree mask, position indices, select token indices,
    // sequence context lengths.
    mSelectTokenIndices.reshape({activeBatchSize, selectTokenSize}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mDraftTreePositionIds.reshape({activeBatchSize, paddedDraftTreeSize});
    mPackedTreeMask.reshape({activeBatchSize, paddedDraftTreeSize, packedTreeMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();
    kernel::prepareEagleDraftProposalInputs(draftTreeMask, draftTreeLength, sequenceStartIndex, mPackedTreeMask,
        mDraftTreePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    auto const key = draftProposalKey(
        draftTreeInputsEmbeds, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (mDraftProposalCudaGraphs.find(key) != mDraftProposalCudaGraphs.end())
    {
        LOG_DEBUG("executeEagleDraftProposalStep(): Use pre-captured CUDA graph for draft proposal step.");
        cudaGraphExec_t graphExec = mDraftProposalCudaGraphs[key].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setOptimizationProfileStatus{true};
        setOptimizationProfileStatus
            &= mTRTExecutionContext->setOptimizationProfileAsync(kDRAFT_MODEL_GENERATION_PROFILE_INDEX, stream);
        if (!setOptimizationProfileStatus)
        {
            LOG_ERROR("Failed to set optimization profile to the engine");
            throw std::runtime_error("Failed to set optimization profile to the engine");
        }

        LOG_INFO("executeEagleDraftProposalStep(): Draft proposal step CUDA graph not captured.");

        // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
        // initialization.
        bool setEngineIOStatus{true};
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kInputsEmbeds, const_cast<void*>(draftTreeInputsEmbeds.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kInputsEmbeds, draftTreeInputsEmbeds.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kBaseModelHiddenStates, const_cast<void*>(baseModelHiddenStates.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kBaseModelHiddenStates, baseModelHiddenStates.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kDraftModelHiddenStates, const_cast<void*>(draftModelHiddenStates.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kDraftModelHiddenStates, draftModelHiddenStates.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().getShape().getTRTDims());

        // For MRope (ND-Rope, context-dependent), reshape to match activeBatchSize (per-batch values needed)
        // For non-MRope (Default Rope), keep batch_size=1 (TensorRT broadcasts via independent rope_batch_size axis)
        if (mConfig.ropeConfig.type == RopeType::kMRope)
        {
            mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
        }

        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

        // Update KV cache shapes to match activeBatchSize for generation context
        setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

        // Differs from prefill step, draft proposal step needs to take real packed mask and position indices.
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mPackedTreeMask.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionMask, mPackedTreeMask.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kAttentionPosId, mDraftTreePositionIds.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionPosId, mDraftTreePositionIds.getShape().getTRTDims());

        // Bind the output tensor into the engine.
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("Failed to bind engine input and output tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mTRTExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("Failed on TensorRT draft proposal stage enqueueV3() call.");
            return false;
        }
    }

    // Note in the draft token proposal step we explicitly don't commit the KVCache since we process the "whole tree" in
    // these steps.
    return true;
}

bool EagleDraftEngineRunner::captureEagleDraftProposalCudaGraph(rt::Tensor const& draftTreeInputsEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mTRTExecutionContext->setOptimizationProfileAsync(kDRAFT_MODEL_GENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    auto const key = draftProposalKey(
        draftTreeInputsEmbeds, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (mDraftProposalCudaGraphs.find(key) != mDraftProposalCudaGraphs.end())
    {
        LOG_INFO("Draft proposal CUDA graph already captured.");
        return true;
    }

    // Here we will simulate the state of the EngineRunner after executing one prefill request for a batched request.
    int32_t const activeBatchSize = draftTreeInputsEmbeds.getShape()[0];
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU,
        DataType::kINT32, "draft_reuse_kv_cache_lengths");

    mLinearKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    // Validate the input tensors.
    bool const validateInputStatus
        = this->draftProposalStepInputValidation(draftTreeInputsEmbeds, baseModelHiddenStates, draftModelHiddenStates,
            draftTreeLength, draftTreeMask, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Draft proposal request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const paddedDraftTreeSize = static_cast<int32_t>(draftTreeInputsEmbeds.getShape()[1]);
    // outputLogits is now 3D: [batch_size, num_tokens, vocab_size]
    int32_t const selectTokenSize = static_cast<int32_t>(outputLogits.getShape()[1]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(paddedDraftTreeSize, 32));

    // Prepare extra input for engine execution. Assemble packed tree mask, position indices, select token indices,
    // sequence context lengths.
    mSelectTokenIndices.reshape({activeBatchSize, selectTokenSize}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mDraftTreePositionIds.reshape({activeBatchSize, paddedDraftTreeSize});
    mPackedTreeMask.reshape({activeBatchSize, paddedDraftTreeSize, packedTreeMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();
    kernel::prepareEagleDraftProposalInputs(draftTreeMask, draftTreeLength, sequenceStartIndex, mPackedTreeMask,
        mDraftTreePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
    // initialization.
    bool setEngineIOStatus{true};

    // Update KV cache shapes to match activeBatchSize for CUDA graph capture
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(draftTreeInputsEmbeds.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kInputsEmbeds, draftTreeInputsEmbeds.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kBaseModelHiddenStates, const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kBaseModelHiddenStates, baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kDraftModelHiddenStates, const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kDraftModelHiddenStates, draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().getShape().getTRTDims());

    // For MRope (ND-Rope, context-dependent), reshape to match activeBatchSize (per-batch values needed)
    // For non-MRope (Default Rope), keep batch_size=1 (TensorRT broadcasts via independent rope_batch_size axis)
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }

    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    // Differs from prefill step, draft proposal step needs to take real packed mask and position indices.
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mPackedTreeMask.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kAttentionMask, mPackedTreeMask.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mDraftTreePositionIds.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kAttentionPosId, mDraftTreePositionIds.getShape().getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution. This will trigger the shape machine of TensorRT engine to avoid cudaGraph capture.
    // error.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT draft proposal stage enqueueV3() call.");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    mDraftProposalCudaGraphs[key] = std::make_pair(graph, graphExec);

    if (!executeStatus)
    {
        LOG_WARNING(
            "captureEagleDraftProposalCudaGraph(): Failed on TensorRT engine enqueueV3() call during CUDA graph "
            "capture.");
        return false;
    }
    else
    {
        LOG_DEBUG("captureEagleDraftProposalCudaGraph(): CUDA graph captured successfully for input shape %s.",
            draftTreeInputsEmbeds.getShape().formatString().c_str());
    }

    return true;
}

bool EagleDraftEngineRunner::acceptDecodeTokenStepInputValidation(rt::Tensor const& acceptedTokensEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& acceptedTokenNums, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates)
{
    // All input tensors shall reside on GPU.
    bool const checkInputsGPUTensor = acceptedTokensEmbeds.getDeviceType() == rt::DeviceType::kGPU
        && baseModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && draftModelHiddenStates.getDeviceType() == rt::DeviceType::kGPU
        && outputLogits.getDeviceType() == rt::DeviceType::kGPU
        && outputHiddenStates.getDeviceType() == rt::DeviceType::kGPU;
    if (!checkInputsGPUTensor)
    {
        LOG_ERROR("Invalid device type of I/O tensors. All inputs and outputs shall reside on GPU.");
        return false;
    }

    // Validate datatypes of the input tensors.
    bool const isInputTypeValid = acceptedTokensEmbeds.getDataType() == DataType::kHALF
        && baseModelHiddenStates.getDataType() == DataType::kHALF
        && draftModelHiddenStates.getDataType() == DataType::kHALF && outputLogits.getDataType() == DataType::kFLOAT
        && outputHiddenStates.getDataType() == DataType::kHALF;
    if (!isInputTypeValid)
    {
        LOG_ERROR(
            "Invalid data type of I/O tensors. Accepted tokens embeds shall be FLOAT16, hidden states I/O shall be "
            "FLOAT16, "
            "output logits shall be FLOAT32, output hidden states shall be FLOAT16.");
        return false;
    }

    // Validate batch size consistency and bounds
    int32_t const batchSize = acceptedTokensEmbeds.getShape()[0];
    bool const isBatchValid = batchSize > 0 && batchSize <= mConfig.maxSupportedBatchSize
        && baseModelHiddenStates.getShape()[0] == batchSize && draftModelHiddenStates.getShape()[0] == batchSize;
    if (!isBatchValid)
    {
        LOG_ERROR(
            "Invalid batch size of the input tensors. Batch size shall be in range [1, %d] and consistent across "
            "tensors, "
            "current accepted tokens embeds shape: %s, base model hidden states shape: %s, draft model hidden states "
            "shape: "
            "%s",
            mConfig.maxSupportedBatchSize, acceptedTokensEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    // Validate acceptedTokenNums tensor
    bool const isAcceptedTokenNumsValid = acceptedTokenNums.getDeviceType() == rt::DeviceType::kGPU
        && acceptedTokenNums.getDataType() == DataType::kINT32 && acceptedTokenNums.getShape()[0] == batchSize;
    if (!isAcceptedTokenNumsValid)
    {
        LOG_ERROR("Invalid acceptedTokenNums tensor. Must be GPU INT32 with shape [%d], got shape: %s", batchSize,
            acceptedTokenNums.getShape().formatString().c_str());
        return false;
    }

    int64_t const acceptedTokenNum = acceptedTokensEmbeds.getShape()[1];
    bool const isAcceptedTokenNumValid = acceptedTokenNum <= mConfig.maxDraftTreeSize
        && baseModelHiddenStates.getShape()[1] == acceptedTokenNum
        && draftModelHiddenStates.getShape()[1] == acceptedTokenNum;
    if (!isAcceptedTokenNumValid)
    {
        LOG_ERROR(
            "Invalid accepted token number of the input tensors. Accepted token number shall be smaller than max limit "
            "%d, And be consistent among input tensors, current accepted tokens embeds shape: %s, base model hidden "
            "states "
            "shape: %s, draft model hidden states shape: %s",
            mConfig.maxDraftTreeSize, acceptedTokensEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    bool const isHiddenSizeValid = acceptedTokensEmbeds.getShape()[2] == mConfig.draftModelHiddenDim
        && baseModelHiddenStates.getShape()[2] == mConfig.baseModelHiddenDim
        && draftModelHiddenStates.getShape()[2] == mConfig.draftModelHiddenDim;
    if (!isHiddenSizeValid)
    {
        LOG_ERROR(
            "Invalid hidden size of the input tensors. Hidden size shall be consistent with the model config, "
            "current acceptedTokensEmbeds shape: %s, baseModelHiddenStates shape: %s, draftModelHiddenStates shape: %s",
            acceptedTokensEmbeds.getShape().formatString().c_str(),
            baseModelHiddenStates.getShape().formatString().c_str(),
            draftModelHiddenStates.getShape().formatString().c_str());
        return false;
    }

    // When accept committed tokens, we only need to collect the logits and hidden states from the "last" token.
    bool const isOutputShapeValid = outputLogits.getShape()[0] == batchSize
        && outputHiddenStates.getShape()[0] == batchSize && outputLogits.getShape()[1] == mConfig.draftModelVocabSize
        && outputHiddenStates.getShape()[1] == mConfig.draftModelHiddenDim;
    if (!isOutputShapeValid)
    {
        LOG_ERROR(
            "Invalid shape of the output tensors. Logits shape shall be [%d, %d], hidden states shape shall be [%d, "
            "%d], "
            "current outputLogits shape: %s, outputHiddenStates shape: %s",
            batchSize, mConfig.draftModelVocabSize, batchSize, mConfig.draftModelHiddenDim,
            outputLogits.getShape().formatString().c_str(), outputHiddenStates.getShape().formatString().c_str());
        return false;
    }

    return true;
}

bool EagleDraftEngineRunner::executeEagleAcceptDecodeTokenStep(rt::Tensor const& acceptedTokensEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& acceptedTokenNums, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool const validateInputStatus = this->acceptDecodeTokenStepInputValidation(acceptedTokensEmbeds,
        baseModelHiddenStates, draftModelHiddenStates, acceptedTokenNums, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Accept decode token request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(acceptedTokensEmbeds.getShape()[0]);
    int32_t const acceptedTokenNum = static_cast<int32_t>(acceptedTokensEmbeds.getShape()[1]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(acceptedTokenNum, 32));
    constexpr int32_t kACCEPT_DECODE_SELECT_TOKEN_LENGTH{1};

    // Prepare extra input for engine execution. Assemble packed tree mask, position indices, select token indices,
    // sequence context lengths.
    mSelectTokenIndices.reshape({activeBatchSize, kACCEPT_DECODE_SELECT_TOKEN_LENGTH}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mDraftTreePositionIds.reshape({activeBatchSize, acceptedTokenNum});
    mPackedTreeMask.reshape({activeBatchSize, acceptedTokenNum, packedTreeMaskLen});
    // We can obtain the sequence start index from KVCache, the current KVCache size denote the start index of the "next
    // token" in the sequence.
    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();

    // Use the provided acceptedTokenNums directly (already on GPU from base model verification)
    // This contains per-batch actual accept counts, NOT the padded length
    kernel::prepareEagleAcceptDecodeTokenInputs(sequenceStartIndex, acceptedTokenNums, mPackedTreeMask,
        mDraftTreePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    auto const key = acceptDecodeTokenKey(
        acceptedTokensEmbeds, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (mAcceptDecodeTokenCudaGraphs.find(key) != mAcceptDecodeTokenCudaGraphs.end())
    {
        LOG_DEBUG(
            "executeEagleAcceptDecodeTokenStep(): Use pre-captured CUDA graph for draft accept decode token step.");
        cudaGraphExec_t graphExec = mAcceptDecodeTokenCudaGraphs[key].second;
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    else
    {
        bool setOptimizationProfileStatus{true};
        setOptimizationProfileStatus
            &= mTRTExecutionContext->setOptimizationProfileAsync(kDRAFT_MODEL_GENERATION_PROFILE_INDEX, stream);
        if (!setOptimizationProfileStatus)
        {
            LOG_ERROR("Failed to set optimization profile to the engine");
            throw std::runtime_error("Failed to set optimization profile to the engine");
        }

        LOG_INFO("executeEagleAcceptDecodeTokenStep(): Draft accept decode token step CUDA graph not captured.");

        // Bind the input and output tensor into the engine. RopeCosSinCache and KVCache are pre-bind during runner
        // initialization.
        bool setEngineIOStatus{true};
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kInputsEmbeds, const_cast<void*>(acceptedTokensEmbeds.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kInputsEmbeds, acceptedTokensEmbeds.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kBaseModelHiddenStates, const_cast<void*>(baseModelHiddenStates.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kBaseModelHiddenStates, baseModelHiddenStates.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kDraftModelHiddenStates, const_cast<void*>(draftModelHiddenStates.rawPointer()));
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kDraftModelHiddenStates, draftModelHiddenStates.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().getShape().getTRTDims());

        // For MRope (ND-Rope, context-dependent), reshape to match activeBatchSize (per-batch values needed)
        // For non-MRope (Default Rope), keep batch_size=1 (TensorRT broadcasts via independent rope_batch_size axis)
        if (mConfig.ropeConfig.type == RopeType::kMRope)
        {
            mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
        }

        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

        // Update KV cache shapes to match activeBatchSize for generation context
        setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

        // Differs from prefill step, draft proposal step needs to take real packed mask and position indices.
        setEngineIOStatus
            &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mPackedTreeMask.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionMask, mPackedTreeMask.getShape().getTRTDims());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kAttentionPosId, mDraftTreePositionIds.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setInputShape(
            binding_names::kAttentionPosId, mDraftTreePositionIds.getShape().getTRTDims());

        // Bind the output tensor into the engine.
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
        setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
            binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

        if (!setEngineIOStatus)
        {
            LOG_ERROR("Failed to bind engine input and output tensors.");
            return false;
        }

        // launch the engine execution.
        bool executeStatus{true};
        executeStatus &= mTRTExecutionContext->enqueueV3(stream);
        if (!executeStatus)
        {
            LOG_ERROR("Failed on TensorRT accept decode token stage enqueueV3() call.");
            return false;
        }
    }

    // Commit the KVCache for accepted tokens.
    mLinearKVCache.commitSequenceLength(acceptedTokenNums, stream);

    LOG_DEBUG("Accept decode token stage execution completed for request with batch size %d.", activeBatchSize);
    return true;
}

bool EagleDraftEngineRunner::captureEagleAcceptDecodeTokenCudaGraph(rt::Tensor const& acceptedTokensEmbeds,
    rt::Tensor const& baseModelHiddenStates, rt::Tensor const& draftModelHiddenStates,
    rt::Tensor const& acceptedTokenNums, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    bool setOptimizationProfileStatus{true};
    setOptimizationProfileStatus
        &= mTRTExecutionContext->setOptimizationProfileAsync(kDRAFT_MODEL_GENERATION_PROFILE_INDEX, stream);
    if (!setOptimizationProfileStatus)
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }

    auto const key = acceptDecodeTokenKey(
        acceptedTokensEmbeds, baseModelHiddenStates, draftModelHiddenStates, outputLogits, outputHiddenStates);
    if (mAcceptDecodeTokenCudaGraphs.find(key) != mAcceptDecodeTokenCudaGraphs.end())
    {
        LOG_INFO("Draft accept decode token CUDA graph already captured.");
        return true;
    }
    bool const validateInputStatus = this->acceptDecodeTokenStepInputValidation(acceptedTokensEmbeds,
        baseModelHiddenStates, draftModelHiddenStates, acceptedTokenNums, outputLogits, outputHiddenStates);
    if (!validateInputStatus)
    {
        LOG_ERROR("Accept decode token request not performed due to invalid input tensors.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(acceptedTokensEmbeds.getShape()[0]);
    int32_t const acceptedTokenNum = static_cast<int32_t>(acceptedTokensEmbeds.getShape()[1]);
    int32_t const packedTreeMaskLen = static_cast<int32_t>(divUp(acceptedTokenNum, 32));
    constexpr int32_t kACCEPT_DECODE_SELECT_TOKEN_LENGTH{1};

    // Reset KV cache for the activeBatchSize to ensure sequenceStartIndex has correct batch dimension
    constexpr int32_t simulateCacheLength{128};
    std::vector<int32_t> reuseKVCacheLengths(activeBatchSize, simulateCacheLength);
    rt::Tensor const reuseKVCacheLengthsTensor(
        reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
    mLinearKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

    mSelectTokenIndices.reshape({activeBatchSize, kACCEPT_DECODE_SELECT_TOKEN_LENGTH}); // 2D tensor [batch, num_tokens]
    mSequenceContextLengths.reshape({activeBatchSize});
    mDraftTreePositionIds.reshape({activeBatchSize, acceptedTokenNum});
    mPackedTreeMask.reshape({activeBatchSize, acceptedTokenNum, packedTreeMaskLen});

    rt::Tensor const& sequenceStartIndex = mLinearKVCache.getKVCacheLengths();

    // Use the provided acceptedTokenNums directly (already on GPU from base model verification)
    // This contains per-batch actual accept counts, NOT the padded length
    kernel::prepareEagleAcceptDecodeTokenInputs(sequenceStartIndex, acceptedTokenNums, mPackedTreeMask,
        mDraftTreePositionIds, mSelectTokenIndices, mSequenceContextLengths, stream);

    bool setEngineIOStatus{true};

    // Update KV cache shapes to match activeBatchSize for CUDA graph capture
    setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kInputsEmbeds, const_cast<void*>(acceptedTokensEmbeds.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kInputsEmbeds, acceptedTokensEmbeds.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kBaseModelHiddenStates, const_cast<void*>(baseModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kBaseModelHiddenStates, baseModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kDraftModelHiddenStates, const_cast<void*>(draftModelHiddenStates.rawPointer()));
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kDraftModelHiddenStates, draftModelHiddenStates.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kContextLengths, mSequenceContextLengths.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kLastTokenIds, mSelectTokenIndices.getShape().getTRTDims());
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kKVCacheStartIndex, mLinearKVCache.getKVCacheLengths().getShape().getTRTDims());

    // For MRope (ND-Rope, context-dependent), reshape to match activeBatchSize (per-batch values needed)
    // For non-MRope (Default Rope), keep batch_size=1 (TensorRT broadcasts via independent rope_batch_size axis)
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        mPosEncCosSinCache.reshape({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
    }

    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());

    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionMask, mPackedTreeMask.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setInputShape(binding_names::kAttentionMask, mPackedTreeMask.getShape().getTRTDims());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kAttentionPosId, mDraftTreePositionIds.rawPointer());
    setEngineIOStatus &= mTRTExecutionContext->setInputShape(
        binding_names::kAttentionPosId, mDraftTreePositionIds.getShape().getTRTDims());

    // Bind the output tensor into the engine.
    setEngineIOStatus &= mTRTExecutionContext->setTensorAddress(binding_names::kLogits, outputLogits.rawPointer());
    setEngineIOStatus
        &= mTRTExecutionContext->setTensorAddress(binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind engine input and output tensors.");
        return false;
    }

    // launch the engine execution. This will trigger the shape machine of TensorRT engine to avoid cudaGraph capture.
    bool executeStatus{true};
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    if (!executeStatus)
    {
        LOG_ERROR("Failed on TensorRT accept decode token stage enqueueV3() call.");
        return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    executeStatus &= mTRTExecutionContext->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    mAcceptDecodeTokenCudaGraphs[key] = std::make_pair(graph, graphExec);

    if (!executeStatus)
    {
        LOG_WARNING(
            "captureEagleAcceptDecodeTokenCudaGraph(): Failed on TensorRT engine enqueueV3() call during CUDA graph "
            "capture.");
        return false;
    }
    else
    {
        LOG_DEBUG("captureEagleAcceptDecodeTokenCudaGraph(): CUDA graph captured successfully for input shape %s.",
            acceptedTokensEmbeds.getShape().formatString().c_str());
    }

    return true;
}

bool EagleDraftEngineRunner::bindKVCacheToEngine(int32_t activeBatchSize)
{
    // Prepare special input binding shape for prefill stage KVCache input.
    Dims const kvCacheDims = {5, {activeBatchSize, 2, mConfig.numKVHeads, mConfig.maxKVCacheCapacity, mConfig.headDim}};
    bool status{true};
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        std::string const pastKeyValuesName = binding_names::formatKVCacheName(i, true);
        std::string const presentKeyValuesName = binding_names::formatKVCacheName(i, false);

        rt::Tensor kvCacheBlock = mLinearKVCache.getKVCacheForDecoderLayer(i);
        status &= mTRTExecutionContext->setTensorAddress(pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mTRTExecutionContext->setTensorAddress(presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
        status &= mTRTExecutionContext->setInputShape(pastKeyValuesName.c_str(), kvCacheDims);
    }
    return status;
}

} // namespace rt
} // namespace trt_edgellm
