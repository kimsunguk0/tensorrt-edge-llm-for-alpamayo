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

#include "qwenViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/mathUtils.h"
#include "common/mmapReader.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "profiling/timer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr char const* kFlexInputName = "final_visual";
constexpr char const* kFlexOutputName = "scene_embeds";
constexpr std::array<char const*, 3> kFlexDeepstackInputNames = {"ds_level0", "ds_level1", "ds_level2"};
constexpr std::array<char const*, 3> kFlexDeepstackOutputNames
    = {"deepstack_scene_0", "deepstack_scene_1", "deepstack_scene_2"};
constexpr char const* kFlexCameraIdsName = "camera_ids";
constexpr char const* kFlexRelativeTimesName = "relative_times";
constexpr std::array<int64_t, 4> kFlexAlpamayoCameraIds = {0, 1, 2, 6};
constexpr std::array<float, 4> kFlexAlpamayoRelativeTimes = {-0.3F, -0.2F, -0.1F, 0.0F};
constexpr int64_t kFlexAlpamayoFramesPerCamera = 4;
constexpr int64_t kFlexAlpamayoImageCount
    = static_cast<int64_t>(kFlexAlpamayoCameraIds.size()) * kFlexAlpamayoFramesPerCamera;

bool isDebugQwenTextPreprocessEnabled()
{
    char const* env = std::getenv("EDGELLM_DEBUG_QWEN_TEXTPRE");
    return env != nullptr && std::strcmp(env, "0") != 0;
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

bool writeBinaryFile(std::filesystem::path const& path, void const* data, size_t size)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        return false;
    }
    ofs.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
    return ofs.good();
}

bool hasIOTensor(nvinfer1::ICudaEngine const& engine, char const* name)
{
    for (int32_t idx = 0; idx < engine.getNbIOTensors(); ++idx)
    {
        char const* tensorName = engine.getIOTensorName(idx);
        if (tensorName != nullptr && std::strcmp(tensorName, name) == 0)
        {
            return true;
        }
    }
    return false;
}

bool hasAllIOTensors(nvinfer1::ICudaEngine const& engine, std::array<char const*, 3> const& names)
{
    return std::all_of(names.begin(), names.end(), [&engine](char const* name) { return hasIOTensor(engine, name); });
}

bool checkTensorDataType(nvinfer1::ICudaEngine const& engine, char const* name, nvinfer1::DataType expected)
{
    nvinfer1::DataType const actual = engine.getTensorDataType(name);
    if (actual != expected)
    {
        LOG_ERROR("QwenViTRunner: FLEX tensor %s has dtype %s, expected %s.", name, getDataTypeString(actual).c_str(),
            getDataTypeString(expected).c_str());
        return false;
    }
    return true;
}
} // namespace

QwenViTRunner::QwenViTRunner(
    std::string const& engineDir, int32_t llmMaxBatchSize, int32_t llmMaxSequenceLength, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
    , mLLMMaxBatchSize(llmMaxBatchSize)
    , mLLMMaxSequenceLength(llmMaxSequenceLength)
{
    if (!validateAndFillConfig(engineDir))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to validate and fill config");
    }
    if (!allocateBuffer(stream))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to allocate buffer");
    }
    if (!tryLoadFlexEngine(engineDir, stream))
    {
        LOG_ERROR("QwenViTRunner::QwenViTRunner(): Failed to load optional FLEX engine");
        throw std::runtime_error("QwenViTRunner::QwenViTRunner(): Failed to load optional FLEX engine");
    }
}

bool QwenViTRunner::tryLoadFlexEngine(std::string const& engineDir, cudaStream_t stream)
{
    namespace fs = std::filesystem;

    fs::path const flexEnginePath = fs::path(engineDir).parent_path() / "flex" / "flex.engine";
    std::error_code ec;
    if (!fs::exists(flexEnginePath, ec))
    {
        return true;
    }

    try
    {
        mFlexRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        auto mmapReader = std::make_unique<file_io::MmapReader>(flexEnginePath);
        mFlexEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            mFlexRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
        if (!mFlexEngine)
        {
            LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): Failed to deserialize FLEX engine: %s",
                flexEnginePath.c_str());
            return false;
        }

        mFlexContext = std::unique_ptr<nvinfer1::IExecutionContext>(mFlexEngine->createExecutionContext());
        if (!mFlexContext)
        {
            LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): Failed to create FLEX execution context.");
            return false;
        }
        if (!mFlexContext->setOptimizationProfileAsync(0, stream))
        {
            LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): Failed to set FLEX optimization profile.");
            return false;
        }

        mFlexHasDeepstackInputs = hasAllIOTensors(*mFlexEngine, kFlexDeepstackInputNames);
        mFlexHasDeepstackOutputs = hasAllIOTensors(*mFlexEngine, kFlexDeepstackOutputNames);
        mFlexHasMetadataInputs
            = hasIOTensor(*mFlexEngine, kFlexCameraIdsName) && hasIOTensor(*mFlexEngine, kFlexRelativeTimesName);

        if (mFlexHasDeepstackInputs != mFlexHasDeepstackOutputs)
        {
            LOG_ERROR(
                "QwenViTRunner::tryLoadFlexEngine(): FLEX engine must have both deepstack inputs and deepstack outputs.");
            return false;
        }
        if (mFlexHasDeepstackInputs && static_cast<int64_t>(mDeepstackFeatures.size()) < 3)
        {
            LOG_ERROR(
                "QwenViTRunner::tryLoadFlexEngine(): FLEX engine requires 3 ViT deepstack features, but visual engine exposes %zu.",
                mDeepstackFeatures.size());
            return false;
        }

        for (char const* name : kFlexDeepstackInputNames)
        {
            if (hasIOTensor(*mFlexEngine, name)
                && !checkTensorDataType(*mFlexEngine, name, nvinfer1::DataType::kHALF))
            {
                return false;
            }
        }
        for (char const* name : kFlexDeepstackOutputNames)
        {
            if (hasIOTensor(*mFlexEngine, name)
                && !checkTensorDataType(*mFlexEngine, name, nvinfer1::DataType::kHALF))
            {
                return false;
            }
        }
        if (mFlexHasMetadataInputs)
        {
            if (!checkTensorDataType(*mFlexEngine, kFlexCameraIdsName, nvinfer1::DataType::kINT64)
                || !checkTensorDataType(*mFlexEngine, kFlexRelativeTimesName, nvinfer1::DataType::kHALF))
            {
                return false;
            }
        }
        if (hasIOTensor(*mFlexEngine, kFlexInputName)
            && !checkTensorDataType(*mFlexEngine, kFlexInputName, nvinfer1::DataType::kHALF))
        {
            return false;
        }
        if (!checkTensorDataType(*mFlexEngine, kFlexOutputName, nvinfer1::DataType::kHALF))
        {
            return false;
        }

        nvinfer1::Dims const flexOutputShape = mFlexEngine->getTensorShape(kFlexOutputName);
        if (flexOutputShape.nbDims == 3 && flexOutputShape.d[1] > 0)
        {
            mFlexMaxSceneTokens = flexOutputShape.d[1];
        }
        if (flexOutputShape.nbDims == 3 && flexOutputShape.d[2] > 0
            && flexOutputShape.d[2] != mConfig.outHiddenSize)
        {
            LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): FLEX hidden size %d does not match ViT/LLM hidden size %d.",
                flexOutputShape.d[2], mConfig.outHiddenSize);
            return false;
        }

        mFlexOutputEmbedding = rt::Tensor({mFlexMaxSceneTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "QwenViTRunner::mFlexOutputEmbedding");
        if (!mFlexContext->setTensorAddress(kFlexOutputName, mFlexOutputEmbedding.rawPointer()))
        {
            LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): Failed to bind FLEX output tensor.");
            return false;
        }

        if (mFlexHasDeepstackOutputs)
        {
            mFlexDeepstackFeatures.clear();
            mFlexDeepstackFeatures.reserve(kFlexDeepstackOutputNames.size());
            for (size_t idx = 0; idx < kFlexDeepstackOutputNames.size(); ++idx)
            {
                mFlexDeepstackFeatures.emplace_back(rt::Tensor({mFlexMaxSceneTokens, mConfig.outHiddenSize},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
                    "QwenViTRunner::mFlexDeepstackFeatures" + std::to_string(idx)));
                if (!mFlexContext->setTensorAddress(
                        kFlexDeepstackOutputNames[idx], mFlexDeepstackFeatures.back().rawPointer()))
                {
                    LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): Failed to bind FLEX output tensor %s.",
                        kFlexDeepstackOutputNames[idx]);
                    return false;
                }
            }
        }

        if (mFlexHasMetadataInputs)
        {
            mFlexCameraIdsHost = rt::Tensor({1, mFlexExpectedVisualTokens}, rt::DeviceType::kCPU,
                nvinfer1::DataType::kINT64, "QwenViTRunner::mFlexCameraIdsHost");
            mFlexCameraIdsDevice = rt::Tensor({1, mFlexExpectedVisualTokens}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kINT64, "QwenViTRunner::mFlexCameraIdsDevice");
            mFlexRelativeTimesHost = rt::Tensor({1, mFlexExpectedVisualTokens, 1}, rt::DeviceType::kCPU,
                nvinfer1::DataType::kHALF, "QwenViTRunner::mFlexRelativeTimesHost");
            mFlexRelativeTimesDevice = rt::Tensor({1, mFlexExpectedVisualTokens, 1}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF, "QwenViTRunner::mFlexRelativeTimesDevice");
        }

        mFlexEnabled = true;
        LOG_INFO(
            "QwenViTRunner: enabled FLEX scene encoder from %s, scene_tokens=%lld, tokens_per_image=%lld, expected_vit_tokens=%lld, deepstack_io=%d, metadata_inputs=%d",
            flexEnginePath.c_str(), static_cast<long long>(mFlexMaxSceneTokens),
            static_cast<long long>(mFlexSceneTokensPerImage), static_cast<long long>(mFlexExpectedVisualTokens),
            mFlexHasDeepstackOutputs ? 1 : 0, mFlexHasMetadataInputs ? 1 : 0);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QwenViTRunner::tryLoadFlexEngine(): %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;

    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to parse config file with error: %s", e.what());
        return false;
    }

    std::string modelTypeStr = jsonConfig["model_type"].get<std::string>();
    mModelType = multimodal::stringToModelType(modelTypeStr);
    if (mModelType != multimodal::ModelType::QWEN2_5_VL && mModelType != multimodal::ModelType::QWEN2_VL
        && mModelType != multimodal::ModelType::QWEN3_VL
        && mModelType != multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Invalid model type: %s", modelTypeStr.c_str());
        return false;
    }

    mConfig.visionStartTokenId = jsonConfig["vision_start_token_id"].get<int32_t>();
    mConfig.visionEndTokenId = jsonConfig.value("vision_end_token_id", 0);
    mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    mConfig.videoTokenId = jsonConfig["video_token_id"].get<int32_t>();

    auto const& subConfig
        = (mModelType == multimodal::ModelType::QWEN2_VL || mModelType == multimodal::ModelType::QWEN2_5_VL)
        ? jsonConfig
        : jsonConfig["text_config"];
    mConfig.vocabSize = subConfig["vocab_size"].get<int32_t>();
    mConfig.mropeTheta = subConfig["rope_theta"].get<float>();

    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        mConfig.windowSize = jsonConfig["vision_config"]["window_size"].get<int64_t>();
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL
        || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        auto visionConfig = jsonConfig["vision_config"];
        auto numPositionEmbeddings = visionConfig["num_position_embeddings"].get<int64_t>();
        mConfig.numGridPerSide = static_cast<int64_t>(std::sqrt(numPositionEmbeddings));
        mConfig.numDeepstackFeatures = visionConfig["deepstack_visual_indexes"].get<std::vector<int64_t>>().size();
    }

    auto builderConfig = jsonConfig["builder_config"];
    mConfig.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    mConfig.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();
    if (mConfig.minImageTokensPerImage <= 0 || mConfig.maxImageTokensPerImage <= 0)
    {
        LOG_ERROR(
            "QwenViTRunner::validateAndFillConfig(): minImageTokensPerImage and maxImageTokensPerImage must be "
            "positive, got %d and %d",
            mConfig.minImageTokensPerImage, mConfig.maxImageTokensPerImage);
        return false;
    }

    // Get preprocessor config
    Json preprocessorConfig;
    std::string preprocessorConfigPath = engineDir + "/preprocessor_config.json";
    std::ifstream preprocessorConfigFileStream(preprocessorConfigPath);
    if (!preprocessorConfigFileStream.is_open())
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to open preprocessor config file: %s",
            preprocessorConfigPath.c_str());
        return false;
    }
    try
    {
        preprocessorConfig = Json::parse(preprocessorConfigFileStream);
        preprocessorConfigFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("QwenViTRunner::validateAndFillConfig(): Failed to parse preprocessor config file with error: %s",
            e.what());
        return false;
    }

    mConfig.patchSize = preprocessorConfig["patch_size"].get<int64_t>();
    mConfig.temporalPatchSize = preprocessorConfig["temporal_patch_size"].get<int64_t>();
    mConfig.mergeSize = preprocessorConfig["merge_size"].get<int64_t>();
    mConfig.imageMean = preprocessorConfig["image_mean"].get<std::vector<float>>();
    mConfig.imageStd = preprocessorConfig["image_std"].get<std::vector<float>>();

    // Get config from engine shapes
    nvinfer1::Dims const inputShapeMax
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims const inputShapeMin
        = mVisualEngine->getProfileShape(binding_names::kVisualInput, 0, nvinfer1::OptProfileSelector::kMIN);
    mConfig.maxHW = inputShapeMax.d[0];
    mConfig.minHW = inputShapeMin.d[0];
    auto maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    mConfig.maxNumImages = maxImageTokens / mConfig.minImageTokensPerImage;
    mConfig.inputDim = mContext->getTensorShape(binding_names::kVisualInput).d[1];
    mConfig.vitPosEmbDim = mContext->getTensorShape(binding_names::kRotaryPosEmb).d[1];
    mConfig.outHiddenSize = mVisualEngine->getTensorShape(binding_names::kVisualOutput).d[1];

    return true;
}

bool QwenViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    mVitInput = rt::Tensor(
        {mConfig.maxHW, mConfig.inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mVitInput");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());

    mRotaryPosEmb = rt::Tensor({mConfig.maxHW, mConfig.vitPosEmbDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "QwenViTRunner::mRotaryPosEmb");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kRotaryPosEmb, mRotaryPosEmb.rawPointer());

    // The size of the tensor is maxNumImages + 1 because the first element is 0.
    mCuSeqlens = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuSeqlens");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kCuSeqlens, mCuSeqlens.rawPointer());
    // Pre-allocate host tensor for cumulative sequence lengths.
    mCuSeqlensHost = rt::Tensor(
        {mConfig.maxNumImages + 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuSeqlensHost");

    mMaxSeqLenCarrier = rt::Tensor(
        {mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mMaxSeqLenCarrier");
    setTensorAddressStatus
        &= mContext->setTensorAddress(binding_names::kMaxSeqLenCarrier, mMaxSeqLenCarrier.rawPointer());

    // In Qwen-VL, VIT input mHW is always numImageTokens * spatial_merge_size ** 2.
    auto const maxImageTokens = mConfig.maxHW / (mConfig.mergeSize * mConfig.mergeSize);
    mOutputEmbedding = rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "QwenViTRunner::mOutputEmbedding");
    setTensorAddressStatus &= mContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());

    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        // Use maxImageTokens as a safe upper bound for cumulative window sequence lengths.
        mCuWindowSeqlens = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuWindowSeqlens");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kCuWindowSeqlens, mCuWindowSeqlens.rawPointer());
        mCuWindowSeqlensHost = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "QwenViTRunner::mCuWindowSeqlensHost");

        mWindowIndexHost = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mWindowIndexHost");
        mWindowIndexDevice = rt::Tensor(
            {maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mWindowIndexDevice");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kWindowIndex, mWindowIndexDevice.rawPointer());

        mReverseWindowIndexHost = rt::Tensor({maxImageTokens}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
            "QwenViTRunner::mReverseWindowIndexHost");
        mReverseWindowIndexDevice = rt::Tensor({maxImageTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
            "QwenViTRunner::mReverseWindowIndexDevice");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.rawPointer());
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL
        || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        mFastPosEmbIdx = rt::Tensor(
            {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mFastPosEmbIdx");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());

        mFastPosEmbWeight = rt::Tensor(
            {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mFastPosEmbWeight");
        setTensorAddressStatus
            &= mContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

        for (int64_t i = 0; i < mConfig.numDeepstackFeatures; ++i)
        {
            // Set tensor name to match the engine binding name.
            std::string const deepstackFeatureName = binding_names::formatDeepstackFeaturesName(i);
            mDeepstackFeatures.emplace_back(rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF, deepstackFeatureName));
            setTensorAddressStatus
                &= mContext->setTensorAddress(deepstackFeatureName.c_str(), mDeepstackFeatures.back().rawPointer());
        }
    }

    if (!setTensorAddressStatus)
    {
        LOG_ERROR("Failed to set tensor address to the engine");
        return false;
    }

    // Copy image mean and std to device to be used in normalizeImage
    auto nbBytes = mConfig.imageMean.size() * sizeof(float);
    auto channels = math::cast<int64_t>(mConfig.imageMean.size());
    mImageMean = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageMean");
    mImageStd = rt::Tensor({channels}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "QwenViTRunner::mImageStd");
    CUDA_CHECK(
        cudaMemcpyAsync(mImageMean.rawPointer(), mConfig.imageMean.data(), nbBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(mImageStd.rawPointer(), mConfig.imageStd.data(), nbBytes, cudaMemcpyHostToDevice, stream));

    // Pre-allocate temporary image buffers for preprocessing
    int64_t const maxImagePixels = mVitInput.getShape().volume();
    // Set max image size to 1xmaxImagePixelsxchannels, will reshape to actual image size in resizeImage
    rt::Tensor resizeBuffer(
        {1, maxImagePixels, channels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "QwenViTRunner::resizeBuffer");
    mResizedImageHost = rt::imageUtils::ImageData(std::move(resizeBuffer));
    mImageDevice
        = rt::Tensor({maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "QwenViTRunner::mImageDevice");
    mNormalizedImageDevice = rt::Tensor(
        {maxImagePixels}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "QwenViTRunner::mNormalizedImageDevice");

    // Pre-allocate tensors for MRoPE position IDs
    mMropePositionIdsHost = rt::Tensor({mLLMMaxBatchSize, 3, mLLMMaxSequenceLength}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kINT64, "QwenViTRunner::mMropePositionIdsHost");
    mMropePositionIdsDevice = rt::Tensor({mLLMMaxBatchSize, 3, mLLMMaxSequenceLength}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "QwenViTRunner::mMropePositionIdsDevice");
    mRopeDeltasHost = rt::Tensor(
        {mLLMMaxBatchSize, 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64, "QwenViTRunner::mRopeDeltasHost");

    return true;
}

void QwenViTRunner::formatPatch(rt::imageUtils::ImageData const& image,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths, int32_t* cuSeqlensData,
    int64_t& cuSeqlensSize, int64_t& maxSeqLen, cudaStream_t stream)
{
    int64_t height = image.height;
    int64_t width = image.width;
    int64_t channels = image.channels;
    unsigned char* imageData = image.data(); // In hwc order

    if (height % (mConfig.patchSize * mConfig.mergeSize) != 0 || width % (mConfig.patchSize * mConfig.mergeSize) != 0)
    {
        throw std::runtime_error("Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));
    }

    std::vector<int64_t> curGrid{1, (height / mConfig.patchSize), (width / mConfig.patchSize)};
    imageGridTHWs.emplace_back(curGrid);
    int64_t curSeqLength = (height / mConfig.patchSize) * (width / mConfig.patchSize);
    int64_t prevCuSeqlen = cuSeqlensData[cuSeqlensSize - 1];
    if (prevCuSeqlen + curSeqLength > mConfig.maxHW || cuSeqlensSize > (mConfig.maxNumImages + 1))
    {
        throw std::runtime_error("cuSeqlens " + std::to_string(prevCuSeqlen + curSeqLength)
            + " exceeds the limitation, maxHW = " + std::to_string(mConfig.maxHW)
            + " or maxNumImages = " + std::to_string(mConfig.maxNumImages) + " of VIT engine.");
    }
    imageTokenLengths.emplace_back(curSeqLength / mConfig.mergeSize / mConfig.mergeSize);
    maxSeqLen = std::max(maxSeqLen, curSeqLength);

    // Reshape pre-allocated temporary buffers to current image dimensions
    check::check(mImageDevice.reshape({mConfig.temporalPatchSize, height, width, channels}), "Tensor reshape failed");
    check::check(
        mNormalizedImageDevice.reshape({mConfig.temporalPatchSize, height, width, channels}), "Tensor reshape failed");

    // Copy image to device. Repeat for T = temporalPatchSize
    auto imageSize = height * width * channels;
    for (int64_t i = 0; i < mConfig.temporalPatchSize; ++i)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mImageDevice.rawPointer()) + i * imageSize, imageData,
            imageSize, cudaMemcpyHostToDevice, stream));
    }

    // mResizedImageHost is reused across images. Ensure the H2D copy completes before the host buffer can be
    // overwritten by the next resize operation.
    if (image.buffer == mResizedImageHost.buffer)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (!mActiveDebugDumpDir.empty())
    {
        auto const imageDir = mActiveDebugDumpDir / format::fmtstr("preprocess_image_%02lld", mActiveDebugImageIndex);
        std::error_code ec;
        std::filesystem::create_directories(imageDir, ec);

        Json hostMeta{
            {"shape", std::vector<int64_t>{height, width, channels}},
            {"dtype", "UINT8"},
            {"file", "host_image.bin"},
        };
        writeBinaryFile(imageDir / "host_image.bin", imageData, static_cast<size_t>(imageSize));

        std::vector<uint8_t> imageDeviceHost;
        copyTensorToHostBytes(mImageDevice, imageDeviceHost, stream);
        Json imageDeviceMeta{
            {"shape", coordsToVector(mImageDevice.getShape())},
            {"dtype", getDataTypeString(mImageDevice.getDataType())},
            {"file", "image_device.bin"},
        };
        writeBinaryFile(imageDir / "image_device.bin", imageDeviceHost.data(), imageDeviceHost.size());

        Json meta{
            {"host_image", hostMeta},
            {"image_device", imageDeviceMeta},
        };
        std::ofstream metaFile(imageDir / "preprocess_stage_meta_pre_norm.json");
        metaFile << meta.dump(2);
    }

    // Normalize image
    kernel::normalizeImage(mImageDevice, mImageMean, mImageStd, mNormalizedImageDevice, stream);

    if (!mActiveDebugDumpDir.empty())
    {
        auto const imageDir = mActiveDebugDumpDir / format::fmtstr("preprocess_image_%02lld", mActiveDebugImageIndex);
        std::vector<uint8_t> normalizedHost;
        copyTensorToHostBytes(mNormalizedImageDevice, normalizedHost, stream);
        Json meta;
        {
            std::ifstream metaFileIn(imageDir / "preprocess_stage_meta_pre_norm.json");
            if (metaFileIn.is_open())
            {
                meta = Json::parse(metaFileIn, nullptr, false);
                if (meta.is_discarded())
                {
                    meta = Json::object();
                }
            }
        }
        meta["normalized_image"] = Json{
            {"shape", coordsToVector(mNormalizedImageDevice.getShape())},
            {"dtype", getDataTypeString(mNormalizedImageDevice.getDataType())},
            {"file", "normalized_image.bin"},
        };
        writeBinaryFile(imageDir / "normalized_image.bin", normalizedHost.data(), normalizedHost.size());
        std::ofstream metaFile(imageDir / "preprocess_stage_meta_pre_transpose.json");
        metaFile << meta.dump(2);
    }

    // Transpose to patch
    kernel::transposeToPatchQwenViT(mNormalizedImageDevice, mVitInput, prevCuSeqlen * mConfig.inputDim,
        mConfig.temporalPatchSize, mConfig.patchSize, mConfig.mergeSize, stream);

    if (!mActiveDebugDumpDir.empty())
    {
        auto const imageDir = mActiveDebugDumpDir / format::fmtstr("preprocess_image_%02lld", mActiveDebugImageIndex);
        int64_t const rawSeqLength = curSeqLength;
        size_t const elemSize = rt::utils::getTypeSize(mVitInput.getDataType());
        size_t const inputOffsetBytes = static_cast<size_t>(prevCuSeqlen * mConfig.inputDim) * elemSize;
        size_t const sliceBytes = static_cast<size_t>(rawSeqLength * mConfig.inputDim) * elemSize;
        std::vector<uint8_t> vitSliceHost(sliceBytes);
        CUDA_CHECK(cudaMemcpyAsync(vitSliceHost.data(),
            static_cast<std::byte const*>(mVitInput.rawPointer()) + inputOffsetBytes, sliceBytes, cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Json meta;
        {
            std::ifstream metaFileIn(imageDir / "preprocess_stage_meta_pre_transpose.json");
            if (metaFileIn.is_open())
            {
                meta = Json::parse(metaFileIn, nullptr, false);
                if (meta.is_discarded())
                {
                    meta = Json::object();
                }
            }
        }
        meta["vit_input_slice"] = Json{
            {"shape", std::vector<int64_t>{rawSeqLength, mConfig.inputDim}},
            {"dtype", getDataTypeString(mVitInput.getDataType())},
            {"file", "vit_input_slice.bin"},
        };
        writeBinaryFile(imageDir / "vit_input_slice.bin", vitSliceHost.data(), vitSliceHost.size());
        std::ofstream metaFile(imageDir / "preprocess_stage_meta.json");
        metaFile << meta.dump(2);
    }

    // Update sequence length
    cuSeqlensData[cuSeqlensSize++] = static_cast<int32_t>(prevCuSeqlen + curSeqLength);
    ++mActiveDebugImageIndex;
}

std::tuple<int64_t, int64_t> QwenViTRunner::getResizedImageSize(
    int64_t const height, int64_t const width, int64_t const maxRatio)
{
    // According to https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    int64_t const factor = mConfig.patchSize * mConfig.mergeSize;
    int64_t const minPixels = mConfig.minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = mConfig.maxImageTokensPerImage * factor * factor;

    auto roundByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::round(static_cast<double>(value) / factor) * factor;
    };
    auto floorByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::floor(static_cast<double>(value) / factor) * factor;
    };
    auto ceilByFactor = [](int64_t value, int64_t factor) -> int64_t {
        return std::ceil(static_cast<double>(value) / factor) * factor;
    };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::runtime_error("absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));
    }

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        hBar = floorByFactor(static_cast<int64_t>(height / beta), factor);
        wBar = floorByFactor(static_cast<int64_t>(width / beta), factor);
    }
    else if (hBar * wBar < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

void QwenViTRunner::imagePreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int64_t>>& imageGridTHWs, std::vector<int64_t>& imageTokenLengths,
    std::vector<int64_t>& numImages, bool doResize, cudaStream_t stream)
{
    // Use pre-allocated pinned host tensor for cumulative sequence lengths
    int32_t* cuSeqlensData = mCuSeqlensHost.dataPointer<int32_t>();
    cuSeqlensData[0] = 0;
    int64_t cuSeqlensSize = 1;
    int64_t maxSeqLen = 0;

    for (auto const& req : request.requests)
    {
        int64_t numImage = 0;
        for (auto const& image : req.imageBuffers)
        {
            if (doResize)
            {
                auto [resizedHeight, resizedWidth] = getResizedImageSize(image.height, image.width);
                rt::imageUtils::resizeImage(image, mResizedImageHost, resizedWidth, resizedHeight);
                formatPatch(mResizedImageHost, imageGridTHWs, imageTokenLengths, cuSeqlensData, cuSeqlensSize,
                    maxSeqLen, stream);
            }
            else
            {
                formatPatch(image, imageGridTHWs, imageTokenLengths, cuSeqlensData, cuSeqlensSize, maxSeqLen, stream);
            }
            ++numImage;
        }
        numImages.emplace_back(numImage);
    }

    int64_t totalSeqLength = cuSeqlensData[cuSeqlensSize - 1];
    if (totalSeqLength == 0)
    {
        check::check(mVitInput.reshape({totalSeqLength, mConfig.inputDim}), "Tensor reshape failed");
        return;
    }

    if (totalSeqLength < mConfig.minHW || totalSeqLength > mConfig.maxHW)
    {
        throw std::runtime_error("totalSeqLength " + std::to_string(totalSeqLength) + " exceeds the limitation, max = "
            + std::to_string(mConfig.maxHW) + ", min = " + std::to_string(mConfig.minHW) + " of VIT engine.");
    }

    // Reshape tensors
    int64_t totalImageTokens = totalSeqLength / (mConfig.mergeSize * mConfig.mergeSize);
    check::check(mVitInput.reshape({totalSeqLength, mConfig.inputDim}), "Tensor reshape failed");
    check::check(mOutputEmbedding.reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    check::check(mMaxSeqLenCarrier.reshape({maxSeqLen}), "Tensor reshape failed");
    // Record performance data
    int64_t imageCount = std::accumulate(numImages.begin(), numImages.end(), int64_t(0));
    mMultimodalMetrics.recordRun(imageCount, totalImageTokens);

    /*
     * Cache optimization for cu_seqlens, rotary position embeddings, and other image grid dependent
     * input tensors. Reuse the data from last round of computation if the image grid sizes are identical.
     * This reduces inference latency by skipping invariant tensor initialization.
     */
    if (imageGridTHWs != mLastImageGridTHWs)
    {
        check::check(mCuSeqlens.reshape({cuSeqlensSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mCuSeqlens.rawPointer(), mCuSeqlensHost.rawPointer(),
            cuSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        check::check(mRotaryPosEmb.reshape({totalSeqLength, mConfig.vitPosEmbDim}), "Tensor reshape failed");
        // Compute rotary position embeddings
        for (size_t i = 0; i < imageGridTHWs.size(); ++i)
        {
            kernel::initRotaryPosEmbQwenViT(
                mRotaryPosEmb, imageGridTHWs[i], mConfig.mergeSize, cuSeqlensData[i], 10000.0f, 1.0f, stream);
        }

        // Compute additional inputs
        if (mModelType == multimodal::ModelType::QWEN2_5_VL)
        {
            check::check(mWindowIndexHost.reshape({totalImageTokens}), "Tensor reshape failed");
            check::check(mWindowIndexDevice.reshape({totalImageTokens}), "Tensor reshape failed");
            check::check(mReverseWindowIndexHost.reshape({totalImageTokens}), "Tensor reshape failed");
            check::check(mReverseWindowIndexDevice.reshape({totalImageTokens}), "Tensor reshape failed");

            getWindowIndex(imageGridTHWs, totalSeqLength, stream);
        }
        else if (mModelType == multimodal::ModelType::QWEN3_VL
            || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
        {
            check::check(mFastPosEmbIdx.reshape({4, totalSeqLength}), "Tensor reshape failed");
            check::check(mFastPosEmbWeight.reshape({4, totalSeqLength}), "Tensor reshape failed");

            for (size_t i = 0; i < imageGridTHWs.size(); ++i)
            {
                kernel::initFastPosEmbedQwenViT(mFastPosEmbIdx, mFastPosEmbWeight, imageGridTHWs[i], mConfig.mergeSize,
                    mConfig.numGridPerSide, cuSeqlensData[i], stream);
            }

            for (int64_t i = 0; i < mConfig.numDeepstackFeatures; ++i)
            {
                check::check(
                    mDeepstackFeatures[i].reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
            }
        }
        mLastImageGridTHWs = imageGridTHWs;
    }
}

void QwenViTRunner::getMRopePositionIds(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int64_t>> const& imageGridTHWs) noexcept
{
    // According to transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.get_rope_index
    // mropePositionIds: (bs, 3, maxPositionEmbeddings), 3 is for T, H, W
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t totalImageIdx = 0;
    int64_t batchOffset = 0;

    for (auto const& inputIds : batchInputIds)
    {
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int64_t startIdx = 0;
        int64_t remainingStartPos = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
            // Text part
            int64_t textLen = it + 1 - start;
            for (int64_t i = 0; i < 3; ++i)
            {
                for (int64_t j = 0; j < textLen; ++j)
                {
                    mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
                }
            }

            // Visual part
            int64_t T = imageGridTHWs[totalImageIdx][0];
            int64_t H = imageGridTHWs[totalImageIdx][1] / mConfig.mergeSize;
            int64_t W = imageGridTHWs[totalImageIdx][2] / mConfig.mergeSize;
            ++totalImageIdx;

            for (int64_t t = 0; t < T; ++t)
            {
                for (int64_t h = 0; h < H; ++h)
                {
                    for (int64_t w = 0; w < W; ++w)
                    {
                        int64_t idx = remainingStartPos + textLen + t * H * W + h * W + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx] = t + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + T * H * W;
            startIdx += std::max(T, std::max(H, W)) + textLen;
            remainingStartPos = start - inputIds.begin();
        }

        // Remaining text part till maxPositionEmbeddings. Treat all generated tokens as text tokens.
        int64_t textLen = maxPositionEmbeddings - remainingStartPos;
        for (int64_t i = 0; i < 3; ++i)
        {
            for (int64_t j = 0; j < textLen; ++j)
            {
                mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
            }
        }

        batchOffset += 3 * maxPositionEmbeddings;
    }
}

void QwenViTRunner::getFlexMRopePositionIds(std::vector<std::vector<int32_t>> const& batchInputIds) noexcept
{
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t batchOffset = 0;

    for (auto const& inputIds : batchInputIds)
    {
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int64_t startIdx = 0;
        int64_t remainingStartPos = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
            int64_t const textLen = it + 1 - start;
            for (int64_t axis = 0; axis < 3; ++axis)
            {
                for (int64_t j = 0; j < textLen; ++j)
                {
                    mropePositionIdsPtr[batchOffset + axis * maxPositionEmbeddings + remainingStartPos + j]
                        = j + startIdx;
                }
            }

            int64_t visualLen = mFlexSceneTokensPerImage;
            int64_t const tokensAfterVisionStart = end - (it + 1);
            if (tokensAfterVisionStart < visualLen)
            {
                visualLen = std::max<int64_t>(tokensAfterVisionStart, 0);
                LOG_WARNING(
                    "QwenViTRunner::getFlexMRopePositionIds(): truncated FLEX visual span to %lld tokens.",
                    static_cast<long long>(visualLen));
            }

            for (int64_t k = 0; k < visualLen; ++k)
            {
                int64_t const idx = remainingStartPos + textLen + k;
                int64_t const pos = textLen + startIdx + k;
                for (int64_t axis = 0; axis < 3; ++axis)
                {
                    mropePositionIdsPtr[batchOffset + axis * maxPositionEmbeddings + idx] = pos;
                }
            }

            start = it + 1 + visualLen;
            startIdx += textLen + visualLen;
            remainingStartPos = start - inputIds.begin();
        }

        int64_t const textLen = maxPositionEmbeddings - remainingStartPos;
        for (int64_t axis = 0; axis < 3; ++axis)
        {
            for (int64_t j = 0; j < textLen; ++j)
            {
                mropePositionIdsPtr[batchOffset + axis * maxPositionEmbeddings + remainingStartPos + j]
                    = j + startIdx;
            }
        }

        batchOffset += 3 * maxPositionEmbeddings;
    }
}

void QwenViTRunner::generateMropeParams(std::vector<std::vector<int32_t>> const& batchInputIds,
    std::vector<std::vector<int64_t>> const& imageGridTHWs, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    int64_t const activeBatchSize = batchInputIds.size();
    auto ropeRotaryCosSinDim = ropeRotaryCosSinDevice.getShape();
    int64_t const maxPositionEmbeddings = ropeRotaryCosSinDim[1];
    int64_t const rotaryDim = ropeRotaryCosSinDim[2];

    bool checkShapeValid = activeBatchSize <= mLLMMaxBatchSize && maxPositionEmbeddings <= mLLMMaxSequenceLength;
    if (!checkShapeValid)
    {
        LOG_ERROR(
            "mropePositionIdsHost shape is not valid. Allowed shape: [%d, 3, %d]. "
            "Got activeBatchSize: %d, maxPositionEmbeddings: %ld",
            mLLMMaxBatchSize, mLLMMaxSequenceLength, activeBatchSize, maxPositionEmbeddings);
        throw std::runtime_error("mropePositionIdsHost shape validation failed");
    }

    // Initialize mropePositionIds and copy to device
    check::check(mMropePositionIdsHost.reshape({activeBatchSize, 3, maxPositionEmbeddings}), "Tensor reshape failed");
    check::check(mMropePositionIdsDevice.reshape({activeBatchSize, 3, maxPositionEmbeddings}), "Tensor reshape failed");
    check::check(mRopeDeltasHost.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (mFlexEnabled)
    {
        getFlexMRopePositionIds(batchInputIds);
    }
    else
    {
        getMRopePositionIds(batchInputIds, imageGridTHWs);
    }

    // Match Hugging Face rope_deltas semantics: max(active_position_ids) + 1 - input_length.
    int64_t const* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t* ropeDeltasPtr = mRopeDeltasHost.dataPointer<int64_t>();
    for (int64_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int64_t const inputLength = static_cast<int64_t>(batchInputIds[batchIdx].size());
        if (inputLength <= 0)
        {
            ropeDeltasPtr[batchIdx] = 0;
            continue;
        }

        int64_t const batchOffset = batchIdx * 3 * maxPositionEmbeddings;
        int64_t maxActivePositionId = 0;
        for (int64_t axis = 0; axis < 3; ++axis)
        {
            int64_t const axisOffset = batchOffset + axis * maxPositionEmbeddings;
            for (int64_t pos = 0; pos < inputLength; ++pos)
            {
                maxActivePositionId = std::max(maxActivePositionId, mropePositionIdsPtr[axisOffset + pos]);
            }
        }
        ropeDeltasPtr[batchIdx] = maxActivePositionId + 1 - inputLength;
        if (mFlexEnabled && batchIdx < static_cast<int64_t>(mFlexRopeDeltaCorrections.size()))
        {
            ropeDeltasPtr[batchIdx] += mFlexRopeDeltaCorrections[batchIdx];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mMropePositionIdsDevice.rawPointer(), mMropePositionIdsHost.rawPointer(),
        activeBatchSize * 3 * maxPositionEmbeddings * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Initialize mrope cosSinCacheDevice
    check::check(
        ropeRotaryCosSinDevice.reshape({activeBatchSize, maxPositionEmbeddings, rotaryDim}), "Tensor reshape failed");
    bool interleaved = (mModelType == multimodal::ModelType::QWEN3_VL);
    kernel::initializeMRopeCosSin(ropeRotaryCosSinDevice.dataPointer<float>(),
        mMropePositionIdsDevice.dataPointer<int64_t>(), mConfig.mropeTheta, rotaryDim, maxPositionEmbeddings,
        activeBatchSize, interleaved, stream);
}

void QwenViTRunner::getWindowIndex(
    std::vector<std::vector<int64_t>> const& imageGridTHWs, int64_t const curHW, cudaStream_t stream)
{
    // Init windowIndex and cuWindowSeqlens
    int64_t* windowIndexPtr = mWindowIndexHost.dataPointer<int64_t>();
    int64_t const windowIndexSize = mWindowIndexHost.getShape()[0];
    int64_t const vitMergerWindowSize = mConfig.windowSize / mConfig.mergeSize / mConfig.patchSize;
    int64_t windowIndexPos = 0;
    int64_t windowIndexValue = 0;

    // Use pre-allocated pinned host tensor for cumulative window sequence lengths
    int32_t* cuWindowSeqlensData = mCuWindowSeqlensHost.dataPointer<int32_t>();
    cuWindowSeqlensData[0] = 0;
    int64_t cuWindowSeqlensSize = 1;

    for (auto const& grid : imageGridTHWs)
    {
        int64_t T = grid[0], H = grid[1], W = grid[2];
        int64_t llmGridH = H / mConfig.mergeSize;
        int64_t llmGridW = W / mConfig.mergeSize;
        int64_t numWindowsH = (llmGridH + vitMergerWindowSize - 1) / vitMergerWindowSize;
        int64_t numWindowsW = (llmGridW + vitMergerWindowSize - 1) / vitMergerWindowSize;

        for (int64_t i = 0; i < numWindowsH; ++i)
        {
            for (int64_t j = 0; j < numWindowsW; ++j)
            {
                int64_t cnt{0};
                for (int64_t m = 0; m < vitMergerWindowSize; ++m)
                {
                    for (int64_t n = 0; n < vitMergerWindowSize; ++n)
                    {
                        int64_t idxH = i * vitMergerWindowSize + m;
                        int64_t idxW = j * vitMergerWindowSize + n;
                        if (idxH < llmGridH && idxW < llmGridW)
                        {
                            windowIndexPtr[windowIndexPos++] = idxH * llmGridW + idxW + windowIndexValue;
                            ++cnt;
                        }
                    }
                }

                int32_t prevCuWindowSeqlen = cuWindowSeqlensData[cuWindowSeqlensSize - 1];
                cuWindowSeqlensData[cuWindowSeqlensSize++]
                    = static_cast<int32_t>(prevCuWindowSeqlen + cnt * mConfig.mergeSize * mConfig.mergeSize);
            }
        }

        windowIndexValue += T * llmGridH * llmGridW;
    }

    if (windowIndexPos * (mConfig.mergeSize * mConfig.mergeSize) != curHW)
    {
        throw std::runtime_error(
            "windowIndex size * (mergeSize * mergeSize) does not match curHW. Got windowIndex size: "
            + std::to_string(windowIndexPos) + ", curHW: " + std::to_string(curHW));
    }

    // Copy cu_window_seqlens
    check::check(mCuWindowSeqlens.reshape({cuWindowSeqlensSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCuWindowSeqlens.rawPointer(), mCuWindowSeqlensHost.rawPointer(),
        cuWindowSeqlensSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Copy window index and reverse window index
    int64_t* reverseWindowIndexPtr = mReverseWindowIndexHost.dataPointer<int64_t>();
    std::iota(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize, 0);
    std::sort(reverseWindowIndexPtr, reverseWindowIndexPtr + windowIndexSize,
        [windowIndexPtr](size_t left, size_t right) { return windowIndexPtr[left] < windowIndexPtr[right]; });

    CUDA_CHECK(cudaMemcpyAsync(mWindowIndexDevice.rawPointer(), mWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mReverseWindowIndexDevice.rawPointer(), mReverseWindowIndexHost.rawPointer(),
        windowIndexSize * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
}

void QwenViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& numImages,
    std::vector<int64_t> const& imageTokenLengths, trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    if (numImages.size() != request.requests.size())
    {
        std::string errorMsg = "QwenViTRunner::textPreprocess() numImages.size() != request.requests.size(), "
            + std::to_string(numImages.size()) + " != " + std::to_string(request.requests.size());
        LOG_ERROR("%s", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }

    int64_t imageIndex = 0;
    // For Qwen2.5-VL/Qwen3-VL: use incrementing IDs (>= vocabSize) for embeddingLookupWithImageInsertion
    // For Qwen3-Omni: keep original imageTokenId and wrap with vision_start/vision_end to match PyTorch,
    // since embeddingLookupMultimodal uses multimodalIndices for indexing
    bool const isQwen3Omni = (mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER);
    int32_t nextImageTokenId = mConfig.vocabSize;

    if (isDebugQwenTextPreprocessEnabled())
    {
        LOG_INFO(
            "QwenViTRunner::textPreprocess debug: requests=%zu imageTokenId=%d videoTokenId=%d visionStart=%d visionEnd=%d vocabSize=%d isQwen3Omni=%d",
            request.requests.size(), mConfig.imageTokenId, mConfig.videoTokenId, mConfig.visionStartTokenId,
            mConfig.visionEndTokenId, mConfig.vocabSize, static_cast<int>(isQwen3Omni));
    }

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;

        // Check if already tokenized (incremental mode)
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            // Already tokenized by another runner, use existing tokens
            ids = batchInputIds[i];
        }
        else
        {
            // First runner to process, tokenize the request
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        int64_t placeholderMatchCount = 0;
        int64_t imagePadCount = 0;
        int64_t visionStartCount = 0;
        int64_t visionEndCount = 0;
        int64_t videoTokenCount = 0;
        for (auto const tokenId : ids)
        {
            if (tokenId == mConfig.imageTokenId || tokenId == mConfig.videoTokenId)
            {
                ++placeholderMatchCount;
            }
            if (tokenId == mConfig.imageTokenId)
            {
                ++imagePadCount;
            }
            if (tokenId == mConfig.visionStartTokenId)
            {
                ++visionStartCount;
            }
            if (tokenId == mConfig.visionEndTokenId)
            {
                ++visionEndCount;
            }
            if (tokenId == mConfig.videoTokenId)
            {
                ++videoTokenCount;
            }
        }

        // insert image tokens
        std::vector<int32_t> newIds;
        size_t const imageIndexStart = static_cast<size_t>(imageIndex);
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                int64_t numImageTokens = imageTokenLengths.at(imageIndex);
                if (mFlexEnabled && !isQwen3Omni)
                {
                    numImageTokens = mFlexSceneTokensPerImage;
                }

                if (isQwen3Omni)
                {
                    // Qwen3-Omni: <|vision_start|> + N×<|image_pad|> + <|vision_end|>
                    // TRT chat template only has <|image_pad|>, no start/end markers
                    newIds.push_back(mConfig.visionStartTokenId);
                    for (int64_t k = 0; k < numImageTokens; ++k)
                    {
                        newIds.push_back(mConfig.imageTokenId);
                    }
                    newIds.push_back(mConfig.visionEndTokenId);
                }
                else
                {
                    // Qwen2.5-VL/Qwen3-VL: use incrementing IDs
                    for (int64_t k = 0; k < numImageTokens; ++k)
                    {
                        newIds.push_back(nextImageTokenId);
                        ++nextImageTokenId;
                    }
                }
                ++imageIndex;
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }

        if (isDebugQwenTextPreprocessEnabled())
        {
            std::ostringstream lengthsStream;
            size_t const imageCountForRequest = static_cast<size_t>(numImages.at(i));
            for (size_t localIdx = 0; localIdx < imageCountForRequest; ++localIdx)
            {
                if (localIdx > 0)
                {
                    lengthsStream << ",";
                }
                lengthsStream << imageTokenLengths.at(imageIndexStart + localIdx);
            }

            int64_t postImagePadCount = 0;
            int64_t postVisionStartCount = 0;
            int64_t postVisionEndCount = 0;
            for (auto const tokenId : newIds)
            {
                if (tokenId == mConfig.imageTokenId)
                {
                    ++postImagePadCount;
                }
                if (tokenId == mConfig.visionStartTokenId)
                {
                    ++postVisionStartCount;
                }
                if (tokenId == mConfig.visionEndTokenId)
                {
                    ++postVisionEndCount;
                }
            }

            LOG_INFO(
                "QwenViTRunner::textPreprocess req=%zu ids=%zu placeholderMatches=%lld imagePad=%lld video=%lld visionStart=%lld visionEnd=%lld numImages=%lld imageTokenLengths=[%s] newIds=%zu postImagePad=%lld postVisionStart=%lld postVisionEnd=%lld",
                i, ids.size(), static_cast<long long>(placeholderMatchCount), static_cast<long long>(imagePadCount),
                static_cast<long long>(videoTokenCount), static_cast<long long>(visionStartCount),
                static_cast<long long>(visionEndCount), static_cast<long long>(numImages.at(i)),
                lengthsStream.str().c_str(), newIds.size(), static_cast<long long>(postImagePadCount),
                static_cast<long long>(postVisionStartCount), static_cast<long long>(postVisionEndCount));
        }

        // Update batchInputIds
        if (i < batchInputIds.size())
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

bool QwenViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    std::vector<std::vector<int64_t>> imageGridTHWs;
    std::vector<int64_t> imageTokenLengths;
    std::vector<int64_t> numImages;

    try
    {
        imagePreprocess(request, imageGridTHWs, imageTokenLengths, numImages, true, stream);
        mFlexRopeDeltaCorrections.clear();
        if (mFlexEnabled)
        {
            size_t imageOffset = 0;
            for (size_t batchIdx = 0; batchIdx < numImages.size(); ++batchIdx)
            {
                int64_t originalImageTokens = 0;
                for (int64_t localIdx = 0; localIdx < numImages[batchIdx]; ++localIdx)
                {
                    originalImageTokens += imageTokenLengths.at(imageOffset + static_cast<size_t>(localIdx));
                }
                imageOffset += static_cast<size_t>(numImages[batchIdx]);

                int64_t const compressedImageTokens = numImages[batchIdx] * mFlexSceneTokensPerImage;
                if (compressedImageTokens > mFlexMaxSceneTokens)
                {
                    throw std::runtime_error("FLEX compressed image tokens " + std::to_string(compressedImageTokens)
                        + " exceed FLEX scene token capacity " + std::to_string(mFlexMaxSceneTokens));
                }
                mFlexRopeDeltaCorrections.emplace_back(originalImageTokens - compressedImageTokens);
            }
        }
        textPreprocess(request, batchedInputIds, numImages, imageTokenLengths, tokenizer);
        generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("QwenViTRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer const* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    if (systemPrompt.empty())
    {
        return true;
    }

    // systemPrompt is already formatted by tokenizer's applyChatTemplate
    std::vector<int32_t> ids = tokenizer->encode(systemPrompt);
    if (ids.empty())
    {
        LOG_ERROR("QwenViTRunner::preprocessSystemPrompt(): Failed to encode system prompt.");
        return false;
    }
    std::vector<std::vector<int32_t>> batchedInputIds;
    batchedInputIds.emplace_back(std::move(ids));
    std::vector<std::vector<int64_t>> imageGridTHWs;
    mFlexRopeDeltaCorrections.assign(batchedInputIds.size(), 0);

    try
    {
        generateMropeParams(batchedInputIds, imageGridTHWs, ropeRotaryCosSinDevice, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("MRope parameter generation failed: %s", e.what());
        return false;
    }

    return true;
}

bool QwenViTRunner::populateFlexMetadata(int64_t totalImageTokens, cudaStream_t stream) noexcept
{
    if (!mFlexHasMetadataInputs)
    {
        return true;
    }
    if (totalImageTokens <= 0 || totalImageTokens % kFlexAlpamayoImageCount != 0)
    {
        LOG_ERROR(
            "QwenViTRunner::populateFlexMetadata(): expected token count divisible by %lld Alpamayo images, got %lld.",
            static_cast<long long>(kFlexAlpamayoImageCount), static_cast<long long>(totalImageTokens));
        return false;
    }

    rt::Coords const cameraIdsShape{1, totalImageTokens};
    rt::Coords const relativeTimesShape{1, totalImageTokens, 1};
    if (!mFlexCameraIdsHost.reshape(cameraIdsShape) || !mFlexCameraIdsDevice.reshape(cameraIdsShape)
        || !mFlexRelativeTimesHost.reshape(relativeTimesShape) || !mFlexRelativeTimesDevice.reshape(relativeTimesShape))
    {
        LOG_ERROR("QwenViTRunner::populateFlexMetadata(): failed to reshape FLEX metadata tensors.");
        return false;
    }

    int64_t* cameraIds = mFlexCameraIdsHost.dataPointer<int64_t>();
    half* relativeTimes = mFlexRelativeTimesHost.dataPointer<half>();
    int64_t const tokensPerImage = totalImageTokens / kFlexAlpamayoImageCount;
    for (size_t cameraIdx = 0; cameraIdx < kFlexAlpamayoCameraIds.size(); ++cameraIdx)
    {
        for (int64_t frameIdx = 0; frameIdx < kFlexAlpamayoFramesPerCamera; ++frameIdx)
        {
            int64_t const imageIdx = static_cast<int64_t>(cameraIdx) * kFlexAlpamayoFramesPerCamera + frameIdx;
            int64_t const tokenOffset = imageIdx * tokensPerImage;
            half const relativeTime = __float2half(kFlexAlpamayoRelativeTimes[frameIdx]);
            for (int64_t tokenIdx = 0; tokenIdx < tokensPerImage; ++tokenIdx)
            {
                int64_t const idx = tokenOffset + tokenIdx;
                cameraIds[idx] = kFlexAlpamayoCameraIds[cameraIdx];
                relativeTimes[idx] = relativeTime;
            }
        }
    }

    size_t const cameraIdsBytes = static_cast<size_t>(totalImageTokens) * sizeof(int64_t);
    size_t const relativeTimesBytes = static_cast<size_t>(totalImageTokens) * sizeof(half);
    cudaError_t status = cudaMemcpyAsync(
        mFlexCameraIdsDevice.rawPointer(), mFlexCameraIdsHost.rawPointer(), cameraIdsBytes, cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess)
    {
        LOG_ERROR("QwenViTRunner::populateFlexMetadata(): failed to copy camera_ids: %s", cudaGetErrorString(status));
        return false;
    }
    status = cudaMemcpyAsync(mFlexRelativeTimesDevice.rawPointer(), mFlexRelativeTimesHost.rawPointer(),
        relativeTimesBytes, cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess)
    {
        LOG_ERROR(
            "QwenViTRunner::populateFlexMetadata(): failed to copy relative_times: %s", cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool QwenViTRunner::infer(cudaStream_t stream) noexcept
{
    // Skip VIT inference if there are no images to process
    // Check if the first dimension (sequence length) is 0, indicating no images
    if (mVitInput.getShape()[0] == 0)
    {
        return true;
    }

    // Profile ViT inference with automatic cleanup
    {
        TIME_STAGE(metrics::StageNames::kVISION_ENCODER, stream);

        bool setEngineIOStatus{true};
        setEngineIOStatus &= mContext->setInputShape(binding_names::kVisualInput, mVitInput.getShape().getTRTDims());
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kRotaryPosEmb, mRotaryPosEmb.getShape().getTRTDims());
        setEngineIOStatus &= mContext->setInputShape(binding_names::kCuSeqlens, mCuSeqlens.getShape().getTRTDims());
        setEngineIOStatus
            &= mContext->setInputShape(binding_names::kMaxSeqLenCarrier, mMaxSeqLenCarrier.getShape().getTRTDims());
        if (mModelType == multimodal::ModelType::QWEN2_5_VL)
        {
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kCuWindowSeqlens, mCuWindowSeqlens.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kWindowIndex, mWindowIndexDevice.getShape().getTRTDims());
            setEngineIOStatus &= mContext->setInputShape(
                binding_names::kReverseWindowIndex, mReverseWindowIndexDevice.getShape().getTRTDims());
        }
        else if (mModelType == multimodal::ModelType::QWEN3_VL
            || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
        {
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.getShape().getTRTDims());
            setEngineIOStatus
                &= mContext->setInputShape(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.getShape().getTRTDims());
        }

        if (!setEngineIOStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to bind engine input tensors.");
            return false;
        }

        bool enqueueStatus = mContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to enqueue engine.");
            return false;
        }
    }

    if (mFlexEnabled)
    {
        int64_t const totalImageTokens = mOutputEmbedding.getShape()[0];
        if (totalImageTokens != mFlexExpectedVisualTokens)
        {
            LOG_ERROR(
                "QwenViTRunner::infer(): FLEX K512 engine currently expects %lld ViT tokens, got %lld. Check image count/resolution.",
                static_cast<long long>(mFlexExpectedVisualTokens), static_cast<long long>(totalImageTokens));
            return false;
        }

        TIME_STAGE(metrics::StageNames::kFLEX_ENCODER, stream);

        rt::Coords const flexInputShape{1, totalImageTokens, mConfig.outHiddenSize};
        rt::Coords const flexCameraIdsShape{1, totalImageTokens};
        rt::Coords const flexRelativeTimesShape{1, totalImageTokens, 1};
        if (!populateFlexMetadata(totalImageTokens, stream))
        {
            return false;
        }

        bool setFlexIOStatus{true};
        setFlexIOStatus &= mFlexContext->setInputShape(kFlexInputName, flexInputShape.getTRTDims());
        setFlexIOStatus &= mFlexContext->setTensorAddress(kFlexInputName, mOutputEmbedding.rawPointer());
        if (mFlexHasDeepstackInputs)
        {
            for (size_t idx = 0; idx < kFlexDeepstackInputNames.size(); ++idx)
            {
                setFlexIOStatus &= mFlexContext->setInputShape(kFlexDeepstackInputNames[idx], flexInputShape.getTRTDims());
                setFlexIOStatus
                    &= mFlexContext->setTensorAddress(kFlexDeepstackInputNames[idx], mDeepstackFeatures[idx].rawPointer());
            }
        }
        if (mFlexHasMetadataInputs)
        {
            setFlexIOStatus &= mFlexContext->setInputShape(kFlexCameraIdsName, flexCameraIdsShape.getTRTDims());
            setFlexIOStatus &= mFlexContext->setTensorAddress(kFlexCameraIdsName, mFlexCameraIdsDevice.rawPointer());
            setFlexIOStatus
                &= mFlexContext->setInputShape(kFlexRelativeTimesName, flexRelativeTimesShape.getTRTDims());
            setFlexIOStatus
                &= mFlexContext->setTensorAddress(kFlexRelativeTimesName, mFlexRelativeTimesDevice.rawPointer());
        }
        setFlexIOStatus &= mFlexContext->setTensorAddress(kFlexOutputName, mFlexOutputEmbedding.rawPointer());
        if (mFlexHasDeepstackOutputs)
        {
            for (size_t idx = 0; idx < kFlexDeepstackOutputNames.size(); ++idx)
            {
                setFlexIOStatus &= mFlexContext->setTensorAddress(
                    kFlexDeepstackOutputNames[idx], mFlexDeepstackFeatures[idx].rawPointer());
            }
        }
        if (!setFlexIOStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to bind FLEX engine tensors.");
            return false;
        }

        nvinfer1::Dims const flexOutputShape = mFlexContext->getTensorShape(kFlexOutputName);
        if (flexOutputShape.nbDims == 3 && flexOutputShape.d[1] > 0)
        {
            if (flexOutputShape.d[1] > mFlexMaxSceneTokens || flexOutputShape.d[2] != mConfig.outHiddenSize)
            {
                LOG_ERROR("QwenViTRunner::infer(): Invalid FLEX output shape [%d, %d, %d].", flexOutputShape.d[0],
                    flexOutputShape.d[1], flexOutputShape.d[2]);
                return false;
            }
            check::check(mFlexOutputEmbedding.reshape({flexOutputShape.d[1], flexOutputShape.d[2]}),
                "Tensor reshape failed");
        }
        else
        {
            check::check(mFlexOutputEmbedding.reshape({mFlexMaxSceneTokens, mConfig.outHiddenSize}),
                "Tensor reshape failed");
        }

        if (mFlexHasDeepstackOutputs)
        {
            int64_t const sceneTokens = mFlexOutputEmbedding.getShape()[0];
            for (size_t idx = 0; idx < kFlexDeepstackOutputNames.size(); ++idx)
            {
                nvinfer1::Dims const flexDeepstackShape = mFlexContext->getTensorShape(kFlexDeepstackOutputNames[idx]);
                int64_t outputTokens = sceneTokens;
                int64_t outputHiddenSize = mConfig.outHiddenSize;
                if (flexDeepstackShape.nbDims == 3 && flexDeepstackShape.d[1] > 0)
                {
                    outputTokens = flexDeepstackShape.d[1];
                    outputHiddenSize = flexDeepstackShape.d[2];
                }
                if (outputTokens > mFlexMaxSceneTokens || outputHiddenSize != mConfig.outHiddenSize)
                {
                    LOG_ERROR("QwenViTRunner::infer(): Invalid FLEX deepstack output shape [%d, %d, %d].",
                        flexDeepstackShape.d[0], flexDeepstackShape.d[1], flexDeepstackShape.d[2]);
                    return false;
                }
                check::check(mFlexDeepstackFeatures[idx].reshape({outputTokens, outputHiddenSize}),
                    "Tensor reshape failed");
            }
        }

        bool enqueueStatus = mFlexContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("QwenViTRunner::infer(): Failed to enqueue FLEX engine.");
            return false;
        }
    }

    return true;
}

rt::OptionalInputTensors QwenViTRunner::getDeepstackFeatures()
{
    if (mModelType != multimodal::ModelType::QWEN3_VL && mModelType != multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        return {};
    }
    if (mFlexEnabled)
    {
        std::vector<std::reference_wrapper<rt::Tensor const>> refs;
        refs.reserve(mFlexDeepstackFeatures.size());
        for (auto const& tensor : mFlexDeepstackFeatures)
        {
            refs.emplace_back(std::cref(tensor));
        }
        return refs;
    }

    // Build vector of references to individual tensors
    std::vector<std::reference_wrapper<rt::Tensor const>> refs;
    refs.reserve(mDeepstackFeatures.size());
    for (auto const& tensor : mDeepstackFeatures)
    {
        refs.emplace_back(std::cref(tensor));
    }
    return refs;
}

rt::Tensor& QwenViTRunner::getOutputEmbedding()
{
    return mFlexEnabled ? mFlexOutputEmbedding : mOutputEmbedding;
}

rt::OptionalInputTensor QwenViTRunner::getPositionIds()
{
    return std::cref(mMropePositionIdsHost);
}

rt::OptionalInputTensor QwenViTRunner::getRopeDeltas()
{
    return std::cref(mRopeDeltasHost);
}

bool QwenViTRunner::dumpDebugInputs(std::filesystem::path const& requestDir, cudaStream_t stream)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(requestDir, ec);
    if (ec)
    {
        LOG_ERROR(
            "QwenViTRunner::dumpDebugInputs(): Failed to create dump directory '%s': %s", requestDir.c_str(),
            ec.message().c_str());
        return false;
    }

    auto dumpTensor = [&](rt::Tensor const& tensor, std::string const& fileName) -> nlohmann::json {
        std::vector<uint8_t> hostData;
        if (!copyTensorToHostBytes(tensor, hostData, stream))
        {
            throw std::runtime_error("Failed to copy tensor to host");
        }
        std::ofstream out(requestDir / fileName, std::ios::binary);
        out.write(reinterpret_cast<char const*>(hostData.data()), static_cast<std::streamsize>(hostData.size()));
        return {{"file", fileName},
            {"shape", coordsToVector(tensor.getShape())},
            {"dtype", getDataTypeString(tensor.getDataType())},
            {"num_bytes", static_cast<int64_t>(hostData.size())}};
    };

    nlohmann::json meta;
    meta["visual_input"] = dumpTensor(mVitInput, "visual_input_request_0.bin");
    meta["rotary_pos_emb"] = dumpTensor(mRotaryPosEmb, "rotary_pos_emb_request_0.bin");
    meta["cu_seqlens"] = dumpTensor(mCuSeqlens, "cu_seqlens_request_0.bin");
    meta["max_seqlen_carrier"] = dumpTensor(mMaxSeqLenCarrier, "max_seqlen_carrier_request_0.bin");
    meta["image_grid_thw"] = mLastImageGridTHWs;

    if (mModelType == multimodal::ModelType::QWEN3_VL
        || mModelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        meta["fast_pos_embed_idx"] = dumpTensor(mFastPosEmbIdx, "fast_pos_embed_idx_request_0.bin");
        meta["fast_pos_embed_weight"] = dumpTensor(mFastPosEmbWeight, "fast_pos_embed_weight_request_0.bin");
    }

    std::ofstream metaFile(requestDir / "visual_inputs_request_0.json");
    metaFile << meta.dump(2);
    return true;
}

void QwenViTRunner::beginDebugInputsDump(std::filesystem::path const& requestDir)
{
    mActiveDebugDumpDir.clear();
    mActiveDebugImageIndex = 0;
    if (char const* env = std::getenv("EDGELLM_DUMP_VIT_PREPROCESS_STAGES"))
    {
        if (std::strcmp(env, "0") != 0 && std::strlen(env) > 0)
        {
            mActiveDebugDumpDir = requestDir;
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
