/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/alpamayoPostVlmRuntime.h"

#include "common/checkMacros.h"
#include "common/npyUtils.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

float elapsedMs(SteadyClock::time_point const& start, SteadyClock::time_point const& end)
{
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
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

bool copyTensorToHostBytes(rt::Tensor const& tensor, std::vector<uint8_t>& hostData, cudaStream_t stream)
{
    int64_t const volume = tensor.getShape().volume();
    if (volume < 0)
    {
        LOG_ERROR("Invalid tensor volume for %s", tensor.getName().c_str());
        return false;
    }

    size_t const elementSize = rt::utils::getTypeSize(tensor.getDataType());
    size_t const byteSize = static_cast<size_t>(volume) * elementSize;
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

bool copyKvCachePrefixDeviceToDevice(
    rt::Tensor const& srcTensor, rt::Tensor& dstTensor, int32_t activeLen, cudaStream_t stream)
{
    Coords const srcShape = srcTensor.getShape();
    Coords const dstShape = dstTensor.getShape();
    if (srcShape.getNumDims() != 6 || dstShape.getNumDims() != 6)
    {
        LOG_ERROR("FM KV device snapshot rank must be 6");
        return false;
    }
    if (srcTensor.getDeviceType() != rt::DeviceType::kGPU || dstTensor.getDeviceType() != rt::DeviceType::kGPU)
    {
        LOG_ERROR("FM KV device snapshot copy requires GPU tensors");
        return false;
    }
    if (srcTensor.getDataType() != dstTensor.getDataType())
    {
        LOG_ERROR("FM KV device snapshot dtype mismatch");
        return false;
    }
    if (srcShape[4] < dstShape[4] || activeLen < 0 || activeLen > dstShape[4])
    {
        LOG_ERROR("FM KV device snapshot prefix copy shape mismatch");
        return false;
    }

    size_t const elementSize = rt::utils::getTypeSize(srcTensor.getDataType());
    int64_t const outerCount = srcShape[0] * srcShape[1] * srcShape[2] * srcShape[3];
    int64_t const headDim = srcShape[5];
    size_t const srcPitch = static_cast<size_t>(srcShape[4]) * headDim * elementSize;
    size_t const dstPitch = static_cast<size_t>(dstShape[4]) * headDim * elementSize;
    size_t const copyWidth = static_cast<size_t>(activeLen) * headDim * elementSize;
    CUDA_CHECK(cudaMemsetAsync(dstTensor.rawPointer(), 0, static_cast<size_t>(dstShape.volume()) * elementSize, stream));
    if (copyWidth > 0)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(dstTensor.rawPointer(), dstPitch, srcTensor.rawPointer(), srcPitch, copyWidth,
            static_cast<size_t>(outerCount), cudaMemcpyDeviceToDevice, stream));
    }
    return true;
}

std::string fmtRequestSuffix(size_t requestIdx)
{
    std::ostringstream oss;
    oss << "request_" << requestIdx;
    return oss.str();
}

} // namespace

AlpamayoPostVlmRuntime::AlpamayoPostVlmRuntime(AlpamayoPostVlmOptions options)
    : mOptions(std::move(options))
{
    if (!mOptions.fmEngine.empty())
    {
        mFmRuntime = std::make_unique<AlpamayoFmRuntime>(mOptions.fmEngine);
    }
}

std::string AlpamayoPostVlmRuntime::removeRouteSpan(std::string const& text)
{
    std::string const startToken = "<|route_start|>";
    std::string const endToken = "<|route_end|>";
    size_t const startPos = text.find(startToken);
    if (startPos == std::string::npos)
    {
        return text;
    }
    size_t const endPos = text.find(endToken, startPos + startToken.size());
    check::check(endPos != std::string::npos, "Found route_start without route_end in request text.");
    return text.substr(0, startPos) + text.substr(endPos + endToken.size());
}

bool AlpamayoPostVlmRuntime::insertRouteSpan(std::string& text, std::string const& navText)
{
    std::string const historyEndToken = "<|traj_history_end|>";
    size_t const historyEndPos = text.find(historyEndToken);
    if (historyEndPos == std::string::npos)
    {
        return false;
    }

    text = removeRouteSpan(text);
    std::string const routeSpan = "<|route_start|>" + navText + "<|route_end|>";
    size_t const insertPos = historyEndPos + historyEndToken.size();
    text.insert(insertPos, routeSpan);
    return true;
}

bool AlpamayoPostVlmRuntime::appendAssistantReplayPrefix(
    LLMGenerationRequest::Request& request, std::string const& replayText)
{
    for (auto msgIt = request.messages.rbegin(); msgIt != request.messages.rend(); ++msgIt)
    {
        if (msgIt->role != "assistant")
        {
            continue;
        }
        for (auto& content : msgIt->contents)
        {
            if (content.type == "text")
            {
                content.content += replayText;
                return true;
            }
        }
    }
    return false;
}

std::optional<std::string> AlpamayoPostVlmRuntime::getEffectiveNavText(LLMGenerationRequest const& request)
{
    if (request.requests.empty())
    {
        return std::nullopt;
    }
    auto const& first = request.requests.front();
    if (!first.navText.has_value() || first.navText->empty())
    {
        return std::nullopt;
    }
    return first.navText;
}

float AlpamayoPostVlmRuntime::getEffectiveNavGuidanceWeight(LLMGenerationRequest const& request, float defaultWeight)
{
    if (request.requests.empty())
    {
        return defaultWeight;
    }
    auto const& first = request.requests.front();
    return first.navGuidanceWeight.value_or(defaultWeight);
}

std::optional<LLMGenerationRequest::ActionSpaceConstants> AlpamayoPostVlmRuntime::getActionSpaceConstants(
    LLMGenerationRequest const& request)
{
    if (request.requests.empty())
    {
        return std::nullopt;
    }
    return request.requests.front().actionSpaceConstants;
}

AlpamayoFmRunConfig AlpamayoPostVlmRuntime::getFmRunConfig(
    LLMGenerationRequest const& request, float defaultGuidanceWeight)
{
    AlpamayoFmRunConfig config;
    config.guidanceWeight = defaultGuidanceWeight;
    if (!request.requests.empty())
    {
        auto const& first = request.requests.front();
        config.seed = first.diffusionSeed.value_or(config.seed);
        config.numSteps = first.diffusionNumSteps.value_or(config.numSteps);
        config.guidanceWeight = first.navGuidanceWeight.value_or(config.guidanceWeight);
    }
    return config;
}

bool AlpamayoPostVlmRuntime::loadEgoHistory(LLMGenerationRequest const& request, std::vector<float>& egoHistoryXyz,
    std::vector<int64_t>& egoHistoryXyzShape, std::vector<float>& egoHistoryRot, std::vector<int64_t>& egoHistoryRotShape)
{
    if (request.requests.empty())
    {
        return false;
    }
    auto const& first = request.requests.front();
    check::check(!first.egoHistoryXYZNpy.empty(), "FM runtime requires ego_history_xyz_npy in the request");
    check::check(!first.egoHistoryRotNpy.empty(), "FM runtime requires ego_history_rot_npy in the request");

    auto const xyz = common::loadNpyFloat32(first.egoHistoryXYZNpy);
    auto const rot = common::loadNpyFloat32(first.egoHistoryRotNpy);
    egoHistoryXyz = xyz.data;
    egoHistoryXyzShape = xyz.shape;
    egoHistoryRot = rot.data;
    egoHistoryRotShape = rot.shape;
    return true;
}

nlohmann::json AlpamayoPostVlmRuntime::tensorToJson(std::vector<float> const& values, std::vector<int64_t> const& shape)
{
    return nlohmann::json{{"shape", shape}, {"data", values}};
}

int32_t AlpamayoPostVlmRuntime::readActiveKvLength(
    LLMInferenceRuntime& runtime, size_t activeBatchSize, cudaStream_t stream)
{
    rt::Tensor kvCacheLengths = runtime.getKVCacheLengths();
    std::vector<uint8_t> hostData;
    check::check(copyTensorToHostBytes(kvCacheLengths, hostData, stream), "Failed to copy KV-cache lengths");
    check::check(kvCacheLengths.getDataType() == nvinfer1::DataType::kINT32, "KV-cache lengths must be INT32");
    size_t const expected = kvCacheLengths.getShape().volume() * sizeof(int32_t);
    check::check(hostData.size() == expected, "KV-cache lengths host byte size mismatch");
    auto const* values = reinterpret_cast<int32_t const*>(hostData.data());
    check::check(activeBatchSize > 0, "activeBatchSize must be positive");
    return values[0];
}

std::optional<AlpamayoFmBranchSnapshot> AlpamayoPostVlmRuntime::captureFmSnapshot(
    LLMInferenceRuntime& runtime, size_t activeBatchSize, cudaStream_t stream, bool requireOwnedDeviceCopy)
{
    rt::Tensor kvCacheBuffer = runtime.getKVCacheBuffer();
    rt::OptionalInputTensor ropeDeltas = runtime.getRopeDeltas();
    check::check(ropeDeltas.has_value(), "FM snapshot capture requires rope_deltas");

    AlpamayoFmBranchSnapshot snapshot;
    snapshot.kvCacheShape = kvCacheBuffer.getShape();
    snapshot.kvCacheDataType = kvCacheBuffer.getDataType();
    snapshot.activeLen = readActiveKvLength(runtime, activeBatchSize, stream);

    std::vector<uint8_t> ropeBytes;
    check::check(copyTensorToHostBytes(ropeDeltas.value().get(), ropeBytes, stream), "Failed to capture FM rope_deltas");
    check::check(ropeDeltas.value().get().getDataType() == nvinfer1::DataType::kINT64,
        "FM rope_deltas must be INT64");
    check::check(ropeBytes.size() >= sizeof(int64_t), "FM rope_deltas payload is too small");
    snapshot.ropeDelta = *reinterpret_cast<int64_t const*>(ropeBytes.data());

    bool const canUseDeviceSnapshot = mFmRuntime != nullptr && kvCacheBuffer.getDeviceType() == rt::DeviceType::kGPU
        && kvCacheBuffer.getDataType() == nvinfer1::DataType::kHALF
        && mFmRuntime->kvCacheDataType() == nvinfer1::DataType::kHALF;
    if (canUseDeviceSnapshot)
    {
        if (requireOwnedDeviceCopy)
        {
            int32_t const targetSeqLen = mFmRuntime->maxSeqLen();
            check::check(targetSeqLen > 0, "FM target sequence length must be positive");
            check::check(snapshot.activeLen <= targetSeqLen,
                "FM active KV length exceeds FM engine max sequence length during device snapshot capture");
            Coords targetShape = snapshot.kvCacheShape;
            targetShape[4] = targetSeqLen;
            snapshot.kvCacheTensor = rt::Tensor(targetShape, rt::DeviceType::kGPU, snapshot.kvCacheDataType, "fm_kv_snapshot");
            if (targetShape == kvCacheBuffer.getShape())
            {
                CUDA_CHECK(cudaMemcpyAsync(snapshot.kvCacheTensor.rawPointer(), kvCacheBuffer.rawPointer(),
                    snapshot.kvCacheTensor.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
            }
            else
            {
                check::check(copyKvCachePrefixDeviceToDevice(kvCacheBuffer, snapshot.kvCacheTensor, snapshot.activeLen, stream),
                    "Failed to capture FM KV device prefix snapshot");
            }
            snapshot.kvCacheShape = targetShape;
        }
        else
        {
            snapshot.kvCacheTensor = std::move(kvCacheBuffer);
        }
        return snapshot;
    }

    check::check(copyTensorToHostBytes(kvCacheBuffer, snapshot.kvCacheBytes, stream), "Failed to capture FM KV snapshot");
    return snapshot;
}

bool AlpamayoPostVlmRuntime::dumpKvSnapshot(LLMInferenceRuntime& runtime, std::filesystem::path const& outputDir,
    size_t requestIdx, size_t activeBatchSize, cudaStream_t stream)
{
    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);
    if (ec)
    {
        LOG_ERROR("Failed to create KV-cache output directory '%s': %s", outputDir.c_str(), ec.message().c_str());
        return false;
    }

    rt::Tensor kvCacheBuffer = runtime.getKVCacheBuffer();
    rt::Tensor kvCacheLengths = runtime.getKVCacheLengths();
    rt::OptionalInputTensor positionIds = runtime.getPositionIds();
    rt::OptionalInputTensor ropeDeltas = runtime.getRopeDeltas();

    std::vector<uint8_t> kvCacheHostData;
    if (!copyTensorToHostBytes(kvCacheBuffer, kvCacheHostData, stream))
    {
        LOG_ERROR("Failed to copy KV-cache buffer to host for request %zu", requestIdx);
        return false;
    }

    std::vector<uint8_t> kvCacheLengthsHostData;
    if (!copyTensorToHostBytes(kvCacheLengths, kvCacheLengthsHostData, stream))
    {
        LOG_ERROR("Failed to copy KV-cache lengths to host for request %zu", requestIdx);
        return false;
    }

    std::vector<uint8_t> positionIdsHostData;
    if (positionIds.has_value() && !copyTensorToHostBytes(positionIds.value().get(), positionIdsHostData, stream))
    {
        LOG_ERROR("Failed to copy position_ids to host for request %zu", requestIdx);
        return false;
    }

    std::vector<uint8_t> ropeDeltasHostData;
    if (ropeDeltas.has_value() && !copyTensorToHostBytes(ropeDeltas.value().get(), ropeDeltasHostData, stream))
    {
        LOG_ERROR("Failed to copy rope_deltas to host for request %zu", requestIdx);
        return false;
    }

    std::string const requestSuffix = fmtRequestSuffix(requestIdx);
    auto const kvCacheFilePath = outputDir / ("kv_cache_" + requestSuffix + ".bin");
    auto const kvCacheLengthsFilePath = outputDir / ("kv_cache_lengths_" + requestSuffix + ".bin");
    auto const positionIdsFilePath = outputDir / ("position_ids_" + requestSuffix + ".bin");
    auto const ropeDeltasFilePath = outputDir / ("rope_deltas_" + requestSuffix + ".bin");
    auto const metaFilePath = outputDir / ("kv_cache_" + requestSuffix + ".json");

    {
        std::ofstream out(kvCacheFilePath, std::ios::binary);
        out.write(reinterpret_cast<char const*>(kvCacheHostData.data()), static_cast<std::streamsize>(kvCacheHostData.size()));
        if (!out.good())
        {
            LOG_ERROR("Failed to write KV-cache file: %s", kvCacheFilePath.c_str());
            return false;
        }
    }
    {
        std::ofstream out(kvCacheLengthsFilePath, std::ios::binary);
        out.write(reinterpret_cast<char const*>(kvCacheLengthsHostData.data()),
            static_cast<std::streamsize>(kvCacheLengthsHostData.size()));
        if (!out.good())
        {
            LOG_ERROR("Failed to write KV-cache lengths file: %s", kvCacheLengthsFilePath.c_str());
            return false;
        }
    }
    if (positionIds.has_value())
    {
        std::ofstream out(positionIdsFilePath, std::ios::binary);
        out.write(reinterpret_cast<char const*>(positionIdsHostData.data()),
            static_cast<std::streamsize>(positionIdsHostData.size()));
    }
    if (ropeDeltas.has_value())
    {
        std::ofstream out(ropeDeltasFilePath, std::ios::binary);
        out.write(
            reinterpret_cast<char const*>(ropeDeltasHostData.data()), static_cast<std::streamsize>(ropeDeltasHostData.size()));
    }

    std::vector<int32_t> activeLengths;
    if (kvCacheLengths.getDataType() == nvinfer1::DataType::kINT32)
    {
        size_t const count = kvCacheLengths.getShape().volume();
        activeLengths.resize(count);
        std::memcpy(activeLengths.data(), kvCacheLengthsHostData.data(), count * sizeof(int32_t));
    }

    nlohmann::json metadata;
    metadata["request_idx"] = requestIdx;
    metadata["layout"] = "[numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]";
    metadata["kv_cache"] = {{"file", kvCacheFilePath.filename().string()},
        {"shape", coordsToVector(kvCacheBuffer.getShape())},
        {"dtype", getDataTypeString(kvCacheBuffer.getDataType())},
        {"num_bytes", kvCacheHostData.size()}};
    metadata["kv_cache_lengths"] = {{"file", kvCacheLengthsFilePath.filename().string()},
        {"shape", coordsToVector(kvCacheLengths.getShape())},
        {"dtype", getDataTypeString(kvCacheLengths.getDataType())},
        {"active_batch_size", activeBatchSize},
        {"active_values", std::vector<int32_t>(activeLengths.begin(), activeLengths.begin() + std::min(activeBatchSize, activeLengths.size()))}};
    if (positionIds.has_value())
    {
        metadata["position_ids"] = {{"file", positionIdsFilePath.filename().string()},
            {"shape", coordsToVector(positionIds.value().get().getShape())},
            {"dtype", getDataTypeString(positionIds.value().get().getDataType())},
            {"num_bytes", positionIdsHostData.size()}};
    }
    if (ropeDeltas.has_value())
    {
        metadata["rope_deltas"] = {{"file", ropeDeltasFilePath.filename().string()},
            {"shape", coordsToVector(ropeDeltas.value().get().getShape())},
            {"dtype", getDataTypeString(ropeDeltas.value().get().getDataType())},
            {"num_bytes", ropeDeltasHostData.size()}};
    }

    std::ofstream meta(metaFilePath);
    meta << metadata.dump(2);
    return meta.good();
}

AlpamayoPostVlmRuntime::BranchArtifacts AlpamayoPostVlmRuntime::runGuidedPass(LLMInferenceRuntime& runtime,
    LLMGenerationRequest const& request, std::string const& navText, size_t requestIdx, cudaStream_t stream)
{
    BranchArtifacts out;
    out.request = request;
    out.request.continueFinalMessage = true;
    out.request.addGenerationPrompt = false;
    if (mOptions.usePrefillKvForFm)
    {
        out.request.maxGenerateLength = 0;
    }

    check::check(out.request.requests.size() == 1,
        "AlpamayoPostVlmRuntime currently supports batch_size=1 only during native nav bring-up.");

    auto& req = out.request.requests[0];
    if (!navText.empty())
    {
        bool inserted = false;
        for (auto& msg : req.messages)
        {
            if (msg.role != "user")
            {
                continue;
            }
            for (auto& content : msg.contents)
            {
                if (content.type == "text")
                {
                    inserted = insertRouteSpan(content.content, navText) || inserted;
                }
            }
        }
        check::check(inserted, "Failed to insert route span into guided request text.");
    }

    check::check(runtime.handleRequest(out.request, out.response, stream), "Guided VLM pass failed");
    out.kvActiveLen = readActiveKvLength(runtime, out.request.requests.size(), stream);
    if (!mOptions.usePrefillKvForFm)
    {
        out.decodedPrefixText = runtime.decodeTokenIds(out.response.outputIds.at(0), false);
    }
    if (mFmRuntime)
    {
        bool const requireOwnedDeviceCopy = !navText.empty() && mOptions.enableNavCfg;
        out.fmSnapshot = captureFmSnapshot(runtime, out.request.requests.size(), stream, requireOwnedDeviceCopy);
    }
    if (mOptions.usePrefillKvForFm)
    {
        out.response.outputIds.assign(out.request.requests.size(), {});
        out.response.outputTexts.assign(out.request.requests.size(), {});
    }
    if (mOptions.dumpNavDualCache)
    {
        out.kvDir = mOptions.navCacheOutputDir / fmtRequestSuffix(requestIdx) / "guided";
        check::check(dumpKvSnapshot(runtime, out.kvDir, 0, out.request.requests.size(), stream),
            "Failed to dump guided KV snapshot");
    }
    return out;
}

AlpamayoPostVlmRuntime::BranchArtifacts AlpamayoPostVlmRuntime::runUnguidedReplayPass(LLMInferenceRuntime& runtime,
    LLMGenerationRequest const& request, std::string const& guidedPrefixText, size_t requestIdx, cudaStream_t stream)
{
    BranchArtifacts out;
    out.request = request;
    out.request.continueFinalMessage = true;
    out.request.addGenerationPrompt = false;
    out.request.maxGenerateLength = 0;

    check::check(out.request.requests.size() == 1,
        "AlpamayoPostVlmRuntime currently supports batch_size=1 only during native nav bring-up.");

    auto& req = out.request.requests[0];
    for (auto& msg : req.messages)
    {
        if (msg.role != "user")
        {
            continue;
        }
        for (auto& content : msg.contents)
        {
            if (content.type == "text")
            {
                content.content = removeRouteSpan(content.content);
            }
        }
    }
    check::check(appendAssistantReplayPrefix(req, guidedPrefixText), "Failed to append guided replay prefix");

    check::check(runtime.handleRequest(out.request, out.response, stream), "Unguided replay VLM pass failed");
    out.kvActiveLen = readActiveKvLength(runtime, out.request.requests.size(), stream);
    if (mFmRuntime)
    {
        out.fmSnapshot = captureFmSnapshot(runtime, out.request.requests.size(), stream, false);
    }
    if (mOptions.usePrefillKvForFm)
    {
        out.response.outputIds.assign(out.request.requests.size(), {});
        out.response.outputTexts.assign(out.request.requests.size(), {});
    }
    if (mOptions.dumpNavDualCache)
    {
        out.kvDir = mOptions.navCacheOutputDir / fmtRequestSuffix(requestIdx) / "unguided";
        check::check(dumpKvSnapshot(runtime, out.kvDir, 0, out.request.requests.size(), stream),
            "Failed to dump unguided KV snapshot");
    }
    return out;
}

bool AlpamayoPostVlmRuntime::handleRequest(LLMInferenceRuntime& runtime, LLMGenerationRequest const& request,
    LLMGenerationResponse& response, nlohmann::json& responseJson, cudaStream_t stream)
{
    auto const totalStart = SteadyClock::now();
    auto const navText = getEffectiveNavText(request);
    auto const navWeight = getEffectiveNavGuidanceWeight(request, mOptions.defaultNavGuidanceWeight);
    auto const fmConstants = getActionSpaceConstants(request);
    auto const fmConfig = getFmRunConfig(request, navWeight);

    auto const guidedStart = SteadyClock::now();
    BranchArtifacts guided = runGuidedPass(runtime, request, navText.value_or(""), 0, stream);
    float const guidedPassMs = ::trt_edgellm::rt::elapsedMs(guidedStart, SteadyClock::now());
    response = guided.response;
    std::optional<BranchArtifacts> unguided;
    float unguidedPassMs = 0.0F;

    responseJson["mode"] = "alpamayo_post_vlm";
    responseJson["fm_engine"] = mOptions.fmEngine;
    responseJson["fm_kv_source"] = mOptions.usePrefillKvForFm ? "backbone_prefill" : "post_decode";
    responseJson["guided"] = {
        {"output_text", guided.response.outputTexts.empty() ? std::string{} : guided.response.outputTexts[0]},
        {"output_token_ids", guided.response.outputIds.empty() ? std::vector<int32_t>{} : guided.response.outputIds[0]},
        {"kv_active_len", guided.kvActiveLen},
        {"used_prefill_kv_for_fm", mOptions.usePrefillKvForFm},
        {"formatted_system_prompt",
            guided.request.formattedRequests.empty() ? std::string{}
                                                     : guided.request.formattedRequests[0].formattedSystemPrompt},
        {"formatted_complete_request",
            guided.request.formattedRequests.empty() ? std::string{}
                                                     : guided.request.formattedRequests[0].formattedCompleteRequest},
        {"decoded_prefix_text_tail",
            guided.decodedPrefixText.size() > 240 ? guided.decodedPrefixText.substr(guided.decodedPrefixText.size() - 240)
                                                  : guided.decodedPrefixText}};

    if (navText.has_value() && mOptions.enableNavCfg)
    {
        auto const unguidedStart = SteadyClock::now();
        unguided = runUnguidedReplayPass(runtime, request, guided.decodedPrefixText, 0, stream);
        unguidedPassMs = ::trt_edgellm::rt::elapsedMs(unguidedStart, SteadyClock::now());
        responseJson["nav"] = {{"enabled", true},
            {"nav_text", *navText},
            {"guidance_weight", navWeight}};
        responseJson["unguided"] = {{"kv_active_len", unguided->kvActiveLen},
            {"formatted_system_prompt",
                unguided->request.formattedRequests.empty() ? std::string{}
                                                            : unguided->request.formattedRequests[0].formattedSystemPrompt},
            {"formatted_complete_request",
                unguided->request.formattedRequests.empty() ? std::string{}
                                                            : unguided->request.formattedRequests[0].formattedCompleteRequest},
            {"output_token_ids",
                unguided->response.outputIds.empty() ? std::vector<int32_t>{} : unguided->response.outputIds[0]}};
    }
    else
    {
        responseJson["nav"] = {{"enabled", false}};
    }

    if (!mFmRuntime)
    {
        responseJson["fm_status"] = "not_configured";
        return true;
    }

    check::check(fmConstants.has_value(),
        "FM runtime requested but no action_space_constants were provided in the request. "
        "This is required to avoid silent decode-default mismatches.");
    check::check(guided.fmSnapshot.has_value(), "Guided FM snapshot is missing");

    std::vector<float> egoHistoryXyz;
    std::vector<int64_t> egoHistoryXyzShape;
    std::vector<float> egoHistoryRot;
    std::vector<int64_t> egoHistoryRotShape;
    auto const egoLoadStart = SteadyClock::now();
    check::check(loadEgoHistory(request, egoHistoryXyz, egoHistoryXyzShape, egoHistoryRot, egoHistoryRotShape),
        "Failed to load ego history for FM runtime");
    float const egoLoadMs = ::trt_edgellm::rt::elapsedMs(egoLoadStart, SteadyClock::now());

    AlpamayoFmRunResult fmResult;
    bool ok = false;
    auto const fmStart = SteadyClock::now();
    if (navText.has_value() && mOptions.enableNavCfg && unguided.has_value())
    {
        check::check(unguided->fmSnapshot.has_value(), "Unguided FM snapshot is missing");
        ok = mFmRuntime->runNavCfg(*guided.fmSnapshot, *unguided->fmSnapshot, *fmConstants, egoHistoryXyz, egoHistoryXyzShape,
            egoHistoryRot, egoHistoryRotShape, fmConfig, fmResult, stream);
        responseJson["fm_mode"] = "nav_cfg";
    }
    else
    {
        ok = mFmRuntime->runNoNav(*guided.fmSnapshot, *fmConstants, egoHistoryXyz, egoHistoryXyzShape, egoHistoryRot,
            egoHistoryRotShape, fmConfig, fmResult, stream);
        responseJson["fm_mode"] = "single_branch";
    }
    check::check(ok, "Native FM runtime execution failed");
    float const fmWallMs = ::trt_edgellm::rt::elapsedMs(fmStart, SteadyClock::now());

    responseJson["fm_status"] = "completed";
    responseJson["fm"] = {
        {"engine_path", mFmRuntime->enginePath()},
        {"seed", fmConfig.seed},
        {"num_steps", fmConfig.numSteps},
        {"guidance_weight", fmConfig.guidanceWeight},
        {"x0", tensorToJson(fmResult.x0, {1, fmResult.horizon, fmResult.actionDim})},
        {"x_final", tensorToJson(fmResult.xFinal, {1, fmResult.horizon, fmResult.actionDim})},
        {"pred_xyz", tensorToJson(fmResult.predXyz, {1, fmResult.horizon, 3})},
        {"pred_rot", tensorToJson(fmResult.predRot, {1, fmResult.horizon, 3, 3})},
        {"timing",
            {{"branch_prepare_ms", fmResult.timing.branchPrepareMs},
                {"kv_convert_ms", fmResult.timing.kvConvertMs},
                {"kv_alloc_ms", fmResult.timing.kvAllocMs},
                {"kv_copy_ms", fmResult.timing.kvCopyMs},
                {"mask_build_ms", fmResult.timing.maskBuildMs},
                {"mask_alloc_ms", fmResult.timing.maskAllocMs},
                {"mask_copy_ms", fmResult.timing.maskCopyMs},
                {"position_build_ms", fmResult.timing.positionBuildMs},
                {"position_alloc_ms", fmResult.timing.positionAllocMs},
                {"position_copy_ms", fmResult.timing.positionCopyMs},
                {"x0_init_ms", fmResult.timing.x0InitMs},
                {"engine_step_total_ms", fmResult.timing.engineStepTotalMs},
                {"engine_step_avg_ms", fmResult.timing.engineStepAvgMs},
                {"decode_postprocess_ms", fmResult.timing.decodePostprocessMs},
                {"total_ms", fmResult.timing.totalMs},
                {"wall_ms", fmWallMs},
                {"num_steps", fmResult.timing.numSteps},
                {"num_branches", fmResult.timing.numBranches}}},
        {"action_space_constants",
            {{"accel_mean", fmConstants->accelMean},
                {"accel_std", fmConstants->accelStd},
                {"curvature_mean", fmConstants->curvatureMean},
                {"curvature_std", fmConstants->curvatureStd},
                {"dt_value", fmConstants->dtValue},
                {"v_lambda", fmConstants->vLambda},
                {"v_ridge", fmConstants->vRidge}}}};
    responseJson["timing"] = {{"guided_pass_ms", guidedPassMs},
        {"unguided_replay_ms", unguidedPassMs},
        {"ego_history_load_ms", egoLoadMs},
        {"fm_wall_ms", fmWallMs},
        {"total_post_vlm_ms", ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now())}};

    return true;
}

} // namespace rt
} // namespace trt_edgellm
