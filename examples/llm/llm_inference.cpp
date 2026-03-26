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

#include "common/checkMacros.h"
#include "common/deltaTrajectoryTokenizer.h"
#include "common/inputLimits.h"
#include "common/npyUtils.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/alpamayoPostVlmRuntime.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

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

std::vector<traj::DeltaTrajectoryTokenizer::Vec3> loadSingleTrajectoryXYZ(common::NpyArrayFloat32 const& array)
{
    check::check(array.shape.size() >= 2, "egoHistoryXYZ must have at least 2 dimensions.");
    check::check(array.shape.back() == 3, "egoHistoryXYZ last dimension must be 3.");
    int64_t const sequenceLength = array.shape[array.shape.size() - 2];
    int64_t const batchCount = static_cast<int64_t>(array.data.size()) / (sequenceLength * 3);
    check::check(batchCount == 1, "Only single-trajectory egoHistoryXYZ arrays are supported.");

    std::vector<traj::DeltaTrajectoryTokenizer::Vec3> xyz(static_cast<size_t>(sequenceLength));
    for (int64_t t = 0; t < sequenceLength; ++t)
    {
        size_t const base = static_cast<size_t>(t * 3);
        xyz[static_cast<size_t>(t)] = {array.data[base], array.data[base + 1], array.data[base + 2]};
    }
    return xyz;
}

std::vector<traj::DeltaTrajectoryTokenizer::Mat3> loadSingleTrajectoryRot(common::NpyArrayFloat32 const& array)
{
    check::check(array.shape.size() >= 3, "egoHistoryRot must have at least 3 dimensions.");
    check::check(array.shape[array.shape.size() - 2] == 3 && array.shape.back() == 3,
        "egoHistoryRot trailing dimensions must be 3x3.");
    int64_t const sequenceLength = array.shape[array.shape.size() - 3];
    int64_t const batchCount = static_cast<int64_t>(array.data.size()) / (sequenceLength * 9);
    check::check(batchCount == 1, "Only single-trajectory egoHistoryRot arrays are supported.");

    std::vector<traj::DeltaTrajectoryTokenizer::Mat3> rot(static_cast<size_t>(sequenceLength));
    for (int64_t t = 0; t < sequenceLength; ++t)
    {
        size_t const base = static_cast<size_t>(t * 9);
        rot[static_cast<size_t>(t)] = {{{array.data[base], array.data[base + 1], array.data[base + 2]},
            {array.data[base + 3], array.data[base + 4], array.data[base + 5]},
            {array.data[base + 6], array.data[base + 7], array.data[base + 8]}}};
    }
    return rot;
}

bool replaceTrajectoryText(std::string& text, std::string const& replacement)
{
    std::string const startToken = "<|traj_history_start|>";
    std::string const endToken = "<|traj_history_end|>";
    size_t const startPos = text.find(startToken);
    if (startPos == std::string::npos)
    {
        return false;
    }

    size_t const endPos = text.find(endToken, startPos + startToken.size());
    check::check(endPos != std::string::npos, "Found traj_history_start without traj_history_end in input text.");

    text = text.substr(0, startPos) + replacement + text.substr(endPos + endToken.size());
    return true;
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

bool dumpKVCacheSnapshot(rt::LLMInferenceRuntime& runtime, std::filesystem::path const& outputDir, size_t requestIdx,
    size_t activeBatchSize, cudaStream_t stream)
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

    std::string const requestSuffix = "request_" + std::to_string(requestIdx);
    std::filesystem::path const kvCacheFilePath = outputDir / ("kv_cache_" + requestSuffix + ".bin");
    std::filesystem::path const kvCacheLengthsFilePath = outputDir / ("kv_cache_lengths_" + requestSuffix + ".bin");
    std::filesystem::path const positionIdsFilePath = outputDir / ("position_ids_" + requestSuffix + ".bin");
    std::filesystem::path const ropeDeltasFilePath = outputDir / ("rope_deltas_" + requestSuffix + ".bin");
    std::filesystem::path const metaFilePath = outputDir / ("kv_cache_" + requestSuffix + ".json");

    {
        std::ofstream kvCacheFile(kvCacheFilePath, std::ios::binary);
        if (!kvCacheFile.is_open())
        {
            LOG_ERROR("Failed to open KV-cache dump file: %s", kvCacheFilePath.c_str());
            return false;
        }
        kvCacheFile.write(reinterpret_cast<char const*>(kvCacheHostData.data()),
            static_cast<std::streamsize>(kvCacheHostData.size()));
        if (!kvCacheFile.good())
        {
            LOG_ERROR("Failed to write KV-cache dump file: %s", kvCacheFilePath.c_str());
            return false;
        }
    }

    {
        std::ofstream kvCacheLengthsFile(kvCacheLengthsFilePath, std::ios::binary);
        if (!kvCacheLengthsFile.is_open())
        {
            LOG_ERROR("Failed to open KV-cache lengths dump file: %s", kvCacheLengthsFilePath.c_str());
            return false;
        }
        kvCacheLengthsFile.write(reinterpret_cast<char const*>(kvCacheLengthsHostData.data()),
            static_cast<std::streamsize>(kvCacheLengthsHostData.size()));
        if (!kvCacheLengthsFile.good())
        {
            LOG_ERROR("Failed to write KV-cache lengths dump file: %s", kvCacheLengthsFilePath.c_str());
            return false;
        }
    }

    if (positionIds.has_value())
    {
        std::ofstream positionIdsFile(positionIdsFilePath, std::ios::binary);
        if (!positionIdsFile.is_open())
        {
            LOG_ERROR("Failed to open position_ids dump file: %s", positionIdsFilePath.c_str());
            return false;
        }
        positionIdsFile.write(
            reinterpret_cast<char const*>(positionIdsHostData.data()), static_cast<std::streamsize>(positionIdsHostData.size()));
        if (!positionIdsFile.good())
        {
            LOG_ERROR("Failed to write position_ids dump file: %s", positionIdsFilePath.c_str());
            return false;
        }
    }

    if (ropeDeltas.has_value())
    {
        std::ofstream ropeDeltasFile(ropeDeltasFilePath, std::ios::binary);
        if (!ropeDeltasFile.is_open())
        {
            LOG_ERROR("Failed to open rope_deltas dump file: %s", ropeDeltasFilePath.c_str());
            return false;
        }
        ropeDeltasFile.write(
            reinterpret_cast<char const*>(ropeDeltasHostData.data()), static_cast<std::streamsize>(ropeDeltasHostData.size()));
        if (!ropeDeltasFile.good())
        {
            LOG_ERROR("Failed to write rope_deltas dump file: %s", ropeDeltasFilePath.c_str());
            return false;
        }
    }

    std::vector<int32_t> kvCacheLengthsValues;
    if (kvCacheLengths.getDataType() == nvinfer1::DataType::kINT32)
    {
        int64_t const lengthsVolume = kvCacheLengths.getShape().volume();
        kvCacheLengthsValues.resize(static_cast<size_t>(lengthsVolume), 0);
        size_t const expectedBytes = kvCacheLengthsValues.size() * sizeof(int32_t);
        if (expectedBytes == kvCacheLengthsHostData.size())
        {
            std::memcpy(kvCacheLengthsValues.data(), kvCacheLengthsHostData.data(), expectedBytes);
        }
    }

    size_t const activeCount = std::min(activeBatchSize, kvCacheLengthsValues.size());
    std::vector<int32_t> activeLengths(kvCacheLengthsValues.begin(), kvCacheLengthsValues.begin() + activeCount);

    std::vector<int64_t> ropeDeltasValues;
    if (ropeDeltas.has_value() && ropeDeltas.value().get().getDataType() == nvinfer1::DataType::kINT64)
    {
        int64_t const ropeDeltasVolume = ropeDeltas.value().get().getShape().volume();
        ropeDeltasValues.resize(static_cast<size_t>(ropeDeltasVolume), 0);
        size_t const expectedBytes = ropeDeltasValues.size() * sizeof(int64_t);
        if (expectedBytes == ropeDeltasHostData.size())
        {
            std::memcpy(ropeDeltasValues.data(), ropeDeltasHostData.data(), expectedBytes);
        }
    }

    Json metadata;
    metadata["request_idx"] = requestIdx;
    metadata["layout"] = "[numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]";
    metadata["kv_cache"] = {{"file", kvCacheFilePath.filename().string()},
        {"shape", coordsToVector(kvCacheBuffer.getShape())},
        {"dtype", getDataTypeString(kvCacheBuffer.getDataType())},
        {"num_bytes", kvCacheHostData.size()}};
    metadata["kv_cache_lengths"] = {{"file", kvCacheLengthsFilePath.filename().string()},
        {"shape", coordsToVector(kvCacheLengths.getShape())},
        {"dtype", getDataTypeString(kvCacheLengths.getDataType())},
        {"num_bytes", kvCacheLengthsHostData.size()},
        {"active_batch_size", activeBatchSize},
        {"active_values", activeLengths},
        {"all_values", kvCacheLengthsValues}};
    if (positionIds.has_value())
    {
        metadata["position_ids"] = {{"file", positionIdsFilePath.filename().string()},
            {"shape", coordsToVector(positionIds.value().get().getShape())},
            {"dtype", getDataTypeString(positionIds.value().get().getDataType())},
            {"num_bytes", positionIdsHostData.size()},
            {"layout", "[batchSize, 3, maxPositionEmbeddings]"}};
    }
    if (ropeDeltas.has_value())
    {
        metadata["rope_deltas"] = {{"file", ropeDeltasFilePath.filename().string()},
            {"shape", coordsToVector(ropeDeltas.value().get().getShape())},
            {"dtype", getDataTypeString(ropeDeltas.value().get().getDataType())},
            {"num_bytes", ropeDeltasHostData.size()},
            {"values", ropeDeltasValues},
            {"formula", "max(active_position_ids)+1-input_length"}};
    }

    std::ofstream metaFile(metaFilePath);
    if (!metaFile.is_open())
    {
        LOG_ERROR("Failed to open KV-cache metadata file: %s", metaFilePath.c_str());
        return false;
    }
    metaFile << metadata.dump(2);
    if (!metaFile.good())
    {
        LOG_ERROR("Failed to write KV-cache metadata file: %s", metaFilePath.c_str());
        return false;
    }

    LOG_INFO("KV-cache snapshot exported for request %zu -> %s", requestIdx, metaFilePath.c_str());
    return true;
}

} // namespace

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    DUMP_KV_CACHE = 916,
    KV_CACHE_OUTPUT_DIR = 917,
    DUMP_PREFILL_INPUTS = 918,
    PREFILL_INPUT_OUTPUT_DIR = 919,
    EAGLE = 910,
    EAGLE_DRAFT_TOP_K = 911,
    EAGLE_DRAFT_STEP = 912,
    EAGLE_VERIFY_TREE_SIZE = 913,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915,
    FM_ENGINE = 920,
    ALPAMAYO_POST_VLM_RUNTIME = 921,
    ALPAMAYO_NAV_CFG = 922,
    DEFAULT_NAV_GUIDANCE_WEIGHT = 923,
    DUMP_NAV_DUAL_CACHE = 924,
    NAV_CACHE_OUTPUT_DIR = 925,
    PERSISTENT_SERVER = 926
};

// Struct to hold Eagle-specific arguments for speculative decoding
struct EagleArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // This controls the branching factor at each level of the draft tree.
    int32_t draftTopK{10};

    // Number of drafting steps to perform with the draft model.
    // Each step extends the draft tree by one more level.
    int32_t draftStep{6};

    // Number of tokens to select from the complete draft tree for base model verification.
    // The total draft tree size is: 1 + draftTopK + (draftStep - 1) * draftTopK * draftTopK
    // This parameter should be <= total draft tree size for optimal performance.
    int32_t verifyTreeSize{60};
};

struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    bool dumpKVCache{false};
    std::string kvCacheOutputDir{"./output/kv_cache"};
    bool dumpPrefillInputs{false};
    std::string prefillInputOutputDir{"./output/prefill_inputs"};
    std::string fmEngine{""};
    bool alpamayoPostVlmRuntime{false};
    bool alpamayoNavCfg{false};
    float defaultNavGuidanceWeight{3.0F};
    bool dumpNavDualCache{false};
    std::string navCacheOutputDir{"./output/nav_dual_cache"};
    bool persistentServer{false};
    // Override parameters (only batchSize and maxGenerateLength can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    EagleArgs eagleArgs;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--dumpKVCache] [--kvCacheOutputDir=<path>] "
                 "[--dumpPrefillInputs] [--prefillInputOutputDir=<path>] [--batchSize=<number>] "
                 "[--maxGenerateLength=<number>] [--fmEngine=<path>] [--alpamayoPostVlmRuntime] "
                 "[--alpamayoNavCfg] [--defaultNavGuidanceWeight=<number>] [--dumpNavDualCache] "
                 "[--navCacheOutputDir=<path>] [--persistentServer] [--eagle] "
                 "[--eagleDraftTopK=<number>] [--eagleDraftStep=<number>] "
                 "[--eagleVerifyTreeSize=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --dumpKVCache             Dump KV-cache snapshot after each successful request" << std::endl;
    std::cerr << "  --kvCacheOutputDir        Directory for KV-cache snapshot artifacts (default: ./output/kv_cache)"
              << std::endl;
    std::cerr << "  --dumpPrefillInputs       Dump prepared prefill inputs (input_ids, inputs_embeds, deepstack embeds)"
              << std::endl;
    std::cerr << "  --prefillInputOutputDir   Directory for prefill input artifacts (default: ./output/prefill_inputs)"
              << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "  --fmEngine                Path to FM TRT one-step engine (for Alpamayo post-VLM runtime)"
              << std::endl;
    std::cerr << "  --alpamayoPostVlmRuntime  Enable Alpamayo post-VLM orchestration path" << std::endl;
    std::cerr << "  --alpamayoNavCfg          Enable nav-aware guided/unguided dual-cache path" << std::endl;
    std::cerr << "  --defaultNavGuidanceWeight  Default nav CFG guidance weight (default: 3.0)" << std::endl;
    std::cerr << "  --dumpNavDualCache        Dump guided/unguided KV-cache artifacts" << std::endl;
    std::cerr << "  --navCacheOutputDir       Directory for nav dual-cache artifacts" << std::endl;
    std::cerr << "  --persistentServer        Keep engines loaded and read newline-delimited JSON commands from stdin"
              << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr << "  --eagle                   Enable Eagle speculative decoding mode" << std::endl;
    std::cerr << "  --eagleDraftTopK          Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            Controls branching factor at each draft tree level" << std::endl;
    std::cerr << "  --eagleDraftStep          Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the draft tree by one more level" << std::endl;
    std::cerr << "  --eagleVerifyTreeSize     Number of tokens for base model verification (default: 60)" << std::endl;
    std::cerr << "                            Total draft tree size: 1 + topK + (step-1) * topK^2" << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, LLMInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"dumpKVCache", no_argument, 0, LLMInferenceOptionId::DUMP_KV_CACHE},
        {"kvCacheOutputDir", required_argument, 0, LLMInferenceOptionId::KV_CACHE_OUTPUT_DIR},
        {"dumpPrefillInputs", no_argument, 0, LLMInferenceOptionId::DUMP_PREFILL_INPUTS},
        {"prefillInputOutputDir", required_argument, 0, LLMInferenceOptionId::PREFILL_INPUT_OUTPUT_DIR},
        {"fmEngine", required_argument, 0, LLMInferenceOptionId::FM_ENGINE},
        {"alpamayoPostVlmRuntime", no_argument, 0, LLMInferenceOptionId::ALPAMAYO_POST_VLM_RUNTIME},
        {"alpamayoNavCfg", no_argument, 0, LLMInferenceOptionId::ALPAMAYO_NAV_CFG},
        {"defaultNavGuidanceWeight", required_argument, 0, LLMInferenceOptionId::DEFAULT_NAV_GUIDANCE_WEIGHT},
        {"dumpNavDualCache", no_argument, 0, LLMInferenceOptionId::DUMP_NAV_DUAL_CACHE},
        {"navCacheOutputDir", required_argument, 0, LLMInferenceOptionId::NAV_CACHE_OUTPUT_DIR},
        {"persistentServer", no_argument, 0, LLMInferenceOptionId::PERSISTENT_SERVER},
        {"eagle", no_argument, 0, LLMInferenceOptionId::EAGLE},
        {"eagleDraftTopK", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_TOP_K},
        {"eagleDraftStep", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_STEP},
        {"eagleVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case LLMInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case LLMInferenceOptionId::DEBUG: args.debug = true; break;
        case LLMInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case LLMInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case LLMInferenceOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case LLMInferenceOptionId::DUMP_KV_CACHE: args.dumpKVCache = true; break;
        case LLMInferenceOptionId::KV_CACHE_OUTPUT_DIR: args.kvCacheOutputDir = optarg; break;
        case LLMInferenceOptionId::DUMP_PREFILL_INPUTS: args.dumpPrefillInputs = true; break;
        case LLMInferenceOptionId::PREFILL_INPUT_OUTPUT_DIR: args.prefillInputOutputDir = optarg; break;
        case LLMInferenceOptionId::FM_ENGINE: args.fmEngine = optarg; break;
        case LLMInferenceOptionId::ALPAMAYO_POST_VLM_RUNTIME: args.alpamayoPostVlmRuntime = true; break;
        case LLMInferenceOptionId::ALPAMAYO_NAV_CFG: args.alpamayoNavCfg = true; break;
        case LLMInferenceOptionId::DEFAULT_NAV_GUIDANCE_WEIGHT:
            try
            {
                args.defaultNavGuidanceWeight = std::stof(optarg);
            }
            catch (std::exception const&)
            {
                LOG_ERROR("Invalid defaultNavGuidanceWeight value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_NAV_DUAL_CACHE: args.dumpNavDualCache = true; break;
        case LLMInferenceOptionId::NAV_CACHE_OUTPUT_DIR: args.navCacheOutputDir = optarg; break;
        case LLMInferenceOptionId::PERSISTENT_SERVER: args.persistentServer = true; break;
        case LLMInferenceOptionId::EAGLE: args.eagleArgs.enabled = true; break;
        case LLMInferenceOptionId::EAGLE_DRAFT_TOP_K:
            try
            {
                args.eagleArgs.draftTopK = std::stoi(optarg);
                if (args.eagleArgs.draftTopK <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftTopK value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftTopK value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_DRAFT_STEP:
            try
            {
                args.eagleArgs.draftStep = std::stoi(optarg);
                if (args.eagleArgs.draftStep <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftStep value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftStep value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE:
            try
            {
                args.eagleArgs.verifyTreeSize = std::stoi(optarg);
                if (args.eagleArgs.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid eagleVerifyTreeSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleVerifyTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength < 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        default: return false;
        }
    }

    if (!args.inputFile.empty())
    {
        LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (!args.outputFile.empty())
    {
        LOG_INFO("args.outputFile: %s", args.outputFile.c_str());
    }

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (args.dumpKVCache)
    {
        LOG_INFO("args.dumpKVCache: enabled");
        LOG_INFO("args.kvCacheOutputDir: %s", args.kvCacheOutputDir.c_str());
    }

    if (args.dumpPrefillInputs)
    {
        LOG_INFO("args.dumpPrefillInputs: enabled");
        LOG_INFO("args.prefillInputOutputDir: %s", args.prefillInputOutputDir.c_str());
    }

    if (args.alpamayoPostVlmRuntime)
    {
        LOG_INFO("args.alpamayoPostVlmRuntime: enabled");
        if (!args.fmEngine.empty())
        {
            LOG_INFO("args.fmEngine: %s", args.fmEngine.c_str());
        }
        if (args.alpamayoNavCfg)
        {
            LOG_INFO("args.alpamayoNavCfg: enabled");
            LOG_INFO("args.defaultNavGuidanceWeight: %f", args.defaultNavGuidanceWeight);
        }
        if (args.dumpNavDualCache)
        {
            LOG_INFO("args.dumpNavDualCache: enabled");
            LOG_INFO("args.navCacheOutputDir: %s", args.navCacheOutputDir.c_str());
        }
        if (args.eagleArgs.enabled)
        {
            LOG_ERROR("ERROR: Alpamayo post-VLM runtime does not support Eagle mode");
            return false;
        }
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (!args.persistentServer)
    {
        if (args.inputFile.empty())
        {
            LOG_ERROR("ERROR: --inputFile is required");
            return false;
        }
        if (args.outputFile.empty())
        {
            LOG_ERROR("ERROR: --outputFile is required");
            return false;
        }
    }
    else
    {
        LOG_INFO("args.persistentServer: enabled");
        if (args.dumpProfile || !args.profileOutputFile.empty())
        {
            LOG_WARNING(
                "Persistent server mode currently disables profile export. Ignoring --dumpProfile/--profileOutputFile.");
            args.dumpProfile = false;
            args.profileOutputFile.clear();
        }
        if (args.dumpPrefillInputs)
        {
            LOG_WARNING("Persistent server mode does not support --dumpPrefillInputs; ignoring this option.");
            args.dumpPrefillInputs = false;
        }
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
    }

    if (args.eagleArgs.enabled)
    {
        LOG_INFO("Eagle mode enabled");
        LOG_INFO("Eagle draft topK: %d", args.eagleArgs.draftTopK);
        LOG_INFO("Eagle draft step: %d", args.eagleArgs.draftStep);
        LOG_INFO("Eagle verify tree size: %d", args.eagleArgs.verifyTreeSize);
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s with error: %s", inputFilePath.string().c_str(), e.what()));
    }

    // Extract global parameters
    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size value: %d (must be positive)", batchSize));

    // Enforce input limits (defined in cpp/common/inputLimits.h) to prevent DoS attacks and
    // excessive resource consumption. Requests exceeding these bounds are rejected early.
    // The actual engine-specific limit will be checked after the runtime is fully initialized.
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("Input rejected: batch_size %d exceeds limit %d. Limit defined in %s.", batchSize,
            limits::security::kReasonableMaxBatchSize, limits::kInputLimitsLocation));

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    check::check(maxGenerateLength >= 0,
        format::fmtstr(
            "Invalid max_generate_length value: %lld (must be non-negative)", static_cast<long long>(maxGenerateLength)));

    // Read apply_chat_template flag (defaults to true)
    bool applyChatTemplate = inputData.value("apply_chat_template", true);

    // Read add_generation_prompt flag (defaults to true)
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);

    // Read continue_final_message flag (defaults to false)
    bool continueFinalMessage = inputData.value("continue_final_message", false);

    // Read enable_thinking flag (defaults to false)
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            check::check(loraPath.is_string(), "LoRA weight path for '" + loraName + "' must be a string");
            check::check(loraWeightsMap.find(loraName) == loraWeightsMap.end(),
                "Lora weights with name " + loraName + " already exists");
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    // Parse requests and create batched requests
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requestsArray = inputData["requests"];
        size_t numRequests = requestsArray.size();

        // Process requests in batches according to batchSize
        for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
        {
            rt::LLMGenerationRequest batchRequest;
            batchRequest.temperature = temperature;
            batchRequest.topP = topP;
            batchRequest.topK = topK;
            batchRequest.maxGenerateLength = maxGenerateLength;
            batchRequest.applyChatTemplate = applyChatTemplate;
            batchRequest.addGenerationPrompt = addGenerationPrompt;
            batchRequest.continueFinalMessage = continueFinalMessage;
            batchRequest.enableThinking = enableThinking;

            // Track LoRA weights for validation
            std::string batchLoraWeightsName = "";
            bool firstInBatch = true;

            // Add requests to this batch (up to batchSize requests)
            size_t endIdx = std::min(startIdx + batchSize, numRequests);
            for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
            {
                auto const& requestItem = requestsArray[requestIdx];

                // Each request must be an object with "messages" key
                check::check(requestItem.is_object(), "Each request must be an object with 'messages' key");

                // These are request level property but currently we don't support the mechanism to group requests
                // manually in the input file. Thus, we adopt simply philosophy that we enable the property for all
                // requests in the batch if any request has set the property.
                bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
                if (saveSystemPromptKVCache)
                {
                    batchRequest.saveSystemPromptKVCache = true;
                }
                bool disableSpecDecode = requestItem.value("disable_spec_decode", false);
                if (disableSpecDecode)
                {
                    batchRequest.disableSpecDecode = true;
                }

                check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
                    "Each request object must contain a 'messages' array");

                auto const& messagesArray = requestItem["messages"];

                // Get per-conversation LoRA name if present
                std::string requestLoraName = "";
                if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
                {
                    requestLoraName = requestItem["lora_name"].get<std::string>();

                    // Validate that the LoRA name exists in available_lora_weights
                    check::check(
                        requestLoraName.empty() || loraWeightsMap.find(requestLoraName) != loraWeightsMap.end(),
                        "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
                }

                // Validate that all requests in this batch use the same LoRA weights
                if (firstInBatch)
                {
                    batchLoraWeightsName = requestLoraName;
                    firstInBatch = false;
                }
                else
                {
                    check::check(requestLoraName == batchLoraWeightsName,
                        "Different LoRA weights within the same batch are not supported");
                }

                // Parse messages into structured format
                std::vector<rt::Message> chatMessages;
                std::vector<rt::imageUtils::ImageData> imageBuffers;
                std::vector<rt::audioUtils::AudioData> audioBuffers;

                std::string egoHistoryXYZNpy;
                if (requestItem.contains("ego_history_xyz_npy") && !requestItem["ego_history_xyz_npy"].is_null())
                {
                    egoHistoryXYZNpy = requestItem["ego_history_xyz_npy"].get<std::string>();
                }

                std::string egoHistoryRotNpy;
                if (requestItem.contains("ego_history_rot_npy") && !requestItem["ego_history_rot_npy"].is_null())
                {
                    egoHistoryRotNpy = requestItem["ego_history_rot_npy"].get<std::string>();
                }

                bool predictYaw = false;
                if (requestItem.contains("predict_yaw") && !requestItem["predict_yaw"].is_null())
                {
                    predictYaw = requestItem["predict_yaw"].get<bool>();
                }

                int32_t trajTokenOffset = 3000;
                if (requestItem.contains("traj_token_offset") && !requestItem["traj_token_offset"].is_null())
                {
                    trajTokenOffset = requestItem["traj_token_offset"].get<int32_t>();
                }

                std::optional<std::string> navText;
                if (requestItem.contains("nav_text") && !requestItem["nav_text"].is_null())
                {
                    navText = requestItem["nav_text"].get<std::string>();
                }

                std::optional<float> navGuidanceWeight;
                if (requestItem.contains("nav_guidance_weight") && !requestItem["nav_guidance_weight"].is_null())
                {
                    navGuidanceWeight = requestItem["nav_guidance_weight"].get<float>();
                }

                std::optional<rt::LLMGenerationRequest::ActionSpaceConstants> actionSpaceConstants;
                if (requestItem.contains("action_space_constants") && !requestItem["action_space_constants"].is_null())
                {
                    auto const& constantsJson = requestItem["action_space_constants"];
                    check::check(constantsJson.is_object(),
                        "'action_space_constants' must be a JSON object when provided");
                    rt::LLMGenerationRequest::ActionSpaceConstants constants;
                    constants.accelMean = constantsJson.at("accel_mean").get<float>();
                    constants.accelStd = constantsJson.at("accel_std").get<float>();
                    constants.curvatureMean = constantsJson.at("curvature_mean").get<float>();
                    constants.curvatureStd = constantsJson.at("curvature_std").get<float>();
                    constants.dtValue = constantsJson.at("dt_value").get<float>();
                    constants.vLambda = constantsJson.at("v_lambda").get<float>();
                    constants.vRidge = constantsJson.at("v_ridge").get<float>();
                    actionSpaceConstants = constants;
                }

                std::optional<uint64_t> diffusionSeed;
                if (requestItem.contains("diffusion_seed") && !requestItem["diffusion_seed"].is_null())
                {
                    diffusionSeed = requestItem["diffusion_seed"].get<uint64_t>();
                }

                std::optional<int32_t> diffusionNumSteps;
                if (requestItem.contains("diffusion_num_steps") && !requestItem["diffusion_num_steps"].is_null())
                {
                    diffusionNumSteps = requestItem["diffusion_num_steps"].get<int32_t>();
                }

                bool const shouldInjectTrajectory = !egoHistoryXYZNpy.empty();
                std::string trajectoryReplacement;
                if (shouldInjectTrajectory)
                {
                    common::NpyArrayFloat32 const xyzArray = common::loadNpyFloat32(egoHistoryXYZNpy);
                    std::vector<traj::DeltaTrajectoryTokenizer::Vec3> const histXYZ = loadSingleTrajectoryXYZ(xyzArray);

                    std::vector<traj::DeltaTrajectoryTokenizer::Mat3> histRot;
                    if (predictYaw)
                    {
                        check::check(!egoHistoryRotNpy.empty(),
                            "predict_yaw is enabled but no ego_history_rot_npy was provided.");
                        common::NpyArrayFloat32 const rotArray = common::loadNpyFloat32(egoHistoryRotNpy);
                        histRot = loadSingleTrajectoryRot(rotArray);
                    }

                    traj::DeltaTrajectoryTokenizer::Config tokenizerConfig;
                    tokenizerConfig.predictYaw = predictYaw;
                    traj::DeltaTrajectoryTokenizer tokenizer(tokenizerConfig);
                    trajectoryReplacement
                        = tokenizer.formatTokens(tokenizer.encodeHistory(histXYZ, histRot), trajTokenOffset);
                }

                // Enforce message count limits
                check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
                    format::fmtstr(
                        "Input rejected: too many messages in request %zu: %zu (max: %zu). Limit defined in %s.",
                        requestIdx, messagesArray.size(), limits::security::kMaxMessagesPerRequest,
                        limits::kInputLimitsLocation));

                for (auto const& messageJson : messagesArray)
                {
                    check::check(messageJson.contains("role") && messageJson.contains("content"),
                        "Each message must have 'role' and 'content' fields");

                    rt::Message chatMsg;
                    chatMsg.role = messageJson["role"].get<std::string>();

                    auto const& contentJson = messageJson["content"];

                    // Support both string (simple text) and array (multimodal) formats
                    if (contentJson.is_string())
                    {
                        // Simple string format - treat as text content
                        std::string contentStr = contentJson.get<std::string>();
                        if (shouldInjectTrajectory)
                        {
                            replaceTrajectoryText(contentStr, trajectoryReplacement);
                        }

                        // Enforce content size limits
                        check::check(contentStr.size() <= limits::security::kMaxMessageContentSizeBytes,
                            format::fmtstr(
                                "Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
                                "Limit defined in %s.",
                                requestIdx, contentStr.size(), limits::security::kMaxMessageContentSizeBytes,
                                limits::kInputLimitsLocation));

                        rt::Message::MessageContent msgContent;
                        msgContent.type = "text";
                        msgContent.content = contentStr;
                        chatMsg.contents.push_back(msgContent);
                    }
                    else if (contentJson.is_array())
                    {
                        // Array format - supports multimodal content
                        // Enforce content item limits
                        check::check(contentJson.size() <= limits::security::kMaxContentItemsPerMessage,
                            format::fmtstr("Input rejected: too many content items in message %zu: %zu (max: %zu). "
                                           "Limit defined in %s.",
                                requestIdx, contentJson.size(), limits::security::kMaxContentItemsPerMessage,
                                limits::kInputLimitsLocation));

                        for (auto const& contentItemJson : contentJson)
                        {
                            check::check(
                                contentItemJson.contains("type"), "Each content item must have a 'type' field");

                            rt::Message::MessageContent msgContent;
                            msgContent.type = contentItemJson["type"].get<std::string>();

                            // Based on type, extract the appropriate field
                            if (msgContent.type == "text")
                            {
                                std::string textContent = contentItemJson["text"].get<std::string>();
                                if (shouldInjectTrajectory)
                                {
                                    replaceTrajectoryText(textContent, trajectoryReplacement);
                                }

                                // Enforce content size limits
                                check::check(textContent.size() <= limits::security::kMaxMessageContentSizeBytes,
                                    format::fmtstr(
                                        "Input rejected: message content too large in request %zu: %zu bytes "
                                        "(max: %zu). Limit defined in %s.",
                                        requestIdx, textContent.size(), limits::security::kMaxMessageContentSizeBytes,
                                        limits::kInputLimitsLocation));

                                msgContent.content = textContent;
                            }
                            else if (msgContent.type == "image")
                            {
                                msgContent.content = contentItemJson["image"].get<std::string>();
                                // TODO: Need to consider multi-turn conversation, and whether to load all images.
                                auto image = rt::imageUtils::loadImageFromFile(msgContent.content);
                                if (image.buffer != nullptr)
                                {
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else if (msgContent.type == "audio")
                            {
                                // Audio content for Qwen3-Omni
                                msgContent.content = contentItemJson["audio"].get<std::string>();
                                rt::audioUtils::AudioData audio;
                                std::string audioPath = msgContent.content;

                                // Check file extension
                                size_t dotPos = audioPath.find_last_of('.');
                                std::string extension;
                                if (dotPos != std::string::npos)
                                {
                                    extension = audioPath.substr(dotPos);
                                }

                                // Support mel-spectrogram .safetensors files directly
                                if (extension == ".safetensors")
                                {
                                    audio.melSpectrogramPath = audioPath;
                                    audio.melSpectrogramFormat = "safetensors";
                                    audioBuffers.push_back(std::move(audio));
                                    LOG_INFO("Loaded mel-spectrogram from: %s (safetensors format)", audioPath.c_str());
                                }
                                else
                                {
                                    LOG_WARNING(
                                        "Unsupported audio format: %s (only .safetensors mel-spectrograms are "
                                        "supported)",
                                        audioPath.c_str());
                                }
                            }
                            else
                            {
                                LOG_ERROR("Content type must be 'text', 'image', 'audio', but got: %s",
                                    msgContent.type.c_str());
                                throw std::runtime_error(
                                    format::fmtstr("Content type must be 'text', 'image', 'audio', but got: %s",
                                        msgContent.type.c_str()));
                            }

                            chatMsg.contents.push_back(msgContent);
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Message content must be a string or an array");
                    }

                    chatMessages.push_back(chatMsg);
                }

                if (shouldInjectTrajectory)
                {
                    bool trajectoryMarkerFound = false;
                    for (auto const& msg : chatMessages)
                    {
                        for (auto const& content : msg.contents)
                        {
                            if (content.type == "text"
                                && content.content.find("<|traj_history_start|>") != std::string::npos)
                            {
                                trajectoryMarkerFound = true;
                                break;
                            }
                        }
                        if (trajectoryMarkerFound)
                        {
                            break;
                        }
                    }
                    check::check(trajectoryMarkerFound,
                        "Trajectory .npy input was provided but no traj_history markers were found in request text.");
                }

                // Create prompt structure with structured messages
                rt::LLMGenerationRequest::Request request;
                request.messages = std::move(chatMessages);
                request.imageBuffers = std::move(imageBuffers);
                request.audioBuffers = std::move(audioBuffers);
                request.egoHistoryXYZNpy = std::move(egoHistoryXYZNpy);
                request.egoHistoryRotNpy = std::move(egoHistoryRotNpy);
                request.predictYaw = predictYaw;
                request.trajTokenOffset = trajTokenOffset;
                request.navText = std::move(navText);
                request.navGuidanceWeight = navGuidanceWeight;
                request.actionSpaceConstants = actionSpaceConstants;
                request.diffusionSeed = diffusionSeed;
                request.diffusionNumSteps = diffusionNumSteps;
                batchRequest.requests.push_back(std::move(request));
            }

            // Set the LoRA weights name for this batch (all requests in this batch use the same LoRA weights)
            if (!batchLoraWeightsName.empty())
            {
                batchRequest.loraWeightsName = batchLoraWeightsName;
            }

            batchedRequests.push_back(std::move(batchRequest));
        }
    }
    else
    {
        throw std::runtime_error("'requests' array not found in input file");
    }

    return std::make_pair(std::move(loraWeightsMap), std::move(batchedRequests));
}

struct PersistentServerCommand
{
    std::string inputFile;
    std::string outputFile;
    bool shutdown{false};
};

PersistentServerCommand parsePersistentServerCommand(std::string const& line)
{
    PersistentServerCommand command;
    Json payload = Json::parse(line);
    if (payload.value("command", "") == "shutdown" || payload.value("shutdown", false))
    {
        command.shutdown = true;
        return command;
    }

    command.inputFile = payload.at("input_file").get<std::string>();
    command.outputFile = payload.at("output_file").get<std::string>();
    check::check(!command.inputFile.empty(), "Persistent server command requires a non-empty input_file.");
    check::check(!command.outputFile.empty(), "Persistent server command requires a non-empty output_file.");
    return command;
}

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "llm_inference");
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    bool const profilerEnabled = args.dumpProfile;
    MemoryMonitor memoryMonitor;

    auto pluginHandles = loadEdgellmPluginLib();

    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    if (!args.inputFile.empty())
    {
        try
        {
            std::tie(loraWeightsMap, batchedRequests) = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
            LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
            LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to parse input file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    if (!args.persistentServer && batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};
    std::unique_ptr<rt::AlpamayoPostVlmRuntime> alpamayoPostVlmRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (args.eagleArgs.enabled)
    {
        rt::EagleDraftingConfig draftingConfig{
            args.eagleArgs.draftTopK, args.eagleArgs.draftStep, args.eagleArgs.verifyTreeSize};
        try
        {
            eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, draftingConfig, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceSpecDecodeRuntime: %s", e.what());
            return EXIT_FAILURE;
        }

        if (!eagleInferenceRuntime->captureDecodingCudaGraph(stream))
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for Eagle decoding usage, proceeding with normal engine execution.");
        }
    }
    else
    {
        try
        {
            llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
            return EXIT_FAILURE;
        }
        if (args.dumpPrefillInputs)
        {
            llmInferenceRuntime->enablePrefillInputDump(args.prefillInputOutputDir);
        }
        if (!llmInferenceRuntime->captureDecodingCUDAGraph(stream))
        {
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
        if (args.alpamayoPostVlmRuntime)
        {
            rt::AlpamayoPostVlmOptions options;
            options.enableNavCfg = args.alpamayoNavCfg;
            options.dumpNavDualCache = args.dumpNavDualCache;
            options.navCacheOutputDir = args.navCacheOutputDir;
            options.fmEngine = args.fmEngine;
            options.defaultNavGuidanceWeight = args.defaultNavGuidanceWeight;
            alpamayoPostVlmRuntime = std::make_unique<rt::AlpamayoPostVlmRuntime>(std::move(options));
        }
    }

    if (args.warmup > 0)
    {
        if (batchedRequests.empty())
        {
            LOG_WARNING("Warmup requested, but no --inputFile was provided. Skipping warmup in persistent server mode.");
        }
        else
        {
            setProfilingEnabled(false);
            LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
            auto& firstRequest = batchedRequests[0];

            for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
            {
                rt::LLMGenerationResponse warmupResponse;
                bool requestStatus = false;
                if (args.eagleArgs.enabled)
                {
                    requestStatus = eagleInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
                }
                else
                {
                    requestStatus = llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
                }

                if (!requestStatus)
                {
                    LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                    return EXIT_FAILURE;
                }
            }
            LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
        }
    }

    auto processRequests = [&](std::string const& inputFileLabel, std::string const& outputFilePath,
                               std::string const& profileOutputFile,
                               std::vector<rt::LLMGenerationRequest> const& activeBatchedRequests,
                               bool enableProfiling) -> int
    {
        if (enableProfiling)
        {
            memoryMonitor.start();
            setProfilingEnabled(true);
        }
        else
        {
            setProfilingEnabled(false);
        }

        nlohmann::json outputData;
        outputData["input_file"] = inputFileLabel;
        outputData["responses"] = nlohmann::json::array();

        bool hasFailedRequest = false;
        std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
        size_t failedCount = 0;

        LOG_INFO("Processing %zu batched requests...", activeBatchedRequests.size());
        for (size_t requestIdx = 0; requestIdx < activeBatchedRequests.size(); ++requestIdx)
        {
            auto const& request = activeBatchedRequests[requestIdx];
            rt::LLMGenerationResponse response;
            nlohmann::json alpamayoPostVlmJson;

            size_t progressInterval = std::max(size_t(1), std::min(activeBatchedRequests.size() / 10, size_t(100)));
            if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == activeBatchedRequests.size() - 1)
            {
                LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, activeBatchedRequests.size(),
                    100.0 * (requestIdx + 1) / activeBatchedRequests.size());
            }

            bool requestStatus = false;
            if (args.eagleArgs.enabled)
            {
                requestStatus = eagleInferenceRuntime->handleRequest(request, response, stream);
            }
            else if (args.alpamayoPostVlmRuntime)
            {
                requestStatus = alpamayoPostVlmRuntime->handleRequest(*llmInferenceRuntime, request, response, alpamayoPostVlmJson, stream);
            }
            else
            {
                requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);
            }

            if (requestStatus)
            {
                if (args.dumpOutput)
                {
                    for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                    {
                        LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                            response.outputTexts[batchIdx].c_str());
                        if (batchIdx < response.outputIds.size())
                        {
                            auto const& ids = response.outputIds[batchIdx];
                            if (!ids.empty())
                            {
                                LOG_INFO("Output token ids for request %zu batch %zu: count=%zu first=%d last=%d", requestIdx,
                                    batchIdx, ids.size(), ids.front(), ids.back());
                            }
                            else
                            {
                                LOG_INFO("Output token ids for request %zu batch %zu: empty", requestIdx, batchIdx);
                            }
                        }
                    }
                }

                if (args.dumpKVCache && !args.eagleArgs.enabled && !args.alpamayoPostVlmRuntime)
                {
                    bool const dumpStatus = dumpKVCacheSnapshot(
                        *llmInferenceRuntime, args.kvCacheOutputDir, requestIdx, request.requests.size(), stream);
                    if (!dumpStatus)
                    {
                        LOG_WARNING("Failed to dump KV-cache snapshot for request %zu", requestIdx);
                    }
                }
            }
            else
            {
                hasFailedRequest = true;
                failedCount++;
                LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
            }

            for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
            {
                nlohmann::json responseJson;
                std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
                responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
                if (requestStatus)
                {
                    responseJson["output_ids"] = response.outputIds[batchIdx];
                    responseJson["output_token_ids"] = response.outputIds[batchIdx];
                }
                responseJson["request_idx"] = requestIdx;
                responseJson["batch_idx"] = batchIdx;
                nlohmann::json messagesJson = nlohmann::json::array();
                for (auto const& msg : request.requests[batchIdx].messages)
                {
                    nlohmann::json msgJson;
                    msgJson["role"] = msg.role;
                    msgJson["content"] = nlohmann::json::array();
                    for (auto const& content : msg.contents)
                    {
                        nlohmann::json contentJson;
                        contentJson["type"] = content.type;
                        if (content.type == "text")
                        {
                            contentJson["text"] = content.content;
                        }
                        else if (content.type == "image")
                        {
                            contentJson["image"] = content.content;
                        }
                        else if (content.type == "video")
                        {
                            contentJson["video"] = content.content;
                        }
                        msgJson["content"].push_back(contentJson);
                    }
                    messagesJson.push_back(msgJson);
                }
                responseJson["messages"] = messagesJson;
                if (args.alpamayoPostVlmRuntime)
                {
                    responseJson["alpamayo_post_vlm"] = alpamayoPostVlmJson;
                    responseJson["formatted_system_prompt"]
                        = alpamayoPostVlmJson.value("guided", nlohmann::json::object()).value("formatted_system_prompt", "");
                    responseJson["formatted_complete_request"]
                        = alpamayoPostVlmJson.value("guided", nlohmann::json::object()).value("formatted_complete_request", "");
                    if (request.requests[batchIdx].navText.has_value())
                    {
                        responseJson["nav_text"] = *request.requests[batchIdx].navText;
                    }
                    if (request.requests[batchIdx].navGuidanceWeight.has_value())
                    {
                        responseJson["nav_guidance_weight"] = *request.requests[batchIdx].navGuidanceWeight;
                    }
                }
                else
                {
                    responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
                    responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
                }
                outputData["responses"].push_back(responseJson);
            }
        }

        outputData["token_metadata"] = {{"traj_future_start_token_id", 155681}, {"traj_future_end_token_id", 155683}};

        LOG_INFO("Processing complete: %zu/%zu batched requests successful", activeBatchedRequests.size() - failedCount,
            activeBatchedRequests.size());
        if (failedCount > 0)
        {
            LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
        }

        if (enableProfiling)
        {
            setProfilingEnabled(false);
            memoryMonitor.stop();
        }

        if (enableProfiling && args.dumpProfile)
        {
            std::ostringstream profileOutput;
            profileOutput << std::endl;
            profileOutput << "=== Performance Summary ===" << std::endl;
            if (args.eagleArgs.enabled)
            {
                auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
                outputPrefillProfile(profileOutput, prefillMetrics);
                outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
                outputMultimodalProfile(profileOutput, multimodalMetrics);
                outputMemoryProfile(profileOutput, memoryMonitor);
            }
            else
            {
                auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
                outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
                outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
                outputMultimodalProfile(profileOutput, multimodalMetrics);
                outputMemoryProfile(profileOutput, memoryMonitor);
            }
            profileOutput << "=====================================" << std::endl;
            LOG_INFO("%s", profileOutput.str().c_str());
        }

        if (enableProfiling && !profileOutputFile.empty())
        {
            try
            {
                nlohmann::json profileJson;

                if (args.eagleArgs.enabled)
                {
                    auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                    auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                    auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
                    addJsonPrefillSummary(profileJson, prefillMetrics);
                    addJsonEagleGenerationSummary(profileJson, eagleGenerationMetrics);
                    addJsonMultimodalSummary(profileJson, multimodalMetrics);
                    addJsonTimingStages(profileJson);
                    addJsonMemorySummary(profileJson, memoryMonitor);
                }
                else
                {
                    auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
                    addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
                    addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
                    addJsonMultimodalSummary(profileJson, multimodalMetrics);
                    addJsonTimingStages(profileJson);
                    addJsonMemorySummary(profileJson, memoryMonitor);
                }

                std::ofstream profileFile(profileOutputFile);
                if (profileFile.is_open())
                {
                    profileFile << profileJson.dump(2);
                    profileFile.close();
                    LOG_INFO("Profile data exported to: %s", profileOutputFile.c_str());
                }
                else
                {
                    LOG_ERROR("Failed to open profile output file: %s", profileOutputFile.c_str());
                    return EXIT_FAILURE;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Failed to write profile output file: %s", e.what());
                return EXIT_FAILURE;
            }
        }

        try
        {
            std::ofstream outputFile(outputFilePath);
            if (outputFile.is_open())
            {
                outputFile << outputData.dump(4);
                outputFile.close();
                LOG_INFO("All responses exported to: %s", outputFilePath.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open output file: %s", outputFilePath.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write output file: %s", e.what());
            return EXIT_FAILURE;
        }

        return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
    };

    if (!args.persistentServer)
    {
        return processRequests(args.inputFile, args.outputFile, args.profileOutputFile, batchedRequests, profilerEnabled);
    }

    LOG_INFO("Persistent server ready. Send newline-delimited JSON commands on stdin.");
    LOG_INFO(R"({"input_file":"/path/request.json","output_file":"/path/output.json"})");
    std::cout << Json{{"status", "ready"}}.dump() << std::endl;
    std::cout.flush();

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        try
        {
            auto const command = parsePersistentServerCommand(line);
            if (command.shutdown)
            {
                std::cout << Json{{"status", "shutdown"}}.dump() << std::endl;
                std::cout.flush();
                break;
            }

            std::unordered_map<std::string, std::string> commandLoraWeightsMap;
            std::vector<rt::LLMGenerationRequest> commandRequests;
            std::tie(commandLoraWeightsMap, commandRequests)
                = parseInputFile(command.inputFile, args.batchSize, args.maxGenerateLength);
            if (!commandLoraWeightsMap.empty() && commandLoraWeightsMap != loraWeightsMap)
            {
                throw std::runtime_error(
                    "Persistent server does not support changing available_lora_weights after startup.");
            }
            if (commandRequests.empty())
            {
                throw std::runtime_error("Persistent server received an input file with no valid requests.");
            }

            int const rc = processRequests(command.inputFile, command.outputFile, std::string{}, commandRequests, false);
            std::cout << Json{{"status", rc == EXIT_SUCCESS ? "ok" : "error"},
                             {"output_file", command.outputFile},
                             {"exit_code", rc}}
                             .dump()
                      << std::endl;
            std::cout.flush();
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Persistent server command failed: %s", e.what());
            std::cout << Json{{"status", "error"}, {"error", e.what()}}.dump() << std::endl;
            std::cout.flush();
        }
    }

    return EXIT_SUCCESS;
}

