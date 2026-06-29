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

#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "profileFormatter.h"
#include "profiling/layerProfiler.h"
#include "runtime/eagleDraftEngineRunner.h"
#include "runtime/llmEngineRunner.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm;

enum LLMBenchOptionId : int
{
    HELP = 801,
    ENGINE_DIR = 802,
    DEBUG = 803,
    BATCH_SIZE = 804,
    INPUT_LEN = 805,
    WARMUP = 806,
    ITERATIONS = 807,
    DUMP_DETAILED_LAYER_PROFILE = 808,
    MODE = 809,
    REUSE_KV_LEN = 810,
    PAST_KV_LEN = 811,
    VERIFY_TREE_SIZE = 812,
    DRAFT_TREE_SIZE = 813,
    SEED = 814
};

enum class BenchMode
{
    kPREFILL,
    kDECODE,
    kEAGLE_VERIFY,
    kEAGLE_DRAFT_PROPOSAL,
    kEAGLE_DRAFT_PREFILL
};

struct LLMBenchArgs
{
    bool help{false};
    std::string engineDir;
    bool debug{false};
    int32_t batchSize{1};
    int32_t inputLen{0};
    int32_t warmup{2};
    int32_t iterations{10};
    bool dumpDetailedLayerProfile{false};

    BenchMode mode{BenchMode::kPREFILL};
    int32_t reuseKVLen{0};
    int32_t pastKVLen{0};
    int32_t verifyTreeSize{0};
    int32_t draftTreeSize{0};
    uint64_t seed{0};
};

std::string modeToString(BenchMode mode)
{
    switch (mode)
    {
    case BenchMode::kPREFILL: return "prefill";
    case BenchMode::kDECODE: return "decode";
    case BenchMode::kEAGLE_VERIFY: return "eagle_verify";
    case BenchMode::kEAGLE_DRAFT_PROPOSAL: return "eagle_draft_proposal";
    case BenchMode::kEAGLE_DRAFT_PREFILL: return "eagle_draft_prefill";
    default: return "unknown";
    }
}

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --engineDir <dir> "
                 "[--debug] [--batchSize <int>] [--warmup <int>] [--iterations <int>] "
                 "[--dumpDetailedLayerProfile] "
                 "[--mode <prefill|decode|eagle_verify|eagle_draft_proposal|eagle_draft_prefill>] "
                 "[--inputLen <int>] [--reuseKVLen <int>] [--pastKVLen <int>] "
                 "[--verifyTreeSize <int>] [--draftTreeSize <int>]"
                 "[--seed <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --engineDir               Provide the output/input TensorRT engine directory path. Required."
              << std::endl;
    std::cerr << "  --debug                   Use debug mode." << std::endl;
    std::cerr << "  --batchSize               Bench: Batch size. Default = 1" << std::endl;
    std::cerr << "  --warmup                  Bench: Warmup iterations. Default = 2" << std::endl;
    std::cerr << "  --iterations              Bench: Measurement iterations. Default = 10" << std::endl;
    std::cerr << "  --dumpDetailedLayerProfile Bench: Enable detailed layer profile dumping. Default = false"
              << std::endl;
    std::cerr << "  --mode                    Bench: Benchmarking mode. Default = prefill" << std::endl;
    std::cerr << "  --inputLen                Bench: Input sequence length (for prefill)." << std::endl;
    std::cerr << "  --reuseKVLen              Bench: Reused KV cache length (for prefill). Default = 0" << std::endl;
    std::cerr << "  --pastKVLen               Bench: Past KV cache length (for decode/verify/draft). Default = 0"
              << std::endl;
    std::cerr << "  --verifyTreeSize          Bench: Verify tree size (for eagle_verify)." << std::endl;
    std::cerr << "  --draftTreeSize           Bench: Draft tree size (for eagle_draft)." << std::endl;
    std::cerr << "  --seed                    Bench: Random seed. Default = 0" << std::endl;
}

bool parseLLMBenchArgs(LLMBenchArgs& args, int argc, char* argv[])
{
    static struct option benchOptions[] = {{"help", no_argument, 0, LLMBenchOptionId::HELP},
        {"engineDir", required_argument, 0, LLMBenchOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, LLMBenchOptionId::DEBUG},
        {"batchSize", required_argument, 0, LLMBenchOptionId::BATCH_SIZE},
        {"inputLen", required_argument, 0, LLMBenchOptionId::INPUT_LEN},
        {"warmup", required_argument, 0, LLMBenchOptionId::WARMUP},
        {"iterations", required_argument, 0, LLMBenchOptionId::ITERATIONS},
        {"dumpDetailedLayerProfile", no_argument, 0, LLMBenchOptionId::DUMP_DETAILED_LAYER_PROFILE},
        {"mode", required_argument, 0, LLMBenchOptionId::MODE},
        {"reuseKVLen", required_argument, 0, LLMBenchOptionId::REUSE_KV_LEN},
        {"pastKVLen", required_argument, 0, LLMBenchOptionId::PAST_KV_LEN},
        {"verifyTreeSize", required_argument, 0, LLMBenchOptionId::VERIFY_TREE_SIZE},
        {"draftTreeSize", required_argument, 0, LLMBenchOptionId::DRAFT_TREE_SIZE},
        {"seed", required_argument, 0, LLMBenchOptionId::SEED}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", benchOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMBenchOptionId::HELP: args.help = true; return true;
        case LLMBenchOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMBenchOptionId::DEBUG: args.debug = true; break;
        case LLMBenchOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::INPUT_LEN:
            try
            {
                args.inputLen = std::stoi(optarg);
                if (args.inputLen <= 0)
                {
                    LOG_ERROR("Invalid inputLen: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid inputLen value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::ITERATIONS:
            try
            {
                args.iterations = std::stoi(optarg);
                if (args.iterations <= 0)
                {
                    LOG_ERROR("Invalid iterations: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid iterations value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::DUMP_DETAILED_LAYER_PROFILE: args.dumpDetailedLayerProfile = true; break;
        case LLMBenchOptionId::MODE:
        {
            std::string modeStr = optarg;
            if (modeStr == "prefill")
            {
                args.mode = BenchMode::kPREFILL;
            }
            else if (modeStr == "decode")
            {
                args.mode = BenchMode::kDECODE;
            }
            else if (modeStr == "eagle_verify")
            {
                args.mode = BenchMode::kEAGLE_VERIFY;
            }
            else if (modeStr == "eagle_draft_proposal")
            {
                args.mode = BenchMode::kEAGLE_DRAFT_PROPOSAL;
            }
            else if (modeStr == "eagle_draft_prefill")
            {
                args.mode = BenchMode::kEAGLE_DRAFT_PREFILL;
            }
            else
            {
                LOG_ERROR("Invalid mode: %s", optarg);
                return false;
            }
            break;
        }
        case LLMBenchOptionId::REUSE_KV_LEN:
            try
            {
                args.reuseKVLen = std::stoi(optarg);
                if (args.reuseKVLen < 0)
                {
                    LOG_ERROR("Invalid reuseKVLen: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid reuseKVLen value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::PAST_KV_LEN:
            try
            {
                args.pastKVLen = std::stoi(optarg);
                if (args.pastKVLen < 0)
                {
                    LOG_ERROR("Invalid pastKVLen: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid pastKVLen value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::VERIFY_TREE_SIZE:
            try
            {
                args.verifyTreeSize = std::stoi(optarg);
                if (args.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid verifyTreeSize: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid verifyTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::DRAFT_TREE_SIZE:
            try
            {
                args.draftTreeSize = std::stoi(optarg);
                if (args.draftTreeSize <= 0)
                {
                    LOG_ERROR("Invalid draftTreeSize: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid draftTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMBenchOptionId::SEED:
            try
            {
                args.seed = std::stoull(optarg);
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid seed value: %s", optarg);
                return false;
            }
            break;
        default: return false;
        }
    }
    return true;
}

// Helper to fill random data
void fillRandomFloat(rt::Tensor& tensor, float minVal, float maxVal, uint64_t seed)
{
    size_t vol = tensor.getShape().volume();
    std::vector<float> hostData(vol);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (size_t i = 0; i < vol; ++i)
    {
        hostData[i] = dis(gen);
    }

    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(float), cudaMemcpyHostToDevice));
}

void fillRandomHalf(rt::Tensor& tensor, float minVal, float maxVal, uint64_t seed)
{
    size_t vol = tensor.getShape().volume();
    std::vector<float> hostData(vol);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (size_t i = 0; i < vol; ++i)
    {
        hostData[i] = dis(gen);
    }

    if (tensor.getDataType() == nvinfer1::DataType::kHALF)
    {
        std::vector<half> halfData(vol);
        for (size_t i = 0; i < vol; ++i)
        {
            halfData[i] = __float2half(hostData[i]);
        }
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), halfData.data(), vol * sizeof(half), cudaMemcpyHostToDevice));
    }
    else
    {
        LOG_ERROR("Unsupported data type for random fill");
    }
}

void fillInt32(rt::Tensor& tensor, int32_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int32_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void fillInt8(rt::Tensor& tensor, int8_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int8_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int8_t), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
    LLMBenchArgs args;
    if ((argc < 2) || (!parseLLMBenchArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse bench args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.engineDir.empty())
    {
        LOG_ERROR("--engineDir is required");
        return EXIT_FAILURE;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    LOG_INFO("Starting Benchmark...");
    LOG_INFO("Mode: %s", modeToString(args.mode).c_str());

    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    trt_edgellm::layerProfiler::enableLayerProfilers();

    std::unique_ptr<rt::LLMEngineRunner> runner;
    std::unique_ptr<rt::EagleDraftEngineRunner> draftRunner;
    int32_t hiddenSize = 0;
    int32_t vocabSize = 0;
    int32_t eagleHiddenDim = 0;
    int32_t numDeepstackFeatures = 0;

    if (args.mode == BenchMode::kPREFILL && args.inputLen <= 0)
    {
        LOG_ERROR("inputLen must be positive for PREFILL mode");
        return EXIT_FAILURE;
    }
    if (args.mode == BenchMode::kEAGLE_VERIFY && args.verifyTreeSize <= 0)
    {
        LOG_ERROR("verifyTreeSize must be positive for EAGLE_VERIFY mode");
        return EXIT_FAILURE;
    }
    if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL && args.draftTreeSize <= 0)
    {
        LOG_ERROR("draftTreeSize must be positive for EAGLE_DRAFT_PROPOSAL mode");
        return EXIT_FAILURE;
    }
    if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL && args.inputLen <= 0)
    {
        LOG_ERROR("inputLen must be positive for EAGLE_DRAFT_PREFILL mode");
        return EXIT_FAILURE;
    }

    // Initialize Runners
    if (args.mode == BenchMode::kPREFILL || args.mode == BenchMode::kDECODE || args.mode == BenchMode::kEAGLE_VERIFY)
    {
        std::filesystem::path enginePath = std::filesystem::path(args.engineDir) / "llm.engine";
        std::filesystem::path configPath = std::filesystem::path(args.engineDir) / "config.json";
        if (!std::filesystem::exists(enginePath))
        {
            LOG_ERROR("LLM engine not found at %s", enginePath.string().c_str());
            return EXIT_FAILURE;
        }
        std::unordered_map<std::string, std::string> loraMap;
        try
        {
            runner = std::make_unique<rt::LLMEngineRunner>(enginePath, configPath, loraMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create LLMEngineRunner: %s", e.what());
            return EXIT_FAILURE;
        }
        auto engineConfig = runner->getEngineConfig();
        hiddenSize = engineConfig.hiddenSize;
        vocabSize = engineConfig.vocabSize;
        eagleHiddenDim = engineConfig.outputHiddenDim;
        numDeepstackFeatures = engineConfig.numDeepstackFeatures;
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL || args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        std::filesystem::path enginePath = std::filesystem::path(args.engineDir) / "eagle_draft.engine";
        std::filesystem::path configPath = std::filesystem::path(args.engineDir) / "draft_config.json";
        if (!std::filesystem::exists(enginePath))
        {
            LOG_ERROR("Eagle draft engine not found at %s", enginePath.string().c_str());
            return EXIT_FAILURE;
        }
        try
        {
            draftRunner = std::make_unique<rt::EagleDraftEngineRunner>(enginePath, configPath, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create EagleDraftEngineRunner: %s", e.what());
            return EXIT_FAILURE;
        }
        auto engineConfig = draftRunner->getDraftEngineConfig();
        hiddenSize = engineConfig.draftModelHiddenDim;
        vocabSize = engineConfig.draftModelVocabSize;
        eagleHiddenDim = engineConfig.draftModelHiddenDim;
    }

    if (runner)
    {
        auto conf = runner->getEngineConfig();
        LOG_INFO(
            "Base Engine Config:\n"
            "  Layers: %d\n"
            "  KV Heads: %d\n"
            "  Head Dim: %d\n"
            "  Rotary Dim: %d\n"
            "  Hidden Size: %d\n"
            "  Vocab Size: %d\n"
            "  Output Vocab Size: %d\n"
            "  Max Supported Batch Size: %d\n"
            "  Max Supported Input Length: %d\n"
            "  Max KV Cache Capacity: %d\n"
            "  Max Supported LoRA Rank: %d\n"
            "  Eagle Enable: %s\n"
            "  Output Hidden Dim: %d\n"
            "  Max Verify Tree Size: %d\n"
            "  Num Deepstack Features: %d",
            conf.numDecoderLayers, conf.numKVHeads, conf.headDim, conf.rotaryDim, conf.hiddenSize, conf.vocabSize,
            conf.outputVocabSize, conf.maxSupportedBatchSize, conf.maxSupportedInputLength, conf.maxKVCacheCapacity,
            conf.maxSupportedLoraRank, conf.enableEagleSpecDecode ? "true" : "false", conf.outputHiddenDim,
            conf.maxVerifyTreeSize, conf.numDeepstackFeatures);
    }
    else if (draftRunner)
    {
        auto conf = draftRunner->getDraftEngineConfig();
        LOG_INFO(
            "Draft Engine Config:\n"
            "  Layers: %d\n"
            "  KV Heads: %d\n"
            "  Head Dim: %d\n"
            "  Rotary Dim: %d\n"
            "  Draft Hidden Size: %d\n"
            "  Base Hidden Size: %d\n"
            "  Draft Vocab Size: %d\n"
            "  Max Supported Batch Size: %d\n"
            "  Max Supported Input Length: %d\n"
            "  Max KV Cache Capacity: %d\n"
            "  Max Draft Tree Size: %d",
            conf.numDecoderLayers, conf.numKVHeads, conf.headDim, conf.rotaryDim, conf.draftModelHiddenDim,
            conf.baseModelHiddenDim, conf.draftModelVocabSize, conf.maxSupportedBatchSize, conf.maxSupportedInputLength,
            conf.maxKVCacheCapacity, conf.maxDraftTreeSize);
    }

    LOG_INFO(
        "Bench Config:\n"
        "  Mode: %s\n"
        "  Batch Size: %d\n"
        "  Iterations: %d\n"
        "  Warmup: %d\n"
        "  Input Len: %d\n"
        "  Reuse KV Len: %d\n"
        "  Past KV Len: %d\n"
        "  Verify Tree Size: %d\n"
        "  Draft Tree Size: %d\n"
        "  Seed: %lu",
        modeToString(args.mode).c_str(), args.batchSize, args.iterations, args.warmup, args.inputLen, args.reuseKVLen,
        args.pastKVLen, args.verifyTreeSize, args.draftTreeSize, args.seed);

    // Prepare Tensors common to multiple modes
    rt::Tensor reuseKVCacheLengths(
        rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "reuse_kv_lengths");
    int32_t kvLenToSet = (args.mode == BenchMode::kPREFILL || args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
        ? args.reuseKVLen
        : args.pastKVLen;
    fillInt32(reuseKVCacheLengths, kvLenToSet);

    rt::Tensor prefillInputs;
    rt::Tensor contextLengths;
    rt::Tensor prefillLogits;
    std::vector<rt::Tensor> prefillDeepstackTensors;
    rt::OptionalInputTensors prefillDeepstackEmbeds;

    rt::Tensor decodeInputs;
    rt::Tensor decodeLogits;

    rt::Tensor verifyInputs;
    rt::Tensor verifyMask;
    rt::Tensor verifyLogits;
    rt::Tensor verifyHiddenStates;

    rt::Tensor draftInputs;
    rt::Tensor draftMask;
    rt::Tensor draftBaseHiddenStates;
    rt::Tensor draftModelHiddenStates;
    rt::Tensor draftTreeLength;
    rt::Tensor draftLogits;
    rt::Tensor draftOutputHiddenStates;

    rt::Tensor draftPrefillInputs;
    rt::Tensor draftBaseHiddenStatesPrefill;
    rt::Tensor draftModelHiddenStatesPrefill;
    rt::Tensor draftContextLengths;
    rt::Tensor draftPrefillLogits;
    rt::Tensor draftPrefillOutputHiddenStates;
    rt::Tensor draftRopeCosSinCache; // Draft engine manages its own rope cache

    if (args.mode == BenchMode::kPREFILL)
    {
        LOG_INFO("Mode PREFILL: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);
        prefillInputs = rt::Tensor(rt::Coords{args.batchSize, args.inputLen, hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "prefill_input");
        fillRandomHalf(prefillInputs, -1.0f, 1.0f, args.seed);

        contextLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "context_lengths");
        std::vector<int32_t> lengthsHost(args.batchSize, args.reuseKVLen + args.inputLen);
        std::memcpy(contextLengths.rawPointer(), lengthsHost.data(), lengthsHost.size() * sizeof(int32_t));

        prefillLogits = rt::Tensor(
            rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "prefill_logits");

        prefillDeepstackTensors.reserve(numDeepstackFeatures);
        for (int32_t i = 0; i < numDeepstackFeatures; ++i)
        {
            prefillDeepstackTensors.emplace_back(rt::Coords{args.batchSize, args.inputLen, hiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
                std::string("prefill_deepstack_embed_") + std::to_string(i));
        }
        for (auto& tensor : prefillDeepstackTensors)
        {
            fillRandomHalf(tensor, 0.0f, 0.0f, args.seed);
            prefillDeepstackEmbeds.push_back(std::cref(tensor));
        }
    }
    else if (args.mode == BenchMode::kDECODE)
    {
        LOG_INFO("Mode DECODE: PastKVLen=%d", args.pastKVLen);
        decodeInputs = rt::Tensor(
            rt::Coords{args.batchSize, 1, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "decode_input");
        fillRandomHalf(decodeInputs, -1.0f, 1.0f, args.seed);

        decodeLogits = rt::Tensor(
            rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "decode_logits");
    }
    else if (args.mode == BenchMode::kEAGLE_VERIFY)
    {
        LOG_INFO("Mode EAGLE_VERIFY: VerifyTreeSize=%d, PastKVLen=%d", args.verifyTreeSize, args.pastKVLen);
        verifyInputs = rt::Tensor(rt::Coords{args.batchSize, args.verifyTreeSize, hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "verify_input");
        fillRandomHalf(verifyInputs, -1.0f, 1.0f, args.seed);

        verifyMask = rt::Tensor(rt::Coords{args.batchSize, args.verifyTreeSize, args.verifyTreeSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "verify_mask");
        fillInt32(verifyMask, 1); // Full mask for simplicity

        int32_t selectTokenSize = args.batchSize * args.verifyTreeSize;
        verifyLogits = rt::Tensor(
            rt::Coords{selectTokenSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "verify_logits");
        verifyHiddenStates = rt::Tensor(rt::Coords{selectTokenSize, eagleHiddenDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "verify_hidden_states");
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL)
    {
        LOG_INFO("Mode EAGLE_DRAFT_PROPOSAL: DraftTreeSize=%d, PastKVLen=%d", args.draftTreeSize, args.pastKVLen);
        draftInputs = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "draft_input");
        fillRandomHalf(draftInputs, -1.0f, 1.0f, args.seed);

        draftMask = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, args.draftTreeSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "draft_mask");
        fillInt8(draftMask, 1);

        draftBaseHiddenStates = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, 1024}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "draft_base_hidden_unused");
        fillRandomHalf(draftBaseHiddenStates, 0.0f, 0.0f, args.seed);

        draftModelHiddenStates = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "draft_hidden_input");
        fillRandomHalf(draftModelHiddenStates, -1.0f, 1.0f, args.seed);

        draftTreeLength = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "draft_tree_len");
        fillInt32(draftTreeLength, args.draftTreeSize);

        int32_t numSelectedTokens = args.draftTreeSize;
        draftLogits = rt::Tensor(rt::Coords{args.batchSize, numSelectedTokens, vocabSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "draft_logits");
        draftOutputHiddenStates = rt::Tensor(rt::Coords{args.batchSize, numSelectedTokens, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "draft_hidden_output");
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        LOG_INFO("Mode EAGLE_DRAFT_PREFILL: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);
        draftPrefillInputs = rt::Tensor(rt::Coords{args.batchSize, args.inputLen, hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "draft_prefill_input");
        fillRandomHalf(draftPrefillInputs, -1.0f, 1.0f, args.seed);

        auto draftConfig = draftRunner->getDraftEngineConfig();
        int32_t baseHiddenDim = draftConfig.baseModelHiddenDim;

        draftBaseHiddenStatesPrefill = rt::Tensor(rt::Coords{args.batchSize, args.inputLen, baseHiddenDim},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "draft_base_hidden_prefill");
        fillRandomHalf(draftBaseHiddenStatesPrefill, -1.0f, 1.0f, args.seed);

        draftModelHiddenStatesPrefill = rt::Tensor(rt::Coords{args.batchSize, args.inputLen, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "draft_hidden_unused_prefill");
        fillRandomHalf(draftModelHiddenStatesPrefill, 0.0f, 0.0f, args.seed);

        draftContextLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "draft_context_lengths");
        std::vector<int32_t> lengthsHost(args.batchSize, args.reuseKVLen + args.inputLen);
        std::memcpy(draftContextLengths.rawPointer(), lengthsHost.data(), lengthsHost.size() * sizeof(int32_t));

        draftPrefillLogits = rt::Tensor(rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "draft_prefill_logits"); // Float32 logits
        draftPrefillOutputHiddenStates = rt::Tensor(rt::Coords{args.batchSize, hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "draft_prefill_output_hidden");

        int32_t rotaryDim = draftConfig.rotaryDim;
        int32_t maxLen = draftConfig.maxSupportedInputLength;
        draftRopeCosSinCache = rt::Tensor(
            rt::Coords{maxLen, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "dummy_base_rope_cache");
        fillRandomFloat(draftRopeCosSinCache, 0.0f, 1.0f, args.seed);
    }

    // Warmup
    LOG_INFO("Warming up (%d iterations)...", args.warmup);
    trt_edgellm::layerProfiler::disableLayerProfilers();

    for (int i = 0; i < args.warmup; ++i)
    {
        if (args.mode == BenchMode::kPREFILL)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            runner->executePrefillStep(
                prefillInputs, contextLengths, prefillDeepstackEmbeds, prefillLogits, std::nullopt, stream);
        }
        else if (args.mode == BenchMode::kDECODE)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            // No hidden states output needed for benchmark decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            runner->executeVanillaDecodingStep(decodeInputs, decodeLogits, outputHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_VERIFY)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            runner->executeEagleBaseTreeDecodingStep(
                verifyInputs, verifyMask, verifyLogits, verifyHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL)
        {
            draftRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            draftRunner->executeEagleDraftProposalStep(draftInputs, draftBaseHiddenStates, draftModelHiddenStates,
                draftTreeLength, draftMask, draftLogits, draftOutputHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
        {
            draftRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            draftRunner->executeEaglePrefillStep(draftPrefillInputs, draftBaseHiddenStatesPrefill,
                draftModelHiddenStatesPrefill, draftContextLengths, draftPrefillLogits, draftPrefillOutputHiddenStates,
                draftRopeCosSinCache, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    trt_edgellm::layerProfiler::enableLayerProfilers();
    LOG_INFO("Running Benchmark (%d iterations)...", args.iterations);

    double totalTime = 0;
    for (int i = 0; i < args.iterations; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (args.mode == BenchMode::kPREFILL)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            runner->executePrefillStep(
                prefillInputs, contextLengths, prefillDeepstackEmbeds, prefillLogits, std::nullopt, stream);
        }
        else if (args.mode == BenchMode::kDECODE)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            // No hidden states output needed for benchmark decoding.
            rt::OptionalOutputTensor const outputHiddenStates{std::nullopt};
            runner->executeVanillaDecodingStep(decodeInputs, decodeLogits, outputHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_VERIFY)
        {
            runner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            runner->executeEagleBaseTreeDecodingStep(
                verifyInputs, verifyMask, verifyLogits, verifyHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL)
        {
            draftRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            draftRunner->executeEagleDraftProposalStep(draftInputs, draftBaseHiddenStates, draftModelHiddenStates,
                draftTreeLength, draftMask, draftLogits, draftOutputHiddenStates, stream);
        }
        else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
        {
            draftRunner->getLinearKVCache().resetForNewSequences(reuseKVCacheLengths, stream);
            draftRunner->executeEaglePrefillStep(draftPrefillInputs, draftBaseHiddenStatesPrefill,
                draftModelHiddenStatesPrefill, draftContextLengths, draftPrefillLogits, draftPrefillOutputHiddenStates,
                draftRopeCosSinCache, stream);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avgTime = totalTime / args.iterations;

    // Output layer profile
    std::ostringstream profileOutput;
    outputLayerProfiles(profileOutput, args.dumpDetailedLayerProfile);
    LOG_INFO("%s", profileOutput.str().c_str());

    LOG_INFO("Results:");
    LOG_INFO("  Mode: %s", modeToString(args.mode).c_str());
    LOG_INFO("  Avg Time: %.3f ms", avgTime);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
