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

#include "builder/builder.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace trt_edgellm;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMBuildOptionId : int
{
    HELP = 701,
    ONNX_DIR = 702,
    ENGINE_DIR = 703,
    MAX_INPUT_LEN = 704,
    MAX_KV_CACHE_CAPACITY = 705,
    DEBUG = 706,
    MAX_BATCH_SIZE = 707,
    MAX_LORA_RANK = 708,
    EAGLE_DRAFT = 709,
    EAGLE_BASE = 710,
    MAX_VERIFY_TREE_SIZE = 711,
    MAX_DRAFT_TREE_SIZE = 712,
    DISABLE_TF32 = 713,
    BUILDER_OPTIMIZATION_LEVEL = 714,
    TACTIC_SOURCES = 715,
    DISABLE_TIMING_CACHE = 716
};

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    int64_t maxInputLen{1024};
    int64_t maxKVCacheCapacity{4096};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // Default to 0 means no LoRA
    bool eagleDraft{false};
    bool eagleBase{false};
    int64_t maxVerifyTreeSize{60};
    int64_t maxDraftTreeSize{60};
    bool disableTF32{false};
    bool disableTimingCache{false};
    int32_t builderOptimizationLevel{-1};
    uint32_t tacticSourcesMask{0};
};

uint32_t tacticSourceBit(char const* name)
{
    std::string value{name ? name : ""};
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (value == "CUBLAS")
    {
        return 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS);
    }
    if (value == "CUBLAS_LT" || value == "CUBLASLT")
    {
        return 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT);
    }
    if (value == "CUDNN")
    {
        return 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN);
    }
    if (value == "EDGE_MASK" || value == "EDGE_MASK_CONVOLUTIONS")
    {
        return 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS);
    }
    if (value == "JIT" || value == "JIT_CONVOLUTIONS")
    {
        return 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS);
    }
    return 0;
}

bool parseTacticSources(std::string const& arg, uint32_t& mask)
{
    if (arg.empty())
    {
        return false;
    }
    std::stringstream ss(arg);
    std::string item;
    uint32_t parsedMask = 0;
    while (std::getline(ss, item, ','))
    {
        if (item.empty())
        {
            continue;
        }
        uint32_t const bit = tacticSourceBit(item.c_str());
        if (bit == 0)
        {
            LOG_ERROR("Unknown tactic source '%s'. Supported values: CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK, JIT", item.c_str());
            return false;
        }
        parsedMask |= bit;
    }
    mask = parsedMask;
    return parsedMask != 0;
}

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnxDir <dir> --engineDir <dir> [--maxInputLen <int>] "
                 "[--maxKVCacheCapacity <int>] [--maxBatchSize <int>] [--debug] [--maxLoraRank <int>]"
                 "[--eagleDraft] [--eagleBase] [--maxVerifyTreeSize <int>] "
                 "[--maxDraftTreeSize <int>] [--disableTF32] [--builderOptimizationLevel <int>] "
                 "[--tacticSources <csv>] [--disableTimingCache]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --onnxDir                 Provide the input ONNX directory path. Required. " << std::endl;
    std::cerr << "  --engineDir               Provide the output TensorRT engine directory path. Required. "
              << std::endl;
    std::cerr << "  --maxInputLen             Provide the maximum input length for the model. Default = 1024"
              << std::endl;
    std::cerr << "  --maxKVCacheCapacity      Provide the maximum KV cache capacity (sequence length). "
                 "Default = 4096"
              << std::endl;
    std::cerr << "  --maxBatchSize            Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --debug                   Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank             Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    std::cerr << "  --eagleDraft              Enable Eagle draft mode" << std::endl;
    std::cerr << "  --eagleBase               Enable Eagle base mode" << std::endl;
    std::cerr << "  --maxVerifyTreeSize       Maximum input_ids tokens passed into Eagle base model for tree "
                 "verification. Default = 60"
              << std::endl;
    std::cerr << "  --maxDraftTreeSize        Maximum input_ids tokens passed into Eagle draft model for draft "
                 "generation. Default = 60"
              << std::endl
              << "  --disableTF32             Disable TF32 during engine build" << std::endl;
    std::cerr << "  --builderOptimizationLevel Override TensorRT builder optimization level" << std::endl;
    std::cerr << "  --tacticSources           Comma-separated tactic source override. Supported: "
                 "CUBLAS,CUBLAS_LT,CUDNN,EDGE_MASK,JIT"
              << std::endl;
    std::cerr << "  --disableTimingCache      Disable TensorRT timing cache reuse during build" << std::endl
              << std::endl;
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, LLMBuildOptionId::HELP},
        {"onnxDir", required_argument, 0, LLMBuildOptionId::ONNX_DIR},
        {"engineDir", required_argument, 0, LLMBuildOptionId::ENGINE_DIR},
        {"maxInputLen", required_argument, 0, LLMBuildOptionId::MAX_INPUT_LEN},
        {"maxKVCacheCapacity", required_argument, 0, LLMBuildOptionId::MAX_KV_CACHE_CAPACITY},
        {"debug", no_argument, 0, LLMBuildOptionId::DEBUG},
        {"maxBatchSize", required_argument, 0, LLMBuildOptionId::MAX_BATCH_SIZE},
        {"maxLoraRank", required_argument, 0, LLMBuildOptionId::MAX_LORA_RANK},
        {"eagleDraft", no_argument, 0, LLMBuildOptionId::EAGLE_DRAFT},
        {"eagleBase", no_argument, 0, LLMBuildOptionId::EAGLE_BASE},
        {"maxVerifyTreeSize", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"maxDraftTreeSize", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"disableTF32", no_argument, 0, LLMBuildOptionId::DISABLE_TF32},
        {"builderOptimizationLevel", required_argument, 0, LLMBuildOptionId::BUILDER_OPTIMIZATION_LEVEL},
        {"tacticSources", required_argument, 0, LLMBuildOptionId::TACTIC_SOURCES},
        {"disableTimingCache", no_argument, 0, LLMBuildOptionId::DISABLE_TIMING_CACHE},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", buildOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMBuildOptionId::HELP: args.help = true; return true;
        case LLMBuildOptionId::ONNX_DIR:
            if (optarg)
            {
                args.onnxDir = optarg;
            }
            else
            {
                LOG_ERROR("--onnxDir requires option argument.");
                return false;
            }
            break;
        case LLMBuildOptionId::ENGINE_DIR:
            if (optarg)
            {
                args.engineDir = optarg;
            }
            else
            {
                LOG_ERROR("--engineDir requires option argument.");
                return false;
            }
            break;
        case LLMBuildOptionId::MAX_INPUT_LEN:
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_KV_CACHE_CAPACITY:
            if (optarg)
            {
                args.maxKVCacheCapacity = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::DEBUG: args.debug = true; break;
        case LLMBuildOptionId::MAX_BATCH_SIZE:
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_LORA_RANK:
            if (optarg)
            {
                args.maxLoraRank = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::EAGLE_DRAFT: args.eagleDraft = true; break;
        case LLMBuildOptionId::EAGLE_BASE: args.eagleBase = true; break;
        case LLMBuildOptionId::MAX_VERIFY_TREE_SIZE:
            if (optarg)
            {
                args.maxVerifyTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_DRAFT_TREE_SIZE:
            if (optarg)
            {
                args.maxDraftTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::DISABLE_TF32: args.disableTF32 = true; break;
        case LLMBuildOptionId::BUILDER_OPTIMIZATION_LEVEL:
            if (optarg)
            {
                args.builderOptimizationLevel = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::TACTIC_SOURCES:
            if (optarg && !parseTacticSources(optarg, args.tacticSourcesMask))
            {
                return false;
            }
            break;
        case LLMBuildOptionId::DISABLE_TIMING_CACHE: args.disableTimingCache = true; break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    LLMBuildArgs args;
    if ((argc < 2) || (!parseLLMBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    // Validate input directory and required files
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    // Create LLMBuilderConfig from args
    builder::LLMBuilderConfig config;
    config.maxInputLen = args.maxInputLen;
    config.maxKVCacheCapacity = args.maxKVCacheCapacity;
    config.maxBatchSize = args.maxBatchSize;
    config.maxLoraRank = args.maxLoraRank;
    config.eagleDraft = args.eagleDraft;
    config.eagleBase = args.eagleBase;
    config.maxVerifyTreeSize = args.maxVerifyTreeSize;
    config.maxDraftTreeSize = args.maxDraftTreeSize;
    config.disableTF32 = args.disableTF32;
    config.disableTimingCache = args.disableTimingCache;
    config.builderOptimizationLevel = args.builderOptimizationLevel;
    config.tacticSourcesMask = args.tacticSourcesMask;

    // Create and run the builder
    builder::LLMBuilder llmBuilder(args.onnxDir, args.engineDir, config);
    if (!llmBuilder.build())
    {
        LOG_ERROR("Failed to build LLM engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("LLM engine built successfully.");
    return EXIT_SUCCESS;
}
