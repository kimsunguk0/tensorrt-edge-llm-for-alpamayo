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
#include "common/deltaTrajectoryTokenizer.h"
#include "common/npyUtils.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

using Json = nlohmann::json;
using namespace trt_edgellm;

namespace
{

struct Args
{
    bool help{false};
    std::string inputFile;
    std::string outputFile;
    std::string egoHistoryXYZNpy;
    std::string egoHistoryRotNpy;
    int32_t tokenOffset{3000};
    bool predictYaw{false};
    bool dumpTokens{false};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " --inputFile=<path> --outputFile=<path> --egoHistoryXYZNpy=<path> "
                 "[--egoHistoryRotNpy=<path>] [--tokenOffset=<number>] [--predictYaw] [--dumpTokens]"
              << std::endl;
}

bool parseArgs(Args& args, int argc, char* argv[])
{
    enum OptionId : int
    {
        HELP = 900,
        INPUT_FILE,
        OUTPUT_FILE,
        EGO_HISTORY_XYZ_NPY,
        EGO_HISTORY_ROT_NPY,
        TOKEN_OFFSET,
        PREDICT_YAW,
        DUMP_TOKENS
    };

    static struct option options[] = {{"help", no_argument, 0, HELP}, {"inputFile", required_argument, 0, INPUT_FILE},
        {"outputFile", required_argument, 0, OUTPUT_FILE},
        {"egoHistoryXYZNpy", required_argument, 0, EGO_HISTORY_XYZ_NPY},
        {"egoHistoryRotNpy", required_argument, 0, EGO_HISTORY_ROT_NPY},
        {"tokenOffset", required_argument, 0, TOKEN_OFFSET}, {"predictYaw", no_argument, 0, PREDICT_YAW},
        {"dumpTokens", no_argument, 0, DUMP_TOKENS}, {0, 0, 0, 0}};

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: args.help = true; return true;
        case INPUT_FILE: args.inputFile = optarg; break;
        case OUTPUT_FILE: args.outputFile = optarg; break;
        case EGO_HISTORY_XYZ_NPY: args.egoHistoryXYZNpy = optarg; break;
        case EGO_HISTORY_ROT_NPY: args.egoHistoryRotNpy = optarg; break;
        case TOKEN_OFFSET: args.tokenOffset = std::stoi(optarg); break;
        case PREDICT_YAW: args.predictYaw = true; break;
        case DUMP_TOKENS: args.dumpTokens = true; break;
        default: return false;
        }
    }

    if (args.inputFile.empty() || args.outputFile.empty() || args.egoHistoryXYZNpy.empty())
    {
        return false;
    }
    if (args.predictYaw && args.egoHistoryRotNpy.empty())
    {
        return false;
    }
    return true;
}

std::vector<traj::DeltaTrajectoryTokenizer::Vec3> loadSingleTrajectoryXYZ(common::NpyArrayFloat32 const& array)
{
    check::check(array.shape.size() >= 2, "egoHistoryXYZ must have at least 2 dimensions.");
    check::check(array.shape.back() == 3, "egoHistoryXYZ last dimension must be 3.");
    int64_t const sequenceLength = array.shape[array.shape.size() - 2];
    int64_t const batchCount = array.data.size() / (sequenceLength * 3);
    check::check(batchCount == 1, "Only single-trajectory egoHistoryXYZ arrays are supported in this tool.");

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
    int64_t const batchCount = array.data.size() / (sequenceLength * 9);
    check::check(batchCount == 1, "Only single-trajectory egoHistoryRot arrays are supported in this tool.");

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

size_t replaceTrajectoryTextInJson(Json& inputData, std::string const& replacement)
{
    size_t replacements = 0;
    check::check(inputData.contains("requests") && inputData["requests"].is_array(),
        "Input JSON must contain a 'requests' array.");

    for (auto& request : inputData["requests"])
    {
        check::check(request.contains("messages") && request["messages"].is_array(),
            "Each request must contain a 'messages' array.");
        for (auto& message : request["messages"])
        {
            if (!message.contains("content"))
            {
                continue;
            }

            Json& content = message["content"];
            if (content.is_string())
            {
                std::string text = content.get<std::string>();
                if (replaceTrajectoryText(text, replacement))
                {
                    content = text;
                    ++replacements;
                }
            }
            else if (content.is_array())
            {
                for (auto& contentItem : content)
                {
                    if (!contentItem.is_object() || contentItem.value("type", "") != "text" || !contentItem.contains("text"))
                    {
                        continue;
                    }
                    std::string text = contentItem["text"].get<std::string>();
                    if (replaceTrajectoryText(text, replacement))
                    {
                        contentItem["text"] = text;
                        ++replacements;
                    }
                }
            }
        }
    }

    return replacements;
}

std::string formatTokenVector(std::vector<int32_t> const& tokens)
{
    std::string output = "[";
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (i > 0)
        {
            output += ", ";
        }
        output += std::to_string(tokens[i]);
    }
    output += "]";
    return output;
}

} // namespace

int main(int argc, char* argv[])
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    try
    {
        common::NpyArrayFloat32 const xyzArray = common::loadNpyFloat32(args.egoHistoryXYZNpy);
        std::vector<traj::DeltaTrajectoryTokenizer::Vec3> const histXYZ = loadSingleTrajectoryXYZ(xyzArray);

        std::vector<traj::DeltaTrajectoryTokenizer::Mat3> histRot;
        if (!args.egoHistoryRotNpy.empty())
        {
            common::NpyArrayFloat32 const rotArray = common::loadNpyFloat32(args.egoHistoryRotNpy);
            histRot = loadSingleTrajectoryRot(rotArray);
        }

        traj::DeltaTrajectoryTokenizer::Config config;
        config.predictYaw = args.predictYaw;
        traj::DeltaTrajectoryTokenizer tokenizer(config);

        std::vector<int32_t> const tokens = tokenizer.encodeHistory(histXYZ, histRot);
        std::string const replacement = tokenizer.formatTokens(tokens, args.tokenOffset);

        if (args.dumpTokens)
        {
            std::cout << "Token count: " << tokens.size() << std::endl;
            std::cout << "Tokens: " << formatTokenVector(tokens) << std::endl;
            std::cout << "Replacement: " << replacement << std::endl;
        }

        std::ifstream inputStream(args.inputFile);
        check::check(inputStream.is_open(), "Failed to open input JSON file: " + args.inputFile);
        Json inputData = Json::parse(inputStream);
        inputStream.close();

        size_t const replacementCount = replaceTrajectoryTextInJson(inputData, replacement);
        check::check(replacementCount > 0, "No trajectory history markers were found in the input JSON.");

        std::ofstream outputStream(args.outputFile);
        check::check(outputStream.is_open(), "Failed to open output JSON file: " + args.outputFile);
        outputStream << inputData.dump(2);
        outputStream.close();

        std::cout << "Replaced trajectory history in " << replacementCount << " text field(s)." << std::endl;
        std::cout << "Output written to: " << args.outputFile << std::endl;
    }
    catch (std::exception const& e)
    {
        std::cerr << "traj_tokenize failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
