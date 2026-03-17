/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/logger.h"
#include "common/tensor.h"
#include "multimodal/qwenViTRunner.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;

namespace fs = std::filesystem;
using namespace trt_edgellm;

namespace
{

struct Args
{
    std::string engineDir{"/models/engines/visual_nq"};
    std::string imageDir{"/root/TensorRT-Edge-LLM/input/images"};
    std::string outputDir{"/root/TensorRT-Edge-LLM/output/visual_dumps/visual_nq"};
    bool help{false};
};

enum OptionId : int
{
    HELP = 601,
    ENGINE_DIR = 602,
    IMAGE_DIR = 603,
    OUTPUT_DIR = 604
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--engineDir DIR] [--imageDir DIR] [--outputDir DIR]\n";
}

bool parseArgs(Args& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, nullptr, OptionId::HELP},
        {"engineDir", required_argument, nullptr, OptionId::ENGINE_DIR},
        {"imageDir", required_argument, nullptr, OptionId::IMAGE_DIR},
        {"outputDir", required_argument, nullptr, OptionId::OUTPUT_DIR}, {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case OptionId::HELP: args.help = true; return true;
        case OptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case OptionId::IMAGE_DIR: args.imageDir = optarg; break;
        case OptionId::OUTPUT_DIR: args.outputDir = optarg; break;
        default: return false;
        }
    }
    return true;
}

std::vector<fs::path> collectImagePaths(fs::path const& imageDir)
{
    std::vector<fs::path> imagePaths;
    for (auto const& entry : fs::directory_iterator(imageDir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
        {
            imagePaths.push_back(entry.path());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

std::string makeNpyHeader(std::string const& descr, std::vector<int64_t> const& shape)
{
    std::ostringstream shapeStream;
    shapeStream << "(";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        shapeStream << shape[i];
        if (shape.size() == 1)
        {
            shapeStream << ",";
        }
        else if (i + 1 < shape.size())
        {
            shapeStream << ", ";
        }
    }
    shapeStream << ")";

    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeStream.str() + ", }";
    size_t preamble = 10; // magic(6) + version(2) + header_len(2)
    size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    if (padding == 16)
    {
        padding = 0;
    }
    header.append(padding, ' ');
    header.push_back('\n');
    return header;
}

void writeNpy(
    fs::path const& outputPath, std::string const& descr, std::vector<int64_t> const& shape, void const* data, size_t bytes)
{
    std::ofstream out(outputPath, std::ios::binary);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + outputPath.string());
    }

    std::string const header = makeNpyHeader(descr, shape);
    char const magic[] = "\x93NUMPY";
    uint8_t const major = 1;
    uint8_t const minor = 0;
    uint16_t const headerLen = static_cast<uint16_t>(header.size());

    out.write(magic, 6);
    out.write(reinterpret_cast<char const*>(&major), sizeof(major));
    out.write(reinterpret_cast<char const*>(&minor), sizeof(minor));
    out.write(reinterpret_cast<char const*>(&headerLen), sizeof(headerLen));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(bytes));
}

template <typename T>
std::vector<T> copyTensorToHost(rt::Tensor const& tensor)
{
    std::vector<T> host(static_cast<size_t>(tensor.getShape().volume()));
    size_t const bytes = host.size() * sizeof(T);

    if (tensor.getDeviceType() == rt::DeviceType::kCPU)
    {
        std::memcpy(host.data(), tensor.rawPointer(), bytes);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.rawPointer(), bytes, cudaMemcpyDeviceToHost));
    }
    return host;
}

std::vector<float> halfToFloat(std::vector<half> const& src)
{
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = __half2float(src[i]);
    }
    return dst;
}

} // namespace

int main(int argc, char** argv)
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

    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);

    fs::create_directories(args.outputDir);
    std::vector<fs::path> imagePaths = collectImagePaths(args.imageDir);
    if (imagePaths.empty())
    {
        throw std::runtime_error("Expected at least 1 image in " + args.imageDir + ", got " + std::to_string(imagePaths.size()));
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    try
    {
        rt::LLMGenerationRequest request;
        request.requests.resize(1);
        auto& imageBuffers = request.requests[0].imageBuffers;
        imageBuffers.reserve(imagePaths.size());
        for (auto const& imagePath : imagePaths)
        {
            imageBuffers.push_back(rt::imageUtils::loadImageFromFile(imagePath.string()));
        }

        rt::QwenViTRunner runner(args.engineDir, 1, 8192, stream);
        if (!runner.debugPrepareImages(request, stream))
        {
            throw std::runtime_error("debugPrepareImages failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const pixelValuesPreInferHalf = copyTensorToHost<half>(runner.getVitInputTensor());
        auto const pixelValuesPreInfer = halfToFloat(pixelValuesPreInferHalf);
        writeNpy(fs::path(args.outputDir) / "pixel_values_pre_infer.npy", "<f4",
            {runner.getVitInputTensor().getShape()[0], runner.getVitInputTensor().getShape()[1]},
            pixelValuesPreInfer.data(), pixelValuesPreInfer.size() * sizeof(float));

        if (!runner.infer(stream))
        {
            throw std::runtime_error("visual infer failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const pixelValuesPostInferHalf = copyTensorToHost<half>(runner.getVitInputTensor());
        auto const pixelValuesPostInfer = halfToFloat(pixelValuesPostInferHalf);
        writeNpy(fs::path(args.outputDir) / "pixel_values.npy", "<f4",
            {runner.getVitInputTensor().getShape()[0], runner.getVitInputTensor().getShape()[1]},
            pixelValuesPostInfer.data(), pixelValuesPostInfer.size() * sizeof(float));
        writeNpy(fs::path(args.outputDir) / "pixel_values_post_infer.npy", "<f4",
            {runner.getVitInputTensor().getShape()[0], runner.getVitInputTensor().getShape()[1]},
            pixelValuesPostInfer.data(), pixelValuesPostInfer.size() * sizeof(float));

        std::vector<int64_t> imageGridFlat;
        auto const& imageGridTHW = runner.getLastImageGridTHWs();
        imageGridFlat.reserve(imageGridTHW.size() * 3);
        for (auto const& grid : imageGridTHW)
        {
            imageGridFlat.insert(imageGridFlat.end(), grid.begin(), grid.end());
        }
        writeNpy(fs::path(args.outputDir) / "image_grid_thw.npy", "<i8", {static_cast<int64_t>(imageGridTHW.size()), 3},
            imageGridFlat.data(), imageGridFlat.size() * sizeof(int64_t));

        auto const poolerOutput = copyTensorToHost<half>(runner.getOutputEmbedding());
        auto const poolerShape = runner.getOutputEmbedding().getShape();
        writeNpy(fs::path(args.outputDir) / "pooler_output_flat.npy", "<f2", {poolerShape[0], poolerShape[1]},
            poolerOutput.data(), poolerOutput.size() * sizeof(half));

        auto const deepstackRefs = runner.getDeepstackFeatures();
        int64_t totalDeepstackRows = 0;
        int64_t hiddenDim = 0;
        for (auto const& ref : deepstackRefs)
        {
            auto const shape = ref.get().getShape();
            totalDeepstackRows += shape[0];
            hiddenDim = shape[1];
        }
        std::vector<half> deepstackConcat;
        deepstackConcat.reserve(static_cast<size_t>(totalDeepstackRows * hiddenDim));
        for (auto const& ref : deepstackRefs)
        {
            auto host = copyTensorToHost<half>(ref.get());
            deepstackConcat.insert(deepstackConcat.end(), host.begin(), host.end());
        }
        writeNpy(fs::path(args.outputDir) / "deepstack_features.npy", "<f2", {totalDeepstackRows, hiddenDim},
            deepstackConcat.data(), deepstackConcat.size() * sizeof(half));

        auto const lastHiddenStateRef = runner.getLastHiddenState();
        bool lastHiddenStateAvailable = lastHiddenStateRef.has_value();
        if (lastHiddenStateAvailable)
        {
            auto const lastHiddenState = copyTensorToHost<half>(lastHiddenStateRef->get());
            auto const lastHiddenShape = lastHiddenStateRef->get().getShape();
            writeNpy(fs::path(args.outputDir) / "last_hidden_state.npy", "<f2", {lastHiddenShape[0], lastHiddenShape[1]},
                lastHiddenState.data(), lastHiddenState.size() * sizeof(half));
        }

        Json manifest{
            {"engine_dir", args.engineDir},
            {"image_dir", args.imageDir},
            {"images", Json::array()},
            {"pixel_values_shape", {runner.getVitInputTensor().getShape()[0], runner.getVitInputTensor().getShape()[1]}},
            {"pixel_values_pre_infer_file", "pixel_values_pre_infer.npy"},
            {"pixel_values_post_infer_file", "pixel_values_post_infer.npy"},
            {"image_grid_thw_shape", {static_cast<int64_t>(imageGridTHW.size()), 3}},
            {"pooler_output_flat_shape", {poolerShape[0], poolerShape[1]}},
            {"deepstack_features_shape", {totalDeepstackRows, hiddenDim}},
            {"last_hidden_state_available", lastHiddenStateAvailable},
            {"last_hidden_state_reason",
                lastHiddenStateAvailable ? ""
                                         : "engine does not export last_hidden_state; release visual engines only "
                                           "export output and deepstack_features"},
        };
        if (lastHiddenStateAvailable)
        {
            auto const lastHiddenShape = lastHiddenStateRef->get().getShape();
            manifest["last_hidden_state_shape"] = {lastHiddenShape[0], lastHiddenShape[1]};
        }
        for (auto const& imagePath : imagePaths)
        {
            manifest["images"].push_back(imagePath.string());
        }
        std::ofstream manifestFile(fs::path(args.outputDir) / "manifest.json");
        manifestFile << manifest.dump(2) << std::endl;
    }
    catch (...)
    {
        cudaStreamDestroy(stream);
        throw;
    }

    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
