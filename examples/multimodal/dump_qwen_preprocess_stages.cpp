/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "runtime/imageUtils.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
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
    std::string engineDir{"/models/engines/visual"};
    std::string imageDir{"/root/TensorRT-Edge-LLM/input/images"};
    std::string outputDir{"/root/TensorRT-Edge-LLM/output/visual_dumps/preprocess_stages"};
    bool help{false};
};

struct Config
{
    int64_t patchSize{};
    int64_t mergeSize{};
    int64_t temporalPatchSize{};
    int64_t minImageTokensPerImage{};
    int64_t maxImageTokensPerImage{};
    std::vector<float> imageMean;
    std::vector<float> imageStd;
};

enum OptionId : int
{
    HELP = 701,
    ENGINE_DIR = 702,
    IMAGE_DIR = 703,
    OUTPUT_DIR = 704
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

Config loadConfig(std::string const& engineDir)
{
    auto loadJson = [](fs::path const& path) -> Json {
        std::ifstream stream(path);
        if (!stream.is_open())
        {
            throw std::runtime_error("Failed to open: " + path.string());
        }
        return Json::parse(stream);
    };

    Json const config = loadJson(fs::path(engineDir) / "config.json");
    Json const preprocessorConfig = loadJson(fs::path(engineDir) / "preprocessor_config.json");

    Config out;
    out.patchSize = preprocessorConfig["patch_size"].get<int64_t>();
    out.temporalPatchSize = preprocessorConfig["temporal_patch_size"].get<int64_t>();
    out.mergeSize = preprocessorConfig["merge_size"].get<int64_t>();
    out.imageMean = preprocessorConfig["image_mean"].get<std::vector<float>>();
    out.imageStd = preprocessorConfig["image_std"].get<std::vector<float>>();

    Json const builderConfig = config["builder_config"];
    out.minImageTokensPerImage = builderConfig["min_image_tokens"].get<int64_t>();
    out.maxImageTokensPerImage = builderConfig["max_image_tokens_per_image"].get<int64_t>();
    return out;
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
    size_t preamble = 10;
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

std::tuple<int64_t, int64_t> getResizedImageSize(
    Config const& config, int64_t const height, int64_t const width, int64_t const maxRatio = 200)
{
    int64_t const factor = config.patchSize * config.mergeSize;
    int64_t const minPixels = config.minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = config.maxImageTokensPerImage * factor * factor;

    auto roundByFactor = [](int64_t value, int64_t factorValue) -> int64_t {
        return std::round(static_cast<double>(value) / factorValue) * factorValue;
    };
    auto floorByFactor = [](int64_t value, int64_t factorValue) -> int64_t {
        return std::floor(static_cast<double>(value) / factorValue) * factorValue;
    };
    auto ceilByFactor = [](int64_t value, int64_t factorValue) -> int64_t {
        return std::ceil(static_cast<double>(value) / factorValue) * factorValue;
    };

    if (std::max(height, width) / std::min(height, width) > maxRatio)
    {
        throw std::runtime_error("absolute aspect ratio must be smaller than " + std::to_string(maxRatio));
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

    Config const config = loadConfig(args.engineDir);
    std::vector<fs::path> imagePaths = collectImagePaths(args.imageDir);
    if (imagePaths.empty())
    {
        throw std::runtime_error("No images found in " + args.imageDir);
    }

    fs::create_directories(args.outputDir);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    try
    {
        rt::Tensor imageMean({static_cast<int64_t>(config.imageMean.size())}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "dump_qwen_preprocess_stages::imageMean");
        rt::Tensor imageStd({static_cast<int64_t>(config.imageStd.size())}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "dump_qwen_preprocess_stages::imageStd");
        CUDA_CHECK(cudaMemcpyAsync(imageMean.rawPointer(), config.imageMean.data(),
            config.imageMean.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(imageStd.rawPointer(), config.imageStd.data(), config.imageStd.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));

        Json manifest{
            {"engine_dir", args.engineDir},
            {"image_dir", args.imageDir},
            {"patch_size", config.patchSize},
            {"merge_size", config.mergeSize},
            {"temporal_patch_size", config.temporalPatchSize},
            {"min_image_tokens_per_image", config.minImageTokensPerImage},
            {"max_image_tokens_per_image", config.maxImageTokensPerImage},
            {"images", Json::array()},
        };

        for (auto const& imagePath : imagePaths)
        {
            auto image = rt::imageUtils::loadImageFromFile(imagePath.string());
            auto [resizedHeight, resizedWidth] = getResizedImageSize(config, image.height, image.width);

            rt::imageUtils::ImageData resized;
            // Allocate a dedicated buffer per image to keep logic simple and explicit.
            resized.buffer = std::make_shared<rt::Tensor>(rt::Tensor({resizedHeight, resizedWidth, image.channels},
                rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "dump_qwen_preprocess_stages::resized"));
            resized.height = resizedHeight;
            resized.width = resizedWidth;
            resized.channels = image.channels;
            rt::imageUtils::resizeImage(image, resized, resizedWidth, resizedHeight);

            std::string const stem = imagePath.stem().string();
            fs::path rawPath = fs::path(args.outputDir) / (stem + ".raw_u8_hwc.npy");
            fs::path resizedPath = fs::path(args.outputDir) / (stem + ".resized_u8_hwc.npy");
            fs::path normalizedPath = fs::path(args.outputDir) / (stem + ".normalized_hwc.npy");
            fs::path patchPath = fs::path(args.outputDir) / (stem + ".patch_values.npy");

            writeNpy(rawPath, "|u1", {image.height, image.width, image.channels}, image.data(),
                static_cast<size_t>(image.height * image.width * image.channels));
            writeNpy(resizedPath, "|u1", {resized.height, resized.width, resized.channels}, resized.data(),
                static_cast<size_t>(resized.height * resized.width * resized.channels));

            rt::Tensor imageDevice({config.temporalPatchSize, resized.height, resized.width, resized.channels}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kUINT8, "dump_qwen_preprocess_stages::imageDevice");
            rt::Tensor normalizedDevice({config.temporalPatchSize, resized.height, resized.width, resized.channels}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF, "dump_qwen_preprocess_stages::normalizedDevice");
            size_t const imageBytes = static_cast<size_t>(resized.height * resized.width * resized.channels);
            for (int64_t t = 0; t < config.temporalPatchSize; ++t)
            {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<unsigned char*>(imageDevice.rawPointer()) + t * imageBytes,
                    resized.data(), imageBytes, cudaMemcpyHostToDevice, stream));
            }
            kernel::normalizeImage(imageDevice, imageMean, imageStd, normalizedDevice, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto normalizedHalf = copyTensorToHost<half>(normalizedDevice);
            auto normalizedFloat = halfToFloat(normalizedHalf);
            writeNpy(normalizedPath, "<f4", {resized.height, resized.width, resized.channels}, normalizedFloat.data(),
                static_cast<size_t>(resized.height * resized.width * resized.channels) * sizeof(float));

            int64_t const seqLength
                = (resized.height / config.patchSize) * (resized.width / config.patchSize);
            int64_t const imageTokenLength = seqLength / (config.mergeSize * config.mergeSize);
            int64_t const inputDim
                = resized.channels * config.temporalPatchSize * config.patchSize * config.patchSize;
            rt::Tensor patchDevice({seqLength, inputDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
                "dump_qwen_preprocess_stages::patchDevice");
            kernel::transposeToPatchQwenViT(normalizedDevice, patchDevice, 0, config.temporalPatchSize,
                config.patchSize, config.mergeSize, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto patchHalf = copyTensorToHost<half>(patchDevice);
            auto patchFloat = halfToFloat(patchHalf);
            writeNpy(patchPath, "<f4", {seqLength, inputDim}, patchFloat.data(), patchFloat.size() * sizeof(float));

            bool rawEqualsResized = false;
            if (image.height == resized.height && image.width == resized.width && image.channels == resized.channels)
            {
                rawEqualsResized = std::memcmp(image.data(), resized.data(), imageBytes) == 0;
            }

            manifest["images"].push_back(Json{
                {"path", imagePath.string()},
                {"stem", stem},
                {"raw_shape", {image.height, image.width, image.channels}},
                {"resized_shape", {resized.height, resized.width, resized.channels}},
                {"seq_length", seqLength},
                {"image_token_length", imageTokenLength},
                {"input_dim", inputDim},
                {"raw_equals_resized", rawEqualsResized},
            });
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
