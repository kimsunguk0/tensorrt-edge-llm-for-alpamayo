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

#include "common/npyUtils.h"

#include "common/checkMacros.h"
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace common
{

namespace
{

template <typename T>
T readLE(std::istream& input)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    check::check(input.good(), "Failed to read NPY header.");
    return value;
}

std::string trim(std::string const& input)
{
    size_t begin = input.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos)
    {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\n\r");
    return input.substr(begin, end - begin + 1);
}

std::vector<int64_t> parseShape(std::string const& shapeText)
{
    std::vector<int64_t> shape;
    std::stringstream ss(shapeText);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        std::string const value = trim(item);
        if (value.empty())
        {
            continue;
        }
        shape.push_back(std::stoll(value));
    }
    check::check(!shape.empty(), "NPY shape must not be empty.");
    return shape;
}

std::vector<int64_t> computeCOrderStrides(std::vector<int64_t> const& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

std::vector<int64_t> computeFortranOrderStrides(std::vector<int64_t> const& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = 1; i < shape.size(); ++i)
    {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

std::vector<float> convertFortranToCOrder(std::vector<float> const& rawData, std::vector<int64_t> const& shape)
{
    std::vector<int64_t> const cStrides = computeCOrderStrides(shape);
    std::vector<int64_t> const fStrides = computeFortranOrderStrides(shape);
    std::vector<float> reordered(rawData.size());

    for (size_t linearIndex = 0; linearIndex < rawData.size(); ++linearIndex)
    {
        int64_t remaining = static_cast<int64_t>(linearIndex);
        int64_t sourceIndex = 0;
        for (size_t dim = 0; dim < shape.size(); ++dim)
        {
            int64_t const idx = remaining / cStrides[dim];
            remaining %= cStrides[dim];
            sourceIndex += idx * fStrides[dim];
        }
        reordered[linearIndex] = rawData[static_cast<size_t>(sourceIndex)];
    }

    return reordered;
}

} // namespace

NpyArrayFloat32 loadNpyFloat32(std::filesystem::path const& filePath)
{
    std::ifstream input(filePath, std::ios::binary);
    check::check(input.is_open(), "Failed to open NPY file: " + filePath.string());

    char magic[6]{};
    input.read(magic, sizeof(magic));
    check::check(input.good(), "Failed to read NPY magic header.");
    check::check(std::string(magic, sizeof(magic)) == std::string("\x93NUMPY", 6),
        "Invalid NPY magic header in file: " + filePath.string());

    uint8_t const major = readLE<uint8_t>(input);
    uint8_t const minor = readLE<uint8_t>(input);
    uint32_t headerLength = 0;
    if (major == 1)
    {
        headerLength = readLE<uint16_t>(input);
    }
    else if (major == 2)
    {
        headerLength = readLE<uint32_t>(input);
    }
    else
    {
        throw std::runtime_error("Unsupported NPY version: " + std::to_string(major) + "." + std::to_string(minor));
    }

    std::string header(headerLength, '\0');
    input.read(header.data(), static_cast<std::streamsize>(headerLength));
    check::check(input.good(), "Failed to read NPY header body.");

    std::regex const descrRegex("'descr'\\s*:\\s*'([^']+)'");
    std::regex const fortranRegex("'fortran_order'\\s*:\\s*(True|False)");
    std::regex const shapeRegex("'shape'\\s*:\\s*\\(([^\\)]*)\\)");
    std::smatch match;

    check::check(std::regex_search(header, match, descrRegex), "Failed to parse NPY dtype.");
    std::string const dtype = match[1].str();
    check::check(dtype == "<f4" || dtype == "|f4", "Only float32 NPY files are supported. Got dtype=" + dtype);

    check::check(std::regex_search(header, match, fortranRegex), "Failed to parse NPY fortran_order.");
    bool const isFortranOrder = match[1].str() == "True";

    check::check(std::regex_search(header, match, shapeRegex), "Failed to parse NPY shape.");
    std::vector<int64_t> const shape = parseShape(match[1].str());

    int64_t const elementCount
        = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
    check::check(elementCount >= 0, "Invalid NPY element count.");

    NpyArrayFloat32 array;
    array.shape = shape;
    array.data.resize(static_cast<size_t>(elementCount));
    input.read(reinterpret_cast<char*>(array.data.data()), static_cast<std::streamsize>(elementCount * sizeof(float)));
    check::check(input.good(), "Failed to read NPY payload from file: " + filePath.string());

    if (isFortranOrder)
    {
        array.data = convertFortranToCOrder(array.data, array.shape);
    }

    return array;
}

} // namespace common
} // namespace trt_edgellm
