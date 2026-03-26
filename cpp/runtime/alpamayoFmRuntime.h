/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/llmRuntimeUtils.h"

#include <NvInfer.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct AlpamayoFmBranchSnapshot
{
    std::vector<uint8_t> kvCacheBytes;
    Coords kvCacheShape;
    nvinfer1::DataType kvCacheDataType{nvinfer1::DataType::kHALF};
    int32_t activeLen{0};
    int64_t ropeDelta{0};
};

struct AlpamayoFmRunConfig
{
    int32_t numSteps{10};
    uint64_t seed{42};
    float guidanceWeight{3.0F};
};

struct AlpamayoFmRunResult
{
    struct Timing
    {
        float branchPrepareMs{0.0F};
        float kvConvertMs{0.0F};
        float kvAllocMs{0.0F};
        float kvCopyMs{0.0F};
        float maskBuildMs{0.0F};
        float maskAllocMs{0.0F};
        float maskCopyMs{0.0F};
        float positionBuildMs{0.0F};
        float positionAllocMs{0.0F};
        float positionCopyMs{0.0F};
        float x0InitMs{0.0F};
        float engineStepTotalMs{0.0F};
        float engineStepAvgMs{0.0F};
        float decodePostprocessMs{0.0F};
        float totalMs{0.0F};
        int32_t numSteps{0};
        int32_t numBranches{1};
    };

    std::vector<float> x0;
    std::vector<float> xFinal;
    std::vector<float> predXyz;
    std::vector<float> predRot;
    int32_t horizon{64};
    int32_t actionDim{2};
    Timing timing;
};

class AlpamayoFmRuntime
{
public:
    explicit AlpamayoFmRuntime(std::string enginePath);
    ~AlpamayoFmRuntime();

    AlpamayoFmRuntime(AlpamayoFmRuntime const&) = delete;
    AlpamayoFmRuntime& operator=(AlpamayoFmRuntime const&) = delete;

    bool runNoNav(AlpamayoFmBranchSnapshot const& branch, LLMGenerationRequest::ActionSpaceConstants const& constants,
        std::vector<float> const& egoHistoryXyz, std::vector<int64_t> const& egoHistoryXyzShape,
        std::vector<float> const& egoHistoryRot, std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config,
        AlpamayoFmRunResult& result);

    bool runNavCfg(AlpamayoFmBranchSnapshot const& guided, AlpamayoFmBranchSnapshot const& unguided,
        LLMGenerationRequest::ActionSpaceConstants const& constants, std::vector<float> const& egoHistoryXyz,
        std::vector<int64_t> const& egoHistoryXyzShape, std::vector<float> const& egoHistoryRot,
        std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config, AlpamayoFmRunResult& result);

    int32_t maxSeqLen() const noexcept;
    int32_t horizon() const noexcept;
    std::string const& enginePath() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace rt
} // namespace trt_edgellm
