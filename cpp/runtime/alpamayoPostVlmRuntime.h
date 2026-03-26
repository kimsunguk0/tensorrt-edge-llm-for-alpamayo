/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/alpamayoFmRuntime.h"
#include "runtime/llmInferenceRuntime.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{

struct AlpamayoPostVlmOptions
{
    bool enableNavCfg{false};
    bool dumpNavDualCache{false};
    std::filesystem::path navCacheOutputDir{"./output/nav_dual_cache"};
    std::string fmEngine{};
    float defaultNavGuidanceWeight{3.0F};
};

class AlpamayoPostVlmRuntime
{
public:
    explicit AlpamayoPostVlmRuntime(AlpamayoPostVlmOptions options);

    bool handleRequest(LLMInferenceRuntime& runtime, LLMGenerationRequest const& request,
        LLMGenerationResponse& response, nlohmann::json& responseJson, cudaStream_t stream);

private:
    struct BranchArtifacts
    {
        LLMGenerationRequest request;
        LLMGenerationResponse response;
        std::string decodedPrefixText;
        int32_t kvActiveLen{0};
        std::filesystem::path kvDir{};
        std::optional<AlpamayoFmBranchSnapshot> fmSnapshot{};
    };

    AlpamayoPostVlmOptions mOptions;
    std::unique_ptr<AlpamayoFmRuntime> mFmRuntime;

    static std::string removeRouteSpan(std::string const& text);
    static bool insertRouteSpan(std::string& text, std::string const& navText);
    static bool appendAssistantReplayPrefix(LLMGenerationRequest::Request& request, std::string const& replayText);
    static std::optional<std::string> getEffectiveNavText(LLMGenerationRequest const& request);
    static float getEffectiveNavGuidanceWeight(LLMGenerationRequest const& request, float defaultWeight);
    static int32_t readActiveKvLength(LLMInferenceRuntime& runtime, size_t activeBatchSize, cudaStream_t stream);
    static std::optional<LLMGenerationRequest::ActionSpaceConstants> getActionSpaceConstants(
        LLMGenerationRequest const& request);
    static AlpamayoFmRunConfig getFmRunConfig(LLMGenerationRequest const& request, float defaultGuidanceWeight);
    static bool loadEgoHistory(LLMGenerationRequest const& request, std::vector<float>& egoHistoryXyz,
        std::vector<int64_t>& egoHistoryXyzShape, std::vector<float>& egoHistoryRot, std::vector<int64_t>& egoHistoryRotShape);
    static nlohmann::json tensorToJson(std::vector<float> const& values, std::vector<int64_t> const& shape);
    static std::optional<AlpamayoFmBranchSnapshot> captureFmSnapshot(LLMInferenceRuntime& runtime, size_t activeBatchSize,
        cudaStream_t stream);

    static bool dumpKvSnapshot(LLMInferenceRuntime& runtime, std::filesystem::path const& outputDir, size_t requestIdx,
        size_t activeBatchSize, cudaStream_t stream);

    BranchArtifacts runGuidedPass(LLMInferenceRuntime& runtime, LLMGenerationRequest const& request,
        std::string const& navText, size_t requestIdx, cudaStream_t stream);
    BranchArtifacts runUnguidedReplayPass(LLMInferenceRuntime& runtime, LLMGenerationRequest const& request,
        std::string const& guidedPrefixText, size_t requestIdx, cudaStream_t stream);
};

} // namespace rt
} // namespace trt_edgellm
