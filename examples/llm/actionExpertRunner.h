#pragma once

#include "actionToTrajExact.h"
#include "common/tensor.h"

#include <NvInferRuntime.h>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace examples
{

struct ActionExpertConfig
{
    std::filesystem::path enginePath;
    int32_t numSteps{10};
    int32_t seed{42};
    float dtValue{0.1F};
    float accelMean{0.0F};
    float accelStd{1.0F};
    float curvatureMean{0.0F};
    float curvatureStd{1.0F};
    float vLambda{1.0e-6F};
    float vRidge{1.0e-4F};
};

struct ActionExpertMetrics
{
    bool integratedGpu{false};
    int64_t inputTokenCount{0};
    int64_t cachedGeneratedTokenCount{0};
    int64_t outputTokenCount{0};
    int64_t trajFutureStartOutputIndex{-1};
    int64_t hookActiveLen{0};
    int64_t runtimeSeqLen{0};
    int64_t offset{0};
    int64_t ropeDelta{0};
    std::vector<double> stepTimesMs;
    double meanStepMs{0.0};
    double maxStepMs{0.0};
    double flowMatchingDecoderMs{0.0};
    double trajMs{0.0};
    double totalMs{0.0};
    double flowMatchingGpuUsedMb{0.0};
    double flowMatchingCpuRssMb{0.0};
    double flowMatchingUnifiedUsedMb{0.0};
    double postProcessingGpuUsedMb{0.0};
    double postProcessingCpuRssMb{0.0};
    double postProcessingUnifiedUsedMb{0.0};
};

struct ActionExpertResult
{
    std::vector<std::array<float, 2>> xFinal;
    TrajectoryDecodeResult trajectory;
    ActionExpertMetrics metrics;
};

class ActionExpertRunner
{
public:
    explicit ActionExpertRunner(std::filesystem::path const& enginePath);
    ~ActionExpertRunner() = default;

    bool run(rt::Tensor const& fullKVCache, int32_t maxBatchSize, int32_t batchIdx, int32_t hookActiveLen,
        int64_t ropeDelta, int32_t trajFutureStartTokenId, std::vector<int32_t> const& outputTokenIds,
        common::NpyArrayFloat32 const& egoHistoryXYZ, common::NpyArrayFloat32 const& egoHistoryRot,
        ActionExpertConfig const& config, ActionExpertResult& result, std::string& errorMessage, cudaStream_t stream);

    int32_t getFutureTokenCount() const
    {
        return mFutureTokenCount;
    }

    int32_t getTargetSeqLen() const
    {
        return mTargetSeqLen;
    }

private:
    using Tensor = rt::Tensor;

    bool initializeBindings();
    bool buildKVCacheInput(rt::Tensor const& fullKVCache, int32_t maxBatchSize, int32_t batchIdx, int32_t runtimeSeqLen,
        cudaStream_t stream);
    bool buildPositionIdsAndAttentionMask(int64_t ropeDelta, int64_t offset, int64_t runtimeSeqLen, cudaStream_t stream);
    bool initializeX(int32_t seed, cudaStream_t stream);
    bool executeStep(float tValue, float dtValue, cudaStream_t stream);
    bool copyFinalXToHost(std::vector<std::array<float, 2>>& xFinal, cudaStream_t stream);

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    rt::Tensor mContextMemory{};

    rt::Tensor mX{};
    rt::Tensor mT{};
    rt::Tensor mDt{};
    rt::Tensor mKVCache{};
    rt::Tensor mAttentionMask{};
    rt::Tensor mPositionIds{};
    rt::Tensor mNextX{};
    rt::Tensor mV{};
    rt::Tensor mFutureTokenEmbeds{};

    rt::Tensor mHostScalar{};
    rt::Tensor mHostPositionIds{};
    rt::Tensor mHostAttentionMask{};
    rt::Tensor mHostX{};
    rt::Tensor mHostFinalX{};

    int32_t mFutureTokenCount{0};
    int32_t mTargetSeqLen{0};
    int32_t mKvNumLayers{0};
    int32_t mKvNumKv{0};
    int32_t mKvNumHeads{0};
    int32_t mKvHeadDim{0};
    nvinfer1::DataType mKVCacheInputType{nvinfer1::DataType::kBF16};
};

bool copyAndPadKVCache(rt::Tensor const& src, int32_t srcBatchSize, int32_t batchIdx, int32_t validSeqLen,
    rt::Tensor& dst, cudaStream_t stream);

} // namespace examples
} // namespace trt_edgellm
