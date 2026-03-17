#include "actionExpertRunner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fstream>
#include <limits>
#include <random>
#include <unistd.h>

namespace trt_edgellm
{
namespace examples
{
namespace
{

struct MemorySnapshot
{
    bool integratedGpu{false};
    double gpuUsedMb{0.0};
    double cpuRssMb{0.0};
    double unifiedUsedMb{0.0};
};

MemorySnapshot sampleMemorySnapshot()
{
    MemorySnapshot snapshot;
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int integrated = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
    snapshot.integratedGpu = integrated == 1;

    long const pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize > 0)
    {
        std::ifstream statm("/proc/self/statm");
        long totalPages = 0;
        long rssPages = 0;
        if (statm.is_open() && (statm >> totalPages >> rssPages))
        {
            size_t const rssBytes = static_cast<size_t>(rssPages) * static_cast<size_t>(pageSize);
            snapshot.cpuRssMb = rt::utils::toMB(rssBytes);
            if (snapshot.integratedGpu)
            {
                snapshot.unifiedUsedMb = snapshot.cpuRssMb;
            }
        }
    }

    size_t freeMem = 0;
    size_t totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    snapshot.gpuUsedMb = rt::utils::toMB(totalMem - freeMem);

    return snapshot;
}

int64_t computeOffset(int32_t hookActiveLen, int32_t trajFutureStartTokenId, std::vector<int32_t> const& outputTokenIds,
    ActionExpertMetrics& metrics)
{
    metrics.outputTokenCount = static_cast<int64_t>(outputTokenIds.size());
    metrics.cachedGeneratedTokenCount = std::max<int64_t>(metrics.outputTokenCount - 1, 0);
    metrics.inputTokenCount = static_cast<int64_t>(hookActiveLen) - metrics.cachedGeneratedTokenCount;
    check::check(metrics.inputTokenCount >= 0, "Invalid action-expert handoff: negative input token count.");

    auto it = std::find(outputTokenIds.begin(), outputTokenIds.end(), trajFutureStartTokenId);
    if (it != outputTokenIds.end())
    {
        metrics.trajFutureStartOutputIndex = static_cast<int64_t>(std::distance(outputTokenIds.begin(), it));
    }
    else
    {
        metrics.trajFutureStartOutputIndex = metrics.outputTokenCount > 0 ? metrics.outputTokenCount - 1 : -1;
    }

    return metrics.inputTokenCount + metrics.trajFutureStartOutputIndex + 1;
}

void writeScalar(rt::Tensor& hostScalar, float value)
{
    hostScalar.reshape({1, 1, 1});
    hostScalar.dataPointer<float>()[0] = value;
}

} // namespace

ActionExpertRunner::ActionExpertRunner(std::filesystem::path const& enginePath)
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    check::check(mRuntime != nullptr, "Failed to create TensorRT runtime for action expert.");

    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    check::check(mmapReader->getData() != nullptr,
        "Failed to map action-expert engine: " + enginePath.string());
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    check::check(mEngine != nullptr, "Failed to deserialize action-expert engine: " + enginePath.string());

    int64_t const contextBytes = static_cast<int64_t>(mEngine->getDeviceMemorySizeV2());
    mContextMemory = rt::Tensor({contextBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "ActionExpertRunner::mContextMemory");
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    check::check(mContext != nullptr, "Failed to create action-expert execution context.");
    mContext->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);

    initializeBindings();
}

bool ActionExpertRunner::initializeBindings()
{
    auto const getShape = [this](char const* name) {
        return rt::Coords(mEngine->getTensorShape(name));
    };
    auto const getType = [this](char const* name) { return mEngine->getTensorDataType(name); };

    rt::Coords const xShape = getShape("x");
    rt::Coords const kvShape = getShape("kv_cache");
    rt::Coords const maskShape = getShape("attention_mask");
    rt::Coords const posShape = getShape("position_ids");
    rt::Coords const nextXShape = getShape("next_x");
    rt::Coords const vShape = getShape("v");
    rt::Coords const futureShape = getShape("future_token_embeds");

    check::check(xShape.getNumDims() == 3 && xShape[0] == 1 && xShape[2] == 2,
        "Unexpected action-expert x shape.");
    check::check(kvShape.getNumDims() == 6 && kvShape[1] == 1,
        "Unexpected action-expert kv_cache shape.");
    check::check(maskShape.getNumDims() == 4 && maskShape[0] == 1 && maskShape[1] == 1,
        "Unexpected action-expert attention_mask shape.");
    check::check(posShape.getNumDims() == 3 && posShape[0] == 3 && posShape[1] == 1,
        "Unexpected action-expert position_ids shape.");

    mFutureTokenCount = static_cast<int32_t>(xShape[1]);
    mTargetSeqLen = static_cast<int32_t>(kvShape[4]);
    mKvNumLayers = static_cast<int32_t>(kvShape[0]);
    mKvNumKv = static_cast<int32_t>(kvShape[2]);
    mKvNumHeads = static_cast<int32_t>(kvShape[3]);
    mKvHeadDim = static_cast<int32_t>(kvShape[5]);
    mKVCacheInputType = getType("kv_cache");

    mX = rt::Tensor(xShape, rt::DeviceType::kGPU, getType("x"), "ActionExpertRunner::mX");
    mT = rt::Tensor(getShape("t"), rt::DeviceType::kGPU, getType("t"), "ActionExpertRunner::mT");
    mDt = rt::Tensor(getShape("dt"), rt::DeviceType::kGPU, getType("dt"), "ActionExpertRunner::mDt");
    mKVCache = rt::Tensor(kvShape, rt::DeviceType::kGPU, mKVCacheInputType, "ActionExpertRunner::mKVCache");
    mAttentionMask = rt::Tensor(maskShape, rt::DeviceType::kGPU, getType("attention_mask"),
        "ActionExpertRunner::mAttentionMask");
    mPositionIds = rt::Tensor(posShape, rt::DeviceType::kGPU, getType("position_ids"),
        "ActionExpertRunner::mPositionIds");
    mNextX = rt::Tensor(nextXShape, rt::DeviceType::kGPU, getType("next_x"), "ActionExpertRunner::mNextX");
    mV = rt::Tensor(vShape, rt::DeviceType::kGPU, getType("v"), "ActionExpertRunner::mV");
    mFutureTokenEmbeds = rt::Tensor(
        futureShape, rt::DeviceType::kGPU, getType("future_token_embeds"), "ActionExpertRunner::mFutureTokenEmbeds");

    mHostScalar = rt::Tensor({1, 1, 1}, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT,
        "ActionExpertRunner::mHostScalar");
    mHostPositionIds = rt::Tensor(posShape, rt::DeviceType::kCPU, getType("position_ids"),
        "ActionExpertRunner::mHostPositionIds");
    mHostAttentionMask = rt::Tensor(maskShape, rt::DeviceType::kCPU, getType("attention_mask"),
        "ActionExpertRunner::mHostAttentionMask");
    mHostX = rt::Tensor(xShape, rt::DeviceType::kCPU, getType("x"), "ActionExpertRunner::mHostX");
    mHostFinalX = rt::Tensor(nextXShape, rt::DeviceType::kCPU, getType("next_x"), "ActionExpertRunner::mHostFinalX");

    bool ok = true;
    ok &= mContext->setInputShape("x", xShape.getTRTDims());
    ok &= mContext->setInputShape("t", getShape("t").getTRTDims());
    ok &= mContext->setInputShape("dt", getShape("dt").getTRTDims());
    ok &= mContext->setInputShape("kv_cache", kvShape.getTRTDims());
    ok &= mContext->setInputShape("attention_mask", maskShape.getTRTDims());
    ok &= mContext->setInputShape("position_ids", posShape.getTRTDims());
    ok &= mContext->setTensorAddress("x", mX.rawPointer());
    ok &= mContext->setTensorAddress("t", mT.rawPointer());
    ok &= mContext->setTensorAddress("dt", mDt.rawPointer());
    ok &= mContext->setTensorAddress("kv_cache", mKVCache.rawPointer());
    ok &= mContext->setTensorAddress("attention_mask", mAttentionMask.rawPointer());
    ok &= mContext->setTensorAddress("position_ids", mPositionIds.rawPointer());
    ok &= mContext->setTensorAddress("next_x", mNextX.rawPointer());
    ok &= mContext->setTensorAddress("v", mV.rawPointer());
    ok &= mContext->setTensorAddress("future_token_embeds", mFutureTokenEmbeds.rawPointer());
    check::check(ok, "Failed to bind action-expert engine tensors.");
    return true;
}

bool ActionExpertRunner::buildKVCacheInput(
    rt::Tensor const& fullKVCache, int32_t maxBatchSize, int32_t batchIdx, int32_t runtimeSeqLen, cudaStream_t stream)
{
    return copyAndPadKVCache(fullKVCache, maxBatchSize, batchIdx, runtimeSeqLen, mKVCache, stream);
}

bool ActionExpertRunner::buildPositionIdsAndAttentionMask(
    int64_t ropeDelta, int64_t offset, int64_t runtimeSeqLen, cudaStream_t stream)
{
    int64_t* positionIds = mHostPositionIds.dataPointer<int64_t>();
    for (int32_t axis = 0; axis < 3; ++axis)
    {
        for (int32_t tokenIdx = 0; tokenIdx < mFutureTokenCount; ++tokenIdx)
        {
            positionIds[axis * mFutureTokenCount + tokenIdx] = ropeDelta + offset + tokenIdx;
        }
    }

    float* attentionMask = mHostAttentionMask.dataPointer<float>();
    size_t const attentionMaskSize = static_cast<size_t>(mHostAttentionMask.getShape().volume());
    std::fill(attentionMask, attentionMask + attentionMaskSize, 0.0F);
    if (offset < runtimeSeqLen)
    {
        float const maskValue = std::numeric_limits<float>::lowest();
        int32_t const totalLen = static_cast<int32_t>(mHostAttentionMask.getShape()[3]);
        for (int32_t q = 0; q < mFutureTokenCount; ++q)
        {
            size_t const rowBase = static_cast<size_t>(q) * static_cast<size_t>(totalLen);
            for (int64_t seq = offset; seq < runtimeSeqLen; ++seq)
            {
                attentionMask[rowBase + static_cast<size_t>(seq)] = maskValue;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mPositionIds.rawPointer(), mHostPositionIds.rawPointer(), mHostPositionIds.getMemoryCapacity(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mAttentionMask.rawPointer(), mHostAttentionMask.rawPointer(),
        mHostAttentionMask.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    return true;
}

bool ActionExpertRunner::initializeX(int32_t seed, cudaStream_t stream)
{
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    float* hostX = mHostX.dataPointer<float>();
    for (int64_t i = 0; i < mHostX.getShape().volume(); ++i)
    {
        hostX[i] = distribution(generator);
    }
    CUDA_CHECK(cudaMemcpyAsync(mX.rawPointer(), mHostX.rawPointer(), mHostX.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    return true;
}

bool ActionExpertRunner::executeStep(float tValue, float dtValue, cudaStream_t stream)
{
    writeScalar(mHostScalar, tValue);
    CUDA_CHECK(cudaMemcpyAsync(mT.rawPointer(), mHostScalar.rawPointer(), mHostScalar.getMemoryCapacity(),
        cudaMemcpyHostToDevice, stream));
    writeScalar(mHostScalar, dtValue);
    CUDA_CHECK(cudaMemcpyAsync(mDt.rawPointer(), mHostScalar.rawPointer(), mHostScalar.getMemoryCapacity(),
        cudaMemcpyHostToDevice, stream));

    if (!mContext->enqueueV3(stream))
    {
        return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(mX.rawPointer(), mNextX.rawPointer(), mX.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));
    return true;
}

bool ActionExpertRunner::copyFinalXToHost(std::vector<std::array<float, 2>>& xFinal, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(
        mHostFinalX.rawPointer(), mX.rawPointer(), mX.getMemoryCapacity(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float const* hostFinalX = mHostFinalX.dataPointer<float>();
    xFinal.resize(static_cast<size_t>(mFutureTokenCount));
    for (int32_t i = 0; i < mFutureTokenCount; ++i)
    {
        xFinal[static_cast<size_t>(i)]
            = {hostFinalX[static_cast<size_t>(i) * 2], hostFinalX[static_cast<size_t>(i) * 2 + 1]};
    }
    return true;
}

bool ActionExpertRunner::run(rt::Tensor const& fullKVCache, int32_t maxBatchSize, int32_t batchIdx, int32_t hookActiveLen,
    int64_t ropeDelta, int32_t trajFutureStartTokenId, std::vector<int32_t> const& outputTokenIds,
    common::NpyArrayFloat32 const& egoHistoryXYZ, common::NpyArrayFloat32 const& egoHistoryRot,
    ActionExpertConfig const& config, ActionExpertResult& result, std::string& errorMessage, cudaStream_t stream)
{
    try
    {
        check::check(batchIdx == 0, "ActionExpertRunner currently supports batchIdx == 0 only.");
        check::check(maxBatchSize >= 1, "Invalid maxBatchSize for action expert.");
        check::check(hookActiveLen > 0, "Action expert requires positive hookActiveLen.");

        result = {};
        result.metrics.hookActiveLen = hookActiveLen;
        result.metrics.ropeDelta = ropeDelta;
        result.metrics.offset = computeOffset(hookActiveLen, trajFutureStartTokenId, outputTokenIds, result.metrics);
        result.metrics.runtimeSeqLen = std::min<int64_t>(result.metrics.offset, hookActiveLen);
        check::check(result.metrics.runtimeSeqLen <= mTargetSeqLen,
            "Action expert runtime seq len exceeds engine target seq len.");

        auto totalStart = std::chrono::steady_clock::now();
        buildKVCacheInput(fullKVCache, maxBatchSize, batchIdx, static_cast<int32_t>(result.metrics.runtimeSeqLen), stream);
        buildPositionIdsAndAttentionMask(
            ropeDelta, result.metrics.offset, result.metrics.runtimeSeqLen, stream);
        initializeX(config.seed, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> timeSteps(static_cast<size_t>(config.numSteps + 1), 0.0F);
        for (int32_t i = 0; i <= config.numSteps; ++i)
        {
            timeSteps[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(config.numSteps);
        }

        result.metrics.stepTimesMs.reserve(static_cast<size_t>(config.numSteps));
        for (int32_t i = 0; i < config.numSteps; ++i)
        {
            auto stepStart = std::chrono::steady_clock::now();
            bool ok = executeStep(timeSteps[static_cast<size_t>(i)],
                timeSteps[static_cast<size_t>(i + 1)] - timeSteps[static_cast<size_t>(i)], stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            check::check(ok, "Action-expert TensorRT executeAsyncV3 failed.");
            double stepMs
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stepStart).count();
            result.metrics.stepTimesMs.push_back(stepMs);
        }

        copyFinalXToHost(result.xFinal, stream);
        result.metrics.flowMatchingDecoderMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - totalStart).count();
        auto const flowSnapshot = sampleMemorySnapshot();
        result.metrics.integratedGpu = flowSnapshot.integratedGpu;
        result.metrics.flowMatchingGpuUsedMb = flowSnapshot.gpuUsedMb;
        result.metrics.flowMatchingCpuRssMb = flowSnapshot.cpuRssMb;
        result.metrics.flowMatchingUnifiedUsedMb = flowSnapshot.unifiedUsedMb;

        auto trajStart = std::chrono::steady_clock::now();
        result.trajectory = decodeActionToTrajectory(result.xFinal, egoHistoryXYZ, egoHistoryRot, config.accelMean,
            config.accelStd, config.curvatureMean, config.curvatureStd, config.dtValue, config.vLambda,
            config.vRidge);
        result.metrics.trajMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trajStart).count();
        auto const postSnapshot = sampleMemorySnapshot();
        result.metrics.postProcessingGpuUsedMb = postSnapshot.gpuUsedMb;
        result.metrics.postProcessingCpuRssMb = postSnapshot.cpuRssMb;
        result.metrics.postProcessingUnifiedUsedMb = postSnapshot.unifiedUsedMb;
        result.metrics.totalMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - totalStart).count();
        if (!result.metrics.stepTimesMs.empty())
        {
            result.metrics.maxStepMs
                = *std::max_element(result.metrics.stepTimesMs.begin(), result.metrics.stepTimesMs.end());
            double totalStepMs = 0.0;
            for (double value : result.metrics.stepTimesMs)
            {
                totalStepMs += value;
            }
            result.metrics.meanStepMs = totalStepMs / result.metrics.stepTimesMs.size();
        }
        return true;
    }
    catch (std::exception const& e)
    {
        errorMessage = e.what();
        return false;
    }
}

} // namespace examples
} // namespace trt_edgellm
