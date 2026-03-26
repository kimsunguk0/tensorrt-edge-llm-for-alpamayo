/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/alpamayoFmRuntime.h"

#include "common/checkMacros.h"

#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

namespace
{

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > Severity::kWARNING)
        {
            return;
        }
        std::cerr << "[FM TRT] " << msg << std::endl;
    }
};

template <typename T>
struct TrtDeleter
{
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;
using SteadyClock = std::chrono::steady_clock;

float elapsedMs(SteadyClock::time_point const& start, SteadyClock::time_point const& end)
{
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
}

struct CudaBuffer
{
    void* ptr{nullptr};
    size_t bytes{0};

    CudaBuffer() = default;
    explicit CudaBuffer(size_t size)
        : bytes(size)
    {
        if (bytes > 0)
        {
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
        }
    }

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr(other.ptr)
        , bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    CudaBuffer(CudaBuffer const&) = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    ~CudaBuffer()
    {
        reset();
    }

    void reset()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

size_t dataTypeSize(nvinfer1::DataType dtype)
{
    using DataType = nvinfer1::DataType;
    switch (dtype)
    {
    case DataType::kFLOAT: return sizeof(float);
    case DataType::kHALF: return 2;
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kBF16: return 2;
    case DataType::kINT64: return sizeof(int64_t);
#endif
    case DataType::kINT32: return sizeof(int32_t);
    case DataType::kBOOL: return sizeof(bool);
    default: throw std::runtime_error("Unsupported TensorRT dtype size request");
    }
}

std::string dataTypeName(nvinfer1::DataType dtype)
{
    using DataType = nvinfer1::DataType;
    switch (dtype)
    {
    case DataType::kFLOAT: return "FLOAT32";
    case DataType::kHALF: return "FLOAT16";
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kBF16: return "BF16";
    case DataType::kINT64: return "INT64";
#endif
    case DataType::kINT32: return "INT32";
    case DataType::kBOOL: return "BOOL";
    default: return "UNKNOWN";
    }
}

int64_t dimsVolume(nvinfer1::Dims const& dims)
{
    int64_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        check::check(dims.d[i] >= 0, "TensorRT dims must be static for Alpamayo FM runtime");
        volume *= dims.d[i];
    }
    return volume;
}

void memcpyToDevice(void* dst, void const* src, size_t bytes)
{
    if (bytes == 0)
    {
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void memcpyToHost(void* dst, void const* src, size_t bytes)
{
    if (bytes == 0)
    {
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

std::vector<float> makeNormalX0(int32_t horizon, int32_t actionDim, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> values(static_cast<size_t>(horizon) * actionDim);
    for (auto& v : values)
    {
        v = dist(rng);
    }
    return values;
}

std::vector<uint8_t> convertHalfBytesToBf16Bytes(std::vector<uint8_t> const& src)
{
    check::check(src.size() % sizeof(uint16_t) == 0, "FP16 KV byte size must be aligned to 2 bytes");
    size_t const count = src.size() / sizeof(uint16_t);
    auto const* srcWords = reinterpret_cast<uint16_t const*>(src.data());
    std::vector<uint16_t> dstWords(count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        __half_raw raw{};
        raw.x = srcWords[i];
        __half halfValue = *reinterpret_cast<__half*>(&raw);
        float f = __half2float(halfValue);
        __nv_bfloat16 bf = __float2bfloat16(f);
        auto const* bfRaw = reinterpret_cast<uint16_t const*>(&bf);
        dstWords[i] = *bfRaw;
    }
    std::vector<uint8_t> bytes(dstWords.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), dstWords.data(), bytes.size());
    return bytes;
}

std::vector<float> extractLastHistoryXYZ(std::vector<float> const& data, std::vector<int64_t> const& shape)
{
    check::check(!shape.empty(), "ego_history_xyz shape must not be empty");
    if (shape.size() == 4)
    {
        check::check(shape[0] == 1, "Only batch_size=1 is supported for ego_history_xyz");
        check::check(shape[3] == 3, "ego_history_xyz last dim must be 3");
        int64_t const trajGroup = shape[1] - 1;
        int64_t const history = shape[2];
        std::vector<float> out(static_cast<size_t>(history) * 3);
        for (int64_t h = 0; h < history; ++h)
        {
            for (int64_t c = 0; c < 3; ++c)
            {
                size_t const src = static_cast<size_t>(((0 * shape[1] + trajGroup) * shape[2] + h) * shape[3] + c);
                out[static_cast<size_t>(h * 3 + c)] = data[src];
            }
        }
        return out;
    }
    if (shape.size() == 3)
    {
        check::check(shape[0] == 1, "Only batch_size=1 is supported for ego_history_xyz");
        check::check(shape[2] == 3, "ego_history_xyz last dim must be 3");
        int64_t const history = shape[1];
        std::vector<float> out(static_cast<size_t>(history) * 3);
        for (int64_t h = 0; h < history; ++h)
        {
            for (int64_t c = 0; c < 3; ++c)
            {
                size_t const src = static_cast<size_t>((0 * shape[1] + h) * shape[2] + c);
                out[static_cast<size_t>(h * 3 + c)] = data[src];
            }
        }
        return out;
    }
    throw std::runtime_error("Unsupported ego_history_xyz shape rank");
}

std::vector<float> extractLastHistoryRot(std::vector<float> const& data, std::vector<int64_t> const& shape)
{
    check::check(!shape.empty(), "ego_history_rot shape must not be empty");
    if (shape.size() == 5)
    {
        check::check(shape[0] == 1, "Only batch_size=1 is supported for ego_history_rot");
        check::check(shape[3] == 3 && shape[4] == 3, "ego_history_rot last dims must be 3x3");
        int64_t const trajGroup = shape[1] - 1;
        int64_t const history = shape[2];
        std::vector<float> out(static_cast<size_t>(history) * 9);
        for (int64_t h = 0; h < history; ++h)
        {
            for (int64_t r = 0; r < 3; ++r)
            {
                for (int64_t c = 0; c < 3; ++c)
                {
                    size_t const src = static_cast<size_t>((((0 * shape[1] + trajGroup) * shape[2] + h) * shape[3] + r)
                        * shape[4]
                        + c);
                    out[static_cast<size_t>(h * 9 + r * 3 + c)] = data[src];
                }
            }
        }
        return out;
    }
    if (shape.size() == 4)
    {
        check::check(shape[0] == 1, "Only batch_size=1 is supported for ego_history_rot");
        check::check(shape[2] == 3 && shape[3] == 3, "ego_history_rot last dims must be 3x3");
        int64_t const history = shape[1];
        std::vector<float> out(static_cast<size_t>(history) * 9);
        for (int64_t h = 0; h < history; ++h)
        {
            for (int64_t r = 0; r < 3; ++r)
            {
                for (int64_t c = 0; c < 3; ++c)
                {
                    size_t const src = static_cast<size_t>(((0 * shape[1] + h) * shape[2] + r) * shape[3] + c);
                    out[static_cast<size_t>(h * 9 + r * 3 + c)] = data[src];
                }
            }
        }
        return out;
    }
    throw std::runtime_error("Unsupported ego_history_rot shape rank");
}

inline double wrapAngle(double x)
{
    return std::atan2(std::sin(x), std::cos(x));
}

inline double& mat(std::vector<double>& a, int n, int r, int c)
{
    return a[static_cast<size_t>(r) * n + c];
}

inline double matc(std::vector<double> const& a, int n, int r, int c)
{
    return a[static_cast<size_t>(r) * n + c];
}

void addThirdOrderDtd(std::vector<double>& lhs, int n, double scale)
{
    if (n < 4)
    {
        return;
    }
    int const coeffs[4] = {-1, 3, -3, 1};
    for (int row = 0; row < n - 3; ++row)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat(lhs, n, row + i, row + j) += scale * static_cast<double>(coeffs[i] * coeffs[j]);
            }
        }
    }
}

bool choleskyInplace(std::vector<double>& a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = matc(a, n, i, j);
            for (int k = 0; k < j; ++k)
            {
                sum -= matc(a, n, i, k) * matc(a, n, j, k);
            }
            if (i == j)
            {
                if (sum <= 0.0)
                {
                    return false;
                }
                mat(a, n, i, j) = std::sqrt(sum);
            }
            else
            {
                mat(a, n, i, j) = sum / matc(a, n, j, j);
            }
        }
        for (int j = i + 1; j < n; ++j)
        {
            mat(a, n, i, j) = 0.0;
        }
    }
    return true;
}

std::vector<double> choleskySolve(std::vector<double> const& lower, std::vector<double> const& rhs, int n)
{
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        double sum = rhs[i];
        for (int j = 0; j < i; ++j)
        {
            sum -= matc(lower, n, i, j) * y[j];
        }
        y[i] = sum / matc(lower, n, i, i);
    }

    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j)
        {
            sum -= matc(lower, n, j, i) * x[j];
        }
        x[i] = sum / matc(lower, n, i, i);
    }
    return x;
}

double estimateV0One(std::vector<float> const& histXyz, std::vector<float> const& histRot, int historyLen,
    double dt, double vLambda, double vRidge)
{
    check::check(historyLen >= 2, "History length must be at least 2");
    int const n = historyLen - 1;

    std::vector<double> theta(historyLen, 0.0);
    double prevRaw = 0.0;
    for (int t = 0; t < historyLen; ++t)
    {
        double raw = std::atan2(static_cast<double>(histRot[static_cast<size_t>(t * 9 + 3)]),
            static_cast<double>(histRot[static_cast<size_t>(t * 9 + 0)]));
        if (t == 0)
        {
            theta[t] = raw;
        }
        else
        {
            theta[t] = theta[t - 1] + wrapAngle(raw - prevRaw);
        }
        prevRaw = raw;
    }

    int const vdim = n + 1;
    std::vector<double> lhs(static_cast<size_t>(vdim) * vdim, 0.0);
    std::vector<double> rhs(vdim, 0.0);

    for (int t = 0; t < n; ++t)
    {
        double dx = static_cast<double>(histXyz[static_cast<size_t>((t + 1) * 3 + 0)] - histXyz[static_cast<size_t>(t * 3 + 0)]);
        double dy = static_cast<double>(histXyz[static_cast<size_t>((t + 1) * 3 + 1)] - histXyz[static_cast<size_t>(t * 3 + 1)]);
        double gx = (2.0 / dt) * dx;
        double gy = (2.0 / dt) * dy;

        double c0 = std::cos(theta[t]);
        double c1 = std::cos(theta[t + 1]);
        double s0 = std::sin(theta[t]);
        double s1 = std::sin(theta[t + 1]);

        mat(lhs, vdim, t, t) += c0 * c0 + s0 * s0;
        mat(lhs, vdim, t, t + 1) += c0 * c1 + s0 * s1;
        mat(lhs, vdim, t + 1, t) += c1 * c0 + s1 * s0;
        mat(lhs, vdim, t + 1, t + 1) += c1 * c1 + s1 * s1;

        rhs[t] += c0 * gx + s0 * gy;
        rhs[t + 1] += c1 * gx + s1 * gy;
    }

    double const smoothScale = vLambda / std::pow(dt, 6.0);
    addThirdOrderDtd(lhs, vdim, smoothScale);
    for (int i = 0; i < vdim; ++i)
    {
        mat(lhs, vdim, i, i) += vRidge;
    }

    check::check(choleskyInplace(lhs, vdim), "Failed Cholesky factorization in action_to_traj exact decode");
    auto velocity = choleskySolve(lhs, rhs, vdim);
    return velocity.back();
}

void actionToTrajExact(std::vector<float> const& action, int horizon, std::vector<float> const& histXyz,
    int historyLen, std::vector<float> const& histRot, LLMGenerationRequest::ActionSpaceConstants const& constants,
    std::vector<float>& predXyz, std::vector<float>& predRot)
{
    predXyz.assign(static_cast<size_t>(horizon) * 3, 0.0F);
    predRot.assign(static_cast<size_t>(horizon) * 9, 0.0F);

    double const v0 = estimateV0One(histXyz, histRot, historyLen, constants.dtValue, constants.vLambda, constants.vRidge);
    std::vector<double> velocity(static_cast<size_t>(horizon) + 1, 0.0);
    std::vector<double> theta(static_cast<size_t>(horizon) + 1, 0.0);
    velocity[0] = v0;
    theta[0] = 0.0;

    for (int t = 0; t < horizon; ++t)
    {
        double accel = static_cast<double>(action[static_cast<size_t>(t * 2 + 0)]) * constants.accelStd + constants.accelMean;
        double kappa = static_cast<double>(action[static_cast<size_t>(t * 2 + 1)]) * constants.curvatureStd
            + constants.curvatureMean;
        velocity[static_cast<size_t>(t + 1)] = velocity[static_cast<size_t>(t)] + accel * constants.dtValue;
        theta[static_cast<size_t>(t + 1)] = theta[static_cast<size_t>(t)] + kappa * velocity[static_cast<size_t>(t)] * constants.dtValue
            + kappa * accel * 0.5 * constants.dtValue * constants.dtValue;
    }

    double xCum = 0.0;
    double yCum = 0.0;
    float const z0 = histXyz[static_cast<size_t>((historyLen - 1) * 3 + 2)];
    double const halfDt = 0.5 * constants.dtValue;
    for (int t = 0; t < horizon; ++t)
    {
        xCum += (velocity[static_cast<size_t>(t)] * std::cos(theta[static_cast<size_t>(t)])
                    + velocity[static_cast<size_t>(t + 1)] * std::cos(theta[static_cast<size_t>(t + 1)]))
            * halfDt;
        yCum += (velocity[static_cast<size_t>(t)] * std::sin(theta[static_cast<size_t>(t)])
                    + velocity[static_cast<size_t>(t + 1)] * std::sin(theta[static_cast<size_t>(t + 1)]))
            * halfDt;

        predXyz[static_cast<size_t>(t * 3 + 0)] = static_cast<float>(xCum);
        predXyz[static_cast<size_t>(t * 3 + 1)] = static_cast<float>(yCum);
        predXyz[static_cast<size_t>(t * 3 + 2)] = z0;

        float c = static_cast<float>(std::cos(theta[static_cast<size_t>(t + 1)]));
        float s = static_cast<float>(std::sin(theta[static_cast<size_t>(t + 1)]));
        predRot[static_cast<size_t>(t * 9 + 0)] = c;
        predRot[static_cast<size_t>(t * 9 + 1)] = -s;
        predRot[static_cast<size_t>(t * 9 + 2)] = 0.0F;
        predRot[static_cast<size_t>(t * 9 + 3)] = s;
        predRot[static_cast<size_t>(t * 9 + 4)] = c;
        predRot[static_cast<size_t>(t * 9 + 5)] = 0.0F;
        predRot[static_cast<size_t>(t * 9 + 6)] = 0.0F;
        predRot[static_cast<size_t>(t * 9 + 7)] = 0.0F;
        predRot[static_cast<size_t>(t * 9 + 8)] = 1.0F;
    }
}

} // namespace

struct AlpamayoFmRuntime::Impl
{
    struct DeviceBranch
    {
        CudaBuffer kvCache;
        CudaBuffer attentionMask;
        CudaBuffer positionIds;
    };

    explicit Impl(std::string path)
        : enginePath(std::move(path))
    {
        std::ifstream in(enginePath, std::ios::binary);
        check::check(in.good(), "Failed to open FM engine: " + enginePath);
        in.seekg(0, std::ios::end);
        size_t const size = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        in.read(buffer.data(), static_cast<std::streamsize>(size));
        check::check(in.good(), "Failed to read FM engine bytes: " + enginePath);

        runtime.reset(nvinfer1::createInferRuntime(logger));
        check::check(runtime != nullptr, "Failed to create TensorRT runtime for FM engine");
        engine.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        check::check(engine != nullptr, "Failed to deserialize FM engine: " + enginePath);
        context.reset(engine->createExecutionContext());
        check::check(context != nullptr, "Failed to create FM execution context");

        xDims = engine->getTensorShape("x");
        nextXDims = engine->getTensorShape("next_x");
        vDims = engine->getTensorShape("v");
        futureDims = engine->getTensorShape("future_token_embeds");
        kvDims = engine->getTensorShape("kv_cache");
        maskDims = engine->getTensorShape("attention_mask");
        posDims = engine->getTensorShape("position_ids");

        horizon = xDims.d[1];
        actionDim = xDims.d[2];
        maxSeqLen = kvDims.d[4];
        nDiffusionTokens = posDims.d[2];
        check::check(horizon > 0 && actionDim > 0 && maxSeqLen > 0 && nDiffusionTokens > 0,
            "FM engine shapes must be static and positive");

        xBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(xDims)) * dataTypeSize(engine->getTensorDataType("x")));
        tBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(engine->getTensorShape("t"))) * dataTypeSize(engine->getTensorDataType("t")));
        dtBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(engine->getTensorShape("dt"))) * dataTypeSize(engine->getTensorDataType("dt")));
        nextXBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(nextXDims)) * dataTypeSize(engine->getTensorDataType("next_x")));
        vBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(vDims)) * dataTypeSize(engine->getTensorDataType("v")));
        futureBuffer = CudaBuffer(static_cast<size_t>(dimsVolume(futureDims)) * dataTypeSize(engine->getTensorDataType("future_token_embeds")));
    }

    DeviceBranch makeDeviceBranch(AlpamayoFmBranchSnapshot const& snapshot, AlpamayoFmRunResult::Timing& timing)
    {
        auto const engineKvType = engine->getTensorDataType("kv_cache");
        check::check(snapshot.kvCacheDataType == nvinfer1::DataType::kHALF,
            "Current native Alpamayo FM runtime supports FLOAT16 VLM KV snapshots only. Got "
                + dataTypeName(snapshot.kvCacheDataType));
        check::check(engineKvType == nvinfer1::DataType::kHALF || engineKvType == nvinfer1::DataType::kBF16,
            "Current native Alpamayo FM runtime only supports FLOAT16/BF16 FM kv_cache inputs. Got "
                + dataTypeName(engineKvType));
        check::check(snapshot.kvCacheShape == Coords(kvDims), "KV snapshot shape does not match FM engine kv_cache input shape");
        check::check(snapshot.activeLen > 0 && snapshot.activeLen <= maxSeqLen, "Invalid FM active KV length");

        auto const kvConvertStart = SteadyClock::now();
        size_t kvBytesSize = snapshot.kvCacheBytes.size();
        uint8_t const* kvBytesPtr = snapshot.kvCacheBytes.data();
        std::vector<uint8_t> convertedKvBytes;
        if (engineKvType == nvinfer1::DataType::kBF16)
        {
            convertedKvBytes = convertHalfBytesToBf16Bytes(snapshot.kvCacheBytes);
            kvBytesSize = convertedKvBytes.size();
            kvBytesPtr = convertedKvBytes.data();
        }
        timing.kvConvertMs += ::trt_edgellm::rt::elapsedMs(kvConvertStart, SteadyClock::now());
        DeviceBranch branch;
        auto const kvAllocStart = SteadyClock::now();
        branch.kvCache = CudaBuffer(kvBytesSize);
        timing.kvAllocMs += ::trt_edgellm::rt::elapsedMs(kvAllocStart, SteadyClock::now());
        auto const kvCopyStart = SteadyClock::now();
        memcpyToDevice(branch.kvCache.ptr, kvBytesPtr, kvBytesSize);
        timing.kvCopyMs += ::trt_edgellm::rt::elapsedMs(kvCopyStart, SteadyClock::now());

        auto const maskBuildStart = SteadyClock::now();
        std::vector<float> attentionMask(static_cast<size_t>(dimsVolume(maskDims)), 0.0F);
        float const maskValue = std::numeric_limits<float>::lowest();
        int32_t const totalKv = maskDims.d[3];
        check::check(totalKv == maxSeqLen + nDiffusionTokens, "Unexpected FM attention mask width");
        for (int32_t q = 0; q < nDiffusionTokens; ++q)
        {
            size_t const rowBase = static_cast<size_t>(q) * totalKv;
            for (int32_t k = snapshot.activeLen; k < maxSeqLen; ++k)
            {
                attentionMask[rowBase + k] = maskValue;
            }
        }
        timing.maskBuildMs += ::trt_edgellm::rt::elapsedMs(maskBuildStart, SteadyClock::now());
        auto const maskAllocStart = SteadyClock::now();
        branch.attentionMask = CudaBuffer(attentionMask.size() * sizeof(float));
        timing.maskAllocMs += ::trt_edgellm::rt::elapsedMs(maskAllocStart, SteadyClock::now());
        auto const maskCopyStart = SteadyClock::now();
        memcpyToDevice(branch.attentionMask.ptr, attentionMask.data(), attentionMask.size() * sizeof(float));
        timing.maskCopyMs += ::trt_edgellm::rt::elapsedMs(maskCopyStart, SteadyClock::now());

        auto const posBuildStart = SteadyClock::now();
        std::vector<int64_t> positionIds(static_cast<size_t>(dimsVolume(posDims)), 0);
        for (int32_t c = 0; c < posDims.d[0]; ++c)
        {
            for (int32_t i = 0; i < posDims.d[2]; ++i)
            {
                size_t const idx = static_cast<size_t>(c) * posDims.d[1] * posDims.d[2] + i;
                positionIds[idx] = snapshot.ropeDelta + snapshot.activeLen + i;
            }
        }
        timing.positionBuildMs += ::trt_edgellm::rt::elapsedMs(posBuildStart, SteadyClock::now());
        auto const posAllocStart = SteadyClock::now();
        branch.positionIds = CudaBuffer(positionIds.size() * sizeof(int64_t));
        timing.positionAllocMs += ::trt_edgellm::rt::elapsedMs(posAllocStart, SteadyClock::now());
        auto const posCopyStart = SteadyClock::now();
        memcpyToDevice(branch.positionIds.ptr, positionIds.data(), positionIds.size() * sizeof(int64_t));
        timing.positionCopyMs += ::trt_edgellm::rt::elapsedMs(posCopyStart, SteadyClock::now());
        return branch;
    }

    bool runOneStep(DeviceBranch const& branch, std::vector<float> const& x, float t, float dt, std::vector<float>& nextX,
        std::vector<float>& v, float* elapsedMsOut = nullptr)
    {
        auto const stepStart = SteadyClock::now();
        check::check(static_cast<int32_t>(x.size()) == horizon * actionDim, "Input x size mismatch for FM one-step");
        nextX.resize(static_cast<size_t>(horizon) * actionDim);
        v.resize(static_cast<size_t>(horizon) * actionDim);

        memcpyToDevice(xBuffer.ptr, x.data(), x.size() * sizeof(float));
        memcpyToDevice(tBuffer.ptr, &t, sizeof(float));
        memcpyToDevice(dtBuffer.ptr, &dt, sizeof(float));

        context->setInputShape("x", xDims);
        context->setInputShape("t", engine->getTensorShape("t"));
        context->setInputShape("dt", engine->getTensorShape("dt"));
        context->setInputShape("kv_cache", kvDims);
        context->setInputShape("attention_mask", maskDims);
        context->setInputShape("position_ids", posDims);

        check::check(context->setTensorAddress("x", xBuffer.ptr), "Failed to bind FM input x");
        check::check(context->setTensorAddress("t", tBuffer.ptr), "Failed to bind FM input t");
        check::check(context->setTensorAddress("dt", dtBuffer.ptr), "Failed to bind FM input dt");
        check::check(context->setTensorAddress("kv_cache", branch.kvCache.ptr), "Failed to bind FM input kv_cache");
        check::check(context->setTensorAddress("attention_mask", branch.attentionMask.ptr),
            "Failed to bind FM input attention_mask");
        check::check(context->setTensorAddress("position_ids", branch.positionIds.ptr),
            "Failed to bind FM input position_ids");
        check::check(context->setTensorAddress("next_x", nextXBuffer.ptr), "Failed to bind FM output next_x");
        check::check(context->setTensorAddress("v", vBuffer.ptr), "Failed to bind FM output v");
        check::check(context->setTensorAddress("future_token_embeds", futureBuffer.ptr),
            "Failed to bind FM output future_token_embeds");

        check::check(context->enqueueV3(0), "FM TensorRT enqueueV3 failed");
        CUDA_CHECK(cudaDeviceSynchronize());

        memcpyToHost(nextX.data(), nextXBuffer.ptr, nextX.size() * sizeof(float));
        memcpyToHost(v.data(), vBuffer.ptr, v.size() * sizeof(float));
        if (elapsedMsOut != nullptr)
        {
            *elapsedMsOut = ::trt_edgellm::rt::elapsedMs(stepStart, SteadyClock::now());
        }
        return true;
    }

    bool runSingleBranch(AlpamayoFmBranchSnapshot const& snapshot, AlpamayoFmRunConfig const& config,
        std::vector<float>& x0, std::vector<float>& xFinal, AlpamayoFmRunResult::Timing& timing)
    {
        auto const totalStart = SteadyClock::now();
        auto const branchStart = SteadyClock::now();
        DeviceBranch branch = makeDeviceBranch(snapshot, timing);
        timing.branchPrepareMs = ::trt_edgellm::rt::elapsedMs(branchStart, SteadyClock::now());

        auto const x0Start = SteadyClock::now();
        x0 = makeNormalX0(horizon, actionDim, config.seed);
        timing.x0InitMs = ::trt_edgellm::rt::elapsedMs(x0Start, SteadyClock::now());
        xFinal = x0;
        std::vector<float> nextX;
        std::vector<float> v;
        float const dt = 1.0F / static_cast<float>(config.numSteps);
        auto const diffusionStart = SteadyClock::now();
        for (int32_t step = 0; step < config.numSteps; ++step)
        {
            float const t = static_cast<float>(step) * dt;
            float stepMs = 0.0F;
            runOneStep(branch, xFinal, t, dt, nextX, v, &stepMs);
            timing.engineStepTotalMs += stepMs;
            xFinal = nextX;
        }
        float const diffusionMs = ::trt_edgellm::rt::elapsedMs(diffusionStart, SteadyClock::now());
        (void) diffusionMs;
        timing.numSteps = config.numSteps;
        timing.numBranches = 1;
        timing.engineStepAvgMs = config.numSteps > 0 ? timing.engineStepTotalMs / static_cast<float>(config.numSteps) : 0.0F;
        timing.totalMs = ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now());
        return true;
    }

    bool runDualBranch(AlpamayoFmBranchSnapshot const& guidedSnapshot, AlpamayoFmBranchSnapshot const& unguidedSnapshot,
        AlpamayoFmRunConfig const& config, std::vector<float>& x0, std::vector<float>& xFinal,
        AlpamayoFmRunResult::Timing& timing)
    {
        auto const totalStart = SteadyClock::now();
        auto const branchStart = SteadyClock::now();
        DeviceBranch guided = makeDeviceBranch(guidedSnapshot, timing);
        DeviceBranch unguided = makeDeviceBranch(unguidedSnapshot, timing);
        timing.branchPrepareMs = ::trt_edgellm::rt::elapsedMs(branchStart, SteadyClock::now());

        auto const x0Start = SteadyClock::now();
        x0 = makeNormalX0(horizon, actionDim, config.seed);
        timing.x0InitMs = ::trt_edgellm::rt::elapsedMs(x0Start, SteadyClock::now());
        xFinal = x0;
        std::vector<float> guidedNext;
        std::vector<float> unguidedNext;
        std::vector<float> guidedV;
        std::vector<float> unguidedV;
        float const dt = 1.0F / static_cast<float>(config.numSteps);
        auto const diffusionStart = SteadyClock::now();
        for (int32_t step = 0; step < config.numSteps; ++step)
        {
            float const t = static_cast<float>(step) * dt;
            float guidedStepMs = 0.0F;
            float unguidedStepMs = 0.0F;
            runOneStep(guided, xFinal, t, dt, guidedNext, guidedV, &guidedStepMs);
            runOneStep(unguided, xFinal, t, dt, unguidedNext, unguidedV, &unguidedStepMs);
            timing.engineStepTotalMs += guidedStepMs + unguidedStepMs;
            for (size_t i = 0; i < xFinal.size(); ++i)
            {
                float const vCfg = (1.0F - config.guidanceWeight) * unguidedV[i] + config.guidanceWeight * guidedV[i];
                xFinal[i] = xFinal[i] + dt * vCfg;
            }
        }
        float const diffusionMs = ::trt_edgellm::rt::elapsedMs(diffusionStart, SteadyClock::now());
        (void) diffusionMs;
        timing.numSteps = config.numSteps;
        timing.numBranches = 2;
        timing.engineStepAvgMs = config.numSteps > 0 ? timing.engineStepTotalMs / static_cast<float>(config.numSteps) : 0.0F;
        timing.totalMs = ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now());
        return true;
    }

    std::string enginePath;
    Logger logger;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;

    nvinfer1::Dims xDims{};
    nvinfer1::Dims nextXDims{};
    nvinfer1::Dims vDims{};
    nvinfer1::Dims futureDims{};
    nvinfer1::Dims kvDims{};
    nvinfer1::Dims maskDims{};
    nvinfer1::Dims posDims{};
    int32_t horizon{0};
    int32_t actionDim{0};
    int32_t maxSeqLen{0};
    int32_t nDiffusionTokens{0};

    CudaBuffer xBuffer;
    CudaBuffer tBuffer;
    CudaBuffer dtBuffer;
    CudaBuffer nextXBuffer;
    CudaBuffer vBuffer;
    CudaBuffer futureBuffer;
};

AlpamayoFmRuntime::AlpamayoFmRuntime(std::string enginePath)
    : mImpl(std::make_unique<Impl>(std::move(enginePath)))
{
}

AlpamayoFmRuntime::~AlpamayoFmRuntime() = default;

bool AlpamayoFmRuntime::runNoNav(AlpamayoFmBranchSnapshot const& branch,
    LLMGenerationRequest::ActionSpaceConstants const& constants, std::vector<float> const& egoHistoryXyz,
    std::vector<int64_t> const& egoHistoryXyzShape, std::vector<float> const& egoHistoryRot,
    std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config, AlpamayoFmRunResult& result)
{
    auto const totalStart = SteadyClock::now();
    std::vector<float> x0;
    std::vector<float> xFinal;
    check::check(mImpl->runSingleBranch(branch, config, x0, xFinal, result.timing), "Failed to run native no-nav FM TRT");

    std::vector<float> histXyz = extractLastHistoryXYZ(egoHistoryXyz, egoHistoryXyzShape);
    std::vector<float> histRot = extractLastHistoryRot(egoHistoryRot, egoHistoryRotShape);
    int32_t const historyLen = static_cast<int32_t>(histXyz.size() / 3);
    std::vector<float> predXyz;
    std::vector<float> predRot;
    auto const decodeStart = SteadyClock::now();
    actionToTrajExact(xFinal, mImpl->horizon, histXyz, historyLen, histRot, constants, predXyz, predRot);
    result.timing.decodePostprocessMs = ::trt_edgellm::rt::elapsedMs(decodeStart, SteadyClock::now());
    result.timing.totalMs = ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now());

    result.x0 = std::move(x0);
    result.xFinal = std::move(xFinal);
    result.predXyz = std::move(predXyz);
    result.predRot = std::move(predRot);
    result.horizon = mImpl->horizon;
    result.actionDim = mImpl->actionDim;
    return true;
}

bool AlpamayoFmRuntime::runNavCfg(AlpamayoFmBranchSnapshot const& guided, AlpamayoFmBranchSnapshot const& unguided,
    LLMGenerationRequest::ActionSpaceConstants const& constants, std::vector<float> const& egoHistoryXyz,
    std::vector<int64_t> const& egoHistoryXyzShape, std::vector<float> const& egoHistoryRot,
    std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config, AlpamayoFmRunResult& result)
{
    auto const totalStart = SteadyClock::now();
    std::vector<float> x0;
    std::vector<float> xFinal;
    check::check(
        mImpl->runDualBranch(guided, unguided, config, x0, xFinal, result.timing), "Failed to run native nav CFG FM TRT");

    std::vector<float> histXyz = extractLastHistoryXYZ(egoHistoryXyz, egoHistoryXyzShape);
    std::vector<float> histRot = extractLastHistoryRot(egoHistoryRot, egoHistoryRotShape);
    int32_t const historyLen = static_cast<int32_t>(histXyz.size() / 3);
    std::vector<float> predXyz;
    std::vector<float> predRot;
    auto const decodeStart = SteadyClock::now();
    actionToTrajExact(xFinal, mImpl->horizon, histXyz, historyLen, histRot, constants, predXyz, predRot);
    result.timing.decodePostprocessMs = ::trt_edgellm::rt::elapsedMs(decodeStart, SteadyClock::now());
    result.timing.totalMs = ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now());

    result.x0 = std::move(x0);
    result.xFinal = std::move(xFinal);
    result.predXyz = std::move(predXyz);
    result.predRot = std::move(predRot);
    result.horizon = mImpl->horizon;
    result.actionDim = mImpl->actionDim;
    return true;
}

int32_t AlpamayoFmRuntime::maxSeqLen() const noexcept
{
    return mImpl->maxSeqLen;
}

int32_t AlpamayoFmRuntime::horizon() const noexcept
{
    return mImpl->horizon;
}

std::string const& AlpamayoFmRuntime::enginePath() const noexcept
{
    return mImpl->enginePath;
}

} // namespace rt
} // namespace trt_edgellm
