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
#include <unordered_map>
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

struct PositionIdsCacheKey
{
    int32_t activeLen{0};
    int64_t ropeDelta{0};

    bool operator==(PositionIdsCacheKey const& other) const noexcept
    {
        return activeLen == other.activeLen && ropeDelta == other.ropeDelta;
    }
};

struct PositionIdsCacheKeyHash
{
    size_t operator()(PositionIdsCacheKey const& key) const noexcept
    {
        size_t const activeHash = std::hash<int32_t>{}(key.activeLen);
        size_t const ropeHash = std::hash<int64_t>{}(key.ropeDelta);
        return activeHash ^ (ropeHash + 0x9e3779b9 + (activeHash << 6) + (activeHash >> 2));
    }
};

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

    bool ensureSize(size_t size)
    {
        if (bytes >= size)
        {
            return false;
        }
        reset();
        bytes = size;
        if (bytes > 0)
        {
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
        }
        return true;
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

bool hasDeviceKvSnapshot(AlpamayoFmBranchSnapshot const& snapshot)
{
    return !snapshot.kvCacheTensor.isEmpty() && snapshot.kvCacheTensor.getDeviceType() == DeviceType::kGPU
        && snapshot.kvCacheTensor.rawPointer() != nullptr;
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
        __half halfValue{raw};
        float f = __half2float(halfValue);
        __nv_bfloat16 bf = __float2bfloat16(f);
        dstWords[i] = static_cast<__nv_bfloat16_raw>(bf).x;
    }
    std::vector<uint8_t> bytes(dstWords.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), dstWords.data(), bytes.size());
    return bytes;
}

bool kvShapesMatchExceptSeqLen(Coords const& snapshotShape, nvinfer1::Dims const& engineShape)
{
    if (snapshotShape.getNumDims() != engineShape.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < engineShape.nbDims; ++i)
    {
        if (i == 4)
        {
            continue;
        }
        if (snapshotShape[i] != engineShape.d[i])
        {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> truncateKvCachePrefix(std::vector<uint8_t> const& src, Coords const& snapshotShape, int32_t activeLen,
    int32_t targetSeqLen, size_t elementBytes)
{
    check::check(snapshotShape.getNumDims() == 6, "FM KV snapshot rank must be 6");
    check::check(snapshotShape[4] >= targetSeqLen, "Target FM kv_cache length exceeds snapshot length");
    check::check(activeLen >= 0 && activeLen <= targetSeqLen, "FM active KV length exceeds target engine length");
    int64_t const outerCount = snapshotShape[0] * snapshotShape[1] * snapshotShape[2] * snapshotShape[3];
    int64_t const snapshotSeqLen = snapshotShape[4];
    int64_t const headDim = snapshotShape[5];
    size_t const srcBlockBytes = static_cast<size_t>(snapshotSeqLen) * headDim * elementBytes;
    size_t const dstBlockBytes = static_cast<size_t>(targetSeqLen) * headDim * elementBytes;
    size_t const copyBytes = static_cast<size_t>(activeLen) * headDim * elementBytes;
    size_t const expectedSrcBytes = static_cast<size_t>(outerCount) * srcBlockBytes;
    check::check(src.size() == expectedSrcBytes, "Unexpected FM snapshot byte size for kv_cache truncation");

    std::vector<uint8_t> dst(static_cast<size_t>(outerCount) * dstBlockBytes, 0);
    for (int64_t idx = 0; idx < outerCount; ++idx)
    {
        auto const* srcPtr = src.data() + static_cast<size_t>(idx) * srcBlockBytes;
        auto* dstPtr = dst.data() + static_cast<size_t>(idx) * dstBlockBytes;
        std::memcpy(dstPtr, srcPtr, copyBytes);
    }
    return dst;
}

void copyKvCachePrefixDeviceToDevice(
    void* dst, Coords const& dstShape, void const* src, Coords const& srcShape, int32_t activeLen, size_t elementBytes,
    cudaStream_t stream)
{
    check::check(srcShape.getNumDims() == 6 && dstShape.getNumDims() == 6, "FM KV snapshot rank must be 6");
    check::check(activeLen >= 0 && activeLen <= dstShape[4], "FM active KV length exceeds target device KV length");
    check::check(srcShape[4] >= dstShape[4], "Source FM KV device snapshot is shorter than target");
    int64_t const outerCount = srcShape[0] * srcShape[1] * srcShape[2] * srcShape[3];
    int64_t const headDim = srcShape[5];
    size_t const srcPitch = static_cast<size_t>(srcShape[4]) * headDim * elementBytes;
    size_t const dstPitch = static_cast<size_t>(dstShape[4]) * headDim * elementBytes;
    size_t const totalDstBytes = static_cast<size_t>(outerCount) * dstPitch;
    size_t const copyWidth = static_cast<size_t>(activeLen) * headDim * elementBytes;
    CUDA_CHECK(cudaMemsetAsync(dst, 0, totalDstBytes, stream));
    if (copyWidth == 0)
    {
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst, dstPitch, src, srcPitch, copyWidth, static_cast<size_t>(outerCount), cudaMemcpyDeviceToDevice, stream));
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

double computePlanarHistoryTravel(std::vector<float> const& histXyz, int historyLen)
{
    if (historyLen < 2)
    {
        return 0.0;
    }

    double travel = 0.0;
    for (int t = 1; t < historyLen; ++t)
    {
        double const dx = static_cast<double>(histXyz[static_cast<size_t>(t * 3 + 0)]
            - histXyz[static_cast<size_t>((t - 1) * 3 + 0)]);
        double const dy = static_cast<double>(histXyz[static_cast<size_t>(t * 3 + 1)]
            - histXyz[static_cast<size_t>((t - 1) * 3 + 1)]);
        travel += std::hypot(dx, dy);
    }
    return travel;
}

double sanitizeInitialForwardSpeed(double v0, std::vector<float> const& histXyz, int historyLen)
{
    double const clampedV0 = std::max(0.0, v0);
    double const historyTravel = computePlanarHistoryTravel(histXyz, historyLen);
    if (historyTravel <= 0.20 || clampedV0 <= 0.25)
    {
        return 0.0;
    }
    return clampedV0;
}

void actionToTrajExact(std::vector<float> const& action, int horizon, std::vector<float> const& histXyz,
    int historyLen, std::vector<float> const& histRot, LLMGenerationRequest::ActionSpaceConstants const& constants,
    std::vector<float>& predXyz, std::vector<float>& predRot)
{
    predXyz.assign(static_cast<size_t>(horizon) * 3, 0.0F);
    predRot.assign(static_cast<size_t>(horizon) * 9, 0.0F);

    double const v0
        = sanitizeInitialForwardSpeed(estimateV0One(histXyz, histRot, historyLen, constants.dtValue, constants.vLambda,
                                         constants.vRidge),
            histXyz, historyLen);
    std::vector<double> velocity(static_cast<size_t>(horizon) + 1, 0.0);
    std::vector<double> theta(static_cast<size_t>(horizon) + 1, 0.0);
    velocity[0] = v0;
    theta[0] = 0.0;

    for (int t = 0; t < horizon; ++t)
    {
        double accel = static_cast<double>(action[static_cast<size_t>(t * 2 + 0)]) * constants.accelStd + constants.accelMean;
        double kappa = static_cast<double>(action[static_cast<size_t>(t * 2 + 1)]) * constants.curvatureStd
            + constants.curvatureMean;
        double const vPrev = velocity[static_cast<size_t>(t)];
        double const vNext = std::max(0.0, vPrev + accel * constants.dtValue);
        velocity[static_cast<size_t>(t + 1)] = vNext;
        double const avgForwardSpeed = 0.5 * (vPrev + vNext);
        theta[static_cast<size_t>(t + 1)]
            = theta[static_cast<size_t>(t)] + kappa * avgForwardSpeed * constants.dtValue;
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
        void* kvCachePtr{nullptr};
        void* attentionMaskPtr{nullptr};
        void* positionIdsPtr{nullptr};
    };

    struct BranchWorkspace
    {
        CudaBuffer kvCache;
        CudaBuffer positionIds;
        std::vector<float> attentionMaskHost;
        std::vector<int64_t> positionIdsHost;
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
        kvDataType = engine->getTensorDataType("kv_cache");
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

    DeviceBranch makeDeviceBranch(AlpamayoFmBranchSnapshot const& snapshot, BranchWorkspace& workspace,
        AlpamayoFmRunResult::Timing& timing, cudaStream_t stream)
    {
        auto const engineKvType = kvDataType;
        auto const engineKvShape = Coords(kvDims);
        check::check(snapshot.kvCacheDataType == nvinfer1::DataType::kHALF,
            "Current native Alpamayo FM runtime supports FLOAT16 VLM KV snapshots only. Got "
                + dataTypeName(snapshot.kvCacheDataType));
        check::check(engineKvType == nvinfer1::DataType::kHALF || engineKvType == nvinfer1::DataType::kBF16,
            "Current native Alpamayo FM runtime only supports FLOAT16/BF16 FM kv_cache inputs. Got "
                + dataTypeName(engineKvType));
        check::check(snapshot.activeLen > 0 && snapshot.activeLen <= maxSeqLen, "Invalid FM active KV length");

        DeviceBranch branch;
        bool const canUseDeviceSnapshot = hasDeviceKvSnapshot(snapshot) && snapshot.kvCacheDataType == engineKvType;
        if (canUseDeviceSnapshot && snapshot.kvCacheShape == engineKvShape)
        {
            branch.kvCachePtr = const_cast<void*>(snapshot.kvCacheTensor.rawPointer());
        }
        else
        {
            auto const kvConvertStart = SteadyClock::now();
            std::vector<uint8_t> kvHostBytes;
            bool const useDeviceTruncation
                = canUseDeviceSnapshot && kvShapesMatchExceptSeqLen(snapshot.kvCacheShape, kvDims) && snapshot.kvCacheShape[4] >= maxSeqLen;
            if (!useDeviceTruncation)
            {
                if (snapshot.kvCacheShape == engineKvShape)
                {
                    kvHostBytes = snapshot.kvCacheBytes;
                }
                else if (kvShapesMatchExceptSeqLen(snapshot.kvCacheShape, kvDims) && snapshot.kvCacheShape[4] >= maxSeqLen)
                {
                    kvHostBytes = truncateKvCachePrefix(
                        snapshot.kvCacheBytes, snapshot.kvCacheShape, snapshot.activeLen, maxSeqLen, sizeof(uint16_t));
                }
                else
                {
                    throw std::runtime_error(
                        "KV snapshot shape does not match FM engine kv_cache input shape: snapshot="
                        + snapshot.kvCacheShape.formatString() + " engine=" + engineKvShape.formatString());
                }
                if (engineKvType == nvinfer1::DataType::kBF16)
                {
                    kvHostBytes = convertHalfBytesToBf16Bytes(kvHostBytes);
                }
            }
            timing.kvConvertMs += ::trt_edgellm::rt::elapsedMs(kvConvertStart, SteadyClock::now());

            size_t const kvBytesSize = static_cast<size_t>(engineKvShape.volume()) * dataTypeSize(engineKvType);
            auto const kvAllocStart = SteadyClock::now();
            bool const kvAllocated = workspace.kvCache.ensureSize(kvBytesSize);
            if (kvAllocated)
            {
                timing.kvAllocMs += ::trt_edgellm::rt::elapsedMs(kvAllocStart, SteadyClock::now());
            }
            auto const kvCopyStart = SteadyClock::now();
            if (useDeviceTruncation)
            {
                copyKvCachePrefixDeviceToDevice(workspace.kvCache.ptr, engineKvShape, snapshot.kvCacheTensor.rawPointer(),
                    snapshot.kvCacheShape, snapshot.activeLen, sizeof(uint16_t), stream);
            }
            else
            {
                memcpyToDevice(workspace.kvCache.ptr, kvHostBytes.data(), kvHostBytes.size());
            }
            timing.kvCopyMs += ::trt_edgellm::rt::elapsedMs(kvCopyStart, SteadyClock::now());
            branch.kvCachePtr = workspace.kvCache.ptr;
        }

        auto maskCacheIt = attentionMaskCache.find(snapshot.activeLen);
        if (maskCacheIt == attentionMaskCache.end())
        {
            auto const maskBuildStart = SteadyClock::now();
            size_t const attentionMaskElems = static_cast<size_t>(dimsVolume(maskDims));
            workspace.attentionMaskHost.assign(attentionMaskElems, 0.0F);
            float const maskValue = std::numeric_limits<float>::lowest();
            int32_t const totalKv = maskDims.d[3];
            check::check(totalKv == maxSeqLen + nDiffusionTokens, "Unexpected FM attention mask width");
            for (int32_t q = 0; q < nDiffusionTokens; ++q)
            {
                size_t const rowBase = static_cast<size_t>(q) * totalKv;
                for (int32_t k = snapshot.activeLen; k < maxSeqLen; ++k)
                {
                    workspace.attentionMaskHost[rowBase + k] = maskValue;
                }
            }
            timing.maskBuildMs += ::trt_edgellm::rt::elapsedMs(maskBuildStart, SteadyClock::now());

            auto const [cacheIt, inserted] = attentionMaskCache.try_emplace(snapshot.activeLen);
            (void) inserted;
            auto const maskAllocStart = SteadyClock::now();
            bool const maskAllocated = cacheIt->second.ensureSize(attentionMaskElems * sizeof(float));
            if (maskAllocated)
            {
                timing.maskAllocMs += ::trt_edgellm::rt::elapsedMs(maskAllocStart, SteadyClock::now());
            }

            auto const maskCopyStart = SteadyClock::now();
            memcpyToDevice(cacheIt->second.ptr, workspace.attentionMaskHost.data(), attentionMaskElems * sizeof(float));
            timing.maskCopyMs += ::trt_edgellm::rt::elapsedMs(maskCopyStart, SteadyClock::now());
            maskCacheIt = cacheIt;
        }
        branch.attentionMaskPtr = maskCacheIt->second.ptr;

        PositionIdsCacheKey const positionKey{snapshot.activeLen, snapshot.ropeDelta};
        auto positionCacheIt = positionIdsCache.find(positionKey);
        if (positionCacheIt == positionIdsCache.end())
        {
            auto const posBuildStart = SteadyClock::now();
            size_t const positionIdsElems = static_cast<size_t>(dimsVolume(posDims));
            workspace.positionIdsHost.assign(positionIdsElems, 0);
            for (int32_t c = 0; c < posDims.d[0]; ++c)
            {
                for (int32_t i = 0; i < posDims.d[2]; ++i)
                {
                    size_t const idx = static_cast<size_t>(c) * posDims.d[1] * posDims.d[2] + i;
                    workspace.positionIdsHost[idx] = snapshot.ropeDelta + snapshot.activeLen + i;
                }
            }
            timing.positionBuildMs += ::trt_edgellm::rt::elapsedMs(posBuildStart, SteadyClock::now());

            auto const [cacheIt, inserted] = positionIdsCache.try_emplace(positionKey);
            (void) inserted;
            auto const posAllocStart = SteadyClock::now();
            bool const positionAllocated = cacheIt->second.ensureSize(positionIdsElems * sizeof(int64_t));
            if (positionAllocated)
            {
                timing.positionAllocMs += ::trt_edgellm::rt::elapsedMs(posAllocStart, SteadyClock::now());
            }

            auto const posCopyStart = SteadyClock::now();
            memcpyToDevice(cacheIt->second.ptr, workspace.positionIdsHost.data(), positionIdsElems * sizeof(int64_t));
            timing.positionCopyMs += ::trt_edgellm::rt::elapsedMs(posCopyStart, SteadyClock::now());
            positionCacheIt = cacheIt;
        }
        branch.positionIdsPtr = positionCacheIt->second.ptr;
        return branch;
    }

    bool enqueueOneStep(DeviceBranch const& branch, void* xPtr, void* nextXPtr, float t, float dt, cudaStream_t stream)
    {
        memcpyToDevice(tBuffer.ptr, &t, sizeof(float));
        memcpyToDevice(dtBuffer.ptr, &dt, sizeof(float));

        context->setInputShape("x", xDims);
        context->setInputShape("t", engine->getTensorShape("t"));
        context->setInputShape("dt", engine->getTensorShape("dt"));
        context->setInputShape("kv_cache", kvDims);
        context->setInputShape("attention_mask", maskDims);
        context->setInputShape("position_ids", posDims);

        check::check(context->setTensorAddress("x", xPtr), "Failed to bind FM input x");
        check::check(context->setTensorAddress("t", tBuffer.ptr), "Failed to bind FM input t");
        check::check(context->setTensorAddress("dt", dtBuffer.ptr), "Failed to bind FM input dt");
        check::check(context->setTensorAddress("kv_cache", branch.kvCachePtr), "Failed to bind FM input kv_cache");
        check::check(context->setTensorAddress("attention_mask", branch.attentionMaskPtr),
            "Failed to bind FM input attention_mask");
        check::check(context->setTensorAddress("position_ids", branch.positionIdsPtr),
            "Failed to bind FM input position_ids");
        check::check(context->setTensorAddress("next_x", nextXPtr), "Failed to bind FM output next_x");
        check::check(context->setTensorAddress("v", vBuffer.ptr), "Failed to bind FM output v");
        check::check(context->setTensorAddress("future_token_embeds", futureBuffer.ptr),
            "Failed to bind FM output future_token_embeds");

        check::check(context->enqueueV3(stream), "FM TensorRT enqueueV3 failed");
        return true;
    }

    bool runOneStepToHost(DeviceBranch const& branch, std::vector<float> const& x, float t, float dt, std::vector<float>& nextX,
        cudaStream_t stream, float* elapsedMsOut = nullptr)
    {
        auto const stepStart = SteadyClock::now();
        check::check(static_cast<int32_t>(x.size()) == horizon * actionDim, "Input x size mismatch for FM one-step");
        nextX.resize(static_cast<size_t>(horizon) * actionDim);

        memcpyToDevice(xBuffer.ptr, x.data(), x.size() * sizeof(float));
        check::check(enqueueOneStep(branch, xBuffer.ptr, nextXBuffer.ptr, t, dt, stream), "FM TensorRT enqueueV3 failed");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        memcpyToHost(nextX.data(), nextXBuffer.ptr, nextX.size() * sizeof(float));
        if (elapsedMsOut != nullptr)
        {
            *elapsedMsOut = ::trt_edgellm::rt::elapsedMs(stepStart, SteadyClock::now());
        }
        return true;
    }

    bool runSingleBranch(AlpamayoFmBranchSnapshot const& snapshot, AlpamayoFmRunConfig const& config, std::vector<float>& x0,
        std::vector<float>& xFinal, AlpamayoFmRunResult::Timing& timing, cudaStream_t stream)
    {
        auto const totalStart = SteadyClock::now();
        auto const branchStart = SteadyClock::now();
        DeviceBranch branch = makeDeviceBranch(snapshot, singleBranchWorkspace, timing, stream);
        timing.branchPrepareMs = ::trt_edgellm::rt::elapsedMs(branchStart, SteadyClock::now());

        auto const x0Start = SteadyClock::now();
        x0 = makeNormalX0(horizon, actionDim, config.seed);
        timing.x0InitMs = ::trt_edgellm::rt::elapsedMs(x0Start, SteadyClock::now());
        xFinal.resize(x0.size());
        float const dt = 1.0F / static_cast<float>(config.numSteps);
        memcpyToDevice(xBuffer.ptr, x0.data(), x0.size() * sizeof(float));
        void* currentXPtr = xBuffer.ptr;
        void* nextXPtr = nextXBuffer.ptr;
        auto const diffusionStart = SteadyClock::now();
        for (int32_t step = 0; step < config.numSteps; ++step)
        {
            float const t = static_cast<float>(step) * dt;
            check::check(enqueueOneStep(branch, currentXPtr, nextXPtr, t, dt, stream), "FM TensorRT enqueueV3 failed");
            std::swap(currentXPtr, nextXPtr);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        timing.engineStepTotalMs = ::trt_edgellm::rt::elapsedMs(diffusionStart, SteadyClock::now());
        memcpyToHost(xFinal.data(), currentXPtr, xFinal.size() * sizeof(float));
        timing.numSteps = config.numSteps;
        timing.numBranches = 1;
        timing.engineStepAvgMs = config.numSteps > 0 ? timing.engineStepTotalMs / static_cast<float>(config.numSteps) : 0.0F;
        timing.totalMs = ::trt_edgellm::rt::elapsedMs(totalStart, SteadyClock::now());
        return true;
    }

    bool runDualBranch(AlpamayoFmBranchSnapshot const& guidedSnapshot, AlpamayoFmBranchSnapshot const& unguidedSnapshot,
        AlpamayoFmRunConfig const& config, std::vector<float>& x0, std::vector<float>& xFinal,
        AlpamayoFmRunResult::Timing& timing, cudaStream_t stream)
    {
        auto const totalStart = SteadyClock::now();
        auto const branchStart = SteadyClock::now();
        DeviceBranch guided = makeDeviceBranch(guidedSnapshot, guidedBranchWorkspace, timing, stream);
        DeviceBranch unguided = makeDeviceBranch(unguidedSnapshot, unguidedBranchWorkspace, timing, stream);
        timing.branchPrepareMs = ::trt_edgellm::rt::elapsedMs(branchStart, SteadyClock::now());

        auto const x0Start = SteadyClock::now();
        x0 = makeNormalX0(horizon, actionDim, config.seed);
        timing.x0InitMs = ::trt_edgellm::rt::elapsedMs(x0Start, SteadyClock::now());
        xFinal = x0;
        std::vector<float> guidedNext;
        std::vector<float> unguidedNext;
        float const dt = 1.0F / static_cast<float>(config.numSteps);
        auto const diffusionStart = SteadyClock::now();
        for (int32_t step = 0; step < config.numSteps; ++step)
        {
            float const t = static_cast<float>(step) * dt;
            float guidedStepMs = 0.0F;
            float unguidedStepMs = 0.0F;
            runOneStepToHost(guided, xFinal, t, dt, guidedNext, stream, &guidedStepMs);
            runOneStepToHost(unguided, xFinal, t, dt, unguidedNext, stream, &unguidedStepMs);
            timing.engineStepTotalMs += guidedStepMs + unguidedStepMs;
            for (size_t i = 0; i < xFinal.size(); ++i)
            {
                xFinal[i] = (1.0F - config.guidanceWeight) * unguidedNext[i] + config.guidanceWeight * guidedNext[i];
            }
        }
        timing.engineStepTotalMs = std::max(
            timing.engineStepTotalMs, ::trt_edgellm::rt::elapsedMs(diffusionStart, SteadyClock::now()));
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
    nvinfer1::DataType kvDataType{nvinfer1::DataType::kHALF};
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
    std::unordered_map<int32_t, CudaBuffer> attentionMaskCache;
    std::unordered_map<PositionIdsCacheKey, CudaBuffer, PositionIdsCacheKeyHash> positionIdsCache;
    BranchWorkspace singleBranchWorkspace;
    BranchWorkspace guidedBranchWorkspace;
    BranchWorkspace unguidedBranchWorkspace;
};

AlpamayoFmRuntime::AlpamayoFmRuntime(std::string enginePath)
    : mImpl(std::make_unique<Impl>(std::move(enginePath)))
{
}

AlpamayoFmRuntime::~AlpamayoFmRuntime() = default;

bool AlpamayoFmRuntime::runNoNav(AlpamayoFmBranchSnapshot const& branch,
    LLMGenerationRequest::ActionSpaceConstants const& constants, std::vector<float> const& egoHistoryXyz,
    std::vector<int64_t> const& egoHistoryXyzShape, std::vector<float> const& egoHistoryRot,
    std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config, AlpamayoFmRunResult& result,
    cudaStream_t stream)
{
    auto const totalStart = SteadyClock::now();
    std::vector<float> x0;
    std::vector<float> xFinal;
    check::check(mImpl->runSingleBranch(branch, config, x0, xFinal, result.timing, stream), "Failed to run native no-nav FM TRT");

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
    std::vector<int64_t> const& egoHistoryRotShape, AlpamayoFmRunConfig const& config, AlpamayoFmRunResult& result,
    cudaStream_t stream)
{
    auto const totalStart = SteadyClock::now();
    std::vector<float> x0;
    std::vector<float> xFinal;
    check::check(
        mImpl->runDualBranch(guided, unguided, config, x0, xFinal, result.timing, stream),
        "Failed to run native nav CFG FM TRT");

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

nvinfer1::DataType AlpamayoFmRuntime::kvCacheDataType() const noexcept
{
    return mImpl->kvDataType;
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
