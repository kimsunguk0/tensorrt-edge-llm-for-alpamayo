#include "actionExpertRunner.h"

#include "common/checkMacros.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace examples
{
namespace
{

template <typename DstT, typename SrcT>
__device__ __forceinline__ DstT convertValue(SrcT value);

template <>
__device__ __forceinline__ half convertValue<half, half>(half value)
{
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 convertValue<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 value)
{
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 convertValue<__nv_bfloat16, half>(half value)
{
    return __float2bfloat16(__half2float(value));
}

template <>
__device__ __forceinline__ half convertValue<half, __nv_bfloat16>(__nv_bfloat16 value)
{
    return __float2half(__bfloat162float(value));
}

template <typename DstT, typename SrcT>
__global__ void copyKvCacheKernel(SrcT const* src, DstT* dst, int32_t srcBatchSize, int32_t batchIdx,
    int32_t layers, int32_t numKv, int32_t numHeads, int32_t srcSeqCapacity, int32_t dstSeqCapacity,
    int32_t headDim, int32_t validSeqLen)
{
    int64_t total = static_cast<int64_t>(layers) * numKv * numHeads * validSeqLen * headDim;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    int32_t d = static_cast<int32_t>(idx % headDim);
    idx /= headDim;
    int32_t seq = static_cast<int32_t>(idx % validSeqLen);
    idx /= validSeqLen;
    int32_t head = static_cast<int32_t>(idx % numHeads);
    idx /= numHeads;
    int32_t kv = static_cast<int32_t>(idx % numKv);
    idx /= numKv;
    int32_t layer = static_cast<int32_t>(idx);

    int64_t srcIndex = (((((static_cast<int64_t>(layer) * srcBatchSize + batchIdx) * numKv + kv) * numHeads + head)
                           * srcSeqCapacity
                          + seq)
                           * headDim
        + d);
    int64_t dstIndex = (((((static_cast<int64_t>(layer) * 1 + 0) * numKv + kv) * numHeads + head) * dstSeqCapacity + seq)
                           * headDim
        + d);
    dst[dstIndex] = convertValue<DstT>(src[srcIndex]);
}

template <typename DstT, typename SrcT>
bool dispatchCopy(rt::Tensor const& src, int32_t srcBatchSize, int32_t batchIdx, int32_t validSeqLen, rt::Tensor& dst,
    cudaStream_t stream)
{
    int32_t const layers = static_cast<int32_t>(dst.getShape()[0]);
    int32_t const numKv = static_cast<int32_t>(dst.getShape()[2]);
    int32_t const numHeads = static_cast<int32_t>(dst.getShape()[3]);
    int32_t const srcSeqCapacity = static_cast<int32_t>(src.getShape()[4]);
    int32_t const dstSeqCapacity = static_cast<int32_t>(dst.getShape()[4]);
    int32_t const headDim = static_cast<int32_t>(dst.getShape()[5]);

    CUDA_CHECK(cudaMemsetAsync(dst.rawPointer(), 0, dst.getMemoryCapacity(), stream));
    int64_t const total = static_cast<int64_t>(layers) * numKv * numHeads * validSeqLen * headDim;
    if (total == 0)
    {
        return true;
    }
    int32_t constexpr blockSize = 256;
    int32_t const gridSize = static_cast<int32_t>((total + blockSize - 1) / blockSize);
    copyKvCacheKernel<DstT, SrcT><<<gridSize, blockSize, 0, stream>>>(src.dataPointer<SrcT>(), dst.dataPointer<DstT>(),
        srcBatchSize, batchIdx, layers, numKv, numHeads, srcSeqCapacity, dstSeqCapacity, headDim, validSeqLen);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

bool copyAndPadKVCache(rt::Tensor const& src, int32_t srcBatchSize, int32_t batchIdx, int32_t validSeqLen,
    rt::Tensor& dst, cudaStream_t stream)
{
    check::check(src.getDeviceType() == rt::DeviceType::kGPU, "Action-expert KV source must reside on GPU.");
    check::check(dst.getDeviceType() == rt::DeviceType::kGPU, "Action-expert KV destination must reside on GPU.");
    check::check(src.getShape().getNumDims() == 6 && dst.getShape().getNumDims() == 6,
        "Action-expert KV tensors must be 6D.");
    check::check(batchIdx >= 0 && batchIdx < srcBatchSize, "Invalid batchIdx for action-expert KV copy.");
    check::check(validSeqLen >= 0 && validSeqLen <= src.getShape()[4] && validSeqLen <= dst.getShape()[4],
        "Invalid validSeqLen for action-expert KV copy.");

    auto const srcType = src.getDataType();
    auto const dstType = dst.getDataType();
    if (srcType == nvinfer1::DataType::kHALF && dstType == nvinfer1::DataType::kBF16)
    {
        return dispatchCopy<__nv_bfloat16, half>(src, srcBatchSize, batchIdx, validSeqLen, dst, stream);
    }
    if (srcType == nvinfer1::DataType::kBF16 && dstType == nvinfer1::DataType::kBF16)
    {
        return dispatchCopy<__nv_bfloat16, __nv_bfloat16>(src, srcBatchSize, batchIdx, validSeqLen, dst, stream);
    }
    if (srcType == nvinfer1::DataType::kHALF && dstType == nvinfer1::DataType::kHALF)
    {
        return dispatchCopy<half, half>(src, srcBatchSize, batchIdx, validSeqLen, dst, stream);
    }
    if (srcType == nvinfer1::DataType::kBF16 && dstType == nvinfer1::DataType::kHALF)
    {
        return dispatchCopy<half, __nv_bfloat16>(src, srcBatchSize, batchIdx, validSeqLen, dst, stream);
    }

    throw std::runtime_error("Unsupported KV dtype conversion in copyAndPadKVCache.");
}

} // namespace examples
} // namespace trt_edgellm
