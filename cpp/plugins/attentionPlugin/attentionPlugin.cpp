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

#include "attentionPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};

// Workaround for CUDA12/13 Thor re-numbering. The kernels themselves have version compatibility.
void applyThorSMRenumberWAR(int32_t& smVersion)
{
    if (smVersion == 110)
    {
        smVersion = 101;
    }
}

// Select KV cache storage datatype based on FP8 enablement
static inline DataType selectKvCacheDataType(bool enableFp8KVCache)
{
    return enableFp8KVCache ? DataType::kFP8 : DataType::kHALF;
}

// Define the mapping of input and output indices of the AttentionPlugin.
constexpr int32_t kIN_QKV_IDX{0};
constexpr int32_t kIN_KV_CACHE_IDX{1};
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{2};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{3};
constexpr int32_t kIN_KV_CACHE_START_IDX{4};
constexpr int32_t kIN_OPTIONAL_ATTN_MASK_IDX{5};
constexpr int32_t kIN_OPTIONAL_ATTN_POS_ID_IDX{6};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

// Reflect the count of Inputs and Outputs of the AttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{5};
constexpr int32_t kNUM_TREE_ATTN_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_FP8_KVCACHE_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};

// Support Tree Attention decoding schema up to 128 tokens in the draft tree per batch.
// We are unable to check this property during shape checking since prefill length is much larger than this value.
constexpr int64_t kMAX_EAGLE_DECODING_TOKENS = 128;

enum class AttentionExecutionMode
{
    kINVALID,
    kNORMAL_PREFILL,
    kCHUNKED_PREFILL,
    kVANILLA_DECODING,
    kTREE_DECODING
};

AttentionExecutionMode deduceModeVanilla(rt::Tensor const& qkvInputTensor, rt::Tensor const& kvCacheStartIdxTensor)
{
    // Empty KVCache Start indices means normal prefill without previous KVCache. Notice single token is also a valid
    // prefill length.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Otherwise, distinguish between chunked prefill and vanilla decoding based on the runtime Sequence Length.
    // Vanilla decoding should always have runtime sequence length of 1.
    int64_t const runtimeSeqLen = qkvInputTensor.getShape()[1];
    if (runtimeSeqLen > 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }
    return AttentionExecutionMode::kVANILLA_DECODING;
}

AttentionExecutionMode deduceModeTreeAttention(
    rt::Tensor const& qkvInputTensor, rt::Tensor const& kvCacheStartIdxTensor, rt::Tensor const& attentionPosIdTensor)
{
    // Normal prefill if there is no previous KVCache.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Under tree attention, each token will be associated with a position id (within the sequence) to perform correct
    // positional encoding. Even for casual decoding with multiple tokens, the position id is still required to be
    // supplied.

    // Note, chunked prefill is very similar to tree decoding, the difference is chunked prefill will have contiguous
    // tokens in the sequence while tree decoding has a "tree" structure described by attention mask and position ids.
    // By convention, we will supply 1 shape for position id tensor under prefill execution.
    int64_t const runtimeSeqLen = qkvInputTensor.getShape()[1];
    int64_t const positionIdLen = attentionPosIdTensor.getShape()[1];

    if (runtimeSeqLen == 1)
    {
        // Also supports single token decoding mode when tree attention is enabled.
        return AttentionExecutionMode::kVANILLA_DECODING;
    }
    else if (positionIdLen == runtimeSeqLen)
    {
        return AttentionExecutionMode::kTREE_DECODING;
    }
    else if (positionIdLen == 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }

    return AttentionExecutionMode::kINVALID;
}

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableFp8KVCache(enableFp8KVCache)
{
    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    bool canImplementFMHA = ContextFMHARunner::canImplement(mHeadSize, mSMVersion, mDataType);
    bool canImplementXQA = DecoderXQARunner::canImplement(
        mNumQHeads, mNumKVHeads, mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache));

    if (!canImplementFMHA || !canImplementXQA)
    {
        LOG_ERROR(
            "Cannot implement AttentionPlugin configuration. FMHA: %s, XQA: %s, SM: %d, HeadSize: %d, NumQHeads: %d, "
            "NumKVHeads: %d",
            canImplementFMHA ? "supported" : "NOT supported", canImplementXQA ? "supported" : "NOT supported",
            mSMVersion, mHeadSize, mNumQHeads, mNumKVHeads);
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }

    // Load FMHA and XQA kernels to device. The kernel code will only be loaded once if
    // multiple AttentionPlugin instances exist in the model.
    // TODO: Fix me to pass spec-decode support through plugin attributes.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    DecoderXQARunner::loadDecodeXQAKernels(
        mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
}

AttentionPlugin::AttentionPlugin(std::string const& name, std::byte const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumQHeads);
    deserializeValue(&data, &length, &mNumKVHeads);
    deserializeValue(&data, &length, &mHeadSize);
    deserializeValue(&data, &length, &mEnableTreeAttention);
    deserializeValue(&data, &length, &mEnableFp8KVCache);

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    // TODO: Fix me too pass spec-deocde support through plugin attributes.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    DecoderXQARunner::loadDecodeXQAKernels(
        mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
}

AttentionPlugin::~AttentionPlugin() {}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin
        = new AttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEnableTreeAttention, mEnableFp8KVCache);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* AttentionPlugin::getPluginType() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      GEMM-QKV tensor (linear FP16) with shape [B, S, Hq+Hk+Hv, D]
    //      KV-cache tensor (linear FP16/FP8) with shape [B, 2, Hkv, Smax, D], here Smax is the kvcache capacity
    //      buffer.
    //      Real context length: [B] (a vector of scalars) with type int32_t.
    //      RoPE cos/sin cache: [B or 1, Smax, D] (a tensor of scalars) with type float.
    //            Rope CosSin can be ND vector depending on rope type.
    //      Start index of the KVCache [B, 0~1] (a vector of scalars) with type int32_t.
    //            0 length indicates there is no existing KVCache for inference.
    //      Optional tree attention mask: [B, S, S] (a tensor of scalars) with type int32_t.
    //      Optional tree attention position ids: [B, S] (a tensor of scalars) with type int32_t.
    //      Optional packed FP8 KV-cache dequant scales (K/V quant->orig): [2] (a tensor of scalars) with type float.
    //            Layout: [k_scale_quant_orig, v_scale_quant_orig].

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [B, S, Hq, D]
    //      KV-cache tensor, same as the above.
    auto checkGemmQKV = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[2] == (mNumQHeads + mNumKVHeads + mNumKVHeads) * mHeadSize;
        }
        return status;
    };

    auto checkKVCache = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        // Support FP16 or FP8 storage;
        if (mEnableFp8KVCache)
        {
            status &= (tensorDesc.type == DataType::kFP8);
        }
        else
        {
            status &= (tensorDesc.type == DataType::kHALF);
        }
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 5;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[1] == 2; // Specify K and V
            status &= tensorDim.d[2] == mNumKVHeads;
            status &= tensorDim.d[4] == mHeadSize;
        }
        return status;
    };

    auto checkSequenceLen = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkPosEncodingCosSin = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        status &= tensorDesc.dims.d[2] <= mHeadSize;
        return status;
    };
    auto checkAttentionMask = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        return status;
    };
    auto checkAttentionPosId = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        return status;
    };
    auto checkKVCacheStartIdx = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkKVScale = [](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        // Packed FP8 KV cache scales: [k_scale_quant_orig, v_scale_quant_orig]
        status &= tensorDesc.dims.nbDims == 1;
        status &= tensorDesc.dims.d[0] == 2;
        return status;
    };

    // Output tensor checks
    auto checkAttentionOutput = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 4;
        if (status)
        {
            auto const tensorDim = tensorDesc.dims;
            status &= tensorDim.d[2] == mNumQHeads;
            status &= tensorDim.d[3] == mHeadSize;
        }
        return status;
    };

    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (mEnableFp8KVCache ? kNUM_FP8_KVCACHE_OPTIONAL_INPUTS : 0);
    bool const checkNumIOs = nbInputs == expectedNbInputs && nbOutputs == kNUM_REQUIRED_OUTPUTS;
    if (!checkNumIOs)
    {
        LOG_ERROR(
            "Invalid number of inputs or outputs for the AttentionPlugin '%s'. Expected %d inputs and %d outputs, but "
            "got %d inputs and %d outputs.",
            mLayerName.c_str(), expectedNbInputs, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_QKV_IDX: result = checkGemmQKV(inOut[0]); break;
        case kIN_KV_CACHE_IDX: result = checkKVCache(inOut[1]); break;
        case kIN_CONTEXT_LENGTH_IDX: result = checkSequenceLen(inOut[2]); break;
        case kIN_ROPE_COS_SIN_IDX: result = checkPosEncodingCosSin(inOut[3]); break;
        case kIN_KV_CACHE_START_IDX: result = checkKVCacheStartIdx(inOut[4]); break;
        default: break;
        }

        // Handle optional inputs (tree attention mask/pos and FP8 scales) with dynamic ordering
        if (result && pos > kIN_KV_CACHE_START_IDX)
        {
            int32_t currentOptionalInputIdx = kIN_KV_CACHE_START_IDX + 1;
            if (mEnableTreeAttention)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionMask(inOut[pos]);
                }
                currentOptionalInputIdx++;
                if (pos == currentOptionalInputIdx)
                {
                    result = checkAttentionPosId(inOut[pos]);
                }
                currentOptionalInputIdx++;
            }
            if (mEnableFp8KVCache)
            {
                if (pos == currentOptionalInputIdx)
                {
                    result = checkKVScale(inOut[pos]);
                }
                currentOptionalInputIdx++;
            }
        }
    }
    else
    {
        int32_t outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: result = checkAttentionOutput(inOut[pos]); break;
        case kOUT_KV_CACHE_IDX: result = checkKVCache(inOut[pos]); break;
        default: break;
        }
    }

    return result;
}

// IPluginV2Ext Methods
DataType AttentionPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    // Output[0] (attention) follows QKV input dtype (HALF). Output[1] (KV cache) follows KV input dtype (HALF or FP8)
    return inputTypes[index];
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
    DimsExprs output;
    if (outputIndex == kOUT_ATTENTION_IDX)
    {
        output.nbDims = 4;
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = exprBuilder.constant(mNumQHeads);
        output.d[3] = exprBuilder.constant(mHeadSize);
    }
    else if (outputIndex == kOUT_KV_CACHE_IDX)
    {
        // Output[1] is KVCache, same shape as input KV cache
        output.nbDims = 5;
        output.d[0] = inputs[1].d[0];
        output.d[1] = inputs[1].d[1];
        output.d[2] = inputs[1].d[2];
        output.d[3] = inputs[1].d[3];
        output.d[4] = inputs[1].d[4];
    }
    return output;
}

void AttentionPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return; // No need to configure anything since we will only use the runtime tensor shapes.
}

// TODO: extend the workspace calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // TensorRT will supply max profile shape for each input/output tensor across all optimization profiles.
    // We will request workspace to keep intermediate tensors under prefill/decode phase executions.
    // Obtain max supported batch size from the input tensor shapes. The QKV input tensor will be in shape
    // [B, S, Hq+Hk+Hv, D] where S is padded the max length of the input sequence within this batch..
    PluginTensorDesc const& qkvInputDesc = inputs[kIN_QKV_IDX];
    int64_t const maxBatchSize = qkvInputDesc.dims.d[0];
    int64_t const maxInputSeqLen = qkvInputDesc.dims.d[1];

    // Obtain max KV cache capacity from the KV cache tensor shape.
    // The KV cache tensor has shape [B, 2, num_kv_heads, capacity, head_dim]
    PluginTensorDesc const& kvCacheDesc = inputs[kIN_KV_CACHE_IDX];
    int64_t const maxKVCacheCapacity = kvCacheDesc.dims.d[3];

    size_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {maxBatchSize + 1}, DataType::kINT32);
    // The workspace will be used to store the Q tensor. For eagle mode, maxDecodingTokens means the number of Q tensor.
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, {maxBatchSize, kMAX_EAGLE_DECODING_TOKENS, mNumQHeads, mHeadSize}, DataType::kHALF);

    // Always reserve workspace memory to prepare for chunked prefill decoding. The implementation should be further
    // optimized to avoid the workspace size overhead.

    // CuTotalKvCacheLens to describe the cumulative length of KV tensors.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
    // KVCache ends that denote the end index of each KVCache lane after adding current contents.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize}, DataType::kINT32);
    // Separate Q Tensor space to keep roped Q results.
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{maxBatchSize, maxInputSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
    // KV Tensor to store concated KV that include pre-cached KV and current KV.
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{maxBatchSize, 2, mNumKVHeads, maxKVCacheCapacity, mHeadSize}, DataType::kHALF);

    // Request another alignment size to align the workspace pointer.
    workspaceSize += kDEVICE_ALIGNMENT;

    LOG_DEBUG("AttentionPlugin workspace size: %zu bytes", workspaceSize);
    return workspaceSize;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{

    // Construct non-owned tensor objects from I/O data pointers and shapes.
    // QKV inputs in the graph will be in shape [B, S, (Hq + Hk + Hv) x D], for convenience,
    // we will use shape of [B, S, Hq + Hk + Hv, D] to represent the tensor.
    PluginTensorDesc const& qkvInputDesc = inputDesc[kIN_QKV_IDX];
    int32_t const runtimeBatchSize = static_cast<int32_t>(qkvInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qkvInputDesc.dims.d[1]);
    check::check(qkvInputDesc.dims.d[2] == (mNumQHeads + mNumKVHeads + mNumKVHeads) * mHeadSize,
        "QKV input shape shall be consistent.");
    rt::Tensor qkvInputTensor(const_cast<void*>(inputs[kIN_QKV_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads + mNumKVHeads + mNumKVHeads, mHeadSize},
        rt::DeviceType::kGPU, qkvInputDesc.type);

    PluginTensorDesc const& contextLengthInputDesc = inputDesc[kIN_CONTEXT_LENGTH_IDX];
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{contextLengthInputDesc.dims}, rt::DeviceType::kGPU, contextLengthInputDesc.type);

    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{posEncodingCosSinDesc.dims}, rt::DeviceType::kGPU, posEncodingCosSinDesc.type);

    PluginTensorDesc const& kvCacheStartIdxInputDesc = inputDesc[kIN_KV_CACHE_START_IDX];
    rt::Tensor const kvCacheStartIdxTensor(const_cast<void*>(inputs[kIN_KV_CACHE_START_IDX]),
        rt::Coords{kvCacheStartIdxInputDesc.dims}, rt::DeviceType::kGPU, kvCacheStartIdxInputDesc.type);

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    // Construct the KVCache tensor from the input KV cache descriptor.
    // This allows KV cache from 0 to maxSeqLen and helps adjust the profile at runtime.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor kvCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);

    // Extract KV cache capacity from the runtime tensor shape.
    int32_t const kvCacheCapacity = static_cast<int32_t>(kvCacheInputDesc.dims.d[3]);

    // Optional Inputs that are not used with Tree Attention enabled.
    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};
    if (mEnableTreeAttention)
    {
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[kIN_OPTIONAL_ATTN_MASK_IDX];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[kIN_OPTIONAL_ATTN_POS_ID_IDX];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_MASK_IDX]),
            rt::Coords{attentionMaskInputDesc.dims}, rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_POS_ID_IDX]),
            rt::Coords{attentionPosIdInputDesc.dims}, rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }

    // Determine the attention execution mode based on the input tensors.
    AttentionExecutionMode executionMode{};
    if (!mEnableTreeAttention)
    {
        executionMode = deduceModeVanilla(qkvInputTensor, kvCacheStartIdxTensor);
    }
    else
    {
        executionMode = deduceModeTreeAttention(qkvInputTensor, kvCacheStartIdxTensor, attentionPosIdTensor);
    }

    // For invalid execution mode, log error and report error return value.
    if (executionMode == AttentionExecutionMode::kINVALID)
    {
        LOG_ERROR("Invalid attention execution mode detected. Abort the AttentionPlugin enqueue() call.");
        return 1;
    }

    // Optional packed FP8 KV cache scales: [k_scale_quant_orig, v_scale_quant_orig]
    // When FP8 KV cache is disabled, this tensor stays empty and downstream kernels ignore it.
    rt::Tensor kvScaleQuantOrigTensor{};
    if (mEnableFp8KVCache)
    {
        // Optional inputs begin after kIN_KV_CACHE_START_IDX; tree attention contributes two inputs if enabled.
        int32_t const kvScaleQuantOrigInputIdx
            = kIN_KV_CACHE_START_IDX + 1 + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0);
        PluginTensorDesc const& kvScaleQuantOrigDesc = inputDesc[kvScaleQuantOrigInputIdx];
        kvScaleQuantOrigTensor = rt::Tensor(const_cast<void*>(inputs[kvScaleQuantOrigInputIdx]),
            rt::Coords{kvScaleQuantOrigDesc.dims}, rt::DeviceType::kGPU, kvScaleQuantOrigDesc.type);
        // Runtime validation (in addition to supportsFormatCombination)
        check::check(
            kvScaleQuantOrigTensor.getDataType() == nvinfer1::DataType::kFLOAT, "kvScaleQuantOrig must be FP32.");
        check::check(kvScaleQuantOrigTensor.getShape().getNumDims() == 1 && kvScaleQuantOrigTensor.getShape()[0] == 2,
            "kvScaleQuantOrig shall have shape [2] with layout [kScaleQuantOrig, vScaleQuantOrig].");
    }

    // Align the workspace pointer so that each tensor assigned from the workspace will align to the device alignment
    // granularity.
    auto const nbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (mEnableFp8KVCache ? kNUM_FP8_KVCACHE_OPTIONAL_INPUTS : 0);
    auto const nbOutputs = kNUM_REQUIRED_OUTPUTS;
    size_t space = getWorkspaceSize(inputDesc, nbInputs, outputDesc, nbOutputs);
    std::byte* alignedWorkspacePtr
        = static_cast<std::byte*>(std::align(kDEVICE_ALIGNMENT, space - kDEVICE_ALIGNMENT, workspace, space));
    if (alignedWorkspacePtr == nullptr)
    {
        LOG_ERROR("Workspace size is too small to hold all data structures with correct alignment");
        return 1;
    }

    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
        rt::Tensor cuQSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);

        rt::Tensor cuKVSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);
        rt::Tensor kvCacheEndIdxsTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);

        kernel::calCuQCuKVSeqLensAndKVEndIdxs(contextLengthTensor, kvCacheStartIdxTensor, cuQSeqLensTensor,
            cuKVSeqLensTensor, kvCacheEndIdxsTensor, runtimeSeqLen, stream);

        AttentionInputLayout attentionInputLayout = executionMode == AttentionExecutionMode::kCHUNKED_PREFILL
            ? AttentionInputLayout::CONTIGUOUS_Q_KV
            : AttentionInputLayout::PACKED_QKV;
        auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
            mHeadSize, mSMVersion, attentionInputLayout);

        // Prepare FMHA_v2 params to launch FMHA kernel
        FusedMultiheadAttentionParamsV2 params{};
        memset(&params, 0, sizeof(params));
        fmhaRunner.setupParams(params);
        params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

        if (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
        {
            // Assign Q tensor to keep roped Q results with layout [B, Sq, Hq, D].
            rt::Tensor qOutTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
            // q: [b, s, hq+hk+hv, d] -> [b, s, hq, d]
            kernel::launchApplyRopeWriteKVContinuousQAndKVCache(ropeCosSinTensor, kvCacheEndIdxsTensor, qkvInputTensor,
                kvCacheTensor, qOutTensor, kvScaleQuantOrigTensor, stream);

            // FMHA kernel with CONTIGUOUS_Q_KV input layout currently only supports FP16 KV cache.
            rt::Tensor transposedKVTensor = assignTensorFromWorkspace(
                alignedWorkspacePtr, {runtimeBatchSize, kvCacheCapacity, 2, mNumKVHeads, mHeadSize}, DataType::kHALF);
            // kvCache: [b, 2, hkv, s, d] -> [b, s, 2, hkv, d]
            kernel::cvtKVLayoutBHSDToBSHD(kvCacheTensor, transposedKVTensor, kvScaleQuantOrigTensor, stream);

            // Set device ptr for FMHA kernel.
            params.s_kv = kvCacheCapacity;
            params.q_ptr = qOutTensor.dataPointer<half>();
            params.kv_ptr = transposedKVTensor.dataPointer<half>();
            params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
            params.o_ptr = attentionOutputTensor.dataPointer<half>();
        }
        else
        { // PACKED_QKV
            kernel::launchApplyRopeWriteKVPackedQKV(
                ropeCosSinTensor, qkvInputTensor, kvCacheTensor, kvScaleQuantOrigTensor, stream);

            params.qkv_ptr = qkvInputTensor.dataPointer<half>();
            params.cu_kv_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();
            params.o_ptr = attentionOutputTensor.dataPointer<half>();
        }

        // Dispatch FMHA kernel
        fmhaRunner.dispatchFMHAKernel(params, stream);
    }
    else
    {
        rt::Tensor qOutTensor = assignTensorFromWorkspace(
            alignedWorkspacePtr, {runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, DataType::kHALF);
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            kernel::launchApplyRopeWriteKVTreeDecoding(ropeCosSinTensor, contextLengthTensor, attentionPosIdTensor,
                qkvInputTensor, kvCacheTensor, qOutTensor, kvScaleQuantOrigTensor, stream);
        }
        else
        {
            kernel::launchApplyRopeWriteKVContinuousQAndKVCache(ropeCosSinTensor, contextLengthTensor, qkvInputTensor,
                kvCacheTensor, qOutTensor, kvScaleQuantOrigTensor, stream);
        }
        // Prepare Decoding attention runner parameter to dispatch kernel
        auto xqaRunner = DecoderXQARunner(mDataType, selectKvCacheDataType(mEnableFp8KVCache), runtimeBatchSize,
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        float const* const kvScales = mEnableFp8KVCache ? kvScaleQuantOrigTensor.dataPointer<float>() : nullptr;
        if (mEnableFp8KVCache)
        {
            float const* const kvScales = kvScaleQuantOrigTensor.dataPointer<float>();
            params.kScale = (kvScales + 0);
            params.vScale = (kvScales + 1);
        }
        params.output = attentionOutputTensor.dataPointer<half>();
        params.qInputPtr = qOutTensor.dataPointer<half>();
        params.kvCache.data = kvCacheTensor.rawPointer();
        params.kvCache.sequence_lengths = contextLengthTensor.dataPointer<int32_t>();
        params.kvCache.capacity = kvCacheCapacity;
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Execute tree attention decoding.
            params.treeAttnMask = attentionMaskTensor.dataPointer<int32_t>();
            params.qSeqLen = runtimeSeqLen;
            xqaRunner.dispatchSpecDecodeXQAKernel(params, stream);
        }
        else
        {
            // Execute vanilla decoding.
            xqaRunner.dispatchXQAKernel(params, stream);
        }
    }
    return 0;
}

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumQHeads) + sizeof(mNumKVHeads) + sizeof(mHeadSize) + sizeof(mEnableTreeAttention)
        + sizeof(mEnableFp8KVCache);
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    std::byte* byteBuffer = static_cast<std::byte*>(buffer);
    serializeValue(&byteBuffer, mNumQHeads);
    serializeValue(&byteBuffer, mNumKVHeads);
    serializeValue(&byteBuffer, mHeadSize);
    serializeValue(&byteBuffer, mEnableTreeAttention);
    serializeValue(&byteBuffer, mEnableFp8KVCache);
}

int32_t AttentionPlugin::initialize() noexcept
{
    return 0;
}

void AttentionPlugin::terminate() noexcept {}

void AttentionPlugin::destroy() noexcept
{
    delete this;
}

AttentionPluginCreator::AttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("enable_tree_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> numQHeads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> numKVHeads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        std::optional<int32_t> enableTreeAttention = parsePluginScalarField<int32_t>("enable_tree_attention", fc);
        // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
        std::optional<int32_t> enableFp8KVCache = parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc);
        int32_t enableFp8KVCacheValue = enableFp8KVCache.value_or(0);

        // Enforce Core parameters are specified.
        bool checkRequiredFields = numQHeads.has_value() && headSize.has_value() && numKVHeads.has_value()
            && enableTreeAttention.has_value();
        if (!checkRequiredFields)
        {
            LOG_ERROR("Missing required AttentionPlugin fields.");
            return nullptr;
        }

        AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(),
            headSize.value(), enableTreeAttention.value(), enableFp8KVCacheValue);

        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* AttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new AttentionPlugin(name, static_cast<std::byte const*>(serialData), serialLength);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
