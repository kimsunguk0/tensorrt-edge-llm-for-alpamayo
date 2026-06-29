/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Adapted from https://github.com/vllm-project/vllm/blob/v0.14.0/csrc/moe/marlin_moe_wna16/kernel.h
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */
#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "../marlin/marlin.cuh"
#include "../marlin/marlin_dtypes.cuh"
#include "../marlin/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                                                                           \
    const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
        const int4 *__restrict__ b_bias_ptr, const float *__restrict__ a_scales_ptr,                                   \
        const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ global_scale_ptr,                            \
        const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,                                                \
        const int32_t *__restrict__ sorted_token_ids_ptr, const int32_t *__restrict__ expert_ids_ptr,                  \
        const int32_t *__restrict__ num_tokens_past_padded_ptr, const float *__restrict__ topk_weights_ptr, int top_k, \
        bool mul_topk_weights, int num_groups, int prob_m, int prob_n, int prob_k, int *locks, bool has_bias,          \
        bool use_atomic_add, bool use_fp32_reduce

namespace MARLIN_NAMESPACE_NAME
{
template <trt_edgellm::marlin_dtypes::ScalarTypeId const a_type_id, // A ScalarType id
    trt_edgellm::marlin_dtypes::ScalarTypeId const b_type_id,       // B ScalarType id
    trt_edgellm::marlin_dtypes::ScalarTypeId const c_type_id,       // C ScalarType id
    trt_edgellm::marlin_dtypes::ScalarTypeId const s_type_id,       // B_SCALE ScalarType id
    int const threads,                                              // number of threads in a threadblock
    int const thread_m_blocks,                                      // number of 16x16 blocks in the m
                                                                    // dimension (batchsize) of the
                                                                    // threadblock
    int const thread_n_blocks,                                      // same for n dimension (output)
    int const thread_k_blocks,                                      // same for k dimension (reduction)
    bool const m_block_size_8,                                      // whether m_block_size == 8
                                                                    // only works when thread_m_blocks == 1
    int const stages,                                               // number of stages for the async global->shared
                                                                    // fetch pipeline
    int const group_blocks,                                         // number of consecutive 16x16 blocks
                                                                    // with a separate quantization scale
    bool const is_zp_float,                                         // is zero point of float16 type?
    bool const dense_identity                                       // skip MoE metadata and use identity row mapping
    >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
