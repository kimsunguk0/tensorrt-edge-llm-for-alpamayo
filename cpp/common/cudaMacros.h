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

#pragma once

#include <cuda.h>

/*!
 * @brief CUDA feature detection helpers
 *
 * Centralizes CUDA-version-based feature macros to avoid scattering `#if CUDA_VERSION ...`
 * checks throughout the codebase.
 *
 * Usage example:
 * @code
 * #include "common/cudaMacros.h"
 *
 * #if SUPPORTS_FP8
 * // Safe to reference __nv_fp8_e4m3 and related helpers.
 * #endif
 * @endcode
 *
 * @note When FP8 is supported (CUDA >= 11.8), this header also includes `<cuda_fp8.h>`,
 * so callers do NOT need to conditionally include it themselves.
 */

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#include <cuda_fp8.h>
#define SUPPORTS_FP8 1
#else
#define SUPPORTS_FP8 0
#endif
