# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TensorRT native attention tests vs numpy reference.

Tests TensorRT native attention layers (IKVCacheUpdateLayer, IAttentionLayer) against numpy.

Requirements:
    - TensorRT >= 10.15
    - pycuda, numpy

Usage:
    python3 -m pytest tests/python-unittests/test_attention_native.py -v
    python3 -m unittest tests.python-unittests.test_attention_native.TestTRTNativeAttentionVsNumpy
"""

from dataclasses import replace
from typing import Optional

import numpy as np
import pytest
import test_attention_utils as utils

# Conditional imports for GPU/TensorRT dependencies
try:
    import pycuda.autoinit  # Initialize CUDA context
    import pycuda.driver as cuda
    import tensorrt as trt
    DEPENDENCIES_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create dummy objects to prevent NameError during class definition
    class DummyModule:

        def __getattr__(self, name):
            return None

    cuda = DummyModule()
    trt = DummyModule()

if not DEPENDENCIES_AVAILABLE:
    _dependency_check_passed = False
    _dependency_check_message = f"DEPENDENCY not available: {IMPORT_ERROR}"
else:
    _dependency_check_passed, _dependency_check_message = utils.check_tensorrt_version(
        trt, 10, 15)


class TensorRTNativeRunner:
    """TensorRT native attention runner (requires TensorRT >= 10.15)."""

    def __init__(self,
                 params: utils.AttentionParams,
                 enable_tree_attention: bool = False):
        self.params = params
        self.enable_tree_attention = enable_tree_attention
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()

    def build_engine(self):
        """Build TensorRT engine with native attention layers."""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     1 << 30)  # 1GB

        p = self.params

        # Add inputs
        qkv_input = network.add_input("qkv", trt.float16,
                                      (-1, -1, p.qkv_hidden_size))
        k_cache_input = network.add_input(
            "k_cache", trt.float16,
            (-1, p.num_kv_heads, p.kv_cache_capacity, p.head_size))
        v_cache_input = network.add_input(
            "v_cache", trt.float16,
            (-1, p.num_kv_heads, p.kv_cache_capacity, p.head_size))
        rope_cos_cache = network.add_input(
            "rope_cos_cache", trt.float16,
            (p.max_position_embeddings, p.head_size // 2))
        rope_sin_cache = network.add_input(
            "rope_sin_cache", trt.float16,
            (p.max_position_embeddings, p.head_size // 2))
        position_ids = network.add_input("position_ids", trt.int32, (-1, -1))
        k_cache_indices = network.add_input("k_cache_indices", trt.int32,
                                            (-1, ))
        present_length_input = network.add_input("present_length", trt.int32,
                                                 (-1, ))
        # Mask input: needed for prefill (causal) or tree attention
        # Regular decode doesn't need mask
        need_mask = p.is_prefill or self.enable_tree_attention
        mask_input = network.add_input("mask", trt.bool,
                                       (1, 1, -1, -1)) if need_mask else None

        # Build network following C++ setupNetworkWithTRTAPI
        self._build_network(network, qkv_input, mask_input, k_cache_input,
                            v_cache_input, rope_cos_cache, rope_sin_cache,
                            position_ids, k_cache_indices,
                            present_length_input)

        # Setup optimization profile
        profile = builder.create_optimization_profile()

        # QKV profile
        profile.set_shape("qkv", (1, 1, p.qkv_hidden_size),
                          (p.batch_size, p.seq_len, p.qkv_hidden_size),
                          (p.max_batch_size, p.max_seq_len, p.qkv_hidden_size))

        # Mask profile: [1, 1, seq_len, present_length]
        if need_mask:
            profile.set_shape("mask", (1, 1, 1, 1),
                              (1, 1, p.seq_len, p.seq_len),
                              (1, 1, p.max_seq_len, p.kv_cache_capacity))

        # K/V cache profiles
        profile.set_shape(
            "k_cache", (1, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            (p.max_batch_size, p.num_kv_heads, p.kv_cache_capacity,
             p.head_size))
        profile.set_shape(
            "v_cache", (1, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            (p.max_batch_size, p.num_kv_heads, p.kv_cache_capacity,
             p.head_size))

        # RoPE cache profiles
        profile.set_shape("rope_cos_cache",
                          (p.max_position_embeddings, p.head_size // 2),
                          (p.max_position_embeddings, p.head_size // 2),
                          (p.max_position_embeddings, p.head_size // 2))
        profile.set_shape("rope_sin_cache",
                          (p.max_position_embeddings, p.head_size // 2),
                          (p.max_position_embeddings, p.head_size // 2),
                          (p.max_position_embeddings, p.head_size // 2))

        # Position IDs profile
        profile.set_shape("position_ids", (1, p.seq_len),
                          (p.batch_size, p.seq_len),
                          (p.max_batch_size, p.max_seq_len))

        # Cache indices profile
        profile.set_shape("k_cache_indices", (1, ), (p.batch_size, ),
                          (p.max_batch_size, ))

        # Present length profile
        profile.set_shape("present_length", (1, ), (p.kv_cache_capacity, ),
                          (p.kv_cache_capacity, ))

        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

    def _build_network(self, network, qkv_input, mask_input, k_cache_input,
                       v_cache_input, rope_cos_cache, rope_sin_cache,
                       position_ids, k_cache_indices, present_length_input):
        """Build the attention network following C++ implementation."""
        p = self.params

        # Slice Q, K, V from QKV input
        # Q: [batch, seq_len, num_q_heads * head_size]
        q_start = 0
        q_size = p.num_q_heads * p.head_size
        slice_q = network.add_slice(qkv_input,
                                    start=[q_start],
                                    shape=[q_size],
                                    stride=[1])
        slice_q.axes = [2]

        # K: [batch, seq_len, num_kv_heads * head_size]
        k_start = p.num_q_heads * p.head_size
        k_size = p.num_kv_heads * p.head_size
        slice_k = network.add_slice(qkv_input,
                                    start=[k_start],
                                    shape=[k_size],
                                    stride=[1])
        slice_k.axes = [2]

        # V: [batch, seq_len, num_kv_heads * head_size]
        v_start = k_start + k_size
        v_size = p.num_kv_heads * p.head_size
        slice_v = network.add_slice(qkv_input,
                                    start=[v_start],
                                    shape=[v_size],
                                    stride=[1])
        slice_v.axes = [2]

        # Reshape Q: [batch, seq_len, num_q_heads, head_size] -> [batch, num_q_heads, seq_len, head_size]
        reshape_q = network.add_shuffle(slice_q.get_output(0))
        reshape_q.reshape_dims = (0, 0, p.num_q_heads, p.head_size)
        reshape_q.second_transpose = (0, 2, 1, 3)

        # Reshape K: [batch, seq_len, num_kv_heads, head_size] -> [batch, num_kv_heads, seq_len, head_size]
        reshape_k = network.add_shuffle(slice_k.get_output(0))
        reshape_k.reshape_dims = (0, 0, p.num_kv_heads, p.head_size)
        reshape_k.second_transpose = (0, 2, 1, 3)

        # Reshape V: [batch, seq_len, num_kv_heads, head_size] -> [batch, num_kv_heads, seq_len, head_size]
        reshape_v = network.add_shuffle(slice_v.get_output(0))
        reshape_v.reshape_dims = (0, 0, p.num_kv_heads, p.head_size)
        reshape_v.second_transpose = (0, 2, 1, 3)

        q_tensor = reshape_q.get_output(0)
        k_tensor = reshape_k.get_output(0)
        v_tensor = reshape_v.get_output(0)

        # Apply RoPE to Q and K
        rope_q = network.add_rotary_embedding(
            q_tensor, rope_cos_cache, rope_sin_cache, False,
            0)  # interleaved=False, rotary_ndims=0 (use all)
        rope_q.set_input(3, position_ids)

        rope_k = network.add_rotary_embedding(k_tensor, rope_cos_cache,
                                              rope_sin_cache, False, 0)
        rope_k.set_input(3, position_ids)

        rope_q_tensor = rope_q.get_output(0)
        rope_k_tensor = rope_k.get_output(0)

        # Update KV cache
        kv_cache_mode = trt.KVCacheMode.LINEAR
        k_cache_layer = network.add_kv_cache_update(k_cache_input,
                                                    rope_k_tensor,
                                                    k_cache_indices,
                                                    kv_cache_mode)
        v_cache_layer = network.add_kv_cache_update(v_cache_input, v_tensor,
                                                    k_cache_indices,
                                                    kv_cache_mode)

        k_cache_output = k_cache_layer.get_output(0)
        v_cache_output = v_cache_layer.get_output(0)

        # Mark cache outputs
        k_cache_output.name = "k_cache_output"
        network.mark_output(k_cache_output)
        v_cache_output.name = "v_cache_output"
        network.mark_output(v_cache_output)

        # Create slice layers to extract present KV from cache
        # Slice along axis 2 (sequence dimension)
        # Start: 0, Size: present_length, Stride: 1
        present_length = network.add_shape(present_length_input)
        present_length_tensor = present_length.get_output(0)

        # Create constant tensors for slice start and stride
        zero_const = network.add_constant((1, ), np.array([0], dtype=np.int32))
        one_const = network.add_constant((1, ), np.array([1], dtype=np.int32))

        k_present_slice = network.add_slice(
            k_cache_output,
            start=(0, 0, 0, 0),
            shape=(0, 0, 1, 0),  # Will be set dynamically
            stride=(1, 1, 1, 1))
        # Slice along axis 2 (sequence dimension in cache: [batch, num_heads, seq, head_size])
        k_present_slice.set_input(1, zero_const.get_output(0))
        k_present_slice.set_input(2, present_length_tensor)
        k_present_slice.set_input(3, one_const.get_output(0))
        k_present_slice.axes = [2]

        v_present_slice = network.add_slice(v_cache_output,
                                            start=(0, 0, 0, 0),
                                            shape=(0, 0, 1, 0),
                                            stride=(1, 1, 1, 1))
        v_present_slice.set_input(1, zero_const.get_output(0))
        v_present_slice.set_input(2, present_length_tensor)
        v_present_slice.set_input(3, one_const.get_output(0))
        v_present_slice.axes = [2]

        k_present = k_present_slice.get_output(0)
        v_present = v_present_slice.get_output(0)

        # Apply Q/K scale using elementwise multiplication
        # Scale Q by sqrt(head_size) to get the same effect as scaling QK^T by scale
        scale_const = network.add_constant((1, 1, 1, 1),
                                           np.array([p.qk_scale],
                                                    dtype=np.float16))
        scaled_q = network.add_elementwise(rope_q_tensor,
                                           scale_const.get_output(0),
                                           trt.ElementWiseOperation.PROD)
        scaled_q_tensor = scaled_q.get_output(0)

        # Add attention layer
        attention = network.add_attention(scaled_q_tensor, k_present,
                                          v_present,
                                          trt.AttentionNormalizationOp.SOFTMAX,
                                          False)
        attention.decomposable = True
        # Mask is used for prefill (causal) or tree attention
        # For regular decode, no mask is needed
        if mask_input is not None:
            attention.mask = mask_input

        # Reshape attention output: [batch, num_heads, seq_len, head_size] -> [batch, seq_len, num_heads*head_size]
        output_transpose = network.add_shuffle(attention.get_output(0))
        output_transpose.first_transpose = (0, 2, 1, 3)
        output_transpose.reshape_dims = (0, 0, p.num_q_heads * p.head_size)

        attention_output = output_transpose.get_output(0)
        attention_output.name = "attention_output"
        network.mark_output(attention_output)

    def execute(self,
                device_buffers: dict,
                qkv_shape: tuple,
                k_cache_shape: tuple,
                v_cache_shape: tuple,
                rope_cos_shape: tuple,
                rope_sin_shape: tuple,
                position_ids_shape: tuple,
                cache_indices_shape: tuple,
                present_length: int,
                mask_shape: tuple = None):
        """Execute the TensorRT engine. Memory must be already copied to device buffers."""
        # Set input shapes
        utils.check_trt(self.context.set_input_shape("qkv", qkv_shape))
        utils.check_trt(self.context.set_input_shape("k_cache", k_cache_shape))
        utils.check_trt(self.context.set_input_shape("v_cache", v_cache_shape))
        utils.check_trt(
            self.context.set_input_shape("rope_cos_cache", rope_cos_shape))
        utils.check_trt(
            self.context.set_input_shape("rope_sin_cache", rope_sin_shape))
        utils.check_trt(
            self.context.set_input_shape("position_ids", position_ids_shape))
        utils.check_trt(
            self.context.set_input_shape("k_cache_indices",
                                         cache_indices_shape))
        utils.check_trt(
            self.context.set_input_shape("present_length", (present_length, )))

        # Set tensor addresses
        utils.check_trt(
            self.context.set_tensor_address("qkv",
                                            int(device_buffers['d_qkv'])))
        utils.check_trt(
            self.context.set_tensor_address("k_cache",
                                            int(device_buffers['d_k_cache'])))
        utils.check_trt(
            self.context.set_tensor_address("v_cache",
                                            int(device_buffers['d_v_cache'])))
        utils.check_trt(
            self.context.set_tensor_address("rope_cos_cache",
                                            int(device_buffers['d_rope_cos'])))
        utils.check_trt(
            self.context.set_tensor_address("rope_sin_cache",
                                            int(device_buffers['d_rope_sin'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "position_ids", int(device_buffers['d_position_ids'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "k_cache_indices", int(device_buffers['d_cache_indices'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "present_length", int(device_buffers['d_present_length'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "attention_output", int(device_buffers['d_attn_output'])))
        utils.check_trt(
            self.context.set_tensor_address("k_cache_output",
                                            int(device_buffers['d_k_cache'])))
        utils.check_trt(
            self.context.set_tensor_address("v_cache_output",
                                            int(device_buffers['d_v_cache'])))

        # Mask is only needed for prefill (causal) or tree attention
        if mask_shape is not None:
            utils.check_trt(self.context.set_input_shape("mask", mask_shape))
            utils.check_trt(
                self.context.set_tensor_address("mask",
                                                int(device_buffers['d_mask'])))

        # Execute
        utils.check_trt(self.context.execute_async_v3(self.stream.handle))


@pytest.mark.skipif(not _dependency_check_passed,
                    reason=_dependency_check_message)
class TestTRTNativeAttentionVsNumpy:
    """Test TensorRT native attention against numpy reference."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test parameters and random seed."""
        self.rng = np.random.default_rng(42)
        self.params = utils.AttentionParams(
            batch_size=4,
            seq_len=1,
            num_q_heads=8,
            num_kv_heads=8,
            head_size=128,
            kv_cache_capacity=64,
            max_batch_size=8,
            max_seq_len=8,
            max_position_embeddings=64,
        )

    def allocate_device_memory(self, params: utils.AttentionParams) -> dict:
        """Allocate device memory buffers for TensorRT execution."""
        p = params

        # Calculate buffer sizes for max shapes
        qkv_size = p.max_batch_size * p.max_seq_len * p.qkv_hidden_size * np.dtype(
            np.float16).itemsize
        # Mask shape: [1, 1, max_seq_len, kv_cache_capacity]
        mask_size = 1 * 1 * p.max_seq_len * p.kv_cache_capacity * np.dtype(
            np.bool_).itemsize
        kv_cache_size = p.max_batch_size * p.num_kv_heads * p.kv_cache_capacity * p.head_size * np.dtype(
            np.float16).itemsize
        rope_cache_size = p.max_position_embeddings * (
            p.head_size // 2) * np.dtype(np.float16).itemsize
        position_ids_size = p.max_batch_size * p.max_seq_len * np.dtype(
            np.int32).itemsize
        cache_indices_size = p.max_batch_size * np.dtype(np.int32).itemsize
        present_length_size = np.dtype(np.int32).itemsize
        attn_output_size = p.max_batch_size * p.max_seq_len * p.num_q_heads * p.head_size * np.dtype(
            np.float16).itemsize

        device_buffers = {
            'd_qkv': cuda.mem_alloc(qkv_size),
            'd_mask': cuda.mem_alloc(mask_size),
            'd_k_cache': cuda.mem_alloc(kv_cache_size),
            'd_v_cache': cuda.mem_alloc(kv_cache_size),
            'd_rope_cos': cuda.mem_alloc(rope_cache_size),
            'd_rope_sin': cuda.mem_alloc(rope_cache_size),
            'd_position_ids': cuda.mem_alloc(position_ids_size),
            'd_cache_indices': cuda.mem_alloc(cache_indices_size),
            'd_present_length': cuda.mem_alloc(present_length_size),
            'd_attn_output': cuda.mem_alloc(attn_output_size),
        }

        return device_buffers

    def free_device_memory(self, device_buffers: dict):
        """Free allocated device memory."""
        if device_buffers is not None:
            for buf in device_buffers.values():
                buf.free()

    def copy_static_inputs_to_device(self, device_buffers: dict,
                                     rope_cos: np.ndarray,
                                     rope_sin: np.ndarray,
                                     stream: cuda.Stream):
        """Copy static input data (RoPE caches) from host to device. Called once before loop."""
        cuda.memcpy_htod_async(device_buffers['d_rope_cos'],
                               rope_cos.astype(np.float16), stream)
        cuda.memcpy_htod_async(device_buffers['d_rope_sin'],
                               rope_sin.astype(np.float16), stream)

    def copy_per_round_inputs_to_device(self,
                                        device_buffers: dict,
                                        qkv: np.ndarray,
                                        position_ids: np.ndarray,
                                        cache_indices: np.ndarray,
                                        stream: cuda.Stream,
                                        mask: Optional[np.ndarray] = None):
        """Copy per-round input data from host to device. Called each iteration."""
        cuda.memcpy_htod_async(device_buffers['d_qkv'], qkv.astype(np.float16),
                               stream)
        cuda.memcpy_htod_async(device_buffers['d_position_ids'],
                               position_ids.astype(np.int32), stream)
        cuda.memcpy_htod_async(device_buffers['d_cache_indices'],
                               cache_indices.astype(np.int32), stream)

        if mask is not None:
            cuda.memcpy_htod_async(device_buffers['d_mask'],
                                   mask.astype(np.bool_), stream)

    def copy_caches_to_device(self, device_buffers: dict, k_cache: np.ndarray,
                              v_cache: np.ndarray, stream: cuda.Stream):
        """Copy KV cache data from host to device."""
        cuda.memcpy_htod_async(device_buffers['d_k_cache'],
                               k_cache.astype(np.float16), stream)
        cuda.memcpy_htod_async(device_buffers['d_v_cache'],
                               v_cache.astype(np.float16), stream)

    def copy_outputs_from_device(self, device_buffers: dict,
                                 attn_output: np.ndarray, k_cache: np.ndarray,
                                 v_cache: np.ndarray, stream: cuda.Stream):
        """Copy output data from device to host."""
        cuda.memcpy_dtoh_async(attn_output, device_buffers['d_attn_output'],
                               stream)
        cuda.memcpy_dtoh_async(k_cache, device_buffers['d_k_cache'], stream)
        cuda.memcpy_dtoh_async(v_cache, device_buffers['d_v_cache'], stream)

    def _run_trt_native_attention_test(self,
                                       num_rounds: int,
                                       seq_len: int,
                                       batch_size: int,
                                       is_prefill: bool = False,
                                       request=None):
        """Run TRT native attention test for specified number of rounds and sequence length.

        Args:
            num_rounds: Number of rounds to run
            seq_len: Sequence length per round
            batch_size: Batch size for this test
            is_prefill: Whether the test is for prefill phase
        """
        p: utils.AttentionParams = replace(self.params,
                                           seq_len=seq_len,
                                           batch_size=batch_size,
                                           is_prefill=is_prefill)

        # Create TensorRT runner
        print(f"\nBuilding TensorRT engine (batch_size={batch_size})...")
        trt_runner = TensorRTNativeRunner(p)
        trt_runner.build_engine()
        print("✓ TensorRT engine built successfully")

        # Allocate device memory
        device_buffers = self.allocate_device_memory(p)
        if request is not None:
            request.addfinalizer(
                lambda: self.free_device_memory(device_buffers))

        # Initialize caches
        np_k_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        np_v_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        trt_k_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        trt_v_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)

        # Shared RoPE caches
        rope_cos_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)
        rope_sin_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)

        # Copy static inputs (RoPE caches) to device once before the loop
        self.copy_static_inputs_to_device(device_buffers, rope_cos_cache,
                                          rope_sin_cache, trt_runner.stream)

        # Copy initial KV caches to device once before the loop
        self.copy_caches_to_device(device_buffers, trt_k_cache, trt_v_cache,
                                   trt_runner.stream)
        trt_runner.stream.synchronize()

        atol = 1e-2
        rtol = 1e-2
        current_pos = 0

        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

            # Create input for this round
            qkv = self.rng.standard_normal(
                (p.batch_size, p.seq_len,
                 p.qkv_hidden_size)).astype(np.float32)
            position_ids = np.arange(current_pos,
                                     current_pos + p.seq_len,
                                     dtype=np.int32)[None, :].repeat(
                                         p.batch_size, axis=0)
            cache_indices = np.full(p.batch_size, current_pos, dtype=np.int32)
            present_length = current_pos + p.seq_len

            if is_prefill:
                causal_mask = utils.create_causal_mask(p.seq_len,
                                                       present_length)
                causal_mask_4d = causal_mask[None, None, :, :]
            else:
                causal_mask = None
                causal_mask_4d = None

            # Run numpy reference
            np_attn_out, np_k_cache, np_v_cache = utils.NumpyAttentionReference.compute_attention(
                qkv, np_k_cache, np_v_cache, rope_cos_cache, rope_sin_cache,
                position_ids, cache_indices, p, causal_mask)

            # Run TensorRT
            # Copy only per-round inputs to device (qkv, position_ids, cache_indices, causal_mask)
            # KV caches and RoPE caches are already on device and reused
            self.copy_per_round_inputs_to_device(device_buffers, qkv,
                                                 position_ids, cache_indices,
                                                 trt_runner.stream,
                                                 causal_mask_4d)

            # Execute TensorRT
            trt_runner.execute(
                device_buffers, qkv.shape, trt_k_cache.shape,
                trt_v_cache.shape, rope_cos_cache.shape, rope_sin_cache.shape,
                position_ids.shape, cache_indices.shape, present_length,
                causal_mask_4d.shape if causal_mask_4d is not None else None)

            # Prepare output buffers
            trt_attn_out = np.zeros(
                (p.batch_size, p.seq_len, p.num_q_heads * p.head_size),
                dtype=np.float16)
            trt_k_cache_out = np.zeros_like(trt_k_cache, dtype=np.float16)
            trt_v_cache_out = np.zeros_like(trt_v_cache, dtype=np.float16)

            # Copy outputs from device
            self.copy_outputs_from_device(device_buffers, trt_attn_out,
                                          trt_k_cache_out, trt_v_cache_out,
                                          trt_runner.stream)

            # Synchronize
            trt_runner.stream.synchronize()

            # Convert to FP32 for comparison
            trt_attn_out_fp32 = trt_attn_out.astype(np.float32)
            trt_k_cache = trt_k_cache_out.astype(np.float32)
            trt_v_cache = trt_v_cache_out.astype(np.float32)

            # Compare results
            attn_match = utils.compare_accuracy_and_report(
                "Attention", round_idx, np_attn_out, trt_attn_out_fp32, atol,
                rtol)
            k_cache_match = utils.compare_accuracy_and_report(
                "K cache", round_idx, np_k_cache, trt_k_cache, atol, rtol)
            v_cache_match = utils.compare_accuracy_and_report(
                "V cache", round_idx, np_v_cache, trt_v_cache, atol, rtol)

            assert attn_match, f"Round {round_idx + 1}: Attention outputs don't match"
            assert k_cache_match, f"Round {round_idx + 1}: K cache outputs don't match"
            assert v_cache_match, f"Round {round_idx + 1}: V cache outputs don't match"

            print(f"  ✓ Round {round_idx + 1} passed")

            current_pos += p.seq_len

        if is_prefill:
            print(
                f"✓ TRT native prefill test passed ({num_rounds} rounds, batch_size={p.batch_size}, seq_len={p.seq_len})"
            )
        else:
            print(
                f"\n✓ TRT native decode test passed ({num_rounds} rounds, batch_size={p.batch_size}, seq_len={p.seq_len})"
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_tensorrt_vs_numpy_prefill(self, batch_size, request):
        """Test TensorRT native vs numpy for multiple rounds (prefill phase)."""
        self._run_trt_native_attention_test(num_rounds=3,
                                            seq_len=self.params.max_seq_len,
                                            batch_size=batch_size,
                                            is_prefill=True,
                                            request=request)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_tensorrt_vs_numpy_decode(self, batch_size, request):
        """Test TensorRT native vs numpy for multiple rounds (decode phase)."""
        self._run_trt_native_attention_test(num_rounds=5,
                                            seq_len=1,
                                            batch_size=batch_size,
                                            request=request)

    def test_tensorrt_vs_numpy_tree_attention(self, request):
        """Test TensorRT native vs numpy: tree attention."""
        p: utils.AttentionParams = replace(self.params, seq_len=4)
        num_rounds = 5

        # Create TensorRT runner
        print("Building TensorRT engine...")
        trt_runner = TensorRTNativeRunner(p, enable_tree_attention=True)
        trt_runner.build_engine()
        print("✓ TensorRT engine built successfully")

        # Allocate device memory
        device_buffers = self.allocate_device_memory(p)
        request.addfinalizer(lambda: self.free_device_memory(device_buffers))

        # Initialize caches
        np_k_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        np_v_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        trt_k_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        trt_v_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)

        # Shared RoPE caches
        rope_cos_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)
        rope_sin_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)

        tree_mask, accepted_indices = utils.get_tree_attention_mask(p.seq_len)

        # Copy static inputs (RoPE caches) to device once before the loop
        self.copy_static_inputs_to_device(device_buffers, rope_cos_cache,
                                          rope_sin_cache, trt_runner.stream)

        # Copy initial KV caches to device once before the loop
        self.copy_caches_to_device(device_buffers, trt_k_cache, trt_v_cache,
                                   trt_runner.stream)
        trt_runner.stream.synchronize()

        # Relaxed tolerance for tree attention
        atol = 1.5e-2
        rtol = 1e-2
        current_pos = 0

        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

            # Create input for this round
            qkv = self.rng.standard_normal(
                (p.batch_size, p.seq_len,
                 p.qkv_hidden_size)).astype(np.float32)

            # Construct position_ids based on tree structure
            # Base position depths: [0, 1, 1, 2] (relative depth in tree)
            base_pos_depth = np.array([0, 1, 1, 2], dtype=np.int32)
            if p.seq_len <= 4:
                pos_depth = base_pos_depth[:p.seq_len]
            else:
                # Extend: token 4+ are children of 0 (depth 1)
                pos_depth = np.concatenate(
                    [base_pos_depth,
                     np.ones(p.seq_len - 4, dtype=np.int32)])
            position_ids = (current_pos + pos_depth)[np.newaxis, :].repeat(
                p.batch_size, axis=0)

            cache_indices = np.full(p.batch_size, current_pos, dtype=np.int32)
            present_length = current_pos + p.seq_len

            # Construct full mask [seq_len, present_length]: [all history | tree mask]
            full_mask = np.ones((p.seq_len, present_length), dtype=np.int32)
            full_mask[:, current_pos:] = tree_mask
            full_mask_4d = full_mask[None, None, :, :]

            # Run numpy reference
            np_attn_out, np_k_cache_out, np_v_cache_out = utils.NumpyAttentionReference.compute_attention(
                qkv,
                np_k_cache,
                np_v_cache,
                rope_cos_cache,
                rope_sin_cache,
                position_ids,
                cache_indices,
                p,
                attention_mask=full_mask)

            # Copy inputs to device
            self.copy_per_round_inputs_to_device(device_buffers, qkv,
                                                 position_ids, cache_indices,
                                                 trt_runner.stream,
                                                 full_mask_4d)

            trt_runner.execute(device_buffers, qkv.shape, trt_k_cache.shape,
                               trt_v_cache.shape, rope_cos_cache.shape,
                               rope_sin_cache.shape, position_ids.shape,
                               cache_indices.shape, present_length,
                               full_mask_4d.shape)

            # Download results from device
            trt_attn_out = np.zeros(
                (p.batch_size, p.seq_len, p.num_q_heads * p.head_size),
                dtype=np.float16)
            trt_k_cache_out = np.zeros_like(trt_k_cache, dtype=np.float16)
            trt_v_cache_out = np.zeros_like(trt_v_cache, dtype=np.float16)

            self.copy_outputs_from_device(device_buffers, trt_attn_out,
                                          trt_k_cache_out, trt_v_cache_out,
                                          trt_runner.stream)
            trt_runner.stream.synchronize()

            # Extract accepted tokens from attention outputs
            np_attn_out_accepted = np_attn_out[:, accepted_indices, :]
            trt_attn_out_accepted = trt_attn_out[:,
                                                 accepted_indices, :].astype(
                                                     np.float32)

            np_k_cache_committed, np_v_cache_committed = utils.commit_kv_cache(
                np_k_cache_out, np_v_cache_out, accepted_indices, current_pos,
                p.seq_len)

            trt_k_cache_committed, trt_v_cache_committed = utils.commit_kv_cache(
                trt_k_cache_out, trt_v_cache_out, accepted_indices,
                current_pos, p.seq_len)

            # Update working copies for next round
            np_k_cache = np_k_cache_committed.astype(np.float32)
            np_v_cache = np_v_cache_committed.astype(np.float32)
            trt_k_cache = trt_k_cache_committed.astype(np.float32)
            trt_v_cache = trt_v_cache_committed.astype(np.float32)

            # Compare all outputs (full coverage)
            trt_attn_out_fp32 = trt_attn_out.astype(np.float32)
            trt_k_cache_out_fp32 = trt_k_cache_out.astype(np.float32)
            trt_v_cache_out_fp32 = trt_v_cache_out.astype(np.float32)

            all_attn_match = utils.compare_accuracy_and_report(
                "Attention (all)", round_idx, np_attn_out, trt_attn_out_fp32,
                atol, rtol)
            all_k_cache_match = utils.compare_accuracy_and_report(
                "K cache (all)", round_idx, np_k_cache_out,
                trt_k_cache_out_fp32, atol, rtol)
            all_v_cache_match = utils.compare_accuracy_and_report(
                "V cache (all)", round_idx, np_v_cache_out,
                trt_v_cache_out_fp32, atol, rtol)

            # Compare accepted outputs only
            attn_match = utils.compare_accuracy_and_report(
                "Attention (accepted)", round_idx, np_attn_out_accepted,
                trt_attn_out_accepted, atol, rtol)
            k_cache_match = utils.compare_accuracy_and_report(
                "K cache (committed)", round_idx, np_k_cache, trt_k_cache,
                atol, rtol)
            v_cache_match = utils.compare_accuracy_and_report(
                "V cache (committed)", round_idx, np_v_cache, trt_v_cache,
                atol, rtol)

            assert all_attn_match, f"Round {round_idx + 1}: Attention outputs (all) don't match"
            assert all_k_cache_match, f"Round {round_idx + 1}: K cache outputs (all) don't match"
            assert all_v_cache_match, f"Round {round_idx + 1}: V cache outputs (all) don't match"
            assert attn_match, f"Round {round_idx + 1}: Attention outputs (accepted) don't match"
            assert k_cache_match, f"Round {round_idx + 1}: K cache outputs (committed) don't match"
            assert v_cache_match, f"Round {round_idx + 1}: V cache outputs (committed) don't match"

            print(
                f"  ✓ Round {round_idx + 1} passed (verified {p.seq_len} tokens, accepted {len(accepted_indices)})"
            )

            # Sync committed cache back to device
            self.copy_caches_to_device(device_buffers, trt_k_cache,
                                       trt_v_cache, trt_runner.stream)
            trt_runner.stream.synchronize()

            # Update sequence position (commitSequenceLength)
            current_pos += len(accepted_indices)

        print(
            f"\n✓ TensorRT vs Numpy tree attention test passed ({num_rounds} decode rounds with batch_size={p.batch_size})"
        )
