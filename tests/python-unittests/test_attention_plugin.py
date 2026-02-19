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
TensorRT attention plugin tests vs numpy reference.

Tests custom attention plugin against numpy reference implementation.

Requirements:
    - Custom plugin library (libNvInfer_edgellm_plugin.so)
    - pycuda, numpy

Usage:
    python3 -m pytest tests/python-unittests/test_attention_plugin.py -v
    python3 -m unittest tests.python-unittests.test_attention_plugin.TestAttentionPluginVsNumpy
"""

import ctypes
import os
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


class AttentionPluginRunner:
    """TensorRT Attention Plugin runner (requires custom plugin library)."""

    def __init__(self,
                 params: utils.AttentionParams,
                 enable_tree_attention: bool = False):
        self.params = params
        self.enable_tree_attention = enable_tree_attention
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.plugin_lib = None

    def load_plugin_library(self):
        """Load the attention plugin library."""
        # Try to find the plugin library
        possible_paths = [
            "build/libNvInfer_edgellm_plugin.so",
        ]

        plugin_path = None
        for path in possible_paths:
            if os.path.exists(path):
                plugin_path = path
                break

        if plugin_path is None:
            raise RuntimeError(
                "Could not find plugin library. Please build the project first."
            )

        # Load the plugin library
        self.plugin_lib = ctypes.CDLL(plugin_path)

        # Initialize plugin registry
        trt.init_libnvinfer_plugins(self.logger, "")

    def build_engine(self):
        """Build TensorRT engine with attention plugin."""
        if self.plugin_lib is None:
            self.load_plugin_library()

        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     1 << 30)  # 1GB

        p = self.params

        # Calculate QKV hidden size
        qkv_hidden_size = (p.num_q_heads + p.num_kv_heads +
                           p.num_kv_heads) * p.head_size

        # Add inputs
        qkv_input = network.add_input("qkv", trt.float16,
                                      (-1, -1, qkv_hidden_size))
        # KV Cache: [B, 2, Hkv, capacity, D]
        kv_cache_input = network.add_input(
            "kv_cache", trt.float16,
            (-1, 2, p.num_kv_heads, p.kv_cache_capacity, p.head_size))
        context_lengths = network.add_input("context_lengths", trt.int32,
                                            (-1, ))
        # RoPE cos/sin cache: [1, max_pos_emb, D]
        rope_cos_sin = network.add_input(
            "rope_cos_sin", trt.float32,
            (1, p.max_position_embeddings, p.head_size))
        kv_cache_indices = network.add_input("kv_cache_indices", trt.int32,
                                             (-1, ))

        # Tree attention specific inputs (optional)
        if self.enable_tree_attention:
            tree_mask = network.add_input("tree_mask", trt.int32, (-1, -1, -1))
            position_ids = network.add_input("position_ids", trt.int32,
                                             (-1, -1))

        # Create plugin
        plugin_registry = trt.get_plugin_registry()
        plugin_creator = plugin_registry.get_plugin_creator(
            "AttentionPlugin", "1")

        if plugin_creator is None:
            raise RuntimeError("AttentionPlugin not found in registry")

        # Create plugin fields
        plugin_fields = [
            trt.PluginField("max_batch_size",
                            np.array([p.max_batch_size], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("kv_cache_capacity",
                            np.array([p.kv_cache_capacity], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("num_q_heads",
                            np.array([p.num_q_heads], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("num_kv_heads",
                            np.array([p.num_kv_heads], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("head_size", np.array([p.head_size],
                                                  dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField("enable_reuse_kv_cache",
                            np.array([1], dtype=np.int32),
                            trt.PluginFieldType.INT32),
            trt.PluginField(
                "enable_tree_attention",
                np.array([self.enable_tree_attention], dtype=np.int32),
                trt.PluginFieldType.INT32),
        ]

        plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
        plugin = plugin_creator.create_plugin("attention",
                                              plugin_field_collection)

        # Add plugin layer
        plugin_inputs = [
            qkv_input, kv_cache_input, context_lengths, rope_cos_sin,
            kv_cache_indices
        ]
        if self.enable_tree_attention:
            plugin_inputs.extend([tree_mask, position_ids])

        plugin_layer = network.add_plugin_v2(plugin_inputs, plugin)

        # Mark outputs
        plugin_layer.get_output(0).name = "attention_output"
        network.mark_output(plugin_layer.get_output(0))
        plugin_layer.get_output(1).name = "kv_cache_output"
        network.mark_output(plugin_layer.get_output(1))

        # Setup optimization profile
        profile = builder.create_optimization_profile()

        # QKV profile
        profile.set_shape("qkv", (1, 1, qkv_hidden_size),
                          (p.batch_size, p.seq_len, qkv_hidden_size),
                          (p.max_batch_size, p.max_seq_len, qkv_hidden_size))

        # KV cache profile
        profile.set_shape(
            "kv_cache",
            (1, 2, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            (p.batch_size, 2, p.num_kv_heads, p.kv_cache_capacity,
             p.head_size), (p.max_batch_size, 2, p.num_kv_heads,
                            p.kv_cache_capacity, p.head_size))

        # Context lengths profile
        profile.set_shape("context_lengths", (1, ), (p.batch_size, ),
                          (p.max_batch_size, ))

        # RoPE cos/sin profile
        profile.set_shape("rope_cos_sin",
                          (1, p.max_position_embeddings, p.head_size),
                          (1, p.max_position_embeddings, p.head_size),
                          (1, p.max_position_embeddings, p.head_size))

        # Cache indices profile
        profile.set_shape("kv_cache_indices", (1, ), (p.batch_size, ),
                          (p.max_batch_size, ))

        # Tree attention specific profiles
        if self.enable_tree_attention:
            # Attention mask profile: [B, S, S]
            profile.set_shape("tree_mask", (1, 1, 1),
                              (p.batch_size, p.seq_len, p.seq_len),
                              (p.max_batch_size, p.max_seq_len, p.max_seq_len))

            # Position IDs profile: [B, S]
            profile.set_shape("position_ids", (1, 1),
                              (p.batch_size, p.seq_len),
                              (p.max_batch_size, p.max_seq_len))

        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

    def execute(self,
                device_buffers: dict,
                qkv_shape: tuple,
                kv_cache_shape: tuple,
                context_lengths_shape: tuple,
                rope_cos_sin_shape: tuple,
                cache_indices_shape: tuple,
                mask_shape: tuple = None,
                position_ids_shape: tuple = None):
        """Execute the TensorRT engine with plugin."""
        # Set input shapes
        utils.check_trt(self.context.set_input_shape("qkv", qkv_shape))
        utils.check_trt(
            self.context.set_input_shape("kv_cache", kv_cache_shape))
        utils.check_trt(
            self.context.set_input_shape("context_lengths",
                                         context_lengths_shape))
        utils.check_trt(
            self.context.set_input_shape("rope_cos_sin", rope_cos_sin_shape))
        utils.check_trt(
            self.context.set_input_shape("kv_cache_indices",
                                         cache_indices_shape))

        # Tree attention specific shapes
        if self.enable_tree_attention:
            if mask_shape is None or position_ids_shape is None:
                raise ValueError(
                    "Tree attention requires mask_shape and position_ids_shape"
                )
            utils.check_trt(
                self.context.set_input_shape("tree_mask", mask_shape))
            utils.check_trt(
                self.context.set_input_shape("position_ids",
                                             position_ids_shape))

        # Set tensor addresses
        utils.check_trt(
            self.context.set_tensor_address("qkv",
                                            int(device_buffers['d_qkv'])))
        utils.check_trt(
            self.context.set_tensor_address("kv_cache",
                                            int(device_buffers['d_kv_cache'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "context_lengths", int(device_buffers['d_context_lengths'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "rope_cos_sin", int(device_buffers['d_rope_cos_sin'])))
        utils.check_trt(
            self.context.set_tensor_address(
                "kv_cache_indices", int(device_buffers['d_cache_indices'])))

        # Tree attention specific tensor addresses
        if self.enable_tree_attention:
            utils.check_trt(
                self.context.set_tensor_address(
                    "tree_mask", int(device_buffers['d_tree_mask'])))
            utils.check_trt(
                self.context.set_tensor_address(
                    "position_ids", int(device_buffers['d_position_ids'])))

        utils.check_trt(
            self.context.set_tensor_address(
                "attention_output", int(device_buffers['d_attn_output'])))
        # Plugin updates KV cache in-place
        utils.check_trt(
            self.context.set_tensor_address("kv_cache_output",
                                            int(device_buffers['d_kv_cache'])))

        # Execute
        utils.check_trt(self.context.execute_async_v3(self.stream.handle))


@pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"Required dependencies not available: {IMPORT_ERROR}")
class TestAttentionPluginVsNumpy:
    """Test TensorRT Attention Plugin against numpy reference."""

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

    def allocate_device_memory_for_plugin(
            self,
            params: utils.AttentionParams,
            enable_tree_attention: bool = False) -> dict:
        """Allocate device memory buffers for plugin execution.

        Args:
            params: Attention parameters
            enable_tree_attention: If True, allocate additional buffers for tree attention
        """
        p = params

        # Calculate buffer sizes for max shapes
        qkv_size = p.max_batch_size * p.max_seq_len * p.qkv_hidden_size * np.dtype(
            np.float16).itemsize
        attn_output_size = p.max_batch_size * p.max_seq_len * p.num_q_heads * p.head_size * np.dtype(
            np.float16).itemsize
        cache_indices_size = p.max_batch_size * np.dtype(np.int32).itemsize

        # Plugin uses combined KV cache [B, 2, Hkv, capacity, D]
        kv_cache_size = p.max_batch_size * 2 * p.num_kv_heads * p.kv_cache_capacity * p.head_size * np.dtype(
            np.float16).itemsize
        # Plugin uses combined RoPE cos/sin [1, max_pos_emb, D]
        rope_cos_sin_size = p.max_position_embeddings * p.head_size * np.dtype(
            np.float32).itemsize
        context_lengths_size = p.max_batch_size * np.dtype(np.int32).itemsize

        device_buffers = {
            'd_qkv': cuda.mem_alloc(qkv_size),
            'd_cache_indices': cuda.mem_alloc(cache_indices_size),
            'd_attn_output': cuda.mem_alloc(attn_output_size),
            'd_kv_cache': cuda.mem_alloc(kv_cache_size),
            'd_rope_cos_sin': cuda.mem_alloc(rope_cos_sin_size),
            'd_context_lengths': cuda.mem_alloc(context_lengths_size),
        }

        # Tree attention specific buffers
        if enable_tree_attention:
            # Tree mask: [max_batch, max_seq, max_seq]
            tree_mask_size = p.max_batch_size * p.max_seq_len * p.max_seq_len * np.dtype(
                np.int32).itemsize
            device_buffers['d_tree_mask'] = cuda.mem_alloc(tree_mask_size)

            # Position IDs: [max_batch, max_seq]
            position_ids_size = p.max_batch_size * p.max_seq_len * np.dtype(
                np.int32).itemsize
            device_buffers['d_position_ids'] = cuda.mem_alloc(
                position_ids_size)

        return device_buffers

    def free_device_memory(self, device_buffers: dict):
        """Free allocated device memory."""
        if device_buffers is not None:
            for buf in device_buffers.values():
                buf.free()

    def copy_plugin_inputs_to_device(
            self,
            device_buffers: dict,
            qkv: np.ndarray,
            kv_cache: np.ndarray,
            context_lengths: np.ndarray,
            rope_cos_sin: np.ndarray,
            cache_indices: np.ndarray,
            stream: cuda.Stream,
            attention_mask: Optional[np.ndarray] = None,
            position_ids: Optional[np.ndarray] = None):
        """Copy plugin input data from host to device.

        Args:
            attention_mask: Optional [B, S, S] packed tree attention mask
            position_ids: Optional [B, S] position IDs
        """
        cuda.memcpy_htod_async(device_buffers['d_qkv'], qkv.astype(np.float16),
                               stream)
        cuda.memcpy_htod_async(device_buffers['d_kv_cache'],
                               kv_cache.astype(np.float16), stream)
        cuda.memcpy_htod_async(device_buffers['d_context_lengths'],
                               context_lengths.astype(np.int32), stream)
        cuda.memcpy_htod_async(device_buffers['d_rope_cos_sin'],
                               rope_cos_sin.astype(np.float32), stream)
        cuda.memcpy_htod_async(device_buffers['d_cache_indices'],
                               cache_indices.astype(np.int32), stream)

        # Tree attention specific inputs
        if attention_mask is not None:
            cuda.memcpy_htod_async(device_buffers['d_tree_mask'],
                                   attention_mask.astype(np.int32), stream)
        if position_ids is not None:
            cuda.memcpy_htod_async(device_buffers['d_position_ids'],
                                   position_ids.astype(np.int32), stream)

    def copy_plugin_outputs_from_device(self, device_buffers: dict,
                                        attn_output: np.ndarray,
                                        kv_cache: np.ndarray,
                                        stream: cuda.Stream):
        """Copy plugin output data from device to host."""
        cuda.memcpy_dtoh_async(attn_output, device_buffers['d_attn_output'],
                               stream)
        cuda.memcpy_dtoh_async(kv_cache, device_buffers['d_kv_cache'], stream)

    def _run_plugin_attention_test(self,
                                   num_rounds: int,
                                   seq_len: int,
                                   batch_size: int,
                                   is_prefill: bool = False,
                                   request=None):
        """Run plugin attention test for specified number of rounds and sequence length.

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

        # Create plugin runner
        print(
            f"\nBuilding TensorRT plugin engine (batch_size={batch_size})...")
        try:
            plugin_runner = AttentionPluginRunner(p)
            plugin_runner.build_engine()
            print("✓ TensorRT plugin engine built successfully")
        except Exception as e:
            pytest.skip(f"Plugin not available: {e}")

        # Allocate device memory for plugin
        device_buffers = self.allocate_device_memory_for_plugin(p)
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

        # Plugin uses combined KV cache [B, 2, Hkv, capacity, D]
        plugin_kv_cache = np.zeros((p.batch_size, 2, p.num_kv_heads,
                                    p.kv_cache_capacity, p.head_size),
                                   dtype=np.float32)

        # Create RoPE caches - generate once and share between numpy and plugin
        # Numpy uses separate cos/sin caches [max_pos_emb, D/2]
        rope_cos_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)
        rope_sin_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)

        # Convert to plugin format: combined cos/sin cache [1, max_pos_emb, D]
        # Layout: [cos_0, cos_1, ..., cos_(D/2-1), sin_0, sin_1, ..., sin_(D/2-1)]
        # This matches the C++ test conversion in attentionPluginTests.cpp
        rope_cos_sin_cache = np.zeros(
            (1, p.max_position_embeddings, p.head_size), dtype=np.float32)
        rope_cos_sin_cache[0, :, :p.head_size // 2] = rope_cos_cache
        rope_cos_sin_cache[0, :, p.head_size // 2:] = rope_sin_cache

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

            # Plugin context_lengths semantics differ by execution mode:
            # - FMHA (prefill): used for cu_q_seqlens, should be current input length
            # - XQA (decode): used for kvCache.sequence_lengths, should be total context
            if is_prefill:
                context_lengths = np.full(p.batch_size,
                                          p.seq_len,
                                          dtype=np.int32)
            else:
                context_lengths = np.full(p.batch_size,
                                          current_pos + p.seq_len,
                                          dtype=np.int32)

            if is_prefill:
                # For prefill phase, create a causal mask
                causal_mask = utils.create_causal_mask(p.seq_len,
                                                       current_pos + p.seq_len)
            else:
                causal_mask = None

            # Run numpy reference
            np_attn_out, np_k_cache, np_v_cache = utils.NumpyAttentionReference.compute_attention(
                qkv, np_k_cache, np_v_cache, rope_cos_cache, rope_sin_cache,
                position_ids, cache_indices, p, causal_mask)

            # Run Plugin
            # Copy inputs to device
            self.copy_plugin_inputs_to_device(device_buffers, qkv,
                                              plugin_kv_cache, context_lengths,
                                              rope_cos_sin_cache,
                                              cache_indices,
                                              plugin_runner.stream)

            # Execute plugin
            plugin_runner.execute(device_buffers, qkv.shape,
                                  plugin_kv_cache.shape, context_lengths.shape,
                                  rope_cos_sin_cache.shape,
                                  cache_indices.shape)

            # Prepare output buffers
            plugin_attn_out = np.zeros(
                (p.batch_size, p.seq_len, p.num_q_heads * p.head_size),
                dtype=np.float16)
            plugin_kv_cache_out = np.zeros_like(plugin_kv_cache,
                                                dtype=np.float16)

            # Copy outputs from device
            self.copy_plugin_outputs_from_device(device_buffers,
                                                 plugin_attn_out,
                                                 plugin_kv_cache_out,
                                                 plugin_runner.stream)

            # Synchronize
            plugin_runner.stream.synchronize()

            # Update plugin cache for next round
            plugin_kv_cache = plugin_kv_cache_out.astype(np.float32)

            # Convert to FP32 for comparison
            plugin_attn_out_fp32 = plugin_attn_out.astype(np.float32)

            # Extract K and V from plugin's combined cache for comparison
            plugin_k_cache_fp32 = plugin_kv_cache_out[:, 0, :, :, :].astype(
                np.float32)
            plugin_v_cache_fp32 = plugin_kv_cache_out[:, 1, :, :, :].astype(
                np.float32)

            # Compare results
            attn_match = utils.compare_accuracy_and_report(
                "Attention", round_idx, np_attn_out, plugin_attn_out_fp32,
                atol, rtol)
            k_cache_match = utils.compare_accuracy_and_report(
                "K cache", round_idx, np_k_cache, plugin_k_cache_fp32, atol,
                rtol)
            v_cache_match = utils.compare_accuracy_and_report(
                "V cache", round_idx, np_v_cache, plugin_v_cache_fp32, atol,
                rtol)

            assert attn_match, f"Round {round_idx + 1}: Plugin attention outputs don't match numpy"
            assert k_cache_match, f"Round {round_idx + 1}: Plugin K cache outputs don't match numpy"
            assert v_cache_match, f"Round {round_idx + 1}: Plugin V cache outputs don't match numpy"

            print(f"  ✓ Round {round_idx + 1} passed")
            current_pos += p.seq_len

        if is_prefill:
            print(
                f"✓ Plugin prefill test passed ({num_rounds} rounds, batch_size={p.batch_size}, seq_len={p.seq_len})"
            )
        else:
            print(
                f"\n✓ Plugin decode test passed ({num_rounds} rounds, batch_size={p.batch_size}, seq_len={p.seq_len})"
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_plugin_vs_numpy_prefill(self, batch_size, request):
        """Test TensorRT Attention Plugin vs numpy for multiple rounds (prefill phase)."""
        self._run_plugin_attention_test(num_rounds=3,
                                        seq_len=self.params.max_seq_len,
                                        batch_size=batch_size,
                                        is_prefill=True,
                                        request=request)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_plugin_vs_numpy_decode(self, batch_size, request):
        """Test TensorRT Attention Plugin vs numpy for multiple rounds (decode phase)."""
        self._run_plugin_attention_test(num_rounds=5,
                                        seq_len=1,
                                        batch_size=batch_size,
                                        request=request)

    def test_plugin_vs_numpy_tree_attention(self, request):
        """Test TensorRT Attention Plugin vs numpy: tree attention."""
        p: utils.AttentionParams = replace(self.params, seq_len=4)
        num_rounds = 5

        # Create tree attention plugin runner
        print("Building TensorRT plugin engine with tree attention...")
        try:
            plugin_runner = AttentionPluginRunner(p,
                                                  enable_tree_attention=True)
            plugin_runner.build_engine()
            print("✓ TensorRT tree attention plugin engine built successfully")
        except Exception as e:
            pytest.skip(f"Tree attention plugin not available: {e}")

        # Allocate device memory (with tree attention support)
        device_buffers = self.allocate_device_memory_for_plugin(
            p, enable_tree_attention=True)
        request.addfinalizer(lambda: self.free_device_memory(device_buffers))

        # Initialize caches (zeros to match plugin)
        np_k_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)
        np_v_cache = np.zeros(
            (p.batch_size, p.num_kv_heads, p.kv_cache_capacity, p.head_size),
            dtype=np.float32)

        # Plugin uses combined KV cache [B, 2, Hkv, capacity, D]
        plugin_kv_cache = np.zeros((p.batch_size, 2, p.num_kv_heads,
                                    p.kv_cache_capacity, p.head_size),
                                   dtype=np.float32)

        # Create RoPE caches - generate once and share between numpy and plugin
        # Numpy uses separate cos/sin caches [max_pos_emb, D/2]
        rope_cos_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)
        rope_sin_cache = self.rng.standard_normal(
            (p.max_position_embeddings, p.head_size // 2)).astype(np.float32)

        # Convert to plugin format: combined cos/sin cache [1, max_pos_emb, D]
        # Layout: [cos_0, cos_1, ..., cos_(D/2-1), sin_0, sin_1, ..., sin_(D/2-1)]
        rope_cos_sin_cache = np.zeros(
            (1, p.max_position_embeddings, p.head_size), dtype=np.float32)
        rope_cos_sin_cache[0, :, :p.head_size // 2] = rope_cos_cache
        rope_cos_sin_cache[0, :, p.head_size // 2:] = rope_sin_cache

        tree_mask, accepted_indices = utils.get_tree_attention_mask(p.seq_len)
        packed_tree_mask = utils.pack_tree_mask(tree_mask, p.seq_len,
                                                p.batch_size)

        atol = 1e-2
        rtol = 1e-2
        current_pos = 0

        for round_idx in range(num_rounds):

            # Generate inputs
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

            # context_lengths = current_pos + seq_len (used by RoPE and XQA kernels)
            context_lengths = np.full(p.batch_size,
                                      current_pos + p.seq_len,
                                      dtype=np.int32)

            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

            # Construct full mask [seq_len, context_lengths[0]]: [all history | tree mask]
            full_mask = np.ones((p.seq_len, context_lengths[0]),
                                dtype=np.int32)
            full_mask[:, current_pos:] = tree_mask

            # Run numpy reference with tree attention mask
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

            # Run Plugin with full tree (for verification)
            self.copy_plugin_inputs_to_device(device_buffers,
                                              qkv,
                                              plugin_kv_cache,
                                              context_lengths,
                                              rope_cos_sin_cache,
                                              cache_indices,
                                              plugin_runner.stream,
                                              attention_mask=packed_tree_mask,
                                              position_ids=position_ids)

            plugin_runner.execute(device_buffers, qkv.shape,
                                  plugin_kv_cache.shape, context_lengths.shape,
                                  rope_cos_sin_cache.shape,
                                  cache_indices.shape, packed_tree_mask.shape,
                                  position_ids.shape)

            # Get outputs (full tree)
            plugin_attn_out = np.zeros(
                (p.batch_size, p.seq_len, p.num_q_heads * p.head_size),
                dtype=np.float16)

            plugin_kv_cache_out = np.zeros_like(plugin_kv_cache,
                                                dtype=np.float16)

            self.copy_plugin_outputs_from_device(device_buffers,
                                                 plugin_attn_out,
                                                 plugin_kv_cache_out,
                                                 plugin_runner.stream)

            plugin_runner.stream.synchronize()

            # Extract accepted tokens from attention outputs
            np_attn_out_accepted = np_attn_out[:, accepted_indices, :]
            plugin_attn_out_accepted = plugin_attn_out[:,
                                                       accepted_indices, :].astype(
                                                           np.float32)

            # Commit KV cache: rearrange to keep only accepted tokens (eagleBaseCommitKVCache)
            np_k_cache_committed, np_v_cache_committed = utils.commit_kv_cache(
                np_k_cache_out, np_v_cache_out, accepted_indices, current_pos,
                p.seq_len)

            plugin_k_cache_out, plugin_v_cache_out = plugin_kv_cache_out[:,
                                                                         0, :, :, :], plugin_kv_cache_out[:,
                                                                                                          1, :, :, :]
            plugin_k_cache_committed, plugin_v_cache_committed = utils.commit_kv_cache(
                plugin_k_cache_out, plugin_v_cache_out, accepted_indices,
                current_pos, p.seq_len)

            # Update working copies for next round
            np_k_cache = np_k_cache_committed.astype(np.float32)
            np_v_cache = np_v_cache_committed.astype(np.float32)
            plugin_k_cache = plugin_k_cache_committed.astype(np.float32)
            plugin_v_cache = plugin_v_cache_committed.astype(np.float32)
            plugin_kv_cache = np.concatenate([
                plugin_k_cache[:, np.newaxis, :, :, :],
                plugin_v_cache[:, np.newaxis, :, :, :]
            ],
                                             axis=1)

            # Compare all outputs (full coverage)
            plugin_attn_out_fp32 = plugin_attn_out.astype(np.float32)
            plugin_k_cache_out_fp32 = plugin_k_cache_out.astype(np.float32)
            plugin_v_cache_out_fp32 = plugin_v_cache_out.astype(np.float32)

            all_attn_match = utils.compare_accuracy_and_report(
                "Attention (all)", round_idx, np_attn_out,
                plugin_attn_out_fp32, atol, rtol)
            all_k_cache_match = utils.compare_accuracy_and_report(
                "K cache (all)", round_idx, np_k_cache_out,
                plugin_k_cache_out_fp32, atol, rtol)
            all_v_cache_match = utils.compare_accuracy_and_report(
                "V cache (all)", round_idx, np_v_cache_out,
                plugin_v_cache_out_fp32, atol, rtol)

            # Compare accepted outputs only
            attn_match = utils.compare_accuracy_and_report(
                "Attention (accepted)", round_idx, np_attn_out_accepted,
                plugin_attn_out_accepted, atol, rtol)
            k_cache_match = utils.compare_accuracy_and_report(
                "K cache (committed)", round_idx, np_k_cache, plugin_k_cache,
                atol, rtol)
            v_cache_match = utils.compare_accuracy_and_report(
                "V cache (committed)", round_idx, np_v_cache, plugin_v_cache,
                atol, rtol)

            assert all_attn_match, f"Round {round_idx + 1}: Plugin attention outputs (all) don't match numpy"
            assert all_k_cache_match, f"Round {round_idx + 1}: Plugin K cache outputs (all) don't match numpy"
            assert all_v_cache_match, f"Round {round_idx + 1}: Plugin V cache outputs (all) don't match numpy"
            assert attn_match, f"Round {round_idx + 1}: Plugin attention outputs (accepted) don't match numpy"
            assert k_cache_match, f"Round {round_idx + 1}: Plugin K cache outputs (committed) don't match numpy"
            assert v_cache_match, f"Round {round_idx + 1}: Plugin V cache outputs (committed) don't match numpy"

            print(
                f"  ✓ Round {round_idx + 1} passed (verified {p.seq_len} tokens, accepted {len(accepted_indices)})"
            )

            # Sync committed cache back to device
            cuda.memcpy_htod_async(device_buffers['d_kv_cache'],
                                   plugin_kv_cache.astype(np.float16),
                                   plugin_runner.stream)
            plugin_runner.stream.synchronize()
            # Update position only by accepted count (not full seq_len)
            current_pos += len(accepted_indices)

        print(
            f"\n✓ Plugin vs Numpy tree attention test passed ({num_rounds} decode rounds with batch_size={p.batch_size})"
        )
