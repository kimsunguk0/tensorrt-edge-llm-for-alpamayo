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
Common utilities and reference implementations for attention tests.

Contains:
    - Numpy reference implementation of attention with RoPE and KV caching
    - Helper functions for mask creation and cache management
    - Shared test parameters and utilities
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def check_trt(return_value: bool) -> bool:
    """Check TensorRT API call return value and raise exception if failed."""
    if not return_value:
        raise Exception("TensorRT API call failed")
    return return_value


# Check TensorRT version requirement
def check_tensorrt_version(trt, required_major, required_minor):
    """Check if TensorRT version meets the minimum requirement of 10.15"""
    try:
        trt_version = trt.__version__
        # Parse version string (e.g., "10.15.0.27" or "10.15.0")
        version_parts = trt_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        if major < required_major or (major == required_major
                                      and minor < required_minor):
            return False, f"TensorRT {trt_version} found, but >= 10.15 is required"
        return True, f"TensorRT {trt_version}"
    except Exception as e:
        return False, f"Failed to check TensorRT version: {e}"


def get_tree_attention_mask(seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the tree attention mask for the given sequence length.
    Fixed tree attention mask for all rounds
    Tree structure (for seq_len >= 4):
         0
       / | \
      1  2 ... seq_len - 1
         |
         3
    Base 4x4 tree mask
    Returns:
        tree_mask: Tree attention mask [seq_len, seq_len]
        accepted_indices: Accepted indices [batch]
    """
    base_tree_mask = np.array(
        [
            [1, 0, 0, 0],  # Token 0: can see self
            [1, 1, 0, 0],  # Token 1: can see 0 and self
            [1, 0, 1, 0],  # Token 2: can see 0 and self
            [1, 0, 1, 1],  # Token 3: can see 0, 2 and self
        ],
        dtype=np.int32)

    if seq_len <= 4:
        # Slice base mask for smaller seq_len
        tree_mask = base_tree_mask[:seq_len, :seq_len].copy()
    else:
        # Extend for larger seq_len: Token 4+ connect to 0 and self
        tree_mask = np.zeros((seq_len, seq_len), dtype=np.int32)
        tree_mask[:4, :4] = base_tree_mask
        for i in range(4, seq_len):
            tree_mask[i, 0] = tree_mask[i, i] = 1

    # Accept only token 0 and 2
    accepted_indices = np.array([0], dtype=np.int32)
    if seq_len > 2:
        accepted_indices = np.concatenate(
            [accepted_indices, np.array([2], dtype=np.int32)])
    return tree_mask, accepted_indices


def pack_tree_mask(mask: np.ndarray, seq_len: int,
                   batch_size: int) -> np.ndarray:
    """Pack tree attention mask to bit-packed format for Plugin (XQA kernel).

    Plugin requires bit-packed mask [B, S, ceil(S/32)] where each row's
    mask values are compressed into 32-bit integers. This allows XQA kernel
    to check mask bits via fast bitwise AND instead of memory reads.
    See kernelSrcs/xqa/mha.cu for implementation details.
    TRT native uses TensorRT's built-in FMHA with unpacked [B, S, S] mask directly.
    """
    num_packed_per_row = (seq_len + 31) // 32
    packed = np.zeros((seq_len, num_packed_per_row), dtype=np.int32)
    for row in range(seq_len):
        packed_val = 0
        for col in range(seq_len):
            if mask[row, col] == 1:
                packed_val |= (1 << col)
        packed[row, 0] = packed_val
    return np.broadcast_to(packed[None, :, :].astype(np.int32),
                           (batch_size, *packed.shape)).copy()


def commit_kv_cache(k_cache: np.ndarray, v_cache: np.ndarray,
                    accepted_indices: np.ndarray, current_pos: int,
                    seq_len: int):
    """Rearrange KV cache to keep only accepted tokens (for tree attention).
    """
    src_pos = current_pos + accepted_indices
    dst_pos = np.arange(current_pos, current_pos + len(accepted_indices))

    k_cache_committed = k_cache.copy()
    v_cache_committed = v_cache.copy()
    # Clear rejected tokens region
    k_cache_committed[:, :, current_pos:current_pos + seq_len, :] = 0
    v_cache_committed[:, :, current_pos:current_pos + seq_len, :] = 0
    # Write only accepted tokens
    k_cache_committed[:, :, dst_pos, :] = k_cache[:, :, src_pos, :]
    v_cache_committed[:, :, dst_pos, :] = v_cache[:, :, src_pos, :]

    return k_cache_committed, v_cache_committed


def create_causal_mask(seq_q: int, seq_k: int) -> np.ndarray:
    """Create lower-right aligned causal mask.

    Args:
        seq_q: Query sequence length
        seq_k: Key sequence length (seq_k >= seq_q)

    Returns:
        Causal mask [seq_q, seq_k] with 1 where attention is allowed, 0 otherwise.
        Token at position i can attend to positions [0, offset + i] where offset = seq_k - seq_q.
    """
    row_indices = np.arange(seq_q)[:, None]  # [seq_q, 1]
    col_indices = np.arange(seq_k)[None, :]  # [1, seq_k]
    offset = seq_k - seq_q
    # Allow attention where col <= row + offset
    mask = (col_indices <= row_indices + offset).astype(np.int32)
    return mask


def compare_accuracy_and_report(name: str, round_idx: int,
                                expected: np.ndarray, actual: np.ndarray,
                                atol: float, rtol: float) -> bool:
    """Compare two arrays and report differences."""
    match = np.allclose(expected, actual, atol=atol, rtol=rtol)
    if not match:
        mismatch_count = np.sum(
            ~np.isclose(expected, actual, atol=atol, rtol=rtol))
        max_abs_diff = np.abs(expected - actual).max()
        max_rel_diff = (np.abs(expected - actual) /
                        (np.abs(expected) + 1e-8)).max()
        print(
            f"  Round {round_idx + 1}: {name} mismatch: {mismatch_count}/{expected.size} elements"
        )
        print(f"  {name} max abs diff: {max_abs_diff:.6f}")
        print(f"  {name} max rel diff: {max_rel_diff:.6f}")
    return match


@dataclass
class AttentionParams:
    """Parameters for attention testing."""
    batch_size: int = 1
    seq_len: int = 1
    num_q_heads: int = 8
    num_kv_heads: int = 8
    head_size: int = 128
    kv_cache_capacity: int = 16
    max_batch_size: int = 8
    max_seq_len: int = 8
    max_position_embeddings: int = 32
    qk_scale: Optional[float] = None
    is_prefill: bool = False

    def __post_init__(self):
        self.qkv_hidden_size = (self.num_q_heads + self.num_kv_heads +
                                self.num_kv_heads) * self.head_size

        # Default to 1.0 / sqrt(head_size) if not specified
        if self.qk_scale is None:
            self.qk_scale = 1.0 / (self.head_size**0.5)


class NumpyAttentionReference:
    """Numpy reference implementation of attention with RoPE and KV caching."""

    @staticmethod
    def apply_rotary_embedding(x: np.ndarray, cos_cache: np.ndarray,
                               sin_cache: np.ndarray,
                               position_ids: np.ndarray) -> np.ndarray:
        """
        Apply rotary position embedding.

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_size]
            cos_cache: Cosine cache [max_pos_emb, head_size // 2]
            sin_cache: Sine cache [max_pos_emb, head_size // 2]
            position_ids: Position indices [batch, seq_len]

        Returns:
            Tensor with rotary embeddings applied
        """
        batch, num_heads, seq_len, head_size = x.shape
        half_dim = head_size // 2

        # Reshape x to separate the two halves for rotation
        x_reshaped = x.reshape(batch, num_heads, seq_len, 2, half_dim)
        x1 = x_reshaped[:, :, :, 0, :]  # First half
        x2 = x_reshaped[:, :, :, 1, :]  # Second half

        # Get cos/sin values for positions
        # cos_cache/sin_cache: [max_pos_emb, half_dim]
        # position_ids: [batch, seq_len]
        cos_vals = cos_cache[position_ids]  # [batch, seq_len, half_dim]
        sin_vals = sin_cache[position_ids]  # [batch, seq_len, half_dim]

        # Add head dimension: [batch, 1, seq_len, half_dim]
        cos_vals = cos_vals[:, np.newaxis, :, :]
        sin_vals = sin_vals[:, np.newaxis, :, :]

        # Apply rotation: rotate_half
        rotated_x1 = x1 * cos_vals - x2 * sin_vals
        rotated_x2 = x1 * sin_vals + x2 * cos_vals

        # Interleave back
        rotated = np.stack([rotated_x1, rotated_x2], axis=3)
        rotated = rotated.reshape(batch, num_heads, seq_len, head_size)

        return rotated

    @staticmethod
    def update_kv_cache(cache: np.ndarray, new_kv: np.ndarray,
                        cache_indices: np.ndarray) -> np.ndarray:
        """
        Update KV cache with new key/value tensors.

        Args:
            cache: Existing cache [batch, num_heads, capacity, head_size]
            new_kv: New K or V values [batch, num_heads, seq_len, head_size]
            cache_indices: Indices where to insert [batch]

        Returns:
            Updated cache
        """
        batch, _, seq_len, _ = new_kv.shape
        updated_cache = cache.copy()

        for b in range(batch):
            idx = cache_indices[b]
            updated_cache[b, :, idx:idx + seq_len, :] = new_kv[b]

        return updated_cache

    @staticmethod
    def scaled_dot_product_attention(
            q: np.ndarray,
            k: np.ndarray,
            v: np.ndarray,
            scale: Optional[float] = None,
            attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query [batch, num_heads, seq_q, head_size]
            k: Key [batch, num_heads, seq_k, head_size]
            v: Value [batch, num_heads, seq_k, head_size]
            scale: Scaling factor (default: 1/sqrt(head_size))
            attention_mask: Optional mask [seq_q, seq_k] with 1 where attention is allowed

        Returns:
            Attention output [batch, num_heads, seq_q, head_size]
        """
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])

        # Q @ K^T
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = np.where(attention_mask == 1, scores, -np.inf)

        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(
            attn_weights, axis=-1, keepdims=True)

        # Apply to values
        output = np.matmul(attn_weights, v)

        return output

    @classmethod
    def compute_attention(
        cls,
        qkv: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        rope_cos_cache: np.ndarray,
        rope_sin_cache: np.ndarray,
        position_ids: np.ndarray,
        cache_indices: np.ndarray,
        params: AttentionParams,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete attention computation with RoPE and KV caching.

        Returns:
            (attention_output, updated_k_cache, updated_v_cache)
        """
        batch = params.batch_size
        seq_len = params.seq_len
        num_q_heads = params.num_q_heads
        num_kv_heads = params.num_kv_heads
        head_size = params.head_size

        # Split QKV: [batch, seq_len, qkv_hidden]
        q_start = 0
        q_size = num_q_heads * head_size
        k_start = q_size
        k_size = num_kv_heads * head_size
        v_start = k_start + k_size
        v_size = num_kv_heads * head_size

        q = qkv[:, :, q_start:q_start + q_size]
        k = qkv[:, :, k_start:k_start + k_size]
        v = qkv[:, :, v_start:v_start + v_size]

        # Reshape: [batch, seq_len, num_heads*head_size] -> [batch, num_heads, seq_len, head_size]
        q = q.reshape(batch, seq_len, num_q_heads,
                      head_size).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_kv_heads,
                      head_size).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_kv_heads,
                      head_size).transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K
        q_rope = cls.apply_rotary_embedding(q, rope_cos_cache, rope_sin_cache,
                                            position_ids)
        k_rope = cls.apply_rotary_embedding(k, rope_cos_cache, rope_sin_cache,
                                            position_ids)

        # Update KV cache
        updated_k_cache = cls.update_kv_cache(k_cache, k_rope, cache_indices)
        updated_v_cache = cls.update_kv_cache(v_cache, v, cache_indices)

        # Get present keys/values (slice from cache)
        present_length = cache_indices[0] + seq_len
        k_present = updated_k_cache[:, :, :present_length, :]
        v_present = updated_v_cache[:, :, :present_length, :]

        # Compute attention with the configured scale
        attn_output = cls.scaled_dot_product_attention(q_rope, k_present,
                                                       v_present,
                                                       params.qk_scale,
                                                       attention_mask)

        # Reshape output: [batch, num_heads, seq_len, head_size] -> [batch, seq_len, num_heads*head_size]
        attn_output = attn_output.transpose(0, 2, 1,
                                            3).reshape(batch, seq_len,
                                                       num_q_heads * head_size)

        return attn_output, updated_k_cache, updated_v_cache
