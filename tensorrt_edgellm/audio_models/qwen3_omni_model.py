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
Qwen3-Omni audio encoder model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen3-Omni audio encoder,
enabling ONNX export with proper attention mechanism handling.
"""

from typing import Any

import torch
import torch.nn as nn
from transformers.models.qwen3_omni.modeling_qwen3_omni import (
    Qwen3OmniAudioAttention, Qwen3OmniAudioEncoder, Qwen3OmniAudioEncoderLayer)

from ..onnx_export.onnx_utils import export_onnx


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    attention_mask: torch.Tensor,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3OmniAudioAttentionPatch(Qwen3OmniAudioAttention):
    """
    Patched version of Qwen3-Omni audio attention for ONNX export.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask with shape [num_attention_elems, num_attention_elems]
            
        Returns:
            Attention output
        """
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            scaling=self.scaling,
            attention_mask=attention_mask,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class Qwen3OmniAudioEncoderLayerPatch(Qwen3OmniAudioEncoderLayer):
    """
    Patched version of Qwen3OmniAudioEncoderLayer with custom attention mechanism.
    
    This class replaces the original attention mechanism with a custom implementation
    that is compatible with ONNX export.
    """

    def __init__(self,
                 config: Any,
                 attn_implementation: str = "eager") -> None:
        super().__init__(config)
        self.self_attn = Qwen3OmniAudioAttentionPatch(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer
            attention_mask (`torch.FloatTensor`, optional): attention mask with shape [num_attention_elems, num_attention_elems]

        Returns:
            hidden_states (`torch.FloatTensor`): output of the layer
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        return hidden_states


class Qwen3OmniAudioEncoderPatch(Qwen3OmniAudioEncoder):
    """
    Patched version of Qwen3OmniAudioEncoder for ONNX export.
    
    This class provides a wrapper around the original Qwen3-Omni audio encoder
    with custom blocks that are compatible with ONNX export.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the patched audio encoder.
        
        Args:
            config: Model configuration object
        """
        super().__init__(config)
        # Replace all blocks with patched versions
        self.layers = nn.ModuleList([
            Qwen3OmniAudioEncoderLayerPatch(config,
                                            config._attn_implementation)
            for _ in range(config.encoder_layers)
        ])

    def forward(self, padded_feature: torch.Tensor,
                padded_mask_after_cnn_indices: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        Three changes have been made to the original forward pass:
        1. padded_feature preprocessing is not TRT friendly and is moved to runtime.
        2. boolean indexing using padded_mask_after_cnn results in NonZero node in ONNX and is not TRT friendly.
        Instead, use padded_mask_after_cnn_indices as model input: padded_mask_after_cnn_indices = torch.nonzero(padded_mask_after_cnn).
        3. cu_seqlens is only used in Flash Attention, not in eager Attention, so after_cnn_lens input is dropped. In its place, attention_mask is used.
        
        Args:
            padded_feature: Padded feature tensor [num_chunks, num_mel_bins, n_window]
            padded_mask_after_cnn_indices: Indices for the boolean padded mask after CNN layers [num_attention_elems, 2]
            attention_mask: Optional attention mask with shape [num_attention_elems, num_attention_elems]

        Returns:
            `torch.Tensor`: hidden_states.
        """
        padded_feature = padded_feature.unsqueeze(1)

        padded_embed = torch.nn.functional.gelu(self.conv2d1(padded_feature))
        padded_embed = torch.nn.functional.gelu(self.conv2d2(padded_embed))
        padded_embed = torch.nn.functional.gelu(self.conv2d3(padded_embed))

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.
            positional_embedding[:padded_embed.shape[1], :].unsqueeze(0).to(
                padded_embed.dtype))
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn_indices[:, 0],
                                     padded_mask_after_cnn_indices[:, 1]]

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states,
                                          attention_mask=attention_mask)
            hidden_states = layer_outputs

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


def export_qwen3_omni_audio(
    model: Qwen3OmniAudioEncoderPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen3-Omni audio encoder model to ONNX format.
    
    This function takes a patched Qwen3-Omni audio encoder model, prepares dummy inputs 
    for ONNX export, and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen3-Omni audio encoder model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """

    # Prepare dummy inputs for ONNX export
    num_mel_bins = model.config.num_mel_bins
    n_window = model.config.n_window
    num_chunks = 3
    padded_mask_chunk_length = 13
    num_attention_elems = 38

    # Create input tensors with appropriate shapes and dtypes
    padded_feature = torch.randn(num_chunks,
                                 num_mel_bins,
                                 n_window * 2,
                                 dtype=torch_dtype,
                                 device=model.device)
    padded_mask_after_cnn = torch.tensor(
        [True] * num_attention_elems + [False] *
        (num_chunks * padded_mask_chunk_length - num_attention_elems),
        device=model.device).reshape(num_chunks, padded_mask_chunk_length)
    padded_mask_after_cnn_indices = torch.nonzero(padded_mask_after_cnn)

    # In this case the attention mask should be a block diagonal matrix with block sizes 26x26, 12x12,
    # as indicated by cu_seqlens, such that only audio signals within each block attend to each other.
    attention_mask = torch.full(
        [num_attention_elems, num_attention_elems],
        torch.finfo(torch_dtype).min,
        device=model.device,
        dtype=torch_dtype,
    )
    cu_seqlens = [0, 26, 38]
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                       cu_seqlens[i - 1]:cu_seqlens[i]] = 0

    inputs = (padded_feature, padded_mask_after_cnn_indices, attention_mask)
    input_names = [
        "padded_feature", "padded_mask_after_cnn_indices", "attention_mask"
    ]
    output_names = ["last_hidden_state"]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        # Model inputs
        'padded_feature': {
            0: 'num_chunks'
        },
        'padded_mask_after_cnn_indices': {
            0: 'num_attention_elems'
        },
        'attention_mask': {
            0: 'num_attention_elems',
            1: 'num_attention_elems'
        },
        # Model outputs
        'last_hidden_state': {
            0: 'num_attention_elems'
        },
    }

    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
