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
Qwen3-Omni visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen3-Omni visual models.
Qwen3-Omni shares the same architecture as Qwen3-VL with slight differences in merger
naming conventions. This module extends the Qwen3-VL implementation with minimal modifications.
"""

from typing import Any, Dict

import torch

from .qwen3_vl_model import Qwen3VLVisionModelPatch, export_qwen3_vl_visual


class Qwen3OmniVisionModelPatch(Qwen3VLVisionModelPatch):
    """
    Patched version of Qwen3-Omni vision model for ONNX export.
    
    This class extends Qwen3VLVisionModelPatch to handle Qwen3-Omni specific weight naming.
    The core architecture is identical to Qwen3-VL, with differences only in VisionPatchMerger
    class naming conventions:
    - Qwen3-Omni uses: ln_q, mlp.0, mlp.2
    - Qwen3-VL expects: norm, linear_fc1, linear_fc2
    
    This class provides automatic key mapping during weight loading to ensure compatibility.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the patched Qwen3-Omni vision transformer.
        
        Args:
            config: Model configuration object (Qwen3VLVisionConfig compatible)
        """
        super().__init__(config)

    def load_omni_state_dict(self,
                             state_dict: Dict[str, torch.Tensor],
                             strict: bool = False) -> Any:
        """
        Load Qwen3-Omni state dict with automatic key mapping.
        
        This method transforms Qwen3-Omni weight keys to match Qwen3-VL naming conventions:
        
        Main merger mappings:
            merger.ln_q.*       -> merger.norm.*
            merger.mlp.0.*      -> merger.linear_fc1.*
            merger.mlp.2.*      -> merger.linear_fc2.*
        
        Deepstack merger mappings:
            merger_list.{i}.ln_q.*       -> deepstack_merger_list.{i}.norm.*
            merger_list.{i}.mlp.0.*      -> deepstack_merger_list.{i}.linear_fc1.*
            merger_list.{i}.mlp.2.*      -> deepstack_merger_list.{i}.linear_fc2.*
        
        Note: mlp.1 (GELU activation) has no learnable parameters and is not mapped.
        
        Args:
            state_dict: Qwen3-Omni model state dictionary
            strict: Whether to strictly enforce that keys match. Default: False
        
        Returns:
            NamedTuple with missing_keys and unexpected_keys lists
        """
        mapped_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Map main merger keys
            if 'merger.ln_q.' in key:
                new_key = key.replace('merger.ln_q.', 'merger.norm.')
            elif 'merger.mlp.0.' in key:
                new_key = key.replace('merger.mlp.0.', 'merger.linear_fc1.')
            elif 'merger.mlp.2.' in key:
                new_key = key.replace('merger.mlp.2.', 'merger.linear_fc2.')

            # Map deepstack merger list keys
            elif 'merger_list.' in key:
                # First change merger_list to deepstack_merger_list
                new_key = key.replace('merger_list.', 'deepstack_merger_list.')
                # Then map the internal structure
                new_key = new_key.replace('.ln_q.', '.norm.')
                new_key = new_key.replace('.mlp.0.', '.linear_fc1.')
                new_key = new_key.replace('.mlp.2.', '.linear_fc2.')

            # All other keys (patch_embed, blocks, pos_embed, etc.) remain unchanged
            mapped_dict[new_key] = value

        # Load mapped weights using parent class method
        return super().load_state_dict(mapped_dict, strict=strict)


def export_qwen3_omni_visual(
    model: Qwen3OmniVisionModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen3-Omni visual model to ONNX format.
    
    This function leverages the Qwen3-VL export implementation since the architectures
    are identical. The exported ONNX model has the same input/output signature as Qwen3-VL:
    
    Inputs (5):
        - input: Pixel values [hw, input_dim]
        - rotary_pos_emb: Rotary position embeddings [hw, rotary_pos_emb_dim]
        - attention_mask: Attention mask [1, hw, hw]
        - fast_pos_embed_idx: Fast position embedding indices [4, hw]
        - fast_pos_embed_weight: Fast position embedding weights [4, hw]
    
    Outputs (4):
        - output: Main visual features [image_token_len, out_hidden_size]
        - deepstack_features.0: First deepstack features [image_token_len, out_hidden_size]
        - deepstack_features.1: Second deepstack features [image_token_len, out_hidden_size]
        - deepstack_features.2: Third deepstack features [image_token_len, out_hidden_size]
    
    Args:
        model: Patched Qwen3-Omni vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model (typically torch.float16)
    """
    # Directly use Qwen3-VL export function since the model structure is identical
    export_qwen3_vl_visual(model, output_dir, torch_dtype)
