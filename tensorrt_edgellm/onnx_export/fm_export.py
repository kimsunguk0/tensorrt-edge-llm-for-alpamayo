# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for exporting Alpamayo FM one-step ONNX models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


def pad_packet_for_static_engine(packet: dict[str, Any], *, max_seq_len: int) -> dict[str, Any]:
    """Pad a replay packet to the static KV prefix length expected by the TRT engine."""
    kv_cache = packet["kv_cache"]
    attention_mask = packet["attention_mask"]
    prefix_len = int(kv_cache.shape[-2])
    if prefix_len > max_seq_len:
        raise ValueError(f"prefix_len={prefix_len} exceeds max_seq_len={max_seq_len}")
    if prefix_len == max_seq_len:
        return dict(packet)

    n_diffusion_tokens = int(packet["position_ids"].shape[-1])
    pad_prefix = max_seq_len - prefix_len
    padded = dict(packet)

    kv_shape = list(kv_cache.shape)
    kv_shape[-2] = pad_prefix
    kv_pad = torch.zeros(kv_shape, dtype=kv_cache.dtype)
    padded["kv_cache"] = torch.cat((kv_cache, kv_pad), dim=-2)

    prefix_mask = attention_mask[..., :prefix_len]
    current_mask = attention_mask[..., prefix_len : prefix_len + n_diffusion_tokens]
    pad_value = torch.finfo(attention_mask.dtype).min
    mask_pad = torch.full(
        (*attention_mask.shape[:-1], pad_prefix),
        pad_value,
        dtype=attention_mask.dtype,
    )
    padded["attention_mask"] = torch.cat((prefix_mask, mask_pad, current_mask), dim=-1)
    padded["kv_cache_padding"] = {
        "original_prefix_len": prefix_len,
        "max_seq_len": max_seq_len,
        "pad_prefix": pad_prefix,
    }
    return padded


def _import_alpamayo_model(alpamayo_src_dir: str | Path):
    alpamayo_src_dir = str(alpamayo_src_dir)
    if alpamayo_src_dir not in sys.path:
        sys.path.insert(0, alpamayo_src_dir)
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    return Alpamayo1_5


class FmOneStepExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])
        self.forward_kwargs: dict[str, Any] = {}
        if model.config.expert_non_causal_attention:
            self.forward_kwargs["is_causal"] = False

    @staticmethod
    def _dense_kv_to_dynamic_cache(kv_cache: torch.Tensor) -> DynamicCache:
        legacy = []
        for layer_idx in range(kv_cache.shape[0]):
            key = kv_cache[layer_idx, :, 0].contiguous()
            value = kv_cache[layer_idx, :, 1].contiguous()
            legacy.append((key, value))
        return DynamicCache.from_legacy_cache(tuple(legacy))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        kv_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_cache = self._dense_kv_to_dynamic_cache(kv_cache)
        prefill_seq_len = prompt_cache.get_seq_length()

        future_token_embeds = self.model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], self.n_diffusion_tokens, -1)

        expert_out = self.model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **self.forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)

        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens :]
        v = self.model.action_out_proj(last_hidden).view(
            -1, *self.model.action_space.get_action_space_dims()
        )
        v = v.to(torch.float32)
        next_x = x.to(torch.float32) + dt.to(torch.float32) * v
        return next_x, v, future_token_embeds.to(torch.float32)


def summarize_tensor(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "abs_max": float(t.abs().max().item()) if t.numel() else 0.0,
        "mean": float(t.float().mean().item()) if t.numel() else 0.0,
    }


def export_fm_model(
    *,
    model_dir: str,
    output_dir: str,
    alpamayo_src_dir: str,
    packet_path: str,
    max_seq_len: int = 8192,
    device: str = "cuda",
    dtype: str = "bf16",
    check_only: bool = False,
) -> dict[str, Any]:
    """Export Alpamayo 1.5 FM one-step graph to ONNX."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype={dtype}; expected one of {sorted(dtype_map)}")
    model_dtype = dtype_map[dtype]

    workdir = Path(output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    Alpamayo1_5 = _import_alpamayo_model(alpamayo_src_dir)

    torch_device = torch.device(device)
    packet = torch.load(packet_path, map_location="cpu")
    padded_packet = pad_packet_for_static_engine(packet, max_seq_len=max_seq_len)

    model = Alpamayo1_5.from_pretrained(
        model_dir,
        dtype=model_dtype,
        local_files_only=True,
    ).to(torch_device)
    model.eval()

    wrapper = FmOneStepExportWrapper(model).to(torch_device)
    wrapper.eval()

    x = padded_packet["x0"].to(device=torch_device, dtype=torch.float32).contiguous()
    t = torch.zeros((1, 1, 1), device=torch_device, dtype=torch.float32)
    dt = torch.full(
        (1, 1, 1),
        float(packet["action_space_constants"]["dt_value"]),
        device=torch_device,
        dtype=torch.float32,
    )
    kv_cache = padded_packet["kv_cache"].to(device=torch_device, dtype=model_dtype).contiguous()
    attention_mask = padded_packet["attention_mask"].to(device=torch_device, dtype=torch.float32).contiguous()
    position_ids = padded_packet["position_ids"].to(device=torch_device, dtype=torch.int64).contiguous()

    with torch.no_grad(), torch.autocast("cuda", dtype=model_dtype):
        next_x, v, future_token_embeds = wrapper(x, t, dt, kv_cache, attention_mask, position_ids)

    check = {
        "model_dir": str(model_dir),
        "packet": str(packet_path),
        "max_seq_len": int(max_seq_len),
        "device": str(torch_device),
        "dtype": dtype,
        "inputs": {
            "x": summarize_tensor(x),
            "t": summarize_tensor(t),
            "dt": summarize_tensor(dt),
            "kv_cache": summarize_tensor(kv_cache),
            "attention_mask": summarize_tensor(attention_mask),
            "position_ids": {
                "shape": list(position_ids.shape),
                "dtype": str(position_ids.dtype),
                "min": int(position_ids.min().item()),
                "max": int(position_ids.max().item()),
            },
        },
        "outputs": {
            "next_x": summarize_tensor(next_x),
            "v": summarize_tensor(v),
            "future_token_embeds": summarize_tensor(future_token_embeds),
        },
    }
    check_summary = workdir / "check_summary.json"
    check_summary.write_text(json.dumps(check, indent=2))

    if check_only:
        result = {
            "workspace_dir": str(workdir),
            "check_only": True,
            "check_summary": str(check_summary),
        }
        return result

    onnx_path = workdir / f"alpamayo15_fm_one_step_{dtype}.onnx"
    torch.onnx.export(
        wrapper,
        (x, t, dt, kv_cache, attention_mask, position_ids),
        str(onnx_path),
        input_names=["x", "t", "dt", "kv_cache", "attention_mask", "position_ids"],
        output_names=["next_x", "v", "future_token_embeds"],
        opset_version=18,
        do_constant_folding=False,
        export_params=True,
        external_data=True,
    )

    result = {
        "onnx_path": str(onnx_path),
        "external_data_exists": onnx_path.exists() and (workdir / f"alpamayo15_fm_one_step_{dtype}.onnx.data").exists(),
        "check_summary": str(check_summary),
    }
    export_summary = workdir / "export_summary.json"
    export_summary.write_text(json.dumps(result, indent=2))
    return result
