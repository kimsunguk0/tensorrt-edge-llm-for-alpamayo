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
"""CLI for exporting Alpamayo FM one-step ONNX models."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from tensorrt_edgellm.onnx_export.fm_export import export_fm_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Alpamayo 1.5 FM one-step model to ONNX format"
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the Alpamayo model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the exported ONNX model")
    parser.add_argument(
        "--alpamayo_src_dir",
        type=str,
        required=True,
        help="Path to the Alpamayo Python source tree (directory containing alpamayo1_5/)",
    )
    parser.add_argument(
        "--packet",
        type=str,
        required=True,
        help="Replay packet (.pt) used to trace the static FM one-step graph",
    )
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Static max KV prefix length for export")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to load the model on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=False,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for export",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        required=False,
        choices=["fp8", "mxfp8", "nvfp4"],
        default=None,
        help="Optional FM quantization mode applied before ONNX export",
    )
    parser.add_argument(
        "--calib_packet_glob",
        type=str,
        required=False,
        default=None,
        help="Optional glob for calibration replay_packet.pt files used during quantization",
    )
    parser.add_argument(
        "--num_calib_packets",
        type=int,
        required=False,
        default=8,
        help="Maximum number of calibration packets to use when quantization is enabled",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Run wrapper validation and write check_summary.json without exporting ONNX",
    )

    args = parser.parse_args()

    try:
        calib_packet_paths = None
        if args.calib_packet_glob:
            from pathlib import Path
            calib_packet_paths = sorted(str(path) for path in Path("/").glob(args.calib_packet_glob.lstrip("/")))
            calib_packet_paths = calib_packet_paths[: args.num_calib_packets]
            if not calib_packet_paths:
                raise ValueError(f"No calibration packets matched glob: {args.calib_packet_glob}")
        result = export_fm_model(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            alpamayo_src_dir=args.alpamayo_src_dir,
            packet_path=args.packet,
            calib_packet_paths=calib_packet_paths,
            max_seq_len=args.max_seq_len,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
            check_only=args.check_only,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during FM model export: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
