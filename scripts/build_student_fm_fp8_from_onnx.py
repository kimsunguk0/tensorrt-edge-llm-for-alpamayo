#!/usr/bin/env python3
"""Quantize a static Alpamayo FM ONNX into an FP8 ONNX using replay packets for calibration."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from modelopt.onnx.llm_export_utils.surgeon_utils import fold_fp8_qdq_to_dq
from modelopt.onnx.quantization.fp8 import CalibrationDataReader, quantize


class ReplayPacketCalibrationReader(CalibrationDataReader):
    def __init__(self, packet_paths: list[Path]):
        self._packet_paths = packet_paths
        self._index = 0

    def __len__(self) -> int:
        return len(self._packet_paths)

    def set_range(self, start_index: int, end_index: int):
        self._packet_paths = self._packet_paths[start_index:end_index]
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._packet_paths):
            return None

        packet = torch.load(self._packet_paths[self._index], map_location="cpu")
        self._index += 1

        dt_value = float(packet["action_space_constants"]["dt_value"])
        feed = {
            "x": packet["x0"].detach().cpu().numpy().astype(np.float32, copy=False),
            "t": np.zeros((1, 1, 1), dtype=np.float32),
            "dt": np.full((1, 1, 1), dt_value, dtype=np.float32),
            "kv_cache": packet["kv_cache"].detach().cpu().numpy().astype(np.float16, copy=False),
            "attention_mask": packet["attention_mask"].detach().cpu().numpy().astype(np.float32, copy=False),
            "position_ids": packet["position_ids"].detach().cpu().numpy().astype(np.int64, copy=False),
        }
        return feed

    def get_first(self) -> dict[str, np.ndarray]:
        self._index = 0
        first = self.get_next()
        if first is None:
            raise RuntimeError("Calibration reader is empty")
        self._index = 0
        return first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", type=Path, required=True, help="Path to the source FP16 ONNX")
    parser.add_argument(
        "--packet-glob",
        type=str,
        required=True,
        help="Glob for replay_packet.pt files used for FP8 calibration",
    )
    parser.add_argument("--num-packets", type=int, default=8, help="Maximum number of calibration packets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model.onnx and onnx_model.data will be written",
    )
    return parser.parse_args()


def save_model(model: onnx.ModelProto, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    data_path = output_dir / "onnx_model.data"
    if onnx_path.exists():
        onnx_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        convert_attribute=True,
    )
    return onnx_path


def main() -> None:
    args = parse_args()
    packet_paths = [Path(p) for p in sorted(glob.glob(args.packet_glob))[: args.num_packets]]
    if not packet_paths:
        raise RuntimeError(f"No replay_packet.pt files matched glob: {args.packet_glob}")

    reader = ReplayPacketCalibrationReader(packet_paths)
    quantized = quantize(
        str(args.input_onnx),
        calibration_method="distribution",
        calibration_data_reader=reader,
        calibration_eps=["cuda:0"],
        op_types_to_quantize=["Gemm", "MatMul"],
        high_precision_dtype="fp16",
        mha_accumulation_dtype="fp16",
        use_external_data_format=True,
        direct_io_types=True,
        log_level="INFO",
    )
    graph = gs.import_onnx(quantized)
    graph = fold_fp8_qdq_to_dq(graph)
    folded = gs.export_onnx(graph)
    onnx_path = save_model(folded, args.output_dir)

    summary = {
        "input_onnx": str(args.input_onnx),
        "output_onnx": str(onnx_path),
        "num_packets": len(packet_paths),
        "packet_paths": [str(p) for p in packet_paths],
        "num_nodes": len(folded.graph.node),
        "fp8_dq_nodes": sum(1 for node in folded.graph.node if node.op_type == "DequantizeLinear"),
    }
    (args.output_dir / "quant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
