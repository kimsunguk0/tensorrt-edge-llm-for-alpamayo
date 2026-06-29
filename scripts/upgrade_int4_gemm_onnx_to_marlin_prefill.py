#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upgrade legacy Int4GroupwiseGemmPlugin ONNX nodes with Marlin prefill inputs.

The existing Alpamayo INT4AWQ ONNX stores plugin weights in the legacy AWQ GEMM
layout [N/2, K] int8. This script keeps those inputs for the fast decode GEMV
path and appends Marlin-packed [K/16, 8*N] weights plus Marlin-permuted scales
for prefill GEMM.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
from onnx import helper


_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)
_MARLIN_OUT_IDX = np.array(
    [(i % 32) * 4 + (i // 32) for i in range(128)], dtype=np.int32)
_MARLIN_ROW_PATTERN = np.array(
    [[0, 1, 8, 9, 0, 1, 8, 9], [2, 3, 10, 11, 2, 3, 10, 11],
     [4, 5, 12, 13, 4, 5, 12, 13], [6, 7, 14, 15, 6, 7, 14, 15]],
    dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_MARLIN_ROW_PATTERN, (32, 1))
_MARLIN_COL_IDX = np.array(
    [[(thread // 32) * 16 + (thread % 32) // 4 + (lane // 4) * 8
      for lane in range(8)] for thread in range(128)],
    dtype=np.int32)


def _tensor_external_data(tensor: onnx.TensorProto) -> Dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _numpy_dtype(data_type: int) -> np.dtype:
    if data_type == onnx.TensorProto.INT8:
        return np.dtype(np.int8)
    if data_type == onnx.TensorProto.FLOAT16:
        return np.dtype(np.float16)
    raise ValueError(f"Unsupported tensor data type {data_type}")


def _read_external_tensor(tensor: onnx.TensorProto,
                          base_dir: Path) -> np.ndarray:
    meta = _tensor_external_data(tensor)
    location = meta["location"]
    offset = int(meta.get("offset", "0"))
    length = int(meta["length"])
    dtype = _numpy_dtype(tensor.data_type)
    with open(base_dir / location, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    return np.frombuffer(data, dtype=dtype).copy().reshape(tuple(tensor.dims))


def _make_external_tensor(name: str, data_type: int, shape: Tuple[int, ...],
                          location: str, offset: int,
                          length: int) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(shape)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.extend([
        onnx.StringStringEntryProto(key="location", value=location),
        onnx.StringStringEntryProto(key="offset", value=str(offset)),
        onnx.StringStringEntryProto(key="length", value=str(length)),
    ])
    return tensor


def _unpack_legacy_awq_weight(packed_int8: np.ndarray, n: int,
                              k: int) -> np.ndarray:
    """Invert int4_gemm_plugin.pack_intweights.

    Returns unpacked weights [N, K] with unsigned int4 values in [0, 15].
    """
    packed_u16 = packed_int8.view(np.uint16).reshape(n // 4, k)
    lanes = np.empty((n // 4, k // 64, 64, 4), dtype=np.uint16)
    lanes[..., 0] = packed_u16.reshape(n // 4, k // 64, 64) & 0xF
    lanes[..., 1] = (packed_u16.reshape(n // 4, k // 64, 64) >> 4) & 0xF
    lanes[..., 2] = (packed_u16.reshape(n // 4, k // 64, 64) >> 8) & 0xF
    lanes[..., 3] = (packed_u16.reshape(n // 4, k // 64, 64) >> 12) & 0xF

    x = lanes.reshape(n // 4, k // 64, 4, 64).transpose(0, 2, 1, 3)
    x = x.reshape(n, k)

    # Invert reorder each 8 weights: [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7].
    reorder8 = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)
    inv8 = np.argsort(reorder8)
    x = x.reshape(n, k // 32, 4, 8)[:, :, :, inv8].reshape(n, k)

    # Invert 32-wide reorder:
    # arange(32).reshape(4,4,2).transpose(1,0,2).reshape(32).
    reorder32 = np.arange(32).reshape(4, 4, 2).transpose(1, 0,
                                                        2).reshape(32)
    inv32 = np.argsort(reorder32)
    x = x.reshape(n, k // 32, 32)[:, :, inv32].reshape(n, k)
    return x.astype(np.int16, copy=False)


def _marlin_permute_scales(scales: np.ndarray, size_k: int, size_n: int,
                           group_size: int) -> np.ndarray:
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).copy()


def _pack_marlin_dense(weights_q: np.ndarray, scales: np.ndarray,
                       group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    n, k = weights_q.shape
    if k % 16 != 0 or n % 64 != 0 or k % group_size != 0:
        raise ValueError(f"Unsupported Marlin shape N={n}, K={k}, G={group_size}")
    if scales.shape != (k // group_size, n):
        raise ValueError(f"Unexpected scales shape {scales.shape}")

    w_kn = weights_q.transpose(1, 0).copy().astype(np.uint32)
    k_tiles, n_tiles = k // 16, n // 64
    tiles = w_kn.reshape(k_tiles, 16, n_tiles, 64).transpose(0, 2, 1, 3)
    gathered = tiles[:, :, _MARLIN_ROW_IDX,
                     _MARLIN_COL_IDX][:, :, :,
                                      _MARLIN_PACK_IDX].astype(np.uint32)
    packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4)
                  | (gathered[:, :, :, 2] << 8)
                  | (gathered[:, :, :, 3] << 12)
                  | (gathered[:, :, :, 4] << 16)
                  | (gathered[:, :, :, 5] << 20)
                  | (gathered[:, :, :, 6] << 24)
                  | (gathered[:, :, :, 7] << 28))

    out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
    for n_tile_id in range(n_tiles):
        out[:, n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                               n_tile_id, :]
    marlin_weight = out.view(np.int8).reshape(k_tiles, 8 * n).copy()
    marlin_scales = _marlin_permute_scales(scales.copy(), k, n, group_size)
    return marlin_weight, marlin_scales


def _copy_runtime_files(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in input_dir.iterdir():
        dst = output_dir / src.name
        if src.name == "model.onnx" or dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, copy_function=os.link)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def upgrade(input_dir: Path, output_dir: Path) -> None:
    _copy_runtime_files(input_dir, output_dir)
    model = onnx.load(input_dir / "model.onnx", load_external_data=False)
    initializers = {init.name: init for init in model.graph.initializer}
    marlin_data_name = "onnx_model_marlin.data"
    marlin_data_path = output_dir / marlin_data_name
    if marlin_data_path.exists():
        marlin_data_path.unlink()

    converted = 0
    skipped = 0
    with open(marlin_data_path, "wb") as marlin_file:
        for node in model.graph.node:
            if node.op_type != "Int4GroupwiseGemmPlugin":
                continue
            attrs = {attr.name: helper.get_attribute_value(attr)
                     for attr in node.attribute}
            if attrs.get("use_marlin", 0):
                skipped += 1
                continue

            gemm_n = int(attrs["gemm_n"])
            gemm_k = int(attrs["gemm_k"])
            group_size = int(attrs["group_size"])
            if not (gemm_k % 16 == 0 and gemm_n % 64 == 0
                    and gemm_k % group_size == 0):
                skipped += 1
                continue

            qweight = _read_external_tensor(initializers[node.input[1]],
                                            input_dir)
            scales = _read_external_tensor(initializers[node.input[2]],
                                           input_dir)
            weights_q = _unpack_legacy_awq_weight(qweight, gemm_n, gemm_k)
            marlin_weight, marlin_scales = _pack_marlin_dense(
                weights_q, scales, group_size)

            weight_offset = marlin_file.tell()
            marlin_file.write(marlin_weight.tobytes(order="C"))
            scale_offset = marlin_file.tell()
            marlin_file.write(marlin_scales.tobytes(order="C"))

            weight_name = f"{node.name}/marlin_qweight"
            scale_name = f"{node.name}/marlin_scales"
            model.graph.initializer.append(
                _make_external_tensor(weight_name, onnx.TensorProto.INT8,
                                      tuple(marlin_weight.shape),
                                      marlin_data_name, weight_offset,
                                      marlin_weight.nbytes))
            model.graph.initializer.append(
                _make_external_tensor(scale_name, onnx.TensorProto.FLOAT16,
                                      tuple(marlin_scales.shape),
                                      marlin_data_name, scale_offset,
                                      marlin_scales.nbytes))
            node.input.extend([weight_name, scale_name])
            node.attribute.extend([helper.make_attribute("use_marlin", 1)])
            converted += 1

    onnx.save(model, output_dir / "model.onnx")
    print(
        f"Converted {converted} Int4GroupwiseGemmPlugin nodes; skipped {skipped}."
    )
    print(f"Wrote {output_dir / 'model.onnx'}")
    print(f"Wrote {marlin_data_path} ({marlin_data_path.stat().st_size} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx-dir", type=Path, required=True)
    parser.add_argument("--output-onnx-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upgrade(args.input_onnx_dir.resolve(), args.output_onnx_dir.resolve())


if __name__ == "__main__":
    main()
