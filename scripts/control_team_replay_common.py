#!/usr/bin/env python3
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"ALPA"
VERSION = 1

COORD_MODE_LOCAL = 0
COORD_MODE_WORLD = 1

FLAG_VALID = 1 << 0
FLAG_LOOPED = 1 << 1
FLAG_END_OF_STREAM = 1 << 2

HEADER_FORMAT = "<4sHHIIIQQHHf"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
POINT_FORMAT = "<5f"
POINT_SIZE = struct.calcsize(POINT_FORMAT)
CRC_SIZE = 4


@dataclass
class PlanBank:
    chunk_id: int
    sample_id: np.ndarray
    t0_us: np.ndarray
    t_rel_s: np.ndarray
    plan_dt_s: float
    plan_points: int
    coord_mode_default: int
    control_dt_default_s: float
    control_points_default: int
    traj_x_local: np.ndarray
    traj_y_local: np.ndarray
    traj_yaw_local: np.ndarray
    traj_v_mps: np.ndarray
    traj_curvature: np.ndarray
    history_x_local: np.ndarray
    history_y_local: np.ndarray
    ref_x_world_enu: np.ndarray
    ref_y_world_enu: np.ndarray
    ref_yaw_world_rad: np.ndarray
    output_text: np.ndarray


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def coord_mode_name(coord_mode: int) -> str:
    return "world" if int(coord_mode) == COORD_MODE_WORLD else "local"


def coord_mode_code(name: str) -> int:
    if name.lower() == "world":
        return COORD_MODE_WORLD
    return COORD_MODE_LOCAL


def load_plan_bank(path: str) -> PlanBank:
    with np.load(path, allow_pickle=False) as npz:
        return PlanBank(
            chunk_id=int(npz["chunk_id"]),
            sample_id=np.asarray(npz["sample_id"], dtype=np.int32),
            t0_us=np.asarray(npz["t0_us"], dtype=np.int64),
            t_rel_s=np.asarray(npz["t_rel_s"], dtype=np.float32),
            plan_dt_s=float(npz["plan_dt_s"]),
            plan_points=int(npz["plan_points"]),
            coord_mode_default=int(npz["coord_mode_default"]),
            control_dt_default_s=float(npz["control_dt_default_s"]),
            control_points_default=int(npz["control_points_default"]),
            traj_x_local=np.asarray(npz["traj_x_local"], dtype=np.float32),
            traj_y_local=np.asarray(npz["traj_y_local"], dtype=np.float32),
            traj_yaw_local=np.asarray(npz["traj_yaw_local"], dtype=np.float32),
            traj_v_mps=np.asarray(npz["traj_v_mps"], dtype=np.float32),
            traj_curvature=np.asarray(npz["traj_curvature"], dtype=np.float32),
            history_x_local=np.asarray(npz["history_x_local"], dtype=np.float32),
            history_y_local=np.asarray(npz["history_y_local"], dtype=np.float32),
            ref_x_world_enu=np.asarray(npz["ref_x_world_enu"], dtype=np.float32),
            ref_y_world_enu=np.asarray(npz["ref_y_world_enu"], dtype=np.float32),
            ref_yaw_world_rad=np.asarray(npz["ref_yaw_world_rad"], dtype=np.float32),
            output_text=np.asarray(npz["output_text"]),
        )


def _interp_scalar_series(values: np.ndarray, src_dt_s: float, times_s: np.ndarray) -> np.ndarray:
    src_times = np.arange(values.shape[0], dtype=np.float32) * np.float32(src_dt_s)
    return np.interp(times_s, src_times, values, left=values[0], right=values[-1]).astype(np.float32)


def sample_plan_points(
    bank: PlanBank,
    plan_idx: int,
    age_s: float,
    control_dt_s: float,
    control_points: int,
    coord_mode: int,
) -> dict[str, np.ndarray]:
    times_s = np.asarray(age_s, dtype=np.float32) + np.arange(control_points, dtype=np.float32) * np.float32(control_dt_s)
    x_local = _interp_scalar_series(bank.traj_x_local[plan_idx], bank.plan_dt_s, times_s)
    y_local = _interp_scalar_series(bank.traj_y_local[plan_idx], bank.plan_dt_s, times_s)
    yaw_unwrapped = np.unwrap(bank.traj_yaw_local[plan_idx].astype(np.float64)).astype(np.float32)
    yaw_local = wrap_angles(_interp_scalar_series(yaw_unwrapped, bank.plan_dt_s, times_s))
    v_mps = _interp_scalar_series(bank.traj_v_mps[plan_idx], bank.plan_dt_s, times_s)
    curvature = _interp_scalar_series(bank.traj_curvature[plan_idx], bank.plan_dt_s, times_s)

    if int(coord_mode) == COORD_MODE_WORLD:
        ref_x = float(bank.ref_x_world_enu[plan_idx])
        ref_y = float(bank.ref_y_world_enu[plan_idx])
        ref_yaw = float(bank.ref_yaw_world_rad[plan_idx])
        c = float(np.cos(ref_yaw))
        s = float(np.sin(ref_yaw))
        x_world = ref_x + c * x_local - s * y_local
        y_world = ref_y + s * x_local + c * y_local
        yaw_world = wrap_angles(yaw_local + ref_yaw)
        return {
            "x": x_world.astype(np.float32),
            "y": y_world.astype(np.float32),
            "yaw": yaw_world.astype(np.float32),
            "v": v_mps.astype(np.float32),
            "curvature": curvature.astype(np.float32),
        }

    return {
        "x": x_local.astype(np.float32),
        "y": y_local.astype(np.float32),
        "yaw": yaw_local.astype(np.float32),
        "v": v_mps.astype(np.float32),
        "curvature": curvature.astype(np.float32),
    }


def build_packet_dict(
    *,
    tx_seq: int,
    plan_seq: int,
    sample_id: int,
    source_t0_us: int,
    tx_time_us: int,
    coord_mode: int,
    dt_s: float,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    v: np.ndarray,
    curvature: np.ndarray,
    flags: int = FLAG_VALID,
) -> dict[str, Any]:
    points = []
    for idx in range(len(x)):
        points.append(
            {
                "x_m": float(x[idx]),
                "y_m": float(y[idx]),
                "yaw_rad": float(yaw[idx]),
                "v_mps": float(v[idx]),
                "curvature": float(curvature[idx]),
            }
        )
    return {
        "header": {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "flags": int(flags),
            "tx_seq": int(tx_seq),
            "plan_seq": int(plan_seq),
            "sample_id": int(sample_id),
            "source_t0_us": int(source_t0_us),
            "tx_time_us": int(tx_time_us),
            "coord_mode": int(coord_mode),
            "coord_mode_name": coord_mode_name(coord_mode),
            "num_points": int(len(points)),
            "dt_s": float(dt_s),
        },
        "points": points,
    }


def pack_packet(packet: dict[str, Any]) -> bytes:
    header = packet["header"]
    header_bytes = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        int(header["version"]),
        int(header["flags"]),
        int(header["tx_seq"]),
        int(header["plan_seq"]),
        int(header["sample_id"]),
        int(header["source_t0_us"]),
        int(header["tx_time_us"]),
        int(header["coord_mode"]),
        int(header["num_points"]),
        float(header["dt_s"]),
    )
    point_bytes = bytearray()
    for point in packet["points"]:
        point_bytes.extend(
            struct.pack(
                POINT_FORMAT,
                float(point["x_m"]),
                float(point["y_m"]),
                float(point["yaw_rad"]),
                float(point["v_mps"]),
                float(point["curvature"]),
            )
        )
    payload = header_bytes + bytes(point_bytes)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack("<I", crc)


def unpack_packet(data: bytes) -> dict[str, Any]:
    if len(data) < HEADER_SIZE + CRC_SIZE:
        raise ValueError("Packet too small")
    header_tuple = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    magic = header_tuple[0]
    if magic != MAGIC:
        raise ValueError(f"Unexpected magic: {magic!r}")
    num_points = int(header_tuple[9])
    expected = HEADER_SIZE + num_points * POINT_SIZE + CRC_SIZE
    if len(data) != expected:
        raise ValueError(f"Packet size mismatch: expected {expected}, got {len(data)}")
    crc_expected = struct.unpack("<I", data[-CRC_SIZE:])[0]
    crc_actual = zlib.crc32(data[:-CRC_SIZE]) & 0xFFFFFFFF
    if crc_actual != crc_expected:
        raise ValueError(f"CRC mismatch: actual {crc_actual}, expected {crc_expected}")

    points = []
    offset = HEADER_SIZE
    for _ in range(num_points):
        x_m, y_m, yaw_rad, v_mps, curvature = struct.unpack(POINT_FORMAT, data[offset : offset + POINT_SIZE])
        points.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "yaw_rad": yaw_rad,
                "v_mps": v_mps,
                "curvature": curvature,
            }
        )
        offset += POINT_SIZE

    return {
        "header": {
            "magic": magic.decode("ascii"),
            "version": int(header_tuple[1]),
            "flags": int(header_tuple[2]),
            "tx_seq": int(header_tuple[3]),
            "plan_seq": int(header_tuple[4]),
            "sample_id": int(header_tuple[5]),
            "source_t0_us": int(header_tuple[6]),
            "tx_time_us": int(header_tuple[7]),
            "coord_mode": int(header_tuple[8]),
            "coord_mode_name": coord_mode_name(int(header_tuple[8])),
            "num_points": num_points,
            "dt_s": float(header_tuple[10]),
            "crc32": int(crc_expected),
        },
        "points": points,
    }
