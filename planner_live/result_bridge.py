from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

import numpy as np

from scripts.control_team_replay_common import (
    COORD_MODE_LOCAL,
    FLAG_VALID,
    build_packet_dict,
    pack_packet,
    wrap_angles,
)

from .result_parser import PlannerResult


DEFAULT_PLAN_DT_S = 0.1
EPS = 1e-6
ACTION_LOG_DISTANCE_MARKS_M = (1.0, 3.0, 5.0, 7.0, 9.0, 11.0)
ACTION_LOG_DISTANCE_MARK_TOLERANCE_M = 0.35


@dataclass(slots=True)
class UdpBridgeConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5001
    payload_mode: str = "binary_alpa"
    send_mode: str = "periodic"
    rate_hz: float = 50.0
    control_dt_s: float = 0.02
    control_points: int = 16
    full_plan: bool = False
    latency_compensate_full_plan: bool = False
    previous_path_blend_ratio: float = 0.0
    previous_path_blend_max_age_s: float = 3.0
    action_log_interval_s: float = 3.0
    action_log_mode: str = "distance_marks"
    action_log_points: int = 16
    action_log_distance_marks_m: tuple[float, ...] = ACTION_LOG_DISTANCE_MARKS_M
    action_log_distance_mark_tolerance_m: float = ACTION_LOG_DISTANCE_MARK_TOLERANCE_M
    opencv_ui_enabled: bool = False
    opencv_ui_width: int = 900
    opencv_ui_height: int = 700
    opencv_ui_window_name: str = "Alpamayo UDP path"
    opencv_ui_display: str | None = None
    opencv_ui_retry_interval_s: float = 5.0
    origin_offset_x_m: float = 0.0
    origin_offset_y_m: float = 0.0
    origin_yaw_offset_rad: float = 0.0
    path_log_dir: Path | None = None


def _interp_series(values: np.ndarray, src_times_s: np.ndarray, dst_times_s: np.ndarray) -> np.ndarray:
    return np.interp(dst_times_s, src_times_s, values, left=values[0], right=values[-1]).astype(np.float32)


def _yaw_from_rotations(pred_rot: np.ndarray) -> np.ndarray:
    if pred_rot.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if pred_rot.ndim != 3 or pred_rot.shape[1:] != (3, 3):
        raise ValueError(f"pred_rot must have shape [N,3,3], got {pred_rot.shape}")
    return np.arctan2(pred_rot[:, 1, 0], pred_rot[:, 0, 0]).astype(np.float32)


def _result_knots(result: PlannerResult) -> dict[str, np.ndarray | float]:
    pred_xyz = np.asarray(result.pred_xyz, dtype=np.float32)
    pred_rot = np.asarray(result.pred_rot, dtype=np.float32)
    if pred_xyz.ndim != 2 or pred_xyz.shape[1] < 2:
        raise ValueError(f"pred_xyz must have shape [N,>=2], got {pred_xyz.shape}")

    plan_dt_s = float(result.plan_dt_s if result.plan_dt_s is not None else DEFAULT_PLAN_DT_S)
    yaw_future = _yaw_from_rotations(pred_rot)

    times_s = np.arange(pred_xyz.shape[0] + 1, dtype=np.float32) * np.float32(plan_dt_s)
    x = np.concatenate([np.zeros((1,), dtype=np.float32), pred_xyz[:, 0].astype(np.float32)])
    y = np.concatenate([np.zeros((1,), dtype=np.float32), pred_xyz[:, 1].astype(np.float32)])
    yaw = np.concatenate([np.zeros((1,), dtype=np.float32), yaw_future.astype(np.float32)])

    vx = np.gradient(x, times_s).astype(np.float32)
    vy = np.gradient(y, times_s).astype(np.float32)
    v = np.sqrt(vx * vx + vy * vy).astype(np.float32)

    yaw_unwrapped = np.unwrap(yaw.astype(np.float64)).astype(np.float32)
    yaw_rate = np.gradient(yaw_unwrapped, times_s).astype(np.float32)
    curvature = np.where(v > EPS, yaw_rate / np.maximum(v, EPS), 0.0).astype(np.float32)

    return {
        "times_s": times_s,
        "x": x,
        "y": y,
        "yaw_unwrapped": yaw_unwrapped,
        "v": v,
        "curvature": curvature,
        "plan_dt_s": plan_dt_s,
    }


def sample_live_result(
    result: PlannerResult,
    *,
    age_s: float,
    control_dt_s: float,
    control_points: int,
) -> dict[str, np.ndarray]:
    knots = _result_knots(result)
    dst_times_s = np.asarray(age_s, dtype=np.float32) + np.arange(control_points, dtype=np.float32) * np.float32(control_dt_s)
    yaw_unwrapped = _interp_series(knots["yaw_unwrapped"], knots["times_s"], dst_times_s)
    return {
        "x": _interp_series(knots["x"], knots["times_s"], dst_times_s),
        "y": _interp_series(knots["y"], knots["times_s"], dst_times_s),
        "yaw": wrap_angles(yaw_unwrapped),
        "v": _interp_series(knots["v"], knots["times_s"], dst_times_s),
        "curvature": _interp_series(knots["curvature"], knots["times_s"], dst_times_s),
    }


def sample_latency_compensated_full_result(result: PlannerResult, *, age_s: float) -> dict[str, np.ndarray]:
    knots = _result_knots(result)
    plan_dt_s = float(knots["plan_dt_s"])
    times_s = np.asarray(knots["times_s"], dtype=np.float32)
    horizon_s = float(times_s[-1]) if times_s.size else 0.0
    start_s = min(max(float(age_s), 0.0), horizon_s)
    dst_times_s = np.float32(start_s) + np.arange(times_s.shape[0], dtype=np.float32) * np.float32(plan_dt_s)

    x_abs = _interp_series(knots["x"], times_s, dst_times_s).astype(np.float64)
    y_abs = _interp_series(knots["y"], times_s, dst_times_s).astype(np.float64)
    yaw_abs = _interp_series(knots["yaw_unwrapped"], times_s, dst_times_s).astype(np.float64)
    origin_x = float(x_abs[0]) if x_abs.size else 0.0
    origin_y = float(y_abs[0]) if y_abs.size else 0.0
    origin_yaw = float(yaw_abs[0]) if yaw_abs.size else 0.0

    dx = x_abs - origin_x
    dy = y_abs - origin_y
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)

    return {
        "x": (dx * cos_yaw + dy * sin_yaw).astype(np.float32),
        "y": (-dx * sin_yaw + dy * cos_yaw).astype(np.float32),
        "yaw": wrap_angles((yaw_abs - origin_yaw).astype(np.float32)),
        "v": _interp_series(knots["v"], times_s, dst_times_s),
        "curvature": _interp_series(knots["curvature"], times_s, dst_times_s),
        "age_s": np.float32(start_s),
    }


def build_live_result_packet(
    result: PlannerResult,
    *,
    tx_seq: int,
    tx_time_us: int,
    control_dt_s: float,
    control_points: int,
) -> dict[str, Any]:
    age_s = max(0.0, (int(tx_time_us) - int(result.t0_us)) / 1_000_000.0)
    sampled = sample_live_result(
        result,
        age_s=age_s,
        control_dt_s=control_dt_s,
        control_points=control_points,
    )
    return build_packet_dict(
        tx_seq=tx_seq,
        plan_seq=result.sequence,
        sample_id=result.sequence,
        source_t0_us=result.t0_us,
        tx_time_us=tx_time_us,
        coord_mode=COORD_MODE_LOCAL,
        dt_s=control_dt_s,
        x=sampled["x"],
        y=sampled["y"],
        yaw=sampled["yaw"],
        v=sampled["v"],
        curvature=sampled["curvature"],
        flags=FLAG_VALID,
    )


def build_full_result_packet(
    result: PlannerResult,
    *,
    tx_seq: int,
    tx_time_us: int,
    latency_compensate: bool = False,
) -> dict[str, Any]:
    knots = _result_knots(result)
    if latency_compensate:
        age_s = max(0.0, (int(tx_time_us) - int(result.t0_us)) / 1_000_000.0)
        sampled = sample_latency_compensated_full_result(result, age_s=age_s)
        packet = build_packet_dict(
            tx_seq=tx_seq,
            plan_seq=result.sequence,
            sample_id=result.sequence,
            source_t0_us=result.t0_us,
            tx_time_us=tx_time_us,
            coord_mode=COORD_MODE_LOCAL,
            dt_s=float(knots["plan_dt_s"]),
            x=sampled["x"],
            y=sampled["y"],
            yaw=sampled["yaw"],
            v=sampled["v"],
            curvature=sampled["curvature"],
            flags=FLAG_VALID,
        )
        packet["header"]["latency_compensated_full_plan"] = True
        packet["header"]["latency_compensation_age_s"] = float(sampled["age_s"])
        return packet
    return build_packet_dict(
        tx_seq=tx_seq,
        plan_seq=result.sequence,
        sample_id=result.sequence,
        source_t0_us=result.t0_us,
        tx_time_us=tx_time_us,
        coord_mode=COORD_MODE_LOCAL,
        dt_s=float(knots["plan_dt_s"]),
        x=np.asarray(knots["x"], dtype=np.float32),
        y=np.asarray(knots["y"], dtype=np.float32),
        yaw=wrap_angles(np.asarray(knots["yaw_unwrapped"], dtype=np.float32)),
        v=np.asarray(knots["v"], dtype=np.float32),
        curvature=np.asarray(knots["curvature"], dtype=np.float32),
        flags=FLAG_VALID,
    )


def _copy_packet(packet: dict[str, Any]) -> dict[str, Any]:
    copied = dict(packet)
    copied["header"] = dict(packet.get("header") or {})
    copied["points"] = [dict(point) for point in packet.get("points") or []]
    return copied


def _wrap_angle_delta_rad(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _previous_path_relative_points(
    previous_packet: dict[str, Any],
    *,
    current_tx_time_us: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    previous_header = dict(previous_packet.get("header") or {})
    previous_points = [dict(point) for point in previous_packet.get("points") or []]
    if not previous_points:
        return [], {"reason": "previous packet has no points"}

    dt_s = max(float(previous_header.get("dt_s") or DEFAULT_PLAN_DT_S), EPS)
    elapsed_s = max(0.0, (int(current_tx_time_us) - int(previous_header.get("tx_time_us", 0))) / 1_000_000.0)
    shift_points = min(max(int(round(elapsed_s / dt_s)), 0), len(previous_points) - 1)
    origin = previous_points[shift_points]
    origin_x = float(origin.get("x_m", 0.0))
    origin_y = float(origin.get("y_m", 0.0))
    origin_yaw = float(origin.get("yaw_rad", 0.0))
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)

    relative_points: list[dict[str, float]] = []
    for idx in range(len(previous_points)):
        source_idx = min(shift_points + idx, len(previous_points) - 1)
        source = previous_points[source_idx]
        dx = float(source.get("x_m", 0.0)) - origin_x
        dy = float(source.get("y_m", 0.0)) - origin_y
        relative_points.append(
            {
                "x_m": dx * cos_yaw + dy * sin_yaw,
                "y_m": -dx * sin_yaw + dy * cos_yaw,
                "yaw_rad": _wrap_angle_delta_rad(float(source.get("yaw_rad", 0.0)) - origin_yaw),
                "v_mps": float(source.get("v_mps", 0.0)),
                "curvature": float(source.get("curvature", 0.0)),
            }
        )

    return relative_points, {
        "elapsed_s": elapsed_s,
        "shift_points": shift_points,
        "previous_plan_seq": previous_header.get("plan_seq"),
        "previous_tx_seq": previous_header.get("tx_seq"),
        "previous_tx_time_us": previous_header.get("tx_time_us"),
    }


def blend_with_previous_full_plan_packet(
    current_packet: dict[str, Any],
    previous_packet: dict[str, Any] | None,
    *,
    ratio: float,
    max_age_s: float,
) -> dict[str, Any]:
    blend_ratio = min(max(float(ratio), 0.0), 1.0)
    if blend_ratio <= 0.0:
        return current_packet

    blended_packet = _copy_packet(current_packet)
    header = dict(blended_packet.get("header") or {})
    header["previous_path_blend_ratio"] = blend_ratio
    header["previous_path_blend_max_age_s"] = float(max_age_s)

    if previous_packet is None:
        header["previous_path_blend_applied"] = False
        header["previous_path_blend_reason"] = "no_previous_packet"
        blended_packet["header"] = header
        return blended_packet

    current_points = [dict(point) for point in blended_packet.get("points") or []]
    previous_relative, previous_meta = _previous_path_relative_points(
        previous_packet,
        current_tx_time_us=int(header.get("tx_time_us", 0)),
    )
    elapsed_s = float(previous_meta.get("elapsed_s", 0.0))
    if elapsed_s > float(max_age_s):
        header["previous_path_blend_applied"] = False
        header["previous_path_blend_reason"] = "previous_packet_too_old"
        header["previous_path_blend_elapsed_s"] = elapsed_s
        blended_packet["header"] = header
        return blended_packet
    if not current_points or not previous_relative:
        header["previous_path_blend_applied"] = False
        header["previous_path_blend_reason"] = "missing_points"
        blended_packet["header"] = header
        return blended_packet

    count = min(len(current_points), len(previous_relative))
    blended_points: list[dict[str, Any]] = []
    for idx, current in enumerate(current_points):
        if idx >= count:
            blended_points.append(current)
            continue
        previous = previous_relative[idx]
        blended = dict(current)
        blended["x_m"] = (1.0 - blend_ratio) * float(current.get("x_m", 0.0)) + blend_ratio * float(
            previous.get("x_m", 0.0)
        )
        blended["y_m"] = (1.0 - blend_ratio) * float(current.get("y_m", 0.0)) + blend_ratio * float(
            previous.get("y_m", 0.0)
        )
        current_yaw = float(current.get("yaw_rad", 0.0))
        previous_yaw = float(previous.get("yaw_rad", 0.0))
        blended["yaw_rad"] = _wrap_angle_delta_rad(
            current_yaw + blend_ratio * _wrap_angle_delta_rad(previous_yaw - current_yaw)
        )
        blended["v_mps"] = (1.0 - blend_ratio) * float(current.get("v_mps", 0.0)) + blend_ratio * float(
            previous.get("v_mps", 0.0)
        )
        blended["curvature"] = (1.0 - blend_ratio) * float(
            current.get("curvature", 0.0)
        ) + blend_ratio * float(previous.get("curvature", 0.0))
        blended_points.append(blended)

    header["previous_path_blend_applied"] = True
    header["previous_path_blend_elapsed_s"] = elapsed_s
    header["previous_path_blend_shift_points"] = int(previous_meta.get("shift_points", 0))
    header["previous_path_blend_previous_plan_seq"] = previous_meta.get("previous_plan_seq")
    header["previous_path_blend_previous_tx_seq"] = previous_meta.get("previous_tx_seq")
    header["previous_path_blend_previous_tx_time_us"] = previous_meta.get("previous_tx_time_us")
    blended_packet["header"] = header
    blended_packet["points"] = blended_points
    return blended_packet


def _inference_time_s(result: PlannerResult) -> float | None:
    total_post_vlm_ms = result.post_vlm_timing.get("total_post_vlm_ms")
    if total_post_vlm_ms is not None:
        return float(total_post_vlm_ms) / 1000.0
    fm_wall_ms = result.fm_timing.get("wall_ms")
    if fm_wall_ms is not None:
        return float(fm_wall_ms) / 1000.0
    return None


def _packet_age_s(packet: dict[str, Any]) -> float:
    header = packet.get("header", {})
    tx_time_us = int(header.get("tx_time_us", 0))
    source_t0_us = int(header.get("source_t0_us", 0))
    return max(0.0, (tx_time_us - source_t0_us) / 1_000_000.0)


def _update_udp_latency_timing(result: PlannerResult, *, packet: dict[str, Any]) -> None:
    actual_offset_ms = _packet_age_s(packet) * 1000.0
    timing = dict(getattr(result, "live_timing", {}) or {})
    timing["udp_actual_offset_ms"] = actual_offset_ms
    inference_time_s = _inference_time_s(result)
    if inference_time_s is not None:
        reported_inference_ms = float(inference_time_s) * 1000.0
        timing["udp_reported_inference_ms"] = reported_inference_ms
        timing["udp_actual_minus_reported_inference_ms"] = actual_offset_ms - reported_inference_ms
    result.live_timing = timing


def _format_latency_breakdown(result: PlannerResult, packet: dict[str, Any]) -> list[str]:
    timing = dict(getattr(result, "live_timing", {}) or {})
    actual_offset_ms = _packet_age_s(packet) * 1000.0
    inference_time_s = _inference_time_s(result)
    lines: list[str] = []
    if inference_time_s is not None:
        reported_inference_ms = float(inference_time_s) * 1000.0
        lines.append(
            "  latency "
            f"actual={actual_offset_ms:.1f}ms "
            f"reported_infer={reported_inference_ms:.1f}ms "
            f"other={actual_offset_ms - reported_inference_ms:.1f}ms"
        )
    else:
        lines.append(f"  latency actual={actual_offset_ms:.1f}ms reported_infer=NA")

    ordered_keys = (
        ("sample_age_at_fetch_done_ms", "sample_age_fetch"),
        ("fetch_latest_sample_ms", "fetch"),
        ("apply_gnss_history_ms", "gnss_hist"),
        ("validate_live_sample_ms", "validate"),
        ("receiver_to_process_start_ms", "queue"),
        ("staging_cleanup_ms", "cleanup"),
        ("write_sample_files_ms", "write_images_npy"),
        ("build_runtime_request_ms", "build_req"),
        ("write_latest_request_ms", "latest_req"),
        ("runtime_request_wall_ms", "runtime_wall"),
        ("runtime_wall_minus_reported_post_vlm_ms", "runtime_unreported"),
        ("write_latest_output_ms", "latest_out"),
        ("extract_trajectory_artifacts_ms", "extract_traj"),
        ("parse_planner_result_ms", "parse"),
        ("serialize_result_json_ms", "json_ser"),
        ("write_result_json_ms", "json_write"),
    )
    parts = []
    for key, label in ordered_keys:
        value = timing.get(key)
        if value is None:
            continue
        try:
            parts.append(f"{label}={float(value):.1f}ms")
        except Exception:
            continue
    if parts:
        lines.append("  breakdown " + " ".join(parts))
    return lines


def _packet_starts_at_origin(points: list[dict[str, Any]]) -> bool:
    if not points:
        return False
    first = points[0]
    return abs(float(first.get("x_m", 0.0))) <= EPS and abs(float(first.get("y_m", 0.0))) <= EPS


def apply_origin_offset_to_packet(
    packet: dict[str, Any],
    *,
    offset_x_m: float,
    offset_y_m: float,
    yaw_offset_rad: float,
) -> dict[str, Any]:
    """Convert rear/local-origin path points into the control sensor frame.

    Positive x offset means the control origin is ahead of the current ego
    origin, so outgoing x is shifted backward. Positive y offset follows our
    local convention (+y left), so it is added to compensate a sensor mounted
    to the vehicle right. Positive yaw offset means the control frame is
    rotated left relative to the current ego frame.
    """
    offset_x = float(offset_x_m)
    offset_y = float(offset_y_m)
    yaw_offset = float(yaw_offset_rad)
    if abs(offset_x) <= EPS and abs(offset_y) <= EPS and abs(yaw_offset) <= EPS:
        return packet
    original_points = list(packet.get("points") or [])
    cos_yaw = math.cos(yaw_offset)
    sin_yaw = math.sin(yaw_offset)
    header = dict(packet.get("header") or {})
    header["has_origin_point"] = _packet_starts_at_origin(original_points)
    header["origin_offset_x_m"] = offset_x
    header["origin_offset_y_m"] = offset_y
    header["origin_yaw_offset_rad"] = yaw_offset
    header["origin_yaw_offset_deg"] = math.degrees(yaw_offset)
    header["origin_offset_convention"] = (
        "translate first: x'=x-offset_x_m, y'=y+offset_y_m; "
        "then rotate by -origin_yaw_offset_rad into control frame"
    )
    shifted_points: list[dict[str, Any]] = []
    for point in original_points:
        translated_x = float(point.get("x_m", 0.0)) - offset_x
        translated_y = float(point.get("y_m", 0.0)) + offset_y
        shifted = dict(point)
        shifted["x_m"] = translated_x * cos_yaw + translated_y * sin_yaw
        shifted["y_m"] = -translated_x * sin_yaw + translated_y * cos_yaw
        shifted["yaw_rad"] = float(
            wrap_angles(np.asarray([float(point.get("yaw_rad", 0.0)) - yaw_offset], dtype=np.float32))[0]
        )
        shifted_points.append(shifted)
    shifted_packet = dict(packet)
    shifted_packet["header"] = header
    shifted_packet["points"] = shifted_points
    return shifted_packet


def _pred_xyz_from_packet_points(points: list[dict[str, Any]]) -> list[list[float]]:
    return [
        [
            float(point.get("x_m", 0.0)),
            float(point.get("y_m", 0.0)),
            0.0,
        ]
        for point in points
    ]


def _integrate_velocity_from_accel(accel: np.ndarray, *, initial_speed_mps: float | None, dt_s: float) -> np.ndarray:
    velocity = np.zeros((accel.shape[0],), dtype=np.float32)
    current_v = max(float(initial_speed_mps or 0.0), 0.0)
    safe_dt_s = max(float(dt_s), EPS)
    for idx, accel_value in enumerate(accel):
        current_v = max(0.0, current_v + float(accel_value) * safe_dt_s)
        velocity[idx] = np.float32(current_v)
    return velocity


def _decode_raw_action(
    result: PlannerResult,
    *,
    initial_speed_mps: float | None = None,
    dt_s: float | None = None,
) -> dict[str, Any] | None:
    x_final = np.asarray(result.x_final, dtype=np.float32)
    if x_final.ndim != 2 or x_final.shape[1] < 2 or x_final.shape[0] == 0:
        return None
    constants = dict(result.action_space_constants or {})
    try:
        accel_mean = float(constants["accel_mean"])
        accel_std = float(constants["accel_std"])
        curvature_mean = float(constants["curvature_mean"])
        curvature_std = float(constants["curvature_std"])
    except KeyError:
        return None

    raw_accel = x_final[:, 0] * np.float32(accel_std) + np.float32(accel_mean)
    # UDP control team convention currently expects curvature with the opposite
    # sign from the FM/Alpamayo decoded action convention.
    curvature = -(x_final[:, 1] * np.float32(curvature_std) + np.float32(curvature_mean))
    action_dt_s = float(dt_s if dt_s is not None else constants.get("dt_value", result.plan_dt_s or DEFAULT_PLAN_DT_S))
    velocity_mps = _integrate_velocity_from_accel(
        raw_accel,
        initial_speed_mps=initial_speed_mps,
        dt_s=action_dt_s,
    )
    return {
        "num_points": int(x_final.shape[0]),
        "normalized_x_final": x_final.astype(np.float32).tolist(),
        # Kept for control-team schema compatibility: this field now carries velocity [m/s].
        "accel_mps2": velocity_mps.astype(np.float32).tolist(),
        "accel_mps2_units": "m/s",
        "raw_accel_mps2": raw_accel.astype(np.float32).tolist(),
        "curvature": curvature.astype(np.float32).tolist(),
        "action_space_constants": constants,
    }


def build_text_result_payload(result: PlannerResult, packet: dict[str, Any]) -> dict[str, Any]:
    points = list(packet["points"])
    initial_speed_mps = float(points[0]["v_mps"]) if points else None
    dt_s = float(packet["header"]["dt_s"])
    actual_offset_s = _packet_age_s(packet)
    packet_has_origin = bool(packet.get("header", {}).get("has_origin_point", _packet_starts_at_origin(points)))
    future_points = points[1:] if packet_has_origin else points
    pred_xyz = _pred_xyz_from_packet_points(future_points)
    raw_curvature = [-float(point["curvature"]) for point in future_points]
    raw_action = _decode_raw_action(result, initial_speed_mps=initial_speed_mps, dt_s=dt_s)
    if raw_action is None:
        raw_action = {
            "num_points": len(future_points),
            "normalized_x_final": [],
            "accel_mps2": [float(point["v_mps"]) for point in future_points],
            "accel_mps2_units": "m/s",
            "raw_accel_mps2": [],
            "curvature": raw_curvature,
            "action_space_constants": {"dt_value": dt_s},
            "source": "fallback_path_velocity",
        }
    else:
        raw_action["source"] = "fm_x_final"
    return {
        "label": "ac_decoded_path",
        "chunk_id": 0,
        "front_frame_id": result.sequence,
        "t0_utc_ns": int(result.t0_us) * 1000,
        "target_offset_s": actual_offset_s,
        "actual_offset_s": actual_offset_s,
        "path_type": "ac_decoded_path",
        "clip_id": result.clip_id,
        "sample_id": result.sequence,
        "t0_us": result.t0_us,
        "final_output": result.output_text,
        "timing": dict(result.post_vlm_timing),
        "plan_dt_s": dt_s,
        "plan_points_no_origin": len(future_points),
        "traj_points_with_origin": len(points),
        "pred_xyz": pred_xyz,
        "pred_xyz_source": "packet_points_xy",
        "packet_points_include_origin": packet_has_origin,
        "model_pred_xyz": result.pred_xyz,
        "pred_yaw_rad": [float(point["yaw_rad"]) for point in future_points],
        "pred_v_mps": [float(point["v_mps"]) for point in future_points],
        "pred_curvature": raw_curvature,
        "packet_header": dict(packet["header"]),
        "packet_points": points,
        "raw_action": raw_action,
        "initial_speed_mps": initial_speed_mps,
        "initial_speed_kph": float(initial_speed_mps * 3.6) if initial_speed_mps is not None else None,
        "initial_speed_source": "packet_points[0].v_mps",
        "sensor_speed_mps": None,
        "sensor_speed_kph": None,
        "sensor_speed_source": "planner_live_unavailable",
        "sensor_speed_t0_utc_ns": None,
        "inference_time_s": _inference_time_s(result),
        "live_timing": dict(getattr(result, "live_timing", {}) or {}),
        "udp_mode": "text_json_live",
    }


def build_udp_path_log_record(
    result: PlannerResult,
    packet: dict[str, Any],
    *,
    target: tuple[str, int],
    payload_mode: str,
    actual_offset_s: float,
    bytes_len: int,
    logged_at_unix: float | None = None,
    logged_at_utc_ns: int | None = None,
) -> dict[str, Any]:
    logged_ns = int(logged_at_utc_ns if logged_at_utc_ns is not None else time.time_ns())
    logged_unix = float(logged_at_unix if logged_at_unix is not None else logged_ns / 1_000_000_000.0)
    record = build_text_result_payload(result, packet)
    record.update(
        {
            "log_type": "udp_sent_path",
            "path_log_schema_version": 1,
            "logged_at_unix": logged_unix,
            "logged_at_utc_ns": logged_ns,
            "target": {"host": str(target[0]), "port": int(target[1])},
            "payload_mode": str(payload_mode),
            "payload_bytes": int(bytes_len),
            "actual_offset_s": float(actual_offset_s),
            "result_completed_at_unix": float(result.completed_at_unix),
            "output_json_path": result.output_json_path,
            "gt_future_available_at_send_time": False,
            "gt_alignment_hint": (
                "Record GNSS/INS separately during the drive and align offline using "
                "packet_header.tx_time_us or packet_header.source_t0_us."
            ),
        }
    )
    return record


def _estimate_accel_from_points(points: list[dict[str, Any]], dt_s: float) -> list[float]:
    if not points:
        return []
    velocities = [float(point.get("v_mps", 0.0)) for point in points]
    if len(velocities) == 1:
        return [0.0]
    accel: list[float] = []
    safe_dt_s = max(float(dt_s), EPS)
    for idx, velocity in enumerate(velocities):
        if idx == 0:
            value = (velocities[1] - velocities[0]) / safe_dt_s
        elif idx == len(velocities) - 1:
            value = (velocities[-1] - velocities[-2]) / safe_dt_s
        else:
            value = (velocities[idx + 1] - velocities[idx - 1]) / (2.0 * safe_dt_s)
        accel.append(float(value))
    return accel


def _packet_future_points(packet: dict[str, Any]) -> list[dict[str, Any]]:
    packet_points = list(packet.get("points") or [])
    packet_has_origin = bool(packet.get("header", {}).get("has_origin_point", _packet_starts_at_origin(packet_points)))
    return packet_points[1:] if packet_has_origin else packet_points


def _format_action_points_preview(result: PlannerResult, packet: dict[str, Any], points: int) -> str:
    packet_points = list(packet.get("points") or [])
    future_points = _packet_future_points(packet)
    limit = min(max(int(points), 0), len(future_points))
    dt_s = float(packet.get("header", {}).get("dt_s", result.plan_dt_s or DEFAULT_PLAN_DT_S))
    initial_speed_mps = float(packet_points[0].get("v_mps", 0.0)) if packet_points else None
    raw_action = _decode_raw_action(result, initial_speed_mps=initial_speed_mps, dt_s=dt_s)
    accel = _estimate_accel_from_points(future_points, dt_s)
    curvature = [-float(point.get("curvature", 0.0)) for point in future_points]
    action_source = "path-derived"
    if raw_action is not None:
        raw_velocity = list(raw_action.get("accel_mps2") or [])
        raw_curvature = list(raw_action.get("curvature") or [])
        if raw_velocity and raw_curvature:
            accel = [float(v) for v in raw_velocity]
            curvature = [float(v) for v in raw_curvature]
            limit = min(limit, len(accel), len(curvature))
            action_source = "fm_x_final"

    rows = []
    for idx in range(limit):
        point = future_points[idx]
        rows.append(
            "  "
            f"{idx:02d} "
            f"v_raw={accel[idx]:6.3f}m/s "
            f"curv={curvature[idx]:+9.6f}1/m "
            f"v_path={float(point.get('v_mps', 0.0)):6.3f}m/s "
            f"x={float(point.get('x_m', 0.0)):7.3f} "
            f"y={float(point.get('y_m', 0.0)):7.3f}"
        )
    return "\n".join(
        [
            "[udp action preview]",
            f"  seq={result.sequence} clip={result.clip_id} dt={dt_s:.3f}s source={action_source} points={limit}/{len(future_points)}",
            *_format_latency_breakdown(result, packet),
            *rows,
        ]
    )


def _path_samples_with_distance(packet: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    future_points = _packet_future_points(packet)
    path_points = [{"x_m": 0.0, "y_m": 0.0}, *future_points]

    if len(path_points) <= 1:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
            max(len(path_points) - 1, 0),
        )

    xs_all = np.asarray([float(point.get("x_m", 0.0)) for point in path_points], dtype=np.float64)
    ys_all = np.asarray([float(point.get("y_m", 0.0)) for point in path_points], dtype=np.float64)
    segment_lengths = np.hypot(np.diff(xs_all), np.diff(ys_all))
    distances = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths)])

    sample_distances = distances[1:]
    sample_xs = xs_all[1:]
    sample_ys = ys_all[1:]
    sample_indices = np.arange(len(future_points), dtype=np.int64)
    finite = np.isfinite(sample_distances) & np.isfinite(sample_xs) & np.isfinite(sample_ys)
    return sample_distances[finite], sample_xs[finite], sample_ys[finite], sample_indices[finite], len(path_points) - 1


def _select_nearest_distance_mark_point(
    distances_m: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    indices: np.ndarray,
    mark_m: float,
    tolerance_m: float,
) -> tuple[float, float, float, int] | None:
    if distances_m.size == 0:
        return None
    deltas = np.abs(distances_m - float(mark_m))
    min_delta = float(np.min(deltas))
    if min_delta > float(tolerance_m):
        return None
    candidate_indices = np.flatnonzero(np.isclose(deltas, min_delta, rtol=0.0, atol=1e-6))
    if candidate_indices.size > 1:
        lower_or_equal = candidate_indices[distances_m[candidate_indices] <= float(mark_m) + 1e-6]
        if lower_or_equal.size:
            idx = int(lower_or_equal[np.argmax(distances_m[lower_or_equal])])
        else:
            idx = int(candidate_indices[np.argmin(distances_m[candidate_indices])])
    else:
        idx = int(candidate_indices[0])
    return float(distances_m[idx]), float(xs[idx]), float(ys[idx]), int(indices[idx])


def _action_values_for_future_points(
    result: PlannerResult,
    *,
    packet_points: list[dict[str, Any]],
    future_points: list[dict[str, Any]],
    dt_s: float,
) -> tuple[list[float], list[float], str]:
    initial_speed_mps = float(packet_points[0].get("v_mps", 0.0)) if packet_points else None
    accel = _estimate_accel_from_points(future_points, dt_s)
    curvature = [-float(point.get("curvature", 0.0)) for point in future_points]
    raw_action = _decode_raw_action(result, initial_speed_mps=initial_speed_mps, dt_s=dt_s)
    if raw_action is None:
        return accel, curvature, "path-derived"

    raw_accel = list(raw_action.get("raw_accel_mps2") or [])
    raw_curvature = list(raw_action.get("curvature") or [])
    if raw_accel and raw_curvature:
        return [float(value) for value in raw_accel], [float(value) for value in raw_curvature], "fm_x_final"
    return accel, curvature, "path-derived"


def _format_action_distance_marks_preview(
    result: PlannerResult,
    packet: dict[str, Any],
    distance_marks_m: tuple[float, ...],
    tolerance_m: float,
) -> str:
    packet_points = list(packet.get("points") or [])
    future_points = _packet_future_points(packet)
    dt_s = float(packet.get("header", {}).get("dt_s", result.plan_dt_s or DEFAULT_PLAN_DT_S))
    accel, curvature, action_source = _action_values_for_future_points(
        result,
        packet_points=packet_points,
        future_points=future_points,
        dt_s=dt_s,
    )

    distances_m, xs, ys, indices, path_point_count = _path_samples_with_distance(packet)
    rows = []
    for mark_m in distance_marks_m:
        selected = _select_nearest_distance_mark_point(distances_m, xs, ys, indices, mark_m, tolerance_m)
        if selected is None:
            rows.append(f"  target_s={mark_m:4.1f}m -> NULL")
        else:
            distance_m, x_m, y_m, point_idx = selected
            accel_text = f"{accel[point_idx]:+7.3f}" if point_idx < len(accel) else "   NULL"
            curvature_text = f"{curvature[point_idx]:+9.6f}" if point_idx < len(curvature) else "     NULL"
            rows.append(
                f"  target_s={mark_m:4.1f}m -> "
                f"s={distance_m:7.3f} x={x_m:7.3f} y={y_m:7.3f} "
                f"a={accel_text}m/s2 c={curvature_text}1/m"
            )
    return "\n".join(
        [
            "[udp action preview]",
            (
                f"  seq={result.sequence} clip={result.clip_id} dt={dt_s:.3f}s "
                f"source={action_source} distance_marks_m={list(distance_marks_m)} "
                f"tolerance_m={float(tolerance_m):.3f} path_points={path_point_count}"
            ),
            *_format_latency_breakdown(result, packet),
            *rows,
        ]
    )


def _format_action_preview(
    result: PlannerResult,
    packet: dict[str, Any],
    *,
    mode: str,
    points: int,
    distance_marks_m: tuple[float, ...],
    distance_mark_tolerance_m: float,
) -> str:
    if mode == "points":
        return _format_action_points_preview(result, packet, points)
    if mode in {"distance_marks", "x_marks"}:
        return _format_action_distance_marks_preview(
            result,
            packet,
            distance_marks_m,
            distance_mark_tolerance_m,
        )
    return f"[udp action preview]\n  seq={result.sequence} clip={result.clip_id} unsupported action_log_mode={mode!r}"


def encode_udp_payload(result: PlannerResult, packet: dict[str, Any], payload_mode: str) -> bytes:
    if payload_mode == "binary_alpa":
        return pack_packet(packet)
    if payload_mode == "text_json":
        payload = build_text_result_payload(result, packet)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raise ValueError(f"unsupported udp payload mode: {payload_mode}")


def _display_from_x_socket(path: str | None) -> str | None:
    if not path:
        return None
    name = os.path.basename(path)
    if not name.startswith("X"):
        return None
    number = name[1:]
    return f":{number}" if number.isdigit() else None


def _x_display_preflight_error(display: str | None) -> str | None:
    if not display:
        return "DISPLAY is not set; pass --udp-opencv-ui-display :N or export DISPLAY=:N before enabling OpenCV UI"
    lib_name = ctypes.util.find_library("X11")
    if not lib_name:
        return "libX11 is unavailable, so OpenCV Qt cannot be checked safely"
    try:
        x11 = ctypes.cdll.LoadLibrary(lib_name)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        ptr = x11.XOpenDisplay(display.encode("utf-8"))
        if not ptr:
            return f"cannot connect to X display {display!r}; check DISPLAY/Xauthority or use --udp-opencv-ui-display"
        x11.XCloseDisplay(ptr)
    except Exception as exc:
        return f"X display preflight failed for {display!r}: {exc}"
    return None


def _available_x_socket_displays() -> list[str]:
    root = "/tmp/.X11-unix"
    if not os.path.isdir(root):
        return []
    displays: list[str] = []
    for name in os.listdir(root):
        display = _display_from_x_socket(name)
        if display is not None:
            displays.append(display)
    return sorted(set(displays), key=lambda item: int(item[1:]) if item[1:].isdigit() else -1, reverse=True)


def _resolve_opencv_display(explicit_display: str | None) -> str:
    candidates: list[str | None] = [
        explicit_display,
        os.environ.get("DISPLAY"),
        _display_from_x_socket(os.environ.get("REMOTE_CONTAINERS_DISPLAY_SOCK")),
    ]
    candidates.extend(_available_x_socket_displays())
    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        error = _x_display_preflight_error(candidate)
        if error is None:
            return candidate
        errors.append(error)
    detail = "; ".join(errors[-5:]) if errors else "no DISPLAY or X socket candidates found"
    raise RuntimeError(f"no usable X display for OpenCV UI: {detail}")


class UdpPathOpenCvUi:
    def __init__(self, *, width: int, height: int, window_name: str, display: str | None = None) -> None:
        resolved_display = _resolve_opencv_display(display)
        os.environ["DISPLAY"] = resolved_display
        os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
        import cv2  # type: ignore

        self.cv2 = cv2
        build_info = cv2.getBuildInformation()
        if "GUI:                           NONE" in build_info or "GUI: NONE" in build_info:
            raise RuntimeError("OpenCV GUI is unavailable; install a GUI-enabled opencv-python build.")
        self.width = int(width)
        self.height = int(height)
        self.window_name = str(window_name)
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(self.window_name, self.width, self.height)
        print(
            f"[udp opencv ui] opened window={self.window_name!r} "
            f"display={resolved_display!r} size={self.width}x{self.height}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def update(
        self,
        *,
        result: PlannerResult,
        packet: dict[str, Any],
        target: tuple[str, int],
        payload_mode: str,
        actual_offset_s: float,
        bytes_len: int,
    ) -> bool:
        cv2 = self.cv2
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (18, 20, 24)
        panel = (56, 72, self.width - 56, self.height - 150)
        origin_px = np.asarray([(panel[0] + panel[2]) // 2, panel[3] - 42], dtype=np.int32)
        points = list(packet.get("points") or [])
        xy = np.asarray([[float(point["x_m"]), float(point["y_m"])] for point in points], dtype=np.float64)
        max_x = max(20.0, float(np.max(xy[:, 0])) + 2.0 if len(xy) else 20.0)
        lateral = max(10.0, float(np.max(np.abs(xy[:, 1]))) + 2.0 if len(xy) else 10.0)
        sx = (panel[2] - panel[0] - 60) / (2.0 * lateral)
        sy = (panel[3] - panel[1] - 70) / max_x
        scale = max(min(sx, sy), 1.0)

        def local_to_px(local_xy: np.ndarray) -> np.ndarray:
            out = np.empty_like(local_xy, dtype=np.int32)
            out[:, 0] = np.round(origin_px[0] - local_xy[:, 1] * scale).astype(np.int32)
            out[:, 1] = np.round(origin_px[1] - local_xy[:, 0] * scale).astype(np.int32)
            return out

        cv2.putText(img, "ALPAMAYO UDP PATH", (56, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.rectangle(img, (panel[0], panel[1]), (panel[2], panel[3]), (72, 72, 72), 1)
        axis_color = (225, 225, 225)
        grid_color = (78, 78, 78)
        cv2.line(img, (origin_px[0], panel[1] + 20), (origin_px[0], panel[3] - 20), grid_color, 1)
        cv2.line(img, (panel[0] + 20, origin_px[1]), (panel[2] - 20, origin_px[1]), grid_color, 1)
        cv2.arrowedLine(img, tuple(origin_px), (origin_px[0], panel[1] + 26), axis_color, 2, cv2.LINE_AA, tipLength=0.04)
        cv2.putText(img, "+x forward", (origin_px[0] + 10, panel[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, axis_color, 1, cv2.LINE_AA)
        cv2.putText(img, "+y left", (panel[0] + 26, origin_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, axis_color, 1, cv2.LINE_AA)
        for x_m in (4.0, 8.0, 16.0):
            y_px = int(round(origin_px[1] - x_m * scale))
            if panel[1] + 20 <= y_px <= panel[3] - 20:
                cv2.line(img, (panel[0] + 20, y_px), (panel[2] - 20, y_px), (58, 58, 58), 1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{int(x_m)}m",
                    (origin_px[0] + 10, y_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (205, 205, 205),
                    1,
                    cv2.LINE_AA,
                )
        for y_m in np.arange(-8.0, 8.0 + 0.5, 0.5):
            if abs(float(y_m)) <= EPS:
                continue
            x_px = int(round(origin_px[0] - y_m * scale))
            if panel[0] + 20 <= x_px <= panel[2] - 20:
                is_meter = abs(float(y_m) - round(float(y_m))) < 1e-6
                color = (68, 68, 68) if is_meter else (44, 44, 44)
                cv2.line(img, (x_px, panel[1] + 20), (x_px, panel[3] - 20), color, 1, cv2.LINE_AA)
                if is_meter:
                    cv2.putText(
                        img,
                        f"{y_m:+.0f}m",
                        (x_px - 22, origin_px[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (205, 205, 205),
                        1,
                        cv2.LINE_AA,
                    )

        if len(xy) >= 2:
            path_px = local_to_px(xy)
            cv2.polylines(img, [path_px.reshape((-1, 1, 2))], False, (70, 230, 120), 3, cv2.LINE_AA)
            for point_px in path_px[:: max(len(path_px) // 16, 1)]:
                cv2.circle(img, tuple(point_px), 3, (90, 255, 160), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(path_px[-1]), 7, (40, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "END", tuple(path_px[-1] + np.asarray([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 210, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(origin_px), 7, (45, 85, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(origin_px), 10, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(img, "ego", tuple(origin_px + np.asarray([10, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        header = packet.get("header", {})
        inference_time_s = _inference_time_s(result)
        texts = [
            f"target={target[0]}:{target[1]} mode={payload_mode} bytes={bytes_len}",
            f"seq={header.get('plan_seq')} tx={header.get('tx_seq')} points={header.get('num_points')} dt={float(header.get('dt_s', 0.0)):.3f}s full_plan={len(points) > 16}",
            f"actual_offset={actual_offset_s:.3f}s inference_time={inference_time_s:.3f}s" if inference_time_s is not None else f"actual_offset={actual_offset_s:.3f}s inference_time=NA",
            f"clip={result.clip_id} output={result.output_text or ''}",
            "sent path: packet_points x_m/y_m",
        ]
        y = self.height - 118
        for text in texts:
            cv2.putText(img, text[:140], (56, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            y += 22
        cv2.putText(img, "Press q or ESC to close visualization", (self.width - 310, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(30) & 0xFF
        return key not in (27, ord("q"))


@dataclass(slots=True)
class UdpPathUiFrame:
    result: PlannerResult
    packet: dict[str, Any]
    target: tuple[str, int]
    payload_mode: str
    actual_offset_s: float
    bytes_len: int


class UdpPathOpenCvUiWorker:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        window_name: str,
        display: str | None,
        retry_interval_s: float,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.window_name = str(window_name)
        self.display = display
        self.retry_interval_s = max(float(retry_interval_s), 0.5)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="planner-udp-opencv-ui", daemon=True)
        self._latest_frame: UdpPathUiFrame | None = None
        self._opencv_ui: UdpPathOpenCvUi | None = None
        self._last_error: str | None = None
        self._next_retry_unix = 0.0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def submit(
        self,
        *,
        result: PlannerResult,
        packet: dict[str, Any],
        target: tuple[str, int],
        payload_mode: str,
        actual_offset_s: float,
        bytes_len: int,
    ) -> None:
        with self._lock:
            self._latest_frame = UdpPathUiFrame(
                result=result,
                packet=packet,
                target=target,
                payload_mode=payload_mode,
                actual_offset_s=actual_offset_s,
                bytes_len=bytes_len,
            )
        self._event.set()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                with self._lock:
                    frame = self._latest_frame
                    self._latest_frame = None
                    self._event.clear()
                if frame is None:
                    continue
                self._update(frame)
        finally:
            if self._opencv_ui is not None:
                self._opencv_ui.close()
                self._opencv_ui = None

    def _update(self, frame: UdpPathUiFrame) -> None:
        now = time.time()
        if self._opencv_ui is None and now < self._next_retry_unix:
            return
        try:
            if self._opencv_ui is None:
                self._opencv_ui = UdpPathOpenCvUi(
                    width=self.width,
                    height=self.height,
                    window_name=self.window_name,
                    display=self.display,
                )
            keep_open = self._opencv_ui.update(
                result=frame.result,
                packet=frame.packet,
                target=frame.target,
                payload_mode=frame.payload_mode,
                actual_offset_s=frame.actual_offset_s,
                bytes_len=frame.bytes_len,
            )
            with self._lock:
                self._last_error = None
            if not keep_open:
                self._opencv_ui.close()
                self._opencv_ui = None
                self._next_retry_unix = time.time() + self.retry_interval_s
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            self._next_retry_unix = time.time() + self.retry_interval_s
            if self._opencv_ui is not None:
                self._opencv_ui.close()
                self._opencv_ui = None
            print(
                f"[udp opencv ui] open/update failed; will retry in "
                f"{self.retry_interval_s:.1f}s: {exc}",
                flush=True,
            )


class UdpResultBridge:
    def __init__(self, config: UdpBridgeConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="planner-udp-bridge", daemon=True)
        self._socket: socket.socket | None = None
        self._latest_result: PlannerResult | None = None
        self._tx_seq = 0
        self._last_error: str | None = None
        self._last_sent_at_unix: float | None = None
        self._last_sent_plan_sequence: int | None = None
        self._last_sent_source_t0_us: int | None = None
        self._last_sent_actual_offset_s: float | None = None
        self._previous_full_plan_blend_source_packet: dict[str, Any] | None = None
        self._last_action_log_at_unix = 0.0
        self._path_log_lock = threading.Lock()
        self._path_log_dir = Path(config.path_log_dir).expanduser() if config.path_log_dir is not None else None
        self._path_log_jsonl_path = self._path_log_dir / "udp_sent_paths.jsonl" if self._path_log_dir else None
        self._path_log_latest_path = self._path_log_dir / "latest_udp_path.json" if self._path_log_dir else None
        self._path_log_count = 0
        self._last_path_log_path: str | None = None
        self._last_path_log_error: str | None = None
        if self._path_log_dir is not None:
            try:
                self._path_log_dir.mkdir(parents=True, exist_ok=True)
                print(f"[udp path log] enabled dir={self._path_log_dir}", flush=True)
            except Exception as exc:
                self._last_path_log_error = str(exc)
                print(f"[udp path log] disabled: cannot create {self._path_log_dir}: {exc}", flush=True)
                self._path_log_dir = None
                self._path_log_jsonl_path = None
                self._path_log_latest_path = None
        self._opencv_ui_enabled = bool(config.opencv_ui_enabled)
        self._opencv_ui_worker: UdpPathOpenCvUiWorker | None = None
        if self._opencv_ui_enabled:
            self._opencv_ui_worker = UdpPathOpenCvUiWorker(
                width=int(config.opencv_ui_width),
                height=int(config.opencv_ui_height),
                window_name=str(config.opencv_ui_window_name),
                display=config.opencv_ui_display,
                retry_interval_s=float(config.opencv_ui_retry_interval_s),
            )
            self._opencv_ui_worker.start()

    def start(self) -> None:
        if not self.config.enabled or self.config.send_mode == "on_result" or self._thread.is_alive():
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._opencv_ui_worker is not None:
            self._opencv_ui_worker.stop()
            self._opencv_ui_worker = None

    def publish_result(self, result: PlannerResult) -> None:
        if self.config.enabled and self.config.send_mode == "on_result":
            self.send_result_once(result)
            return
        with self._lock:
            self._latest_result = result

    def send_result_once(self, result: PlannerResult) -> dict[str, Any]:
        if not self.config.enabled:
            return {"udp_sent": False, "reason": "udp bridge disabled"}

        tx_time_us = time.time_ns() // 1000
        with self._lock:
            tx_seq = self._tx_seq

        if self.config.full_plan:
            packet = build_full_result_packet(
                result,
                tx_seq=tx_seq,
                tx_time_us=tx_time_us,
                latency_compensate=bool(self.config.latency_compensate_full_plan),
            )
        else:
            packet = build_live_result_packet(
                result,
                tx_seq=tx_seq,
                tx_time_us=tx_time_us,
                control_dt_s=self.config.control_dt_s,
                control_points=self.config.control_points,
            )
        blend_source_packet = _copy_packet(packet) if self.config.full_plan else None
        if self.config.full_plan and float(self.config.previous_path_blend_ratio) > 0.0:
            with self._lock:
                previous_packet = _copy_packet(self._previous_full_plan_blend_source_packet) if (
                    self._previous_full_plan_blend_source_packet is not None
                ) else None
            packet = blend_with_previous_full_plan_packet(
                packet,
                previous_packet,
                ratio=float(self.config.previous_path_blend_ratio),
                max_age_s=float(self.config.previous_path_blend_max_age_s),
            )
        packet = apply_origin_offset_to_packet(
            packet,
            offset_x_m=float(self.config.origin_offset_x_m),
            offset_y_m=float(self.config.origin_offset_y_m),
            yaw_offset_rad=float(self.config.origin_yaw_offset_rad),
        )

        actual_offset_s = _packet_age_s(packet)
        _update_udp_latency_timing(result, packet=packet)
        target = (self.config.host, self.config.port)
        payload = encode_udp_payload(result, packet, self.config.payload_mode)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(payload, target)
        finally:
            sock.close()

        with self._lock:
            self._tx_seq += 1
            self._last_error = None
            self._last_sent_at_unix = time.time()
            self._last_sent_plan_sequence = result.sequence
            self._last_sent_source_t0_us = result.t0_us
            self._last_sent_actual_offset_s = actual_offset_s
            if blend_source_packet is not None:
                self._previous_full_plan_blend_source_packet = blend_source_packet

        self._maybe_log_action_preview(result, packet)
        self._update_opencv_ui(
            result=result,
            packet=packet,
            target=target,
            actual_offset_s=actual_offset_s,
            bytes_len=len(payload),
        )
        path_log_path = self._write_path_log(
            result=result,
            packet=packet,
            target=target,
            actual_offset_s=actual_offset_s,
            bytes_len=len(payload),
        )

        return {
            "udp_sent": True,
            "target": f"{self.config.host}:{self.config.port}",
            "payload_mode": self.config.payload_mode,
            "bytes": len(payload),
            "tx_seq": tx_seq,
            "plan_seq": result.sequence,
            "source_t0_us": result.t0_us,
            "actual_offset_s": actual_offset_s,
            "path_log_path": path_log_path,
        }

    def send_packet_once(self, result: PlannerResult, packet: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enabled:
            return {"udp_sent": False, "reason": "udp bridge disabled"}

        tx_time_us = time.time_ns() // 1000
        with self._lock:
            tx_seq = self._tx_seq

        packet_to_send = dict(packet)
        packet_to_send["header"] = dict(packet.get("header") or {})
        packet_to_send["points"] = [dict(point) for point in packet.get("points") or []]
        packet_to_send["header"]["tx_seq"] = tx_seq
        packet_to_send["header"]["tx_time_us"] = tx_time_us
        packet_to_send = apply_origin_offset_to_packet(
            packet_to_send,
            offset_x_m=float(self.config.origin_offset_x_m),
            offset_y_m=float(self.config.origin_offset_y_m),
            yaw_offset_rad=float(self.config.origin_yaw_offset_rad),
        )

        actual_offset_s = _packet_age_s(packet_to_send)
        _update_udp_latency_timing(result, packet=packet_to_send)
        target = (self.config.host, self.config.port)
        payload = encode_udp_payload(result, packet_to_send, self.config.payload_mode)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(payload, target)
        finally:
            sock.close()

        with self._lock:
            self._tx_seq += 1
            self._last_error = None
            self._last_sent_at_unix = time.time()
            self._last_sent_plan_sequence = result.sequence
            self._last_sent_source_t0_us = result.t0_us
            self._last_sent_actual_offset_s = actual_offset_s

        self._maybe_log_action_preview(result, packet_to_send)
        self._update_opencv_ui(
            result=result,
            packet=packet_to_send,
            target=target,
            actual_offset_s=actual_offset_s,
            bytes_len=len(payload),
        )
        path_log_path = self._write_path_log(
            result=result,
            packet=packet_to_send,
            target=target,
            actual_offset_s=actual_offset_s,
            bytes_len=len(payload),
        )

        return {
            "udp_sent": True,
            "target": f"{self.config.host}:{self.config.port}",
            "payload_mode": self.config.payload_mode,
            "bytes": len(payload),
            "tx_seq": tx_seq,
            "plan_seq": result.sequence,
            "source_t0_us": result.t0_us,
            "actual_offset_s": actual_offset_s,
            "path_log_path": path_log_path,
            "prebuilt_packet": True,
        }

    def _write_path_log(
        self,
        *,
        result: PlannerResult,
        packet: dict[str, Any],
        target: tuple[str, int],
        actual_offset_s: float,
        bytes_len: int,
    ) -> str | None:
        if self._path_log_jsonl_path is None or self._path_log_latest_path is None:
            return None
        try:
            logged_ns = time.time_ns()
            record = build_udp_path_log_record(
                result,
                packet,
                target=target,
                payload_mode=self.config.payload_mode,
                actual_offset_s=actual_offset_s,
                bytes_len=bytes_len,
                logged_at_utc_ns=logged_ns,
                logged_at_unix=logged_ns / 1_000_000_000.0,
            )
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            latest_text = json.dumps(record, indent=2, ensure_ascii=False)
            with self._path_log_lock:
                assert self._path_log_jsonl_path is not None
                assert self._path_log_latest_path is not None
                with self._path_log_jsonl_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(line + "\n")
                latest_tmp_path = self._path_log_latest_path.with_suffix(self._path_log_latest_path.suffix + ".tmp")
                latest_tmp_path.write_text(latest_text, encoding="utf-8")
                latest_tmp_path.replace(self._path_log_latest_path)
                self._path_log_count += 1
                self._last_path_log_path = str(self._path_log_jsonl_path)
                self._last_path_log_error = None
            return str(self._path_log_jsonl_path)
        except Exception as exc:
            with self._path_log_lock:
                self._last_path_log_error = str(exc)
            print(f"[udp path log] write failed: {exc}", flush=True)
            return None

    def _maybe_log_action_preview(self, result: PlannerResult, packet: dict[str, Any]) -> None:
        interval_s = float(self.config.action_log_interval_s)
        mode = str(self.config.action_log_mode)
        points = int(self.config.action_log_points)
        distance_marks_m = tuple(float(value) for value in self.config.action_log_distance_marks_m)
        distance_mark_tolerance_m = float(self.config.action_log_distance_mark_tolerance_m)
        if interval_s < 0.0:
            return
        if mode == "points" and points <= 0:
            return
        if mode in {"distance_marks", "x_marks"} and not distance_marks_m:
            return
        now = time.time()
        with self._lock:
            if now - self._last_action_log_at_unix < interval_s:
                return
            self._last_action_log_at_unix = now
        print(
            _format_action_preview(
                result,
                packet,
                mode=mode,
                points=points,
                distance_marks_m=distance_marks_m,
                distance_mark_tolerance_m=distance_mark_tolerance_m,
            ),
            flush=True,
        )

    def _update_opencv_ui(
        self,
        *,
        result: PlannerResult,
        packet: dict[str, Any],
        target: tuple[str, int],
        actual_offset_s: float,
        bytes_len: int,
    ) -> None:
        if self._opencv_ui_worker is None:
            return
        self._opencv_ui_worker.submit(
            result=result,
            packet=packet,
            target=target,
            payload_mode=self.config.payload_mode,
            actual_offset_s=actual_offset_s,
            bytes_len=bytes_len,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            opencv_ui_last_error = self._opencv_ui_worker.last_error if self._opencv_ui_worker is not None else None
            return {
                "udp_bridge_enabled": self.config.enabled,
                "udp_bridge_alive": self.config.send_mode == "on_result" or self._thread.is_alive(),
                "udp_bridge_target": f"{self.config.host}:{self.config.port}" if self.config.enabled else None,
                "udp_payload_mode": self.config.payload_mode if self.config.enabled else None,
                "udp_send_mode": self.config.send_mode if self.config.enabled else None,
                "udp_full_plan_mode": self.config.full_plan if self.config.enabled else None,
                "udp_previous_path_blend_ratio": self.config.previous_path_blend_ratio if self.config.enabled else None,
                "udp_previous_path_blend_max_age_s": self.config.previous_path_blend_max_age_s
                if self.config.enabled
                else None,
                "udp_last_tx_seq": self._tx_seq - 1 if self._tx_seq > 0 else None,
                "udp_last_plan_sequence": self._last_sent_plan_sequence,
                "udp_last_source_t0_us": self._last_sent_source_t0_us,
                "udp_last_actual_offset_s": self._last_sent_actual_offset_s,
                "udp_last_sent_at_unix": self._last_sent_at_unix,
                "udp_last_error": self._last_error,
                "udp_opencv_ui_enabled": self._opencv_ui_enabled,
                "udp_opencv_ui_last_error": opencv_ui_last_error,
                "udp_path_log_dir": str(self._path_log_dir) if self._path_log_dir is not None else None,
                "udp_path_log_jsonl": str(self._path_log_jsonl_path) if self._path_log_jsonl_path is not None else None,
                "udp_path_log_latest": str(self._path_log_latest_path) if self._path_log_latest_path is not None else None,
                "udp_path_log_count": self._path_log_count,
                "udp_last_path_log_path": self._last_path_log_path,
                "udp_last_path_log_error": self._last_path_log_error,
            }

    def _run_loop(self) -> None:
        period_s = 1.0 / max(self.config.rate_hz, EPS)
        target = (self.config.host, self.config.port)
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            with self._lock:
                result = self._latest_result
            if result is not None and self._socket is not None:
                tx_time_us = time.time_ns() // 1000
                try:
                    if self.config.full_plan:
                        packet = build_full_result_packet(
                            result,
                            tx_seq=self._tx_seq,
                            tx_time_us=tx_time_us,
                            latency_compensate=bool(self.config.latency_compensate_full_plan),
                        )
                    else:
                        packet = build_live_result_packet(
                            result,
                            tx_seq=self._tx_seq,
                            tx_time_us=tx_time_us,
                            control_dt_s=self.config.control_dt_s,
                            control_points=self.config.control_points,
                        )
                    packet = apply_origin_offset_to_packet(
                        packet,
                        offset_x_m=float(self.config.origin_offset_x_m),
                        offset_y_m=float(self.config.origin_offset_y_m),
                        yaw_offset_rad=float(self.config.origin_yaw_offset_rad),
                    )
                    actual_offset_s = _packet_age_s(packet)
                    _update_udp_latency_timing(result, packet=packet)
                    payload = encode_udp_payload(result, packet, self.config.payload_mode)
                    self._socket.sendto(payload, target)
                    with self._lock:
                        self._tx_seq += 1
                        self._last_error = None
                        self._last_sent_at_unix = time.time()
                        self._last_sent_plan_sequence = result.sequence
                        self._last_sent_source_t0_us = result.t0_us
                        self._last_sent_actual_offset_s = actual_offset_s
                    self._maybe_log_action_preview(result, packet)
                    self._update_opencv_ui(
                        result=result,
                        packet=packet,
                        target=target,
                        actual_offset_s=actual_offset_s,
                        bytes_len=len(payload),
                    )
                    self._write_path_log(
                        result=result,
                        packet=packet,
                        target=target,
                        actual_offset_s=actual_offset_s,
                        bytes_len=len(payload),
                    )
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"udp send failed: {exc}"

            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s <= 0:
                next_deadline = time.perf_counter()
                continue
            self._stop_event.wait(timeout=sleep_s)
