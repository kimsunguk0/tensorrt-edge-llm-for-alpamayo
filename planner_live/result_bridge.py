from __future__ import annotations

from dataclasses import dataclass
import json
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
    action_log_interval_s: float = 3.0
    action_log_points: int = 16


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
) -> dict[str, Any]:
    knots = _result_knots(result)
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
    future_points = points[1:] if points else []
    raw_curvature = [-float(point["curvature"]) for point in future_points]
    raw_action = _decode_raw_action(result, initial_speed_mps=initial_speed_mps, dt_s=dt_s)
    if raw_action is None:
        raw_action = {
            "num_points": max(len(points) - 1, 0),
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
        "plan_points_no_origin": max(len(points) - 1, 0),
        "traj_points_with_origin": len(points),
        "pred_xyz": result.pred_xyz,
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
        "udp_mode": "text_json_live",
    }


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


def _format_action_preview(result: PlannerResult, packet: dict[str, Any], points: int) -> str:
    packet_points = list(packet.get("points") or [])
    future_points = packet_points[1:] if len(packet_points) > 1 else packet_points
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
            *rows,
        ]
    )


def encode_udp_payload(result: PlannerResult, packet: dict[str, Any], payload_mode: str) -> bytes:
    if payload_mode == "binary_alpa":
        return pack_packet(packet)
    if payload_mode == "text_json":
        payload = build_text_result_payload(result, packet)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raise ValueError(f"unsupported udp payload mode: {payload_mode}")


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
        self._last_action_log_at_unix = 0.0

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
            )
        else:
            packet = build_live_result_packet(
                result,
                tx_seq=tx_seq,
                tx_time_us=tx_time_us,
                control_dt_s=self.config.control_dt_s,
                control_points=self.config.control_points,
            )

        actual_offset_s = _packet_age_s(packet)
        target = (self.config.host, self.config.port)
        payload = encode_udp_payload(result, packet, self.config.payload_mode)
        self._maybe_log_action_preview(result, packet)
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

        return {
            "udp_sent": True,
            "target": f"{self.config.host}:{self.config.port}",
            "payload_mode": self.config.payload_mode,
            "bytes": len(payload),
            "tx_seq": tx_seq,
            "plan_seq": result.sequence,
            "source_t0_us": result.t0_us,
            "actual_offset_s": actual_offset_s,
        }

    def _maybe_log_action_preview(self, result: PlannerResult, packet: dict[str, Any]) -> None:
        interval_s = float(self.config.action_log_interval_s)
        points = int(self.config.action_log_points)
        if interval_s < 0.0 or points <= 0:
            return
        now = time.time()
        with self._lock:
            if now - self._last_action_log_at_unix < interval_s:
                return
            self._last_action_log_at_unix = now
        print(_format_action_preview(result, packet, points), flush=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "udp_bridge_enabled": self.config.enabled,
                "udp_bridge_alive": self.config.send_mode == "on_result" or self._thread.is_alive(),
                "udp_bridge_target": f"{self.config.host}:{self.config.port}" if self.config.enabled else None,
                "udp_payload_mode": self.config.payload_mode if self.config.enabled else None,
                "udp_send_mode": self.config.send_mode if self.config.enabled else None,
                "udp_full_plan_mode": self.config.full_plan if self.config.enabled else None,
                "udp_last_tx_seq": self._tx_seq - 1 if self._tx_seq > 0 else None,
                "udp_last_plan_sequence": self._last_sent_plan_sequence,
                "udp_last_source_t0_us": self._last_sent_source_t0_us,
                "udp_last_actual_offset_s": self._last_sent_actual_offset_s,
                "udp_last_sent_at_unix": self._last_sent_at_unix,
                "udp_last_error": self._last_error,
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
                        )
                    else:
                        packet = build_live_result_packet(
                            result,
                            tx_seq=self._tx_seq,
                            tx_time_us=tx_time_us,
                            control_dt_s=self.config.control_dt_s,
                            control_points=self.config.control_points,
                        )
                    actual_offset_s = _packet_age_s(packet)
                    self._maybe_log_action_preview(result, packet)
                    self._socket.sendto(encode_udp_payload(result, packet, self.config.payload_mode), target)
                    with self._lock:
                        self._tx_seq += 1
                        self._last_error = None
                        self._last_sent_at_unix = time.time()
                        self._last_sent_plan_sequence = result.sequence
                        self._last_sent_source_t0_us = result.t0_us
                        self._last_sent_actual_offset_s = actual_offset_s
                except Exception as exc:
                    with self._lock:
                        self._last_error = f"udp send failed: {exc}"

            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s <= 0:
                next_deadline = time.perf_counter()
                continue
            self._stop_event.wait(timeout=sleep_s)
