from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import threading
import time
from typing import Any
import urllib.request

import numpy as np

from scripts.control_team_replay_common import COORD_MODE_LOCAL, FLAG_VALID, build_packet_dict, wrap_angles

from .result_bridge import UdpResultBridge, sample_latency_compensated_full_result
from .result_parser import PlannerResult


EPS = 1e-6
_BRACKET_VECTOR_RE_TEMPLATE = r"(?:^|\s){key}\s*=\s*\[([^\]]+)\]"
_SCALAR_RE_TEMPLATE = r"(?:^|\s){key}\s*=\s*([-+0-9.eE]+)"


@dataclass(frozen=True, slots=True)
class LivePathManagerConfig:
    enabled: bool = False
    rate_hz: float = 10.0
    health_url: str = "http://127.0.0.1:18080/healthz"
    pose_timeout_s: float = 0.2
    max_plan_age_s: float = 3.0
    min_plan_arc_m: float = 2.0
    min_remaining_distance_m: float = 2.0
    max_projection_distance_m: float = 5.0
    output_points: int = 65
    plan_blend_s: float = 0.0
    stabilize_local_frame: bool = False
    yaw_filter_tau_s: float = 0.35
    yaw_max_rate_rad_s: float = 1.5
    projection_arc_filter_tau_s: float = 0.35
    fixed_arc_step_m: float = 0.0
    log_interval_s: float = 1.0


@dataclass(frozen=True, slots=True)
class LivePose:
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float | None
    timestamp_utc_ns: int | None
    receive_time_utc_ns: int
    source: str


@dataclass(slots=True)
class CachedPathPlan:
    result: PlannerResult
    registered_at_unix: float
    registered_at_utc_ns: int
    source_t0_us: int
    plan_seq: int
    plan_dt_s: float
    global_xy: np.ndarray
    global_yaw_unwrapped: np.ndarray
    v_mps: np.ndarray
    curvature: np.ndarray
    arc_m: np.ndarray
    initial_age_s: float
    initial_pose: LivePose
    blend_duration_s: float = 0.0
    blend_started_at_unix: float | None = None
    blend_from_plan_seq: int | None = None
    blend_from_global_xy: np.ndarray | None = None
    blend_from_global_yaw_unwrapped: np.ndarray | None = None
    blend_from_v_mps: np.ndarray | None = None
    blend_from_curvature: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Projection:
    segment_idx: int
    ratio: float
    arc_m: float
    distance_m: float
    point_xy: np.ndarray


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _numeric_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        parsed = _coerce_float(item)
        if parsed is None:
            return None
        out.append(float(parsed))
    return out


def _parse_text_vector(payload: str, *keys: str) -> list[float] | None:
    for key in keys:
        pattern = _BRACKET_VECTOR_RE_TEMPLATE.format(key=re.escape(key))
        match = re.search(pattern, payload)
        if match is None:
            continue
        values: list[float] = []
        for token in match.group(1).split(","):
            parsed = _coerce_float(token.strip())
            if parsed is None:
                values = []
                break
            values.append(float(parsed))
        if values:
            return values
    return None


def _parse_text_scalar(payload: str, *keys: str) -> float | None:
    for key in keys:
        pattern = _SCALAR_RE_TEMPLATE.format(key=re.escape(key))
        match = re.search(pattern, payload)
        if match is None:
            continue
        parsed = _coerce_float(match.group(1))
        if parsed is not None:
            return float(parsed)
    return None


def _find_utm(payload: Any, *, _depth: int = 0) -> list[float] | None:
    if _depth > 6 or payload is None:
        return None
    if isinstance(payload, bytes):
        return _find_utm(payload.decode("utf-8", errors="replace"), _depth=_depth + 1)
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            return _find_utm(json.loads(stripped), _depth=_depth + 1)
        except json.JSONDecodeError:
            vector = _parse_text_vector(stripped, "gnss_utm", "utm", "utm_enu", "utm_xyz")
            return vector if vector is not None and len(vector) >= 2 else None
    if isinstance(payload, dict):
        for key in ("gnss_utm", "utm", "utm_enu", "utm_xyz"):
            vector = _numeric_vector(payload.get(key))
            if vector is not None and len(vector) >= 2:
                return vector
        for key in ("latest_fix", "gnss_latest", "gnss_fix", "fix", "gnss", "localization"):
            if key in payload:
                nested = _find_utm(payload[key], _depth=_depth + 1)
                if nested is not None:
                    return nested
        for value in payload.values():
            nested = _find_utm(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _find_utm(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


def _find_timestamp_utc_ns(payload: Any, *, _depth: int = 0) -> int | None:
    if _depth > 6 or payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("timestamp_utc_ns", "t0_utc_ns", "receive_time_utc_ns"):
            parsed = _coerce_int(payload.get(key))
            if parsed is not None:
                return parsed
        for key in ("latest_fix", "gnss_latest", "gnss_fix", "fix", "gnss", "localization"):
            if key in payload:
                nested = _find_timestamp_utc_ns(payload[key], _depth=_depth + 1)
                if nested is not None:
                    return nested
        for value in payload.values():
            nested = _find_timestamp_utc_ns(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _find_timestamp_utc_ns(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


def _find_yaw_rad(payload: Any, *, _depth: int = 0) -> float | None:
    if _depth > 6 or payload is None:
        return None
    if isinstance(payload, bytes):
        return _find_yaw_rad(payload.decode("utf-8", errors="replace"), _depth=_depth + 1)
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            return _find_yaw_rad(json.loads(stripped), _depth=_depth + 1)
        except json.JSONDecodeError:
            yaw_rad = _parse_text_scalar(stripped, "ins_yaw_rad", "yaw_rad", "heading_rad", "heading_enu_rad")
            if yaw_rad is not None:
                return float(yaw_rad)
            yaw_deg = _parse_text_scalar(stripped, "ins_yaw_deg", "yaw_deg", "heading_deg")
            if yaw_deg is not None:
                return math.radians(float(yaw_deg))
            euler_deg = _parse_text_vector(stripped, "euler_deg", "euler")
            if euler_deg is not None and len(euler_deg) >= 3:
                return math.radians(float(euler_deg[2]))
            return None
    if isinstance(payload, dict):
        for key in ("ins_yaw_rad", "yaw_rad", "heading_rad", "heading_enu_rad"):
            parsed = _coerce_float(payload.get(key))
            if parsed is not None:
                return float(parsed)
        for key in ("ins_yaw_deg", "yaw_deg", "heading_deg"):
            parsed = _coerce_float(payload.get(key))
            if parsed is not None:
                return math.radians(float(parsed))
        euler = payload.get("euler_deg") or payload.get("euler")
        vector = _numeric_vector(euler)
        if vector is not None and len(vector) >= 3:
            return math.radians(float(vector[2]))
        for key in ("imu", "orientation", "ins", "latest_fix", "gnss", "localization"):
            if key in payload:
                nested = _find_yaw_rad(payload[key], _depth=_depth + 1)
                if nested is not None:
                    return nested
        for value in payload.values():
            nested = _find_yaw_rad(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _find_yaw_rad(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


def _find_speed_mps(payload: Any, *, _depth: int = 0) -> float | None:
    if _depth > 6 or payload is None:
        return None
    if isinstance(payload, bytes):
        return _find_speed_mps(payload.decode("utf-8", errors="replace"), _depth=_depth + 1)
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            return _find_speed_mps(json.loads(stripped), _depth=_depth + 1)
        except json.JSONDecodeError:
            speed = _parse_text_scalar(stripped, "speed_mps", "speed", "ego_speed_mps")
            if speed is not None:
                return float(speed)
            velocity = _parse_text_vector(stripped, "velocity_mps", "vel")
            if velocity is not None and len(velocity) >= 2:
                return float(math.hypot(velocity[0], velocity[1]))
            return None
    if isinstance(payload, dict):
        for key in ("speed_mps", "speed", "ego_speed_mps"):
            parsed = _coerce_float(payload.get(key))
            if parsed is not None:
                return parsed
        vector = _numeric_vector(payload.get("velocity_mps"))
        if vector is not None and len(vector) >= 2:
            return float(math.hypot(vector[0], vector[1]))
        for value in payload.values():
            nested = _find_speed_mps(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for value in payload:
            nested = _find_speed_mps(value, _depth=_depth + 1)
            if nested is not None:
                return nested
    return None


class HealthzPoseSource:
    def __init__(self, health_url: str) -> None:
        self.health_url = str(health_url)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def label(self) -> str:
        return self.health_url

    def read_pose(self, timeout_s: float) -> LivePose:
        receive_time_utc_ns = time.time_ns()
        with urllib.request.urlopen(self.health_url, timeout=float(timeout_s)) as response:
            raw = response.read()
        try:
            payload: Any = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            payload = raw
        utm = _find_utm(payload)
        yaw_rad = _find_yaw_rad(payload)
        if utm is None or len(utm) < 2:
            raise RuntimeError("healthz response did not contain gnss_utm/utm")
        if yaw_rad is None:
            raise RuntimeError("healthz response did not contain ins_yaw_rad/yaw_rad")
        return LivePose(
            x_m=float(utm[0]),
            y_m=float(utm[1]),
            yaw_rad=float(wrap_angles(np.asarray([yaw_rad], dtype=np.float32))[0]),
            speed_mps=_find_speed_mps(payload),
            timestamp_utc_ns=_find_timestamp_utc_ns(payload),
            receive_time_utc_ns=receive_time_utc_ns,
            source=self.health_url,
        )


def _path_arc(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    if xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths)])


def _local_to_global(local_xy: np.ndarray, pose: LivePose) -> np.ndarray:
    cos_yaw = math.cos(float(pose.yaw_rad))
    sin_yaw = math.sin(float(pose.yaw_rad))
    x = local_xy[:, 0].astype(np.float64)
    y = local_xy[:, 1].astype(np.float64)
    return np.column_stack(
        [
            float(pose.x_m) + x * cos_yaw - y * sin_yaw,
            float(pose.y_m) + x * sin_yaw + y * cos_yaw,
        ]
    )


def _global_to_local(global_xy: np.ndarray, pose: LivePose) -> np.ndarray:
    dx = global_xy[:, 0].astype(np.float64) - float(pose.x_m)
    dy = global_xy[:, 1].astype(np.float64) - float(pose.y_m)
    cos_yaw = math.cos(float(pose.yaw_rad))
    sin_yaw = math.sin(float(pose.yaw_rad))
    return np.column_stack([dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw])


def _wrap_angle_scalar(angle_rad: float) -> float:
    return float(wrap_angles(np.asarray([angle_rad], dtype=np.float32))[0])


def _with_pose_yaw(pose: LivePose, yaw_rad: float) -> LivePose:
    return LivePose(
        x_m=pose.x_m,
        y_m=pose.y_m,
        yaw_rad=_wrap_angle_scalar(yaw_rad),
        speed_mps=pose.speed_mps,
        timestamp_utc_ns=pose.timestamp_utc_ns,
        receive_time_utc_ns=pose.receive_time_utc_ns,
        source=pose.source,
    )


def _project_to_path(point_xy: np.ndarray, path_xy: np.ndarray, arc_m: np.ndarray) -> Projection | None:
    if path_xy.shape[0] < 2:
        return None
    starts = path_xy[:-1]
    ends = path_xy[1:]
    seg = ends - starts
    seg_len2 = np.sum(seg * seg, axis=1)
    valid = seg_len2 > EPS
    if not np.any(valid):
        return None
    rel = point_xy[None, :] - starts
    ratio_all = np.zeros((seg.shape[0],), dtype=np.float64)
    ratio_all[valid] = np.clip(np.sum(rel[valid] * seg[valid], axis=1) / seg_len2[valid], 0.0, 1.0)
    projected = starts + ratio_all[:, None] * seg
    distance = np.linalg.norm(projected - point_xy[None, :], axis=1)
    distance[~valid] = np.inf
    idx = int(np.argmin(distance))
    segment_length = float(math.sqrt(seg_len2[idx]))
    proj_arc = float(arc_m[idx] + ratio_all[idx] * segment_length)
    return Projection(
        segment_idx=idx,
        ratio=float(ratio_all[idx]),
        arc_m=proj_arc,
        distance_m=float(distance[idx]),
        point_xy=projected[idx].astype(np.float64),
    )


def _projection_at_arc(
    target_arc_m: float,
    point_xy: np.ndarray,
    path_xy: np.ndarray,
    arc_m: np.ndarray,
) -> Projection | None:
    if path_xy.shape[0] < 2 or arc_m.shape[0] != path_xy.shape[0]:
        return None
    target = float(np.clip(float(target_arc_m), float(arc_m[0]), float(arc_m[-1])))
    idx = int(np.searchsorted(arc_m, target, side="right") - 1)
    idx = max(0, min(idx, path_xy.shape[0] - 2))
    segment_length = float(max(arc_m[idx + 1] - arc_m[idx], EPS))
    ratio = float(np.clip((target - float(arc_m[idx])) / segment_length, 0.0, 1.0))
    point = path_xy[idx] * (1.0 - ratio) + path_xy[idx + 1] * ratio
    return Projection(
        segment_idx=idx,
        ratio=ratio,
        arc_m=target,
        distance_m=float(np.linalg.norm(point - point_xy.astype(np.float64))),
        point_xy=point.astype(np.float64),
    )


def _interp_segment(values: np.ndarray, idx: int, ratio: float) -> float:
    if values.shape[0] == 0:
        return 0.0
    if idx >= values.shape[0] - 1:
        return float(values[-1])
    return float(values[idx] * (1.0 - ratio) + values[idx + 1] * ratio)


def _pad_or_trim(values: np.ndarray, output_points: int) -> np.ndarray:
    target = max(int(output_points), 2)
    if values.shape[0] >= target:
        return values[:target]
    pad_count = target - values.shape[0]
    pad = np.repeat(values[-1:], pad_count, axis=0)
    return np.concatenate([values, pad], axis=0)


def _resample_path_values(
    target_arc_m: np.ndarray,
    source_arc_m: np.ndarray,
    source_values: np.ndarray,
    fallback_values: np.ndarray,
) -> np.ndarray:
    if target_arc_m.size == 0:
        return fallback_values.copy()
    if source_arc_m.size < 2 or source_values.shape[0] < 2:
        return fallback_values.copy()
    target = target_arc_m.astype(np.float64)
    valid = target <= float(source_arc_m[-1]) + EPS
    if source_values.ndim == 1:
        out = fallback_values.astype(np.float64).copy()
        out[valid] = np.interp(target[valid], source_arc_m, source_values)
        return out
    out = fallback_values.astype(np.float64).copy()
    for dim in range(source_values.shape[1]):
        out[valid, dim] = np.interp(target[valid], source_arc_m, source_values[:, dim])
    return out


def _sample_path_values(target_arc_m: np.ndarray, source_arc_m: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    if target_arc_m.size == 0:
        return source_values[:0].copy()
    target = np.clip(target_arc_m.astype(np.float64), float(source_arc_m[0]), float(source_arc_m[-1]))
    if source_values.ndim == 1:
        return np.interp(target, source_arc_m, source_values).astype(np.float64)
    out = np.empty((target.shape[0], source_values.shape[1]), dtype=np.float64)
    for dim in range(source_values.shape[1]):
        out[:, dim] = np.interp(target, source_arc_m, source_values[:, dim])
    return out


class LivePathManager:
    def __init__(
        self,
        config: LivePathManagerConfig,
        *,
        udp_bridge: UdpResultBridge,
        pose_source: Any | None = None,
    ) -> None:
        self.config = config
        self.udp_bridge = udp_bridge
        self.pose_source = pose_source if pose_source is not None else HealthzPoseSource(config.health_url)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="planner-live-path-manager", daemon=True)
        self._plan: CachedPathPlan | None = None
        self._last_error: str | None = None
        self._last_pose: LivePose | None = None
        self._last_publish_info: dict[str, Any] | None = None
        self._last_publish_at_unix: float | None = None
        self._published_count = 0
        self._dropped_publish_count = 0
        self._accepted_plan_count = 0
        self._rejected_plan_count = 0
        self._last_log_at_unix = 0.0
        self._filtered_yaw_unwrapped: float | None = None
        self._filtered_pose_time_unix: float | None = None
        self._projection_plan_seq: int | None = None
        self._projection_arc_m: float | None = None
        self._projection_time_unix: float | None = None

    def start(self) -> None:
        self.pose_source.start()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.pose_source.stop()

    def publish_result(self, result: PlannerResult) -> dict[str, Any]:
        try:
            pose = self.pose_source.read_pose(float(self.config.pose_timeout_s))
            pose = self._stabilize_pose(pose)
            plan = self._build_cached_plan(result, pose)
        except Exception as exc:
            with self._lock:
                self._rejected_plan_count += 1
                self._last_error = f"plan registration failed seq={result.sequence}: {exc}"
            return {"path_manager_plan_accepted": False, "reason": str(exc)}

        if float(plan.arc_m[-1]) < float(self.config.min_plan_arc_m):
            reason = f"path arc {float(plan.arc_m[-1]):.3f}m < min_plan_arc_m={self.config.min_plan_arc_m:.3f}m"
            with self._lock:
                self._rejected_plan_count += 1
                self._last_error = f"plan rejected seq={result.sequence}: {reason}"
            return {"path_manager_plan_accepted": False, "reason": reason}

        with self._lock:
            previous_plan = self._plan
            if previous_plan is not None and float(self.config.plan_blend_s) > EPS:
                self._attach_plan_blend(plan, previous_plan, pose)
            self._plan = plan
            self._projection_plan_seq = None
            self._projection_arc_m = None
            self._projection_time_unix = None
            self._last_pose = pose
            self._last_error = None
            self._accepted_plan_count += 1
        return {
            "path_manager_plan_accepted": True,
            "plan_seq": result.sequence,
            "initial_age_s": plan.initial_age_s,
            "path_arc_m": float(plan.arc_m[-1]),
            "pose_source": pose.source,
            "plan_blend_s": float(plan.blend_duration_s),
            "plan_blend_from_seq": plan.blend_from_plan_seq,
        }

    def _build_cached_plan(self, result: PlannerResult, pose: LivePose) -> CachedPathPlan:
        now_us = time.time_ns() // 1000
        initial_age_s = max(0.0, (int(now_us) - int(result.t0_us)) / 1_000_000.0)
        sampled = sample_latency_compensated_full_result(result, age_s=initial_age_s)
        local_xy = np.column_stack(
            [np.asarray(sampled["x"], dtype=np.float64), np.asarray(sampled["y"], dtype=np.float64)]
        )
        global_xy = _local_to_global(local_xy, pose)
        local_yaw = np.unwrap(np.asarray(sampled["yaw"], dtype=np.float64))
        global_yaw = np.unwrap(local_yaw + float(pose.yaw_rad))
        return CachedPathPlan(
            result=result,
            registered_at_unix=time.time(),
            registered_at_utc_ns=time.time_ns(),
            source_t0_us=int(result.t0_us),
            plan_seq=int(result.sequence),
            plan_dt_s=float(result.plan_dt_s if result.plan_dt_s is not None else 0.1),
            global_xy=global_xy.astype(np.float64),
            global_yaw_unwrapped=global_yaw.astype(np.float64),
            v_mps=np.asarray(sampled["v"], dtype=np.float64),
            curvature=np.asarray(sampled["curvature"], dtype=np.float64),
            arc_m=_path_arc(global_xy),
            initial_age_s=initial_age_s,
            initial_pose=pose,
        )

    def _resolved_plan_arrays(
        self, plan: CachedPathPlan, *, now_unix: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if (
            plan.blend_duration_s <= EPS
            or plan.blend_started_at_unix is None
            or plan.blend_from_global_xy is None
            or plan.blend_from_global_yaw_unwrapped is None
            or plan.blend_from_v_mps is None
            or plan.blend_from_curvature is None
        ):
            return plan.global_xy, plan.global_yaw_unwrapped, plan.v_mps, plan.curvature, plan.arc_m, 1.0

        elapsed_s = max(0.0, (time.time() if now_unix is None else float(now_unix)) - float(plan.blend_started_at_unix))
        alpha = min(1.0, elapsed_s / max(float(plan.blend_duration_s), EPS))
        if alpha >= 1.0:
            return plan.global_xy, plan.global_yaw_unwrapped, plan.v_mps, plan.curvature, plan.arc_m, 1.0

        global_xy = plan.blend_from_global_xy * (1.0 - alpha) + plan.global_xy * alpha
        global_yaw = plan.blend_from_global_yaw_unwrapped * (1.0 - alpha) + plan.global_yaw_unwrapped * alpha
        v_mps = plan.blend_from_v_mps * (1.0 - alpha) + plan.v_mps * alpha
        curvature = plan.blend_from_curvature * (1.0 - alpha) + plan.curvature * alpha
        return global_xy, global_yaw, v_mps, curvature, _path_arc(global_xy), float(alpha)

    def _attach_plan_blend(self, plan: CachedPathPlan, previous_plan: CachedPathPlan, pose: LivePose) -> None:
        previous_global_xy, previous_yaw, previous_v, previous_curvature, previous_arc, _ = self._resolved_plan_arrays(
            previous_plan
        )
        projection = _project_to_path(
            np.asarray([float(pose.x_m), float(pose.y_m)], dtype=np.float64),
            previous_global_xy,
            previous_arc,
        )
        if projection is None or projection.distance_m > float(self.config.max_projection_distance_m):
            return

        idx = projection.segment_idx
        ratio = projection.ratio
        start_global_xy = projection.point_xy.reshape(1, 2)
        remaining_global_xy = np.concatenate([start_global_xy, previous_global_xy[idx + 1 :]], axis=0)
        if remaining_global_xy.shape[0] < 2:
            return

        start_yaw = _interp_segment(previous_yaw, idx, ratio)
        remaining_yaw = np.concatenate([np.asarray([start_yaw], dtype=np.float64), previous_yaw[idx + 1 :]])
        start_v = _interp_segment(previous_v, idx, ratio)
        remaining_v = np.concatenate([np.asarray([start_v], dtype=np.float64), previous_v[idx + 1 :]])
        start_curvature = _interp_segment(previous_curvature, idx, ratio)
        remaining_curvature = np.concatenate(
            [np.asarray([start_curvature], dtype=np.float64), previous_curvature[idx + 1 :]]
        )
        remaining_arc = _path_arc(remaining_global_xy)
        if float(remaining_arc[-1]) < EPS:
            return

        target_arc = plan.arc_m
        blend_from_xy = _resample_path_values(target_arc, remaining_arc, remaining_global_xy, plan.global_xy)
        blend_from_yaw = _resample_path_values(target_arc, remaining_arc, remaining_yaw, plan.global_yaw_unwrapped)
        blend_from_yaw = plan.global_yaw_unwrapped + np.unwrap(
            wrap_angles((blend_from_yaw - plan.global_yaw_unwrapped).astype(np.float32)).astype(np.float64)
        )
        blend_from_v = _resample_path_values(target_arc, remaining_arc, remaining_v, plan.v_mps)
        blend_from_curvature = _resample_path_values(target_arc, remaining_arc, remaining_curvature, plan.curvature)

        plan.blend_duration_s = max(0.0, float(self.config.plan_blend_s))
        plan.blend_started_at_unix = time.time()
        plan.blend_from_plan_seq = previous_plan.plan_seq
        plan.blend_from_global_xy = blend_from_xy.astype(np.float64)
        plan.blend_from_global_yaw_unwrapped = blend_from_yaw.astype(np.float64)
        plan.blend_from_v_mps = blend_from_v.astype(np.float64)
        plan.blend_from_curvature = blend_from_curvature.astype(np.float64)

    def _run_loop(self) -> None:
        period_s = 1.0 / max(float(self.config.rate_hz), EPS)
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            self._publish_tick()
            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s <= 0.0:
                next_deadline = time.perf_counter()
                continue
            self._stop_event.wait(timeout=sleep_s)

    def _publish_tick(self) -> None:
        with self._lock:
            plan = self._plan
        if plan is None:
            return
        try:
            pose = self.pose_source.read_pose(float(self.config.pose_timeout_s))
            pose = self._stabilize_pose(pose)
            packet, debug = self._build_packet(plan, pose)
            if packet is None:
                with self._lock:
                    self._dropped_publish_count += 1
                    self._last_error = str(debug.get("reason", "path manager skipped publish"))
                    self._last_pose = pose
                self._maybe_log_skip(debug)
                return
            info = self.udp_bridge.send_packet_once(plan.result, packet)
            info.update(debug)
            with self._lock:
                self._published_count += 1
                self._last_publish_at_unix = time.time()
                self._last_publish_info = info
                self._last_pose = pose
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._dropped_publish_count += 1
                self._last_error = f"path manager publish failed: {exc}"

    def _stabilize_pose(self, pose: LivePose) -> LivePose:
        if not bool(self.config.stabilize_local_frame):
            return pose

        pose_time_unix = float(pose.receive_time_utc_ns) / 1_000_000_000.0
        raw_yaw = float(pose.yaw_rad)
        tau_s = max(0.0, float(self.config.yaw_filter_tau_s))
        max_rate = max(0.0, float(self.config.yaw_max_rate_rad_s))
        with self._lock:
            if self._filtered_yaw_unwrapped is None or self._filtered_pose_time_unix is None:
                self._filtered_yaw_unwrapped = raw_yaw
                self._filtered_pose_time_unix = pose_time_unix
                return _with_pose_yaw(pose, raw_yaw)

            dt_s = pose_time_unix - float(self._filtered_pose_time_unix)
            if dt_s <= 0.0 or dt_s > 1.0:
                self._filtered_yaw_unwrapped = raw_yaw
                self._filtered_pose_time_unix = pose_time_unix
                return _with_pose_yaw(pose, raw_yaw)

            raw_delta = _wrap_angle_scalar(raw_yaw - float(self._filtered_yaw_unwrapped))
            if max_rate > EPS:
                max_step = max_rate * dt_s
                raw_delta = float(np.clip(raw_delta, -max_step, max_step))
            raw_unwrapped = float(self._filtered_yaw_unwrapped) + raw_delta
            alpha = 1.0 if tau_s <= EPS else float(np.clip(dt_s / (tau_s + dt_s), 0.0, 1.0))
            self._filtered_yaw_unwrapped = float(self._filtered_yaw_unwrapped) + alpha * (
                raw_unwrapped - float(self._filtered_yaw_unwrapped)
            )
            self._filtered_pose_time_unix = pose_time_unix
            return _with_pose_yaw(pose, float(self._filtered_yaw_unwrapped))

    def _stabilize_projection(
        self,
        plan: CachedPathPlan,
        projection: Projection,
        pose: LivePose,
        global_xy: np.ndarray,
        arc_m: np.ndarray,
        *,
        now_unix: float,
    ) -> Projection:
        if not bool(self.config.stabilize_local_frame):
            return projection

        raw_arc = float(projection.arc_m)
        tau_s = max(0.0, float(self.config.projection_arc_filter_tau_s))
        speed_mps = _coerce_float(pose.speed_mps)
        if speed_mps is None:
            speed_mps = 0.0
        speed_mps = float(np.clip(speed_mps, 0.0, 15.0))
        with self._lock:
            if (
                self._projection_plan_seq != plan.plan_seq
                or self._projection_arc_m is None
                or self._projection_time_unix is None
            ):
                self._projection_plan_seq = plan.plan_seq
                self._projection_arc_m = raw_arc
                self._projection_time_unix = float(now_unix)
                return projection

            dt_s = max(0.0, float(now_unix) - float(self._projection_time_unix))
            if dt_s <= 0.0 or dt_s > 1.0:
                tracked_arc = raw_arc
            else:
                predicted_arc = float(self._projection_arc_m) + speed_mps * dt_s
                alpha = 1.0 if tau_s <= EPS else float(np.clip(dt_s / (tau_s + dt_s), 0.0, 1.0))
                tracked_arc = predicted_arc + alpha * (raw_arc - predicted_arc)
                tracked_arc = max(float(self._projection_arc_m), tracked_arc)

            tracked_arc = float(np.clip(tracked_arc, float(arc_m[0]), float(arc_m[-1])))
            self._projection_arc_m = tracked_arc
            self._projection_time_unix = float(now_unix)

        point_xy = np.asarray([float(pose.x_m), float(pose.y_m)], dtype=np.float64)
        stable_projection = _projection_at_arc(tracked_arc, point_xy, global_xy, arc_m)
        return projection if stable_projection is None else stable_projection

    def _build_packet(self, plan: CachedPathPlan, pose: LivePose) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        now_us = time.time_ns() // 1000
        now_unix = float(now_us) / 1_000_000.0
        plan_age_s = max(0.0, (int(now_us) - int(plan.source_t0_us)) / 1_000_000.0)
        if plan_age_s > float(self.config.max_plan_age_s):
            return None, {
                "reason": f"plan age {plan_age_s:.3f}s > max_plan_age_s={self.config.max_plan_age_s:.3f}s",
                "plan_age_s": plan_age_s,
                "plan_seq": plan.plan_seq,
            }

        global_xy, global_yaw_unwrapped, v_mps_values, curvature_values, arc_m, blend_alpha = self._resolved_plan_arrays(
            plan
        )
        projection = _project_to_path(
            np.asarray([float(pose.x_m), float(pose.y_m)], dtype=np.float64),
            global_xy,
            arc_m,
        )
        if projection is None:
            return None, {"reason": "projection unavailable", "plan_age_s": plan_age_s, "plan_seq": plan.plan_seq}
        if projection.distance_m > float(self.config.max_projection_distance_m):
            return None, {
                "reason": (
                    f"projection distance {projection.distance_m:.3f}m > "
                    f"max_projection_distance_m={self.config.max_projection_distance_m:.3f}m"
                ),
                "plan_age_s": plan_age_s,
                "plan_seq": plan.plan_seq,
                "projection_distance_m": projection.distance_m,
            }

        raw_projection = projection
        projection = self._stabilize_projection(
            plan,
            projection,
            pose,
            global_xy,
            arc_m,
            now_unix=now_unix,
        )

        remaining_distance_m = float(arc_m[-1] - projection.arc_m)
        if remaining_distance_m < float(self.config.min_remaining_distance_m):
            return None, {
                "reason": (
                    f"remaining distance {remaining_distance_m:.3f}m < "
                    f"min_remaining_distance_m={self.config.min_remaining_distance_m:.3f}m"
                ),
                "plan_age_s": plan_age_s,
                "plan_seq": plan.plan_seq,
                "projection_distance_m": projection.distance_m,
                "remaining_distance_m": remaining_distance_m,
            }

        idx = projection.segment_idx
        ratio = projection.ratio
        output_points = max(int(self.config.output_points), 2)
        fixed_arc_step_m = max(0.0, float(self.config.fixed_arc_step_m))
        if fixed_arc_step_m > EPS:
            target_arc_m = projection.arc_m + np.arange(output_points, dtype=np.float64) * fixed_arc_step_m
            target_arc_m = np.clip(target_arc_m, float(arc_m[0]), float(arc_m[-1]))
            sampled_global_xy = _sample_path_values(target_arc_m, arc_m, global_xy)
            local_xy = _global_to_local(sampled_global_xy, pose)
            yaw_global = _sample_path_values(target_arc_m, arc_m, global_yaw_unwrapped)
            yaw_local = wrap_angles((yaw_global - float(pose.yaw_rad)).astype(np.float32)).astype(np.float64)
            v_mps = _sample_path_values(target_arc_m, arc_m, v_mps_values)
            curvature = _sample_path_values(target_arc_m, arc_m, curvature_values)
        else:
            start_global_xy = projection.point_xy.reshape(1, 2)
            remaining_global_xy = np.concatenate([start_global_xy, global_xy[idx + 1 :]], axis=0)
            local_xy = _global_to_local(remaining_global_xy, pose)

            start_yaw = _interp_segment(global_yaw_unwrapped, idx, ratio)
            yaw_global = np.concatenate(
                [np.asarray([start_yaw], dtype=np.float64), global_yaw_unwrapped[idx + 1 :]]
            )
            yaw_local = wrap_angles((yaw_global - float(pose.yaw_rad)).astype(np.float32)).astype(np.float64)
            v_mps = np.concatenate(
                [np.asarray([_interp_segment(v_mps_values, idx, ratio)], dtype=np.float64), v_mps_values[idx + 1 :]]
            )
            curvature = np.concatenate(
                [
                    np.asarray([_interp_segment(curvature_values, idx, ratio)], dtype=np.float64),
                    curvature_values[idx + 1 :],
                ]
            )

            local_xy = _pad_or_trim(local_xy, output_points)
            yaw_local = _pad_or_trim(yaw_local.reshape(-1, 1), output_points).reshape(-1)
            v_mps = _pad_or_trim(v_mps.reshape(-1, 1), output_points).reshape(-1)
            curvature = _pad_or_trim(curvature.reshape(-1, 1), output_points).reshape(-1)

        packet = build_packet_dict(
            tx_seq=0,
            plan_seq=plan.plan_seq,
            sample_id=plan.plan_seq,
            source_t0_us=plan.source_t0_us,
            tx_time_us=now_us,
            coord_mode=COORD_MODE_LOCAL,
            dt_s=plan.plan_dt_s,
            x=local_xy[:, 0].astype(np.float32),
            y=local_xy[:, 1].astype(np.float32),
            yaw=yaw_local.astype(np.float32),
            v=v_mps.astype(np.float32),
            curvature=curvature.astype(np.float32),
            flags=FLAG_VALID,
        )
        packet["header"].update(
            {
                "path_manager_enabled": True,
                "path_manager_mode": "gnss_projection",
                "path_manager_rate_hz": float(self.config.rate_hz),
                "path_manager_plan_age_s": plan_age_s,
                "path_manager_initial_age_s": float(plan.initial_age_s),
                "path_manager_projection_distance_m": float(projection.distance_m),
                "path_manager_raw_projection_distance_m": float(raw_projection.distance_m),
                "path_manager_projection_arc_m": float(projection.arc_m),
                "path_manager_raw_projection_arc_m": float(raw_projection.arc_m),
                "path_manager_remaining_distance_m": remaining_distance_m,
                "path_manager_plan_blend_s": float(plan.blend_duration_s),
                "path_manager_plan_blend_alpha": float(blend_alpha),
                "path_manager_plan_blend_active": bool(blend_alpha < 1.0),
                "path_manager_plan_blend_from_seq": plan.blend_from_plan_seq,
                "path_manager_stabilize_local_frame": bool(self.config.stabilize_local_frame),
                "path_manager_yaw_filter_tau_s": float(self.config.yaw_filter_tau_s),
                "path_manager_yaw_max_rate_rad_s": float(self.config.yaw_max_rate_rad_s),
                "path_manager_projection_arc_filter_tau_s": float(self.config.projection_arc_filter_tau_s),
                "path_manager_fixed_arc_step_m": float(fixed_arc_step_m),
                "path_manager_pose_source": pose.source,
                "path_manager_pose_receive_time_utc_ns": int(pose.receive_time_utc_ns),
                "path_manager_pose_timestamp_utc_ns": pose.timestamp_utc_ns,
                "path_manager_pose_x_m": float(pose.x_m),
                "path_manager_pose_y_m": float(pose.y_m),
                "path_manager_pose_yaw_rad": float(pose.yaw_rad),
                "path_manager_pose_speed_mps": pose.speed_mps,
                "path_manager_output_points": output_points,
            }
        )
        return packet, {
            "plan_seq": plan.plan_seq,
            "plan_age_s": plan_age_s,
            "projection_distance_m": float(projection.distance_m),
            "raw_projection_distance_m": float(raw_projection.distance_m),
            "remaining_distance_m": remaining_distance_m,
            "plan_blend_s": float(plan.blend_duration_s),
            "plan_blend_alpha": float(blend_alpha),
            "plan_blend_active": bool(blend_alpha < 1.0),
            "plan_blend_from_seq": plan.blend_from_plan_seq,
            "stabilize_local_frame": bool(self.config.stabilize_local_frame),
            "fixed_arc_step_m": float(fixed_arc_step_m),
            "path_manager_packet": True,
        }

    def _maybe_log_skip(self, debug: dict[str, Any]) -> None:
        interval_s = float(self.config.log_interval_s)
        if interval_s < 0.0:
            return
        now = time.time()
        if now - self._last_log_at_unix < interval_s:
            return
        self._last_log_at_unix = now
        print(f"[path manager] skip publish: {debug}", flush=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            plan = self._plan
            pose = self._last_pose
            return {
                "path_manager_enabled": True,
                "path_manager_alive": self._thread.is_alive(),
                "path_manager_rate_hz": float(self.config.rate_hz),
                "path_manager_plan_blend_s": float(self.config.plan_blend_s),
                "path_manager_stabilize_local_frame": bool(self.config.stabilize_local_frame),
                "path_manager_yaw_filter_tau_s": float(self.config.yaw_filter_tau_s),
                "path_manager_yaw_max_rate_rad_s": float(self.config.yaw_max_rate_rad_s),
                "path_manager_projection_arc_filter_tau_s": float(self.config.projection_arc_filter_tau_s),
                "path_manager_fixed_arc_step_m": float(self.config.fixed_arc_step_m),
                "path_manager_pose_source": getattr(self.pose_source, "label", str(self.pose_source)),
                "path_manager_latest_plan_seq": None if plan is None else plan.plan_seq,
                "path_manager_latest_plan_age_s": None
                if plan is None
                else max(0.0, time.time() - float(plan.source_t0_us) / 1_000_000.0),
                "path_manager_latest_plan_arc_m": None if plan is None else float(plan.arc_m[-1]),
                "path_manager_accepted_plan_count": self._accepted_plan_count,
                "path_manager_rejected_plan_count": self._rejected_plan_count,
                "path_manager_published_count": self._published_count,
                "path_manager_dropped_publish_count": self._dropped_publish_count,
                "path_manager_last_publish_at_unix": self._last_publish_at_unix,
                "path_manager_last_publish_info": self._last_publish_info,
                "path_manager_last_error": self._last_error,
                "path_manager_last_pose": None
                if pose is None
                else {
                    "x_m": pose.x_m,
                    "y_m": pose.y_m,
                    "yaw_rad": pose.yaw_rad,
                    "speed_mps": pose.speed_mps,
                    "timestamp_utc_ns": pose.timestamp_utc_ns,
                    "receive_time_utc_ns": pose.receive_time_utc_ns,
                    "source": pose.source,
                },
            }
