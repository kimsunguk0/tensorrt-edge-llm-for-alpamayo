from __future__ import annotations

from typing import Any

import numpy as np

from .sample_contract import (
    DEFAULT_LAST_ROT_ABS_TOL,
    DEFAULT_LAST_XYZ_ABS_TOL,
    EXPECTED_EGO_HISTORY_ROT_SHAPE,
    EXPECTED_EGO_HISTORY_XYZ_SHAPE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEGACY_CAMERA_ORDER_ALIASES,
    NUM_FRAMES_PER_CAMERA,
    REQUIRED_SAMPLE_KEYS,
    SUPPORTED_CAMERA_CONFIGS,
)


class ValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def _dtype_name(value: Any) -> str:
    return str(np.asarray(value).dtype)


def _check_shape(issues: list[str], name: str, value: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != expected_shape:
        issues.append(f"{name} shape {arr.shape} != {expected_shape}")
    return arr


def _as_list(value: Any) -> list[Any]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.tolist()


def _as_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    raise ValueError(f"expected scalar or 1-element array, got shape {arr.shape}")


def _camera_order(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_list(value))


def _expected_camera_order(camera_indices: tuple[int, ...], camera_order: tuple[str, ...]) -> tuple[str, ...] | None:
    expected_order = SUPPORTED_CAMERA_CONFIGS.get(camera_indices)
    if expected_order is not None and camera_order == expected_order:
        return expected_order
    aliased_order = LEGACY_CAMERA_ORDER_ALIASES.get(camera_order)
    if aliased_order is not None and aliased_order == expected_order:
        return aliased_order
    return None


def validate_live_sample(
    sample: dict[str, Any],
    *,
    last_xyz_abs_tol: float = DEFAULT_LAST_XYZ_ABS_TOL,
    last_rot_abs_tol: float = DEFAULT_LAST_ROT_ABS_TOL,
) -> dict[str, Any]:
    issues: list[str] = []

    for key in REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            issues.append(f"missing required key: {key}")

    if issues:
        raise ValidationError(issues)

    image_frames = np.asarray(sample["image_frames"])
    if image_frames.ndim != 5:
        issues.append(f"image_frames ndim {image_frames.ndim} != 5")
        num_cameras = 0
    else:
        num_cameras = int(image_frames.shape[0])
        expected_tail = (NUM_FRAMES_PER_CAMERA, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        if tuple(image_frames.shape[1:]) != expected_tail:
            issues.append(f"image_frames shape tail {image_frames.shape[1:]} != {expected_tail}")
    if image_frames.dtype != np.uint8:
        issues.append(f"image_frames dtype {_dtype_name(image_frames)} != uint8")

    camera_indices = np.asarray(sample["camera_indices"])
    if camera_indices.shape != (num_cameras,):
        issues.append(f"camera_indices shape {camera_indices.shape} != ({num_cameras},)")
    if camera_indices.dtype != np.int32:
        issues.append(f"camera_indices dtype {_dtype_name(camera_indices)} != int32")
    camera_index_tuple = tuple(int(x) for x in camera_indices.reshape(-1).tolist())
    if camera_index_tuple not in SUPPORTED_CAMERA_CONFIGS:
        supported = [list(indices) for indices in SUPPORTED_CAMERA_CONFIGS]
        issues.append(f"camera_indices {list(camera_index_tuple)} must be one of {supported}")

    ego_history_xyz = _check_shape(issues, "ego_history_xyz", sample["ego_history_xyz"], EXPECTED_EGO_HISTORY_XYZ_SHAPE)
    if ego_history_xyz.dtype != np.float32:
        issues.append(f"ego_history_xyz dtype {_dtype_name(ego_history_xyz)} != float32")

    ego_history_rot = _check_shape(issues, "ego_history_rot", sample["ego_history_rot"], EXPECTED_EGO_HISTORY_ROT_SHAPE)
    if ego_history_rot.dtype != np.float32:
        issues.append(f"ego_history_rot dtype {_dtype_name(ego_history_rot)} != float32")

    relative_timestamps = np.asarray(sample["relative_timestamps"])
    if relative_timestamps.shape != (num_cameras, NUM_FRAMES_PER_CAMERA):
        issues.append(
            f"relative_timestamps shape {relative_timestamps.shape} != "
            f"({num_cameras}, {NUM_FRAMES_PER_CAMERA})"
        )
    if relative_timestamps.dtype != np.float32:
        issues.append(f"relative_timestamps dtype {_dtype_name(relative_timestamps)} != float32")

    absolute_timestamps = np.asarray(sample["absolute_timestamps"])
    if absolute_timestamps.shape != (num_cameras, NUM_FRAMES_PER_CAMERA):
        issues.append(
            f"absolute_timestamps shape {absolute_timestamps.shape} != "
            f"({num_cameras}, {NUM_FRAMES_PER_CAMERA})"
        )
    if absolute_timestamps.dtype != np.int64:
        issues.append(f"absolute_timestamps dtype {_dtype_name(absolute_timestamps)} != int64")

    raw_camera_order = _camera_order(sample["camera_order"])
    camera_order = _expected_camera_order(camera_index_tuple, raw_camera_order)
    if camera_order is None:
        supported = [
            {"camera_indices": list(indices), "camera_order": list(order)}
            for indices, order in SUPPORTED_CAMERA_CONFIGS.items()
        ]
        issues.append(
            "camera_order must be one of the supported camera configs; "
            f"supported={supported}; got indices={list(camera_index_tuple)} order={list(raw_camera_order)}"
        )

    try:
        t0_us = int(_as_scalar(sample["t0_us"]))
    except Exception:
        issues.append(f"t0_us is not int-like: {sample['t0_us']!r}")
        t0_us = 0

    try:
        fixed_delta_seconds = float(_as_scalar(sample["fixed_delta_seconds"]))
    except Exception:
        issues.append(f"fixed_delta_seconds is not float-like: {sample['fixed_delta_seconds']!r}")
        fixed_delta_seconds = 0.0

    try:
        clip_id = str(_as_scalar(sample["clip_id"]))
    except Exception:
        clip_id = str(sample["clip_id"])
    if not clip_id:
        issues.append("clip_id must not be empty")
    if fixed_delta_seconds <= 0.0:
        issues.append(f"fixed_delta_seconds must be > 0, got {fixed_delta_seconds}")

    if ego_history_xyz.shape == EXPECTED_EGO_HISTORY_XYZ_SHAPE and not np.all(np.isfinite(ego_history_xyz)):
        issues.append("ego_history_xyz contains non-finite values")

    if ego_history_rot.shape == EXPECTED_EGO_HISTORY_ROT_SHAPE and not np.all(np.isfinite(ego_history_rot)):
        issues.append("ego_history_rot contains non-finite values")

    if issues:
        raise ValidationError(issues)

    normalized = dict(sample)
    normalized["image_frames"] = image_frames.astype(np.uint8, copy=False)
    normalized["camera_indices"] = camera_indices.astype(np.int32, copy=False)
    normalized["ego_history_xyz"] = ego_history_xyz.astype(np.float32, copy=False)
    normalized["ego_history_rot"] = ego_history_rot.astype(np.float32, copy=False)
    normalized["relative_timestamps"] = relative_timestamps.astype(np.float32, copy=False)
    normalized["absolute_timestamps"] = absolute_timestamps.astype(np.int64, copy=False)
    normalized["camera_order"] = list(camera_order)
    normalized["t0_us"] = t0_us
    normalized["fixed_delta_seconds"] = fixed_delta_seconds
    normalized["clip_id"] = clip_id
    return normalized
