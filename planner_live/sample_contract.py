from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_SAMPLE_KEYS = (
    "image_frames",
    "camera_indices",
    "ego_history_xyz",
    "ego_history_rot",
    "relative_timestamps",
    "absolute_timestamps",
    "t0_us",
    "fixed_delta_seconds",
    "clip_id",
    "camera_order",
)

PLANNER_CAMERA_IDS = (0, 1, 2, 6)
PLANNER_CAMERA_ORDER = ("left", "front", "right", "front_tele")
LIVE_4CAM_CAMERA_IDS = (1, 6, 0, 2)
LIVE_4CAM_CAMERA_ORDER = ("front", "front_tele", "left", "right")
FRONT_TELE_CAMERA_IDS = (1, 6)
FRONT_TELE_CAMERA_ORDER = ("front", "front_tele")
FRONT_LEFT_CAMERA_IDS = (1, 0)
FRONT_LEFT_CAMERA_ORDER = ("front", "left")
LEFT_FRONT_CAMERA_IDS = (0, 1)
LEFT_FRONT_CAMERA_ORDER = ("left", "front")
SINGLE_CAMERA_CONFIGS = {
    (0,): ("left",),
    (1,): ("front",),
    (2,): ("right",),
    (6,): ("front_tele",),
}
LEGACY_CAMERA_ORDER = (
    "front_left_camera",
    "front_camera",
    "front_right_camera",
    "front_telephoto_camera",
)

SUPPORTED_CAMERA_CONFIGS = {
    PLANNER_CAMERA_IDS: PLANNER_CAMERA_ORDER,
    LIVE_4CAM_CAMERA_IDS: LIVE_4CAM_CAMERA_ORDER,
    FRONT_TELE_CAMERA_IDS: FRONT_TELE_CAMERA_ORDER,
    FRONT_LEFT_CAMERA_IDS: FRONT_LEFT_CAMERA_ORDER,
    LEFT_FRONT_CAMERA_IDS: LEFT_FRONT_CAMERA_ORDER,
    **SINGLE_CAMERA_CONFIGS,
}

LEGACY_CAMERA_ORDER_ALIASES = {
    LEGACY_CAMERA_ORDER: PLANNER_CAMERA_ORDER,
    ("front_camera", "front_telephoto_camera", "front_left_camera", "front_right_camera"): LIVE_4CAM_CAMERA_ORDER,
    ("front_camera", "front_left_camera"): FRONT_LEFT_CAMERA_ORDER,
    ("front_left_camera", "front_camera"): LEFT_FRONT_CAMERA_ORDER,
    ("front_left_camera",): ("left",),
    ("front_camera",): ("front",),
    ("front_right_camera",): ("right",),
    ("front_telephoto_camera",): ("front_tele",),
}

NUM_FRAMES_PER_CAMERA = 4
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 576

EXPECTED_IMAGE_SHAPE = (4, 4, 3, 320, 576)
EXPECTED_CAMERA_INDICES_SHAPE = (4,)
EXPECTED_EGO_HISTORY_XYZ_SHAPE = (1, 1, 16, 3)
EXPECTED_EGO_HISTORY_ROT_SHAPE = (1, 1, 16, 3, 3)
EXPECTED_TIMESTAMP_SHAPE = (4, 4)

DEFAULT_LAST_XYZ_ABS_TOL = 1e-3
DEFAULT_LAST_ROT_ABS_TOL = 1e-3


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    sequence: int
    t0_us: int
    clip_id: str
    received_at_unix: float
    source_headers: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "t0_us": self.t0_us,
            "clip_id": self.clip_id,
            "received_at_unix": self.received_at_unix,
            "source_headers": dict(self.source_headers),
        }


@dataclass(slots=True)
class SampleEnvelope:
    sample: dict[str, Any]
    metadata: SampleMetadata
