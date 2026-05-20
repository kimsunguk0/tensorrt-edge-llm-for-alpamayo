from __future__ import annotations

import unittest

import numpy as np

from planner_live.sample_contract import (
    FRONT_TELE_CAMERA_IDS,
    FRONT_TELE_CAMERA_ORDER,
    LEGACY_CAMERA_ORDER,
    PLANNER_CAMERA_IDS,
    PLANNER_CAMERA_ORDER,
    SINGLE_CAMERA_CONFIGS,
)
from planner_live.sample_validator import ValidationError, validate_live_sample


def make_valid_sample(
    *,
    camera_indices: tuple[int, ...] = PLANNER_CAMERA_IDS,
    camera_order: tuple[str, ...] = PLANNER_CAMERA_ORDER,
) -> dict[str, object]:
    num_cameras = len(camera_indices)
    ego_history_rot = np.zeros((1, 1, 16, 3, 3), dtype=np.float32)
    ego_history_rot[0, 0, -1] = np.eye(3, dtype=np.float32)
    return {
        "image_frames": np.zeros((num_cameras, 4, 3, 320, 576), dtype=np.uint8),
        "camera_indices": np.asarray(camera_indices, dtype=np.int32),
        "ego_history_xyz": np.zeros((1, 1, 16, 3), dtype=np.float32),
        "ego_history_rot": ego_history_rot,
        "relative_timestamps": np.zeros((num_cameras, 4), dtype=np.float32),
        "absolute_timestamps": np.zeros((num_cameras, 4), dtype=np.int64),
        "t0_us": 123,
        "fixed_delta_seconds": 0.1,
        "clip_id": "clip-1",
        "camera_order": list(camera_order),
    }


class SampleValidationTests(unittest.TestCase):
    def test_accepts_canonical_contract(self) -> None:
        normalized = validate_live_sample(make_valid_sample())
        self.assertEqual(normalized["camera_order"], list(PLANNER_CAMERA_ORDER))

    def test_accepts_legacy_camera_aliases(self) -> None:
        normalized = validate_live_sample(make_valid_sample(camera_order=LEGACY_CAMERA_ORDER))
        self.assertEqual(normalized["camera_order"], list(PLANNER_CAMERA_ORDER))

    def test_accepts_front_and_front_tele_contract(self) -> None:
        normalized = validate_live_sample(
            make_valid_sample(camera_indices=FRONT_TELE_CAMERA_IDS, camera_order=FRONT_TELE_CAMERA_ORDER)
        )
        self.assertEqual(normalized["camera_order"], list(FRONT_TELE_CAMERA_ORDER))

    def test_accepts_single_camera_contracts(self) -> None:
        for camera_indices, camera_order in SINGLE_CAMERA_CONFIGS.items():
            with self.subTest(camera_indices=camera_indices):
                normalized = validate_live_sample(
                    make_valid_sample(camera_indices=camera_indices, camera_order=camera_order)
                )
                self.assertEqual(normalized["camera_order"], list(camera_order))

    def test_accepts_single_camera_legacy_alias(self) -> None:
        normalized = validate_live_sample(make_valid_sample(camera_indices=(1,), camera_order=("front_camera",)))
        self.assertEqual(normalized["camera_order"], ["front"])

    def test_accepts_npz_scalar_arrays(self) -> None:
        sample = make_valid_sample(camera_indices=FRONT_TELE_CAMERA_IDS, camera_order=FRONT_TELE_CAMERA_ORDER)
        sample["t0_us"] = np.asarray([123], dtype=np.int64)
        sample["fixed_delta_seconds"] = np.asarray([0.1], dtype=np.float32)
        sample["clip_id"] = np.asarray(["clip-1"])
        normalized = validate_live_sample(sample)
        self.assertEqual(normalized["t0_us"], 123)
        self.assertEqual(normalized["clip_id"], "clip-1")

    def test_rejects_missing_key(self) -> None:
        sample = make_valid_sample()
        sample.pop("clip_id")
        with self.assertRaisesRegex(ValidationError, "missing required key: clip_id"):
            validate_live_sample(sample)

    def test_rejects_wrong_camera_order(self) -> None:
        sample = make_valid_sample()
        sample["camera_order"] = ["rear", "front", "right", "front_tele"]
        with self.assertRaisesRegex(ValidationError, "camera_order must be one of"):
            validate_live_sample(sample)

    def test_rejects_wrong_shape(self) -> None:
        sample = make_valid_sample()
        sample["relative_timestamps"] = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "relative_timestamps shape"):
            validate_live_sample(sample)


if __name__ == "__main__":
    unittest.main()
