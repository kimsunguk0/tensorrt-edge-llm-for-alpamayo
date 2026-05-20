from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .sample_contract import SampleMetadata


def _reshape_tensor_field(field: dict[str, Any] | None) -> np.ndarray | None:
    if not field:
        return None
    data = np.asarray(field["data"], dtype=np.float32)
    shape = tuple(int(x) for x in field["shape"])
    return data.reshape(shape)


@dataclass(slots=True)
class PlannerResult:
    sequence: int
    t0_us: int
    clip_id: str
    completed_at_unix: float
    output_text: str | None
    fm_mode: str | None
    fm_status: str | None
    plan_dt_s: float | None
    pred_xyz: list[list[float]]
    pred_rot: list[Any]
    post_vlm_timing: dict[str, Any]
    fm_timing: dict[str, Any]
    output_json_path: str
    x_final: list[list[float]] = field(default_factory=list)
    action_space_constants: dict[str, Any] = field(default_factory=dict)
    trajectory_json_path: str | None = None
    trajectory_xyz_npy_path: str | None = None
    trajectory_rot_npy_path: str | None = None
    dashboard_png_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "t0_us": self.t0_us,
            "clip_id": self.clip_id,
            "completed_at_unix": self.completed_at_unix,
            "output_text": self.output_text,
            "fm_mode": self.fm_mode,
            "fm_status": self.fm_status,
            "plan_dt_s": self.plan_dt_s,
            "pred_xyz": self.pred_xyz,
            "pred_rot": self.pred_rot,
            "x_final": self.x_final,
            "action_space_constants": self.action_space_constants,
            "post_vlm_timing": self.post_vlm_timing,
            "fm_timing": self.fm_timing,
            "output_json_path": self.output_json_path,
            "trajectory_json_path": self.trajectory_json_path,
            "trajectory_xyz_npy_path": self.trajectory_xyz_npy_path,
            "trajectory_rot_npy_path": self.trajectory_rot_npy_path,
            "dashboard_png_path": self.dashboard_png_path,
        }


def parse_planner_result(
    output_file: Path,
    metadata: SampleMetadata,
    *,
    trajectory_json_path: Path | None = None,
    trajectory_xyz_npy_path: Path | None = None,
    trajectory_rot_npy_path: Path | None = None,
    dashboard_png_path: Path | None = None,
    completed_at_unix: float | None = None,
) -> PlannerResult:
    payload = json.loads(output_file.read_text())
    response = payload["responses"][0]
    post_vlm = response.get("alpamayo_post_vlm", {})
    fm = post_vlm.get("fm", {})

    pred_xyz = _reshape_tensor_field(fm.get("pred_xyz"))
    pred_rot = _reshape_tensor_field(fm.get("pred_rot"))
    x_final = _reshape_tensor_field(fm.get("x_final"))
    action_space_constants = fm.get("action_space_constants", {})

    pred_xyz_list = pred_xyz[0].tolist() if pred_xyz is not None and pred_xyz.ndim >= 2 else []
    pred_rot_list = pred_rot[0].tolist() if pred_rot is not None and pred_rot.ndim >= 3 else []
    x_final_list = x_final[0].tolist() if x_final is not None and x_final.ndim >= 2 else []
    plan_dt_value = action_space_constants.get("dt_value")
    plan_dt_s = float(plan_dt_value) if plan_dt_value is not None else None

    return PlannerResult(
        sequence=metadata.sequence,
        t0_us=metadata.t0_us,
        clip_id=metadata.clip_id,
        completed_at_unix=completed_at_unix if completed_at_unix is not None else time.time(),
        output_text=response.get("output_text"),
        fm_mode=post_vlm.get("fm_mode"),
        fm_status=post_vlm.get("fm_status"),
        plan_dt_s=plan_dt_s,
        pred_xyz=pred_xyz_list,
        pred_rot=pred_rot_list,
        x_final=x_final_list,
        action_space_constants=dict(action_space_constants),
        post_vlm_timing=dict(post_vlm.get("timing", {})),
        fm_timing=dict(fm.get("timing", {})),
        output_json_path=str(output_file),
        trajectory_json_path=str(trajectory_json_path) if trajectory_json_path is not None else None,
        trajectory_xyz_npy_path=str(trajectory_xyz_npy_path) if trajectory_xyz_npy_path is not None else None,
        trajectory_rot_npy_path=str(trajectory_rot_npy_path) if trajectory_rot_npy_path is not None else None,
        dashboard_png_path=str(dashboard_png_path) if dashboard_png_path is not None else None,
    )
