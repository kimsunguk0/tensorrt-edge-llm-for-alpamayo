from __future__ import annotations

from pathlib import Path


MODELS_ROOT = Path("/workspace/models/alpamayo_runtime")
FM_ROOT = MODELS_ROOT / "fm"

STUDENT_STEP4_V2_MXFP8_DIR = FM_ROOT / "student_teacher_structured_reflow_consistency_step4_v2_20260409_mxfp8"
STUDENT_STEP4_V2_MXFP8_ENGINE = (
    STUDENT_STEP4_V2_MXFP8_DIR / "student_teacher_structured_reflow_consistency_step4_v2_20260409_one_step_mxfp8_thor.plan"
)

STUDENT_STEP4_MXFP8_ENGINE = (
    FM_ROOT
    / "student_teacher_structured_reflow_consistency_step4_mxfp8"
    / "student_teacher_structured_reflow_consistency_step4_one_step_mxfp8_thor.plan"
)
LEGACY_FP16_ENGINE = (
    FM_ROOT / "alpamayo15_fm_one_step_fp16_true" / "alpamayo15_fm_one_step_fp16_true_thor.plan"
)
LEGACY_MXFP8_ENGINE = FM_ROOT / "alpamayo15_fm_one_step_mxfp8_thor.plan"


def preferred_fm_engine_candidates(default_models_root: Path | None = None) -> list[Path]:
    if default_models_root is None:
        fm_root = FM_ROOT
    else:
        fm_root = default_models_root / "fm"
    return [
        fm_root
        / "student_teacher_structured_reflow_consistency_step4_v2_20260409_mxfp8"
        / "student_teacher_structured_reflow_consistency_step4_v2_20260409_one_step_mxfp8_thor.plan",
        fm_root
        / "student_teacher_structured_reflow_consistency_step4_mxfp8"
        / "student_teacher_structured_reflow_consistency_step4_one_step_mxfp8_thor.plan",
        fm_root / "alpamayo15_fm_one_step_fp16_true" / "alpamayo15_fm_one_step_fp16_true_thor.plan",
        fm_root / "alpamayo15_fm_one_step_mxfp8_thor.plan",
    ]


def first_existing_fm_engine(default_models_root: Path | None = None, extra_candidates: list[Path] | None = None) -> Path:
    candidates = preferred_fm_engine_candidates(default_models_root)
    if extra_candidates:
        candidates.extend(extra_candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
