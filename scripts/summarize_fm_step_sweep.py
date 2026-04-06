#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FM step sweep against a baseline step count")
    parser.add_argument("--run-root", type=Path, required=True, help="Root dir containing stepXX subdirs")
    parser.add_argument("--baseline-step", type=int, default=10)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    return parser.parse_args()


def load_output_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    outputs = {}
    for path in sorted(run_dir.glob("output_*.json")):
        sample_tag = path.stem.split("output_", 1)[1]
        outputs[sample_tag] = json.loads(path.read_text())
    return outputs


def load_profile_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    profiles = {}
    for path in sorted(run_dir.glob("profile_*.json")):
        sample_tag = path.stem.split("profile_", 1)[1]
        profiles[sample_tag] = json.loads(path.read_text())
    return profiles


def reshape_field(field: dict[str, Any]) -> np.ndarray:
    data = np.asarray(field["data"], dtype=np.float32)
    return data.reshape(tuple(int(x) for x in field["shape"]))


def summarize_pair(
    *,
    baseline_output: dict[str, Any],
    candidate_output: dict[str, Any],
) -> dict[str, Any]:
    base_resp = baseline_output["responses"][0]
    cand_resp = candidate_output["responses"][0]
    base_post = base_resp["alpamayo_post_vlm"]
    cand_post = cand_resp["alpamayo_post_vlm"]

    base_xyz = reshape_field(base_post["fm"]["pred_xyz"])[0]
    cand_xyz = reshape_field(cand_post["fm"]["pred_xyz"])[0]
    base_rot = reshape_field(base_post["fm"]["pred_rot"])[0]
    cand_rot = reshape_field(cand_post["fm"]["pred_rot"])[0]

    delta_xyz = cand_xyz - base_xyz
    delta_rot = cand_rot - base_rot
    delta_xy_norm = np.linalg.norm(delta_xyz[:, :2], axis=1)

    base_t = base_post["timing"]
    cand_t = cand_post["timing"]
    base_fm_t = base_post["fm"]["timing"]
    cand_fm_t = cand_post["fm"]["timing"]

    return {
        "text_same": str(base_resp.get("output_text") or "") == str(cand_resp.get("output_text") or ""),
        "pred_xyz_mae": float(np.mean(np.abs(delta_xyz))),
        "pred_xyz_rmse": float(np.sqrt(np.mean(delta_xyz**2))),
        "pred_xyz_max_abs": float(np.max(np.abs(delta_xyz))),
        "pred_rot_mae": float(np.mean(np.abs(delta_rot))),
        "pred_rot_rmse": float(np.sqrt(np.mean(delta_rot**2))),
        "max_xy_delta_m": float(np.max(delta_xy_norm)),
        "end_dx": float(delta_xyz[-1, 0]),
        "end_dy": float(delta_xyz[-1, 1]),
        "end_dz": float(delta_xyz[-1, 2]),
        "baseline_total_post_vlm_ms": float(base_t["total_post_vlm_ms"]),
        "candidate_total_post_vlm_ms": float(cand_t["total_post_vlm_ms"]),
        "delta_total_post_vlm_ms": float(cand_t["total_post_vlm_ms"] - base_t["total_post_vlm_ms"]),
        "baseline_fm_wall_ms": float(base_t["fm_wall_ms"]),
        "candidate_fm_wall_ms": float(cand_t["fm_wall_ms"]),
        "delta_fm_wall_ms": float(cand_t["fm_wall_ms"] - base_t["fm_wall_ms"]),
        "baseline_fm_engine_step_total_ms": float(base_fm_t["engine_step_total_ms"]),
        "candidate_fm_engine_step_total_ms": float(cand_fm_t["engine_step_total_ms"]),
        "delta_fm_engine_step_total_ms": float(cand_fm_t["engine_step_total_ms"] - base_fm_t["engine_step_total_ms"]),
        "baseline_num_steps": int(base_fm_t["num_steps"]),
        "candidate_num_steps": int(cand_fm_t["num_steps"]),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row]
    return float(sum(vals) / len(vals)) if vals else None


def main() -> None:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = args.run_root / f"step{args.baseline_step:02d}"
    baseline_outputs = load_output_map(baseline_dir)

    all_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for step in args.steps:
        run_dir = args.run_root / f"step{step:02d}"
        outputs = load_output_map(run_dir)
        step_rows: list[dict[str, Any]] = []
        for sample_tag, baseline_output in sorted(baseline_outputs.items()):
            if sample_tag not in outputs:
                raise FileNotFoundError(f"Missing sample {sample_tag} in {run_dir}")
            row = {
                "sample_tag": sample_tag,
                "step": step,
            }
            row.update(summarize_pair(baseline_output=baseline_output, candidate_output=outputs[sample_tag]))
            step_rows.append(row)
            all_rows.append(row)

        aggregate_rows.append(
            {
                "step": step,
                "num_samples": len(step_rows),
                "text_same_count": int(sum(bool(r["text_same"]) for r in step_rows)),
                "pred_xyz_mae_mean": mean(step_rows, "pred_xyz_mae"),
                "pred_xyz_rmse_mean": mean(step_rows, "pred_xyz_rmse"),
                "pred_xyz_max_abs_mean": mean(step_rows, "pred_xyz_max_abs"),
                "max_xy_delta_m_mean": mean(step_rows, "max_xy_delta_m"),
                "candidate_total_post_vlm_ms_mean": mean(step_rows, "candidate_total_post_vlm_ms"),
                "delta_total_post_vlm_ms_mean": mean(step_rows, "delta_total_post_vlm_ms"),
                "candidate_fm_wall_ms_mean": mean(step_rows, "candidate_fm_wall_ms"),
                "delta_fm_wall_ms_mean": mean(step_rows, "delta_fm_wall_ms"),
                "candidate_fm_engine_step_total_ms_mean": mean(step_rows, "candidate_fm_engine_step_total_ms"),
                "delta_fm_engine_step_total_ms_mean": mean(step_rows, "delta_fm_engine_step_total_ms"),
            }
        )

    rows_csv = args.summary_dir / "summary_rows.csv"
    aggregates_csv = args.summary_dir / "aggregate_rows.csv"
    summary_json = args.summary_dir / "summary.json"

    row_fields = sorted({key for row in all_rows for key in row.keys()})
    with rows_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    agg_fields = sorted({key for row in aggregate_rows for key in row.keys()})
    with aggregates_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    payload = {
        "baseline_step": args.baseline_step,
        "steps": args.steps,
        "aggregates": aggregate_rows,
        "rows": all_rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    print(rows_csv)
    print(aggregates_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
