#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate steering/control yaw bias from straight dummy GNSS runs. "
            "Each run should contain summary.json from straight_dummy_gnss_test.py."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Run directories. You may append ':yaw_deg' when metadata does not "
            "contain dummy_initial_yaw_deg, e.g. run_foo:-1.0."
        ),
    )
    parser.add_argument(
        "--min-forward-m",
        type=float,
        default=5.0,
        help="Ignore runs with less forward motion than this.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_run_arg(value: str) -> tuple[Path, float | None]:
    if ":" not in value:
        return Path(value), None
    path_text, yaw_text = value.rsplit(":", 1)
    try:
        return Path(path_text), float(yaw_text)
    except ValueError:
        return Path(value), None


def yaw_from_metadata(run_dir: Path) -> float | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = load_json(metadata_path)
    yaw = metadata.get("dummy_initial_yaw_deg")
    if yaw is not None:
        return float(yaw)
    yaw_rad = metadata.get("dummy_initial_yaw_rad")
    if yaw_rad is not None:
        return math.degrees(float(yaw_rad))
    return None


def main() -> int:
    args = parse_args()
    rows: list[dict[str, float | str]] = []
    for item in args.runs:
        run_dir, yaw_override = split_run_arg(item)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"[skip] no summary.json: {run_dir}")
            continue
        summary = load_json(summary_path)
        yaw_cmd = yaw_override if yaw_override is not None else yaw_from_metadata(run_dir)
        if yaw_cmd is None:
            yaw_cmd = 0.0
        x = float(summary.get("final_x_m", 0.0) or 0.0)
        y = float(summary.get("final_y_m_left_positive", 0.0) or 0.0)
        if abs(x) < float(args.min_forward_m):
            print(f"[skip] forward too short x={x:.3f}m: {run_dir}")
            continue
        residual_deg = math.degrees(math.atan2(y, x))
        rows.append(
            {
                "run": run_dir.name,
                "yaw_cmd_deg": float(yaw_cmd),
                "forward_m": x,
                "left_y_m": y,
                "residual_left_deg": residual_deg,
            }
        )

    if not rows:
        print("no usable runs")
        return 1

    print("run,yaw_cmd_deg,forward_m,left_y_m,residual_left_deg")
    for row in rows:
        print(
            f"{row['run']},{row['yaw_cmd_deg']:.4f},"
            f"{row['forward_m']:.3f},{row['left_y_m']:.3f},"
            f"{row['residual_left_deg']:.4f}"
        )

    if len(rows) >= 2:
        yaw = np.asarray([float(row["yaw_cmd_deg"]) for row in rows], dtype=np.float64)
        residual = np.asarray([float(row["residual_left_deg"]) for row in rows], dtype=np.float64)
        if float(np.ptp(yaw)) > 1e-6:
            slope, intercept = np.polyfit(yaw, residual, 1)
            zero_cmd = -float(intercept) / float(slope)
            print()
            print(f"fit: residual_left_deg = {slope:.4f} * yaw_cmd_deg + {intercept:.4f}")
            print(f"dummy yaw command for zero residual: {zero_cmd:+.4f} deg")
            print(
                "equivalent live option approximately: "
                f"--udp-origin-yaw-offset-deg {-zero_cmd:+.4f}"
            )
            print("(positive live offset rotates a straight outgoing path to the right in this codebase)")
    elif len(rows) == 1:
        row = rows[0]
        yaw_cmd = float(row["yaw_cmd_deg"])
        residual = float(row["residual_left_deg"])
        estimated_zero_cmd = yaw_cmd - residual
        print()
        print(f"single-run rough residual: {residual:+.4f} deg left")
        print(f"rough dummy yaw command for zero residual: {estimated_zero_cmd:+.4f} deg")
        print(
            "rough equivalent live option: "
            f"--udp-origin-yaw-offset-deg {-estimated_zero_cmd:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
