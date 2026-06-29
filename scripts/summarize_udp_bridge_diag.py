#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np


LD_GOAL_RE = re.compile(r"^ld(.+)_goal_y_m$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fm/udp_bridge.py path diagnostic CSV logs."
    )
    parser.add_argument("csv", nargs="+", help="Diagnostic CSV path(s), e.g. /tmp/udp_bridge_path_diag_*.csv")
    parser.add_argument("--event", choices=["recv", "control", "all"], default="recv")
    parser.add_argument("--name", action="append", default=[], help="Optional display name for each CSV.")
    return parser.parse_args()


def read_rows(path: Path, event: str) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if event != "all":
        rows = [row for row in rows if row.get("event") == event]
    return rows


def values(rows: Iterable[dict[str, str]], key: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        text = row.get(key, "")
        if text == "":
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def summarize_array(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:4d} mean={arr.mean():+.5f} med={np.median(arr):+.5f} "
        f"std={arr.std():.5f} p05={np.percentile(arr, 5):+.5f} "
        f"p95={np.percentile(arr, 95):+.5f} pos={100.0 * np.mean(arr > 0.0):5.1f}%"
    )


def ld_tags(fieldnames: list[str]) -> list[str]:
    tags = []
    for field in fieldnames:
        match = LD_GOAL_RE.match(field)
        if match:
            tags.append(match.group(1))
    return sorted(set(tags), key=lambda item: float(item))


def print_summary(path: Path, rows: list[dict[str, str]], name: str | None) -> None:
    label = name or path.name
    print(f"\n== {label} ==")
    print(f"path: {path}")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    fields = list(rows[0].keys())
    for key in ("source_age_s", "tx_age_s", "payload_actual_offset_s", "inference_time_s"):
        arr = values(rows, key)
        if arr.size:
            print(f"{key:28s} {summarize_array(arr)}")

    for tag in ld_tags(fields):
        for suffix in ("goal_y_m", "kappa_raw", "goal_y_delta_m"):
            key = f"ld{tag}_{suffix}"
            arr = values(rows, key)
            if arr.size:
                print(f"{key:28s} {summarize_array(arr)}")

    for key in ("std_y_m", "dy_step_rms_m", "d2y_step_rms_m", "max_abs_y_m"):
        arr = values(rows, key)
        if arr.size:
            print(f"{key:28s} {summarize_array(arr)}")


def main() -> None:
    args = parse_args()
    names = list(args.name)
    for idx, item in enumerate(args.csv):
        path = Path(item)
        rows = read_rows(path, args.event)
        name = names[idx] if idx < len(names) else None
        print_summary(path, rows, name)


if __name__ == "__main__":
    main()
