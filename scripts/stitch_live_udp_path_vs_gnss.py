#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch sent live UDP local paths into the path the controller would have followed between "
            "successive plan updates, then compare it against GNSS."
        )
    )
    parser.add_argument("--path-log", type=Path, required=True)
    parser.add_argument("--gnss-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-path-arc-m", type=float, default=2.0)
    parser.add_argument(
        "--max-segment-s",
        type=float,
        default=2.0,
        help="Cap how long one sent path may be stitched when the next packet is delayed or missing.",
    )
    parser.add_argument(
        "--skip-gaps-longer-than-s",
        type=float,
        default=3.0,
        help="Skip packet intervals longer than this as session gaps.",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def wrap_angle_rad(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception as exc:
                raise RuntimeError(f"failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def read_gnss_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp_ns = as_float(row.get("timestamp_utc_ns"))
            east = as_float(row.get("utm_easting_m"))
            north = as_float(row.get("utm_northing_m"))
            if not (math.isfinite(timestamp_ns) and math.isfinite(east) and math.isfinite(north)):
                continue
            yaw = as_float(row.get("ins_yaw_rad"))
            if not math.isfinite(yaw):
                yaw = as_float(row.get("yaw_rad"))
            vel_x = as_float(row.get("vel_x_mps"))
            vel_y = as_float(row.get("vel_y_mps"))
            if not math.isfinite(yaw) and math.isfinite(vel_x) and math.isfinite(vel_y):
                if math.hypot(vel_x, vel_y) > 0.2:
                    yaw = math.atan2(vel_y, vel_x)
            rows.append(
                {
                    "time_s": float(timestamp_ns) / 1_000_000_000.0,
                    "east": east,
                    "north": north,
                    "yaw": yaw,
                    "speed": math.hypot(vel_x, vel_y)
                    if math.isfinite(vel_x) and math.isfinite(vel_y)
                    else float("nan"),
                }
            )
    if len(rows) < 2:
        raise RuntimeError(f"not enough GNSS rows in {path}")
    rows.sort(key=lambda item: item["time_s"])
    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    yaw = np.asarray([row["yaw"] for row in rows], dtype=np.float64)
    finite_yaw = np.isfinite(yaw)
    if not finite_yaw.any():
        yaw = np.zeros_like(time_s)
    elif not finite_yaw.all():
        yaw = np.interp(time_s, time_s[finite_yaw], yaw[finite_yaw])
    return {
        "time_s": time_s,
        "east": np.asarray([row["east"] for row in rows], dtype=np.float64),
        "north": np.asarray([row["north"] for row in rows], dtype=np.float64),
        "yaw": np.unwrap(yaw),
        "speed": np.asarray([row["speed"] for row in rows], dtype=np.float64),
    }


def interp_gnss(gnss: dict[str, np.ndarray], times_s: np.ndarray) -> dict[str, np.ndarray]:
    src_t = gnss["time_s"]
    if np.any(times_s < src_t[0]) or np.any(times_s > src_t[-1]):
        raise ValueError("requested time outside GNSS range")
    return {
        "east": np.interp(times_s, src_t, gnss["east"]),
        "north": np.interp(times_s, src_t, gnss["north"]),
        "yaw": np.interp(times_s, src_t, gnss["yaw"]),
        "speed": np.interp(times_s, src_t, gnss["speed"]),
    }


def packet_arrays(points: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([as_float(point.get("x_m")) for point in points], dtype=np.float64),
        "y": np.asarray([as_float(point.get("y_m")) for point in points], dtype=np.float64),
        "yaw": np.asarray([as_float(point.get("yaw_rad")) for point in points], dtype=np.float64),
        "curvature": np.asarray([as_float(point.get("curvature")) for point in points], dtype=np.float64),
    }


def path_arc_m(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def local_to_world(
    *,
    origin_east: float,
    origin_north: float,
    origin_yaw: float,
    local_x: np.ndarray,
    local_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    east = origin_east + local_x * cos_yaw - local_y * sin_yaw
    north = origin_north + local_x * sin_yaw + local_y * cos_yaw
    return east, north


def world_to_local(
    *,
    origin_east: float,
    origin_north: float,
    origin_yaw: float,
    world_east: np.ndarray,
    world_north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dx = world_east - origin_east
    dy = world_north - origin_north
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    x = dx * cos_yaw + dy * sin_yaw
    y = -dx * sin_yaw + dy * cos_yaw
    return x, y


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def stitch(path_rows: list[dict[str, Any]], gnss: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    segments: list[dict[str, np.ndarray | int | float]] = []
    skipped: dict[str, int] = {}
    tx_times = []
    for row in path_rows:
        header = dict(row.get("packet_header") or {})
        tx_time_us = int(header.get("tx_time_us", 0))
        tx_times.append(tx_time_us / 1_000_000.0)

    for idx, row in enumerate(path_rows):
        header = dict(row.get("packet_header") or {})
        points = list(row.get("packet_points") or [])
        if len(points) < 2:
            skipped["not_enough_points"] = skipped.get("not_enough_points", 0) + 1
            continue
        tx_time_s = int(header.get("tx_time_us", 0)) / 1_000_000.0
        if not (gnss["time_s"][0] <= tx_time_s <= gnss["time_s"][-1]):
            skipped["tx_outside_gnss"] = skipped.get("tx_outside_gnss", 0) + 1
            continue
        next_tx_time_s = tx_times[idx + 1] if idx + 1 < len(tx_times) else None
        if next_tx_time_s is None:
            interval_s = min(float(args.max_segment_s), float(header.get("dt_s", 0.1)) * (len(points) - 1))
        else:
            interval_s = next_tx_time_s - tx_time_s
        if interval_s <= 0.0:
            skipped["non_positive_interval"] = skipped.get("non_positive_interval", 0) + 1
            continue
        if interval_s > float(args.skip_gaps_longer_than_s):
            skipped["session_gap"] = skipped.get("session_gap", 0) + 1
            continue

        arrays = packet_arrays(points)
        arc_m = path_arc_m(arrays["x"], arrays["y"])
        if arc_m < float(args.min_path_arc_m):
            skipped["short_path"] = skipped.get("short_path", 0) + 1
            continue

        dt_s = max(float(header.get("dt_s") or 0.1), EPS)
        rel_times_s = np.arange(len(points), dtype=np.float64) * dt_s
        segment_s = min(interval_s, float(args.max_segment_s), float(rel_times_s[-1]))
        keep = rel_times_s <= segment_s + EPS
        if int(np.count_nonzero(keep)) < 2:
            skipped["short_segment"] = skipped.get("short_segment", 0) + 1
            continue
        rel_times_s = rel_times_s[keep]
        x = arrays["x"][keep]
        y = arrays["y"][keep]

        origin = interp_gnss(gnss, np.asarray([tx_time_s], dtype=np.float64))
        east, north = local_to_world(
            origin_east=float(origin["east"][0]),
            origin_north=float(origin["north"][0]),
            origin_yaw=float(origin["yaw"][0]),
            local_x=x,
            local_y=y,
        )
        abs_times_s = tx_time_s + rel_times_s
        if np.any(abs_times_s > gnss["time_s"][-1]):
            skipped["segment_outside_gnss"] = skipped.get("segment_outside_gnss", 0) + 1
            continue
        gt = interp_gnss(gnss, abs_times_s)
        gt_east = gt["east"]
        gt_north = gt["north"]
        error = np.hypot(east - gt_east, north - gt_north)
        segments.append(
            {
                "plan_seq": int(header.get("plan_seq", -1)),
                "tx_seq": int(header.get("tx_seq", -1)),
                "tx_time_s": float(tx_time_s),
                "time_s": abs_times_s,
                "east": east,
                "north": north,
                "gt_east": gt_east,
                "gt_north": gt_north,
                "error": error,
                "arc_m": float(arc_m),
                "duration_s": float(rel_times_s[-1]),
            }
        )

    if not segments:
        raise RuntimeError(f"no stitchable segments; skipped={skipped}")

    stitched_time = np.concatenate([np.asarray(seg["time_s"], dtype=np.float64) for seg in segments])
    stitched_east = np.concatenate([np.asarray(seg["east"], dtype=np.float64) for seg in segments])
    stitched_north = np.concatenate([np.asarray(seg["north"], dtype=np.float64) for seg in segments])
    gt_east = np.concatenate([np.asarray(seg["gt_east"], dtype=np.float64) for seg in segments])
    gt_north = np.concatenate([np.asarray(seg["gt_north"], dtype=np.float64) for seg in segments])
    error = np.concatenate([np.asarray(seg["error"], dtype=np.float64) for seg in segments])

    return {
        "segments": segments,
        "skipped": skipped,
        "stitched_time_s": stitched_time,
        "stitched_east": stitched_east,
        "stitched_north": stitched_north,
        "gt_east": gt_east,
        "gt_north": gt_north,
        "error": error,
    }


def draw_plot(result: dict[str, Any], out_path: Path, *, title: str) -> None:
    stitched_east = np.asarray(result["stitched_east"], dtype=np.float64)
    stitched_north = np.asarray(result["stitched_north"], dtype=np.float64)
    gt_east = np.asarray(result["gt_east"], dtype=np.float64)
    gt_north = np.asarray(result["gt_north"], dtype=np.float64)
    times = np.asarray(result["stitched_time_s"], dtype=np.float64)
    error = np.asarray(result["error"], dtype=np.float64)

    origin_east = float(gt_east[0])
    origin_north = float(gt_north[0])
    if gt_east.size >= 2:
        origin_yaw = math.atan2(float(gt_north[-1] - gt_north[0]), float(gt_east[-1] - gt_east[0]))
    else:
        origin_yaw = 0.0
    stitched_x, stitched_y = world_to_local(
        origin_east=origin_east,
        origin_north=origin_north,
        origin_yaw=origin_yaw,
        world_east=stitched_east,
        world_north=stitched_north,
    )
    gt_x, gt_y = world_to_local(
        origin_east=origin_east,
        origin_north=origin_north,
        origin_yaw=origin_yaw,
        world_east=gt_east,
        world_north=gt_north,
    )

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(gt_x, gt_y, color="#111827", linewidth=2.2, label="actual GNSS")
    ax.plot(stitched_x, stitched_y, color="#dc2626", linewidth=2.0, label="stitched sent path")
    ax.scatter(gt_x[0], gt_y[0], s=42, color="#111827", zorder=5, label="start")
    ax.scatter(stitched_x[-1], stitched_y[-1], s=42, color="#dc2626", zorder=5, label="stitched end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")
    ax.set_title(title)
    ax.legend(loc="best")

    rel_t = times - times[0]
    ax_err.plot(rel_t, error, color="#7c3aed", linewidth=1.8)
    ax_err.grid(True, alpha=0.25)
    ax_err.set_xlabel("stitched time [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.set_title("Distance to actual GNSS at same timestamp")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(args.output_dir)
    error = np.asarray(result["error"], dtype=np.float64)
    times = np.asarray(result["stitched_time_s"], dtype=np.float64)
    summary = {
        "path_log": str(args.path_log),
        "gnss_csv": str(args.gnss_csv),
        "segments": int(len(result["segments"])),
        "stitched_points": int(error.size),
        "duration_s": float(times[-1] - times[0]) if times.size else 0.0,
        "skipped": result["skipped"],
        "max_segment_s": float(args.max_segment_s),
        "skip_gaps_longer_than_s": float(args.skip_gaps_longer_than_s),
        "error_m": summarize(error),
    }
    summary_path = args.output_dir / "stitched_live_path_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.output_dir / "stitched_live_path_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "stitched_east_m", "stitched_north_m", "gt_east_m", "gt_north_m", "error_m"])
        for row in zip(
            result["stitched_time_s"],
            result["stitched_east"],
            result["stitched_north"],
            result["gt_east"],
            result["gt_north"],
            result["error"],
        ):
            writer.writerow([float(value) for value in row])

    npz_path = args.output_dir / "stitched_live_path.npz"
    np.savez_compressed(
        npz_path,
        stitched_time_s=result["stitched_time_s"],
        stitched_east=result["stitched_east"],
        stitched_north=result["stitched_north"],
        gt_east=result["gt_east"],
        gt_north=result["gt_north"],
        error=result["error"],
    )

    plot_path = args.output_dir / "stitched_live_path_vs_gnss.png"
    title = args.title or "stitched sent path vs actual GNSS"
    draw_plot(result, plot_path, title=title)

    report_path = args.output_dir / "stitched_live_path_report.md"
    err = summary["error_m"]
    lines = [
        "# Stitched Live Path vs GNSS",
        "",
        f"- path log: `{args.path_log}`",
        f"- GNSS CSV: `{args.gnss_csv}`",
        f"- segments: {summary['segments']}",
        f"- stitched points: {summary['stitched_points']}",
        f"- duration: {summary['duration_s']:.3f} s",
        f"- skipped: `{summary['skipped']}`",
        "",
        "## Error",
        "",
        f"- mean: {err['mean']:.3f} m",
        f"- median: {err['median']:.3f} m",
        f"- p90: {err['p90']:.3f} m",
        f"- p95: {err['p95']:.3f} m",
        f"- max: {err['max']:.3f} m",
        "",
        f"![stitched path](./{plot_path.name})",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "points_csv": str(csv_path),
        "npz": str(npz_path),
        "plot_png": str(plot_path),
        "report_md": str(report_path),
        **summary,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.path_log)
    gnss = read_gnss_csv(args.gnss_csv)
    result = stitch(rows, gnss, args)
    output = write_outputs(result, args)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
