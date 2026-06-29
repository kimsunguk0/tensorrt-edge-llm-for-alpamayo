#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "output" / "planner_live" / "results"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "planner_live_path_sessions"
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Alpamayo planner_live result_seq*.json paths over a time window. "
            "Outputs y@LD statistics, pure-pursuit implied curvature, compact CSV/NPZ, and PNG plots."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--since-unix", type=float, default=None)
    parser.add_argument("--until-unix", type=float, default=None)
    parser.add_argument("--last-n", type=int, default=0)
    parser.add_argument("--ld-m", default="5,10,15", help="Comma-separated LD/x distances to analyze.")
    parser.add_argument("--origin-offset-x-m", type=float, default=0.0)
    parser.add_argument("--origin-offset-y-m", type=float, default=0.0)
    parser.add_argument("--origin-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--min-forward-m", type=float, default=10.0)
    parser.add_argument("--max-plot-paths", type=int, default=50)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_ld_values(value: str) -> list[float]:
    lds: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        ld = float(token)
        if ld <= 0:
            raise ValueError("--ld-m values must be > 0")
        lds.append(ld)
    if not lds:
        raise ValueError("--ld-m must contain at least one value")
    return sorted(set(lds))


def seq_from_path(path: Path) -> int:
    match = re.search(r"result_seq(\d+)", path.name)
    return int(match.group(1)) if match else -1


def filtered_result_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(args.results_dir.glob("result_seq*.json"), key=lambda p: p.stat().st_mtime)
    if args.since_unix is not None:
        files = [path for path in files if path.stat().st_mtime >= float(args.since_unix)]
    if args.until_unix is not None:
        # Small grace window: result file writes can land just after child shutdown handling.
        files = [path for path in files if path.stat().st_mtime <= float(args.until_unix) + 1.0]
    if args.last_n and args.last_n > 0:
        files = files[-int(args.last_n) :]
    return files


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clean_xy(pred_xyz: Any) -> tuple[np.ndarray, np.ndarray] | None:
    xyz = np.asarray(pred_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 2 or xyz.shape[0] < 2:
        return None
    x = np.concatenate([np.zeros((1,), dtype=np.float64), xyz[:, 0].astype(np.float64)])
    y = np.concatenate([np.zeros((1,), dtype=np.float64), xyz[:, 1].astype(np.float64)])
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None
    return x, y


def apply_origin_offset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    offset_x_m: float,
    offset_y_m: float,
    yaw_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_offset_deg))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    translated_x = x.astype(np.float64) - float(offset_x_m)
    translated_y = y.astype(np.float64) + float(offset_y_m)
    x_send = translated_x * cos_y + translated_y * sin_y
    y_send = -translated_x * sin_y + translated_y * cos_y
    return x_send, y_send


def interp_y_at_x(x: np.ndarray, y: np.ndarray, target_x: float) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.nanmax(x)) < target_x:
        return None
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    last_x: float | None = None
    for xx, yy in zip(xs, ys):
        xx_f = float(xx)
        yy_f = float(yy)
        if last_x is not None and abs(xx_f - last_x) <= EPS:
            unique_y[-1] = yy_f
        else:
            unique_x.append(xx_f)
            unique_y.append(yy_f)
            last_x = xx_f
    if len(unique_x) < 2 or max(unique_x) < target_x:
        return None
    return float(np.interp(float(target_x), np.asarray(unique_x), np.asarray(unique_y)))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "positive_pct": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "positive_pct": float(np.mean(arr > 0.0) * 100.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    lds: list[float],
    primary_ld: float,
) -> None:
    if not rows:
        return
    def row_float(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        return float(value) if value is not None else float("nan")

    index = np.arange(len(rows), dtype=np.int32)
    model_y = np.asarray([row_float(row, f"model_y_at_{primary_ld:g}m") for row in rows], dtype=np.float64)
    out_y = np.asarray([row_float(row, f"outgoing_y_at_{primary_ld:g}m") for row in rows], dtype=np.float64)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=140)

    ax = axs[0, 0]
    ax.plot(index, model_y, linewidth=1.1, label=f"model y@{primary_ld:g}m")
    if np.isfinite(out_y).any() and np.nanmax(np.abs(out_y - model_y)) > 1e-4:
        ax.plot(index, out_y, linewidth=1.1, label=f"outgoing y@{primary_ld:g}m", alpha=0.8)
    if len(model_y) >= 20:
        kernel = np.ones(20, dtype=np.float64) / 20.0
        roll = np.convolve(np.nan_to_num(model_y, nan=0.0), kernel, mode="valid")
        ax.plot(np.arange(len(roll)) + 19, roll, color="crimson", linewidth=2.0, label="model rolling mean 20")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"y at LD={primary_ld:g}m over session")
    ax.set_xlabel("usable result index")
    ax.set_ylabel("y [m] (+ left)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axs[0, 1]
    values = model_y[np.isfinite(model_y)]
    ax.hist(values, bins=40, color="gray", edgecolor="white")
    if values.size:
        ax.axvline(float(np.mean(values)), color="crimson", label=f"mean {np.mean(values):+.3f}m")
        ax.axvline(float(np.median(values)), color="navy", label=f"median {np.median(values):+.3f}m")
        pos_pct = float(np.mean(values > 0.0) * 100.0)
        ax.set_title(f"model y@{primary_ld:g} distribution: pos {pos_pct:.1f}%")
    else:
        ax.set_title("model y distribution")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("y [m] (+ left)")
    ax.set_ylabel("count")
    ax.legend()

    ax = axs[1, 0]
    if paths:
        sample_indices = np.linspace(0, len(paths) - 1, min(len(paths), 50), dtype=np.int32)
        for idx in sample_indices:
            item = paths[int(idx)]
            x = item["model_x"]
            y = item["model_y"]
            y_ld = item.get("model_primary_y", np.nan)
            color = "crimson" if np.isfinite(y_ld) and y_ld > 0 else "steelblue"
            mask = (x >= 0.0) & (x <= max(max(lds) + 10.0, 25.0))
            ax.plot(x[mask], y[mask], color=color, alpha=0.32, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(primary_ld, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("sample model paths, red means y@LD > 0")
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left [m]")
    ax.grid(True, alpha=0.25)

    ax = axs[1, 1]
    kappa = np.asarray([row_float(row, f"model_pp_kappa_at_{primary_ld:g}m") for row in rows], dtype=np.float64)
    ax.scatter(model_y, kappa, s=8, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("pure pursuit implied curvature from y@LD")
    ax.set_xlabel(f"model y@{primary_ld:g}m [m] (+ left)")
    ax.set_ylabel("2*y/LD^2 [1/m]")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    lds = parse_ld_values(args.ld_m)
    primary_ld = 10.0 if 10.0 in lds else lds[0]
    output_dir = args.output_dir
    if output_dir is None:
        session_name = args.session_name or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
        output_dir = args.output_root / session_name
    ensure_dir(output_dir)

    files = filtered_result_files(args)
    rows: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    model_xyz_list: list[np.ndarray] = []
    outgoing_xyz_list: list[np.ndarray] = []

    for path in files:
        result = load_result(path)
        if result is None:
            continue
        xy = clean_xy(result.get("pred_xyz"))
        if xy is None:
            continue
        model_x, model_y = xy
        if float(np.nanmax(model_x)) < float(args.min_forward_m):
            continue
        out_x, out_y = apply_origin_offset(
            model_x,
            model_y,
            offset_x_m=float(args.origin_offset_x_m),
            offset_y_m=float(args.origin_offset_y_m),
            yaw_offset_deg=float(args.origin_yaw_offset_deg),
        )
        row: dict[str, Any] = {
            "seq": int(result.get("sequence", seq_from_path(path))),
            "t0_us": result.get("t0_us"),
            "mtime_unix": path.stat().st_mtime,
            "result_json": str(path),
            "plan_dt_s": result.get("plan_dt_s"),
            "model_x_end_m": float(model_x[-1]),
            "model_y_end_m": float(model_y[-1]),
            "outgoing_x_end_m": float(out_x[-1]),
            "outgoing_y_end_m": float(out_y[-1]),
            "fm_wall_ms": (result.get("fm_timing") or {}).get("wall_ms"),
            "guided_pass_ms": (result.get("post_vlm_timing") or {}).get("guided_pass_ms"),
            "total_post_vlm_ms": (result.get("post_vlm_timing") or {}).get("total_post_vlm_ms"),
        }
        for ld in lds:
            model_y_ld = interp_y_at_x(model_x, model_y, ld)
            outgoing_y_ld = interp_y_at_x(out_x, out_y, ld)
            row[f"model_y_at_{ld:g}m"] = model_y_ld
            row[f"outgoing_y_at_{ld:g}m"] = outgoing_y_ld
            row[f"model_pp_kappa_at_{ld:g}m"] = 2.0 * model_y_ld / (ld * ld) if model_y_ld is not None else None
            row[f"outgoing_pp_kappa_at_{ld:g}m"] = (
                2.0 * outgoing_y_ld / (ld * ld) if outgoing_y_ld is not None else None
            )
            row[f"model_equiv_yaw_deg_at_{ld:g}m"] = (
                math.degrees(math.atan2(model_y_ld, ld)) if model_y_ld is not None else None
            )
            row[f"outgoing_equiv_yaw_deg_at_{ld:g}m"] = (
                math.degrees(math.atan2(outgoing_y_ld, ld)) if outgoing_y_ld is not None else None
            )
        rows.append(row)
        paths.append(
            {
                "seq": row["seq"],
                "model_x": model_x.copy(),
                "model_y": model_y.copy(),
                "outgoing_x": out_x.copy(),
                "outgoing_y": out_y.copy(),
                "model_primary_y": row.get(f"model_y_at_{primary_ld:g}m"),
            }
        )
        if len(model_x) == 65:
            model_xyz_list.append(np.stack([model_x, model_y, np.zeros_like(model_x)], axis=-1))
            outgoing_xyz_list.append(np.stack([out_x, out_y, np.zeros_like(out_x)], axis=-1))

    rows = sorted(rows, key=lambda row: (float(row.get("mtime_unix") or 0.0), int(row.get("seq") or -1)))
    csv_path = output_dir / "path_ld_analysis.csv"
    write_csv(csv_path, rows)

    if model_xyz_list:
        np.savez_compressed(
            output_dir / "paths_compact.npz",
            model_xyz=np.stack(model_xyz_list, axis=0).astype(np.float32),
            outgoing_xyz=np.stack(outgoing_xyz_list, axis=0).astype(np.float32),
        )

    summary: dict[str, Any] = {
        "status": "ok" if rows else "no_usable_results",
        "results_dir": str(args.results_dir),
        "output_dir": str(output_dir),
        "file_count_selected": len(files),
        "usable_results": len(rows),
        "since_unix": args.since_unix,
        "until_unix": args.until_unix,
        "ld_m": lds,
        "primary_ld_m": primary_ld,
        "origin_offset_x_m": float(args.origin_offset_x_m),
        "origin_offset_y_m": float(args.origin_offset_y_m),
        "origin_yaw_offset_deg": float(args.origin_yaw_offset_deg),
        "csv": str(csv_path),
        "compact_npz": str(output_dir / "paths_compact.npz") if model_xyz_list else None,
    }
    if rows:
        summary.update(
            {
                "seq_first": rows[0]["seq"],
                "seq_last": rows[-1]["seq"],
                "mtime_first_unix": rows[0]["mtime_unix"],
                "mtime_last_unix": rows[-1]["mtime_unix"],
                "mtime_first_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(rows[0]["mtime_unix"]))),
                "mtime_last_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(rows[-1]["mtime_unix"]))),
            }
        )
        for frame in ("model", "outgoing"):
            for ld in lds:
                y_key = f"{frame}_y_at_{ld:g}m"
                k_key = f"{frame}_pp_kappa_at_{ld:g}m"
                yaw_key = f"{frame}_equiv_yaw_deg_at_{ld:g}m"
                summary[f"{y_key}_summary"] = summarize(
                    [float(row[y_key]) for row in rows if row.get(y_key) is not None]
                )
                summary[f"{k_key}_summary"] = summarize(
                    [float(row[k_key]) for row in rows if row.get(k_key) is not None]
                )
                summary[f"{yaw_key}_summary"] = summarize(
                    [float(row[yaw_key]) for row in rows if row.get(yaw_key) is not None]
                )
        summary["model_y_end_m_summary"] = summarize([float(row["model_y_end_m"]) for row in rows])
        summary["outgoing_y_end_m_summary"] = summarize([float(row["outgoing_y_end_m"]) for row in rows])
        summary["fm_wall_ms_summary"] = summarize(
            [float(row["fm_wall_ms"]) for row in rows if row.get("fm_wall_ms") is not None]
        )
        summary["guided_pass_ms_summary"] = summarize(
            [float(row["guided_pass_ms"]) for row in rows if row.get("guided_pass_ms") is not None]
        )
        primary_model = summary.get(f"model_y_at_{primary_ld:g}m_summary", {})
        primary_out = summary.get(f"outgoing_y_at_{primary_ld:g}m_summary", {})
        summary["interpretation"] = {
            "left_positive": True,
            "model_primary_left_bias": (
                primary_model.get("median") is not None and float(primary_model["median"]) > 0.0
            ),
            "model_primary_positive_pct": primary_model.get("positive_pct"),
            "outgoing_primary_positive_pct": primary_out.get("positive_pct"),
        }

    png_path = output_dir / "path_ld_analysis.png"
    plot_analysis(png_path, rows=rows, paths=paths, lds=lds, primary_ld=primary_ld)
    summary["png"] = str(png_path) if png_path.exists() else None

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
