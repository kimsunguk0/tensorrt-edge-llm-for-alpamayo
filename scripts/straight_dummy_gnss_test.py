#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from send_dummy_raw_action_udp import build_payload, load_simple_yaml  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config" / "dummy_straight_path_udp.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "straight_dummy_gnss_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally send the straight dummy Alpamayo UDP path while logging "
            "live GNSS from /healthz, then summarize actual vehicle motion."
        )
    )
    parser.add_argument("--dummy-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--send-straight", action="store_true", help="Send the straight dummy path during logging.")
    parser.add_argument("--udp-host", default=None, help="Override udp_host in dummy config.")
    parser.add_argument("--udp-port", type=int, default=None, help="Override udp_port in dummy config.")
    parser.add_argument("--send-hz", type=float, default=None, help="Override send_hz in dummy config.")
    parser.add_argument("--dummy-speed-mps", type=float, default=None, help="Override dummy forward speed.")
    parser.add_argument("--dummy-curvature", type=float, default=None, help="Override dummy curvature [1/m].")
    parser.add_argument("--dummy-yaw-deg", type=float, default=None, help="Override dummy path initial yaw [deg].")
    parser.add_argument("--dummy-y-m", type=float, default=None, help="Override dummy path initial lateral y [m].")
    parser.add_argument("--dummy-x-m", type=float, default=None, help="Override dummy path initial x [m].")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--poll-hz", type=float, default=10.0)
    parser.add_argument("--sensor-health-url", default="http://127.0.0.1:18080/healthz")
    parser.add_argument("--health-timeout-s", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--heading-source",
        choices=("auto", "motion", "imu", "manual"),
        default="auto",
        help=(
            "Reference heading for local analysis. 'motion' uses early GNSS movement, "
            "'imu' uses healthz euler yaw, and 'manual' uses --reference-heading-deg."
        ),
    )
    parser.add_argument("--reference-heading-deg", type=float, default=None)
    parser.add_argument("--min-heading-distance-m", type=float, default=0.5)
    parser.add_argument("--max-heading-window-s", type=float, default=5.0)
    parser.add_argument("--print-every-s", type=float, default=1.0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(text: str, key: str, expected: int | None = None) -> list[float] | None:
    match = re.search(rf"{re.escape(key)}=\[([^\]]+)\]", text)
    if not match:
        return None
    try:
        values = [float(token.strip()) for token in match.group(1).split(",")]
    except ValueError:
        return None
    if expected is not None and len(values) != expected:
        return None
    return values


def parse_float_value(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([-+0-9.eE]+)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_int_value(text: str, key: str) -> int | None:
    value = parse_float_value(text, key)
    return int(value) if value is not None else None


def fetch_health(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_health_row(health: dict[str, Any], *, sample_idx: int) -> dict[str, Any]:
    now_unix = time.time()
    now_us = time.time_ns() // 1000
    gnss_text = str(health.get("gnss", ""))
    imu_text = str(health.get("imu", ""))
    latest_store = health.get("latest_store") if isinstance(health.get("latest_store"), dict) else {}
    latest_summary = (
        latest_store.get("latest_sample_summary") if isinstance(latest_store.get("latest_sample_summary"), dict) else {}
    )

    utm = parse_float_list(gnss_text, "utm", expected=3)
    vel = parse_float_list(gnss_text, "vel", expected=3)
    cov = parse_float_list(gnss_text, "cov_diag_m2", expected=3)
    euler = parse_float_list(imu_text, "euler_deg", expected=3)

    row: dict[str, Any] = {
        "sample_idx": sample_idx,
        "unix_s": now_unix,
        "host_time_us": now_us,
        "clip_id": health.get("clip_id"),
        "latest_seq": latest_store.get("latest_seq"),
        "latest_t0_us": latest_store.get("latest_t0_us"),
        "fix_type": parse_int_value(gnss_text, "fix_type"),
        "fix_age_ms": parse_float_value(gnss_text, "fix_age_ms"),
        "lat": parse_float_value(gnss_text, "lat"),
        "lon": parse_float_value(gnss_text, "lon"),
        "alt_m": parse_float_value(gnss_text, "alt"),
        "utm_e_m": utm[0] if utm else None,
        "utm_n_m": utm[1] if utm else None,
        "utm_z_m": utm[2] if utm else None,
        "vel_e_mps": vel[0] if vel else None,
        "vel_n_mps": vel[1] if vel else None,
        "vel_u_mps": vel[2] if vel else None,
        "cov_e_m2": cov[0] if cov else None,
        "cov_n_m2": cov[1] if cov else None,
        "cov_z_m2": cov[2] if cov else None,
        "imu_roll_deg": euler[0] if euler else None,
        "imu_pitch_deg": euler[1] if euler else None,
        "imu_yaw_deg": euler[2] if euler else None,
        "gnss_utm_valid": latest_summary.get("gnss_utm_valid"),
        "gnss_covariance_valid": latest_summary.get("gnss_covariance_valid"),
        "yaw_valid": latest_summary.get("yaw_valid"),
        "raw_gnss": gnss_text,
        "raw_imu": imu_text,
    }
    speed = None
    if row["vel_e_mps"] is not None and row["vel_n_mps"] is not None:
        speed = math.hypot(float(row["vel_e_mps"]), float(row["vel_n_mps"]))
    row["speed_mps"] = speed
    return row


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def choose_heading_rad(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float | None, str]:
    valid = [row for row in rows if finite_float(row.get("utm_e_m")) is not None and finite_float(row.get("utm_n_m")) is not None]
    if not valid:
        return None, "no_valid_utm"

    if args.heading_source == "manual":
        if args.reference_heading_deg is None:
            return None, "manual_heading_missing"
        return math.radians(float(args.reference_heading_deg)), "manual"

    if args.heading_source in {"auto", "motion"}:
        first = valid[0]
        e0 = float(first["utm_e_m"])
        n0 = float(first["utm_n_m"])
        t0 = float(first["unix_s"])
        max_t = t0 + max(float(args.max_heading_window_s), 0.0)
        for row in valid[1:]:
            if float(row["unix_s"]) > max_t:
                break
            de = float(row["utm_e_m"]) - e0
            dn = float(row["utm_n_m"]) - n0
            if math.hypot(de, dn) >= float(args.min_heading_distance_m):
                return math.atan2(dn, de), "early_motion"
        if args.heading_source == "motion":
            return None, "motion_too_short"

    imu_yaws = [
        math.radians(float(row["imu_yaw_deg"]))
        for row in valid[: max(1, min(20, len(valid)))]
        if finite_float(row.get("imu_yaw_deg")) is not None
    ]
    if args.heading_source in {"auto", "imu"} and imu_yaws:
        return float(np.angle(np.mean(np.exp(1j * np.asarray(imu_yaws, dtype=np.float64))))), "imu_euler_yaw"

    return None, "no_heading"


def analyze_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    valid = [row for row in rows if finite_float(row.get("utm_e_m")) is not None and finite_float(row.get("utm_n_m")) is not None]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "valid_utm_rows": len(valid),
        "health_errors": len(rows) - len(valid),
    }
    if len(valid) < 2:
        summary["status"] = "not_enough_valid_utm"
        return summary

    e = np.asarray([float(row["utm_e_m"]) for row in valid], dtype=np.float64)
    n = np.asarray([float(row["utm_n_m"]) for row in valid], dtype=np.float64)
    t = np.asarray([float(row["unix_s"]) for row in valid], dtype=np.float64)
    speed = np.asarray(
        [np.nan if finite_float(row.get("speed_mps")) is None else float(row["speed_mps"]) for row in valid],
        dtype=np.float64,
    )
    cov_e = np.asarray(
        [np.nan if finite_float(row.get("cov_e_m2")) is None else float(row["cov_e_m2"]) for row in valid],
        dtype=np.float64,
    )
    cov_n = np.asarray(
        [np.nan if finite_float(row.get("cov_n_m2")) is None else float(row["cov_n_m2"]) for row in valid],
        dtype=np.float64,
    )

    heading_rad, heading_source = choose_heading_rad(valid, args)
    summary["heading_source"] = heading_source
    if heading_rad is None:
        heading_rad = math.atan2(float(n[-1] - n[0]), float(e[-1] - e[0]))
        summary["heading_source_fallback"] = "overall_displacement"

    de = e - e[0]
    dn = n - n[0]
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    local_x = de * cos_h + dn * sin_h
    local_y = -de * sin_h + dn * cos_h
    step_dist = np.hypot(np.diff(e), np.diff(n))
    duration_s = max(float(t[-1] - t[0]), 0.0)
    path_length_m = float(np.sum(step_dist))
    displacement_m = float(math.hypot(float(e[-1] - e[0]), float(n[-1] - n[0])))
    avg_speed_path_mps = path_length_m / duration_s if duration_s > 0 else 0.0
    end_segment = min(len(valid) - 1, max(1, int(round(min(2.0, duration_s) * float(args.poll_hz)))))
    if end_segment > 0:
        end_heading = math.atan2(float(n[-1] - n[-1 - end_segment]), float(e[-1] - e[-1 - end_segment]))
        heading_change_deg = math.degrees(math.atan2(math.sin(end_heading - heading_rad), math.cos(end_heading - heading_rad)))
    else:
        heading_change_deg = None

    summary.update(
        {
            "status": "ok",
            "duration_s": duration_s,
            "path_length_m": path_length_m,
            "displacement_m": displacement_m,
            "avg_speed_path_mps": avg_speed_path_mps,
            "avg_speed_gnss_mps": float(np.nanmean(speed)) if np.isfinite(speed).any() else None,
            "max_speed_gnss_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else None,
            "heading_deg": math.degrees(heading_rad),
            "final_x_m": float(local_x[-1]),
            "final_y_m_left_positive": float(local_y[-1]),
            "mean_y_m_left_positive": float(np.mean(local_y)),
            "rms_y_m": float(np.sqrt(np.mean(local_y * local_y))),
            "max_left_y_m": float(np.max(local_y)),
            "max_right_y_m": float(np.min(local_y)),
            "max_abs_y_m": float(np.max(np.abs(local_y))),
            "end_heading_change_deg_left_positive": heading_change_deg,
            "cov_e_sigma_mean_m": float(np.nanmean(np.sqrt(cov_e))) if np.isfinite(cov_e).any() else None,
            "cov_n_sigma_mean_m": float(np.nanmean(np.sqrt(cov_n))) if np.isfinite(cov_n).any() else None,
            "cov_e_sigma_max_m": float(np.nanmax(np.sqrt(cov_e))) if np.isfinite(cov_e).any() else None,
            "cov_n_sigma_max_m": float(np.nanmax(np.sqrt(cov_n))) if np.isfinite(cov_n).any() else None,
            "first_utm": [float(e[0]), float(n[0]), finite_float(valid[0].get("utm_z_m"))],
            "last_utm": [float(e[-1]), float(n[-1]), finite_float(valid[-1].get("utm_z_m"))],
        }
    )
    return summary


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


def main() -> int:
    args = parse_args()
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.poll_hz <= 0:
        raise ValueError("--poll-hz must be > 0")

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = args.output_root / run_name
    ensure_dir(run_dir)

    config = load_simple_yaml(args.dummy_config)
    if args.dummy_speed_mps is not None:
        config["accel_mps2"] = float(args.dummy_speed_mps)
    if args.dummy_curvature is not None:
        config["curvature"] = float(args.dummy_curvature)
    if args.dummy_yaw_deg is not None:
        config["initial_yaw_rad"] = math.radians(float(args.dummy_yaw_deg))
    if args.dummy_y_m is not None:
        config["initial_y_m"] = float(args.dummy_y_m)
    if args.dummy_x_m is not None:
        config["initial_x_m"] = float(args.dummy_x_m)
    udp_host = str(args.udp_host or config.get("udp_host", "127.0.0.1"))
    udp_port = int(args.udp_port or config.get("udp_port", 5005))
    send_hz = float(args.send_hz or config.get("send_hz", 10.0))
    send_interval_s = 1.0 / send_hz if send_hz > 0 else None
    payload = build_payload(config)
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    metadata = {
        "dummy_config": str(args.dummy_config),
        "send_straight": bool(args.send_straight),
        "udp_host": udp_host,
        "udp_port": udp_port,
        "send_hz": send_hz,
        "duration_s": float(args.duration_s),
        "poll_hz": float(args.poll_hz),
        "sensor_health_url": args.sensor_health_url,
        "payload_bytes": len(payload_bytes),
        "curvature": payload.get("raw_action", {}).get("curvature", [None])[0],
        "velocity_mps": payload.get("raw_action", {}).get("accel_mps2", [None])[0],
        "dummy_initial_x_m": config.get("initial_x_m"),
        "dummy_initial_y_m": config.get("initial_y_m"),
        "dummy_initial_yaw_rad": config.get("initial_yaw_rad"),
        "dummy_initial_yaw_deg": math.degrees(float(config.get("initial_yaw_rad", 0.0))),
        "plan_points": len(payload.get("pred_xyz", [])),
        "plan_dt_s": payload.get("plan_dt_s"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.send_straight else None
    rows: list[dict[str, Any]] = []
    raw_jsonl = (run_dir / "health_raw.jsonl").open("w", encoding="utf-8")
    start = time.monotonic()
    next_send = start
    next_poll = start
    next_print = start
    sent = 0
    polled = 0
    print(
        "[straight-gnss-test] "
        f"send={args.send_straight} target={udp_host}:{udp_port} duration={args.duration_s:.1f}s "
        f"poll_hz={args.poll_hz:.1f} out={run_dir}",
        flush=True,
    )
    try:
        while True:
            now = time.monotonic()
            if now - start >= float(args.duration_s):
                break

            if sock is not None and send_interval_s is not None and now >= next_send:
                sock.sendto(payload_bytes, (udp_host, udp_port))
                sent += 1
                next_send += send_interval_s
                if next_send < now - send_interval_s:
                    next_send = now + send_interval_s

            if now >= next_poll:
                try:
                    health = fetch_health(str(args.sensor_health_url), float(args.health_timeout_s))
                    raw_jsonl.write(json.dumps({"host_time_us": time.time_ns() // 1000, "health": health}) + "\n")
                    row = parse_health_row(health, sample_idx=polled)
                except Exception as exc:
                    row = {
                        "sample_idx": polled,
                        "unix_s": time.time(),
                        "host_time_us": time.time_ns() // 1000,
                        "error": repr(exc),
                    }
                rows.append(row)
                polled += 1
                next_poll += 1.0 / float(args.poll_hz)

            if now >= next_print:
                last = rows[-1] if rows else {}
                print(
                    "[straight-gnss-test] "
                    f"t={now - start:5.1f}s sent={sent} rows={len(rows)} "
                    f"utm=({last.get('utm_e_m')},{last.get('utm_n_m')}) "
                    f"speed={last.get('speed_mps')}",
                    flush=True,
                )
                next_print = now + max(float(args.print_every_s), 0.2)

            sleep_until = min(next_poll, next_send if sock is not None and send_interval_s is not None else next_poll)
            time.sleep(max(0.001, min(0.02, sleep_until - time.monotonic())))
    except KeyboardInterrupt:
        print("\n[straight-gnss-test] interrupted", flush=True)
    finally:
        raw_jsonl.close()
        if sock is not None:
            sock.close()

    csv_path = run_dir / "gnss_log.csv"
    write_csv(csv_path, rows)
    summary = analyze_rows(rows, args)
    summary.update({"udp_sent_packets": sent, "csv": str(csv_path), "run_dir": str(run_dir)})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
