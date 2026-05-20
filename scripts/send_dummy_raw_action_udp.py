#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import socket
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "dummy_raw_action_udp.yaml"


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no}: expected 'key: value'")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")
        config[key] = _parse_scalar(value)
    return config


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))


def _int_config(config: dict[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def build_payload(config: dict[str, Any]) -> dict[str, Any]:
    num_points = _int_config(config, "num_points", 64)
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    plan_dt_s = _float_config(config, "plan_dt_s", 0.1)
    inference_time_s = _float_config(config, "inference_time_s", 0.3)
    accel_mps2_value = _float_config(config, "accel_mps2", 0.0)
    curvature = _float_config(config, "curvature", 0.0)
    yaw = _float_config(config, "initial_yaw_rad", 0.0)
    x_m = _float_config(config, "initial_x_m", 0.0)
    y_m = _float_config(config, "initial_y_m", 0.0)
    z_m = _float_config(config, "z_m", 0.0)

    # The control side requires path-like fields to have the same length as
    # raw_action. Build a coherent constant-velocity/constant-curvature trace.
    velocity_mps = accel_mps2_value
    pred_xyz: list[list[float]] = []
    pred_yaw_rad: list[float] = []
    for _ in range(num_points):
        yaw += curvature * velocity_mps * plan_dt_s
        x_m += velocity_mps * math.cos(yaw) * plan_dt_s
        y_m += velocity_mps * math.sin(yaw) * plan_dt_s
        pred_xyz.append([float(x_m), float(y_m), float(z_m)])
        pred_yaw_rad.append(float(yaw))

    return {
        "raw_action": {
            "accel_mps2": [float(accel_mps2_value)] * num_points,
            "curvature": [float(curvature)] * num_points,
        },
        "pred_xyz": pred_xyz,
        "pred_yaw_rad": pred_yaw_rad,
        "pred_v_mps": [float(velocity_mps)] * num_points,
        "plan_dt_s": float(plan_dt_s),
        "inference_time_s": float(inference_time_s),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send dummy Alpamayo raw_action JSON UDP packets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Simple YAML config path")
    parser.add_argument("--host", default=None, help="Override udp_host from config")
    parser.add_argument("--port", type=int, default=None, help="Override udp_port from config")
    parser.add_argument("--send-hz", type=float, default=None, help="Override send_hz from config")
    parser.add_argument("--once", action="store_true", help="Send one packet and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without sending")
    parser.add_argument("--print-payload", action="store_true", help="Print full JSON payload at startup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_simple_yaml(args.config)
    payload = build_payload(config)
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    udp_host = str(args.host or config.get("udp_host", "127.0.0.1"))
    udp_port = int(args.port or config.get("udp_port", 5005))
    send_hz = float(args.send_hz or config.get("send_hz", 10.0))
    if send_hz <= 0:
        raise ValueError("send_hz must be > 0")
    interval_s = 1.0 / send_hz

    accel = payload["raw_action"]["accel_mps2"][0]
    curvature = payload["raw_action"]["curvature"][0]
    print(
        "[dummy-udp] target="
        f"{udp_host}:{udp_port} send_hz={send_hz:.3f} points={len(payload['pred_xyz'])} "
        f"plan_dt_s={payload['plan_dt_s']:.3f} inference_time_s={payload['inference_time_s']:.3f} "
        f"accel_mps2_field={accel:.6f} curvature={curvature:.6f} bytes={len(payload_bytes)}"
    )
    if args.print_payload or args.dry_run:
        print(json.dumps(payload, indent=2))
    if args.dry_run:
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    next_log_t = 0.0
    try:
        while True:
            sock.sendto(payload_bytes, (udp_host, udp_port))
            sent += 1
            now = time.monotonic()
            if now >= next_log_t:
                print(f"[dummy-udp] sent={sent} last_bytes={len(payload_bytes)}")
                next_log_t = now + 3.0
            if args.once:
                break
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print(f"\n[dummy-udp] stopped sent={sent}")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
