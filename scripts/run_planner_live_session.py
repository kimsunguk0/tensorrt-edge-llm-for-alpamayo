#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "planner_live_path_sessions"
DEFAULT_PLANNER_OUTPUT_ROOT = REPO_ROOT / "output" / "planner_live"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run planner_live_service as a session, then analyze all generated paths "
            "between process start and stop."
        )
    )
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--ld-m", default="5,10,15")
    parser.add_argument("--min-forward-m", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --, e.g. -- python3 -m ...")
    return parser.parse_args()


def strip_leading_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def option_value(command: list[str], name: str, default: str | None = None) -> str | None:
    for idx, token in enumerate(command):
        if token == name and idx + 1 < len(command):
            return command[idx + 1]
        prefix = f"{name}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return default


def infer_results_dir(command: list[str], override: Path | None) -> Path:
    if override is not None:
        return override
    output_root = option_value(command, "--output-root")
    if output_root is None:
        return DEFAULT_PLANNER_OUTPUT_ROOT / "results"
    return Path(output_root) / "results"


def float_option(command: list[str], name: str, default: float) -> float:
    value = option_value(command, name)
    if value is None:
        return float(default)
    return float(value)


def main() -> int:
    args = parse_args()
    command = strip_leading_separator(list(args.command))
    if not command:
        raise SystemExit("missing command after --")

    session_name = args.session_name or time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
    session_dir = args.output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    results_dir = infer_results_dir(command, args.results_dir)
    offset_x = float_option(command, "--udp-origin-offset-x-m", 0.0)
    offset_y = float_option(command, "--udp-origin-offset-y-m", 0.0)
    yaw_offset_deg = float_option(command, "--udp-origin-yaw-offset-deg", 0.0)

    start_unix = time.time()
    metadata = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "start_unix": start_unix,
        "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_unix)),
        "command": command,
        "results_dir": str(results_dir),
        "origin_offset_x_m": offset_x,
        "origin_offset_y_m": offset_y,
        "origin_yaw_offset_deg": yaw_offset_deg,
    }
    (session_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[planner-session] start session={session_name}", flush=True)
    print(f"[planner-session] results_dir={results_dir}", flush=True)
    print(f"[planner-session] command={' '.join(command)}", flush=True)
    log_path = session_dir / "planner_stdout.log"

    returncode = 0
    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            returncode = int(proc.wait())
    except KeyboardInterrupt:
        print("\n[planner-session] Ctrl+C: stopping planner...", flush=True)
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
        returncode = int(proc.returncode if proc is not None and proc.returncode is not None else 130)
    finally:
        end_unix = time.time()

    metadata.update(
        {
            "end_unix": end_unix,
            "end_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_unix)),
            "duration_s": end_unix - start_unix,
            "returncode": returncode,
            "planner_stdout_log": str(log_path),
        }
    )
    (session_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    analyze_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_planner_live_paths.py"),
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(session_dir),
        "--since-unix",
        str(start_unix),
        "--until-unix",
        str(end_unix),
        "--ld-m",
        str(args.ld_m),
        "--min-forward-m",
        str(args.min_forward_m),
        "--origin-offset-x-m",
        str(offset_x),
        "--origin-offset-y-m",
        str(offset_y),
        "--origin-yaw-offset-deg",
        str(yaw_offset_deg),
    ]
    print(f"[planner-session] analyzing paths into {session_dir}", flush=True)
    analysis = subprocess.run(analyze_cmd, cwd=str(REPO_ROOT), text=True)
    if analysis.returncode != 0:
        print(f"[planner-session] analysis failed returncode={analysis.returncode}", flush=True)
        return analysis.returncode
    print(f"[planner-session] done: {session_dir}", flush=True)
    return returncode if returncode not in (0, 130, -2) else 0


if __name__ == "__main__":
    raise SystemExit(main())
