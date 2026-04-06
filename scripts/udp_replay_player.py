#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from control_team_replay_common import (
    COORD_MODE_LOCAL,
    FLAG_END_OF_STREAM,
    FLAG_LOOPED,
    FLAG_VALID,
    build_packet_dict,
    coord_mode_code,
    coord_mode_name,
    load_plan_bank,
    pack_packet,
    sample_plan_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved plan bank as UDP packets for control integration.")
    parser.add_argument("--plan-bank", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    parser.add_argument("--coord-mode", choices=["local", "world"], default="local")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--start-plan-index", type=int, default=0)
    parser.add_argument("--verbose-every", type=int, default=50)
    args = parser.parse_args()

    bank = load_plan_bank(args.plan_bank)
    control_dt = float(args.control_dt if args.control_dt is not None else bank.control_dt_default_s)
    control_points = int(args.control_points if args.control_points is not None else bank.control_points_default)
    coord_mode = coord_mode_code(args.coord_mode)

    if not (0 <= args.start_plan_index < len(bank.sample_id)):
        raise ValueError(f"start-plan-index out of range: {args.start_plan_index}")

    base_t_rel = float(bank.t_rel_s[args.start_plan_index])
    total_duration = float(bank.t_rel_s[-1] - base_t_rel)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    start_monotonic = time.perf_counter()
    tx_seq = 0
    last_plan_idx = args.start_plan_index

    print(
        json.dumps(
            {
                "plan_bank": args.plan_bank,
                "host": args.host,
                "port": args.port,
                "coord_mode": coord_mode_name(coord_mode),
                "control_dt_s": control_dt,
                "control_points": control_points,
                "num_samples": int(len(bank.sample_id)),
                "chunk_id": int(bank.chunk_id),
                "loop": bool(args.loop),
                "playback_rate": float(args.playback_rate),
            },
            indent=2,
        )
    )

    while True:
        wall_now = time.perf_counter()
        elapsed = (wall_now - start_monotonic) * args.playback_rate
        if args.max_seconds is not None and elapsed >= args.max_seconds:
            print(f"[replay] reached max-seconds={args.max_seconds:.3f}, stopping")
            break

        playback_t = base_t_rel + elapsed
        looped = False
        if playback_t > float(bank.t_rel_s[-1]):
            if not args.loop:
                playback_t = float(bank.t_rel_s[-1])
            else:
                if total_duration <= 1e-6:
                    playback_t = base_t_rel
                else:
                    playback_t = base_t_rel + ((playback_t - base_t_rel) % total_duration)
                looped = True

        plan_idx = int(max(args.start_plan_index, min(len(bank.sample_id) - 1, bank.t_rel_s.searchsorted(playback_t, side="right") - 1)))
        age_s = float(max(0.0, playback_t - float(bank.t_rel_s[plan_idx])))
        sampled = sample_plan_points(
            bank=bank,
            plan_idx=plan_idx,
            age_s=age_s,
            control_dt_s=control_dt,
            control_points=control_points,
            coord_mode=coord_mode,
        )
        flags = FLAG_VALID
        if looped:
            flags |= FLAG_LOOPED
        if plan_idx == len(bank.sample_id) - 1 and not args.loop:
            flags |= FLAG_END_OF_STREAM

        packet = build_packet_dict(
            tx_seq=tx_seq,
            plan_seq=plan_idx,
            sample_id=int(bank.sample_id[plan_idx]),
            source_t0_us=int(bank.t0_us[plan_idx]),
            tx_time_us=time.time_ns() // 1000,
            coord_mode=coord_mode,
            dt_s=control_dt,
            x=sampled["x"],
            y=sampled["y"],
            yaw=sampled["yaw"],
            v=sampled["v"],
            curvature=sampled["curvature"],
            flags=flags,
        )
        sock.sendto(pack_packet(packet), target)

        if tx_seq % max(1, args.verbose_every) == 0 or plan_idx != last_plan_idx:
            print(
                f"[replay] tx={tx_seq} plan_idx={plan_idx} sample_id={int(bank.sample_id[plan_idx])} "
                f"t_rel={float(bank.t_rel_s[plan_idx]):.3f}s age={age_s:.3f}s mode={coord_mode_name(coord_mode)}"
            )
        last_plan_idx = plan_idx
        tx_seq += 1

        if not args.loop and plan_idx == len(bank.sample_id) - 1 and age_s >= (bank.plan_dt_s * max(bank.plan_points - 1, 0)):
            print("[replay] reached final plan horizon, stopping")
            break

        next_deadline = start_monotonic + (tx_seq * control_dt) / max(args.playback_rate, 1e-6)
        sleep_s = next_deadline - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    sock.close()


if __name__ == "__main__":
    main()
