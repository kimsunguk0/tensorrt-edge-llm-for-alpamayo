#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct FM replay_packet.pt files from Alpamayo post-VLM outputs plus dumped KV snapshots."
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing output_*.json files")
    parser.add_argument(
        "--nav-cache-root",
        type=Path,
        required=True,
        help="Directory containing per-request guided KV dumps exported via --dumpNavDualCache",
    )
    parser.add_argument("--packet-root", type=Path, required=True, help="Directory to write replay_packet.pt files")
    parser.add_argument(
        "--request-glob",
        default="output_*.json",
        help="Glob for output JSON files under --output-root (default: output_*.json)",
    )
    return parser.parse_args()


def tensor_from_json(obj: dict, dtype: torch.dtype) -> torch.Tensor:
    shape = list(obj["shape"])
    data = obj["data"]
    return torch.tensor(data, dtype=dtype).reshape(shape)


def build_attention_mask(active_len: int, horizon: int) -> torch.Tensor:
    width = active_len + horizon
    mask = torch.zeros((1, 1, horizon, width), dtype=torch.float32)
    return mask


def build_position_ids(active_len: int, rope_delta: int, horizon: int) -> torch.Tensor:
    values = torch.arange(active_len, active_len + horizon, dtype=torch.int64) + int(rope_delta)
    return values.view(1, 1, horizon).repeat(3, 1, 1)


def load_guided_dump(guided_dir: Path) -> tuple[torch.Tensor, int, int]:
    meta_path = guided_dir / "kv_cache_request_0.json"
    meta = json.loads(meta_path.read_text())

    kv_meta = meta["kv_cache"]
    kv_shape = list(kv_meta["shape"])
    kv_path = guided_dir / kv_meta["file"]
    kv_np = np.fromfile(kv_path, dtype=np.float16).reshape(kv_shape)

    active_len = int(meta["kv_cache_lengths"]["active_values"][0])
    rope_meta = meta.get("rope_deltas")
    rope_delta = 0
    if rope_meta:
        rope_path = guided_dir / rope_meta["file"]
        rope_np = np.fromfile(rope_path, dtype=np.int64).reshape(rope_meta["shape"])
        rope_delta = int(rope_np.reshape(-1)[0])

    kv_np = kv_np[..., :active_len, :]
    kv = torch.from_numpy(kv_np.copy())
    return kv, active_len, rope_delta


def main() -> None:
    args = parse_args()
    args.packet_root.mkdir(parents=True, exist_ok=True)

    output_paths = sorted(args.output_root.glob(args.request_glob))
    if not output_paths:
        raise RuntimeError(f"No outputs matched {args.request_glob} under {args.output_root}")

    built = []
    for output_path in output_paths:
        base = output_path.name.replace("output_", "").replace(".json", "")
        request_stem = f"request_{base}"
        guided_dir = args.nav_cache_root / request_stem / "guided"
        if not guided_dir.exists():
            raise FileNotFoundError(f"Missing guided dump directory: {guided_dir}")

        output_obj = json.loads(output_path.read_text())
        apv = output_obj["responses"][0]["alpamayo_post_vlm"]
        fm = apv["fm"]

        x0 = tensor_from_json(fm["x0"], torch.float32).contiguous()
        horizon = int(x0.shape[1])
        kv_cache, active_len, rope_delta = load_guided_dump(guided_dir)
        attention_mask = build_attention_mask(active_len, horizon)
        position_ids = build_position_ids(active_len, rope_delta, horizon)

        packet = {
            "x0": x0,
            "kv_cache": kv_cache.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "position_ids": position_ids.contiguous(),
            "action_space_constants": {
                key: float(value) for key, value in fm["action_space_constants"].items()
            },
            "source_output_json": str(output_path),
            "source_guided_dump_dir": str(guided_dir),
            "kv_active_len": int(active_len),
            "rope_delta": int(rope_delta),
        }

        packet_dir = args.packet_root / base
        packet_dir.mkdir(parents=True, exist_ok=True)
        packet_path = packet_dir / "replay_packet.pt"
        torch.save(packet, packet_path)
        built.append(
            {
                "sample": base,
                "packet_path": str(packet_path),
                "active_len": int(active_len),
                "rope_delta": int(rope_delta),
            }
        )

    summary_path = args.packet_root / "summary.json"
    summary_path.write_text(json.dumps({"count": len(built), "packets": built}, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
