#!/usr/bin/env python3

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pull latest CARLA sample from a remote producer and run llm_inference."
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://192.168.10.183:8765",
        help="Base URL of the PC-side sample server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Sleep interval when no new sample is available",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one fetch + one inference and exit",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=-1,
        help="Maximum number of inference runs. -1 means unlimited.",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=repo_root / "engines" / "llm_kv",
        help="Path to LLM engine directory",
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=repo_root / "engines" / "visual",
        help="Path to multimodal engine directory",
    )
    parser.add_argument(
        "--llm-inference-bin",
        type=Path,
        default=repo_root / "build" / "examples" / "llm" / "llm_inference",
        help="Path to llm_inference binary",
    )
    parser.add_argument(
        "--template-input",
        type=Path,
        default=repo_root / "input" / "requests" / "input_fixed_template.json",
        help="Template request JSON with image slots and traj placeholder",
    )
    parser.add_argument(
        "--live-request-file",
        type=Path,
        default=repo_root / "input" / "requests" / "input_live_runtime.json",
        help="Generated request JSON for each inference run",
    )
    parser.add_argument(
        "--live-image-dir",
        type=Path,
        default=repo_root / "input" / "images" / "live_runtime",
        help="Directory where fetched images are written",
    )
    parser.add_argument(
        "--live-ego-dir",
        type=Path,
        default=repo_root / "input" / "ego" / "live_runtime",
        help="Directory where fetched ego-history npy files are written",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output" / "runs" / "live_runtime",
        help="Directory for inference outputs",
    )
    parser.add_argument(
        "--kv-cache-dir",
        type=Path,
        default=repo_root / "output" / "kv_cache" / "live_runtime",
        help="Directory for KV-cache dumps",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup runs passed to llm_inference",
    )
    parser.add_argument(
        "--dump-profile",
        action="store_true",
        help="Enable llm_inference profile output",
    )
    parser.add_argument(
        "--dump-kv-cache",
        action="store_true",
        help="Enable llm_inference KV-cache dumps",
    )
    parser.add_argument(
        "--keep-last-only",
        action="store_true",
        help="Delete previous outputs before writing the next run",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fetch_latest_sample(server_url: str, timeout: float):
    latest_url = server_url.rstrip("/") + "/latest"
    request = urllib.request.Request(latest_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            headers = response.headers
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return None, None
        raise

    npz = np.load(io.BytesIO(payload), allow_pickle=False)
    sample = {
        "image_frames": npz["image_frames"],
        "camera_indices": npz["camera_indices"],
        "ego_history_xyz": npz["ego_history_xyz"],
        "ego_history_rot": npz["ego_history_rot"],
        "relative_timestamps": npz["relative_timestamps"],
        "absolute_timestamps": npz["absolute_timestamps"],
        "t0_us": int(npz["t0_us"][0]),
        "fixed_delta_seconds": float(npz["fixed_delta_seconds"][0]),
        "clip_id": str(npz["clip_id"][0]),
        "camera_order": [str(x) for x in npz["camera_order"].tolist()],
    }
    metadata = {
        "sequence": int(headers.get("X-Sample-Sequence", "0")),
        "t0_us": int(headers.get("X-T0-US", str(sample["t0_us"]))),
        "clip_id": headers.get("X-Clip-ID", sample["clip_id"]),
    }
    return sample, metadata


def write_sample_files(sample, image_dir: Path, ego_dir: Path):
    ensure_dir(image_dir)
    ensure_dir(ego_dir)

    image_frames = sample["image_frames"]
    num_cams, num_steps = image_frames.shape[:2]
    image_paths = []
    for cam_idx in range(num_cams):
        for step_idx in range(num_steps):
            image_hwc = np.transpose(image_frames[cam_idx, step_idx], (1, 2, 0))
            image_path = image_dir / f"cam{cam_idx}_f{step_idx}.png"
            Image.fromarray(image_hwc).save(image_path)
            image_paths.append(image_path)

    xyz_path = ego_dir / "ego_history_xyz.npy"
    rot_path = ego_dir / "ego_history_rot.npy"
    np.save(xyz_path, sample["ego_history_xyz"].astype(np.float32))
    np.save(rot_path, sample["ego_history_rot"].astype(np.float32))
    return image_paths, xyz_path, rot_path


def build_runtime_request(
    template_input: Path,
    output_request: Path,
    image_dir: Path,
    xyz_path: Path,
    rot_path: Path,
):
    data = json.loads(template_input.read_text())

    for request in data.get("requests", []):
        request["ego_history_xyz_npy"] = str(xyz_path)
        request["ego_history_rot_npy"] = str(rot_path)

        image_counter = 0
        for message in request.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type") == "image":
                    item["image"] = str(image_dir / f"cam{image_counter // 4}_f{image_counter % 4}.png")
                    image_counter += 1

        if image_counter != 16:
            raise RuntimeError(f"Expected 16 image slots in template, found {image_counter}")

    ensure_dir(output_request.parent)
    output_request.write_text(json.dumps(data, indent=2))


def remove_tree_contents(path: Path):
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_llm_inference(args, request_path: Path, run_name: str):
    ensure_dir(args.output_dir)
    output_file = args.output_dir / f"output_{run_name}.json"
    profile_file = args.output_dir / f"profile_{run_name}.json"
    kv_dir = args.kv_cache_dir / run_name

    cmd = [
        str(args.llm_inference_bin),
        "--engineDir",
        str(args.engine_dir),
        "--multimodalEngineDir",
        str(args.multimodal_engine_dir),
        "--inputFile",
        str(request_path),
        "--outputFile",
        str(output_file),
        "--warmup",
        str(args.warmup),
    ]

    if args.dump_profile:
        cmd.extend(["--dumpProfile", "--profileOutputFile", str(profile_file)])

    if args.dump_kv_cache:
        cmd.extend(["--dumpKVCache", "--kvCacheOutputDir", str(kv_dir)])

    print(">> Running llm_inference")
    print("   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_file, profile_file if args.dump_profile else None, kv_dir if args.dump_kv_cache else None


def main():
    args = parse_args()

    if not args.llm_inference_bin.exists():
        raise FileNotFoundError(f"llm_inference not found: {args.llm_inference_bin}")
    if not args.template_input.exists():
        raise FileNotFoundError(f"Template input not found: {args.template_input}")

    ensure_dir(args.live_image_dir)
    ensure_dir(args.live_ego_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.kv_cache_dir)

    last_sequence = None
    completed_runs = 0

    while args.max_runs < 0 or completed_runs < args.max_runs:
        sample, metadata = fetch_latest_sample(args.server_url, args.timeout)
        if sample is None:
            print(">> sample not ready, waiting...")
            time.sleep(args.poll_interval)
            continue

        sequence = metadata["sequence"]
        if last_sequence is not None and sequence == last_sequence:
            print(f">> no new sample yet (sequence={sequence}), waiting...")
            time.sleep(args.poll_interval)
            continue

        run_name = f"seq{sequence:06d}_t0_{metadata['t0_us']}"
        print(
            f">> Pulled sample sequence={sequence}, t0_us={metadata['t0_us']}, "
            f"clip_id={metadata['clip_id']}, image_frames={sample['image_frames'].shape}, "
            f"ego_history_xyz={sample['ego_history_xyz'].shape}"
        )

        if args.keep_last_only:
            remove_tree_contents(args.live_image_dir)
            remove_tree_contents(args.live_ego_dir)

        image_paths, xyz_path, rot_path = write_sample_files(sample, args.live_image_dir, args.live_ego_dir)
        _ = image_paths  # Paths are implied by the template rewrite.
        build_runtime_request(args.template_input, args.live_request_file, args.live_image_dir, xyz_path, rot_path)
        output_file, profile_file, kv_dir = run_llm_inference(args, args.live_request_file, run_name)

        print(f">> output: {output_file}")
        if profile_file is not None:
            print(f">> profile: {profile_file}")
        if kv_dir is not None:
            print(f">> kv_cache: {kv_dir}")

        last_sequence = sequence
        completed_runs += 1

        if args.once:
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(">> interrupted by user")
        sys.exit(130)
