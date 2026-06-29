#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fm_model_defaults import first_existing_fm_engine


CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

ALPAMAYO_CANONICAL_CAMERA_ORDER = (0, 1, 2, 6)

DEFAULT_ACTION_SPACE_CONSTANTS = {
    "accel_mean": 0.029052734375,
    "accel_std": 0.6796875,
    "curvature_mean": 0.0002689361572265625,
    "curvature_std": 0.026123046875,
    "dt_value": 0.1,
    "v_lambda": 0.000001,
    "v_ridge": 0.0001,
}


def ordered_camera_positions(sample: dict[str, Any]) -> list[tuple[int, int]]:
    camera_indices = [int(x) for x in np.asarray(sample["camera_indices"]).tolist()]
    by_id = {cam_id: cam_pos for cam_pos, cam_id in enumerate(camera_indices)}
    if set(camera_indices) == set(ALPAMAYO_CANONICAL_CAMERA_ORDER):
        return [(by_id[cam_id], cam_id) for cam_id in ALPAMAYO_CANONICAL_CAMERA_ORDER]
    return list(enumerate(camera_indices))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pull live samples and run Alpamayo 1.5 + TRT-Edge-LLM v0.6.0 end-to-end."
    )
    parser.add_argument("--server-url", type=str, default="http://192.168.10.183:8765", help="Base URL of the sample server")
    parser.add_argument("--sample-endpoint", type=str, default="/latest", help="HTTP endpoint to poll for the newest sample")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Sleep interval when no new sample is available")
    parser.add_argument("--once", action="store_true", help="Run one fetch + one inference and exit")
    parser.add_argument("--max-runs", type=int, default=-1, help="Maximum number of inference runs. -1 means unlimited")

    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("/alpamayo_vlm_engines/alpa1.5"),
        help="Path to the LLM engine directory",
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/alpamayo_vlm_engines/alpa1.5"),
        help="Path to the multimodal engine directory",
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=first_existing_fm_engine(
            extra_candidates=[
                Path("/root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan"),
                Path("/root/test/output/alpamayo15_fm_one_step_mxfp8/alpamayo15_fm_one_step_mxfp8_thor.plan"),
            ]
        ),
        help="Path to the FM TRT engine",
    )
    parser.add_argument(
        "--llm-inference-bin",
        type=Path,
        default=repo_root / "build" / "examples" / "llm" / "llm_inference",
        help="Path to llm_inference binary",
    )
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=repo_root / "build" / "libNvInfer_edgellm_plugin.so",
        help="Path to libNvInfer_edgellm_plugin.so",
    )

    parser.add_argument(
        "--live-request-file",
        type=Path,
        default=repo_root / "input" / "requests" / "input_live_runtime_alpamayo15.json",
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
        "--trajectory-dir",
        type=Path,
        default=repo_root / "output" / "trajectories" / "live_runtime",
        help="Directory for extracted predicted trajectories",
    )
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=repo_root / "output" / "dashboards" / "live_runtime",
        help="Directory for dashboard PNG outputs",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs passed to llm_inference")
    parser.add_argument("--dump-profile", action="store_true", help="Enable llm_inference profile output")
    parser.add_argument("--dump-kv-cache", action="store_true", help="Enable llm_inference KV-cache dumps")
    parser.add_argument(
        "--persistent-llm-inference",
        action="store_true",
        help="Keep a single llm_inference process alive and send request/output file pairs over stdin",
    )
    parser.add_argument("--keep-last-only", action="store_true", help="Delete previous live inputs before writing the next run")

    parser.add_argument("--max-generate-length", type=int, default=20, help="Generation length for the VLM stage")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling")
    parser.add_argument(
        "--traj-token-offset",
        type=int,
        default=3000,
        help="Discrete trajectory token offset used by Alpamayo 1.5",
    )
    parser.add_argument("--diffusion-seed", type=int, default=42, help="Seed for the post-VLM FM diffusion loop")
    parser.add_argument("--diffusion-num-steps", type=int, default=2, help="Number of FM diffusion steps")
    parser.add_argument(
        "--alpamayo-fm-use-prefill-kv",
        action="store_true",
        help="Feed backbone prefill KV directly into FM and skip CoT decode in llm_inference",
    )
    parser.add_argument("--action-space-constants-json", type=Path, default=None, help="Optional JSON file overriding action-space constants")

    parser.add_argument("--nav-text", type=str, default=None, help="Optional navigation instruction")
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0, help="Guidance weight for nav CFG")
    parser.add_argument("--dump-nav-dual-cache", action="store_true", help="Dump guided/unguided dual-cache artifacts")
    parser.add_argument(
        "--nav-cache-output-dir",
        type=Path,
        default=repo_root / "output" / "nav_cache" / "live_runtime",
        help="Directory for guided/unguided nav cache dumps",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_tree_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def fetch_latest_sample(server_url: str, sample_endpoint: str, timeout: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    latest_url = server_url.rstrip("/") + "/" + sample_endpoint.lstrip("/")
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


def _image_extension(image_format: str) -> str:
    normalized = str(image_format).lower()
    if normalized == "png":
        return ".png"
    if normalized == "ppm":
        return ".ppm"
    raise ValueError(f"unsupported image format: {image_format!r}")


def _write_rgb_image(image_hwc: np.ndarray, image_path: Path, image_format: str) -> None:
    normalized = str(image_format).lower()
    if normalized == "png":
        Image.fromarray(image_hwc).save(image_path, compress_level=0)
        return
    if normalized == "ppm":
        contiguous = np.ascontiguousarray(image_hwc, dtype=np.uint8)
        height, width, channels = contiguous.shape
        if channels != 3:
            raise ValueError(f"PPM writer expects RGB image with 3 channels, got shape={contiguous.shape}")
        with image_path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(contiguous.tobytes())
        return
    raise ValueError(f"unsupported image format: {image_format!r}")


def write_sample_files(
    sample: dict[str, Any],
    image_dir: Path,
    ego_dir: Path,
    *,
    image_format: str = "png",
) -> tuple[list[Path], Path, Path]:
    ensure_dir(image_dir)
    ensure_dir(ego_dir)

    image_frames = sample["image_frames"]
    camera_indices = [int(x) for x in np.asarray(sample["camera_indices"]).tolist()]
    num_cams, num_steps = image_frames.shape[:2]
    if num_cams != len(camera_indices):
        raise RuntimeError(f"camera_indices length mismatch: num_cams={num_cams}, indices={camera_indices}")

    image_paths: list[Path] = []
    image_suffix = _image_extension(image_format)
    for cam_pos, cam_id in ordered_camera_positions(sample):
        for step_idx in range(num_steps):
            image_hwc = np.transpose(image_frames[cam_pos, step_idx], (1, 2, 0))
            image_path = image_dir / f"cam{cam_id}_f{step_idx}{image_suffix}"
            _write_rgb_image(image_hwc, image_path, image_format)
            image_paths.append(image_path)

    xyz_path = ego_dir / "ego_history_xyz.npy"
    rot_path = ego_dir / "ego_history_rot.npy"
    np.save(xyz_path, sample["ego_history_xyz"].astype(np.float32))
    np.save(rot_path, sample["ego_history_rot"].astype(np.float32))
    return image_paths, xyz_path, rot_path


def load_action_space_constants(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_ACTION_SPACE_CONSTANTS)
    data = json.loads(path.read_text())
    required = set(DEFAULT_ACTION_SPACE_CONSTANTS.keys())
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"Missing action-space constants keys: {missing}")
    return {key: float(data[key]) for key in DEFAULT_ACTION_SPACE_CONSTANTS.keys()}


def build_user_content(
    image_dir: Path,
    camera_indices: list[int],
    num_frames_per_camera: int,
    nav_text: str | None,
    image_format: str = "png",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_suffix = _image_extension(image_format)
    for cam_id in camera_indices:
        cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
        content.append({"type": "text", "text": f"{cam_name}: "})
        for frame_idx in range(num_frames_per_camera):
            image_path = image_dir / f"cam{cam_id}_f{frame_idx}{image_suffix}"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing frame file: {image_path}")
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image", "image": str(image_path)})

    hist_placeholder = "<|traj_history_start|>" + ("<|traj_history|>" * 48) + "<|traj_history_end|>"
    route_section = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else ""
    prompt_text = (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
    content.append({"type": "text", "text": f"{hist_placeholder}{route_section}{prompt_text}"})
    return content


def build_inline_user_content(
    sample: dict[str, Any],
    nav_text: str | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_frames = np.asarray(sample["image_frames"])
    num_frames_per_camera = int(image_frames.shape[1])
    for cam_pos, cam_id in ordered_camera_positions(sample):
        cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
        content.append({"type": "text", "text": f"{cam_name}: "})
        for frame_idx in range(num_frames_per_camera):
            image_hwc = np.ascontiguousarray(np.transpose(image_frames[cam_pos, frame_idx], (1, 2, 0)), dtype=np.uint8)
            height, width, channels = image_hwc.shape
            if channels != 3:
                raise ValueError(f"inline image expects RGB shape [H,W,3], got {image_hwc.shape}")
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append(
                {
                    "type": "image",
                    "image": f"inline://cam{cam_id}_f{frame_idx}.rgb",
                    "image_rgb_u8": {
                        "shape": [int(height), int(width), int(channels)],
                        "encoding": "base64",
                        "data_b64": base64.b64encode(image_hwc.tobytes()).decode("ascii"),
                    },
                }
            )

    hist_placeholder = "<|traj_history_start|>" + ("<|traj_history|>" * 48) + "<|traj_history_end|>"
    route_section = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else ""
    prompt_text = (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
    content.append({"type": "text", "text": f"{hist_placeholder}{route_section}{prompt_text}"})
    return content


def build_runtime_request(
    *,
    sample: dict[str, Any],
    xyz_path: Path | None,
    rot_path: Path | None,
    image_dir: Path | None,
    output_request: Path,
    action_space_constants: dict[str, float],
    nav_text: str | None,
    nav_guidance_weight: float,
    traj_token_offset: int,
    diffusion_seed: int,
    diffusion_num_steps: int,
    max_generate_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    image_format: str = "png",
    inline_inputs: bool = False,
    json_indent: int | None = 2,
) -> dict[str, Any]:
    camera_indices = [cam_id for _, cam_id in ordered_camera_positions(sample)]
    num_frames_per_camera = int(sample["image_frames"].shape[1])
    if inline_inputs:
        user_content = build_inline_user_content(sample, nav_text=nav_text)
    else:
        if xyz_path is None or rot_path is None or image_dir is None:
            raise ValueError("file-based runtime request requires xyz_path, rot_path, and image_dir")
        user_content = build_user_content(
            image_dir=image_dir,
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames_per_camera,
            nav_text=nav_text,
            image_format=image_format,
        )

    request_item: dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a driving assistant that generates safe and accurate actions.",
                    }
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<|cot_start|>"}],
            },
        ],
        "traj_token_offset": traj_token_offset,
        "action_space_constants": action_space_constants,
        "diffusion_seed": int(diffusion_seed),
        "diffusion_num_steps": int(diffusion_num_steps),
    }
    if inline_inputs:
        ego_xyz = np.asarray(sample["ego_history_xyz"], dtype=np.float32)
        ego_rot = np.asarray(sample["ego_history_rot"], dtype=np.float32)
        request_item["ego_history_xyz"] = {
            "shape": [int(x) for x in ego_xyz.shape],
            "dtype": "float32",
            "data": ego_xyz.reshape(-1).astype(float).tolist(),
        }
        request_item["ego_history_rot"] = {
            "shape": [int(x) for x in ego_rot.shape],
            "dtype": "float32",
            "data": ego_rot.reshape(-1).astype(float).tolist(),
        }
    else:
        request_item["ego_history_xyz_npy"] = str(xyz_path)
        request_item["ego_history_rot_npy"] = str(rot_path)
    if nav_text:
        request_item["nav_text"] = nav_text
        request_item["nav_guidance_weight"] = float(nav_guidance_weight)

    request = {
        "batch_size": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_generate_length": max_generate_length,
        "apply_chat_template": True,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "enable_thinking": False,
        "requests": [request_item],
    }

    ensure_dir(output_request.parent)
    if json_indent is None:
        output_request.write_text(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
    else:
        output_request.write_text(json.dumps(request, indent=int(json_indent), ensure_ascii=False))
    return request


class PersistentLLMInferenceClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen[str] | None = None
        self.quiet_logs = bool(getattr(args, "quiet_llm_logs", False))

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.args.plugin_lib.exists():
            env["EDGELLM_PLUGIN_PATH"] = str(self.args.plugin_lib)
            build_dir = str(self.args.plugin_lib.parent)
            env["LD_LIBRARY_PATH"] = build_dir + (
                f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
            )
        return env

    def _build_cmd(self) -> list[str]:
        cmd = [
            str(self.args.llm_inference_bin),
            "--engineDir",
            str(self.args.engine_dir),
            "--multimodalEngineDir",
            str(self.args.multimodal_engine_dir),
            "--fmEngine",
            str(self.args.fm_engine),
            "--alpamayoPostVlmRuntime",
            "--persistentServer",
            "--warmup",
            str(self.args.warmup),
        ]
        if self.args.alpamayo_fm_use_prefill_kv:
            cmd.append("--alpamayoFmUsePrefillKv")
        if self.args.nav_text:
            cmd.append("--alpamayoNavCfg")
        return cmd

    def _read_status(self, timeout_sec: float) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Persistent llm_inference process is not running")
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Persistent llm_inference exited early with code {self.process.returncode}")
                time.sleep(0.01)
                continue
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                if not self.quiet_logs:
                    print(f">> llm_inference: {text}")
                continue
            if isinstance(obj, dict) and "status" in obj:
                return obj
        raise TimeoutError("Timed out waiting for persistent llm_inference status response")

    def start(self) -> None:
        if self.process is not None:
            return
        if self.args.dump_profile:
            print(">> Warning: --dump-profile is not supported in persistent llm_inference mode yet; ignoring profile output.")
        cmd = self._build_cmd()
        if not self.quiet_logs:
            print(">> Starting persistent llm_inference")
            print("   " + " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._build_env(),
        )
        ready = self._read_status(timeout_sec=120.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Persistent llm_inference did not become ready: {ready}")
        if not self.quiet_logs:
            print(">> Persistent llm_inference ready")

    def run(self, request_path: Path, output_file: Path) -> None:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        command = {"input_file": str(request_path), "output_file": str(output_file)}
        if bool(getattr(self.args, "compact_output_json", False)):
            command["compact_output_json"] = True
        if bool(getattr(self.args, "minimal_output_json", False)):
            command["minimal_output_json"] = True
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        status = self._read_status(timeout_sec=300.0)
        if status.get("status") != "ok":
            raise RuntimeError(f"Persistent llm_inference request failed: {status}")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
            self._read_status(timeout_sec=10.0)
        except Exception:
            pass
        finally:
            try:
                self.process.wait(timeout=20.0)
            except Exception:
                self.process.kill()
            self.process = None


def run_llm_inference(
    args: argparse.Namespace,
    request_path: Path,
    run_name: str,
    persistent_client: PersistentLLMInferenceClient | None = None,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    ensure_dir(args.output_dir)
    output_file = args.output_dir / f"output_{run_name}.json"
    profile_file = args.output_dir / f"profile_{run_name}.json"
    kv_dir = args.kv_cache_dir / run_name
    nav_cache_dir = args.nav_cache_output_dir / run_name

    if persistent_client is not None:
        persistent_client.run(request_path=request_path, output_file=output_file)
        return (
            output_file,
            None,
            None,
            None,
        )

    cmd = [
        str(args.llm_inference_bin),
        "--engineDir",
        str(args.engine_dir),
        "--multimodalEngineDir",
        str(args.multimodal_engine_dir),
        "--fmEngine",
        str(args.fm_engine),
        "--alpamayoPostVlmRuntime",
        "--inputFile",
        str(request_path),
        "--outputFile",
        str(output_file),
        "--warmup",
        str(args.warmup),
    ]

    if args.nav_text:
        cmd.append("--alpamayoNavCfg")

    if args.alpamayo_fm_use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")

    if args.dump_profile:
        cmd.extend(["--dumpProfile", "--profileOutputFile", str(profile_file)])

    if args.dump_kv_cache:
        cmd.extend(["--dumpKVCache", "--kvCacheOutputDir", str(kv_dir)])

    if args.dump_nav_dual_cache:
        cmd.extend(["--dumpNavDualCache", "--navCacheOutputDir", str(nav_cache_dir)])

    print(">> Running llm_inference")
    print("   " + " ".join(cmd))
    env = dict(os.environ)
    if args.plugin_lib.exists():
        env["EDGELLM_PLUGIN_PATH"] = str(args.plugin_lib)
        build_dir = str(args.plugin_lib.parent)
        env["LD_LIBRARY_PATH"] = build_dir + (
            f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
        )
        print(f">> Using EDGELLM_PLUGIN_PATH={args.plugin_lib}")
    else:
        print(f">> Warning: plugin library not found at {args.plugin_lib}")
    subprocess.run(cmd, check=True, env=env)
    return (
        output_file,
        profile_file if args.dump_profile else None,
        kv_dir if args.dump_kv_cache else None,
        nav_cache_dir if args.dump_nav_dual_cache else None,
    )


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    data = np.asarray(field["data"], dtype=np.float32)
    shape = tuple(int(x) for x in field["shape"])
    return data.reshape(shape)


def load_stage_times(profile_file: Path | None) -> dict[str, float]:
    if profile_file is None or not profile_file.exists():
        return {}
    obj = json.loads(profile_file.read_text())
    out: dict[str, float] = {}
    for stage in obj.get("stages", []):
        out[str(stage["stage_id"])] = float(stage["average_time_per_run_ms"])
    return out


def extract_cot_text(output_text: str | None) -> str:
    if not output_text:
        return ""
    text = str(output_text)
    if "<|cot_end|>" not in text:
        return ""
    return text.split("<|cot_end|>", 1)[0].strip()


def extract_final_text(output_text: str | None) -> str:
    if not output_text:
        return ""
    text = str(output_text)
    if "<|cot_end|>" in text:
        text = text.split("<|cot_end|>", 1)[1]
    return text.strip()


def summarize_dashboard_timing(
    *,
    response: dict[str, Any],
    profile_file: Path | None,
) -> list[str]:
    post_vlm = response.get("alpamayo_post_vlm", {})
    fm = post_vlm.get("fm", {})
    fm_timing = fm.get("timing", {})
    post_timing = post_vlm.get("timing", {})
    stages = load_stage_times(profile_file)

    vision_ms = stages.get("vision_encoder")
    prefill_ms = stages.get("llm_prefill")
    generation_ms = stages.get("llm_generation")
    guided_pass_ms = post_timing.get("guided_pass_ms")
    preprocess_est = None
    if guided_pass_ms is not None and vision_ms is not None and prefill_ms is not None and generation_ms is not None:
        preprocess_est = max(float(guided_pass_ms) - vision_ms - prefill_ms - generation_ms, 0.0)
    llm_total_ms = None
    if prefill_ms is not None or generation_ms is not None:
        llm_total_ms = float(prefill_ms or 0.0) + float(generation_ms or 0.0)
    fm_total_ms = None
    if "fm_wall_ms" in post_timing:
        fm_total_ms = float(post_timing["fm_wall_ms"])

    lines = []
    if preprocess_est is not None:
        lines.append(f"preprocess: {preprocess_est:.2f} ms")
    if vision_ms is not None:
        lines.append(f"vit: {vision_ms:.2f} ms")
    if llm_total_ms is not None:
        lines.append(f"llm: {llm_total_ms:.2f} ms")
    if fm_total_ms is not None:
        lines.append(f"fm: {fm_total_ms:.2f} ms")
    if preprocess_est is not None or vision_ms is not None or llm_total_ms is not None or fm_total_ms is not None:
        lines.append("---")
    if vision_ms is not None:
        lines.append(f"vision_encoder: {vision_ms:.2f} ms")
    if prefill_ms is not None:
        lines.append(f"llm_prefill: {prefill_ms:.2f} ms")
    if generation_ms is not None:
        lines.append(f"llm_generation: {generation_ms:.2f} ms")
    if "ego_history_load_ms" in post_timing:
        lines.append(f"ego_history_load: {float(post_timing['ego_history_load_ms']):.2f} ms")
    if "fm_wall_ms" in post_timing:
        lines.append(f"fm_wall: {float(post_timing['fm_wall_ms']):.2f} ms")
    if "branch_prepare_ms" in fm_timing:
        lines.append(f"fm_branch_prepare: {float(fm_timing['branch_prepare_ms']):.2f} ms")
    if "engine_step_total_ms" in fm_timing:
        lines.append(f"fm_engine_step_total: {float(fm_timing['engine_step_total_ms']):.2f} ms")
    if "decode_postprocess_ms" in fm_timing:
        lines.append(f"fm_decode_post: {float(fm_timing['decode_postprocess_ms']):.2f} ms")
    if "total_post_vlm_ms" in post_timing:
        lines.append(f"total_post_vlm: {float(post_timing['total_post_vlm_ms']):.2f} ms")
    return lines


def save_live_dashboard(
    *,
    sample: dict[str, Any],
    output_file: Path,
    profile_file: Path | None,
    dashboard_dir: Path,
    run_name: str,
    metadata: dict[str, Any],
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    result = json.loads(output_file.read_text())
    response = result["responses"][0]
    post_vlm = response["alpamayo_post_vlm"]
    fm = post_vlm["fm"]

    pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
    pred_rot = reshape_tensor_field(fm["pred_rot"])[0]
    hist_xyz = np.asarray(sample["ego_history_xyz"], dtype=np.float32)[0, 0]
    hist_rot = np.asarray(sample["ego_history_rot"], dtype=np.float32)[0, 0]
    imgs = np.asarray(sample["image_frames"], dtype=np.uint8)[:, -1]
    camera_titles = [str(x).replace("_", " ").title() for x in sample["camera_order"]]
    cot_text = extract_cot_text(response.get("output_text"))
    final_text = extract_final_text(response.get("output_text"))
    timing_lines = summarize_dashboard_timing(response=response, profile_file=profile_file)

    fig = plt.figure(figsize=(18.6, 11.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05], width_ratios=[1.0, 1.0, 1.18])

    image_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    for ax, img, title in zip(image_axes, imgs, camera_titles):
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    right_gs = gs[0:2, 2].subgridspec(2, 1, height_ratios=[1.28, 0.92], hspace=0.12)
    ax_traj = fig.add_subplot(right_gs[0, 0])
    ax_cot = fig.add_subplot(right_gs[1, 0])
    ax_cot.axis("off")

    header_lines = [
        f"sequence: {int(metadata['sequence'])}",
        f"clip_id: {metadata['clip_id']}",
        f"t0_us: {int(metadata['t0_us'])}",
        f"fm_mode: {post_vlm.get('fm_mode')}",
        f"fm_status: {post_vlm.get('fm_status')}",
    ]
    nav_info = post_vlm.get("nav")
    if nav_info:
        header_lines.append(f"nav: {json.dumps(nav_info, ensure_ascii=False)}")

    hist_display_x = hist_xyz[:, 1]
    hist_display_y = hist_xyz[:, 0]
    pred_display_x = pred_xyz[:, 1]
    pred_display_y = pred_xyz[:, 0]
    pred_path_display = np.vstack([hist_xyz[-1:, 1], hist_xyz[-1:, 0]]).T
    pred_path_display = np.vstack([pred_path_display, np.stack([pred_xyz[:, 1], pred_xyz[:, 0]], axis=1)])

    ax_traj.plot(hist_display_x, hist_display_y, color="#1f77b4", linewidth=2.0, alpha=0.8, label="ego history")
    hist_step = 3
    ax_traj.quiver(
        hist_display_x[::hist_step],
        hist_display_y[::hist_step],
        hist_rot[::hist_step, 1, 0],
        hist_rot[::hist_step, 0, 0],
        color="#1f77b4",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.003,
        alpha=0.6,
    )

    ax_traj.plot(
        pred_path_display[:, 0],
        pred_path_display[:, 1],
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="pred trajectory",
    )
    pred_step = 4
    ax_traj.quiver(
        pred_display_x[::pred_step],
        pred_display_y[::pred_step],
        pred_rot[::pred_step, 1, 0],
        pred_rot[::pred_step, 0, 0],
        color="#2ca02c",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.003,
        alpha=0.6,
    )
    ax_traj.scatter([0.0], [0.0], color="black", marker="x", s=90, label="current t0")

    route_display = np.concatenate(
        [
            np.stack([hist_xyz[:, 1], hist_xyz[:, 0]], axis=1),
            pred_path_display,
            np.zeros((1, 2), dtype=np.float32),
        ],
        axis=0,
    )
    x_min, y_min = np.min(route_display, axis=0)
    x_max, y_max = np.max(route_display, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 4.0)
    half_extent = 0.5 * span + 0.6
    x_center = 0.5 * float(x_min + x_max)
    y_center = 0.5 * float(y_min + y_max)
    ax_traj.set_xlim(x_center - half_extent, x_center + half_extent)
    ax_traj.set_ylim(y_center - half_extent, y_center + half_extent)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("Local Y (Right) [m]")
    ax_traj.set_ylabel("Local X (Forward) [m]")
    ax_traj.set_title("Top-Down Local Trajectory", fontsize=13, fontweight="bold")
    ax_traj.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_traj.yaxis.set_major_locator(MultipleLocator(1.0))
    ax_traj.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_traj.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_traj.grid(which="major", color="#8091a7", alpha=0.34, linewidth=0.9)
    ax_traj.grid(which="minor", color="#c3cfdd", alpha=0.20, linewidth=0.45)
    ax_traj.legend(loc="upper left")

    cot_block = textwrap.fill(cot_text if cot_text else "(no CoT captured)", width=38)
    ax_cot.text(
        0.0,
        1.0,
        "CoT\n\n" + cot_block,
        fontsize=11,
        va="top",
        family="monospace",
    )

    bottom_gs = gs[2, :].subgridspec(1, 3, width_ratios=[1.0, 1.1, 1.2], wspace=0.16)
    ax_summary = fig.add_subplot(bottom_gs[0, 0])
    ax_decision = fig.add_subplot(bottom_gs[0, 1])
    ax_timing = fig.add_subplot(bottom_gs[0, 2])
    for ax in (ax_summary, ax_decision, ax_timing):
        ax.axis("off")

    horizon_s = float(pred_xyz.shape[0]) * float(fm["action_space_constants"]["dt_value"])
    summary_block = (
        "Live Alpamayo 1.5 Dashboard\n\n"
        + "\n".join(header_lines)
        + f"\nplan_points: {int(pred_xyz.shape[0])}"
        + f"\nplan_horizon_s: {horizon_s:.2f}"
        + f"\nfixed_delta_s: {float(sample['fixed_delta_seconds']):.3f}"
        + "\n\ncameras:\n"
        + "\n".join([f"- {title}" for title in camera_titles])
    )
    ax_summary.text(0.0, 1.0, summary_block, fontsize=10.5, va="top", family="monospace")

    decision_lines = []
    if nav_info:
        decision_lines.append("Nav")
        decision_lines.append(textwrap.fill(json.dumps(nav_info, ensure_ascii=False), width=44))
        decision_lines.append("")
    decision_lines.append("Final Output")
    decision_lines.append(textwrap.fill(final_text if final_text else "(empty)", width=44))
    ax_decision.text(0.0, 1.0, "\n".join(decision_lines), fontsize=10.5, va="top", family="monospace")

    wrapped_timing = "\n".join(timing_lines) if timing_lines else "timing unavailable"
    ax_timing.text(0.0, 1.0, "Timing\n\n" + wrapped_timing, fontsize=10.5, va="top", family="monospace")

    ensure_dir(dashboard_dir)
    dashboard_path = dashboard_dir / f"dashboard_{run_name}.png"
    latest_path = dashboard_dir / "latest_dashboard.png"
    fig.savefig(dashboard_path, dpi=140)
    fig.savefig(latest_path, dpi=140)
    plt.close(fig)
    return dashboard_path


def extract_trajectory_artifacts(
    *,
    output_file: Path,
    trajectory_dir: Path,
    run_name: str,
    metadata: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[Path, Path, Path]:
    result = json.loads(output_file.read_text())
    response = result["responses"][0]
    post_vlm = response["alpamayo_post_vlm"]
    fm = post_vlm["fm"]

    pred_xyz = reshape_tensor_field(fm["pred_xyz"])
    pred_rot = reshape_tensor_field(fm["pred_rot"])
    x0 = reshape_tensor_field(fm["x0"])
    x_final = reshape_tensor_field(fm["x_final"])

    pred_xyz = pred_xyz[0]
    pred_rot = pred_rot[0]
    x0 = x0[0]
    x_final = x_final[0]

    dt_value = float(fm["action_space_constants"]["dt_value"])
    relative_timestamps = [round((idx + 1) * dt_value, 6) for idx in range(pred_xyz.shape[0])]

    summary = {
        "sequence": int(metadata["sequence"]),
        "t0_us": int(metadata["t0_us"]),
        "clip_id": str(metadata["clip_id"]),
        "fixed_delta_seconds": float(sample["fixed_delta_seconds"]),
        "camera_indices": [int(x) for x in np.asarray(sample["camera_indices"]).tolist()],
        "camera_order": [str(x) for x in sample["camera_order"]],
        "relative_timestamps_sec": relative_timestamps,
        "output_text": response.get("output_text"),
        "fm_mode": post_vlm.get("fm_mode"),
        "fm_status": post_vlm.get("fm_status"),
        "nav": post_vlm.get("nav"),
        "pred_xyz_shape": list(pred_xyz.shape),
        "pred_rot_shape": list(pred_rot.shape),
        "x0_shape": list(x0.shape),
        "x_final_shape": list(x_final.shape),
        "pred_xyz": pred_xyz.tolist(),
        "pred_rot": pred_rot.tolist(),
        "x0": x0.tolist(),
        "x_final": x_final.tolist(),
    }

    ensure_dir(trajectory_dir)
    traj_json = trajectory_dir / f"trajectory_{run_name}.json"
    traj_xyz_npy = trajectory_dir / f"pred_xyz_{run_name}.npy"
    traj_rot_npy = trajectory_dir / f"pred_rot_{run_name}.npy"

    traj_json.write_text(json.dumps(summary, indent=2))
    np.save(traj_xyz_npy, pred_xyz.astype(np.float32))
    np.save(traj_rot_npy, pred_rot.astype(np.float32))

    latest_json = trajectory_dir / "latest_trajectory.json"
    latest_xyz = trajectory_dir / "latest_pred_xyz.npy"
    latest_rot = trajectory_dir / "latest_pred_rot.npy"
    latest_json.write_text(json.dumps(summary, indent=2))
    np.save(latest_xyz, pred_xyz.astype(np.float32))
    np.save(latest_rot, pred_rot.astype(np.float32))

    return traj_json, traj_xyz_npy, traj_rot_npy


def main() -> None:
    args = parse_args()
    persistent_client: PersistentLLMInferenceClient | None = None

    if not args.llm_inference_bin.exists():
        raise FileNotFoundError(f"llm_inference not found: {args.llm_inference_bin}")
    if not args.engine_dir.exists():
        raise FileNotFoundError(f"engineDir not found: {args.engine_dir}")
    if not args.multimodal_engine_dir.exists():
        raise FileNotFoundError(f"multimodalEngineDir not found: {args.multimodal_engine_dir}")
    if not args.fm_engine.exists():
        raise FileNotFoundError(f"fmEngine not found: {args.fm_engine}")

    ensure_dir(args.live_image_dir)
    ensure_dir(args.live_ego_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.kv_cache_dir)
    ensure_dir(args.trajectory_dir)
    ensure_dir(args.dashboard_dir)
    if args.dump_nav_dual_cache:
        ensure_dir(args.nav_cache_output_dir)

    action_space_constants = load_action_space_constants(args.action_space_constants_json)

    last_sequence = None
    completed_runs = 0

    try:
        if args.persistent_llm_inference:
            persistent_client = PersistentLLMInferenceClient(args)

        while args.max_runs < 0 or completed_runs < args.max_runs:
            sample, metadata = fetch_latest_sample(args.server_url, args.sample_endpoint, args.timeout)
            if sample is None:
                print(">> sample not ready, waiting...")
                time.sleep(args.poll_interval)
                continue

            sequence = int(metadata["sequence"])
            if last_sequence is not None and sequence == last_sequence:
                print(f">> no new sample yet (sequence={sequence}), waiting...")
                time.sleep(args.poll_interval)
                continue

            run_name = f"seq{sequence:06d}_t0_{metadata['t0_us']}"
            print(
                f">> Pulled sample sequence={sequence}, t0_us={metadata['t0_us']}, "
                f"clip_id={metadata['clip_id']}, image_frames={sample['image_frames'].shape}, "
                f"ego_history_xyz={sample['ego_history_xyz'].shape}, camera_indices={sample['camera_indices'].tolist()}"
            )

            if args.keep_last_only:
                remove_tree_contents(args.live_image_dir)
                remove_tree_contents(args.live_ego_dir)

            _, xyz_path, rot_path = write_sample_files(sample, args.live_image_dir, args.live_ego_dir)
            build_runtime_request(
                sample=sample,
                xyz_path=xyz_path,
                rot_path=rot_path,
                image_dir=args.live_image_dir,
                output_request=args.live_request_file,
                action_space_constants=action_space_constants,
                nav_text=args.nav_text,
                nav_guidance_weight=args.nav_guidance_weight,
                traj_token_offset=args.traj_token_offset,
                diffusion_seed=args.diffusion_seed,
                diffusion_num_steps=args.diffusion_num_steps,
                max_generate_length=args.max_generate_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            output_file, profile_file, kv_dir, nav_cache_dir = run_llm_inference(
                args, args.live_request_file, run_name, persistent_client=persistent_client
            )
            traj_json, traj_xyz_npy, traj_rot_npy = extract_trajectory_artifacts(
                output_file=output_file,
                trajectory_dir=args.trajectory_dir,
                run_name=run_name,
                metadata=metadata,
                sample=sample,
            )
            dashboard_png = save_live_dashboard(
                sample=sample,
                output_file=output_file,
                profile_file=profile_file,
                dashboard_dir=args.dashboard_dir,
                run_name=run_name,
                metadata=metadata,
            )

            latest_output = args.output_dir / "latest_output.json"
            latest_output.write_text(output_file.read_text())

            print(f">> output: {output_file}")
            print(f">> trajectory json: {traj_json}")
            print(f">> trajectory xyz: {traj_xyz_npy}")
            print(f">> trajectory rot: {traj_rot_npy}")
            print(f">> dashboard png: {dashboard_png}")
            if profile_file is not None:
                print(f">> profile: {profile_file}")
            if kv_dir is not None:
                print(f">> kv_cache: {kv_dir}")
            if nav_cache_dir is not None:
                print(f">> nav_cache: {nav_cache_dir}")

            last_sequence = sequence
            completed_runs += 1

            if args.once:
                break

            time.sleep(args.poll_interval)
    finally:
        if persistent_client is not None:
            persistent_client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(">> interrupted by user")
        sys.exit(130)
