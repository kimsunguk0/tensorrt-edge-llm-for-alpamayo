#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import numpy as np


DEFAULT_OUTPUT_ROOT = Path("/workspace/live_camera_ego_history_capture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save live /latest camera frames and ego history samples.")
    parser.add_argument("--latest-url", default="http://127.0.0.1:18080/latest")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl+C or --limit.")
    parser.add_argument("--limit", type=int, default=0, help="0 means no sample limit.")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--latest-frame-only", action="store_true", help="Save only the newest frame per camera.")
    parser.add_argument("--no-images", action="store_true", help="Do not write PNG images, only NPZ/CSV metadata.")
    parser.add_argument("--no-full-npz", action="store_true", help="Do not write per-sample compact NPZ files.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Save repeated /latest t0_us samples too.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_latest_npz(url: str, timeout_s: float) -> dict[str, np.ndarray]:
    with urllib.request.urlopen(url, timeout=float(timeout_s)) as response:
        raw = response.read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(-1)[0].item()
    else:
        return arr.tolist()
    if isinstance(item, np.generic):
        return item.item()
    return item


def camera_names(sample: dict[str, np.ndarray], num_cameras: int) -> list[str]:
    if "camera_order" not in sample:
        return [f"cam{idx}" for idx in range(num_cameras)]
    names = [str(item) for item in np.asarray(sample["camera_order"]).reshape(-1).tolist()]
    if len(names) < num_cameras:
        names.extend(f"cam{idx}" for idx in range(len(names), num_cameras))
    return names[:num_cameras]


def save_png(path: Path, chw_rgb: np.ndarray) -> None:
    import cv2

    image = np.asarray(chw_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image with shape [3,H,W], got {image.shape}")
    hwc_rgb = np.transpose(image, (1, 2, 0))
    hwc_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), hwc_bgr):
        raise RuntimeError(f"failed to write image: {path}")


def yaw_from_rot(rot: np.ndarray) -> float:
    return float(np.arctan2(float(rot[1, 0]), float(rot[0, 0])))


def _first_vector(sample: dict[str, np.ndarray], key: str, size: int) -> list[float | None]:
    if key not in sample:
        return [None] * size
    arr = np.asarray(sample[key]).reshape(-1)
    out: list[float | None] = []
    for idx in range(size):
        if idx < arr.size:
            out.append(float(arr[idx]))
        else:
            out.append(None)
    return out


def _first_scalar(sample: dict[str, np.ndarray], key: str) -> Any:
    if key not in sample:
        return None
    return scalar(sample[key])


def sample_nav_metadata(sample: dict[str, np.ndarray]) -> dict[str, Any]:
    lat, lon, alt = _first_vector(sample, "gnss_lla", 3)
    utm_e, utm_n, utm_alt = _first_vector(sample, "gnss_utm", 3)
    cov = np.asarray(sample.get("gnss_covariance_enu_m2", []), dtype=np.float64).reshape(-1)
    return {
        "gnss_utc_us": _first_scalar(sample, "gnss_utc_us"),
        "gnss_lat": lat,
        "gnss_lon": lon,
        "gnss_alt": alt,
        "utm_easting_m": utm_e,
        "utm_northing_m": utm_n,
        "utm_alt_m": utm_alt,
        "utm_zone": _first_scalar(sample, "gnss_utm_zone"),
        "utm_northp": _first_scalar(sample, "gnss_utm_northp"),
        "utm_valid": _first_scalar(sample, "gnss_utm_valid"),
        "cov_xx_m2": float(cov[0]) if cov.size >= 1 else None,
        "cov_yy_m2": float(cov[4]) if cov.size >= 5 else None,
        "cov_zz_m2": float(cov[8]) if cov.size >= 9 else None,
        "cov_valid": _first_scalar(sample, "gnss_covariance_valid"),
        "yaw_deg": _first_scalar(sample, "yaw_deg"),
        "yaw_valid": _first_scalar(sample, "yaw_valid"),
    }


def write_ego_history_csv(path: Path, xyz: np.ndarray, rot: np.ndarray, nav: dict[str, Any]) -> None:
    fields = [
        "history_idx",
        "x_m",
        "y_m",
        "z_m",
        "yaw_rad",
        "sample_gnss_utc_us",
        "sample_utm_easting_m",
        "sample_utm_northing_m",
        "sample_utm_alt_m",
        "sample_utm_zone",
        "sample_utm_northp",
        "sample_yaw_deg",
    ]
    lines = [",".join(fields) + "\n"]
    for idx in range(int(xyz.shape[0])):
        x, y, z = xyz[idx].tolist()
        row = {
            "history_idx": idx,
            "x_m": f"{float(x):.9f}",
            "y_m": f"{float(y):.9f}",
            "z_m": f"{float(z):.9f}",
            "yaw_rad": f"{yaw_from_rot(rot[idx]):.9f}",
            "sample_gnss_utc_us": nav.get("gnss_utc_us"),
            "sample_utm_easting_m": nav.get("utm_easting_m"),
            "sample_utm_northing_m": nav.get("utm_northing_m"),
            "sample_utm_alt_m": nav.get("utm_alt_m"),
            "sample_utm_zone": nav.get("utm_zone"),
            "sample_utm_northp": nav.get("utm_northp"),
            "sample_yaw_deg": nav.get("yaw_deg"),
        }
        lines.append(",".join("" if row[field] is None else str(row[field]) for field in fields) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def compact_npz_payload(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep_prefixes = ("gnss_",)
    keep_names = {
        "image_frames",
        "camera_indices",
        "camera_order",
        "ego_history_xyz",
        "ego_history_rot",
        "relative_timestamps",
        "absolute_timestamps",
        "t0_us",
        "fixed_delta_seconds",
        "clip_id",
        "yaw_deg",
        "yaw_valid",
    }
    return {key: value for key, value in sample.items() if key in keep_names or key.startswith(keep_prefixes)}


SAMPLES_CSV_FIELDS = [
    "seq",
    "t0_us",
    "gnss_utc_us",
    "receive_time_unix_s",
    "clip_id",
    "num_cameras",
    "num_frames_per_camera",
    "camera_order",
    "camera_indices",
    "gnss_lat",
    "gnss_lon",
    "gnss_alt",
    "utm_easting_m",
    "utm_northing_m",
    "utm_alt_m",
    "utm_zone",
    "utm_northp",
    "utm_valid",
    "cov_xx_m2",
    "cov_yy_m2",
    "cov_zz_m2",
    "cov_valid",
    "yaw_deg",
    "yaw_valid",
    "relative_timestamps",
    "absolute_timestamps",
    "saved_image_count",
    "first_image_path",
    "ego_history_xyz_npy",
    "ego_history_rot_npy",
    "ego_history_csv",
    "sample_npz",
]


def samples_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {field: summary.get(field) for field in SAMPLES_CSV_FIELDS}
    row["camera_order"] = "|".join(str(item) for item in (summary.get("camera_order") or []))
    row["camera_indices"] = "|".join(str(item) for item in (summary.get("camera_indices") or []))
    row["relative_timestamps"] = json.dumps(summary.get("relative_timestamps"), separators=(",", ":"))
    row["absolute_timestamps"] = json.dumps(summary.get("absolute_timestamps"), separators=(",", ":"))
    saved_images = summary.get("saved_images") or []
    row["saved_image_count"] = len(saved_images)
    row["first_image_path"] = saved_images[0] if saved_images else None
    return row


def save_sample(
    *,
    sample: dict[str, np.ndarray],
    output_root: Path,
    seq: int,
    latest_frame_only: bool,
    save_images: bool,
    save_full_npz: bool,
) -> dict[str, Any]:
    t0_us = int(scalar(sample["t0_us"]))
    clip_id = str(scalar(sample.get("clip_id", ""))) if "clip_id" in sample else ""
    sample_stem = f"sample_{seq:06d}_t0_{t0_us}"

    images_root = output_root / "images" / sample_stem
    ego_root = output_root / "ego_history"
    npz_root = output_root / "npz"
    ensure_dir(ego_root)
    ensure_dir(npz_root)

    image_paths: list[str] = []
    image_frames = np.asarray(sample["image_frames"], dtype=np.uint8)
    num_cameras = int(image_frames.shape[0])
    num_frames = int(image_frames.shape[1])
    names = camera_names(sample, num_cameras)

    frame_indices = [num_frames - 1] if latest_frame_only else list(range(num_frames))
    if save_images:
        ensure_dir(images_root)
        for cam_idx in range(num_cameras):
            cam_name = names[cam_idx].replace("/", "_").replace(" ", "_")
            for frame_idx in frame_indices:
                image_path = images_root / f"cam{cam_idx}_{cam_name}_f{frame_idx}.png"
                save_png(image_path, image_frames[cam_idx, frame_idx])
                image_paths.append(str(image_path))

    ego_xyz = np.asarray(sample["ego_history_xyz"], dtype=np.float32)
    ego_rot = np.asarray(sample["ego_history_rot"], dtype=np.float32)
    nav = sample_nav_metadata(sample)
    ego_xyz_path = ego_root / f"{sample_stem}_ego_history_xyz.npy"
    ego_rot_path = ego_root / f"{sample_stem}_ego_history_rot.npy"
    ego_csv_path = ego_root / f"{sample_stem}_ego_history.csv"
    np.save(ego_xyz_path, ego_xyz)
    np.save(ego_rot_path, ego_rot)
    write_ego_history_csv(ego_csv_path, ego_xyz[0, 0], ego_rot[0, 0], nav)

    npz_path: Path | None = None
    if save_full_npz:
        npz_path = npz_root / f"{sample_stem}.npz"
        np.savez_compressed(npz_path, **compact_npz_payload(sample))

    absolute_timestamps = sample.get("absolute_timestamps")
    relative_timestamps = sample.get("relative_timestamps")
    summary: dict[str, Any] = {
        "seq": int(seq),
        "t0_us": int(t0_us),
        "clip_id": clip_id,
        "num_cameras": int(num_cameras),
        "num_frames_per_camera": int(num_frames),
        "camera_order": names,
        "camera_indices": sample.get("camera_indices", np.asarray([], dtype=np.int32)).reshape(-1).astype(int).tolist(),
        "image_shape": list(image_frames.shape),
        "saved_images": image_paths,
        "ego_history_xyz_npy": str(ego_xyz_path),
        "ego_history_rot_npy": str(ego_rot_path),
        "ego_history_csv": str(ego_csv_path),
        "sample_npz": str(npz_path) if npz_path is not None else None,
        "relative_timestamps": None if relative_timestamps is None else relative_timestamps.tolist(),
        "absolute_timestamps": None if absolute_timestamps is None else absolute_timestamps.tolist(),
        "receive_time_unix_s": time.time(),
    }
    summary.update(nav)
    for key in (
        "gnss_utc_us",
        "gnss_lla",
        "gnss_utm",
        "gnss_utm_zone",
        "gnss_utm_northp",
        "gnss_utm_valid",
        "gnss_covariance_enu_m2",
        "gnss_covariance_valid",
        "yaw_deg",
        "yaw_valid",
    ):
        if key in sample:
            raw_key = f"{key}_raw" if key in summary else key
            summary[raw_key] = sample[key].tolist()
    return summary


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.output_root / "images")
    ensure_dir(args.output_root / "ego_history")
    ensure_dir(args.output_root / "npz")

    manifest_path = args.output_root / "manifest.json"
    samples_jsonl_path = args.output_root / "samples.jsonl"
    samples_csv_path = args.output_root / "samples.csv"
    period_s = 1.0 / max(float(args.rate_hz), 1e-6)
    started_s = time.monotonic()
    saved = 0
    duplicate_count = 0
    error_count = 0
    last_error: str | None = None
    last_t0_us: int | None = None
    interrupted = False

    try:
        csv_write_header = not samples_csv_path.exists() or samples_csv_path.stat().st_size == 0
        with samples_jsonl_path.open("a", encoding="utf-8") as samples_jsonl, samples_csv_path.open(
            "a", encoding="utf-8", newline=""
        ) as samples_csv:
            samples_writer = csv.DictWriter(samples_csv, fieldnames=SAMPLES_CSV_FIELDS)
            if csv_write_header:
                samples_writer.writeheader()
            while args.limit <= 0 or saved < int(args.limit):
                tick_s = time.monotonic()
                if args.duration_s > 0 and tick_s - started_s >= float(args.duration_s):
                    break
                try:
                    sample = fetch_latest_npz(str(args.latest_url), float(args.timeout_s))
                    t0_us = int(scalar(sample["t0_us"]))
                    if not args.keep_duplicates and last_t0_us == t0_us:
                        duplicate_count += 1
                    else:
                        summary = save_sample(
                            sample=sample,
                            output_root=args.output_root,
                            seq=saved,
                            latest_frame_only=bool(args.latest_frame_only),
                            save_images=not bool(args.no_images),
                            save_full_npz=not bool(args.no_full_npz),
                        )
                        samples_jsonl.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
                        samples_jsonl.flush()
                        samples_writer.writerow(samples_csv_row(summary))
                        samples_csv.flush()
                        print(
                            "saved",
                            f"seq={saved}",
                            f"t0_us={t0_us}",
                            f"utm_e={summary.get('utm_easting_m')}",
                            f"utm_n={summary.get('utm_northing_m')}",
                            f"cams={summary['num_cameras']}",
                            f"frames={summary['num_frames_per_camera']}",
                            f"images={len(summary['saved_images'])}",
                            flush=True,
                        )
                        saved += 1
                        last_t0_us = t0_us
                        last_error = None
                except KeyboardInterrupt:
                    interrupted = True
                    raise
                except Exception as exc:
                    error_count += 1
                    last_error = str(exc)
                    print(f"save_error count={error_count} error={last_error}", flush=True)
                time.sleep(max(period_s - (time.monotonic() - tick_s), 0.0))
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; stopping live camera/ego capture", flush=True)
    finally:
        manifest = {
            "latest_url": str(args.latest_url),
            "output_root": str(args.output_root),
            "samples_jsonl": str(samples_jsonl_path),
            "samples_csv": str(samples_csv_path),
            "rate_hz": float(args.rate_hz),
            "saved_samples": int(saved),
            "duplicate_skips": int(duplicate_count),
            "errors": int(error_count),
            "last_error": last_error,
            "latest_frame_only": bool(args.latest_frame_only),
            "save_images": not bool(args.no_images),
            "save_full_npz": not bool(args.no_full_npz),
            "interrupted": interrupted,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
