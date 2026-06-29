#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from build_portfolio_preview_frame import (
    CANVAS_SIZE,
    cover_resize,
    draw_bev_panel,
    draw_camera_chip,
    find_legacy_artifact_dir,
    official_points_xy,
    packet_points_xy,
    load_font,
    load_jsonl_by_sample,
    wrap_text_to_width,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 720p/10fps portfolio video from Alpamayo comparison outputs."
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/straight10_20260624_test1_legacy_official_gap10/2026-06-24-test1"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/data/2026-06-24-test1"),
    )
    parser.add_argument(
        "--model",
        choices=("legacy_seed23", "official_10b"),
        required=True,
        help="Path source for the BEV panel.",
    )
    parser.add_argument(
        "--cot-source",
        choices=("same", "legacy_seed23", "official_10b"),
        default="official_10b",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Optional frame output directory. Defaults next to the MP4.",
    )
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help="Use only samples that have predictions instead of filling the 10Hz camera timeline.",
    )
    parser.add_argument(
        "--crf-note",
        action="store_true",
        help="No-op placeholder for callers used to ffmpeg options; OpenCV mp4v is used here.",
    )
    return parser.parse_args()


def load_prediction_sample_ids(predictions_jsonl: Path) -> list[int]:
    sample_ids: set[int] = set()
    with predictions_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in (None, "official_10b_ae"):
                continue
            sample_ids.add(int(row["sample_id"]))
    return sorted(sample_ids)


def load_legacy_sample_ids(artifact_root: Path) -> list[int]:
    sample_ids: set[int] = set()
    for path in artifact_root.glob("request_*_sid*_t0_*"):
        match = re.search(r"_sid(\d+)_", path.name)
        if match:
            sample_ids.add(int(match.group(1)))
    return sorted(sample_ids)


def sample_sequence(
    key_sample_ids: list[int],
    keyframes_only: bool,
    sample_start: int | None = None,
    sample_end: int | None = None,
) -> list[int]:
    if keyframes_only:
        return key_sample_ids
    first = key_sample_ids[0] if sample_start is None else int(sample_start)
    last = key_sample_ids[-1] if sample_end is None else int(sample_end)
    return list(range(first, last + 1))


def latest_key_sample(sample_id: int, key_sample_ids: list[int]) -> int:
    current = key_sample_ids[0]
    for key in key_sample_ids:
        if key <= sample_id:
            current = key
        else:
            break
    return current


class CameraReader:
    def __init__(self, dataset_root: Path, camera_name: str) -> None:
        self.dataset_root = dataset_root
        self.camera_name = camera_name
        self.frames = pd.read_parquet(dataset_root / "sensors" / camera_name / "frames.parquet")
        self.rows = self.frames.set_index("frame_id")
        self.caps: dict[int, cv2.VideoCapture] = {}
        self.next_frame_idx: dict[int, int] = {}

    def read(self, frame_id: int) -> Image.Image:
        rec = self.rows.loc[int(frame_id)]
        chunk_id = int(rec["chunk_id"])
        frame_idx = int(rec["frame_index_in_chunk"])
        cap = self.caps.get(chunk_id)
        if cap is None:
            video_path = self.dataset_root / "sensors" / self.camera_name / "chunks" / f"chunk_{chunk_id:04d}.mkv"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"failed to open {video_path}")
            self.caps[chunk_id] = cap
        if self.next_frame_idx.get(chunk_id) != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read {self.camera_name} frame_id={frame_id} idx={frame_idx}")
        self.next_frame_idx[chunk_id] = frame_idx + 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        self.next_frame_idx.clear()


def load_cot(root: Path, sample_id: int, cot_source: str, model: str) -> str:
    source = model if cot_source == "same" else cot_source
    if source == "official_10b":
        row = load_jsonl_by_sample(root / "official_10b" / "predictions.jsonl", sample_id)
        return str(row.get("cot", "")).strip() or "(10B CoT unavailable)"
    return "Legacy seed23 prefill-to-AE run does not decode a CoT string; trajectory is generated directly from prefill KV."


def load_path_bundle(
    root: Path,
    sample_id: int,
    model: str,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], str]:
    artifact_dir = find_legacy_artifact_dir(root, sample_id)
    gt_points = packet_points_xy(artifact_dir / "gt_path.json")
    if model == "legacy_seed23":
        pred_points = packet_points_xy(artifact_dir / "final_path.json")
        model_label = "legacy seed23"
    else:
        row = load_jsonl_by_sample(root / "official_10b" / "predictions.jsonl", sample_id)
        pred_points = official_points_xy(row)
        model_label = "official 10B"
    return pred_points, gt_points, model_label


def render_portfolio_frame(
    *,
    front: Image.Image,
    left: Image.Image,
    right: Image.Image,
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    cot: str,
    model_label: str,
) -> Image.Image:
    canvas = cover_resize(front, CANVAS_SIZE).convert("RGBA")
    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, CANVAS_SIZE[0], 150), fill=(0, 0, 0, 55))
    draw.rectangle((0, 560, CANVAS_SIZE[0], CANVAS_SIZE[1]), fill=(0, 0, 0, 92))
    canvas.alpha_composite(overlay)

    draw_camera_chip(canvas, left, (26, 24), (202, 112), "LEFT")
    draw_camera_chip(canvas, right, (238, 24), (202, 112), "RIGHT")
    bev_panel = draw_bev_panel(pred_points, gt_points, model_label=model_label, size=(332, 244))
    canvas.alpha_composite(bev_panel, (922, 24))

    draw = ImageDraw.Draw(canvas)
    cot_font = load_font(20, bold=False)
    lines = wrap_text_to_width(cot, cot_font, CANVAS_SIZE[0] - 72)[:3]
    total_h = max(1, len(lines)) * 28
    start_y = 604 if len(lines) == 1 else max(580, 640 - total_h)
    for idx, line in enumerate(lines):
        draw.text((36, start_y + idx * 28), line, fill=(245, 248, 255, 240), font=cot_font)
    return canvas.convert("RGB")


def write_video(frame_paths: list[Path], output: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {output}")
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    official_key_ids = load_prediction_sample_ids(args.comparison_root / "official_10b" / "predictions.jsonl")
    legacy_key_ids = load_legacy_sample_ids(args.comparison_root / "legacy_seed23" / "artifacts")
    path_key_ids = legacy_key_ids if args.model == "legacy_seed23" else official_key_ids
    if not path_key_ids:
        raise RuntimeError(f"No path key samples found for {args.model} under {args.comparison_root}")
    cot_key_ids = official_key_ids if args.cot_source in ("same", "official_10b") else legacy_key_ids
    if not cot_key_ids:
        raise RuntimeError(f"No CoT key samples found for {args.cot_source} under {args.comparison_root}")
    sample_ids = sample_sequence(
        path_key_ids,
        args.keyframes_only,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
    )
    frame_dir = args.frame_dir or args.output.with_suffix("").parent / (args.output.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    samples = pd.read_parquet(args.dataset_root / "sample_index_10hz.parquet").set_index("sample_id")
    front_reader = CameraReader(args.dataset_root, "camera_front")
    left_reader = CameraReader(args.dataset_root, "camera_left")
    right_reader = CameraReader(args.dataset_root, "camera_right")
    path_cache: dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]], str]] = {}
    cot_cache: dict[int, str] = {}
    try:
        for frame_idx, sid in enumerate(sample_ids):
            path_sid = latest_key_sample(sid, path_key_ids)
            cot_sid = latest_key_sample(sid, cot_key_ids)
            frame_path = frame_dir / f"frame_{frame_idx:05d}_sid{sid:05d}_pathsid{path_sid:05d}_cotsid{cot_sid:05d}.png"
            if not frame_path.exists():
                row = samples.loc[int(sid)]
                bundle = path_cache.get(path_sid)
                if bundle is None:
                    bundle = load_path_bundle(args.comparison_root, path_sid, args.model)
                    path_cache[path_sid] = bundle
                pred_points, gt_points, model_label = bundle
                cot = cot_cache.get(cot_sid)
                if cot is None:
                    cot = load_cot(args.comparison_root, cot_sid, args.cot_source, args.model)
                    cot_cache[cot_sid] = cot
                image = render_portfolio_frame(
                    front=front_reader.read(int(row["front_frame_id"])),
                    left=left_reader.read(int(row["left_frame_id"])),
                    right=right_reader.read(int(row["right_frame_id"])),
                    pred_points=pred_points,
                    gt_points=gt_points,
                    cot=cot,
                    model_label=model_label,
                )
                image.save(frame_path, quality=96)
            frame_paths.append(frame_path)
    finally:
        front_reader.close()
        left_reader.close()
        right_reader.close()

    write_video(frame_paths, args.output, args.fps)
    duration_s = len(frame_paths) / args.fps
    summary = {
        "output": str(args.output),
        "frame_dir": str(frame_dir),
        "model": args.model,
        "cot_source": args.cot_source,
        "fps": args.fps,
        "frames": len(frame_paths),
        "duration_s": duration_s,
        "path_key_sample_ids": path_key_ids,
        "cot_key_sample_ids": cot_key_ids,
        "sample_range": [sample_ids[0], sample_ids[-1]],
        "keyframes_only": args.keyframes_only,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
