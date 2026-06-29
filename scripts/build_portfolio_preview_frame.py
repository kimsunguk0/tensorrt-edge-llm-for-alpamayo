#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (1280, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 720p portfolio preview frame with cameras, BEV path, and CoT text."
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
        help="Dataset root used to pull original-resolution camera frames. Falls back to request PNGs if unavailable.",
    )
    parser.add_argument("--sample-id", type=int, default=423)
    parser.add_argument(
        "--camera-sample-id",
        type=int,
        default=None,
        help="Camera frame sample id. Defaults to --sample-id. Useful when holding the latest path across 10Hz camera frames.",
    )
    parser.add_argument(
        "--model",
        choices=("legacy_seed23", "official_10b"),
        default="official_10b",
        help="Path source used for the BEV trajectory.",
    )
    parser.add_argument(
        "--cot-source",
        choices=("same", "legacy_seed23", "official_10b"),
        default="same",
        help="CoT text source. Use --model legacy_seed23 --cot-source official_10b for legacy path with 10B scene reasoning.",
    )
    parser.add_argument(
        "--bottom-mode",
        choices=("cot_only", "full"),
        default="cot_only",
        help="Bottom overlay style. cot_only shows only the CoT sentence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_by_sample(path: Path, sample_id: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("sample_id", -1)) == sample_id:
                return row
    raise FileNotFoundError(f"sample_id {sample_id} not found in {path}")


def find_legacy_artifact_dir(root: Path, sample_id: int) -> Path:
    matches = sorted((root / "legacy_seed23" / "artifacts").glob(f"request_*_sid{sample_id:05d}_*"))
    if not matches:
        raise FileNotFoundError(f"legacy artifact for sample_id {sample_id} not found")
    return matches[0]


def find_request_json(root: Path, sample_id: int) -> Path:
    matches = sorted((root / "legacy_seed23" / "requests" / "requests").glob(f"request_*_sid{sample_id:05d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"request JSON for sample_id {sample_id} not found")
    return matches[0]


def latest_request_images(request_json: Path) -> dict[str, Path]:
    request = load_json(request_json)
    content = request["requests"][0]["messages"][1]["content"]
    current_label = ""
    images: dict[str, list[Path]] = {}
    for item in content:
        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if text.endswith("camera:"):
                current_label = text[:-1]
        elif item.get("type") == "image" and current_label:
            images.setdefault(current_label, []).append(Path(item["image"]))
    return {label: paths[-1] for label, paths in images.items() if paths}


def read_original_frame(dataset_root: Path, camera_name: str, frame_id: int) -> Image.Image:
    import cv2
    import pandas as pd

    frames = pd.read_parquet(dataset_root / "sensors" / camera_name / "frames.parquet")
    row = frames.loc[frames["frame_id"] == int(frame_id)]
    if row.empty:
        raise FileNotFoundError(f"{camera_name} frame_id {frame_id} not found")
    rec = row.iloc[0]
    chunk_id = int(rec["chunk_id"])
    frame_idx = int(rec["frame_index_in_chunk"])
    video_path = dataset_root / "sensors" / camera_name / "chunks" / f"chunk_{chunk_id:04d}.mkv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read {video_path} frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def original_camera_images(dataset_root: Path, sample_id: int) -> dict[str, Image.Image]:
    import pandas as pd

    samples = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    row = samples.loc[samples["sample_id"] == int(sample_id)]
    if row.empty:
        raise FileNotFoundError(f"sample_id {sample_id} not found in {dataset_root}")
    rec = row.iloc[0]
    return {
        "Front camera": read_original_frame(dataset_root, "camera_front", int(rec["front_frame_id"])),
        "Front left camera": read_original_frame(dataset_root, "camera_left", int(rec["left_frame_id"])),
        "Front right camera": read_original_frame(dataset_root, "camera_right", int(rec["right_frame_id"])),
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = size
    src_w, src_h = image.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))
    return image.resize(size, Image.Resampling.LANCZOS)


def fit_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def packet_points_xy(path_json: Path) -> list[tuple[float, float]]:
    data = load_json(path_json)
    return [(float(p["x_m"]), float(p["y_m"])) for p in data["packet_points"]]


def official_points_xy(row: dict[str, Any]) -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    for xyz in row["pred_xyz"]:
        points.append((float(xyz[0]), float(xyz[1])))
    return points


def path_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def point_at_fraction(points: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    target = path_length(points) * fraction
    walked = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-6:
            continue
        if walked + seg >= target:
            t = (target - walked) / seg
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        walked += seg
    return points[-1]


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
    dash_px: int,
    gap_px: int,
) -> None:
    for p0, p1 in zip(points, points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-6:
            continue
        ux = (x1 - x0) / length
        uy = (y1 - y0) / length
        pos = 0.0
        while pos < length:
            end = min(length, pos + dash_px)
            draw.line(
                [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
                fill=color,
                width=width,
            )
            pos += dash_px + gap_px


def draw_bev_panel(
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    model_label: str,
    size: tuple[int, int] = (332, 244),
) -> Image.Image:
    scale = 3
    panel_w, panel_h = size
    img = Image.new("RGBA", (panel_w * scale, panel_h * scale), (7, 11, 17, 210))
    draw = ImageDraw.Draw(img)
    font = load_font(9 * scale, bold=False)
    title_font = load_font(12 * scale, bold=True)
    small_font = load_font(8 * scale, bold=False)

    w, h = img.size
    pad_l = 34 * scale
    pad_r = 14 * scale
    pad_t = 24 * scale
    pad_b = 28 * scale
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    all_points = pred_points + gt_points
    max_x = max([p[0] for p in all_points] + [8.0])
    x_max = min(max(10.0, max_x + 0.8), 24.0)
    y_abs = max([abs(p[1]) for p in all_points if p[0] <= x_max] + [1.8])
    y_lim = min(max(2.0, y_abs + 0.35), 5.0)

    def to_screen(point: tuple[float, float]) -> tuple[float, float]:
        forward_x, lateral_y = point
        # Vehicle +y is left; screen x grows right, so negate y to make visual left/right correct.
        display_lat = -lateral_y
        sx = pad_l + (display_lat + y_lim) / (2 * y_lim) * plot_w
        sy = pad_t + (1.0 - max(0.0, min(forward_x, x_max)) / x_max) * plot_h
        return sx, sy

    # Panel and plot backgrounds.
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=16 * scale, fill=(7, 11, 17, 218))
    draw.rounded_rectangle(
        (pad_l, pad_t, pad_l + plot_w, pad_t + plot_h),
        radius=6 * scale,
        fill=(11, 18, 28, 215),
        outline=(80, 94, 115, 175),
        width=1 * scale,
    )

    # Grid: 0.5 m lateral, 2 m forward, with stronger 1/4 m lines.
    grid_minor = (120, 144, 180, 50)
    grid_major = (148, 163, 184, 92)
    lat = -math.floor(y_lim * 2) / 2
    while lat <= y_lim + 1e-6:
        sx = pad_l + (lat + y_lim) / (2 * y_lim) * plot_w
        is_major = abs(round(lat) - lat) < 1e-6
        draw.line((sx, pad_t, sx, pad_t + plot_h), fill=grid_major if is_major else grid_minor, width=1 * scale)
        lat += 0.5
    fwd = 0.0
    while fwd <= x_max + 1e-6:
        sy = pad_t + (1.0 - fwd / x_max) * plot_h
        is_major = abs((fwd / 4.0) - round(fwd / 4.0)) < 1e-6
        draw.line((pad_l, sy, pad_l + plot_w, sy), fill=grid_major if is_major else grid_minor, width=1 * scale)
        if fwd > 0:
            draw.text((5 * scale, sy - 5 * scale), f"{int(fwd)}m", fill=(210, 219, 234, 210), font=small_font)
        fwd += 2.0

    center_x = pad_l + plot_w / 2
    draw.line((center_x, pad_t, center_x, pad_t + plot_h), fill=(226, 232, 240, 120), width=1 * scale)
    draw.text((pad_l, 4 * scale), "BEV path", fill=(245, 248, 255, 245), font=title_font)
    draw.text((pad_l + 86 * scale, 7 * scale), model_label, fill=(148, 163, 184, 245), font=font)
    draw.text((pad_l, h - 20 * scale), "LEFT", fill=(203, 213, 225, 200), font=small_font)
    draw.text((pad_l + plot_w - 34 * scale, h - 20 * scale), "RIGHT", fill=(203, 213, 225, 200), font=small_font)

    pred_screen = [to_screen(p) for p in pred_points if p[0] <= x_max]
    gt_screen = [to_screen(p) for p in gt_points if p[0] <= x_max]
    draw_dashed_polyline(draw, gt_screen, (235, 238, 245, 190), 2 * scale, 7 * scale, 5 * scale)
    draw_polyline(draw, pred_screen, (45, 212, 255, 255), 4 * scale)

    ld_point = to_screen(point_at_fraction(pred_points, 0.25))
    r = 5 * scale
    draw.ellipse((ld_point[0] - r, ld_point[1] - r, ld_point[0] + r, ld_point[1] + r), fill=(255, 214, 102, 255))
    draw.ellipse((ld_point[0] - r, ld_point[1] - r, ld_point[0] + r, ld_point[1] + r), outline=(15, 23, 42, 255), width=1 * scale)
    draw.text((ld_point[0] + 6 * scale, ld_point[1] - 6 * scale), "25%", fill=(255, 230, 145, 255), font=small_font)

    # Ego marker.
    ego = to_screen((0.0, 0.0))
    draw.polygon(
        [
            (ego[0], ego[1] - 8 * scale),
            (ego[0] - 5 * scale, ego[1] + 6 * scale),
            (ego[0] + 5 * scale, ego[1] + 6 * scale),
        ],
        fill=(255, 255, 255, 245),
    )
    return img.resize(size, Image.Resampling.LANCZOS)


def draw_camera_chip(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    label: str,
) -> None:
    chip = fit_resize(image, size)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = xy
    w, h = size
    draw.rounded_rectangle((x - 3, y - 3, x + w + 3, y + h + 24), radius=9, fill=(0, 0, 0, 120))
    canvas.alpha_composite(overlay)
    canvas.paste(chip, xy)
    draw = ImageDraw.Draw(canvas)
    font = load_font(13, bold=True)
    draw.rectangle((x, y + h - 24, x + w, y + h), fill=(0, 0, 0, 135))
    draw.text((x + 8, y + h - 20), label, fill=(245, 248, 255, 240), font=font)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=5, outline=(255, 255, 255, 150), width=1)


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_frame(args: argparse.Namespace) -> None:
    root = args.comparison_root
    artifact_dir = find_legacy_artifact_dir(root, args.sample_id)
    request_json = find_request_json(root, args.sample_id)
    camera_sample_id = int(args.camera_sample_id) if args.camera_sample_id is not None else int(args.sample_id)
    try:
        images = original_camera_images(args.dataset_root, camera_sample_id)
        front = images["Front camera"]
        left = images["Front left camera"]
        right = images["Front right camera"]
    except Exception:
        request_images = latest_request_images(request_json)
        front = Image.open(request_images["Front camera"])
        left = Image.open(request_images["Front left camera"])
        right = Image.open(request_images["Front right camera"])

    gt_points = packet_points_xy(artifact_dir / "gt_path.json")
    if args.model == "legacy_seed23":
        pred_points = packet_points_xy(artifact_dir / "final_path.json")
        model_label = "legacy seed23"
    else:
        row = load_jsonl_by_sample(root / "official_10b" / "predictions.jsonl", args.sample_id)
        pred_points = official_points_xy(row)
        model_label = "official 10B"

    cot_source = args.model if args.cot_source == "same" else args.cot_source
    if cot_source == "official_10b":
        cot_row = load_jsonl_by_sample(root / "official_10b" / "predictions.jsonl", args.sample_id)
        cot = str(cot_row.get("cot", "")).strip() or "(10B CoT unavailable)"
        cot_label = "CoT by 10B"
    else:
        cot = "Legacy seed23 prefill-to-AE run does not decode a CoT string; trajectory is generated directly from prefill KV."
        cot_label = "CoT"

    canvas = cover_resize(front, CANVAS_SIZE).convert("RGBA")

    # Slight vignette makes the overlays readable without hiding the road.
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
    title_font = load_font(20, bold=True)
    body_font = load_font(18, bold=False)
    small_font = load_font(13, bold=False)
    if args.bottom_mode == "full":
        title = f"Alpamayo planner | sample {args.sample_id} | path: {model_label}"
        if cot_source != args.model:
            title += " | reasoning: official 10B"
        draw.text((36, 585), title, fill=(255, 255, 255, 245), font=title_font)
        cot_text_x = 148
        draw.text((36, 613), cot_label, fill=(255, 214, 102, 245), font=small_font)
        for idx, line in enumerate(wrap_text_to_width(cot, body_font, CANVAS_SIZE[0] - cot_text_x - 36)[:3]):
            draw.text((cot_text_x, 606 + idx * 25), line, fill=(235, 241, 250, 235), font=body_font)
    else:
        cot_font = load_font(20, bold=False)
        lines = wrap_text_to_width(cot, cot_font, CANVAS_SIZE[0] - 72)[:3]
        total_h = max(1, len(lines)) * 28
        start_y = 604 if len(lines) == 1 else max(580, 640 - total_h)
        for idx, line in enumerate(lines):
            draw.text((36, start_y + idx * 28), line, fill=(245, 248, 255, 240), font=cot_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.output, quality=96)
    print(args.output)


def main() -> None:
    build_frame(parse_args())


if __name__ == "__main__":
    main()
