#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize distill smoke trajectories on 1m x 1m grid")
    p.add_argument("--input-pt", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/distill_smoke_5.pt")
    p.add_argument("--out-dir", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/plots")
    p.add_argument("--grid-step-m", type=float, default=1.0)
    p.add_argument("--pixels-per-meter", type=int, default=40)
    p.add_argument("--padding-m", type=float, default=2.0)
    return p.parse_args()


def bounds_xy(curves: list[torch.Tensor], padding_m: float, grid_step_m: float) -> tuple[float, float, float, float]:
    xs = torch.cat([c[:, 0] for c in curves])
    ys = torch.cat([c[:, 1] for c in curves])
    x_min = math.floor((float(xs.min().item()) - padding_m) / grid_step_m) * grid_step_m
    x_max = math.ceil((float(xs.max().item()) + padding_m) / grid_step_m) * grid_step_m
    y_min = math.floor((float(ys.min().item()) - padding_m) / grid_step_m) * grid_step_m
    y_max = math.ceil((float(ys.max().item()) + padding_m) / grid_step_m) * grid_step_m
    return x_min, x_max, y_min, y_max


def main() -> None:
    args = parse_args()
    data = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(data.get("samples", [])):
        run_name = sample.get("run_name", f"sample_{i:03d}")
        teacher = sample["teacher"]
        pred = teacher["pred_xyz"].to(torch.float32)
        x_final = teacher["x_final"].to(torch.float32)
        x0 = teacher["x0"].to(torch.float32)

        pred_xy = pred[:, :2]
        # visualize latent x0/x_final as tiny side traces (scaled for readability)
        x0_xy = x0[:, :2] * 0.2
        xf_xy = x_final[:, :2] * 0.2

        x_min, x_max, y_min, y_max = bounds_xy([pred_xy, x0_xy, xf_xy], args.padding_m, args.grid_step_m)
        ppm = args.pixels_per_meter
        margin = 70
        w = int(round((x_max - x_min) * ppm)) + margin * 2
        h = int(round((y_max - y_min) * ppm)) + margin * 2

        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def to_px(x: float, y: float) -> tuple[int, int]:
            px = margin + int(round((x - x_min) * ppm))
            py = h - margin - int(round((y - y_min) * ppm))
            return px, py

        gx = x_min
        while gx <= x_max + 1e-6:
            draw.line([to_px(gx, y_min), to_px(gx, y_max)], fill=(220, 220, 220), width=1)
            gx += args.grid_step_m

        gy = y_min
        while gy <= y_max + 1e-6:
            draw.line([to_px(x_min, gy), to_px(x_max, gy)], fill=(220, 220, 220), width=1)
            gy += args.grid_step_m

        p_a = to_px(x_min, y_min)
        p_b = to_px(x_max, y_max)
        draw.rectangle([min(p_a[0], p_b[0]), min(p_a[1], p_b[1]), max(p_a[0], p_b[0]), max(p_a[1], p_b[1])], outline=(120, 120, 120), width=2)

        def poly(curve: torch.Tensor, color: tuple[int, int, int], width: int) -> None:
            pts = [to_px(float(p[0].item()), float(p[1].item())) for p in curve]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=width)
            for pt in pts:
                draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=color)

        poly(pred_xy, (37, 99, 235), 3)
        poly(x0_xy, (148, 163, 184), 2)
        poly(xf_xy, (220, 38, 38), 2)
        origin = to_px(0.0, 0.0)
        draw.ellipse([origin[0]-4, origin[1]-4, origin[0]+4, origin[1]+4], fill=(0, 0, 0))

        draw.text((10, 10), f"run={run_name}", fill=(0, 0, 0))
        draw.text((10, 26), f"clip_id={sample['meta'].get('clip_id','')}", fill=(0, 0, 0))
        draw.text((10, 42), f"grid={args.grid_step_m:.1f}m x {args.grid_step_m:.1f}m", fill=(0, 0, 0))
        draw.text((10, 58), "blue=pred_xyz, gray=x0*0.2, red=x_final*0.2", fill=(0, 0, 0))

        out_png = out_dir / f"traj_grid_{run_name}.png"
        img.save(out_png)
        print(out_png)


if __name__ == "__main__":
    main()
