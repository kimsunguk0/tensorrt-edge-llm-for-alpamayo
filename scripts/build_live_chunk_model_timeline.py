#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)
IDENTITY_RE = re.compile(r"output_chunk(\d+)_sid(\d+)_t0_(\d+)\.json$")
CAMERA_LABELS = {
    0: "Front Left",
    1: "Front",
    2: "Front Right",
    6: "Front Tele",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * slat * slat)
    x = (N + alt_m) * clat * clon
    y = (N + alt_m) * clat * slon
    z = (N * (1.0 - E2_WGS84) + alt_m) * slat
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu(xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref_xyz = geodetic_to_ecef(
        np.asarray([ref_lat_deg], dtype=np.float64),
        np.asarray([ref_lon_deg], dtype=np.float64),
        np.asarray([ref_alt_m], dtype=np.float64),
    )[0]
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)
    rot = np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )
    return (xyz - ref_xyz) @ rot.T


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.float32).reshape(tuple(field["shape"]))


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def interp_series(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    return np.interp(times_dst, times_src, values_src, left=values_src[0], right=values_src[-1]).astype(np.float32)


def parse_identity(path: Path) -> tuple[int, int, int]:
    m = IDENTITY_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse identity from {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def write_overview_png(out_path: Path, world_t0_xy: np.ndarray, world_plan_xy: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(world_t0_xy[:, 0], world_t0_xy[:, 1], color="#c8c1b4", linewidth=2.0)
    stride = max(1, len(world_plan_xy) // 20)
    for idx in range(0, len(world_plan_xy), stride):
        alpha = 0.12 + 0.45 * (idx / max(len(world_plan_xy) - 1, 1))
        ax.plot(world_plan_xy[idx, :, 0], world_plan_xy[idx, :, 1], color="#1f77b4", linewidth=1.2, alpha=alpha)
    ax.scatter([world_t0_xy[0, 0]], [world_t0_xy[0, 1]], color="#2ca02c", s=60, label="start")
    ax.scatter([world_t0_xy[-1, 0]], [world_t0_xy[-1, 1]], color="#d62728", s=60, label="end")
    ax.set_title("chunk0011 model timeline overview")
    ax.set_xlabel("x_world_enu [m]")
    ax.set_ylabel("y_world_enu [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_viewer_html(out_path: Path, data_json_name: str) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk Model Timeline Replay</title>
  <style>
    :root {
      --bg: #f6f2e8;
      --panel: rgba(255,255,255,0.94);
      --ink: #202124;
      --muted: #6e695f;
      --line: #dccfb9;
    }
    body {
      margin: 0;
      padding: 18px;
      background: radial-gradient(circle at top right, #fffaf1 0%, var(--bg) 62%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }
    .layout {
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 38px rgba(80, 63, 27, 0.08);
    }
    .images {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .image-card img {
      width: 100%;
      border-radius: 12px;
      display: block;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }
    .label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .plots {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    canvas {
      width: 100%;
      height: 420px;
      border-radius: 14px;
      background: #fffdf8;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }
    .controls {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    input[type=range] {
      width: 100%;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
    }
    .meta {
      font-size: 14px;
      line-height: 1.55;
      color: var(--ink);
      white-space: pre-line;
    }
    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .swatch {
      display: inline-block;
      width: 18px;
      height: 3px;
      vertical-align: middle;
      margin-right: 6px;
      border-radius: 999px;
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="panel">
      <h2 id="title">Chunk Model Timeline Replay</h2>
      <div class="controls">
        <div>
          <input id="slider" type="range" min="0" max="0" value="0" />
        </div>
        <div class="row">
          <button id="play">Play</button>
          <button id="prev">Prev</button>
          <button id="next">Next</button>
          <span id="counter"></span>
        </div>
        <div class="legend">
          <span><span class="swatch" style="background:#6d7b8d"></span>history</span>
          <span><span class="swatch" style="background:#1f77b4"></span>model plan</span>
          <span><span class="swatch" style="background:#d62728"></span>50Hz packet</span>
          <span><span class="swatch" style="background:#333"></span>ego now</span>
        </div>
      </div>
      <div id="meta" class="meta"></div>
      <div id="images" class="images" style="margin-top:14px"></div>
    </div>
    <div class="panel">
      <div class="plots">
        <div>
          <div class="label">World ENU Replay</div>
          <canvas id="world"></canvas>
        </div>
        <div>
          <div class="label">Local Control View</div>
          <canvas id="local"></canvas>
        </div>
      </div>
    </div>
  </div>
  <script>
    const dataUrl = "__DATA_JSON__";
    const worldCanvas = document.getElementById("world");
    const localCanvas = document.getElementById("local");
    const slider = document.getElementById("slider");
    const playBtn = document.getElementById("play");
    const prevBtn = document.getElementById("prev");
    const nextBtn = document.getElementById("next");
    const counter = document.getElementById("counter");
    const meta = document.getElementById("meta");
    const title = document.getElementById("title");
    const images = document.getElementById("images");
    let payload = null;
    let idx = 0;
    let timer = null;

    function fitCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * ratio);
      canvas.height = Math.round(rect.height * ratio);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return ctx;
    }

    function project(pt, bounds, width, height, pad) {
      const w = bounds.max_x - bounds.min_x || 1;
      const h = bounds.max_y - bounds.min_y || 1;
      const sx = (width - pad * 2) / w;
      const sy = (height - pad * 2) / h;
      const s = Math.min(sx, sy);
      const ox = pad + (width - pad * 2 - w * s) * 0.5;
      const oy = pad + (height - pad * 2 - h * s) * 0.5;
      return [ox + (pt[0] - bounds.min_x) * s, height - (oy + (pt[1] - bounds.min_y) * s)];
    }

    function drawPolyline(ctx, pts, bounds, color, width, dash) {
      if (!pts || pts.length === 0) return;
      const rect = ctx.canvas.getBoundingClientRect();
      const pad = 22;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      pts.forEach((pt, i) => {
        const [x, y] = project(pt, bounds, rect.width, rect.height, pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function drawDot(ctx, pt, bounds, color, radius) {
      const rect = ctx.canvas.getBoundingClientRect();
      const pad = 22;
      const [x, y] = project(pt, bounds, rect.width, rect.height, pad);
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function clearCanvas(ctx) {
      const rect = ctx.canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, rect.width, rect.height);
    }

    function renderImageCards() {
      images.innerHTML = "";
      for (const panel of payload.camera_panels) {
        const card = document.createElement("div");
        card.className = "image-card";
        card.innerHTML = `<div class="label">${panel.label}</div><img id="img-${panel.camera_id}" />`;
        images.appendChild(card);
      }
    }

    function localBounds() {
      const e = payload.local_extent_m || 1;
      return { min_x: -e, max_x: e, min_y: -e, max_y: e };
    }

    function render() {
      const sample = payload.samples[idx];
      slider.value = String(idx);
      counter.textContent = `${idx + 1} / ${payload.samples.length}`;

      meta.textContent =
        `chunk: ${payload.chunk_id}\n` +
        `sample_id: ${sample.sample_id}\n` +
        `t_rel: ${sample.t_rel_s.toFixed(2)} s\n` +
        `t0_us: ${sample.t0_us}\n` +
        `plan_dt: ${sample.plan_dt_s.toFixed(2)} s\n` +
        `control: ${payload.control_points} pts @ ${payload.control_dt_s.toFixed(2)} s\n` +
        `output_text: ${sample.output_text ?? ""}`;

      for (const panel of payload.camera_panels) {
        const camId = panel.camera_id;
        const img = document.getElementById(`img-${camId}`);
        if (img) {
          img.src = sample.images[String(camId)];
        }
      }

      const worldCtx = fitCanvas(worldCanvas);
      clearCanvas(worldCtx);
      drawPolyline(worldCtx, payload.world_path_t0_xy, payload.world_bounds, "#d6d0c5", 2.0, []);
      drawPolyline(worldCtx, sample.history_world_xy, payload.world_bounds, "#6d7b8d", 2.2, [6, 4]);
      drawPolyline(worldCtx, sample.plan_world_xy, payload.world_bounds, "#1f77b4", 2.6, []);
      drawPolyline(worldCtx, sample.packet_world_xy, payload.world_bounds, "#d62728", 3.2, []);
      drawDot(worldCtx, sample.history_world_xy[sample.history_world_xy.length - 1], payload.world_bounds, "#222", 4.2);

      const localCtx = fitCanvas(localCanvas);
      clearCanvas(localCtx);
      const lb = localBounds();
      drawPolyline(localCtx, sample.history_local_xy, lb, "#6d7b8d", 2.2, [6, 4]);
      drawPolyline(localCtx, sample.plan_local_xy, lb, "#1f77b4", 2.6, []);
      drawPolyline(localCtx, sample.packet_local_xy, lb, "#d62728", 3.2, []);
      drawDot(localCtx, [0, 0], lb, "#222", 4.2);
    }

    function setIndex(next) {
      idx = Math.max(0, Math.min(payload.samples.length - 1, next));
      render();
    }

    function togglePlay() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
        playBtn.textContent = "Play";
      } else {
        timer = window.setInterval(() => {
          if (idx >= payload.samples.length - 1) {
            togglePlay();
            return;
          }
          setIndex(idx + 1);
        }, 100);
        playBtn.textContent = "Pause";
      }
    }

    fetch(dataUrl)
      .then((resp) => resp.json())
      .then((data) => {
        payload = data;
        title.textContent = `Chunk ${String(payload.chunk_id).padStart(4, "0")} Model Timeline Replay`;
        renderImageCards();
        slider.max = String(payload.samples.length - 1);
        slider.addEventListener("input", (ev) => setIndex(Number(ev.target.value)));
        playBtn.addEventListener("click", togglePlay);
        prevBtn.addEventListener("click", () => setIndex(idx - 1));
        nextBtn.addEventListener("click", () => setIndex(idx + 1));
        window.addEventListener("resize", render);
        render();
      })
      .catch((err) => {
        meta.textContent = `Failed to load viewer data: ${err}`;
      });
  </script>
</body>
</html>
"""
    out_path.write_text(html.replace("__DATA_JSON__", data_json_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model timeline replay viewer for a fully inferred raw chunk.")
    parser.add_argument("--dataset-root", default="/root/live_dataset/2025-03-31-test2")
    parser.add_argument("--chunk-id", type=int, default=11)
    parser.add_argument("--request-bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing output_chunkXXXX_sid*.json files")
    parser.add_argument(
        "--viewer-dir",
        type=Path,
        default=Path("/root/TensorRT-Edge-LLM-v060/output/dashboards/real_rutuime/chunk0011_model_timeline_replay"),
    )
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    viewer_dir = args.viewer_dir
    ensure_dir(viewer_dir)

    manifest = json.loads((args.request_bank_root / "manifest.json").read_text())
    manifest_by_sample: dict[int, dict[str, Any]] = {int(item["sample_id"]): item for item in manifest}

    summary = json.loads((args.request_bank_root / "summary.json").read_text())
    ref_lat, ref_lon, ref_alt = summary["chunk_ref_lla"]
    camera_runtime_order = summary.get("camera_runtime_order") or [
        {"semantic_name": "left", "camera_id": 0},
        {"semantic_name": "front", "camera_id": 1},
        {"semantic_name": "right", "camera_id": 2},
        {"semantic_name": "front_tele", "camera_id": 6},
    ]
    camera_ids = [int(item["camera_id"]) for item in camera_runtime_order]
    camera_panels = [
        {
            "camera_id": int(item["camera_id"]),
            "semantic_name": str(item["semantic_name"]),
            "label": CAMERA_LABELS.get(int(item["camera_id"]), str(item["semantic_name"])),
        }
        for item in camera_runtime_order
    ]

    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    image_symlink = viewer_dir / "images_bank"
    safe_symlink(args.request_bank_root / "images", image_symlink)

    output_paths = sorted(args.output_root.glob(f"output_chunk{args.chunk_id:04d}_sid*_t0_*.json"))
    if not output_paths:
        raise RuntimeError(f"No output files found in {args.output_root}")

    sample_records: list[dict[str, Any]] = []
    world_t0_list: list[list[float]] = []
    world_plan_list: list[np.ndarray] = []
    local_extent = 1.0

    for output_path in output_paths:
        chunk_id, sample_id, t0_us = parse_identity(output_path)
        meta = manifest_by_sample.get(sample_id)
        if meta is None:
            continue
        t0_utc_ns = int(meta["t0_utc_ns"])

        t0_lat = float(np.interp([t0_utc_ns], utc, lat)[0])
        t0_lon = float(np.interp([t0_utc_ns], utc, lon)[0])
        t0_alt = float(np.interp([t0_utc_ns], utc, alt)[0])
        t0_world = ecef_to_enu(
            geodetic_to_ecef(np.asarray([t0_lat]), np.asarray([t0_lon]), np.asarray([t0_alt])),
            ref_lat,
            ref_lon,
            ref_alt,
        )[0].astype(np.float32)

        yaw_times = np.asarray([t0_utc_ns - 100_000_000, t0_utc_ns, t0_utc_ns + 100_000_000], dtype=np.int64)
        yaw_lat = np.interp(yaw_times, utc, lat)
        yaw_lon = np.interp(yaw_times, utc, lon)
        yaw_alt = np.interp(yaw_times, utc, alt)
        yaw_world_pts = ecef_to_enu(geodetic_to_ecef(yaw_lat, yaw_lon, yaw_alt), ref_lat, ref_lon, ref_alt)
        yaw_vec = yaw_world_pts[2, :2] - yaw_world_pts[0, :2]
        t0_yaw = float(np.arctan2(yaw_vec[1], yaw_vec[0]))
        c = math.cos(t0_yaw)
        s = math.sin(t0_yaw)
        R0 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        req_path = Path(meta["request_json"])
        req_obj = json.loads(req_path.read_text())
        request_item = req_obj["requests"][0]
        hist_xyz = np.load(request_item["ego_history_xyz_npy"]).astype(np.float32)[0, 0]

        output = json.loads(output_path.read_text())["responses"][0]
        post = output["alpamayo_post_vlm"]
        fm = post["fm"]
        pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
        pred_rot = reshape_tensor_field(fm["pred_rot"])[0]
        plan_dt_s = float(fm["action_space_constants"]["dt_value"])
        plan_times = np.arange(len(pred_xyz), dtype=np.float32) * plan_dt_s
        yaw_local = yaw_from_rot(pred_rot)
        yaw_unwrapped = unwrap_angles(yaw_local)

        speed = np.zeros(len(pred_xyz), dtype=np.float32)
        speed[0] = float(np.linalg.norm(pred_xyz[0, :2]) / max(plan_dt_s, 1e-6))
        if len(pred_xyz) > 1:
            speed[1:] = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1) / max(plan_dt_s, 1e-6)

        curvature = np.zeros(len(pred_xyz), dtype=np.float32)
        if len(pred_xyz) > 2:
            ds = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1)
            dyaw = np.diff(yaw_unwrapped)
            curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
            curvature[0] = curvature[1]

        packet_times = np.arange(args.control_points, dtype=np.float32) * args.control_dt
        packet_x = interp_series(plan_times, pred_xyz[:, 0], packet_times)
        packet_y = interp_series(plan_times, pred_xyz[:, 1], packet_times)
        packet_local = np.stack([packet_x, packet_y], axis=1).astype(np.float32)

        hist_world = (hist_xyz @ R0.T + t0_world).astype(np.float32)
        plan_world = (pred_xyz @ R0.T + t0_world).astype(np.float32)
        packet_world = (np.pad(packet_local, ((0, 0), (0, 1))) @ R0.T + t0_world).astype(np.float32)

        local_extent = max(
            local_extent,
            float(np.max(np.abs(np.concatenate([hist_xyz[:, :2], pred_xyz[:, :2], packet_local], axis=0)))),
        )
        world_t0_list.append(t0_world[:2].tolist())
        world_plan_list.append(plan_world[:, :2])

        stem = req_path.stem.replace("request_", "")
        sample_records.append(
            {
                "sample_id": sample_id,
                "t0_us": t0_us,
                "t_rel_s": round(float((t0_utc_ns - int(manifest[0]["t0_utc_ns"])) / 1e9), 3),
                "plan_dt_s": plan_dt_s,
                "output_text": output.get("output_text"),
                "timing": post.get("timing", {}),
                "history_local_xy": hist_xyz[:, :2].round(5).tolist(),
                "plan_local_xy": pred_xyz[:, :2].round(5).tolist(),
                "packet_local_xy": packet_local.round(5).tolist(),
                "history_world_xy": hist_world[:, :2].round(5).tolist(),
                "plan_world_xy": plan_world[:, :2].round(5).tolist(),
                "packet_world_xy": packet_world[:, :2].round(5).tolist(),
                "images": {
                    str(cam_id): f"images_bank/{stem}/cam{cam_id}_f3.png"
                    for cam_id in camera_ids
                },
            }
        )

    if not sample_records:
        raise RuntimeError("No sample records could be built from outputs")

    world_points = np.concatenate([np.asarray(world_t0_list, dtype=np.float32)] + world_plan_list, axis=0)
    world_bounds = {
        "min_x": float(np.min(world_points[:, 0])),
        "max_x": float(np.max(world_points[:, 0])),
        "min_y": float(np.min(world_points[:, 1])),
        "max_y": float(np.max(world_points[:, 1])),
    }

    data = {
        "chunk_id": args.chunk_id,
        "num_samples": len(sample_records),
        "control_dt_s": args.control_dt,
        "control_points": args.control_points,
        "camera_panels": camera_panels,
        "local_extent_m": local_extent,
        "world_bounds": world_bounds,
        "world_path_t0_xy": world_t0_list,
        "samples": sample_records,
    }

    data_json = viewer_dir / f"chunk{args.chunk_id:04d}_model_viewer_data.json"
    data_json.write_text(json.dumps(data))
    html_path = viewer_dir / f"chunk{args.chunk_id:04d}_model_timeline_viewer.html"
    write_viewer_html(html_path, data_json.name)
    write_overview_png(
        viewer_dir / f"chunk{args.chunk_id:04d}_model_timeline_overview.png",
        np.asarray(world_t0_list, dtype=np.float32),
        np.stack(world_plan_list, axis=0),
    )

    summary_out = {
        "chunk_id": args.chunk_id,
        "num_outputs": len(output_paths),
        "num_samples_in_viewer": len(sample_records),
        "viewer_html": str(html_path),
        "viewer_data_json": str(data_json),
        "overview_png": str(viewer_dir / f"chunk{args.chunk_id:04d}_model_timeline_overview.png"),
    }
    (viewer_dir / "summary.json").write_text(json.dumps(summary_out, indent=2))
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
