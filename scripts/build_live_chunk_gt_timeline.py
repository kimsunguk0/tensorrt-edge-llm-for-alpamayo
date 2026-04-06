#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)

CAMERA_SPECS = [
    ("front", "camera_front", "front_frame_id"),
    ("front_tele", "camera_front_tele", "front_tele_frame_id"),
    ("left", "camera_left", "left_frame_id"),
    ("right", "camera_right", "right_frame_id"),
]


@dataclass
class CameraFrameRef:
    sample_idx: int
    sample_id: int
    frame_id: int
    frame_index_in_chunk: int
    image_relpath: str


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


def yaw_to_rot(yaw_rad: np.ndarray) -> np.ndarray:
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    rot = np.zeros((len(yaw_rad), 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 1] = -sin_y
    rot[:, 1, 0] = sin_y
    rot[:, 1, 1] = cos_y
    rot[:, 2, 2] = 1.0
    return rot


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def interp_series(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    return np.interp(times_dst, times_src, values_src, left=values_src[0], right=values_src[-1]).astype(np.float32)


def build_pose_bundle(
    gnss_valid: pd.DataFrame,
    chunk_ref_lla: tuple[float, float, float],
    t0_utc_ns: int,
    history_len: int,
    future_len: int,
    dt_ns: int,
) -> dict[str, np.ndarray]:
    hist_times = np.asarray([t0_utc_ns - (history_len - 1 - i) * dt_ns for i in range(history_len)], dtype=np.int64)
    fut_times = np.asarray([t0_utc_ns + (i + 1) * dt_ns for i in range(future_len)], dtype=np.int64)
    all_times = np.concatenate([hist_times, fut_times], axis=0)

    utc = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt = gnss_valid["alt"].to_numpy(dtype=np.float64)

    interp_lat = np.interp(all_times, utc, lat)
    interp_lon = np.interp(all_times, utc, lon)
    interp_alt = np.interp(all_times, utc, alt)

    interp_xyz_ecef = geodetic_to_ecef(interp_lat, interp_lon, interp_alt)
    world_xyz = ecef_to_enu(interp_xyz_ecef, *chunk_ref_lla).astype(np.float32)

    world_vel_xy = np.zeros((len(all_times), 2), dtype=np.float32)
    world_vel_xy[1:-1] = (world_xyz[2:, :2] - world_xyz[:-2, :2]) / float(2 * dt_ns / 1e9)
    world_vel_xy[0] = (world_xyz[1, :2] - world_xyz[0, :2]) / float(dt_ns / 1e9)
    world_vel_xy[-1] = (world_xyz[-1, :2] - world_xyz[-2, :2]) / float(dt_ns / 1e9)
    world_yaw = np.arctan2(world_vel_xy[:, 1], world_vel_xy[:, 0]).astype(np.float32)
    world_rot = yaw_to_rot(world_yaw)

    hist_world_xyz = world_xyz[:history_len]
    fut_world_xyz = world_xyz[history_len:]
    hist_world_rot = world_rot[:history_len]
    fut_world_rot = world_rot[history_len:]

    p0 = hist_world_xyz[-1]
    R0 = hist_world_rot[-1]
    R0_t = R0.T

    hist_local_xyz = ((hist_world_xyz - p0) @ R0).astype(np.float32)
    fut_local_xyz = ((fut_world_xyz - p0) @ R0).astype(np.float32)
    hist_local_rot = np.einsum("ij,tjk->tik", R0_t, hist_world_rot).astype(np.float32)
    fut_local_rot = np.einsum("ij,tjk->tik", R0_t, fut_world_rot).astype(np.float32)

    return {
        "history_times_utc_ns": hist_times,
        "future_times_utc_ns": fut_times,
        "history_world_xyz": hist_world_xyz,
        "future_world_xyz": fut_world_xyz,
        "history_world_rot": hist_world_rot,
        "future_world_rot": fut_world_rot,
        "history_local_xyz": hist_local_xyz,
        "future_local_xyz": fut_local_xyz,
        "history_local_rot": hist_local_rot,
        "future_local_rot": fut_local_rot,
        "world_yaw": world_yaw,
    }


def build_control_packet(
    future_local_xyz: np.ndarray,
    future_local_rot: np.ndarray,
    future_world_xyz: np.ndarray,
    future_times_s: np.ndarray,
    control_dt_s: float,
    control_points: int,
) -> dict[str, np.ndarray]:
    future_local_yaw = yaw_from_rot(future_local_rot)
    future_local_yaw_unwrapped = unwrap_angles(future_local_yaw)
    packet_times = np.arange(control_points, dtype=np.float32) * control_dt_s

    packet_local_x = interp_series(future_times_s, future_local_xyz[:, 0], packet_times)
    packet_local_y = interp_series(future_times_s, future_local_xyz[:, 1], packet_times)
    packet_local_z = interp_series(future_times_s, future_local_xyz[:, 2], packet_times)
    packet_local_yaw = wrap_angles(interp_series(future_times_s, future_local_yaw_unwrapped.astype(np.float32), packet_times))

    future_speed = np.zeros(len(future_local_xyz), dtype=np.float32)
    future_speed[0] = float(np.linalg.norm(future_local_xyz[0, :2]) / max(future_times_s[0], 1e-6))
    if len(future_local_xyz) > 1:
        future_speed[1:] = np.linalg.norm(future_local_xyz[1:, :2] - future_local_xyz[:-1, :2], axis=1) / max(
            future_times_s[1] - future_times_s[0], 1e-6
        )
    packet_speed = interp_series(future_times_s, future_speed, packet_times)

    future_curvature = np.zeros(len(future_local_xyz), dtype=np.float32)
    if len(future_local_xyz) > 2:
        ds = np.linalg.norm(future_local_xyz[1:, :2] - future_local_xyz[:-1, :2], axis=1)
        dyaw = np.diff(future_local_yaw_unwrapped)
        future_curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
        future_curvature[0] = future_curvature[1]
    packet_curvature = interp_series(future_times_s, future_curvature, packet_times)

    packet_world_x = interp_series(future_times_s, future_world_xyz[:, 0], packet_times)
    packet_world_y = interp_series(future_times_s, future_world_xyz[:, 1], packet_times)
    packet_world_z = interp_series(future_times_s, future_world_xyz[:, 2], packet_times)

    return {
        "packet_times_s": packet_times,
        "packet_local_xyz": np.stack([packet_local_x, packet_local_y, packet_local_z], axis=1).astype(np.float32),
        "packet_world_xyz": np.stack([packet_world_x, packet_world_y, packet_world_z], axis=1).astype(np.float32),
        "packet_local_yaw": packet_local_yaw.astype(np.float32),
        "packet_speed_mps": packet_speed.astype(np.float32),
        "packet_curvature": packet_curvature.astype(np.float32),
    }


def export_camera_images(
    dataset_root: Path,
    chunk_id: int,
    samples: pd.DataFrame,
    image_root: Path,
    width: int,
    height: int,
) -> dict[str, list[str]]:
    image_root.mkdir(parents=True, exist_ok=True)
    relpaths_by_camera: dict[str, list[str]] = {}

    for semantic_name, sensor_name, frame_col in CAMERA_SPECS:
        frame_table = pd.read_parquet(dataset_root / "sensors" / sensor_name / "frames.parquet")
        frame_table = frame_table[frame_table["chunk_id"] == chunk_id][["frame_id", "frame_index_in_chunk"]].copy()
        frame_lookup = dict(
            zip(
                frame_table["frame_id"].astype(int).tolist(),
                frame_table["frame_index_in_chunk"].astype(int).tolist(),
                strict=True,
            )
        )

        refs: list[CameraFrameRef] = []
        relpaths: list[str] = []
        cam_dir = image_root / semantic_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx, row in samples.reset_index(drop=True).iterrows():
            frame_id = int(row[frame_col])
            frame_index = frame_lookup[frame_id]
            relpath = f"images/{semantic_name}/{sample_idx:04d}.jpg"
            refs.append(
                CameraFrameRef(
                    sample_idx=sample_idx,
                    sample_id=int(row["sample_id"]),
                    frame_id=frame_id,
                    frame_index_in_chunk=frame_index,
                    image_relpath=relpath,
                )
            )
            relpaths.append(relpath)

        video_path = dataset_root / "sensors" / sensor_name / "chunks" / f"chunk_{chunk_id:04d}.mkv"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        refs.sort(key=lambda item: item.frame_index_in_chunk)
        ref_pos = 0
        next_ref = refs[ref_pos] if refs else None
        frame_index = 0
        while next_ref is not None:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Unexpected EOF while extracting {video_path} at frame {frame_index}")
            if frame_index == next_ref.frame_index_in_chunk:
                resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                out_path = image_root.parent / next_ref.image_relpath
                cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                ref_pos += 1
                next_ref = refs[ref_pos] if ref_pos < len(refs) else None
            frame_index += 1
        cap.release()
        relpaths_by_camera[semantic_name] = relpaths

    return relpaths_by_camera


def write_overview_png(out_path: Path, t0_world_xy: np.ndarray, futures_world_xy: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t0_world_xy[:, 0], t0_world_xy[:, 1], color="#6d7b8d", linewidth=2.2, label="ego_t0_path")
    stride = max(1, len(futures_world_xy) // 24)
    for idx in range(0, len(futures_world_xy), stride):
        alpha = 0.15 + 0.45 * (idx / max(len(futures_world_xy) - 1, 1))
        ax.plot(futures_world_xy[idx, :, 0], futures_world_xy[idx, :, 1], color="#1f77b4", linewidth=1.2, alpha=alpha)
    ax.scatter([t0_world_xy[0, 0]], [t0_world_xy[0, 1]], color="#2ca02c", s=60, label="start")
    ax.scatter([t0_world_xy[-1, 0]], [t0_world_xy[-1, 1]], color="#d62728", s=60, label="end")
    ax.set_title("chunk0011 raw GT timeline overview")
    ax.set_xlabel("x_world_enu [m]")
    ax.set_ylabel("y_world_enu [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_viewer_payload(
    samples: pd.DataFrame,
    relpaths_by_camera: dict[str, list[str]],
    arrays: dict[str, np.ndarray],
    history_len: int,
    future_len: int,
) -> dict[str, Any]:
    local_extent = float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        arrays["history_local_xyz"][:, :, :2].reshape(-1, 2),
                        arrays["future_local_xyz"][:, :, :2].reshape(-1, 2),
                        arrays["packet_local_xyz"][:, :, :2].reshape(-1, 2),
                    ],
                    axis=0,
                )
            )
        )
    )
    world_points = np.concatenate(
        [
            arrays["t0_world_xyz"][:, None, :2],
            arrays["history_world_xyz"][:, :, :2],
            arrays["future_world_xyz"][:, :, :2],
            arrays["packet_world_xyz"][:, :, :2],
        ],
        axis=1,
    ).reshape(-1, 2)
    min_xy = np.min(world_points, axis=0)
    max_xy = np.max(world_points, axis=0)
    world_bounds = {
        "min_x": float(min_xy[0]),
        "max_x": float(max_xy[0]),
        "min_y": float(min_xy[1]),
        "max_y": float(max_xy[1]),
    }

    payload = {
        "chunk_id": int(samples["chunk_id"].iloc[0]),
        "num_samples": int(len(samples)),
        "history_len": history_len,
        "future_len": future_len,
        "control_points": int(arrays["packet_local_xyz"].shape[1]),
        "control_dt_s": float(arrays["packet_dt_s"][0]),
        "plan_dt_s": float(arrays["plan_dt_s"][0]),
        "local_extent_m": local_extent,
        "world_bounds": world_bounds,
        "world_path_t0_xy": arrays["t0_world_xyz"][:, :2].round(5).tolist(),
        "samples": [],
    }

    for idx, row in samples.reset_index(drop=True).iterrows():
        payload["samples"].append(
            {
                "sample_id": int(row["sample_id"]),
                "t0_utc_ns": int(row["t0_utc_ns"]),
                "t_rel_s": round(float((int(row["t0_utc_ns"]) - int(samples["t0_utc_ns"].iloc[0])) / 1e9), 3),
                "gnss_row_id": int(row["gnss_ins_row_id"]),
                "images": {
                    cam: relpaths_by_camera[cam][idx]
                    for cam, _, _ in CAMERA_SPECS
                },
                "history_local_xy": arrays["history_local_xyz"][idx, :, :2].round(5).tolist(),
                "future_local_xy": arrays["future_local_xyz"][idx, :, :2].round(5).tolist(),
                "packet_local_xy": arrays["packet_local_xyz"][idx, :, :2].round(5).tolist(),
                "history_world_xy": arrays["history_world_xyz"][idx, :, :2].round(5).tolist(),
                "future_world_xy": arrays["future_world_xyz"][idx, :, :2].round(5).tolist(),
                "packet_world_xy": arrays["packet_world_xyz"][idx, :, :2].round(5).tolist(),
                "packet_speed_mps": arrays["packet_speed_mps"][idx].round(5).tolist(),
                "packet_curvature": arrays["packet_curvature"][idx].round(6).tolist(),
            }
        )
    return payload


def write_viewer_html(out_path: Path, data_filename: str) -> None:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk Timeline Replay</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: rgba(255,255,255,0.94);
      --ink: #202124;
      --muted: #6e695f;
      --line: #dccfb9;
      --blue: #1f77b4;
      --red: #d62728;
      --gray: #6d7b8d;
    }}
    body {{
      margin: 0;
      padding: 18px;
      background: radial-gradient(circle at top right, #fffaf1 0%, var(--bg) 62%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 38px rgba(80, 63, 27, 0.08);
    }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .image-card img {{
      width: 100%;
      border-radius: 12px;
      display: block;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .plots {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    canvas {{
      width: 100%;
      height: 420px;
      border-radius: 14px;
      background: #fffdf8;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}
    .controls {{
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    input[type=range] {{
      width: 100%;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
    }}
    .meta {{
      font-size: 14px;
      line-height: 1.55;
      color: var(--ink);
      white-space: pre-line;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }}
    .swatch {{
      display: inline-block;
      width: 18px;
      height: 3px;
      vertical-align: middle;
      margin-right: 6px;
      border-radius: 999px;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="panel">
      <h2 id="title">Chunk Timeline Replay</h2>
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
          <span><span class="swatch" style="background:#1f77b4"></span>future/plan</span>
          <span><span class="swatch" style="background:#d62728"></span>50Hz packet</span>
          <span><span class="swatch" style="background:#333"></span>ego now</span>
        </div>
      </div>
      <div id="meta" class="meta"></div>
      <div class="images" style="margin-top:14px">
        <div class="image-card"><div class="label">Front</div><img id="img-front" /></div>
        <div class="image-card"><div class="label">Front Tele</div><img id="img-front_tele" /></div>
        <div class="image-card"><div class="label">Left</div><img id="img-left" /></div>
        <div class="image-card"><div class="label">Right</div><img id="img-right" /></div>
      </div>
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
    const dataUrl = "__DATA_FILENAME__";
    const worldCanvas = document.getElementById("world");
    const localCanvas = document.getElementById("local");
    const slider = document.getElementById("slider");
    const playBtn = document.getElementById("play");
    const prevBtn = document.getElementById("prev");
    const nextBtn = document.getElementById("next");
    const counter = document.getElementById("counter");
    const meta = document.getElementById("meta");
    const title = document.getElementById("title");
    const imageIds = ["front", "front_tele", "left", "right"];
    let payload = null;
    let idx = 0;
    let timer = null;

    function fitCanvas(canvas) {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * ratio);
      canvas.height = Math.round(rect.height * ratio);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return ctx;
    }}

    function project(pt, bounds, width, height, pad) {{
      const w = bounds.max_x - bounds.min_x || 1;
      const h = bounds.max_y - bounds.min_y || 1;
      const sx = (width - pad * 2) / w;
      const sy = (height - pad * 2) / h;
      const s = Math.min(sx, sy);
      const ox = pad + (width - pad * 2 - w * s) * 0.5;
      const oy = pad + (height - pad * 2 - h * s) * 0.5;
      return [
        ox + (pt[0] - bounds.min_x) * s,
        height - (oy + (pt[1] - bounds.min_y) * s),
      ];
    }}

    function drawPolyline(ctx, pts, bounds, color, width, dash) {{
      if (!pts || pts.length === 0) return;
      const rect = ctx.canvas.getBoundingClientRect();
      const pad = 22;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      pts.forEach((pt, i) => {{
        const [x, y] = project(pt, bounds, rect.width, rect.height, pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();
    }}

    function drawDot(ctx, pt, bounds, color, radius) {{
      const rect = ctx.canvas.getBoundingClientRect();
      const pad = 22;
      const [x, y] = project(pt, bounds, rect.width, rect.height, pad);
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }}

    function clearCanvas(ctx) {{
      const rect = ctx.canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, rect.width, rect.height);
    }}

    function localBounds() {{
      const e = payload.local_extent_m || 1;
      return {{ min_x: -e, max_x: e, min_y: -e, max_y: e }};
    }}

    function render() {{
      const sample = payload.samples[idx];
      slider.value = String(idx);
      counter.textContent = `${{idx + 1}} / ${{payload.samples.length}}`;

      meta.textContent =
        `chunk: ${{payload.chunk_id}}\\n` +
        `sample_id: ${{sample.sample_id}}\\n` +
        `t_rel: ${{sample.t_rel_s.toFixed(2)}} s\\n` +
        `t0_utc_ns: ${{sample.t0_utc_ns}}\\n` +
        `gnss_row_id: ${{sample.gnss_row_id}}\\n` +
        `plan_dt: ${{payload.plan_dt_s.toFixed(2)}} s\\n` +
        `control: ${{payload.control_points}} pts @ ${{payload.control_dt_s.toFixed(2)}} s`;

      imageIds.forEach((key) => {{
        document.getElementById(`img-${{key}}`).src = sample.images[key];
      }});

      const worldCtx = fitCanvas(worldCanvas);
      clearCanvas(worldCtx);
      drawPolyline(worldCtx, payload.world_path_t0_xy, payload.world_bounds, "#d6d0c5", 2.0, []);
      drawPolyline(worldCtx, sample.history_world_xy, payload.world_bounds, "#6d7b8d", 2.2, [6, 4]);
      drawPolyline(worldCtx, sample.future_world_xy, payload.world_bounds, "#1f77b4", 2.6, []);
      drawPolyline(worldCtx, sample.packet_world_xy, payload.world_bounds, "#d62728", 3.2, []);
      drawDot(worldCtx, sample.history_world_xy[sample.history_world_xy.length - 1], payload.world_bounds, "#222", 4.2);

      const localCtx = fitCanvas(localCanvas);
      clearCanvas(localCtx);
      const lb = localBounds();
      drawPolyline(localCtx, sample.history_local_xy, lb, "#6d7b8d", 2.2, [6, 4]);
      drawPolyline(localCtx, sample.future_local_xy, lb, "#1f77b4", 2.6, []);
      drawPolyline(localCtx, sample.packet_local_xy, lb, "#d62728", 3.2, []);
      drawDot(localCtx, [0, 0], lb, "#222", 4.2);
    }}

    function setIndex(next) {{
      idx = Math.max(0, Math.min(payload.samples.length - 1, next));
      render();
    }}

    function togglePlay() {{
      if (timer) {{
        window.clearInterval(timer);
        timer = null;
        playBtn.textContent = "Play";
      }} else {{
        timer = window.setInterval(() => {{
          if (idx >= payload.samples.length - 1) {{
            togglePlay();
            return;
          }}
          setIndex(idx + 1);
        }}, 100);
        playBtn.textContent = "Pause";
      }}
    }}

    fetch(dataUrl)
      .then((resp) => resp.json())
      .then((data) => {{
        payload = data;
        title.textContent = `Chunk ${{String(payload.chunk_id).padStart(4, "0")}} Raw GT Timeline Replay`;
        slider.max = String(payload.samples.length - 1);
        slider.addEventListener("input", (ev) => setIndex(Number(ev.target.value)));
        playBtn.addEventListener("click", togglePlay);
        prevBtn.addEventListener("click", () => setIndex(idx - 1));
        nextBtn.addEventListener("click", () => setIndex(idx + 1));
        window.addEventListener("resize", render);
        render();
      })
      .catch((err) => {{
        meta.textContent = `Failed to load viewer data: ${{err}}`;
      }});
  </script>
</body>
</html>
"""
    out_path.write_text(html.replace("__DATA_FILENAME__", data_filename))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw live-dataset chunk timeline replay viewer.")
    parser.add_argument("--dataset-root", default="/root/live_dataset/2025-03-31-test2")
    parser.add_argument("--chunk-id", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        default="/root/TensorRT-Edge-LLM-v060/output/dashboards/real_rutuime/chunk0011_raw_gt_timeline_replay",
    )
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--future-len", type=int, default=64)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--control-dt-s", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = out_dir / "images"

    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[["frame_id", "chunk_id"]]
    sample_index = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    samples = sample_index[sample_index["chunk_id"] == args.chunk_id].copy().sort_values("t0_utc_ns").reset_index(drop=True)
    if samples.empty:
        raise RuntimeError(f"No samples found for chunk {args.chunk_id}")

    gnss = pd.read_parquet(dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet")
    gnss_valid = gnss[gnss["lat"].notna() & gnss["lon"].notna() & gnss["alt"].notna()].copy().sort_values("timestamp_utc_ns")
    dt_ns = int(round(args.dt_s * 1e9))

    first_t0 = int(samples["t0_utc_ns"].iloc[0])
    utc_valid = gnss_valid["timestamp_utc_ns"].to_numpy(dtype=np.int64)
    lat_valid = gnss_valid["lat"].to_numpy(dtype=np.float64)
    lon_valid = gnss_valid["lon"].to_numpy(dtype=np.float64)
    alt_valid = gnss_valid["alt"].to_numpy(dtype=np.float64)
    chunk_ref_lla = (
        float(np.interp([first_t0], utc_valid, lat_valid)[0]),
        float(np.interp([first_t0], utc_valid, lon_valid)[0]),
        float(np.interp([first_t0], utc_valid, alt_valid)[0]),
    )

    n = len(samples)
    history_world_xyz = np.zeros((n, args.history_len, 3), dtype=np.float32)
    future_world_xyz = np.zeros((n, args.future_len, 3), dtype=np.float32)
    history_local_xyz = np.zeros((n, args.history_len, 3), dtype=np.float32)
    future_local_xyz = np.zeros((n, args.future_len, 3), dtype=np.float32)
    history_local_rot = np.zeros((n, args.history_len, 3, 3), dtype=np.float32)
    future_local_rot = np.zeros((n, args.future_len, 3, 3), dtype=np.float32)
    t0_world_xyz = np.zeros((n, 3), dtype=np.float32)
    t0_world_yaw = np.zeros((n,), dtype=np.float32)
    packet_local_xyz = np.zeros((n, args.control_points, 3), dtype=np.float32)
    packet_world_xyz = np.zeros((n, args.control_points, 3), dtype=np.float32)
    packet_local_yaw = np.zeros((n, args.control_points), dtype=np.float32)
    packet_speed_mps = np.zeros((n, args.control_points), dtype=np.float32)
    packet_curvature = np.zeros((n, args.control_points), dtype=np.float32)

    future_times_s = np.arange(args.future_len, dtype=np.float32) * args.dt_s + args.dt_s
    for idx, row in samples.iterrows():
        pose = build_pose_bundle(
            gnss_valid=gnss_valid,
            chunk_ref_lla=chunk_ref_lla,
            t0_utc_ns=int(row["t0_utc_ns"]),
            history_len=args.history_len,
            future_len=args.future_len,
            dt_ns=dt_ns,
        )
        history_world_xyz[idx] = pose["history_world_xyz"]
        future_world_xyz[idx] = pose["future_world_xyz"]
        history_local_xyz[idx] = pose["history_local_xyz"]
        future_local_xyz[idx] = pose["future_local_xyz"]
        history_local_rot[idx] = pose["history_local_rot"]
        future_local_rot[idx] = pose["future_local_rot"]
        t0_world_xyz[idx] = pose["history_world_xyz"][-1]
        t0_world_yaw[idx] = pose["world_yaw"][args.history_len - 1]

        packet = build_control_packet(
            future_local_xyz=pose["future_local_xyz"],
            future_local_rot=pose["future_local_rot"],
            future_world_xyz=pose["future_world_xyz"],
            future_times_s=future_times_s,
            control_dt_s=args.control_dt_s,
            control_points=args.control_points,
        )
        packet_local_xyz[idx] = packet["packet_local_xyz"]
        packet_world_xyz[idx] = packet["packet_world_xyz"]
        packet_local_yaw[idx] = packet["packet_local_yaw"]
        packet_speed_mps[idx] = packet["packet_speed_mps"]
        packet_curvature[idx] = packet["packet_curvature"]

    relpaths_by_camera = export_camera_images(
        dataset_root=dataset_root,
        chunk_id=args.chunk_id,
        samples=samples,
        image_root=image_root,
        width=args.image_width,
        height=args.image_height,
    )

    arrays = {
        "history_world_xyz": history_world_xyz,
        "future_world_xyz": future_world_xyz,
        "history_local_xyz": history_local_xyz,
        "future_local_xyz": future_local_xyz,
        "history_local_rot": history_local_rot,
        "future_local_rot": future_local_rot,
        "t0_world_xyz": t0_world_xyz,
        "t0_world_yaw": t0_world_yaw,
        "packet_local_xyz": packet_local_xyz,
        "packet_world_xyz": packet_world_xyz,
        "packet_local_yaw": packet_local_yaw,
        "packet_speed_mps": packet_speed_mps,
        "packet_curvature": packet_curvature,
        "packet_dt_s": np.full((n,), args.control_dt_s, dtype=np.float32),
        "plan_dt_s": np.full((n,), args.dt_s, dtype=np.float32),
        "sample_id": samples["sample_id"].to_numpy(dtype=np.int32),
        "t0_utc_ns": samples["t0_utc_ns"].to_numpy(dtype=np.int64),
    }

    np.savez_compressed(out_dir / f"chunk{args.chunk_id:04d}_gt_plan_bank.npz", **arrays)

    viewer_payload = build_viewer_payload(
        samples=samples,
        relpaths_by_camera=relpaths_by_camera,
        arrays=arrays,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    data_json = out_dir / f"chunk{args.chunk_id:04d}_viewer_data.json"
    data_json.write_text(json.dumps(viewer_payload))

    summary = {
        "dataset_root": str(dataset_root),
        "chunk_id": args.chunk_id,
        "num_samples": int(n),
        "sample_id_min": int(samples["sample_id"].min()),
        "sample_id_max": int(samples["sample_id"].max()),
        "t0_start_utc_ns": int(samples["t0_utc_ns"].min()),
        "t0_end_utc_ns": int(samples["t0_utc_ns"].max()),
        "span_seconds": float((int(samples["t0_utc_ns"].max()) - int(samples["t0_utc_ns"].min())) / 1e9),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "plan_dt_s": args.dt_s,
        "control_dt_s": args.control_dt_s,
        "control_points": args.control_points,
        "chunk_ref_lla": [float(x) for x in chunk_ref_lla],
        "outputs": {
            "plan_bank_npz": str(out_dir / f"chunk{args.chunk_id:04d}_gt_plan_bank.npz"),
            "viewer_html": str(out_dir / f"chunk{args.chunk_id:04d}_raw_gt_timeline_viewer.html"),
            "viewer_data_json": str(data_json),
            "overview_png": str(out_dir / f"chunk{args.chunk_id:04d}_raw_gt_timeline_overview.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_overview_png(
        out_dir / f"chunk{args.chunk_id:04d}_raw_gt_timeline_overview.png",
        t0_world_xy=t0_world_xyz[:, :2],
        futures_world_xy=future_world_xyz[:, :, :2],
    )
    write_viewer_html(out_dir / f"chunk{args.chunk_id:04d}_raw_gt_timeline_viewer.html", data_json.name)

    readme = f"""# Chunk {args.chunk_id:04d} Raw GT Timeline Replay

This directory contains a chunk-wide replay viewer built directly from `/root/live_dataset/2025-03-31-test2`.

- Source chunk: `chunk_{args.chunk_id:04d}`
- Samples: `{n}` at 10 Hz-equivalent `t0`
- History: `{args.history_len}` steps
- Future: `{args.future_len}` steps
- Plan dt: `{args.dt_s:.2f}s`
- Control replay: `{args.control_points}` points @ `{args.control_dt_s:.2f}s`

Important:
- This is a raw-dataset GT/stub replay, not model inference output.
- Camera panels show the current frame at each `t0`.
- The blue path is the GT future from interpolated GNSS position.
- The red path is the 50 Hz control packet horizon resampled from that GT future.

Open via a local browser over HTTP:

- `chunk{args.chunk_id:04d}_raw_gt_timeline_viewer.html`
"""
    (out_dir / "README.md").write_text(readme)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
