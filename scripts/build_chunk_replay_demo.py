#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    data = np.asarray(field["data"], dtype=np.float32)
    shape = tuple(field["shape"])
    return data.reshape(shape)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def interp_series(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    return np.interp(times_dst, times_src, values_src, left=values_src[0], right=values_src[-1]).astype(np.float32)


def make_data_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_sample_identity(path: Path) -> dict[str, int]:
    match = re.search(r"chunk(\d+)_sid(\d+)_t0_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse sample identity from {path}")
    return {
        "chunk_id": int(match.group(1)),
        "sample_id": int(match.group(2)),
        "t0_us": int(match.group(3)),
    }


@dataclass
class ReplayData:
    request_path: Path
    output_path: Path
    chunk_id: int
    sample_id: int
    t0_us: int
    step_count: int
    output_text: str
    plan_dt_s: float
    horizon: int
    control_dt_s: float
    control_points: int
    hist_xyz: np.ndarray
    hist_rot: np.ndarray
    pred_xyz: np.ndarray
    pred_rot: np.ndarray
    yaw_local: np.ndarray
    speed_mps: np.ndarray
    curvature: np.ndarray
    packets: list[dict[str, Any]]
    image_uris: list[str]
    image_labels: list[str]
    timing: dict[str, Any]
    source_request: dict[str, Any]


def build_replay_data(
    request_path: Path,
    output_path: Path,
    control_dt_s: float,
    control_points: int,
    latest_frame_only: bool,
) -> ReplayData:
    identity = parse_sample_identity(output_path)

    request_root = json.loads(request_path.read_text())
    request = request_root["requests"][0]
    response = json.loads(output_path.read_text())["responses"][0]
    post = response["alpamayo_post_vlm"]
    fm = post["fm"]

    hist_xyz = np.load(request["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    hist_rot = np.load(request["ego_history_rot_npy"]).astype(np.float32)[0, 0]
    pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
    pred_rot = reshape_tensor_field(fm["pred_rot"])[0]

    yaw_local = np.arctan2(pred_rot[:, 1, 0], pred_rot[:, 0, 0]).astype(np.float32)
    plan_dt_s = float(fm["action_space_constants"]["dt_value"])
    plan_times = np.arange(len(pred_xyz), dtype=np.float32) * plan_dt_s

    speed = np.zeros(len(pred_xyz), dtype=np.float32)
    speed[0] = float(np.linalg.norm(pred_xyz[0, :2]) / max(plan_dt_s, 1e-6))
    if len(pred_xyz) > 1:
        delta_xy = pred_xyz[1:, :2] - pred_xyz[:-1, :2]
        speed[1:] = np.linalg.norm(delta_xy, axis=1) / max(plan_dt_s, 1e-6)

    yaw_unwrapped = unwrap_angles(yaw_local)
    ds = np.zeros(len(pred_xyz), dtype=np.float32)
    if len(pred_xyz) > 1:
        ds[1:] = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1)
    curvature = np.zeros(len(pred_xyz), dtype=np.float32)
    if len(pred_xyz) > 2:
        dyaw = np.diff(yaw_unwrapped)
        curvature[1:] = (dyaw / np.maximum(ds[1:], 1e-4)).astype(np.float32)
        curvature[0] = curvature[1]

    resample_times = np.arange(control_points, dtype=np.float32) * control_dt_s
    packet_count = max(1, int(math.ceil((plan_times[-1] if len(plan_times) else 0.0) / control_dt_s)) + 1)

    packets: list[dict[str, Any]] = []
    for tx_seq in range(packet_count):
        tx_time_s = tx_seq * control_dt_s
        query_times = tx_time_s + resample_times
        x = interp_series(plan_times, pred_xyz[:, 0], query_times)
        y = interp_series(plan_times, pred_xyz[:, 1], query_times)
        z = interp_series(plan_times, pred_xyz[:, 2], query_times)
        yaw = wrap_angles(interp_series(plan_times, yaw_unwrapped.astype(np.float32), query_times))
        v = interp_series(plan_times, speed, query_times)
        kappa = interp_series(plan_times, curvature, query_times)
        packets.append(
            {
                "header": {
                    "magic": "ALPA",
                    "version": 1,
                    "flags": 1,
                    "coord_mode": "ego_local",
                    "tx_seq": tx_seq,
                    "plan_seq": 1,
                    "source_sample_id": identity["sample_id"],
                    "source_chunk_id": identity["chunk_id"],
                    "source_t0_us": identity["t0_us"],
                    "tx_time_offset_s": round(float(tx_time_s), 6),
                    "num_points": control_points,
                    "dt_s": control_dt_s,
                    "horizon_s": round(float(control_points * control_dt_s), 6),
                },
                "points": [
                    {
                        "x_m": round(float(px), 6),
                        "y_m": round(float(py), 6),
                        "z_m": round(float(pz), 6),
                        "yaw_rad": round(float(pyaw), 6),
                        "v_mps": round(float(pv), 6),
                        "curvature": round(float(pk), 6),
                    }
                    for px, py, pz, pyaw, pv, pk in zip(x, y, z, yaw, v, kappa, strict=True)
                ],
            }
        )

    image_paths: list[Path] = []
    image_labels: list[str] = []
    for message in request["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and "image" in item:
                image_path = Path(item["image"])
                label = image_path.stem
                if latest_frame_only and not label.endswith("_f3"):
                    continue
                image_paths.append(image_path)
                image_labels.append(label)

    image_uris = [make_data_uri(path) for path in image_paths]

    return ReplayData(
        request_path=request_path,
        output_path=output_path,
        chunk_id=identity["chunk_id"],
        sample_id=identity["sample_id"],
        t0_us=identity["t0_us"],
        step_count=int(fm["num_steps"]),
        output_text=response["output_text"],
        plan_dt_s=plan_dt_s,
        horizon=len(pred_xyz),
        control_dt_s=control_dt_s,
        control_points=control_points,
        hist_xyz=hist_xyz,
        hist_rot=hist_rot,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        yaw_local=yaw_local,
        speed_mps=speed,
        curvature=curvature,
        packets=packets,
        image_uris=image_uris,
        image_labels=image_labels,
        timing=post["timing"],
        source_request=request,
    )


def write_npz(replay: ReplayData, out_path: Path) -> None:
    np.savez(
        out_path,
        chunk_id=np.int32(replay.chunk_id),
        sample_id=np.int32(replay.sample_id),
        t0_us=np.int64(replay.t0_us),
        step_count=np.int32(replay.step_count),
        plan_dt_s=np.float32(replay.plan_dt_s),
        control_dt_s=np.float32(replay.control_dt_s),
        control_points=np.int32(replay.control_points),
        hist_xyz=replay.hist_xyz,
        hist_rot=replay.hist_rot,
        pred_xyz_local=replay.pred_xyz,
        pred_rot_local=replay.pred_rot,
        yaw_local=replay.yaw_local,
        speed_mps=replay.speed_mps,
        curvature=replay.curvature,
    )


def make_json_serializable(replay: ReplayData) -> dict[str, Any]:
    return {
        "meta": {
            "chunk_id": replay.chunk_id,
            "sample_id": replay.sample_id,
            "t0_us": replay.t0_us,
            "coord_mode": "ego_local",
            "output_text": replay.output_text,
            "step_count": replay.step_count,
            "plan_dt_s": replay.plan_dt_s,
            "horizon": replay.horizon,
            "control_dt_s": replay.control_dt_s,
            "control_points": replay.control_points,
            "request_json": str(replay.request_path),
            "output_json": str(replay.output_path),
            "timing": replay.timing,
        },
        "history": {
            "xyz": replay.hist_xyz.tolist(),
            "rot": replay.hist_rot.tolist(),
        },
        "plan": {
            "pred_xyz_local": replay.pred_xyz.tolist(),
            "pred_rot_local": replay.pred_rot.tolist(),
            "yaw_local": replay.yaw_local.tolist(),
            "speed_mps": replay.speed_mps.tolist(),
            "curvature": replay.curvature.tolist(),
        },
    }


def pack_packet_binary(packet: dict[str, Any]) -> bytes:
    header = packet["header"]
    points = packet["points"]
    magic = 0x414C5041
    header_blob = struct.pack(
        "<IHHIIIIQQfHff",
        magic,
        int(header["version"]),
        int(header["flags"]),
        0,
        int(header["tx_seq"]),
        int(header["plan_seq"]),
        int(header["source_sample_id"]),
        int(header["source_t0_us"]),
        int(header["source_t0_us"]),
        float(header["tx_time_offset_s"]),
        int(header["num_points"]),
        float(header["dt_s"]),
        float(header["horizon_s"]),
    )
    point_blob = b"".join(
        struct.pack(
            "<ffffff",
            float(point["x_m"]),
            float(point["y_m"]),
            float(point["z_m"]),
            float(point["yaw_rad"]),
            float(point["v_mps"]),
            float(point["curvature"]),
        )
        for point in points
    )
    return header_blob + point_blob


def write_overview_png(replay: ReplayData, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    hist = replay.hist_xyz
    ax.plot(hist[:, 0], hist[:, 1], color="#7f7f7f", lw=2.0, label="ego_history")
    ax.plot(replay.pred_xyz[:, 0], replay.pred_xyz[:, 1], color="#1f77b4", lw=2.5, label="full_plan")

    sample_indices = [0, min(len(replay.packets) // 3, len(replay.packets) - 1), min((2 * len(replay.packets)) // 3, len(replay.packets) - 1)]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    labels = ["packet@0ms", "packet@mid1", "packet@mid2"]
    for idx, color, label in zip(sample_indices, colors, labels, strict=True):
        packet = replay.packets[idx]
        pts = np.asarray([[p["x_m"], p["y_m"]] for p in packet["points"]], dtype=np.float32)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.9, label=label)
        ax.scatter([pts[0, 0]], [pts[0, 1]], color=color, s=30)

    ax.scatter([0.0], [0.0], color="black", s=60, label="ego_now")
    ax.set_title(
        f"chunk{replay.chunk_id:04d} replay preview | sample {replay.sample_id} | step {replay.step_count}",
        fontsize=13,
    )
    ax.set_xlabel("x_local [m]")
    ax.set_ylabel("y_local [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_viewer_html(replay: ReplayData, out_path: Path) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk {replay.chunk_id:04d} Replay Viewer</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #202124;
      --muted: #6f6a60;
      --grid: #d8cfbf;
      --accent: #1f77b4;
      --accent2: #d62728;
      --accent3: #2e8b57;
      --line: #d4c6ad;
    }}
    body {{
      margin: 0;
      padding: 20px;
      background: radial-gradient(circle at top left, #fffaf0 0%, var(--bg) 65%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    h1 {{ margin: 0 0 10px; font-size: 26px; }}
    .sub {{ color: var(--muted); margin-bottom: 16px; }}
    .layout {{
      display: grid;
      grid-template-columns: 430px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(70, 50, 20, 0.08);
      padding: 16px;
    }}
    .cam-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .cam-grid img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      display: block;
      background: #ddd;
    }}
    .cam-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }}
    .stat {{
      background: #faf5ea;
      border-radius: 12px;
      padding: 10px;
      border: 1px solid #eadfc8;
      font-size: 13px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      background: linear-gradient(180deg, #fffefb 0%, #fbf7ef 100%);
      border-radius: 16px;
      border: 1px solid var(--line);
      display: block;
    }}
    .controls {{ display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
    input[type=range] {{ flex: 1 1 420px; }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      background: var(--ink);
      color: white;
      cursor: pointer;
    }}
    code {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      background: #f7f2e9;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .mono {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      background: #f7f2e9;
      border: 1px solid #eadfc8;
      border-radius: 12px;
      padding: 10px;
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <h1>Chunk {replay.chunk_id:04d} Replay Viewer</h1>
  <div class="sub">Local-frame replay prototype for control bridging. This is the same inference result resampled into 50 Hz packet horizons.</div>

  <div class="layout">
    <div class="panel">
      <div class="cam-grid" id="camGrid"></div>
      <div class="stats">
        <div class="stat"><strong>Output text</strong><br />{replay.output_text}</div>
        <div class="stat"><strong>Mode</strong><br />ego_local, step {replay.step_count}, horizon {replay.horizon} x {replay.plan_dt_s:.2f}s</div>
        <div class="stat"><strong>Sample</strong><br />chunk {replay.chunk_id:04d}, sid {replay.sample_id}, t0 {replay.t0_us}</div>
        <div class="stat"><strong>Control packet</strong><br />{replay.control_points} points x {replay.control_dt_s:.2f}s</div>
      </div>
      <div class="mono" id="packetText"></div>
    </div>

    <div class="panel">
      <canvas id="trajCanvas" width="920" height="920"></canvas>
      <div class="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <input id="frameSlider" type="range" min="0" max="{len(replay.packets) - 1}" value="0" />
        <span id="frameLabel"></span>
      </div>
      <div class="sub">Blue: full inference plan, Red: current 50 Hz control packet horizon, Gray: ego history, Black: ego now.</div>
    </div>
  </div>

  <script>
    const replay = {json.dumps({
        "meta": {
            "chunk_id": replay.chunk_id,
            "sample_id": replay.sample_id,
            "t0_us": replay.t0_us,
            "step_count": replay.step_count,
            "output_text": replay.output_text,
            "plan_dt_s": replay.plan_dt_s,
            "control_dt_s": replay.control_dt_s,
            "control_points": replay.control_points,
            "timing": replay.timing,
        },
        "history": replay.hist_xyz[:, :2].round(6).tolist(),
        "plan": replay.pred_xyz[:, :2].round(6).tolist(),
        "packet_paths": [[[p["x_m"], p["y_m"]] for p in packet["points"]] for packet in replay.packets],
        "packets": replay.packets,
        "images": [{"label": label, "uri": uri} for label, uri in zip(replay.image_labels, replay.image_uris)],
    })};

    const canvas = document.getElementById("trajCanvas");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("frameSlider");
    const frameLabel = document.getElementById("frameLabel");
    const packetText = document.getElementById("packetText");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");
    const camGrid = document.getElementById("camGrid");

    for (const image of replay.images) {{
      const wrapper = document.createElement("div");
      wrapper.innerHTML = `<img src="${{image.uri}}" alt="${{image.label}}" /><div class="cam-label">${{image.label}}</div>`;
      camGrid.appendChild(wrapper);
    }}

    const allPts = [...replay.history, ...replay.plan, ...replay.packet_paths.flat()];
    const xs = allPts.map(p => p[0]);
    const ys = allPts.map(p => p[1]);
    const xMin = Math.min(...xs, -0.5), xMax = Math.max(...xs, 0.5);
    const yMin = Math.min(...ys, -0.5), yMax = Math.max(...ys, 0.5);
    const pad = 0.12 * Math.max(xMax - xMin, yMax - yMin);
    const view = {{
      x0: xMin - pad,
      x1: xMax + pad,
      y0: yMin - pad,
      y1: yMax + pad,
    }};

    function toCanvas(pt) {{
      const margin = 70;
      const w = canvas.width - margin * 2;
      const h = canvas.height - margin * 2;
      const x = margin + (pt[0] - view.x0) / (view.x1 - view.x0) * w;
      const y = canvas.height - margin - (pt[1] - view.y0) / (view.y1 - view.y0) * h;
      return [x, y];
    }}

    function drawGrid() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fbf7ef";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#d8cfbf";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 8]);
      for (let gx = Math.ceil(view.x0); gx <= Math.floor(view.x1); gx += 1) {{
        const [x0, y0] = toCanvas([gx, view.y0]);
        const [x1, y1] = toCanvas([gx, view.y1]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }}
      for (let gy = Math.ceil(view.y0); gy <= Math.floor(view.y1); gy += 1) {{
        const [x0, y0] = toCanvas([view.x0, gy]);
        const [x1, y1] = toCanvas([view.x1, gy]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }}
      ctx.setLineDash([]);
    }}

    function drawPolyline(points, color, width, dashed=false) {{
      if (!points.length) return;
      ctx.beginPath();
      const start = toCanvas(points[0]);
      ctx.moveTo(start[0], start[1]);
      for (let i = 1; i < points.length; i++) {{
        const c = toCanvas(points[i]);
        ctx.lineTo(c[0], c[1]);
      }}
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [12, 9] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    function drawMarker(point, color, radius) {{
      const c = toCanvas(point);
      ctx.beginPath();
      ctx.arc(c[0], c[1], radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function drawAxes() {{
      const originX = toCanvas([0, view.y0])[0];
      const originY = toCanvas([view.x0, 0])[1];
      ctx.strokeStyle = "#9d927f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(originX, 40);
      ctx.lineTo(originX, canvas.height - 40);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(40, originY);
      ctx.lineTo(canvas.width - 40, originY);
      ctx.stroke();
    }}

    function renderFrame(idx) {{
      const packet = replay.packets[idx];
      const packetPath = replay.packet_paths[idx];
      drawGrid();
      drawAxes();
      drawPolyline(replay.history, "#7f7f7f", 4);
      drawPolyline(replay.plan, "#1f77b4", 4, true);
      drawPolyline(packetPath, "#d62728", 5);
      drawMarker([0, 0], "#000000", 8);
      drawMarker(replay.plan[replay.plan.length - 1], "#1f77b4", 7);
      drawMarker(packetPath[0], "#d62728", 7);
      drawMarker(packetPath[packetPath.length - 1], "#2e8b57", 7);

      frameLabel.textContent = `packet ${{idx + 1}} / ${{replay.packets.length}} | tx=${{packet.header.tx_time_offset_s.toFixed(2)}}s`;
      const first = packet.points[0];
      const last = packet.points[packet.points.length - 1];
      packetText.textContent =
        `header.magic=${{packet.header.magic}}\\n` +
        `coord_mode=${{packet.header.coord_mode}}\\n` +
        `tx_seq=${{packet.header.tx_seq}}  plan_seq=${{packet.header.plan_seq}}\\n` +
        `sample_id=${{packet.header.source_sample_id}}  chunk=${{packet.header.source_chunk_id}}\\n` +
        `tx_time_offset_s=${{packet.header.tx_time_offset_s}}\\n` +
        `num_points=${{packet.header.num_points}} dt=${{packet.header.dt_s}} horizon=${{packet.header.horizon_s}}\\n\\n` +
        `first point: x=${{first.x_m.toFixed(3)}}, y=${{first.y_m.toFixed(3)}}, yaw=${{first.yaw_rad.toFixed(3)}}, v=${{first.v_mps.toFixed(3)}}, kappa=${{first.curvature.toFixed(3)}}\\n` +
        `last point:  x=${{last.x_m.toFixed(3)}}, y=${{last.y_m.toFixed(3)}}, yaw=${{last.yaw_rad.toFixed(3)}}, v=${{last.v_mps.toFixed(3)}}, kappa=${{last.curvature.toFixed(3)}}\\n\\n` +
        `output_text: ${{replay.meta.output_text}}`;
    }}

    let playing = false;
    let timer = null;
    function stop() {{
      playing = false;
      playBtn.textContent = "Play";
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
    }}
    function start() {{
      if (playing) return;
      playing = true;
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        const next = (Number(slider.value) + 1) % replay.packets.length;
        slider.value = String(next);
        renderFrame(next);
      }}, 80);
    }}

    slider.addEventListener("input", () => renderFrame(Number(slider.value)));
    playBtn.addEventListener("click", () => playing ? stop() : start());
    resetBtn.addEventListener("click", () => {{
      stop();
      slider.value = "0";
      renderFrame(0);
    }});

    renderFrame(0);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_readme(replay: ReplayData, out_path: Path) -> None:
    stem = f"chunk{replay.chunk_id:04d}_step{replay.step_count:02d}"
    lines = [
        "# Chunk Replay Demo",
        "",
        f"- chunk: `{replay.chunk_id:04d}`",
        f"- sample id: `{replay.sample_id}`",
        f"- t0_us: `{replay.t0_us}`",
        f"- inference source: `{replay.output_path}`",
        f"- request source: `{replay.request_path}`",
        f"- coord mode: `ego_local`",
        f"- chosen FM steps: `{replay.step_count}`",
        f"- plan dt: `{replay.plan_dt_s:.2f}s`",
        f"- control packet: `{replay.control_points}` points @ `{replay.control_dt_s:.2f}s`",
        "",
        "## Files",
        "",
        f"- `plan_bank_{stem}_local.npz`: plan bank prototype for replay",
        f"- `plan_bank_{stem}_local.json`: readable version of the same data",
        f"- `packet_preview_{stem}_local.json`: all 50 Hz packet horizons",
        f"- `packet_preview_{stem}_local.bin`: first packet encoded as a binary preview",
        f"- `{stem}_replay_viewer.html`: self-contained browser viewer",
        f"- `{stem}_replay_overview.png`: static overview image",
        "",
        "## Notes",
        "",
        "- This prototype stays in `ego_local` coordinates because the current FM output is local-frame.",
        "- If the controller expects world/map coordinates later, we will need reference world pose metadata and a transform step.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-chunk replay demo from offline inference output.")
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    parser.add_argument("--latest-frame-only", action="store_true")
    args = parser.parse_args()

    replay = build_replay_data(
        request_path=args.request_json,
        output_path=args.output_json,
        control_dt_s=args.control_dt,
        control_points=args.control_points,
        latest_frame_only=args.latest_frame_only,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"chunk{replay.chunk_id:04d}_step{replay.step_count:02d}"
    npz_path = out_dir / f"plan_bank_{stem}_local.npz"
    json_path = out_dir / f"plan_bank_{stem}_local.json"
    packet_json_path = out_dir / f"packet_preview_{stem}_local.json"
    packet_bin_path = out_dir / f"packet_preview_{stem}_local.bin"
    html_path = out_dir / f"{stem}_replay_viewer.html"
    png_path = out_dir / f"{stem}_replay_overview.png"
    readme_path = out_dir / "README.md"

    write_npz(replay, npz_path)
    json_path.write_text(json.dumps(make_json_serializable(replay), indent=2), encoding="utf-8")
    packet_json_path.write_text(json.dumps({"packets": replay.packets}, indent=2), encoding="utf-8")
    packet_bin_path.write_bytes(pack_packet_binary(replay.packets[0]))
    write_overview_png(replay, png_path)
    write_viewer_html(replay, html_path)
    write_readme(replay, readme_path)

    print(json.dumps(
        {
            "chunk_id": replay.chunk_id,
            "sample_id": replay.sample_id,
            "step_count": replay.step_count,
            "plan_bank_npz": str(npz_path),
            "plan_bank_json": str(json_path),
            "packet_preview_json": str(packet_json_path),
            "packet_preview_bin": str(packet_bin_path),
            "viewer_html": str(html_path),
            "overview_png": str(png_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
