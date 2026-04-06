#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reshape_tensor_field(field: dict[str, Any]) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.float32).reshape(tuple(field["shape"]))


def make_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).astype(np.float32)


def unwrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw.astype(np.float64))


def wrap_angles(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def interp_series(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    return np.interp(times_dst, times_src, values_src, left=values_src[0], right=values_src[-1]).astype(np.float32)


IDENTITY_RE = re.compile(r"chunk(\d{4})_sid(\d+)_t0_(\d+)")


def parse_identity(path: Path) -> tuple[int, int, int]:
    m = IDENTITY_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse chunk/sample/t0 from {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


@dataclass
class Sample:
    chunk_id: int
    sample_id: int
    t0_us: int
    request_path: Path
    output_path: Path
    output_text: str
    hist_xyz: np.ndarray
    pred_xyz: np.ndarray
    pred_rot: np.ndarray
    yaw_local: np.ndarray
    speed_mps: np.ndarray
    curvature: np.ndarray
    packet_path: np.ndarray
    image_uris: list[str]
    image_labels: list[str]
    timing: dict[str, Any]
    plan_dt_s: float
    step_count: int


def load_sample(output_path: Path, request_path: Path, latest_frame_only: bool, control_dt_s: float, control_points: int) -> Sample:
    chunk_id, sample_id, t0_us = parse_identity(output_path)
    output = json.loads(output_path.read_text())["responses"][0]
    request = json.loads(request_path.read_text())
    if "requests" in request:
        request_item = request["requests"][0]
    else:
        request_item = request

    fm = output["alpamayo_post_vlm"]["fm"]
    hist_xyz = np.load(request_item["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    pred_xyz = reshape_tensor_field(fm["pred_xyz"])[0]
    pred_rot = reshape_tensor_field(fm["pred_rot"])[0]
    yaw_local = yaw_from_rot(pred_rot)

    plan_dt_s = float(fm["action_space_constants"]["dt_value"])
    times = np.arange(len(pred_xyz), dtype=np.float32) * plan_dt_s

    speed = np.zeros(len(pred_xyz), dtype=np.float32)
    speed[0] = float(np.linalg.norm(pred_xyz[0, :2]) / max(plan_dt_s, 1e-6))
    if len(pred_xyz) > 1:
        speed[1:] = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1) / max(plan_dt_s, 1e-6)

    curvature = np.zeros(len(pred_xyz), dtype=np.float32)
    if len(pred_xyz) > 2:
        ds = np.linalg.norm(pred_xyz[1:, :2] - pred_xyz[:-1, :2], axis=1)
        dyaw = np.diff(unwrap_angles(yaw_local))
        curvature[1:] = (dyaw / np.maximum(ds, 1e-4)).astype(np.float32)
        curvature[0] = curvature[1]

    packet_times = np.arange(control_points, dtype=np.float32) * control_dt_s
    packet_x = interp_series(times, pred_xyz[:, 0], packet_times)
    packet_y = interp_series(times, pred_xyz[:, 1], packet_times)
    packet_path = np.stack([packet_x, packet_y], axis=1)

    image_paths: list[Path] = []
    image_labels: list[str] = []
    for message in request_item["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and "image" in item:
                img = Path(item["image"])
                label = img.stem
                if latest_frame_only and not label.endswith("_f3"):
                    continue
                image_paths.append(img)
                image_labels.append(label)

    return Sample(
        chunk_id=chunk_id,
        sample_id=sample_id,
        t0_us=t0_us,
        request_path=request_path,
        output_path=output_path,
        output_text=output["output_text"],
        hist_xyz=hist_xyz,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        yaw_local=yaw_local,
        speed_mps=speed,
        curvature=curvature,
        packet_path=packet_path,
        image_uris=[make_data_uri(path) for path in image_paths],
        image_labels=image_labels,
        timing=output["alpamayo_post_vlm"]["timing"],
        plan_dt_s=plan_dt_s,
        step_count=int(fm["num_steps"]),
    )


def find_request_for_sample(
    chunk_id: int,
    sample_id: int,
    t0_us: int,
    request_search_roots: list[Path],
) -> Path | None:
    names = [
        f"request_fp16_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}.json",
        f"request_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}_step10.json",
        f"request_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}_step08.json",
        f"request_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}_step06.json",
        f"request_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}_step04.json",
        f"request_chunk{chunk_id:04d}_sid{sample_id:05d}_t0_{t0_us}_step02.json",
    ]
    for root in request_search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def collect_samples(
    chunk_id: int,
    output_root: Path,
    request_search_roots: list[Path],
    latest_frame_only: bool,
    control_dt_s: float,
    control_points: int,
) -> list[Sample]:
    output_paths = sorted(output_root.glob(f"output_chunk{chunk_id:04d}_sid*_t0_*.json"))
    samples: list[Sample] = []
    for output_path in output_paths:
        c, sample_id, t0_us = parse_identity(output_path)
        request_path = find_request_for_sample(c, sample_id, t0_us, request_search_roots)
        if request_path is None:
            continue
        samples.append(load_sample(output_path, request_path, latest_frame_only, control_dt_s, control_points))
    samples.sort(key=lambda s: s.t0_us)
    return samples


def write_summary_png(samples: list[Sample], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    palette = plt.get_cmap("tab20")
    for idx, sample in enumerate(samples):
        color = palette(idx % 20)
        ax.plot(sample.pred_xyz[:, 0], sample.pred_xyz[:, 1], color=color, lw=2.0, alpha=0.85)
        ax.scatter([sample.pred_xyz[-1, 0]], [sample.pred_xyz[-1, 1]], color=color, s=22)
    ax.scatter([0.0], [0.0], color="black", s=60, label="ego_now")
    ax.set_title(f"chunk{samples[0].chunk_id:04d} timeline overview ({len(samples)} samples)")
    ax.set_xlabel("x_local [m]")
    ax.set_ylabel("y_local [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_viewer_html(samples: list[Sample], out_path: Path) -> None:
    chunk_id = samples[0].chunk_id
    data = {
        "chunk_id": chunk_id,
        "count": len(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "t0_us": sample.t0_us,
                "output_text": sample.output_text,
                "plan_dt_s": sample.plan_dt_s,
                "step_count": sample.step_count,
                "timing": sample.timing,
                "history": sample.hist_xyz[:, :2].round(6).tolist(),
                "plan": sample.pred_xyz[:, :2].round(6).tolist(),
                "packet": sample.packet_path.round(6).tolist(),
                "images": [{"label": label, "uri": uri} for label, uri in zip(sample.image_labels, sample.image_uris)],
            }
            for sample in samples
        ],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Chunk {chunk_id:04d} Timeline Replay</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: rgba(255,255,255,0.93);
      --ink: #1e1f21;
      --muted: #6d685f;
      --line: #dccfb9;
      --accent: #1f77b4;
      --danger: #d62728;
    }}
    body {{
      margin: 0;
      padding: 20px;
      background: radial-gradient(circle at top right, #fffaf1 0%, var(--bg) 60%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 14px 30px rgba(70, 50, 20, 0.08);
      padding: 16px;
    }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color: var(--muted); margin-bottom: 12px; }}
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
    }}
    .cam-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .stat {{
      margin-top: 12px;
      background: #faf5ea;
      border: 1px solid #eadfc8;
      border-radius: 12px;
      padding: 10px;
      font-size: 13px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffefb 0%, #fbf7ef 100%);
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    input[type=range] {{ flex: 1 1 440px; }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      background: var(--ink);
      color: white;
      cursor: pointer;
    }}
    .mono {{
      margin-top: 12px;
      white-space: pre-wrap;
      font-family: "Consolas", monospace;
      font-size: 12px;
      background: #f7f2e9;
      border: 1px solid #eadfc8;
      border-radius: 12px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <h1>Chunk {chunk_id:04d} Timeline Replay</h1>
  <div class="sub">This viewer steps across all saved samples that belong to the same chunk. If only one sample exists in the workspace, the timeline will only have one state.</div>

  <div class="layout">
    <div class="panel">
      <div class="cam-grid" id="camGrid"></div>
      <div class="stat" id="metaBox"></div>
      <div class="mono" id="timelineBox"></div>
    </div>
    <div class="panel">
      <canvas id="trajCanvas" width="920" height="920"></canvas>
      <div class="controls">
        <button id="playBtn">Play</button>
        <button id="resetBtn">Reset</button>
        <input id="sampleSlider" type="range" min="0" max="{len(samples) - 1}" value="0" />
        <span id="sampleLabel"></span>
      </div>
      <div class="sub">Gray: ego history, Blue: selected sample's full plan, Red: selected sample's first 0.5 s control packet horizon.</div>
    </div>
  </div>

  <script>
    const data = {json.dumps(data)};
    const canvas = document.getElementById("trajCanvas");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("sampleSlider");
    const sampleLabel = document.getElementById("sampleLabel");
    const metaBox = document.getElementById("metaBox");
    const timelineBox = document.getElementById("timelineBox");
    const camGrid = document.getElementById("camGrid");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");

    const allPts = [];
    for (const sample of data.samples) {{
      allPts.push(...sample.history, ...sample.plan, ...sample.packet);
    }}
    const xs = allPts.map(p => p[0]);
    const ys = allPts.map(p => p[1]);
    const xMin = Math.min(...xs, -0.5), xMax = Math.max(...xs, 0.5);
    const yMin = Math.min(...ys, -0.5), yMax = Math.max(...ys, 0.5);
    const pad = 0.12 * Math.max(xMax - xMin, yMax - yMin);
    const view = {{ x0: xMin - pad, x1: xMax + pad, y0: yMin - pad, y1: yMax + pad }};

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
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }}
      for (let gy = Math.ceil(view.y0); gy <= Math.floor(view.y1); gy += 1) {{
        const [x0, y0] = toCanvas([view.x0, gy]);
        const [x1, y1] = toCanvas([view.x1, gy]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
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
      ctx.beginPath(); ctx.moveTo(originX, 40); ctx.lineTo(originX, canvas.height - 40); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(40, originY); ctx.lineTo(canvas.width - 40, originY); ctx.stroke();
    }}

    function loadImages(sample) {{
      camGrid.innerHTML = "";
      for (const image of sample.images) {{
        const wrapper = document.createElement("div");
        wrapper.innerHTML = `<img src="${{image.uri}}" alt="${{image.label}}" /><div class="cam-label">${{image.label}}</div>`;
        camGrid.appendChild(wrapper);
      }}
    }}

    function renderSample(idx) {{
      const sample = data.samples[idx];
      loadImages(sample);
      drawGrid();
      drawAxes();
      drawPolyline(sample.history, "#7f7f7f", 4);
      drawPolyline(sample.plan, "#1f77b4", 4, true);
      drawPolyline(sample.packet, "#d62728", 5);
      drawMarker([0, 0], "#000000", 8);
      drawMarker(sample.plan[sample.plan.length - 1], "#1f77b4", 7);
      drawMarker(sample.packet[0], "#d62728", 7);

      sampleLabel.textContent = `sample ${{idx + 1}} / ${{data.count}}`;
      metaBox.innerHTML =
        `<strong>chunk</strong> ${{data.chunk_id.toString().padStart(4, "0")}}<br/>` +
        `<strong>sample id</strong> ${{sample.sample_id}}<br/>` +
        `<strong>t0_us</strong> ${{sample.t0_us}}<br/>` +
        `<strong>steps</strong> ${{sample.step_count}}<br/>` +
        `<strong>plan dt</strong> ${{sample.plan_dt_s.toFixed(2)}}s<br/>` +
        `<strong>output</strong> ${{sample.output_text}}`;

      const firstT0 = data.samples[0].t0_us;
      const relSec = (sample.t0_us - firstT0) / 1e6;
      timelineBox.textContent =
        `timeline index=${{idx}} / ${{data.count - 1}}\\n` +
        `relative chunk time=${{relSec.toFixed(3)}}s\\n` +
        `saved samples in workspace=${{data.count}}\\n\\n` +
        `timing:\\n` +
        `guided_pass_ms=${{sample.timing.guided_pass_ms}}\\n` +
        `fm_wall_ms=${{sample.timing.fm_wall_ms}}\\n` +
        `total_post_vlm_ms=${{sample.timing.total_post_vlm_ms}}`;
    }}

    let playing = false;
    let timer = null;
    function stop() {{
      playing = false;
      playBtn.textContent = "Play";
      if (timer) {{ clearInterval(timer); timer = null; }}
    }}
    function start() {{
      if (playing) return;
      playing = true;
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        const next = (Number(slider.value) + 1) % data.count;
        slider.value = String(next);
        renderSample(next);
      }}, 350);
    }}

    slider.addEventListener("input", () => renderSample(Number(slider.value)));
    playBtn.addEventListener("click", () => playing ? stop() : start());
    resetBtn.addEventListener("click", () => {{
      stop();
      slider.value = "0";
      renderSample(0);
    }});

    renderSample(0);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_readme(samples: list[Sample], out_path: Path, output_root: Path) -> None:
    chunk_id = samples[0].chunk_id
    lines = [
        "# Chunk Timeline Replay",
        "",
        f"- chunk: `{chunk_id:04d}`",
        f"- matched samples in current workspace: `{len(samples)}`",
        f"- output search root: `{output_root}`",
        "",
        "## Important",
        "",
        "- This viewer only includes samples that are actually saved in the current workspace.",
        "- If you expected an entire 60 s chunk but only one sample is listed here, that means only one extracted t0 for this chunk is currently present on disk.",
        "- Once more t0 samples are extracted for the same chunk, rerunning this script will automatically expand the timeline.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chunk timeline replay viewer from all saved samples for one chunk.")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing output_chunkXXXX_sid*.json files")
    parser.add_argument("--request-roots", type=Path, nargs="+", required=True, help="Directories to search for matching request JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--control-points", type=int, default=25)
    parser.add_argument("--latest-frame-only", action="store_true")
    args = parser.parse_args()

    samples = collect_samples(
        chunk_id=args.chunk_id,
        output_root=args.output_root,
        request_search_roots=args.request_roots,
        latest_frame_only=args.latest_frame_only,
        control_dt_s=args.control_dt,
        control_points=args.control_points,
    )
    if not samples:
        raise SystemExit(f"No saved samples found for chunk {args.chunk_id:04d}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"chunk{args.chunk_id:04d}_timeline_replay_viewer.html"
    png_path = out_dir / f"chunk{args.chunk_id:04d}_timeline_overview.png"
    summary_path = out_dir / f"chunk{args.chunk_id:04d}_timeline_summary.json"
    readme_path = out_dir / "README.md"

    summary = {
        "chunk_id": args.chunk_id,
        "num_samples_found": len(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "t0_us": sample.t0_us,
                "output_json": str(sample.output_path),
                "request_json": str(sample.request_path),
            }
            for sample in samples
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_png(samples, png_path)
    write_viewer_html(samples, html_path)
    write_readme(samples, readme_path, args.output_root)
    print(json.dumps(
        {
            "chunk_id": args.chunk_id,
            "num_samples_found": len(samples),
            "viewer_html": str(html_path),
            "overview_png": str(png_path),
            "summary_json": str(summary_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
