#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fm_model_defaults import first_existing_fm_engine
from run_raw_dataset_one_shot_udp import build_result_artifacts, ensure_exists, load_pandas
from run_request_bank_persistent import build_env, read_status


BUILD_REQUEST_BANK = SCRIPT_DIR / "build_live_chunk_request_bank.py"


@dataclass
class RequestEntry:
    request_path: Path
    sample_id: int
    front_frame_id: int
    t0_utc_ns: int
    t0_us: int
    actual_offset_s: float
    ego_history_xyz_npy: str
    ego_history_rot_npy: str
    selected_frames: dict[str, list[int]]


@dataclass
class ReplaySelection:
    entry: RequestEntry
    processed_index: int
    selected_index: int
    skipped_since_last: int
    skipped_total: int
    desired_offset_s: float | None


class ReplayViewerState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "phase": "initializing",
            "message": "starting replay",
            "started_unix_s": time.time(),
            "updated_unix_s": time.time(),
            "current_request_index": 0,
            "total_requests": 0,
            "udp_sent_count": 0,
            "replay_mode": "sequential",
            "skipped_request_count": 0,
            "alpamayo_nav_cfg": False,
            "current_nav": None,
            "latest": None,
        }

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)
            self._state["updated_unix_s"] = time.time()

    def set_latest(
        self,
        *,
        final_summary: dict[str, Any],
        ac_summary: dict[str, Any],
        gt_summary: dict[str, Any],
        chosen_source: str,
        tx_row: dict[str, Any],
        nav_payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._state["latest"] = {
                "chosen_source": chosen_source,
                "tx_row": copy.deepcopy(tx_row),
                "final_path": copy.deepcopy(final_summary),
                "ac_decoded_path": copy.deepcopy(ac_summary),
                "gt_path": copy.deepcopy(gt_summary),
                "nav": copy.deepcopy(nav_payload),
            }
            self._state["updated_unix_s"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chunk UDP Replay Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1117;
      --panel: #121b24;
      --panel2: #18232e;
      --text: #e6eef5;
      --muted: #93a7b8;
      --cyan: #67d5ff;
      --green: #72e39c;
      --orange: #ffb86b;
      --pink: #ff79c6;
      --border: rgba(255,255,255,0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #0e151d, #091017);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .wrap {
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
    }
    .top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }
    .title {
      font-size: 22px;
      font-weight: 700;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(560px, 1.15fr) minmax(420px, 0.95fr);
      align-items: start;
      gap: 16px;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .image-card {
      display: grid;
      gap: 8px;
    }
    .final-banner {
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(103,213,255,0.16), rgba(103,213,255,0.06));
      border: 1px solid rgba(103,213,255,0.22);
      display: grid;
      gap: 6px;
    }
    .final-banner-label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .final-banner-text {
      font-size: 22px;
      line-height: 1.25;
      font-weight: 700;
      color: #eef8ff;
      word-break: break-word;
    }
    .image-title {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .viewer-img {
      width: 100%;
      height: 200px;
      object-fit: cover;
      display: block;
      border-radius: 12px;
      background: #091017;
      border: 1px solid var(--border);
    }
    .canvas {
      width: 100%;
      height: 360px;
      display: block;
      border-radius: 12px;
      background: #091017;
      border: 1px solid var(--border);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .card {
      background: var(--panel2);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .value {
      font-size: 16px;
      font-weight: 600;
      word-break: break-word;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 7px;
      vertical-align: -1px;
    }
    .lg-final::before { background: var(--cyan); }
    .lg-ac::before { background: var(--green); }
    .lg-gt::before { background: var(--orange); }
    .lg-selected::before { background: var(--pink); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--panel2);
      border-radius: 10px;
      padding: 12px;
      max-height: 340px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.45;
    }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      .image-grid { grid-template-columns: 1fr; }
      .viewer-img { height: 220px; }
      .canvas { height: 320px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel top">
      <div>
        <div class="title">Chunk UDP Replay Viewer</div>
        <div class="muted">Auto-refreshing live replay status and latest generated paths</div>
      </div>
      <div class="muted" id="refresh-meta">waiting for first refresh...</div>
    </div>
    <div class="grid">
      <div class="panel">
        <div class="label" style="margin-bottom: 10px;">Current Input Images</div>
        <div class="image-grid">
          <div class="image-card">
            <div class="image-title">Left</div>
            <img id="img-left" class="viewer-img" alt="Left camera">
          </div>
          <div class="image-card">
            <div class="image-title">Front</div>
            <img id="img-front" class="viewer-img" alt="Front camera">
          </div>
          <div class="image-card">
            <div class="image-title">Right</div>
            <img id="img-right" class="viewer-img" alt="Right camera">
          </div>
          <div class="image-card">
            <div class="image-title">Front Tele</div>
            <img id="img-front-tele" class="viewer-img" alt="Front tele camera">
          </div>
        </div>
        <div class="final-banner">
          <div class="final-banner-label">Final Output</div>
          <div id="final-banner-text" class="final-banner-text">(waiting for first result)</div>
        </div>
      </div>
      <div class="stack">
        <div class="panel">
          <canvas id="traj-canvas" class="canvas"></canvas>
          <div class="legend">
            <span class="lg-final">final path</span>
          </div>
        </div>
        <div class="panel">
          <div class="stats">
            <div class="card"><div class="label">Phase</div><div class="value" id="phase">-</div></div>
            <div class="card"><div class="label">Progress</div><div class="value" id="progress">-</div></div>
            <div class="card"><div class="label">Replay Mode</div><div class="value" id="replay-mode">-</div></div>
            <div class="card"><div class="label">Skipped</div><div class="value" id="skipped-count">-</div></div>
            <div class="card"><div class="label">UDP Target</div><div class="value" id="udp-target">-</div></div>
            <div class="card"><div class="label">Viewer URL</div><div class="value" id="viewer-url">-</div></div>
            <div class="card"><div class="label">UDP Sent Count</div><div class="value" id="udp-count">-</div></div>
            <div class="card"><div class="label">Current Sample</div><div class="value" id="sample-id">-</div></div>
            <div class="card"><div class="label">Selected Path</div><div class="value" id="selected-path">-</div></div>
            <div class="card"><div class="label">Nav Mode</div><div class="value" id="nav-mode">-</div></div>
            <div class="card"><div class="label">Nav Weight</div><div class="value" id="nav-weight">-</div></div>
            <div class="card"><div class="label">Nav Source</div><div class="value" id="nav-source">-</div></div>
            <div class="card"><div class="label">Latest t0_us</div><div class="value" id="t0-us">-</div></div>
            <div class="card"><div class="label">Last Elapsed</div><div class="value" id="elapsed">-</div></div>
          </div>
          <div class="label" style="margin-bottom: 6px;">Final Output</div>
          <pre id="final-output">(waiting for first result)</pre>
          <div class="label" style="margin: 14px 0 6px;">Nav Input</div>
          <pre id="nav-input">(no nav input)</pre>
          <div class="label" style="margin: 14px 0 6px;">Effective Route Prompt</div>
          <pre id="nav-prompt">(no route span)</pre>
          <div class="label" style="margin: 14px 0 6px;">Latest Status</div>
          <pre id="status-json">{}</pre>
        </div>
      </div>
    </div>
  </div>
  <script>
    const phaseEl = document.getElementById("phase");
    const progressEl = document.getElementById("progress");
    const replayModeEl = document.getElementById("replay-mode");
    const skippedCountEl = document.getElementById("skipped-count");
    const udpTargetEl = document.getElementById("udp-target");
    const viewerUrlEl = document.getElementById("viewer-url");
    const udpCountEl = document.getElementById("udp-count");
    const sampleEl = document.getElementById("sample-id");
    const selectedEl = document.getElementById("selected-path");
    const navModeEl = document.getElementById("nav-mode");
    const navWeightEl = document.getElementById("nav-weight");
    const navSourceEl = document.getElementById("nav-source");
    const t0El = document.getElementById("t0-us");
    const elapsedEl = document.getElementById("elapsed");
    const finalBannerEl = document.getElementById("final-banner-text");
    const finalOutputEl = document.getElementById("final-output");
    const navInputEl = document.getElementById("nav-input");
    const navPromptEl = document.getElementById("nav-prompt");
    const statusEl = document.getElementById("status-json");
    const refreshMetaEl = document.getElementById("refresh-meta");
    const canvas = document.getElementById("traj-canvas");
    const imageEls = {
      left: document.getElementById("img-left"),
      front: document.getElementById("img-front"),
      right: document.getElementById("img-right"),
      front_tele: document.getElementById("img-front-tele"),
    };
    let zoomScale = 1.0;
    let zoomDirty = false;

    function setText(el, value) {
      el.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function setupCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {ctx, width: rect.width, height: rect.height};
    }

    function defaultViewExtents(path) {
      let maxForward = 8.0;
      let maxLateral = 2.5;
      if (path && Array.isArray(path.packet_points)) {
        for (const point of path.packet_points) {
          maxForward = Math.max(maxForward, Number(point.x_m || 0));
          maxLateral = Math.max(maxLateral, Math.abs(Number(point.y_m || 0)));
        }
      }
      const forward = Math.min(140.0, Math.max(10.0, maxForward * 1.08 + 1.0));
      const rear = Math.min(8.0, Math.max(1.5, forward * 0.14));
      const lateral = Math.min(30.0, Math.max(4.0, maxLateral * 1.25));
      return { forward, rear, lateral };
    }

    function project(point, originX, originY, scale) {
      return {
        x: originX - point.y_m * scale,
        y: originY - point.x_m * scale,
      };
    }

    function drawGrid(ctx, width, height, originX, originY, scale, extents) {
      ctx.fillStyle = "#091017";
      ctx.fillRect(0, 0, width, height);
      const stepMinor = 0.5;
      const stepMajor = 1.0;
      for (let lateral = -extents.lateral; lateral <= extents.lateral + 1e-6; lateral += stepMinor) {
        const isMajor = Math.abs(lateral / stepMajor - Math.round(lateral / stepMajor)) < 1e-6;
        ctx.strokeStyle = isMajor ? "rgba(255,255,255,0.13)" : "rgba(255,255,255,0.05)";
        ctx.lineWidth = isMajor ? 1.2 : 0.8;

        const xLeft = originX - lateral * scale;
        ctx.beginPath();
        ctx.moveTo(xLeft, 0);
        ctx.lineTo(xLeft, height);
        ctx.stroke();
      }

      for (let longitudinal = -extents.rear; longitudinal <= extents.forward + 1e-6; longitudinal += stepMinor) {
        const isMajor = Math.abs(longitudinal / stepMajor - Math.round(longitudinal / stepMajor)) < 1e-6;
        ctx.strokeStyle = isMajor ? "rgba(255,255,255,0.13)" : "rgba(255,255,255,0.05)";
        ctx.lineWidth = isMajor ? 1.2 : 0.8;

        const yUp = originY - longitudinal * scale;
        ctx.beginPath();
        ctx.moveTo(0, yUp);
        ctx.lineTo(width, yUp);
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(originX, 0);
      ctx.lineTo(originX, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, originY);
      ctx.lineTo(width, originY);
      ctx.stroke();
    }

    function drawPath(ctx, width, height, path, color, lineWidth, originX, originY, scale, dashed=false) {
      if (!path || !Array.isArray(path.packet_points) || path.packet_points.length === 0) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dashed ? [8, 6] : []);
      ctx.beginPath();
      path.packet_points.forEach((point, idx) => {
        const p = project(point, originX, originY, scale);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      for (let idx = 0; idx < path.packet_points.length; idx += 4) {
        const p = project(path.packet_points[idx], originX, originY, scale);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, idx === 0 ? 4.5 : 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    function drawScene(state) {
      const {ctx, width, height} = setupCanvas();
      const latest = state.latest || {};
      const finalPath = latest.final_path || null;
      const baseExtents = defaultViewExtents(finalPath);
      const extents = {
        forward: baseExtents.forward * zoomScale,
        rear: baseExtents.rear * zoomScale,
        lateral: baseExtents.lateral * zoomScale,
      };
      const pad = 26.0;
      const drawW = Math.max(width - pad * 2.0, 1.0);
      const drawH = Math.max(height - pad * 2.0, 1.0);
      const scale = Math.min(
        drawW / (2.0 * extents.lateral),
        drawH / (extents.forward + extents.rear),
      );
      const originX = width / 2;
      const originY = pad + extents.forward * scale;
      drawGrid(ctx, width, height, originX, originY, scale, extents);
      drawPath(ctx, width, height, finalPath, "#67d5ff", 2.4, originX, originY, scale, false);
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(originX, originY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.82)";
      ctx.font = "12px sans-serif";
      ctx.fillText(`front +${extents.forward.toFixed(1)} m`, 18, 24);
      ctx.fillText(`rear -${extents.rear.toFixed(1)} m`, 18, height - 12);
    }

    async function loadState() {
      const urls = [
        `/api/state?ts=${Date.now()}`,
        `./latest_view_state.json?ts=${Date.now()}`,
      ];
      let lastError = null;
      for (const url of urls) {
        try {
          const response = await fetch(url, {cache: "no-store"});
          if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
          return await response.json();
        } catch (err) {
          lastError = err;
        }
      }
      throw lastError || new Error("state fetch failed");
    }

    async function refresh() {
      try {
        const state = await loadState();
        setText(phaseEl, state.phase);
        const cur = state.current_request_index || 0;
        const total = state.total_requests || 0;
        setText(progressEl, `${cur}/${total}`);
        setText(replayModeEl, state.replay_mode || "sequential");
        setText(skippedCountEl, state.skipped_request_count || 0);
        setText(udpTargetEl, `${state.udp_host || "-"}:${state.udp_port || "-"}`);
        setText(viewerUrlEl, state.viewer_url || window.location.href);
        setText(udpCountEl, state.udp_sent_count);
        const latest = state.latest || {};
        const txRow = latest.tx_row || {};
        const nav = state.current_nav || latest.nav || {};
        const navMode = nav.nav_text ? (state.alpamayo_nav_cfg ? "nav_cfg" : "nav_text") : "none";
        setText(sampleEl, txRow.sample_id);
        setText(selectedEl, latest.chosen_source || state.udp_path_source || "-");
        setText(navModeEl, navMode);
        setText(navWeightEl, nav.nav_guidance_weight);
        setText(navSourceEl, nav.oracle_nav_source || nav.route_injection_mode);
        setText(t0El, txRow.t0_us);
        setText(elapsedEl, txRow.request_elapsed_s ? `${Number(txRow.request_elapsed_s).toFixed(2)} s` : "-");
        const finalText = (latest.ac_decoded_path && latest.ac_decoded_path.final_output) || state.message || "(waiting for first result)";
        setText(finalBannerEl, finalText);
        setText(finalOutputEl, finalText);
        setText(navInputEl, nav.nav_text || "(no nav input)");
        setText(navPromptEl, nav.effective_prompt_excerpt || nav.request_prompt_excerpt || "(no route span)");
        const stamp = Date.now();
        const imageMap = {
          left: "./latest_left.png",
          front: "./latest_front.png",
          right: "./latest_right.png",
          front_tele: "./latest_front_tele.png",
        };
        for (const [key, el] of Object.entries(imageEls)) {
          if (el) {
            el.src = `${imageMap[key]}?ts=${stamp}`;
          }
        }
        statusEl.textContent = JSON.stringify({
          phase: state.phase,
          message: state.message,
          current_request_index: state.current_request_index,
          total_requests: state.total_requests,
          replay_mode: state.replay_mode || null,
          alpamayo_nav_cfg: state.alpamayo_nav_cfg || false,
          skipped_request_count: state.skipped_request_count || 0,
          udp_sent_count: state.udp_sent_count,
          viewer_url: state.viewer_url || null,
          latest_sample_id: txRow.sample_id || null,
          latest_output_text: txRow.output_text || null,
          latest_nav_text: nav.nav_text || null,
          nav_mode: navMode,
          nav_prompt_excerpt: nav.effective_prompt_excerpt || null,
        }, null, 2);
        refreshMetaEl.textContent = `last refresh ${new Date().toLocaleTimeString()}`;
        drawScene(state);
      } catch (err) {
        refreshMetaEl.textContent = `viewer refresh failed: ${err}`;
      }
    }

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 0.88 : 1.14;
      zoomScale = Math.max(0.45, Math.min(4.0, zoomScale * factor));
      zoomDirty = true;
      refresh();
    }, {passive: false});

    canvas.addEventListener("dblclick", () => {
      zoomDirty = false;
      zoomScale = 1.0;
      refresh();
    });

    window.addEventListener("resize", () => refresh());
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def _read_artifact_file(artifact_root: Path, relative_path: str) -> tuple[bytes, str] | None:
    normalized = relative_path.lstrip("/")
    candidate = (artifact_root / normalized).resolve()
    try:
        candidate.relative_to(artifact_root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
    return candidate.read_bytes(), content_type


def _truncate_text(text: str | None, limit: int = 320) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    keep = max(32, limit - 3)
    head = keep // 2
    tail = keep - head
    return text[:head] + "..." + text[-tail:]


def _remove_route_span(text: str) -> str:
    start_token = "<|route_start|>"
    end_token = "<|route_end|>"
    start = text.find(start_token)
    if start < 0:
        return text
    end = text.find(end_token, start + len(start_token))
    if end < 0:
        return text
    return text[:start] + text[end + len(end_token) :]


def _insert_route_span(text: str, nav_text: str) -> str:
    history_end_token = "<|traj_history_end|>"
    history_end = text.find(history_end_token)
    if history_end < 0:
        return text
    route_span = f"<|route_start|>{nav_text}<|route_end|>"
    text = _remove_route_span(text)
    insert_pos = history_end + len(history_end_token)
    return text[:insert_pos] + route_span + text[insert_pos:]


def _extract_prompt_excerpt(text: str | None) -> str | None:
    if not text:
        return None
    anchor = text.find("<|traj_history_end|>")
    if anchor < 0:
        anchor = text.find("<|route_start|>")
    if anchor < 0:
        return _truncate_text(text, 320)
    start = max(0, anchor - 48)
    end = min(len(text), anchor + 240)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + "..."
    return excerpt


def extract_request_nav_payload(request_path: Path) -> dict[str, Any] | None:
    try:
        request_obj = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"request_path": str(request_path), "parse_error": str(exc)}

    requests = request_obj.get("requests") or []
    if not requests:
        return None
    req = requests[0]
    nav_text = req.get("nav_text")
    nav_guidance_weight = req.get("nav_guidance_weight")
    oracle_nav_source = req.get("oracle_nav_source")

    user_texts: list[str] = []
    for msg in req.get("messages", []):
        if msg.get("role") != "user":
            continue
        for content in msg.get("content", []):
            if content.get("type") == "text":
                user_texts.append(str(content.get("text", "")))

    prompt_text = None
    for text in user_texts:
        if "<|traj_history_end|>" in text or "<|route_start|>" in text:
            prompt_text = text
            break
    if prompt_text is None and user_texts:
        prompt_text = user_texts[-1]

    route_span = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else None
    request_has_route_span = bool(prompt_text and "<|route_start|>" in prompt_text)
    effective_prompt = _insert_route_span(prompt_text, nav_text) if (prompt_text and nav_text) else prompt_text
    if nav_text:
        route_injection_mode = "request_embedded" if request_has_route_span else "runtime_insert_after_traj_history_end"
    else:
        route_injection_mode = "none"

    return {
        "request_path": str(request_path),
        "nav_text": nav_text,
        "nav_guidance_weight": nav_guidance_weight,
        "oracle_nav_source": oracle_nav_source,
        "route_span": route_span,
        "route_injection_mode": route_injection_mode,
        "request_prompt_excerpt": _extract_prompt_excerpt(prompt_text),
        "effective_prompt_excerpt": _extract_prompt_excerpt(effective_prompt),
    }


def start_viewer_server(host: str, port: int, artifact_root: Path, state: ReplayViewerState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/viewer"):
                body = _viewer_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/state":
                payload = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            static = _read_artifact_file(artifact_root, path)
            if static is not None:
                body, content_type = static
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                if path.endswith(".json"):
                    self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404, "Not found")

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def persist_viewer_state(artifact_root: Path, state: ReplayViewerState) -> None:
    (artifact_root / "latest_view_state.json").write_text(
        json.dumps(state.snapshot(), indent=2),
        encoding="utf-8",
    )


def allocate_viewer_server(
    host: str,
    requested_port: int,
    artifact_root: Path,
    state: ReplayViewerState,
) -> tuple[ThreadingHTTPServer, int]:
    last_error: Exception | None = None
    for port in range(int(requested_port), int(requested_port) + 20):
        try:
            server = start_viewer_server(host, port, artifact_root, state)
            return server, port
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to bind viewer server near port {requested_port}: {last_error}")


def try_open_viewer_url(url: str) -> bool:
    try:
        if shutil.which("xdg-open") is not None and os.name != "nt":
            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        return bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        return False


def resolve_viewer_advertise_host(bind_host: str) -> str:
    if bind_host not in {"0.0.0.0", "::"}:
        return bind_host
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
        for token in output.split():
            if token and not token.startswith("127.") and not token.startswith("172.17.") and not token.startswith("172.18."):
                return token
    except Exception:
        pass
    return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Replay one raw live-dataset chunk through Alpamayo and send each inferred path to control via UDP."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-id", type=int, default=1)
    parser.add_argument("--work-root", type=Path, default=repo_root / "output" / "live_chunk_udp_replay")
    parser.add_argument("--request-bank-root", type=Path, default=None)
    parser.add_argument("--request-limit", type=int, default=-1)
    parser.add_argument("--request-stride", type=int, default=1)
    parser.add_argument(
        "--replay-mode",
        choices=["sequential", "latest_only"],
        default="sequential",
        help="`sequential` processes every 0.1s sample in order. "
        "`latest_only` emulates live latest-frame behavior and skips stale samples while inference is busy.",
    )
    parser.add_argument(
        "--target-offset-s",
        type=float,
        default=None,
        help="Start replay from the request nearest to this chunk-relative offset in seconds. "
        "Useful with --request-limit 1 for one-shot replay at a chosen time.",
    )
    parser.add_argument("--rebuild-request-bank", action="store_true")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--llm-inference-bin", type=Path, default=repo_root / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=repo_root / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=first_existing_fm_engine(),
    )
    parser.add_argument("--alpamayo-nav-cfg", action="store_true")
    parser.add_argument("--alpamayo-fm-use-prefill-kv", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=600.0)
    parser.add_argument("--udp-host", type=str, default="192.168.0.29")
    parser.add_argument("--udp-port", type=int, default=5001)
    parser.add_argument(
        "--udp-path-source",
        choices=["ac_decoded", "final", "gt"],
        default="ac_decoded",
        help="Which path variant to packetize and send. Default matches the old a,c-decoded control path flow.",
    )
    parser.add_argument("--viewer-host", type=str, default="127.0.0.1")
    parser.add_argument("--viewer-port", type=int, default=8780)
    parser.add_argument("--open-viewer", action="store_true")
    parser.add_argument("--disable-viewer", action="store_true")
    parser.add_argument("--skip-udp", action="store_true")
    parser.add_argument("--skip-existing-outputs", action="store_true")
    return parser.parse_args()


def build_request_bank(args: argparse.Namespace, request_bank_root: Path) -> None:
    request_bank_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BUILD_REQUEST_BANK),
        "--dataset-root",
        str(args.dataset_root),
        "--chunk-id",
        str(args.chunk_id),
        "--output-root",
        str(request_bank_root),
        "--history-len",
        str(args.history_len),
        "--dt-s",
        str(args.dt_s),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--traj-token-offset",
        str(args.traj_token_offset),
        "--diffusion-seed",
        str(args.diffusion_seed),
        "--diffusion-num-steps",
        str(args.diffusion_num_steps),
        "--max-generate-length",
        str(args.max_generate_length),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
    ]
    if args.request_limit > 0 and args.target_offset_s is None:
        cmd.extend(["--limit", str(args.request_limit)])
    if args.nav_text:
        cmd.extend(["--nav-text", args.nav_text])
    print("[chunk-replay] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_entries(request_bank_root: Path, stride: int) -> tuple[list[RequestEntry], dict[str, Any]]:
    manifest_path = request_bank_root / "manifest.json"
    summary_path = request_bank_root / "summary.json"
    ensure_exists(manifest_path, "request bank manifest")
    ensure_exists(summary_path, "request bank summary")
    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not manifest_rows:
        raise RuntimeError(f"No request entries found in {manifest_path}")

    t0_start_ns = int(min(int(row["t0_utc_ns"]) for row in manifest_rows))
    entries: list[RequestEntry] = []
    for row in sorted(manifest_rows, key=lambda item: int(item["t0_utc_ns"])):
        selected_frames = row.get("selected_frames", {})
        front_ids = selected_frames.get("front") or []
        front_frame_id = int(front_ids[-1]) if front_ids else int(row["sample_id"])
        actual_offset_s = float((int(row["t0_utc_ns"]) - t0_start_ns) / 1e9)
        entries.append(
            RequestEntry(
                request_path=Path(row["request_json"]),
                sample_id=int(row["sample_id"]),
                front_frame_id=front_frame_id,
                t0_utc_ns=int(row["t0_utc_ns"]),
                t0_us=int(row["t0_us"]),
                actual_offset_s=actual_offset_s,
                ego_history_xyz_npy=str(row["ego_history_xyz_npy"]),
                ego_history_rot_npy=str(row["ego_history_rot_npy"]),
                selected_frames={str(k): [int(x) for x in v] for k, v in selected_frames.items()},
            )
        )
    if stride > 1:
        entries = entries[::stride]
    return entries, summary


def choose_summary(
    udp_path_source: str,
    final_summary: dict[str, Any],
    ac_summary: dict[str, Any],
    gt_summary: dict[str, Any],
) -> dict[str, Any]:
    if udp_path_source == "ac_decoded":
        return ac_summary
    if udp_path_source == "gt":
        return gt_summary
    return final_summary


def select_entries_from_target_offset(entries: list[RequestEntry], target_offset_s: float | None) -> tuple[list[RequestEntry], int | None]:
    if target_offset_s is None or not entries:
        return entries, None
    target = float(target_offset_s)
    nearest_idx = min(range(len(entries)), key=lambda idx: abs(entries[idx].actual_offset_s - target))
    return entries[nearest_idx:], nearest_idx


def iter_replay_selections(entries: list[RequestEntry], replay_mode: str) -> list[ReplaySelection]:
    if replay_mode == "sequential":
        selections: list[ReplaySelection] = []
        skipped_total = 0
        for idx, entry in enumerate(entries, start=1):
            selections.append(
                ReplaySelection(
                    entry=entry,
                    processed_index=idx,
                    selected_index=idx - 1,
                    skipped_since_last=0,
                    skipped_total=skipped_total,
                    desired_offset_s=entry.actual_offset_s,
                )
            )
        return selections
    raise RuntimeError("iter_replay_selections() is only valid for sequential mode")


def iter_latest_only_replay(entries: list[RequestEntry]):
    if not entries:
        return

    processed_index = 0
    skipped_total = 0
    next_idx = 0
    replay_start_wall = time.time()
    replay_start_offset_s = float(entries[0].actual_offset_s)

    while next_idx < len(entries):
        desired_offset_s = replay_start_offset_s + (time.time() - replay_start_wall)
        if entries[next_idx].actual_offset_s > desired_offset_s:
            yield None, desired_offset_s
            continue

        selected_idx = next_idx
        while selected_idx + 1 < len(entries) and entries[selected_idx + 1].actual_offset_s <= desired_offset_s:
            selected_idx += 1

        skipped_since_last = selected_idx - next_idx
        skipped_total += skipped_since_last
        processed_index += 1
        selection = ReplaySelection(
            entry=entries[selected_idx],
            processed_index=processed_index,
            selected_index=selected_idx,
            skipped_since_last=skipped_since_last,
            skipped_total=skipped_total,
            desired_offset_s=desired_offset_s,
        )
        next_idx = selected_idx + 1
        yield selection, desired_offset_s


def _select_position_speed_rows(gnss: Any) -> Any:
    valid = gnss.dropna(subset=["timestamp_utc_ns", "lat", "lon", "alt"]).copy()
    quality_mask = None
    if "num_sats" in valid.columns:
        quality_mask = valid["num_sats"].notna()
    if "hdop" in valid.columns:
        hdop_mask = valid["hdop"].notna()
        quality_mask = hdop_mask if quality_mask is None else (quality_mask | hdop_mask)
    if quality_mask is not None and int(quality_mask.sum()) >= 2:
        valid = valid.loc[quality_mask].copy()
    sort_cols = ["timestamp_utc_ns", "row_id"] if "row_id" in valid.columns else ["timestamp_utc_ns"]
    valid = valid.sort_values(sort_cols)
    return valid.drop_duplicates(subset=["timestamp_utc_ns"], keep="last").reset_index(drop=True)


def _position_speed_lookup(gnss: Any) -> tuple[list[int], list[float], str]:
    import numpy as np

    valid = _select_position_speed_rows(gnss)
    if len(valid) < 2:
        return [], [], "unavailable"

    times = valid["timestamp_utc_ns"].astype("int64").to_numpy()
    lat = valid["lat"].astype("float64").to_numpy()
    lon = valid["lon"].astype("float64").to_numpy()
    lat0 = float(np.deg2rad(lat[0]))
    meters_per_deg_lat = 111132.92
    meters_per_deg_lon = 111412.84 * float(np.cos(lat0))
    x = (lon - lon[0]) * meters_per_deg_lon
    y = (lat - lat[0]) * meters_per_deg_lat
    dt = np.diff(times.astype("float64")) / 1e9
    dist = np.hypot(np.diff(x), np.diff(y))
    segment_speed = np.divide(dist, dt, out=np.zeros_like(dist), where=dt > 0.0)
    speeds = np.zeros(len(times), dtype=np.float64)
    speeds[1:] = segment_speed
    speeds[0] = segment_speed[0] if len(segment_speed) else 0.0
    speeds = np.clip(speeds, 0.0, 80.0)
    return times.astype("int64").tolist(), speeds.astype("float64").tolist(), "gnss_position_delta"


def load_sensor_speed_lookup(dataset_root: Path) -> tuple[list[int], list[float], str]:
    pd = load_pandas()
    gnss_path = dataset_root / "sensors" / "gnss_ins" / "gnss_ins.parquet"
    ensure_exists(gnss_path, "gnss_ins parquet")
    gnss = pd.read_parquet(gnss_path)
    valid = gnss.dropna(subset=["timestamp_utc_ns", "vx", "vy"]).copy()
    valid = valid.sort_values("timestamp_utc_ns")
    if not valid.empty:
        speeds_series = ((valid["vx"].astype("float64") ** 2 + valid["vy"].astype("float64") ** 2) ** 0.5).astype("float64")
        sane = speeds_series.notna() & (speeds_series >= 0.0) & (speeds_series <= 80.0)
        if int(sane.sum()) >= max(2, int(0.8 * len(speeds_series))):
            return valid["timestamp_utc_ns"].astype("int64").tolist(), speeds_series.tolist(), "gnss_vx_vy_nearest_t0"
    return _position_speed_lookup(gnss)


def lookup_nearest_sensor_speed(timestamps_utc_ns: list[int], speeds_mps: list[float], target_t0_utc_ns: int) -> tuple[float | None, int | None]:
    if not timestamps_utc_ns:
        return None, None
    import bisect

    idx = bisect.bisect_left(timestamps_utc_ns, int(target_t0_utc_ns))
    candidates: list[int] = []
    if idx < len(timestamps_utc_ns):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(timestamps_utc_ns[i] - int(target_t0_utc_ns)))
    return float(speeds_mps[best_idx]), int(timestamps_utc_ns[best_idx])


def build_text_udp_payload(
    *,
    summary: dict[str, Any],
    sensor_speed_mps: float | None,
    sensor_speed_t0_utc_ns: int | None,
    sensor_speed_source: str,
    inference_time_s: float,
) -> dict[str, Any]:
    payload = copy.deepcopy(summary)
    packet_points = payload.get("packet_points") or []
    initial_speed_mps = None
    if packet_points:
        initial_speed_mps = float(packet_points[0].get("v_mps", 0.0))
    payload["initial_speed_mps"] = initial_speed_mps
    payload["initial_speed_kph"] = float(initial_speed_mps * 3.6) if initial_speed_mps is not None else None
    payload["initial_speed_source"] = "packet_points[0].v_mps"
    payload["sensor_speed_mps"] = float(sensor_speed_mps) if sensor_speed_mps is not None else None
    payload["sensor_speed_kph"] = float(sensor_speed_mps * 3.6) if sensor_speed_mps is not None else None
    payload["sensor_speed_source"] = sensor_speed_source
    payload["sensor_speed_t0_utc_ns"] = int(sensor_speed_t0_utc_ns) if sensor_speed_t0_utc_ns is not None else None
    payload["inference_time_s"] = float(inference_time_s)
    return payload


def update_latest_viewer_images(artifact_root: Path, request_bank_root: Path, request_path: Path) -> None:
    sample_stem = request_path.stem
    if sample_stem.startswith("request_"):
        sample_stem = sample_stem[len("request_") :]
    sample_image_dir = request_bank_root / "images" / sample_stem
    image_map = {
        "latest_left.png": sample_image_dir / "cam0_f3.png",
        "latest_front.png": sample_image_dir / "cam1_f3.png",
        "latest_right.png": sample_image_dir / "cam2_f3.png",
        "latest_front_tele.png": sample_image_dir / "cam6_f3.png",
    }
    for dest_name, src_path in image_map.items():
        if src_path.exists():
            shutil.copy2(src_path, artifact_root / dest_name)


def main() -> None:
    args = parse_args()
    ensure_exists(args.dataset_root, "dataset root")
    for path, description in [
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engineDir"),
        (args.multimodal_engine_dir, "multimodalEngineDir"),
        (args.fm_engine, "fmEngine"),
    ]:
        ensure_exists(path, description)

    chunk_tag = f"chunk{args.chunk_id:04d}"
    request_bank_root = args.request_bank_root or (args.work_root / chunk_tag / "request_bank")
    output_root = args.work_root / chunk_tag / "outputs"
    artifact_root = args.work_root / chunk_tag / "artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    viewer_state = ReplayViewerState()
    viewer_state.update(
        phase="starting",
        message="preparing request bank and runtime",
        udp_host=args.udp_host,
        udp_port=int(args.udp_port),
        udp_path_source=args.udp_path_source,
        replay_mode=args.replay_mode,
        alpamayo_nav_cfg=bool(args.alpamayo_nav_cfg),
        chunk_id=int(args.chunk_id),
        dataset_root=str(args.dataset_root),
    )
    viewer_server: ThreadingHTTPServer | None = None
    persist_viewer_state(artifact_root, viewer_state)
    if not args.disable_viewer:
        try:
            (artifact_root / "index.html").write_text(_viewer_html(), encoding="utf-8")
            viewer_server, actual_viewer_port = allocate_viewer_server(
                args.viewer_host,
                int(args.viewer_port),
                artifact_root,
                viewer_state,
            )
            viewer_url = f"http://{resolve_viewer_advertise_host(args.viewer_host)}:{actual_viewer_port}/viewer"
            viewer_state.update(
                viewer_host=args.viewer_host,
                viewer_port=int(actual_viewer_port),
                viewer_url=viewer_url,
            )
            persist_viewer_state(artifact_root, viewer_state)
            print(f"[chunk-replay] viewer: {viewer_url}", flush=True)
            if int(actual_viewer_port) != int(args.viewer_port):
                print(
                    "[chunk-replay] viewer requested port "
                    f"{int(args.viewer_port)} was busy; using {int(actual_viewer_port)} instead. "
                    "If you are accessing from outside Docker, make sure that port is published too.",
                    flush=True,
                )
            if args.open_viewer:
                opened = try_open_viewer_url(viewer_url)
                print(f"[chunk-replay] viewer auto-open: {'requested' if opened else 'failed'}", flush=True)
        except Exception as exc:
            print(f"[chunk-replay] viewer disabled automatically: {exc}", flush=True)

    if args.rebuild_request_bank or not (request_bank_root / "summary.json").exists():
        build_request_bank(args, request_bank_root)
    else:
        print(f"[chunk-replay] reusing request bank: {request_bank_root}", flush=True)

    entries, request_bank_summary = load_entries(request_bank_root, max(1, int(args.request_stride)))
    entries, start_entry_index = select_entries_from_target_offset(entries, args.target_offset_s)
    sensor_speed_times_utc_ns, sensor_speeds_mps, sensor_speed_source = load_sensor_speed_lookup(args.dataset_root)
    history_len = int(request_bank_summary["history_len"])
    if args.request_limit > 0:
        entries = entries[: args.request_limit]
    if not entries:
        raise RuntimeError("No request entries selected after applying stride/limit")
    if args.target_offset_s is not None and start_entry_index is not None:
        first = entries[0]
        print(
            "[chunk-replay] target offset "
            f"{float(args.target_offset_s):.3f}s -> selected sample_id={first.sample_id} "
            f"actual_offset_s={first.actual_offset_s:.6f}",
            flush=True,
        )
    viewer_state.update(
        phase="request_bank_ready",
        message="request bank loaded",
        total_requests=len(entries),
        replay_mode=args.replay_mode,
        request_bank_root=str(request_bank_root),
        artifact_root=str(artifact_root),
        output_root=str(output_root),
        target_offset_s=float(args.target_offset_s) if args.target_offset_s is not None else None,
        selected_start_sample_id=int(entries[0].sample_id),
        selected_start_actual_offset_s=float(entries[0].actual_offset_s),
    )
    persist_viewer_state(artifact_root, viewer_state)

    cmd = [
        str(args.llm_inference_bin),
        "--engineDir",
        str(args.engine_dir),
        "--multimodalEngineDir",
        str(args.multimodal_engine_dir),
        "--fmEngine",
        str(args.fm_engine),
        "--alpamayoPostVlmRuntime",
        "--persistentServer",
        "--warmup",
        str(args.warmup),
    ]
    if args.alpamayo_fm_use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")
    if args.alpamayo_nav_cfg:
        cmd.append("--alpamayoNavCfg")

    print("[chunk-replay] starting persistent llm_inference", flush=True)
    print("[chunk-replay] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(args.plugin_lib),
    )

    sock: socket.socket | None = None
    target = (args.udp_host, int(args.udp_port))
    tx_log: list[dict[str, Any]] = []
    start_time = time.time()
    try:
        viewer_state.update(phase="starting_runtime", message="loading persistent llm_inference")
        ready = read_status(proc, timeout_s=120.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected ready state: {ready}")
        print("[chunk-replay] persistent llm_inference ready", flush=True)
        viewer_state.update(phase="runtime_ready", message="persistent llm_inference ready")
        persist_viewer_state(artifact_root, viewer_state)

        if not args.skip_udp:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(
                "[chunk-replay] udp text send enabled "
                f"target={args.udp_host}:{args.udp_port} "
                "mode=text_json_once_64pts",
                flush=True,
            )

        total = len(entries)
        if args.replay_mode == "latest_only":
            selection_iter = iter_latest_only_replay(entries)
        else:
            selection_iter = ((selection, selection.desired_offset_s) for selection in iter_replay_selections(entries, args.replay_mode))

        for maybe_selection, desired_offset_s in selection_iter:
            if maybe_selection is None:
                viewer_state.update(
                    phase="waiting_for_replay_time",
                    message="waiting for next chunk timestamp",
                    current_request_index=len(tx_log),
                    total_requests=total,
                    skipped_request_count=tx_log[-1]["skipped_total"] if tx_log else 0,
                    desired_offset_s=float(desired_offset_s) if desired_offset_s is not None else None,
                )
                persist_viewer_state(artifact_root, viewer_state)
                time.sleep(0.02)
                continue

            selection = maybe_selection
            idx = selection.processed_index
            entry = selection.entry
            nav_payload = extract_request_nav_payload(entry.request_path)
            viewer_state.update(
                phase="processing",
                message=f"processing request {idx}/{total}",
                current_request_index=idx,
                total_requests=total,
                skipped_request_count=selection.skipped_total,
                current_sample_id=entry.sample_id,
                current_t0_us=entry.t0_us,
                current_request_json=str(entry.request_path),
                current_nav=nav_payload,
                desired_offset_s=float(selection.desired_offset_s) if selection.desired_offset_s is not None else None,
            )
            persist_viewer_state(artifact_root, viewer_state)
            output_name = entry.request_path.name.replace("request_", "output_")
            output_path = output_root / output_name
            if args.skip_existing_outputs and output_path.exists():
                print(f"[chunk-replay] skip existing output {output_path.name}", flush=True)
                continue

            payload = {"input_file": str(entry.request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            request_t0 = time.time()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=args.timeout_per_request)
            if status.get("status") != "ok":
                raise RuntimeError(f"Request failed for {entry.request_path.name}: {status}")
            inference_time_s = time.time() - request_t0

            normalized_metadata = {
                "chunk_id": args.chunk_id,
                "sample_id": entry.sample_id,
                "front_frame_id": entry.front_frame_id,
                "t0_utc_ns": entry.t0_utc_ns,
                "t0_us": entry.t0_us,
                "target_offset_s": entry.actual_offset_s,
                "actual_offset_s": entry.actual_offset_s,
                "request_json": str(entry.request_path),
                "ego_history_xyz_npy": entry.ego_history_xyz_npy,
                "ego_history_rot_npy": entry.ego_history_rot_npy,
                "selected_frames": entry.selected_frames,
            }
            sample_artifact_root = artifact_root / entry.request_path.stem
            sample_artifact_root.mkdir(parents=True, exist_ok=True)
            final_summary, ac_summary, gt_summary, _ = build_result_artifacts(
                output_path=output_path,
                metadata=normalized_metadata,
                dataset_root=args.dataset_root,
                history_len=history_len,
                artifact_root=sample_artifact_root,
            )
            chosen_summary = choose_summary(args.udp_path_source, final_summary, ac_summary, gt_summary)
            sensor_speed_mps, sensor_speed_t0_utc_ns = lookup_nearest_sensor_speed(
                sensor_speed_times_utc_ns,
                sensor_speeds_mps,
                entry.t0_utc_ns,
            )
            udp_payload = build_text_udp_payload(
                summary=chosen_summary,
                sensor_speed_mps=sensor_speed_mps,
                sensor_speed_t0_utc_ns=sensor_speed_t0_utc_ns,
                sensor_speed_source=sensor_speed_source,
                inference_time_s=inference_time_s,
            )
            udp_payload_bytes = json.dumps(udp_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

            udp_sent = False
            udp_mode = "disabled"
            if sock is not None:
                sock.sendto(udp_payload_bytes, target)
                udp_sent = True
                udp_mode = "text_json_once"

            elapsed = time.time() - request_t0
            total_elapsed = time.time() - start_time
            tx_row = {
                "request_index": idx,
                "request_count": total,
                "selected_entry_index": int(selection.selected_index),
                "sample_id": entry.sample_id,
                "t0_us": entry.t0_us,
                "request_json": str(entry.request_path),
                "output_json": str(output_path),
                "artifact_root": str(sample_artifact_root),
                "replay_mode": args.replay_mode,
                "desired_offset_s": float(selection.desired_offset_s) if selection.desired_offset_s is not None else None,
                "actual_offset_s": float(entry.actual_offset_s),
                "skipped_since_last": int(selection.skipped_since_last),
                "skipped_total": int(selection.skipped_total),
                "udp_path_source": args.udp_path_source,
                "udp_host": args.udp_host,
                "udp_port": int(args.udp_port),
                "udp_sent": udp_sent,
                "udp_mode": udp_mode,
                "udp_num_points": int(len(udp_payload.get("packet_points") or [])),
                "initial_speed_mps": udp_payload.get("initial_speed_mps"),
                "sensor_speed_mps": udp_payload.get("sensor_speed_mps"),
                "output_text": chosen_summary.get("final_output"),
                "nav_text": nav_payload.get("nav_text") if nav_payload else None,
                "nav_guidance_weight": nav_payload.get("nav_guidance_weight") if nav_payload else None,
                "nav_source": nav_payload.get("oracle_nav_source") if nav_payload else None,
                "traj_points_with_origin": int(chosen_summary.get("traj_points_with_origin", 0)),
                "inference_time_s": inference_time_s,
                "request_elapsed_s": elapsed,
                "total_elapsed_s": total_elapsed,
            }
            tx_log.append(tx_row)
            update_latest_viewer_images(artifact_root, request_bank_root, entry.request_path)
            (artifact_root / "latest_final_path.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
            (artifact_root / "latest_ac_decoded_path.json").write_text(json.dumps(ac_summary, indent=2), encoding="utf-8")
            (artifact_root / "latest_gt_path.json").write_text(json.dumps(gt_summary, indent=2), encoding="utf-8")
            (artifact_root / "latest_tx_row.json").write_text(json.dumps(tx_row, indent=2), encoding="utf-8")
            (sample_artifact_root / "udp_payload.json").write_text(json.dumps(udp_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (artifact_root / "latest_udp_payload.json").write_text(json.dumps(udp_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            viewer_state.update(
                phase="sent" if udp_sent else "completed_sample",
                message=f"processed sample {entry.sample_id}",
                udp_sent_count=sum(1 for row in tx_log if row.get("udp_sent")),
                skipped_request_count=selection.skipped_total,
            )
            viewer_state.set_latest(
                final_summary=final_summary,
                ac_summary=ac_summary,
                gt_summary=gt_summary,
                chosen_source=args.udp_path_source,
                tx_row=tx_row,
                nav_payload=nav_payload,
            )
            persist_viewer_state(artifact_root, viewer_state)
            print(
                f"[chunk-replay] {idx}/{total} sample_id={entry.sample_id} "
                f"path={args.udp_path_source} udp_sent={udp_sent} "
                f"skipped={selection.skipped_since_last} elapsed={elapsed:.2f}s",
                flush=True,
            )
    finally:
        if sock is not None:
            sock.close()
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                proc.stdin.flush()
            read_status(proc, timeout_s=10.0)
        except Exception:
            pass
        try:
            proc.wait(timeout=20.0)
        except Exception:
            proc.kill()
        if viewer_server is not None:
            viewer_server.shutdown()
            viewer_server.server_close()

    summary = {
        "dataset_root": str(args.dataset_root),
        "chunk_id": int(args.chunk_id),
        "replay_mode": args.replay_mode,
        "request_bank_root": str(request_bank_root),
        "output_root": str(output_root),
        "artifact_root": str(artifact_root),
        "selected_request_count": len(entries),
        "processed_request_count": len(tx_log),
        "skipped_request_count": int(tx_log[-1]["skipped_total"]) if tx_log else 0,
        "udp_host": args.udp_host,
        "udp_port": int(args.udp_port),
        "udp_path_source": args.udp_path_source,
        "udp_enabled": bool(not args.skip_udp),
        "alpamayo_nav_cfg": bool(args.alpamayo_nav_cfg),
        "fm_engine": str(args.fm_engine),
        "tx_log_path": str(artifact_root / "tx_log.json"),
    }
    (artifact_root / "tx_log.json").write_text(json.dumps(tx_log, indent=2), encoding="utf-8")
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    viewer_state.update(phase="finished", message="chunk replay completed")
    persist_viewer_state(artifact_root, viewer_state)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
