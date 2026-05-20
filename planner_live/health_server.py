from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import urlparse


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Planner Live Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1720;
      --panel: #16212b;
      --panel-alt: #1c2a36;
      --text: #e9f1f6;
      --muted: #98adbe;
      --accent: #4cc9f0;
      --good: #61d095;
      --warn: #ffb454;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #101820, #0b1117);
      color: var(--text);
    }
    .wrap {
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .topbar, .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px 16px;
    }
    .topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px 18px;
    }
    .title {
      font-size: 22px;
      font-weight: 700;
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(560px, 1.55fr) minmax(400px, 1fr);
      gap: 16px;
    }
    .viewer-img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      background: #0b1117;
      min-height: 280px;
    }
    .route-canvas {
      width: 100%;
      height: 520px;
      display: block;
      border-radius: 12px;
      background: #0b1117;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    .value {
      font-size: 16px;
      font-weight: 600;
      word-break: break-word;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d7e6f3;
      background: var(--panel-alt);
      border-radius: 10px;
      padding: 12px;
      max-height: 300px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.4;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .route-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .status-good { color: var(--good); }
    .status-warn { color: var(--warn); }
    @media (max-width: 1080px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .route-meta { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <div class="title">Planner Live Viewer</div>
        <div class="meta">Auto-refreshing dashboard + latest parsed result</div>
      </div>
      <div class="meta" id="refresh-meta">waiting for first refresh...</div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="row" style="margin-bottom: 12px;">
          <div>
            <div class="label">Dashboard Status</div>
            <div class="value" id="dashboard-status">waiting for image...</div>
          </div>
          <div>
            <div class="label">Service</div>
            <div class="value" id="service-running">-</div>
          </div>
          <div>
            <div class="label">Inference Busy</div>
            <div class="value" id="inference-busy">-</div>
          </div>
          <div>
            <div class="label">Last Success Sequence</div>
            <div class="value" id="last-seq">-</div>
          </div>
        </div>
        <img id="dashboard-img" class="viewer-img" alt="Latest planner dashboard">
      </div>

      <div class="panel stack">
        <div class="row">
          <div>
            <div class="label">Route Status</div>
            <div class="value" id="route-status">waiting for route...</div>
          </div>
          <div>
            <div class="label">Planner Result</div>
            <div class="value" id="result-status">waiting for result...</div>
          </div>
        </div>

        <canvas id="route-canvas" class="route-canvas"></canvas>

        <div class="route-meta">
          <div>
            <div class="label">Clip ID</div>
            <div class="value" id="clip-id">-</div>
          </div>
          <div>
            <div class="label">Last Success t0_us</div>
            <div class="value" id="last-t0">-</div>
          </div>
          <div>
            <div class="label">Plan dt</div>
            <div class="value" id="plan-dt">-</div>
          </div>
          <div>
            <div class="label">Plan Points</div>
            <div class="value" id="plan-points">-</div>
          </div>
        </div>

        <div>
          <div class="label">Final Output</div>
          <pre id="cot-text">(waiting for planner output)</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dashboardImg = document.getElementById("dashboard-img");
    const dashboardStatus = document.getElementById("dashboard-status");
    const resultStatus = document.getElementById("result-status");
    const routeStatus = document.getElementById("route-status");
    const refreshMeta = document.getElementById("refresh-meta");
    const serviceRunning = document.getElementById("service-running");
    const inferenceBusy = document.getElementById("inference-busy");
    const lastSeq = document.getElementById("last-seq");
    const lastT0 = document.getElementById("last-t0");
    const clipId = document.getElementById("clip-id");
    const planDt = document.getElementById("plan-dt");
    const planPoints = document.getElementById("plan-points");
    const cotText = document.getElementById("cot-text");
    const routeCanvas = document.getElementById("route-canvas");
    let latestTrajectory = null;
    let routeHalfExtentM = 6.0;
    let routeZoomDirty = false;

    async function fetchJson(url) {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return await response.json();
    }

    function setText(idEl, value) {
      idEl.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function extractCot(text) {
      if (!text) {
        return "";
      }
      const marker = "<|cot_end|>";
      const idx = String(text).indexOf(marker);
      if (idx < 0) {
        return "";
      }
      return String(text).slice(0, idx).trim();
    }

    function extractDecision(text) {
      if (!text) {
        return "";
      }
      const marker = "<|cot_end|>";
      const idx = String(text).indexOf(marker);
      if (idx < 0) {
        return String(text).trim();
      }
      return String(text).slice(idx + marker.length).trim();
    }

    function setupCanvas() {
      const rect = routeCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (routeCanvas.width !== width || routeCanvas.height !== height) {
        routeCanvas.width = width;
        routeCanvas.height = height;
      }
      const ctx = routeCanvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }

    function isMultiple(value, step) {
      return Math.abs(value / step - Math.round(value / step)) < 1e-6;
    }

    function defaultHalfExtentForTrajectory(trajectory) {
      if (!trajectory || !Array.isArray(trajectory.pred_xyz) || trajectory.pred_xyz.length === 0) {
        return 6.0;
      }
      let maxAbs = 0.0;
      for (const point of trajectory.pred_xyz) {
        const x = Number(point[0]) || 0.0;
        const y = Number(point[1]) || 0.0;
        maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
      }
      return Math.max(Math.ceil((maxAbs + 0.8) * 2.0) / 2.0, 3.0);
    }

    function drawRoute(trajectory) {
      const { ctx, width, height } = setupCanvas();
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0b1117";
      ctx.fillRect(0, 0, width, height);

      if (!trajectory || !Array.isArray(trajectory.pred_xyz) || trajectory.pred_xyz.length === 0) {
        ctx.fillStyle = "#98adbe";
        ctx.font = "16px sans-serif";
        ctx.fillText("No trajectory available yet", 20, 36);
        return;
      }

      const path = [[0.0, 0.0], ...trajectory.pred_xyz.map((point) => [Number(point[1]) || 0.0, Number(point[0]) || 0.0])];
      const halfExtent = Math.max(routeHalfExtentM, 0.5);
      const minDisplayX = -halfExtent;
      const maxDisplayX = halfExtent;
      const minDisplayY = -halfExtent;
      const maxDisplayY = halfExtent;
      const pad = 28.0;
      const drawW = Math.max(width - pad * 2.0, 1.0);
      const drawH = Math.max(height - pad * 2.0, 1.0);

      function toCanvas(point) {
        const x = pad + ((point[0] - minDisplayX) / (maxDisplayX - minDisplayX)) * drawW;
        const y = pad + ((maxDisplayY - point[1]) / (maxDisplayY - minDisplayY)) * drawH;
        return [x, y];
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(pad, pad, drawW, drawH);
      ctx.clip();

      const minorStep = 0.1;
      const majorStep = 1.0;
      for (let gx = Math.floor(minDisplayX / minorStep) * minorStep; gx <= maxDisplayX + 1e-9; gx += minorStep) {
        const [cx] = toCanvas([gx, 0.0]);
        const major = isMultiple(gx, majorStep);
        ctx.strokeStyle = Math.abs(gx) < 1e-6 ? "#5cc8ff" : major ? "rgba(128,145,167,0.32)" : "rgba(195,207,221,0.13)";
        ctx.lineWidth = Math.abs(gx) < 1e-6 ? 1.4 : major ? 0.95 : 0.45;
        ctx.beginPath();
        ctx.moveTo(cx, pad);
        ctx.lineTo(cx, pad + drawH);
        ctx.stroke();
      }

      for (let gy = Math.floor(minDisplayY / minorStep) * minorStep; gy <= maxDisplayY + 1e-9; gy += minorStep) {
        const [, cy] = toCanvas([0.0, gy]);
        const major = isMultiple(gy, majorStep);
        ctx.strokeStyle = Math.abs(gy) < 1e-6 ? "#5cc8ff" : major ? "rgba(128,145,167,0.32)" : "rgba(195,207,221,0.13)";
        ctx.lineWidth = Math.abs(gy) < 1e-6 ? 1.4 : major ? 0.95 : 0.45;
        ctx.beginPath();
        ctx.moveTo(pad, cy);
        ctx.lineTo(pad + drawW, cy);
        ctx.stroke();
      }

      ctx.strokeStyle = "#33d17a";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      path.forEach((point, idx) => {
        const [cx, cy] = toCanvas(point);
        if (idx === 0) {
          ctx.moveTo(cx, cy);
        } else {
          ctx.lineTo(cx, cy);
        }
      });
      ctx.stroke();

      ctx.fillStyle = "#33d17a";
      for (let idx = 1; idx < path.length; idx += 4) {
        const [cx, cy] = toCanvas(path[idx]);
        ctx.beginPath();
        ctx.arc(cx, cy, 2.8, 0, Math.PI * 2.0);
        ctx.fill();
      }

      const [originX, originY] = toCanvas([0.0, 0.0]);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(originX - 7.0, originY);
      ctx.lineTo(originX + 7.0, originY);
      ctx.moveTo(originX, originY - 7.0);
      ctx.lineTo(originX, originY + 7.0);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = "#e9f1f6";
      ctx.font = "600 14px sans-serif";
      ctx.fillText("Top-Down Local Route (origin-centered, wheel zoom)", 18, 22);
      ctx.fillStyle = "#98adbe";
      ctx.font = "12px sans-serif";
      ctx.fillText(`y: ${minDisplayX.toFixed(1)} to ${maxDisplayX.toFixed(1)} m`, 18, height - 14);
      ctx.fillText(`x: ${minDisplayY.toFixed(1)} to ${maxDisplayY.toFixed(1)} m`, width - 160, height - 14);
      ctx.fillText(`scale: +/-${halfExtent.toFixed(1)} m`, width - 160, 22);
    }

    async function refreshView() {
      const stamp = Date.now();
      refreshMeta.textContent = `last refresh ${new Date(stamp).toLocaleTimeString()}`;

      try {
        const status = await fetchJson(`/status?ts=${stamp}`);
        setText(serviceRunning, status.service_running);
        setText(inferenceBusy, status.inference_busy);
        setText(lastSeq, status.last_success_sequence);
        setText(lastT0, status.last_success_t0_us);
      } catch (err) {
        setText(serviceRunning, `status fetch failed: ${err}`);
      }

      try {
        const result = await fetchJson(`/artifacts/latest_result.json?ts=${stamp}`);
        setText(clipId, result.clip_id);
        setText(planDt, result.plan_dt_s === null || result.plan_dt_s === undefined ? "-" : `${Number(result.plan_dt_s).toFixed(2)} s`);
        cotText.textContent = extractDecision(result.output_text) || extractCot(result.output_text) || "(no final output captured)";
        resultStatus.textContent = "latest result loaded";
        resultStatus.className = "value status-good";
      } catch (err) {
        resultStatus.textContent = `latest result unavailable: ${err}`;
        resultStatus.className = "value status-warn";
        cotText.textContent = "(latest result unavailable)";
      }

      try {
        latestTrajectory = await fetchJson(`/artifacts/latest_trajectory.json?ts=${stamp}`);
        if (!routeZoomDirty) {
          routeHalfExtentM = defaultHalfExtentForTrajectory(latestTrajectory);
        }
        setText(planPoints, Array.isArray(latestTrajectory.pred_xyz) ? latestTrajectory.pred_xyz.length : "-");
        drawRoute(latestTrajectory);
        routeStatus.textContent = "latest route loaded";
        routeStatus.className = "value status-good";
      } catch (err) {
        latestTrajectory = null;
        setText(planPoints, "-");
        drawRoute(null);
        routeStatus.textContent = `latest route unavailable: ${err}`;
        routeStatus.className = "value status-warn";
      }

      dashboardImg.src = `/artifacts/latest_dashboard.png?ts=${stamp}`;
    }

    dashboardImg.addEventListener("load", () => {
      dashboardStatus.textContent = "latest dashboard loaded";
      dashboardStatus.className = "value status-good";
    });

    dashboardImg.addEventListener("error", () => {
      dashboardStatus.textContent = "latest dashboard unavailable";
      dashboardStatus.className = "value status-warn";
    });

    refreshView();
    setInterval(refreshView, 1000);
    window.addEventListener("resize", () => drawRoute(latestTrajectory));
    routeCanvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const direction = event.deltaY > 0 ? 1.12 : 1.0 / 1.12;
      routeHalfExtentM = Math.min(Math.max(routeHalfExtentM * direction, 0.5), 200.0);
      routeZoomDirty = true;
      drawRoute(latestTrajectory);
    }, { passive: false });
    routeCanvas.addEventListener("dblclick", () => {
      routeZoomDirty = false;
      routeHalfExtentM = defaultHalfExtentForTrajectory(latestTrajectory);
      drawRoute(latestTrajectory);
    });
  </script>
</body>
</html>
"""


def _manual_viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Planner Manual Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1720;
      --panel: #16212b;
      --text: #e9f1f6;
      --muted: #98adbe;
      --accent: #4cc9f0;
      --good: #61d095;
      --warn: #ffb454;
      --bad: #ff7b72;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #101820, #0b1117);
      color: var(--text);
    }
    .wrap {
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px 16px;
    }
    .topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px 18px;
    }
    .title {
      font-size: 22px;
      font-weight: 700;
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    .value {
      font-size: 16px;
      font-weight: 600;
      word-break: break-word;
    }
    .status-good { color: var(--good); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--bad); }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 12px;
      align-items: center;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 10px;
      padding: 12px 16px;
      background: linear-gradient(135deg, #2d8cff, #4cc9f0);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .viewer-frame {
      width: 100%;
      min-height: 1180px;
      border: 0;
      border-radius: 12px;
      background: #0b1117;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 10px;
      background: rgba(255,255,255,0.04);
      padding: 12px;
      color: #d7e6f3;
      font-size: 13px;
      line-height: 1.45;
    }
    @media (max-width: 1080px) {
      .row {
        grid-template-columns: 1fr 1fr;
      }
    }
    @media (max-width: 720px) {
      .row {
        grid-template-columns: 1fr;
      }
      .viewer-frame {
        min-height: 900px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel topbar">
      <div>
        <div class="title">Planner Manual Viewer</div>
        <div class="meta">Click once to fetch the latest sample, run one inference, visualize it, and send UDP once.</div>
      </div>
      <div class="meta" id="refresh-meta">waiting for first refresh...</div>
    </div>

    <div class="panel controls">
      <div class="actions">
        <button id="run-once-button">Generate Route Once</button>
        <div class="value" id="manual-status">idle</div>
      </div>

      <div class="row">
        <div>
          <div class="label">Service</div>
          <div class="value" id="service-running">-</div>
        </div>
        <div>
          <div class="label">Manual Busy</div>
          <div class="value" id="manual-busy">-</div>
        </div>
        <div>
          <div class="label">Manual Count</div>
          <div class="value" id="manual-count">-</div>
        </div>
        <div>
          <div class="label">Last Success Sequence</div>
          <div class="value" id="last-seq">-</div>
        </div>
      </div>

      <pre id="manual-response">(waiting for manual trigger)</pre>
    </div>

    <div class="panel">
      <iframe id="viewer-frame" class="viewer-frame" src="/viewer"></iframe>
    </div>
  </div>

  <script>
    const runOnceButton = document.getElementById("run-once-button");
    const manualStatus = document.getElementById("manual-status");
    const manualResponse = document.getElementById("manual-response");
    const serviceRunning = document.getElementById("service-running");
    const manualBusy = document.getElementById("manual-busy");
    const manualCount = document.getElementById("manual-count");
    const lastSeq = document.getElementById("last-seq");
    const refreshMeta = document.getElementById("refresh-meta");
    const viewerFrame = document.getElementById("viewer-frame");

    async function fetchJson(url, options = undefined) {
      const response = await fetch(url, { cache: "no-store", ...options });
      const text = await response.text();
      let payload = {};
      try {
        payload = text ? JSON.parse(text) : {};
      } catch (_) {
        payload = { raw_text: text };
      }
      if (!response.ok) {
        const message = payload.error || payload.raw_text || `${response.status} ${response.statusText}`;
        throw new Error(message);
      }
      return payload;
    }

    function setText(el, value) {
      el.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    }

    async function refreshStatus() {
      const stamp = Date.now();
      refreshMeta.textContent = `last refresh ${new Date(stamp).toLocaleTimeString()}`;
      try {
        const status = await fetchJson(`/status?ts=${stamp}`);
        setText(serviceRunning, status.service_running);
        setText(manualBusy, status.manual_run_busy);
        setText(manualCount, status.manual_run_count);
        setText(lastSeq, status.last_success_sequence);
        if (status.manual_run_busy) {
          manualStatus.textContent = "running";
          manualStatus.className = "value status-warn";
          runOnceButton.disabled = true;
        } else if (status.manual_last_status === "completed") {
          manualStatus.textContent = "completed";
          manualStatus.className = "value status-good";
          runOnceButton.disabled = false;
        } else if (status.manual_last_status === "failed") {
          manualStatus.textContent = "failed";
          manualStatus.className = "value status-bad";
          runOnceButton.disabled = false;
        } else {
          manualStatus.textContent = "idle";
          manualStatus.className = "value";
          runOnceButton.disabled = false;
        }
      } catch (err) {
        manualStatus.textContent = `status fetch failed: ${err}`;
        manualStatus.className = "value status-bad";
      }
    }

    async function runManualOnce() {
      runOnceButton.disabled = true;
      manualStatus.textContent = "running";
      manualStatus.className = "value status-warn";
      manualResponse.textContent = "Running one-shot planner inference...";
      try {
        const result = await fetchJson("/actions/manual-run-once", { method: "POST" });
        manualResponse.textContent = JSON.stringify(result, null, 2);
        if (result.ok) {
          manualStatus.textContent = "completed";
          manualStatus.className = "value status-good";
          viewerFrame.src = `/viewer?ts=${Date.now()}`;
        } else {
          manualStatus.textContent = "failed";
          manualStatus.className = "value status-bad";
        }
      } catch (err) {
        manualResponse.textContent = String(err);
        manualStatus.textContent = "failed";
        manualStatus.className = "value status-bad";
      } finally {
        runOnceButton.disabled = false;
        refreshStatus();
      }
    }

    runOnceButton.addEventListener("click", runManualOnce);
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    return "application/octet-stream"


def _make_handler(
    snapshot_provider: Callable[[], dict[str, Any]],
    artifact_provider: Callable[[], dict[str, Path | None]] | None,
    manual_run_once: Callable[[], dict[str, Any]] | None,
) -> type[BaseHTTPRequestHandler]:
    class PlannerHealthHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, body: bytes, *, status_code: int, content_type: str) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/viewer"}:
                self._send_bytes(
                    _viewer_html().encode("utf-8"),
                    status_code=200,
                    content_type="text/html; charset=utf-8",
                )
                return
            if path in {"/viewer-manual", "/viewer_manual"}:
                self._send_bytes(
                    _manual_viewer_html().encode("utf-8"),
                    status_code=200,
                    content_type="text/html; charset=utf-8",
                )
                return

            if path in {
                "/artifacts/latest_dashboard.png",
                "/artifacts/latest_result.json",
                "/artifacts/latest_trajectory.json",
            }:
                if artifact_provider is None:
                    self.send_error(404, "Artifacts Not Available")
                    return
                artifacts = artifact_provider()
                key = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if key == "latest_dashboard":
                    artifact_path = artifacts.get("latest_dashboard")
                elif key == "latest_result":
                    artifact_path = artifacts.get("latest_result")
                else:
                    artifact_path = artifacts.get("latest_trajectory")
                if artifact_path is None or not artifact_path.exists():
                    self.send_error(404, "Artifact Not Found")
                    return
                self._send_bytes(
                    artifact_path.read_bytes(),
                    status_code=200,
                    content_type=_content_type_for_path(artifact_path),
                )
                return

            if path not in {"/healthz", "/status"}:
                self.send_error(404, "Not Found")
                return

            snapshot = snapshot_provider()
            status_code = 200 if snapshot.get("service_running", False) else 503
            if path == "/healthz":
                payload = {
                    "service_running": snapshot.get("service_running", False),
                    "runtime_alive": snapshot.get("runtime_alive", False),
                    "inference_busy": snapshot.get("inference_busy", False),
                    "last_received_sequence": snapshot.get("last_received_sequence"),
                    "last_success_sequence": snapshot.get("last_success_sequence"),
                }
            else:
                payload = snapshot

            self._send_bytes(
                json.dumps(payload, indent=2).encode("utf-8"),
                status_code=status_code,
                content_type="application/json; charset=utf-8",
            )

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/actions/manual-run-once":
                self.send_error(404, "Not Found")
                return
            if manual_run_once is None:
                self.send_error(404, "Manual Trigger Not Available")
                return
            payload = manual_run_once()
            status_code = 200 if bool(payload.get("ok")) else 503
            self._send_bytes(
                json.dumps(payload, indent=2).encode("utf-8"),
                status_code=status_code,
                content_type="application/json; charset=utf-8",
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return PlannerHealthHandler


class HealthServer:
    def __init__(
        self,
        host: str,
        port: int,
        snapshot_provider: Callable[[], dict[str, Any]],
        artifact_provider: Callable[[], dict[str, Path | None]] | None = None,
        manual_run_once: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._server = ThreadingHTTPServer((host, port), _make_handler(snapshot_provider, artifact_provider, manual_run_once))
        self._thread = threading.Thread(target=self._server.serve_forever, name="planner-health-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])
