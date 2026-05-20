#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.parse
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_live_chunk_udp_replay import (  # noqa: E402
    ReplayViewerState,
    persist_viewer_state,
    resolve_viewer_advertise_host,
    update_latest_viewer_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved chunk UDP payloads from a browser-controlled viewer.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chunks", type=str, default="0,1,2,3")
    parser.add_argument("--viewer-dir", type=Path, default=None)
    parser.add_argument("--viewer-host", type=str, default="0.0.0.0")
    parser.add_argument("--viewer-port", type=int, default=8780)
    parser.add_argument("--udp-host", type=str, required=True)
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--send-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--udp-payload-mode",
        choices=["path", "action_ac", "control_schema"],
        default="path",
        help="Send saved path JSON, compact raw accel/curvature actions, or control-team schema.",
    )
    return parser.parse_args()


def parse_chunks(value: str) -> list[int]:
    chunks: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chunks.extend(range(int(start), int(end) + 1))
        else:
            chunks.append(int(token))
    return chunks


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_items(run_root: Path, chunks: list[int]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for chunk_id in chunks:
        chunk_root = run_root / f"chunk{chunk_id:04d}"
        artifact_root = chunk_root / "artifacts"
        if not (artifact_root / "tx_log.json").exists():
            artifact_root = chunk_root / "precompute_udp_replay" / "artifacts"
        tx_log_path = artifact_root / "tx_log.json"
        rows = json.loads(tx_log_path.read_text(encoding="utf-8"))
        request_bank_root = chunk_root / "request_bank"
        for row in rows:
            sample_artifact = Path(row["artifact_root"])
            payload_path = sample_artifact / "udp_payload.json"
            items.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_root": str(chunk_root),
                    "request_bank_root": str(request_bank_root),
                    "sample_artifact": str(sample_artifact),
                    "payload_path": str(payload_path),
                    "request_path": row.get("request_json"),
                    "tx_row": row,
                }
            )
    return items


def path_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_action_ac_payload(path_payload: dict[str, Any]) -> dict[str, Any]:
    raw_action = path_payload.get("raw_action") or {}
    accel = list(raw_action.get("accel_mps2") or [])
    curvature = list(raw_action.get("curvature") or [])
    if not accel or not curvature:
        raise RuntimeError("Saved payload does not contain raw_action.accel_mps2 and raw_action.curvature")
    count = min(len(accel), len(curvature))
    accel = accel[:count]
    curvature = curvature[:count]
    constants = raw_action.get("action_space_constants") or {}
    dt_s = float(constants.get("dt_value", path_payload.get("plan_dt_s", 0.1)))
    return {
        "label": "raw_action",
        "payload_format": "raw_action_ac_json",
        "control_mode": "raw_action_ac",
        "path_type": "raw_action_ac",
        "chunk_id": path_payload.get("chunk_id"),
        "front_frame_id": path_payload.get("front_frame_id"),
        "t0_utc_ns": path_payload.get("t0_utc_ns"),
        "t0_us": path_payload.get("t0_us"),
        "target_offset_s": path_payload.get("target_offset_s"),
        "actual_offset_s": path_payload.get("actual_offset_s"),
        "source_chunk_id": path_payload.get("source_chunk_id"),
        "source_sample_id": path_payload.get("source_sample_id"),
        "source_actual_offset_s": path_payload.get("source_actual_offset_s"),
        "source_t0_us": path_payload.get("source_t0_us"),
        "source_t0_utc_ns": path_payload.get("source_t0_utc_ns"),
        "final_output": path_payload.get("final_output"),
        "dt_s": dt_s,
        "action_dt_s": dt_s,
        "num_points": int(count),
        "accel_mps2": [float(v) for v in accel],
        "curvature": [float(v) for v in curvature],
        "action_points": [
            {"t_s": float((idx + 1) * dt_s), "a_mps2": float(accel[idx]), "curvature": float(curvature[idx])}
            for idx in range(count)
        ],
        "initial_speed_mps": path_payload.get("initial_speed_mps"),
        "initial_speed_kph": path_payload.get("initial_speed_kph"),
        "initial_speed_source": path_payload.get("initial_speed_source"),
        "sensor_speed_mps": path_payload.get("sensor_speed_mps"),
        "sensor_speed_kph": path_payload.get("sensor_speed_kph"),
        "sensor_speed_source": path_payload.get("sensor_speed_source"),
        "sensor_speed_t0_utc_ns": path_payload.get("sensor_speed_t0_utc_ns"),
        "raw_action": copy.deepcopy(raw_action),
    }


def build_control_schema_payload(path_payload: dict[str, Any]) -> dict[str, Any]:
    raw_action = copy.deepcopy(path_payload.get("raw_action") or {})
    accel = list(raw_action.get("accel_mps2") or [])
    curvature = list(raw_action.get("curvature") or [])
    pred_xyz = list(path_payload.get("pred_xyz") or [])
    pred_yaw_rad = list(path_payload.get("pred_yaw_rad") or [])
    pred_v_mps = list(path_payload.get("pred_v_mps") or [])
    count = min(len(accel), len(curvature), len(pred_xyz), len(pred_yaw_rad), len(pred_v_mps))
    if count < 2:
        raise RuntimeError("Control schema requires matching time series with N >= 2")
    raw_action["accel_mps2"] = [float(v) for v in accel[:count]]
    raw_action["curvature"] = [float(v) for v in curvature[:count]]
    raw_action["num_points"] = int(count)
    if "normalized_x_final" in raw_action:
        raw_action["normalized_x_final"] = raw_action["normalized_x_final"][:count]
    return {
        "raw_action": raw_action,
        "pred_xyz": pred_xyz[:count],
        "pred_yaw_rad": [float(v) for v in pred_yaw_rad[:count]],
        "pred_v_mps": [float(v) for v in pred_v_mps[:count]],
        "plan_dt_s": float(path_payload.get("plan_dt_s", 0.1)),
        "inference_time_s": float(path_payload.get("inference_time_s", 0.0)),
    }


def viewer_html() -> str:
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Saved UDP Replay Controller</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071016;
      --panel: #101b24;
      --panel2: #172533;
      --text: #edf6ff;
      --muted: #8ea5b8;
      --blue: #5ed7ff;
      --green: #71e49b;
      --orange: #ffb45f;
      --red: #ff6b74;
      --border: rgba(255,255,255,0.1);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, rgba(94,215,255,0.12), transparent 34%), var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .wrap { padding: 18px; display: grid; gap: 14px; }
    .panel { background: rgba(16,27,36,0.94); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }
    .top { display: flex; justify-content: space-between; gap: 16px; align-items: center; flex-wrap: wrap; }
    .title { font-size: 22px; font-weight: 800; }
    .muted { color: var(--muted); font-size: 14px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; }
    button {
      border: 0;
      color: #061018;
      font-weight: 800;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      background: var(--blue);
    }
    button.stop { background: var(--red); color: #1c0507; }
    button.restart { background: var(--orange); color: #201003; }
    button:disabled { opacity: 0.45; cursor: wait; }
    .grid { display: grid; grid-template-columns: minmax(560px, 1.1fr) minmax(420px, 0.9fr); gap: 14px; align-items: start; }
    .images { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
    .image-title, .label { color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
    img { width: 100%; height: 190px; object-fit: cover; border-radius: 12px; border: 1px solid var(--border); background: #050b10; }
    canvas { width: 100%; height: 380px; display: block; border-radius: 12px; background: #050b10; border: 1px solid var(--border); }
    .stats { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
    .card { background: var(--panel2); border-radius: 11px; padding: 10px; }
    .value { font-size: 16px; font-weight: 750; word-break: break-word; }
    .banner { margin-top: 12px; padding: 14px; border: 1px solid rgba(94,215,255,.22); border-radius: 12px; background: rgba(94,215,255,.08); }
    .banner-text { font-size: 21px; font-weight: 800; line-height: 1.25; margin-top: 4px; }
    pre { margin: 0; background: var(--panel2); border-radius: 11px; padding: 10px; max-height: 240px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
    @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } .images { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel top">
      <div>
        <div class="title">Saved UDP Replay Controller</div>
        <div class="muted">저장된 chunk 결과를 버튼으로 UDP 전송 / 중지 / 처음부터 재전송</div>
      </div>
      <div class="controls">
        <button id="start">전송 시작</button>
        <button id="stop" class="stop">중지</button>
        <button id="restart" class="restart">처음부터 다시 전송</button>
      </div>
    </div>
    <div class="grid">
      <div class="panel">
        <div class="images">
          <div><div class="image-title">Left</div><img id="left"></div>
          <div><div class="image-title">Front</div><img id="front"></div>
          <div><div class="image-title">Right</div><img id="right"></div>
          <div><div class="image-title">Front Tele</div><img id="frontTele"></div>
        </div>
        <div class="banner">
          <div class="label">Final Output</div>
          <div id="finalText" class="banner-text">(waiting)</div>
        </div>
      </div>
      <div class="panel">
        <canvas id="canvas"></canvas>
        <div class="muted" style="margin-top:8px">cyan: final/ac_decoded path, orange: GT</div>
      </div>
      <div class="panel">
        <div class="stats">
          <div class="card"><div class="label">State</div><div id="state" class="value">-</div></div>
          <div class="card"><div class="label">Progress</div><div id="progress" class="value">-</div></div>
          <div class="card"><div class="label">UDP Target</div><div id="udp" class="value">-</div></div>
          <div class="card"><div class="label">UDP Payload</div><div id="udpPayloadMode" class="value">-</div></div>
          <div class="card"><div class="label">Current</div><div id="current" class="value">-</div></div>
        </div>
      </div>
      <div class="panel">
        <div class="label" style="margin-bottom: 8px">Latest Status</div>
        <pre id="status">{}</pre>
      </div>
    </div>
  </div>
<script>
const qs = (id) => document.getElementById(id);
const imgs = [["left","latest_left.png"],["front","latest_front.png"],["right","latest_right.png"],["frontTele","latest_front_tele.png"]];

async function command(action) {
  for (const id of ["start","stop","restart"]) qs(id).disabled = true;
  try {
    await fetch(`/api/${action}`, {method: "POST"});
  } finally {
    setTimeout(() => { for (const id of ["start","stop","restart"]) qs(id).disabled = false; }, 300);
  }
}
qs("start").onclick = () => command("start");
qs("stop").onclick = () => command("stop");
qs("restart").onclick = () => command("restart");

function points(path) {
  if (!path) return [];
  if (Array.isArray(path.packet_points)) return path.packet_points;
  if (Array.isArray(path.points)) return path.points.map(p => ({x_m:p.x_m ?? p[0], y_m:p.y_m ?? p[1]}));
  return [];
}
function setupCanvas() {
  const c = qs("canvas");
  const rect = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(1, Math.floor(rect.width * dpr));
  c.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w:rect.width, h:rect.height};
}
function drawPath(ctx, pts, color, ox, oy, scale, dashed=false) {
  if (!pts.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash(dashed ? [8,6] : []);
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = ox - Number(p.y_m || 0) * scale;
    const y = oy - Number(p.x_m || 0) * scale;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  pts.forEach((p, i) => {
    if (i % 4 !== 0 && i !== 0) return;
    const x = ox - Number(p.y_m || 0) * scale;
    const y = oy - Number(p.x_m || 0) * scale;
    ctx.beginPath(); ctx.arc(x,y,i===0?5:2.6,0,Math.PI*2); ctx.fill();
  });
  ctx.restore();
}
function draw(state) {
  const {ctx,w,h} = setupCanvas();
  ctx.fillStyle = "#050b10"; ctx.fillRect(0,0,w,h);
  const latest = state.latest || {};
  const finalPts = points(latest.ac_decoded_path || latest.final_path || (latest.udp_payload ? {packet_points: latest.udp_payload.packet_points} : null));
  const gtPts = points(latest.gt_path);
  const all = finalPts.concat(gtPts);
  let maxX = 12, maxY = 5;
  for (const p of all) { maxX = Math.max(maxX, Number(p.x_m || 0)); maxY = Math.max(maxY, Math.abs(Number(p.y_m || 0))); }
  const rear = 2, forward = Math.min(140, Math.max(16, maxX * 1.12));
  const lateral = Math.min(40, Math.max(6, maxY * 1.4));
  const pad = 24;
  const scale = Math.min((w-pad*2)/(lateral*2), (h-pad*2)/(forward+rear));
  const ox = w/2, oy = pad + forward*scale;
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = 1;
  for (let y=-rear; y<=forward; y+=1) { const py=oy-y*scale; ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(w,py); ctx.stroke(); }
  for (let x=-lateral; x<=lateral; x+=1) { const px=ox-x*scale; ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke(); }
  ctx.strokeStyle = "rgba(255,255,255,.35)";
  ctx.beginPath(); ctx.moveTo(ox,0); ctx.lineTo(ox,h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,oy); ctx.lineTo(w,oy); ctx.stroke();
  drawPath(ctx, gtPts, "#ffb45f", ox, oy, scale, true);
  drawPath(ctx, finalPts, "#5ed7ff", ox, oy, scale, false);
}
async function refresh() {
  try {
    const res = await fetch(`/latest_view_state.json?ts=${Date.now()}`, {cache:"no-store"});
    const state = await res.json();
    qs("state").textContent = state.phase || "-";
    qs("progress").textContent = `${state.current_request_index || 0} / ${state.total_requests || 0}`;
    const latest = state.latest || {};
    const tx = latest.tx_row || {};
    qs("udp").textContent = `${tx.udp_host || state.udp_host || "-"}:${tx.udp_port || state.udp_port || "-"}`;
    qs("udpPayloadMode").textContent = tx.udp_payload_mode || state.udp_payload_mode || "-";
    qs("current").textContent = `chunk ${tx.chunk_id ?? tx.source_chunk_id ?? "-"} / sample ${tx.sample_id ?? "-"}`;
    const summary = latest.ac_decoded_path || latest.final_path || {};
    qs("finalText").textContent = summary.final_output || (latest.udp_payload || {}).final_output || "(waiting)";
    qs("status").textContent = JSON.stringify({phase: state.phase, message: state.message, latest_tx: tx}, null, 2);
    for (const [id, src] of imgs) qs(id).src = `${src}?ts=${Date.now()}`;
    draw(state);
  } catch (err) {
    qs("status").textContent = String(err);
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""


class ReplayController:
    def __init__(
        self,
        *,
        items: list[dict[str, Any]],
        viewer_dir: Path,
        udp_host: str,
        udp_port: int,
        send_interval_s: float,
        udp_payload_mode: str,
        state: ReplayViewerState,
    ) -> None:
        self.items = items
        self.viewer_dir = viewer_dir
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.send_interval_s = send_interval_s
        self.udp_payload_mode = udp_payload_mode
        self.state = state
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.next_index = 0
        self.tx_log: list[dict[str, Any]] = []

    def start(self, restart: bool = False) -> dict[str, Any]:
        with self.lock:
            if restart:
                self.stop_event.set()
                old_thread = self.thread
            else:
                old_thread = None

        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=2.0)

        with self.lock:
            if restart:
                self.next_index = 0
                self.tx_log = []
            if self.thread is not None and self.thread.is_alive():
                return {"ok": True, "message": "already running", "next_index": self.next_index}
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return {"ok": True, "message": "started", "next_index": self.next_index}

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.state.update(phase="stopped", message="UDP replay stopped by viewer button")
        persist_viewer_state(self.viewer_dir, self.state)
        return {"ok": True, "message": "stopping", "next_index": self.next_index}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.thread is not None and self.thread.is_alive()
            return {
                "running": running,
                "next_index": self.next_index,
                "total": len(self.items),
                "udp_host": self.udp_host,
                "udp_port": self.udp_port,
                "send_interval_s": self.send_interval_s,
                "udp_payload_mode": self.udp_payload_mode,
            }

    def _publish_item(self, item: dict[str, Any], index: int, sock: socket.socket) -> None:
        sample_artifact = Path(item["sample_artifact"])
        payload = json.loads(Path(item["payload_path"]).read_text(encoding="utf-8"))
        tx_row = copy.deepcopy(item["tx_row"])
        chunk_id = int(item["chunk_id"])

        path_payload = payload
        if self.udp_payload_mode == "action_ac":
            payload = build_action_ac_payload(path_payload)
        elif self.udp_payload_mode == "control_schema":
            payload = build_control_schema_payload(path_payload)
        payload.update(
            {
                "replay_index": index + 1,
                "replay_count": len(self.items),
                "replay_send_unix_s": time.time(),
                "replay_elapsed_s": index * self.send_interval_s,
                "send_interval_s": self.send_interval_s,
                "source_chunk_id": chunk_id,
                "source_sample_id": tx_row.get("sample_id"),
                "source_actual_offset_s": tx_row.get("actual_offset_s"),
                "udp_payload_mode": self.udp_payload_mode,
            }
        )
        sock.sendto(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            (self.udp_host, self.udp_port),
        )

        for name in ("final_path.json", "ac_decoded_path.json", "gt_path.json"):
            src = sample_artifact / name
            if src.exists():
                (self.viewer_dir / f"latest_{name}").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(self.viewer_dir / "latest_udp_payload.json", payload)

        request_path = item.get("request_path")
        if request_path:
            update_latest_viewer_images(self.viewer_dir, Path(item["request_bank_root"]), Path(request_path))

        tx_row.update(
            {
                "chunk_id": chunk_id,
                "udp_host": self.udp_host,
                "udp_port": self.udp_port,
                "udp_sent": True,
                "udp_mode": f"text_json_once_controlled_resend_{self.udp_payload_mode}",
                "udp_payload_mode": self.udp_payload_mode,
                "replay_index": index + 1,
                "replay_count": len(self.items),
            }
        )
        self.tx_log.append(tx_row)
        write_json(self.viewer_dir / "latest_tx_row.json", tx_row)
        write_json(self.viewer_dir / "controlled_resend_tx_log.json", self.tx_log)

        final_summary = path_json(sample_artifact / "final_path.json")
        ac_summary = path_json(sample_artifact / "ac_decoded_path.json")
        gt_summary = path_json(sample_artifact / "gt_path.json")
        self.state.set_latest(
            final_summary=final_summary,
            ac_summary=ac_summary,
            gt_summary=gt_summary,
            chosen_source="ac_decoded",
            tx_row=tx_row,
        )
        self.state.update(
            phase="running",
            message=f"sent saved path {index + 1}/{len(self.items)}",
            current_request_index=index + 1,
            total_requests=len(self.items),
            udp_sent_count=index + 1,
            udp_host=self.udp_host,
            udp_port=self.udp_port,
            udp_payload_mode=self.udp_payload_mode,
        )
        persist_viewer_state(self.viewer_dir, self.state)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.state.update(phase="running", message="UDP replay started")
            persist_viewer_state(self.viewer_dir, self.state)
            while True:
                if self.stop_event.is_set():
                    return
                with self.lock:
                    if self.next_index >= len(self.items):
                        self.state.update(phase="finished", message="UDP replay finished")
                        persist_viewer_state(self.viewer_dir, self.state)
                        return
                    index = self.next_index
                    item = self.items[index]
                    self.next_index += 1

                self._publish_item(item, index, sock)
                print(
                    f"[controlled-resend] sent {index + 1}/{len(self.items)} "
                    f"chunk={item['chunk_id']} sample_id={item['tx_row'].get('sample_id')}",
                    flush=True,
                )

                deadline = time.monotonic() + self.send_interval_s
                while not self.stop_event.is_set() and time.monotonic() < deadline:
                    time.sleep(min(0.05, deadline - time.monotonic()))
        finally:
            sock.close()


def start_server(host: str, port: int, viewer_dir: Path, controller: ReplayController) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path in {"/", "/viewer"}:
                self._send_bytes(viewer_html().encode("utf-8"), "text/html; charset=utf-8")
                return
            rel = path.lstrip("/")
            if rel == "api/status":
                self._send_json(controller.snapshot())
                return
            target = (viewer_dir / rel).resolve()
            try:
                target.relative_to(viewer_dir.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not target.exists() or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send_bytes(target.read_bytes(), content_type)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/start":
                self._send_json(controller.start(restart=False))
            elif parsed.path == "/api/stop":
                self._send_json(controller.stop())
            elif parsed.path == "/api/restart":
                self._send_json(controller.start(restart=True))
            else:
                self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, value: Any) -> None:
            self._send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json")

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            server.actual_port = candidate  # type: ignore[attr-defined]
            return server
        except OSError:
            continue
    raise RuntimeError(f"Failed to bind viewer server near port {port}")


def main() -> None:
    args = parse_args()
    if args.send_interval_s <= 0:
        raise ValueError("--send-interval-s must be positive")
    run_root = args.run_root.resolve()
    viewer_dir = (args.viewer_dir or (run_root / "controlled_resend_viewer")).resolve()
    viewer_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(run_root, parse_chunks(args.chunks))
    state = ReplayViewerState()
    state.update(
        phase="idle",
        message="viewer ready; press start to send UDP",
        replay_mode="browser_controlled_saved_payload_resend",
        total_requests=len(items),
        current_request_index=0,
        udp_sent_count=0,
        udp_host=args.udp_host,
        udp_port=int(args.udp_port),
        udp_payload_mode=args.udp_payload_mode,
    )
    persist_viewer_state(viewer_dir, state)

    controller = ReplayController(
        items=items,
        viewer_dir=viewer_dir,
        udp_host=args.udp_host,
        udp_port=int(args.udp_port),
        send_interval_s=float(args.send_interval_s),
        udp_payload_mode=args.udp_payload_mode,
        state=state,
    )
    server = start_server(args.viewer_host, int(args.viewer_port), viewer_dir, controller)
    actual_port = int(getattr(server, "actual_port", args.viewer_port))
    viewer_url = f"http://{resolve_viewer_advertise_host(args.viewer_host)}:{actual_port}/viewer"
    state.update(viewer_host=args.viewer_host, viewer_port=actual_port, viewer_url=viewer_url)
    persist_viewer_state(viewer_dir, state)
    print(f"[controlled-resend] viewer: {viewer_url}", flush=True)
    print(f"[controlled-resend] target: {args.udp_host}:{int(args.udp_port)}", flush=True)
    print(f"[controlled-resend] udp payload mode: {args.udp_payload_mode}", flush=True)
    print(f"[controlled-resend] loaded payloads: {len(items)}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        controller.stop()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
