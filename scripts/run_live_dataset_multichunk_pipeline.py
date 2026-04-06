#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_REQUEST_BANK = SCRIPT_DIR / "build_live_chunk_request_bank.py"
RUN_REQUEST_BANK = SCRIPT_DIR / "run_request_bank_persistent.py"
BUILD_VIEWER = SCRIPT_DIR / "build_live_chunk_model_timeline.py"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> None:
    print("[pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def discover_chunk_counts(dataset_root: Path, chunks: list[int]) -> dict[int, int]:
    sample_index = pd.read_parquet(dataset_root / "sample_index_10hz.parquet")
    front_frames = pd.read_parquet(dataset_root / "sensors" / "camera_front" / "frames.parquet")[["frame_id", "chunk_id"]]
    merged = sample_index.merge(front_frames, left_on="front_frame_id", right_on="frame_id", how="left")
    counts = merged[merged["chunk_id"].isin(chunks)].groupby("chunk_id").size().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


def count_outputs(output_root: Path, chunk_id: int) -> int:
    return len(list(output_root.glob(f"output_chunk{chunk_id:04d}_sid*_t0_*.json")))


def write_index_html(index_path: Path, progress: dict) -> None:
    rows = []
    for item in progress["chunks"]:
        viewer_rel = item.get("viewer_rel")
        viewer_link = f'<a href="{viewer_rel}/chunk{item["chunk_id"]:04d}_model_timeline_viewer.html">open viewer</a>' if viewer_rel else "-"
        summary_rel = item.get("summary_rel")
        summary_link = f'<a href="{summary_rel}">summary</a>' if summary_rel else "-"
        rows.append(
            f"""
            <tr>
              <td>{item['chunk_id']:04d}</td>
              <td>{item.get('num_requests_expected', '-')}</td>
              <td>{item.get('num_outputs_done', 0)}</td>
              <td>{item.get('status', 'pending')}</td>
              <td>{viewer_link}</td>
              <td>{summary_link}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Live Dataset Multi-Chunk Progress</title>
  <style>
    body {{
      font-family: "Segoe UI", sans-serif;
      margin: 24px;
      background: #f8f4ea;
      color: #202124;
    }}
    .panel {{
      background: rgba(255,255,255,0.95);
      border: 1px solid #d6c8b2;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(80,63,27,0.08);
      margin-bottom: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #ece2d3;
      padding: 10px;
      text-align: left;
    }}
    th {{
      background: #fffaf1;
    }}
    code {{
      background: #f3ede2;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h2>Dataset Progress</h2>
    <div>dataset: <code>{progress['dataset_root']}</code></div>
    <div>chunks: <code>{", ".join(str(x) for x in progress['chunk_ids'])}</code></div>
    <div>updated_utc: <code>{progress['updated_utc']}</code></div>
    <div>current_stage: <code>{progress.get('current_stage', 'idle')}</code></div>
    <div>current_chunk: <code>{progress.get('current_chunk', '-') }</code></div>
  </div>
  <div class="panel">
    <table>
      <thead>
        <tr>
          <th>chunk</th>
          <th>expected requests</th>
          <th>outputs done</th>
          <th>status</th>
          <th>viewer</th>
          <th>summary</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run request-bank -> inference -> viewer pipeline for multiple live dataset chunks.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chunks", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dashboard-root", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = args.output_root
    dashboard_root = args.dashboard_root
    ensure_dir(output_root)
    ensure_dir(dashboard_root)

    counts = discover_chunk_counts(dataset_root, args.chunks)
    progress = {
        "dataset_root": str(dataset_root),
        "chunk_ids": args.chunks,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_stage": "starting",
        "current_chunk": None,
        "chunks": [],
    }
    for chunk_id in args.chunks:
        progress["chunks"].append(
            {
                "chunk_id": chunk_id,
                "num_requests_expected": counts.get(chunk_id, 0),
                "num_outputs_done": 0,
                "status": "pending",
                "viewer_rel": None,
                "summary_rel": None,
            }
        )

    def save_progress() -> None:
        progress["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        progress_path = dashboard_root / "progress.json"
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        write_index_html(dashboard_root / "index.html", progress)

    save_progress()

    for item in progress["chunks"]:
        chunk_id = int(item["chunk_id"])
        progress["current_chunk"] = chunk_id
        request_bank_root = output_root / f"chunk{chunk_id:04d}_request_bank"
        output_json_root = output_root / f"chunk{chunk_id:04d}_outputs"
        viewer_dir = dashboard_root / f"chunk{chunk_id:04d}_model_timeline_replay"

        item["status"] = "building_request_bank"
        progress["current_stage"] = "building_request_bank"
        save_progress()

        if not (args.skip_existing and (request_bank_root / "summary.json").exists()):
            run_cmd(
                [
                    sys.executable,
                    str(BUILD_REQUEST_BANK),
                    "--dataset-root",
                    str(dataset_root),
                    "--chunk-id",
                    str(chunk_id),
                    "--output-root",
                    str(request_bank_root),
                ]
            )

        item["status"] = "running_inference"
        progress["current_stage"] = "running_inference"
        save_progress()

        run_cmd(
            [
                sys.executable,
                str(RUN_REQUEST_BANK),
                "--request-root",
                str(request_bank_root / "requests"),
                "--output-root",
                str(output_json_root),
                "--skip-existing",
            ]
        )

        item["num_outputs_done"] = count_outputs(output_json_root, chunk_id)
        item["status"] = "building_viewer"
        progress["current_stage"] = "building_viewer"
        save_progress()

        run_cmd(
            [
                sys.executable,
                str(BUILD_VIEWER),
                "--dataset-root",
                str(dataset_root),
                "--chunk-id",
                str(chunk_id),
                "--request-bank-root",
                str(request_bank_root),
                "--output-root",
                str(output_json_root),
                "--viewer-dir",
                str(viewer_dir),
            ]
        )

        item["num_outputs_done"] = count_outputs(output_json_root, chunk_id)
        item["status"] = "done"
        item["viewer_rel"] = viewer_dir.relative_to(dashboard_root).as_posix()
        item["summary_rel"] = f"{item['viewer_rel']}/summary.json"
        progress["current_stage"] = "chunk_done"
        save_progress()

    progress["current_stage"] = "complete"
    progress["current_chunk"] = None
    save_progress()
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
