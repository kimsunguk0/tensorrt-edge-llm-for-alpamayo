#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Variant:
    name: str
    engine_dir: Path
    multimodal_engine_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VLM engine variants with a shared multimodal request.")
    parser.add_argument(
        "--request-file",
        type=Path,
        required=True,
        help="Input JSON request for llm_inference. Use a pure multimodal request without FM-specific runtime flags.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where per-run profiles and aggregated summaries will be written.",
    )
    parser.add_argument(
        "--llm-inference-bin",
        type=Path,
        default=Path("/workspace/alpamayo_vlm/build/examples/llm/llm_inference"),
        help="Path to llm_inference binary.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured runs per variant.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup count forwarded to llm_inference for each run.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Timeout per llm_inference invocation.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant spec in the form name::engine_dir::multimodal_engine_dir. Repeat for multiple variants.",
    )
    return parser.parse_args()


def parse_variant(spec: str) -> Variant:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid variant spec: {spec!r}")
    name, engine_dir, multimodal_engine_dir = parts
    return Variant(name=name, engine_dir=Path(engine_dir), multimodal_engine_dir=Path(multimodal_engine_dir))


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["stage_id"]: stage for stage in profile.get("stages", [])}


def pick_vision_ms(profile: dict[str, Any]) -> float | None:
    stages = stage_map(profile)
    for key in ("vision_encoder", "multimodal_processing"):
        stage = stages.get(key)
        if stage is not None:
            return float(stage["average_time_per_run_ms"])
    return None


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    if len(values) == 1:
        return {
            "count": 1,
            "mean": values[0],
            "median": values[0],
            "min": values[0],
            "max": values[0],
            "stdev": 0.0,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def run_variant(
    *,
    llm_inference_bin: Path,
    request_file: Path,
    output_root: Path,
    variant: Variant,
    runs: int,
    warmup: int,
    timeout_s: float,
) -> dict[str, Any]:
    variant_root = output_root / variant.name
    variant_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for run_idx in range(1, runs + 1):
        output_file = variant_root / f"output_run{run_idx:02d}.json"
        profile_file = variant_root / f"profile_run{run_idx:02d}.json"
        stdout_file = variant_root / f"stdout_run{run_idx:02d}.log"
        stderr_file = variant_root / f"stderr_run{run_idx:02d}.log"
        cmd = [
            str(llm_inference_bin),
            "--engineDir",
            str(variant.engine_dir),
            "--multimodalEngineDir",
            str(variant.multimodal_engine_dir),
            "--inputFile",
            str(request_file),
            "--outputFile",
            str(output_file),
            "--dumpProfile",
            "--profileOutputFile",
            str(profile_file),
            "--warmup",
            str(warmup),
        ]
        start = time.time()
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        wall_s = time.time() - start
        stdout_file.write_text(result.stdout, encoding="utf-8", errors="ignore")
        stderr_file.write_text(result.stderr, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            raise RuntimeError(
                f"Variant {variant.name} run {run_idx} failed with exit code {result.returncode}. "
                f"See {stdout_file} and {stderr_file}"
            )
        profile = load_json(profile_file)
        stages = stage_map(profile)
        row = {
            "run_index": run_idx,
            "wall_s": wall_s,
            "vision_encoder_ms": pick_vision_ms(profile),
            "prefill_ms": maybe_float(profile.get("prefill", {}).get("average_time_per_run_ms")),
            "prefill_tokens_per_run": maybe_float(profile.get("prefill", {}).get("average_tokens_per_run")),
            "decode_total_ms": maybe_float(stages.get("llm_generation", {}).get("average_time_per_run_ms")),
            "decode_ms_per_token": maybe_float(profile.get("generation", {}).get("average_time_per_token_ms")),
            "generated_tokens": maybe_float(profile.get("generation", {}).get("generated_tokens")),
        }
        run_rows.append(row)

    summary = {
        "variant": variant.name,
        "engine_dir": str(variant.engine_dir),
        "multimodal_engine_dir": str(variant.multimodal_engine_dir),
        "runs": run_rows,
        "metrics": {
            "vision_encoder_ms": summarize([x["vision_encoder_ms"] for x in run_rows if x["vision_encoder_ms"] is not None]),
            "prefill_ms": summarize([x["prefill_ms"] for x in run_rows if x["prefill_ms"] is not None]),
            "decode_total_ms": summarize([x["decode_total_ms"] for x in run_rows if x["decode_total_ms"] is not None]),
            "decode_ms_per_token": summarize([x["decode_ms_per_token"] for x in run_rows if x["decode_ms_per_token"] is not None]),
            "wall_s": summarize([x["wall_s"] for x in run_rows if x["wall_s"] is not None]),
        },
    }
    (variant_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_report(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    report = {"summaries": summaries}
    (output_root / "benchmark_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def combined_stage_metric(summary: dict[str, Any], stat_key: str) -> float | None:
        metric_keys = ("vision_encoder_ms", "prefill_ms", "decode_total_ms")
        values: list[float] = []
        for metric_key in metric_keys:
            metric = summary["metrics"].get(metric_key)
            if metric is None or metric.get(stat_key) is None:
                return None
            values.append(float(metric[stat_key]))
        return sum(values)

    metric_rows = [
        ("ViT mean ms", "vision_encoder_ms", "mean"),
        ("ViT median ms", "vision_encoder_ms", "median"),
        ("Prefill mean ms", "prefill_ms", "mean"),
        ("Prefill median ms", "prefill_ms", "median"),
        ("Decode mean ms/run", "decode_total_ms", "mean"),
        ("Decode median ms/run", "decode_total_ms", "median"),
        ("Decode mean ms/token", "decode_ms_per_token", "mean"),
        ("Decode median ms/token", "decode_ms_per_token", "median"),
        ("Stage total mean ms", "__combined_stage_total__", "mean"),
        ("Stage total median ms", "__combined_stage_total__", "median"),
        ("Wall mean s", "wall_s", "mean"),
        ("Wall median s", "wall_s", "median"),
    ]

    csv_path = output_root / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *[summary["variant"] for summary in summaries]])
        for row_label, metric_key, stat_key in metric_rows:
            row = [row_label]
            for summary in summaries:
                if metric_key == "__combined_stage_total__":
                    row.append(combined_stage_metric(summary, stat_key))
                    continue
                metric = summary["metrics"].get(metric_key)
                row.append(metric.get(stat_key) if metric else None)
            writer.writerow(row)

    md_lines = [
        "| Metric | " + " | ".join(summary["variant"] for summary in summaries) + " |",
        "|---|" + "|".join("---:" for _ in summaries) + "|",
    ]
    for row_label, metric_key, stat_key in metric_rows:
        row_values: list[str] = []
        for summary in summaries:
            if metric_key == "__combined_stage_total__":
                combined_value = combined_stage_metric(summary, stat_key)
                if combined_value is None:
                    row_values.append("-")
                else:
                    row_values.append(f"{combined_value:.3f}")
                continue
            metric = summary["metrics"].get(metric_key)
            if metric is None:
                row_values.append("-")
            else:
                row_values.append(f"{metric.get(stat_key, float('nan')):.3f}")
        md_lines.append("| " + row_label + " | " + " | ".join(row_values) + " |")
    (output_root / "benchmark_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_exists(args.llm_inference_bin, "llm_inference")
    ensure_exists(args.request_file, "request file")
    variants = [parse_variant(spec) for spec in args.variant]
    if not variants:
        raise RuntimeError("At least one --variant is required.")
    for variant in variants:
        ensure_exists(variant.engine_dir, f"engine dir for {variant.name}")
        ensure_exists(variant.multimodal_engine_dir, f"multimodal engine dir for {variant.name}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[benchmark] running {variant.name}", flush=True)
        summaries.append(
            run_variant(
                llm_inference_bin=args.llm_inference_bin,
                request_file=args.request_file,
                output_root=args.output_root,
                variant=variant,
                runs=int(args.runs),
                warmup=int(args.warmup),
                timeout_s=float(args.timeout_s),
            )
        )

    build_report(args.output_root, summaries)
    print(json.dumps({"output_root": str(args.output_root), "variants": [x["variant"] for x in summaries]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[benchmark] error: {exc}", file=sys.stderr)
        raise
