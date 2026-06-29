#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_flex_legacy_threeway_dataset_compare import (  # noqa: E402
    DEFAULT_LEGACY_FM,
    VariantConfig,
    build_dataset_request_bank,
    build_variant_artifacts,
    ensure_exists,
    path_xy,
    prepare_variant_requests,
    timing_value,
    write_json,
)
from scripts.run_legacy_seed23_latency_rollout import stitch_latency_rollout  # noqa: E402
from scripts.run_request_bank_persistent import build_env, read_status  # noqa: E402


@dataclass(frozen=True)
class CameraMode:
    name: str
    label: str
    color: str
    camera_semantics: tuple[str, ...]


CAMERA_MODES = [
    CameraMode(
        name="all4",
        label="4cam left/front/right/tele",
        color="#2563eb",
        camera_semantics=("left", "front", "right", "front_tele"),
    ),
    CameraMode(
        name="no_tele",
        label="3cam left/front/right",
        color="#dc2626",
        camera_semantics=("left", "front", "right"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy prefill-KV seed 23 with all 4 cameras vs tele removed."
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[
            REPO_ROOT / "data" / "2026-06-12-test1",
            REPO_ROOT / "data" / "2026-06-12-test2",
        ],
    )
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "output" / "legacy_seed23_tele_ablation_20260612")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--llm-inference-bin", type=Path, default=REPO_ROOT / "build" / "examples" / "llm" / "llm_inference")
    parser.add_argument("--plugin-lib", type=Path, default=REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so")
    parser.add_argument("--engine-dir", type=Path, default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5"))
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=Path("/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild"),
    )
    parser.add_argument("--fm-engine", type=Path, default=DEFAULT_LEGACY_FM)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout-per-request", type=float, default=900.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def quantile_or_none(values: list[float | None], q: float) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return None
    return float(np.quantile(np.asarray(finite, dtype=np.float64), q))


def run_outputs_with_wall_times(
    *,
    variant: VariantConfig,
    request_paths: list[Path],
    output_root: Path,
    llm_inference_bin: Path,
    plugin_lib: Path,
    warmup: int,
    timeout_per_request: float,
    skip_existing: bool,
) -> tuple[list[Path], dict[str, float]]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [output_root / p.name.replace("request_", "output_") for p in request_paths]
    wall_path = output_root / "request_wall_times.json"
    if skip_existing and wall_path.exists() and all(path.exists() for path in output_paths):
        wall_times = {str(k): float(v) for k, v in load_json(wall_path).items()}
        print(f"[tele-ablation] reuse outputs for {variant.name}", flush=True)
        return output_paths, wall_times

    cmd = [
        str(llm_inference_bin),
        "--engineDir",
        str(variant.engine_dir),
        "--multimodalEngineDir",
        str(variant.multimodal_engine_dir),
        "--fmEngine",
        str(variant.fm_engine),
        "--alpamayoPostVlmRuntime",
        "--persistentServer",
        "--warmup",
        str(warmup),
    ]
    if variant.use_prefill_kv:
        cmd.append("--alpamayoFmUsePrefillKv")

    print(f"[tele-ablation] starting {variant.name}", flush=True)
    print("[tele-ablation] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=build_env(plugin_lib),
    )

    wall_times: dict[str, float] = {}
    completed = 0
    start = time.time()
    try:
        ready = read_status(proc, timeout_s=180.0)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected ready state for {variant.name}: {ready}")
        print(f"[tele-ablation] {variant.name} ready", flush=True)

        for request_path, output_path in zip(request_paths, output_paths, strict=True):
            if skip_existing and output_path.exists() and request_path.stem in wall_times:
                completed += 1
                continue
            payload = {"input_file": str(request_path), "output_file": str(output_path)}
            assert proc.stdin is not None
            t0 = time.perf_counter()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            status = read_status(proc, timeout_s=timeout_per_request)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if status.get("status") != "ok":
                raise RuntimeError(f"{variant.name} failed for {request_path.name}: {status}")
            wall_times[request_path.stem] = float(wall_ms)
            completed += 1
            elapsed = time.time() - start
            eta = (elapsed / max(completed, 1)) * (len(request_paths) - completed)
            print(
                f"[tele-ablation] {variant.name} done {completed}/{len(request_paths)} "
                f"{request_path.name} wall={wall_ms:.1f}ms | eta {eta/60:.1f}m",
                flush=True,
            )
    finally:
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

    write_json(wall_path, wall_times)
    return output_paths, wall_times


def attach_wall_times(artifacts: dict[str, dict[str, Any]], wall_times: dict[str, float]) -> None:
    for stem, item in artifacts.items():
        wall_ms = wall_times.get(stem)
        if wall_ms is None:
            continue
        final = item["final"]
        timing = dict(final.get("timing", {}))
        timing["request_wall_ms"] = float(wall_ms)
        final["timing"] = timing
        final_path = Path(item["artifact_root"]) / "final_path.json"
        if final_path.exists():
            write_json(final_path, final)


def per_sample_quality(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ade: list[float] = []
    fde: list[float] = []
    path_len: list[float] = []
    abs_curv: list[float] = []
    abs_accel: list[float] = []
    abs_jerk: list[float] = []

    for item in artifacts.values():
        final = item["final"]
        gt = item["gt"]
        pred = path_xy(final)
        gt_pts = path_xy(gt)
        n = min(len(pred), len(gt_pts))
        if n <= 1:
            continue
        err = np.linalg.norm(pred[:n] - gt_pts[:n], axis=1)
        ade.append(float(np.mean(err)))
        fde.append(float(err[-1]))
        diffs = np.diff(pred, axis=0)
        path_len.append(float(np.linalg.norm(diffs, axis=1).sum()))

        curv = np.asarray(final.get("pred_curvature", []), dtype=np.float64)
        if curv.size:
            abs_curv.extend(np.abs(curv).tolist())

        v = np.asarray(final.get("pred_v_mps", []), dtype=np.float64)
        dt_s = float(final.get("plan_dt_s", 0.1))
        if v.size >= 2 and dt_s > 0:
            accel = np.diff(v) / dt_s
            abs_accel.extend(np.abs(accel).tolist())
            if accel.size >= 2:
                jerk = np.diff(accel) / dt_s
                abs_jerk.extend(np.abs(jerk).tolist())

    return {
        "sample_count": int(len(ade)),
        "local_ade_mean_m": float(np.mean(ade)) if ade else None,
        "local_ade_p90_m": float(np.quantile(ade, 0.90)) if ade else None,
        "local_fde_mean_m": float(np.mean(fde)) if fde else None,
        "local_fde_p90_m": float(np.quantile(fde, 0.90)) if fde else None,
        "bad_case_rate_fde_gt_3m": float(np.mean(np.asarray(fde) > 3.0)) if fde else None,
        "path_len_mean_m": float(np.mean(path_len)) if path_len else None,
        "abs_curvature_mean": float(np.mean(abs_curv)) if abs_curv else None,
        "abs_curvature_p90": float(np.quantile(abs_curv, 0.90)) if abs_curv else None,
        "abs_accel_mean_mps2": float(np.mean(abs_accel)) if abs_accel else None,
        "abs_jerk_p90_mps3": float(np.quantile(abs_jerk, 0.90)) if abs_jerk else None,
    }


def timing_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    timings = [item["final"].get("timing", {}) for item in artifacts.values()]
    keys = [
        "request_wall_ms",
        "total_post_vlm_ms",
        "guided_pass_ms",
        "fm_wall_ms",
        "ego_history_load_ms",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = [t.get(key) for t in timings]
        out[f"{key}_mean"] = mean_or_none(values)
        out[f"{key}_p50"] = quantile_or_none(values, 0.50)
        out[f"{key}_p90"] = quantile_or_none(values, 0.90)
    return out


def draw_dataset_compare(
    *,
    out_path: Path,
    dataset_name: str,
    mode_summaries: dict[str, dict[str, Any]],
    mode_npz_paths: dict[str, Path],
) -> None:
    fig = plt.figure(figsize=(15, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.28, wspace=0.18)
    fig.patch.set_facecolor("#f8fafc")

    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("white")
    gt_dense = None
    loaded: dict[str, Any] = {}
    for mode in CAMERA_MODES:
        data = np.load(mode_npz_paths[mode.name])
        loaded[mode.name] = data
        if gt_dense is None:
            gt_dense = data["gt_dense_world_xy"]
        stitched = data["stitched_world_xy"]
        ax.plot(stitched[:, 0], stitched[:, 1], color=mode.color, linewidth=2.0, label=mode.label)
    assert gt_dense is not None
    ax.plot(gt_dense[:, 0], gt_dense[:, 1], color="#475569", linewidth=2.0, linestyle="--", label="GNSS GT")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("ENU east [m]")
    ax.set_ylabel("ENU north [m]")
    ax.set_title(f"{dataset_name}: stitched path with/without tele")
    ax.legend(fontsize=8)

    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.set_facecolor("white")
    for mode in CAMERA_MODES:
        data = loaded[mode.name]
        times = data["stitched_times_ns"]
        rel_t = (times.astype(np.float64) - float(times[0])) / 1e9
        ax_err.plot(rel_t, data["errors_m"], color=mode.color, linewidth=1.5, label=mode.label)
    ax_err.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax_err.set_xlabel("rollout time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.set_title("GT error")
    ax_err.legend(fontsize=8)

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    lines = []
    for mode in CAMERA_MODES:
        item = mode_summaries[mode.name]
        lines.extend(
            [
                f"{mode.label}",
                f"  wall {item['timing']['request_wall_ms_mean']:.1f}ms mean / {item['timing']['request_wall_ms_p90']:.1f}ms p90",
                f"  local ADE {item['quality']['local_ade_mean_m']:.2f}m, FDE {item['quality']['local_fde_mean_m']:.2f}m",
                f"  stitched ADE {item['stitched']['ade_m']:.2f}m, p90 {item['stitched']['p90_error_m']:.2f}m",
            ]
        )
    ax_txt.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax_txt.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95, "pad": 8},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rel_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return float((new - old) / old * 100.0)


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2f}{suffix}"


def write_report(path: Path, all_results: dict[str, Any]) -> None:
    lines = [
        "# Legacy Seed 23 Tele Camera Ablation",
        "",
        "Legacy prefill-KV, FM seed 23, diffusion 2-step에서 같은 샘플을 4cam과 tele 제거 3cam으로 각각 실행했다.",
        "`request_wall_ms`는 persistent server가 request를 받은 뒤 output JSON을 받을 때까지의 wall time이다.",
        "`total_post_vlm_ms`는 output JSON 내부의 Alpamayo post-VLM timing이다.",
        "",
        "| dataset | mode | images | wall mean ms | wall p90 ms | post mean ms | guided mean ms | fm mean ms | local ADE m | local FDE m | stitched ADE m | stitched p90 m | PNG |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset_name, dataset_result in all_results["datasets"].items():
        png = dataset_result["comparison_png"]
        for mode in CAMERA_MODES:
            item = dataset_result["modes"][mode.name]
            timing = item["timing"]
            quality = item["quality"]
            stitched = item["stitched"]
            images = len(mode.camera_semantics) * 4
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset_name,
                        mode.name,
                        str(images),
                        fmt(timing["request_wall_ms_mean"]),
                        fmt(timing["request_wall_ms_p90"]),
                        fmt(timing["total_post_vlm_ms_mean"]),
                        fmt(timing["guided_pass_ms_mean"]),
                        fmt(timing["fm_wall_ms_mean"]),
                        fmt(quality["local_ade_mean_m"]),
                        fmt(quality["local_fde_mean_m"]),
                        fmt(stitched["ade_m"]),
                        fmt(stitched["p90_error_m"]),
                        png if mode.name == "all4" else "",
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Delta: no_tele vs all4", ""])
    lines.append("| dataset | wall delta | post delta | local ADE delta | local FDE delta | stitched ADE delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset_name, dataset_result in all_results["datasets"].items():
        all4 = dataset_result["modes"]["all4"]
        no = dataset_result["modes"]["no_tele"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset_name,
                    fmt(rel_change(no["timing"]["request_wall_ms_mean"], all4["timing"]["request_wall_ms_mean"]), "%"),
                    fmt(rel_change(no["timing"]["total_post_vlm_ms_mean"], all4["timing"]["total_post_vlm_ms_mean"]), "%"),
                    fmt(rel_change(no["quality"]["local_ade_mean_m"], all4["quality"]["local_ade_mean_m"]), "%"),
                    fmt(rel_change(no["quality"]["local_fde_mean_m"], all4["quality"]["local_fde_mean_m"]), "%"),
                    fmt(rel_change(no["stitched"]["ade_m"], all4["stitched"]["ade_m"]), "%"),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for p, desc in [
        (args.llm_inference_bin, "llm_inference"),
        (args.plugin_lib, "plugin lib"),
        (args.engine_dir, "engine dir"),
        (args.multimodal_engine_dir, "multimodal engine dir"),
        (args.fm_engine, "fm engine"),
    ]:
        ensure_exists(p, desc)

    variant = VariantConfig(
        name=f"legacy_prefill_seed_{args.seed}",
        label=f"Legacy prefill KV seed {args.seed}",
        color="#2563eb",
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        use_prefill_kv=True,
        diffusion_seed=int(args.seed),
        diffusion_num_steps=int(args.diffusion_num_steps),
        max_generate_length=int(args.max_generate_length),
    )

    all_results: dict[str, Any] = {
        "work_root": str(args.work_root),
        "seed": int(args.seed),
        "diffusion_num_steps": int(args.diffusion_num_steps),
        "sample_count": int(args.sample_count),
        "datasets": {},
    }

    for dataset_root in args.dataset_roots:
        dataset_root = dataset_root.resolve()
        ensure_exists(dataset_root, "dataset root")
        dataset_name = dataset_root.name
        dataset_result: dict[str, Any] = {"modes": {}}
        mode_npz_paths: dict[str, Path] = {}
        print(f"[tele-ablation] dataset {dataset_name}", flush=True)

        for mode in CAMERA_MODES:
            mode_root = args.work_root / dataset_name / mode.name
            request_bank_root = mode_root / "request_bank"
            print(f"[tele-ablation] build request bank {dataset_name} {mode.name}", flush=True)
            manifest_rows, request_summary = build_dataset_request_bank(
                dataset_root=dataset_root,
                out_root=request_bank_root,
                chunk_id=args.chunk_id,
                sample_count=args.sample_count,
                width=args.width,
                height=args.height,
                history_len=args.history_len,
                dt_s=args.dt_s,
                traj_token_offset=args.traj_token_offset,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                camera_semantics=mode.camera_semantics,
            )
            variant_root = mode_root / "variant" / variant.name
            request_paths = prepare_variant_requests(
                manifest_rows=manifest_rows,
                variant=variant,
                out_root=variant_root / "request_bank",
            )
            output_paths, wall_times = run_outputs_with_wall_times(
                variant=variant,
                request_paths=request_paths,
                output_root=variant_root / "outputs",
                llm_inference_bin=args.llm_inference_bin,
                plugin_lib=args.plugin_lib,
                warmup=args.warmup,
                timeout_per_request=args.timeout_per_request,
                skip_existing=args.skip_existing,
            )
            artifacts = build_variant_artifacts(
                variant=variant,
                output_paths=output_paths,
                request_paths=request_paths,
                manifest_rows=manifest_rows,
                dataset_root=dataset_root,
                history_len=args.history_len,
                artifact_root=variant_root / "artifacts",
            )
            attach_wall_times(artifacts, wall_times)
            stitched = stitch_latency_rollout(
                dataset_name=f"{dataset_name}_{mode.name}",
                dataset_root=dataset_root,
                request_summary=request_summary,
                manifest_rows=manifest_rows,
                artifacts=artifacts,
                dt_s=args.dt_s,
                out_root=mode_root,
            )
            mode_npz_paths[mode.name] = mode_root / "stitched_latency_rollout.npz"
            dataset_result["modes"][mode.name] = {
                "label": mode.label,
                "camera_semantics": list(mode.camera_semantics),
                "request_bank_summary": request_summary,
                "timing": timing_summary(artifacts),
                "quality": per_sample_quality(artifacts),
                "stitched": stitched,
                "wall_times_json": str(variant_root / "outputs" / "request_wall_times.json"),
                "mode_root": str(mode_root),
            }

        compare_png = args.work_root / dataset_name / f"{dataset_name}_all4_vs_no_tele_stitched.png"
        draw_dataset_compare(
            out_path=compare_png,
            dataset_name=dataset_name,
            mode_summaries=dataset_result["modes"],
            mode_npz_paths=mode_npz_paths,
        )
        dataset_result["comparison_png"] = str(compare_png)
        all_results["datasets"][dataset_name] = dataset_result

    write_json(args.work_root / "summary.json", all_results)
    write_report(args.work_root / "tele_ablation_report.md", all_results)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
