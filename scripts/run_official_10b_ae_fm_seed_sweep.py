#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_official_10b_hf_request_bank import (
    build_model_inputs,
    build_sample_from_manifest_row,
    finite_float,
    load_manifest_rows,
    patch_torchvision_nms_schema,
)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("empty seed list")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Alpamayo-1.5-10B Action Expert FM seed sweep. "
            "The VLM rollout/KV is generated once per sample with a fixed seed; "
            "only the flow-matching initial noise seed is swept."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/alpamayo1.5/model"))
    parser.add_argument("--vlm-name-or-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--code-src",
        type=Path,
        default=Path("/workspace/alpamayo1.5/code/alpamayo1.5/src"),
    )
    parser.add_argument(
        "--request-bank-root",
        type=Path,
        default=Path(
            "/workspace/alpamayo_vlm/output/flex_legacy_threeway_20260612/"
            "2026-06-12-test1/request_bank"
        ),
    )
    parser.add_argument(
        "--gt-artifact-root",
        type=Path,
        default=None,
        help="Optional root containing request_*/gt_path.json. Defaults to sibling legacy_prefill artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "official_10b_ae_fm_seed_sweep_20260612_test1_10samples",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--fm-seeds",
        default="42,2,0,1,7,11,17,23,99,1234",
        help="Comma-separated flow-matching seeds. Includes 42 and 2 by default.",
    )
    parser.add_argument("--vlm-seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def request_stem(row: dict[str, Any]) -> str:
    return Path(row["request_json"]).stem


def path_xy_from_pred(pred_xyz: list[list[float]] | np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_xyz, dtype=np.float32)
    pts = np.zeros((pred.shape[0] + 1, 2), dtype=np.float32)
    pts[1:, 0] = pred[:, 0]
    pts[1:, 1] = pred[:, 1]
    return pts


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def short_text(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def prepare_official_ae_context(
    *,
    model: Any,
    model_inputs: dict[str, Any],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )

    data = {
        "tokenized_data": dict(model_inputs["tokenized_data"]),
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = data["tokenized_data"].pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
    )
    device = input_ids.device

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.top_p = args.top_p
    generation_config.temperature = args.temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = args.max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None if args.top_k < 0 else args.top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    torch.cuda.manual_seed_all(args.vlm_seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **data["tokenized_data"],
        )
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star = vlm_outputs.sequences.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    offset = model._find_eos_offset(
        sequences=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        device=device,
    )
    prefix_mask = data["tokenized_data"].get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, 1, dim=0)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )
    extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
    cot = ""
    if isinstance(extra, dict) and extra.get("cot"):
        cot = str(extra["cot"][0])

    if hasattr(vlm_outputs, "logits"):
        del vlm_outputs.logits
    torch.cuda.empty_cache()

    return {
        "device": device,
        "prompt_cache": prompt_cache,
        "prefill_seq_len": prefill_seq_len,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "n_diffusion_tokens": n_diffusion_tokens,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "cot": cot,
        "generated_tokens": int(vlm_outputs.sequences.shape[1] - input_ids.shape[1]),
        "sequence_length": int(vlm_outputs.sequences.shape[1]),
        "forward_kwargs": {"is_causal": False} if model.config.expert_non_causal_attention else {},
    }


def sample_ae_for_fm_seed(
    *,
    model: Any,
    ctx: dict[str, Any],
    fm_seed: int,
    diffusion_steps: int,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_cache = ctx["prompt_cache"]
    prefill_seq_len = int(ctx["prefill_seq_len"])
    n_diffusion_tokens = int(ctx["n_diffusion_tokens"])
    position_ids = ctx["position_ids"]
    attention_mask = ctx["attention_mask"]
    forward_kwargs = ctx["forward_kwargs"]

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b_star = x.shape[0]
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)
        expert_out_base = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

    torch.cuda.manual_seed_all(int(fm_seed))
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        sampled_action = model.diffusion.sample(
            batch_size=1,
            step_fn=step_fn,
            device=ctx["device"],
            return_all_steps=False,
            inference_step=int(diffusion_steps),
        )
    hist_xyz = ctx["ego_history_xyz"][:, -1]
    hist_rot = ctx["ego_history_rot"][:, -1]
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
    return (
        pred_xyz.detach().cpu().float().numpy()[0],
        pred_rot.detach().cpu().float().numpy()[0],
    )


def make_payload(
    *,
    row: dict[str, Any],
    pred_xyz: np.ndarray,
    pred_rot: np.ndarray,
    cot: str,
    runtime_s: float,
    vlm_runtime_s: float,
    fm_seed: int,
    vlm_seed: int,
    generated_tokens: int,
    sequence_length: int,
    diffusion_steps: int,
) -> dict[str, Any]:
    return {
        "label": "official_10b_ae_fm_seed",
        "model": "nvidia/Alpamayo-1.5-10B",
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "front_frame_id": int((row.get("selected_frames") or {}).get("front", [-1])[-1]),
        "t0_utc_ns": int(row["t0_utc_ns"]),
        "t0_us": int(row["t0_us"]),
        "vlm_seed": int(vlm_seed),
        "fm_seed": int(fm_seed),
        "diffusion_steps": int(diffusion_steps),
        "plan_dt_s": 0.1,
        "plan_points_no_origin": int(pred_xyz.shape[0]),
        "pred_xyz": pred_xyz.astype(float).tolist(),
        "pred_rot": pred_rot.reshape(pred_rot.shape[0], -1).astype(float).tolist(),
        "cot": cot,
        "runtime_s": finite_float(runtime_s),
        "vlm_runtime_s": finite_float(vlm_runtime_s),
        "generated_tokens": int(generated_tokens),
        "sequence_length": int(sequence_length),
    }


def maybe_load_gt_points(gt_artifact_root: Path | None, row: dict[str, Any]) -> np.ndarray | None:
    if gt_artifact_root is None:
        return None
    path = gt_artifact_root / request_stem(row) / "gt_path.json"
    if not path.exists():
        return None
    data = load_json(path)
    return path_xy_from_pred(data["pred_xyz"])


def draw_sample_overlay(
    *,
    out_path: Path,
    row: dict[str, Any],
    seed_payloads: list[dict[str, Any]],
    gt_points: np.ndarray | None,
) -> None:
    fig = plt.figure(figsize=(13.5, 10.8), dpi=160)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.85, 2.2], hspace=0.24, wspace=0.08)
    fig.patch.set_facecolor("#f8fafc")

    image_dir = Path(row["image_dir"])
    for idx, cam_id in enumerate([0, 1, 2, 6]):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.set_title(f"cam{cam_id} f3", fontsize=9)
        ax_img.axis("off")
        image_path = image_dir / f"cam{cam_id}_f3.png"
        if image_path.exists():
            ax_img.imshow(plt.imread(image_path))
        else:
            ax_img.text(0.5, 0.5, "missing", ha="center", va="center")

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    cot = seed_payloads[0].get("cot") or ""
    endpoints = []
    for item in seed_payloads:
        pts = path_xy_from_pred(item["pred_xyz"])
        endpoints.append(
            f"s{item['fm_seed']}:({pts[-1,0]:.1f},{pts[-1,1]:.1f}) L={path_length(pts):.1f}"
        )
    text = (
        f"chunk {row['chunk_id']:04d} | sample {row['sample_id']} | t0_us {row['t0_us']} | "
        f"official 10B AE FM seed sweep | VLM seed {seed_payloads[0]['vlm_seed']} | "
        f"{seed_payloads[0]['diffusion_steps']}-step\n"
        f"CoT: {short_text(cot)}\n"
        + " | ".join(endpoints)
    )
    ax_text.text(
        0.01,
        0.95,
        text,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        transform=ax_text.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.96, "pad": 6},
    )

    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor("#ffffff")
    hist_xyz = np.load(row["ego_history_xyz_npy"]).astype(np.float32)[0, 0]
    ax.plot(hist_xyz[:, 1], hist_xyz[:, 0], color="#0f172a", linewidth=1.6, alpha=0.72, label="ego history")

    all_pts = [hist_xyz[:, :2]]
    if gt_points is not None:
        all_pts.append(gt_points)
        ax.plot(gt_points[:, 1], gt_points[:, 0], color="#64748b", linewidth=2.0, alpha=0.72, linestyle="--", label="GNSS GT")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(seed_payloads), 10)))
    for idx, item in enumerate(seed_payloads):
        pts = path_xy_from_pred(item["pred_xyz"])
        all_pts.append(pts)
        lw = 3.2 if int(item["fm_seed"]) in (42, 2) else 2.0
        alpha = 0.98 if int(item["fm_seed"]) in (42, 2) else 0.78
        ax.plot(
            pts[:, 1],
            pts[:, 0],
            color=colors[idx],
            linewidth=lw,
            alpha=alpha,
            label=f"FM seed {item['fm_seed']}",
        )
        ax.scatter(pts[-1, 1], pts[-1, 0], color=colors[idx], s=20, zorder=5)

    ax.scatter([0.0], [0.0], color="#111827", s=28, zorder=6, label="t0")
    stack = np.concatenate([np.asarray(p, dtype=np.float32)[:, :2] for p in all_pts], axis=0)
    x_forward_min, y_lat_min = stack.min(axis=0)
    x_forward_max, y_lat_max = stack.max(axis=0)
    forward_pad = max(5.0, float(x_forward_max - x_forward_min) * 0.15)
    lat_pad = max(3.0, float(y_lat_max - y_lat_min) * 0.20)
    ax.set_ylim(float(x_forward_min - forward_pad), float(x_forward_max + forward_pad))
    ax.set_xlim(float(y_lat_min - lat_pad), float(y_lat_max + lat_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.set_title("Official 10B AE flow-matching seed sweep")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 5, thumb_width: int = 640) -> Path:
    if not image_paths:
        raise RuntimeError("No images for contact sheet")
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        aspect = img.height / max(img.width, 1)
        thumb = img.resize((thumb_width, int(thumb_width * aspect)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
        labels.append(path.stem)

    label_h = 22
    cell_w = thumb_width
    cell_h = max(img.height for img in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 8, y + 4), labels[idx], fill="#0f172a")
        sheet.paste(thumb, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    fm_seeds = parse_int_list(args.fm_seeds)
    gt_artifact_root = args.gt_artifact_root
    if gt_artifact_root is None:
        gt_artifact_root = args.request_bank_root.parent / "variants" / "legacy_prefill" / "artifacts"

    patch_torchvision_nms_schema()
    sys.path.insert(0, str(args.code_src))
    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(args.request_bank_root, args.limit)
    out_jsonl = args.output_root / "predictions.jsonl"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    with (args.model_dir / "config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["vlm_name_or_path"] = args.vlm_name_or_path
    config_dict["attn_implementation"] = args.attn_implementation
    config = Alpamayo1_5Config(**config_dict)

    print(f"[official10b-seed] load {args.model_dir} dtype={dtype} attn={args.attn_implementation}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        str(args.model_dir),
        config=config,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    done: set[tuple[int, int, int]] = set()
    existing_payloads: dict[tuple[int, int], list[dict[str, Any]]] = {}
    if args.skip_existing and out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (int(item["chunk_id"]), int(item["sample_id"]), int(item["fm_seed"]))
            done.add(key)
            existing_payloads.setdefault((key[0], key[1]), []).append(item)
        print(f"[official10b-seed] reuse {len(done)} existing seed rows", flush=True)

    write_json(
        args.output_root / "summary.json",
        {
            "model_dir": str(args.model_dir),
            "request_bank_root": str(args.request_bank_root),
            "output_root": str(args.output_root),
            "sample_count": len(rows),
            "fm_seeds": fm_seeds,
            "vlm_seed": int(args.vlm_seed),
            "diffusion_steps": int(args.diffusion_steps),
            "top_p": float(args.top_p),
            "temperature": float(args.temperature),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "gt_artifact_root": str(gt_artifact_root) if gt_artifact_root is not None else None,
        },
    )

    overlay_paths: list[Path] = []
    started = time.time()
    with out_jsonl.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            chunk_id = int(row["chunk_id"])
            sample_id = int(row["sample_id"])
            pending_seeds = [
                seed for seed in fm_seeds if (chunk_id, sample_id, int(seed)) not in done
            ]
            sample_payloads = list(existing_payloads.get((chunk_id, sample_id), []))

            vlm_runtime_s = 0.0
            ctx: dict[str, Any] | None = None
            if pending_seeds:
                data = build_sample_from_manifest_row(row)
                model_inputs = build_model_inputs(
                    helper=helper,
                    processor=processor,
                    data=data,
                    device="cuda",
                )
                vlm_t0 = time.time()
                ctx = prepare_official_ae_context(
                    model=model,
                    model_inputs=model_inputs,
                    dtype=dtype,
                    args=args,
                )
                vlm_runtime_s = time.time() - vlm_t0
                print(
                    f"[official10b-seed] {idx}/{len(rows)} sid{sample_id:05d} "
                    f"vlm {vlm_runtime_s:.2f}s cot={short_text(ctx['cot'], 80)}",
                    flush=True,
                )
                del model_inputs, data

            for seed in pending_seeds:
                assert ctx is not None
                seed_t0 = time.time()
                pred_xyz, pred_rot = sample_ae_for_fm_seed(
                    model=model,
                    ctx=ctx,
                    fm_seed=int(seed),
                    diffusion_steps=args.diffusion_steps,
                    dtype=dtype,
                )
                runtime_s = time.time() - seed_t0
                payload = make_payload(
                    row=row,
                    pred_xyz=pred_xyz,
                    pred_rot=pred_rot,
                    cot=ctx["cot"],
                    runtime_s=runtime_s,
                    vlm_runtime_s=vlm_runtime_s,
                    fm_seed=int(seed),
                    vlm_seed=int(args.vlm_seed),
                    generated_tokens=int(ctx["generated_tokens"]),
                    sequence_length=int(ctx["sequence_length"]),
                    diffusion_steps=int(args.diffusion_steps),
                )
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
                sample_payloads.append(payload)
                print(
                    f"[official10b-seed] sid{sample_id:05d} fm_seed={seed} "
                    f"{runtime_s:.2f}s end=({pred_xyz[-1,0]:.2f},{pred_xyz[-1,1]:.2f})",
                    flush=True,
                )
                torch.cuda.empty_cache()

            if ctx is not None:
                del ctx
                torch.cuda.empty_cache()

            sample_payloads = sorted(sample_payloads, key=lambda item: fm_seeds.index(int(item["fm_seed"])) if int(item["fm_seed"]) in fm_seeds else 999)
            overlay_path = args.output_root / "overlays" / f"overlay_seed_sweep_{request_stem(row).removeprefix('request_')}.png"
            draw_sample_overlay(
                out_path=overlay_path,
                row=row,
                seed_payloads=sample_payloads,
                gt_points=maybe_load_gt_points(gt_artifact_root, row),
            )
            overlay_paths.append(overlay_path)
            print(
                f"[official10b-seed] rendered {overlay_path.name} total={(time.time()-started)/60:.1f}m",
                flush=True,
            )

    contact_sheet = make_contact_sheet(overlay_paths, args.output_root / "contact_sheet_official_10b_ae_fm_seed_sweep.png")
    print(json.dumps({"output_root": str(args.output_root), "contact_sheet": str(contact_sheet), "overlays": len(overlay_paths)}, indent=2))


if __name__ == "__main__":
    main()
