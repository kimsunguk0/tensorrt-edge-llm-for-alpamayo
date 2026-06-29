#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList


CAMERA_ID_BY_SEMANTIC = {
    "left": 0,
    "front": 1,
    "right": 2,
    "front_tele": 6,
}

_TORCHVISION_SCHEMA_LIB: Any | None = None


def patch_torchvision_nms_schema() -> None:
    global _TORCHVISION_SCHEMA_LIB
    try:
        _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_SCHEMA_LIB.define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official HF Alpamayo-1.5-10B checkpoint on a flat request bank "
            "and write Action Expert plus VLM discrete-128 trajectory JSONL rows."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/alpamayo1.5/model"))
    parser.add_argument(
        "--vlm-name-or-path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Backbone config/tokenizer source used by the official checkpoint wrapper.",
    )
    parser.add_argument(
        "--code-src",
        type=Path,
        default=Path("/workspace/alpamayo1.5/code/alpamayo1.5/src"),
    )
    parser.add_argument("--request-bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional sample ids to run from the request bank manifest.",
    )
    parser.add_argument(
        "--nav-text",
        default=None,
        help="Optional route instruction to insert with <|route_start|>...<|route_end|>.",
    )
    parser.add_argument(
        "--use-nav-prompt",
        action="store_true",
        help="Use the navigation-style prompt even when --nav-text is absent.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "ae", "discrete"), default="both")
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--ae-max-generation-length", type=int, default=256)
    parser.add_argument("--discrete-max-generation-length", type=int, default=320)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def request_stem(row: dict[str, Any]) -> str:
    return Path(row["request_json"]).name.removeprefix("request_").removesuffix(".json")


def load_manifest_rows(
    request_bank_root: Path,
    limit: int,
    sample_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    manifest_path = request_bank_root / "manifest.json"
    rows = load_json(manifest_path)
    rows = sorted(rows, key=lambda row: int(row["t0_utc_ns"]))
    if sample_ids:
        wanted = set(int(x) for x in sample_ids)
        rows = [row for row in rows if int(row["sample_id"]) in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_sample_from_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    image_dir = Path(row["image_dir"])
    selected_frames = row["selected_frames"]
    ordered = sorted(
        ((CAMERA_ID_BY_SEMANTIC[name], name) for name in selected_frames.keys()),
        key=lambda item: item[0],
    )
    camera_indices = torch.tensor([camera_id for camera_id, _ in ordered], dtype=torch.int64)

    per_camera_frames: list[torch.Tensor] = []
    for camera_id, _semantic_name in ordered:
        frames = [
            load_image_tensor(image_dir / f"cam{camera_id}_f{frame_idx}.png")
            for frame_idx in range(4)
        ]
        per_camera_frames.append(torch.stack(frames, dim=0))
    image_frames = torch.stack(per_camera_frames, dim=0)

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": torch.from_numpy(np.load(row["ego_history_xyz_npy"])).float(),
        "ego_history_rot": torch.from_numpy(np.load(row["ego_history_rot_npy"])).float(),
    }


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def token_span_count(sequences: torch.Tensor, start_id: int, end_id: int) -> int:
    seq = sequences[0].detach().cpu().tolist()
    try:
        start = len(seq) - 1 - seq[::-1].index(start_id)
    except ValueError:
        return 0
    try:
        end = seq.index(end_id, start + 1)
    except ValueError:
        end = len(seq)
    return max(0, end - start - 1)


def build_model_inputs(
    *,
    helper: Any,
    processor: Any,
    data: dict[str, Any],
    device: str,
    nav_text: str | None = None,
    use_nav_prompt: bool = False,
) -> dict[str, Any]:
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
        nav_text=nav_text,
        use_nav_prompt=use_nav_prompt,
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": tokenized,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def run_official_ae(
    *,
    model: Any,
    model_inputs: dict[str, Any],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            top_k=None if args.top_k < 0 else args.top_k,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.ae_max_generation_length,
            diffusion_kwargs={"inference_step": args.diffusion_steps},
            return_extra=True,
        )
    pred_xyz_np = pred_xyz.detach().cpu().float().numpy()[0, 0, 0]
    pred_rot_np = pred_rot.detach().cpu().float().numpy()[0, 0, 0]
    cot = None
    if isinstance(extra, dict) and "cot" in extra:
        cot = str(extra["cot"][0, 0, 0])
    return pred_xyz_np, pred_rot_np, cot


def run_official_discrete(
    *,
    model: Any,
    model_inputs: dict[str, Any],
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str | None, dict[str, Any]]:
    from alpamayo1_5.models.token_utils import StopAfterEOS, extract_text_tokens, extract_traj_tokens

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

    generation_config = model.vlm.generation_config
    generation_config.top_p = args.top_p
    generation_config.temperature = args.temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = args.discrete_max_generation_length
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None if args.top_k < 0 else args.top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    future_end_id = model.special_token_ids["traj_future_end"]
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=future_end_id)])

    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        generated = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=LogitsProcessorList(),
            **data["tokenized_data"],
        )

    traj_tokens = extract_traj_tokens(
        output_tokens=generated.sequences,
        special_token_ids=model.special_token_ids,
        tokens_per_future_traj=model.config.tokens_per_future_traj,
        future_token_start_idx=model.future_token_start_idx,
        traj_tokenizer_vocab_size=model.traj_tokenizer.vocab_size,
    )
    hist_xyz = data["ego_history_xyz"][:, -1]
    hist_rot = data["ego_history_rot"][:, -1]
    with torch.no_grad():
        pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(hist_xyz, hist_rot, traj_tokens)

    extra = extract_text_tokens(model.tokenizer, generated.sequences)
    cot = None
    if isinstance(extra, dict) and "cot" in extra and extra["cot"]:
        cot = str(extra["cot"][0])

    meta = {
        "generated_tokens": int(generated.sequences.shape[1] - input_ids.shape[1]),
        "discrete_token_count_between_markers": token_span_count(
            generated.sequences,
            model.special_token_ids["traj_future_start"],
            model.special_token_ids["traj_future_end"],
        ),
        "expected_discrete_tokens": int(model.config.tokens_per_future_traj),
        "decoded_text": model.tokenizer.batch_decode(
            generated.sequences[:, input_ids.shape[1] :], skip_special_tokens=False
        )[0],
    }
    return (
        pred_xyz.detach().cpu().float().numpy()[0],
        pred_rot.detach().cpu().float().numpy()[0],
        cot,
        meta,
    )


def make_payload(
    *,
    row: dict[str, Any],
    label: str,
    pred_xyz: np.ndarray,
    pred_rot: np.ndarray,
    cot: str | None,
    runtime_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "label": label,
        "model": "nvidia/Alpamayo-1.5-10B",
        "chunk_id": int(row["chunk_id"]),
        "sample_id": int(row["sample_id"]),
        "front_frame_id": int((row.get("selected_frames") or {}).get("front", [-1])[-1]),
        "t0_utc_ns": int(row["t0_utc_ns"]),
        "t0_us": int(row["t0_us"]),
        "plan_dt_s": 0.1,
        "plan_points_no_origin": int(pred_xyz.shape[0]),
        "pred_xyz": pred_xyz.astype(float).tolist(),
        "pred_rot": pred_rot.reshape(pred_rot.shape[0], -1).astype(float).tolist(),
        "cot": cot,
        "runtime_s": finite_float(runtime_s),
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    args = parse_args()
    patch_torchvision_nms_schema()
    sys.path.insert(0, str(args.code_src))

    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_root / "predictions.jsonl"
    summary_path = args.output_root / "summary.json"
    rows = load_manifest_rows(args.request_bank_root, args.limit, args.sample_ids)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    with (args.model_dir / "config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["vlm_name_or_path"] = args.vlm_name_or_path
    config_dict["attn_implementation"] = args.attn_implementation
    config = Alpamayo1_5Config(**config_dict)

    print(f"[official10b] load {args.model_dir} dtype={dtype} attn={args.attn_implementation}", flush=True)
    model = Alpamayo1_5.from_pretrained(
        str(args.model_dir),
        config=config,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    summary = {
        "model_dir": str(args.model_dir),
        "request_bank_root": str(args.request_bank_root),
        "output_root": str(args.output_root),
        "sample_count": len(rows),
        "sample_ids": args.sample_ids,
        "nav_text": args.nav_text,
        "use_nav_prompt": bool(args.use_nav_prompt),
        "mode": args.mode,
        "seed": args.seed,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "top_k": None if args.top_k < 0 else args.top_k,
        "ae_max_generation_length": args.ae_max_generation_length,
        "discrete_max_generation_length": args.discrete_max_generation_length,
        "diffusion_steps": args.diffusion_steps,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "labels": ["official_10b_ae", "official_10b_discrete128"] if args.mode == "both" else [args.mode],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    done_keys: set[tuple[int, int, str]] = set()
    if args.skip_existing and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                done_keys.add((int(item["chunk_id"]), int(item["sample_id"]), str(item["label"])))
        print(f"[official10b] reuse {len(done_keys)} existing rows", flush=True)

    started = time.time()
    with out_jsonl.open("a", encoding="utf-8") as f:
        for index, row in enumerate(rows, start=1):
            chunk_id = int(row["chunk_id"])
            sample_id = int(row["sample_id"])
            data = build_sample_from_manifest_row(row)
            model_inputs = build_model_inputs(
                helper=helper,
                processor=processor,
                data=data,
                device="cuda",
                nav_text=args.nav_text,
                use_nav_prompt=args.use_nav_prompt,
            )

            if args.mode in ("both", "ae") and (chunk_id, sample_id, "official_10b_ae") not in done_keys:
                sample_start = time.time()
                pred_xyz, pred_rot, cot = run_official_ae(
                    model=model,
                    model_inputs=model_inputs,
                    dtype=dtype,
                    args=args,
                )
                elapsed = time.time() - sample_start
                payload = make_payload(
                    row=row,
                    label="official_10b_ae",
                    pred_xyz=pred_xyz,
                    pred_rot=pred_rot,
                    cot=cot,
                    runtime_s=elapsed,
                    extra={"diffusion_steps": int(args.diffusion_steps)},
                )
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[official10b] {index}/{len(rows)} sid{sample_id:05d} ae {elapsed:.2f}s",
                    flush=True,
                )
                torch.cuda.empty_cache()

            if args.mode in ("both", "discrete") and (
                chunk_id,
                sample_id,
                "official_10b_discrete128",
            ) not in done_keys:
                sample_start = time.time()
                pred_xyz, pred_rot, cot, meta = run_official_discrete(
                    model=model,
                    model_inputs=model_inputs,
                    dtype=dtype,
                    args=args,
                )
                elapsed = time.time() - sample_start
                payload = make_payload(
                    row=row,
                    label="official_10b_discrete128",
                    pred_xyz=pred_xyz,
                    pred_rot=pred_rot,
                    cot=cot,
                    runtime_s=elapsed,
                    extra=meta,
                )
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[official10b] {index}/{len(rows)} sid{sample_id:05d} discrete {elapsed:.2f}s "
                    f"tokens={meta['discrete_token_count_between_markers']}",
                    flush=True,
                )
                torch.cuda.empty_cache()

            total_elapsed = time.time() - started
            print(f"[official10b] progress total={total_elapsed / 60:.1f}m", flush=True)
            del model_inputs, data
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
