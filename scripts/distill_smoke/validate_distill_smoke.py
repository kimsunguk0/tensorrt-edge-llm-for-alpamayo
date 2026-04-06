#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate distill smoke dataset structure and numeric sanity")
    p.add_argument("--input-pt", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/distill_smoke_5.pt")
    p.add_argument("--output-json", default="/root/TensorRT-Edge-LLM-v060/output/distill_smoke/distill_smoke_5.validation.json")
    return p.parse_args()


def is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def main() -> None:
    args = parse_args()
    input_pt = Path(args.input_pt)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(input_pt, map_location="cpu", weights_only=False)
    samples = data.get("samples", [])

    errors: list[str] = []
    checks: list[dict] = []

    for i, s in enumerate(samples):
        run_name = s.get("run_name")
        prefix = f"sample[{i}]({run_name})"
        teacher = s.get("teacher", {})
        x0 = teacher.get("x0")
        x_final = teacher.get("x_final")
        pred_xyz = teacher.get("pred_xyz")
        pred_rot = teacher.get("pred_rot")

        local_errors: list[str] = []
        if not isinstance(x0, torch.Tensor) or tuple(x0.shape) != (64, 2):
            local_errors.append("x0 shape must be [64,2]")
        if not isinstance(x_final, torch.Tensor) or tuple(x_final.shape) != (64, 2):
            local_errors.append("x_final shape must be [64,2]")
        if not isinstance(pred_xyz, torch.Tensor) or tuple(pred_xyz.shape) != (64, 3):
            local_errors.append("pred_xyz shape must be [64,3]")
        if not isinstance(pred_rot, torch.Tensor) or tuple(pred_rot.shape) != (64, 3, 3):
            local_errors.append("pred_rot shape must be [64,3,3]")

        for name, t in [("x0", x0), ("x_final", x_final), ("pred_xyz", pred_xyz), ("pred_rot", pred_rot)]:
            if isinstance(t, torch.Tensor) and not is_finite_tensor(t):
                local_errors.append(f"{name} contains NaN/Inf")

        image_files = s.get("images", {}).get("files", [])
        if not image_files:
            local_errors.append("images.files is empty")
        else:
            missing = [p for p in image_files if not Path(p).exists()]
            if missing:
                local_errors.append(f"missing image files: {len(missing)}")

        ego_xyz = Path(s.get("paths", {}).get("ego_xyz_npy", ""))
        ego_rot = Path(s.get("paths", {}).get("ego_rot_npy", ""))
        if not ego_xyz.exists() or not ego_rot.exists():
            local_errors.append("ego history npy file missing")

        checks.append(
            {
                "run_name": run_name,
                "num_images": len(image_files),
                "ok": len(local_errors) == 0,
                "errors": local_errors,
            }
        )
        errors.extend([f"{prefix}: {e}" for e in local_errors])

    report = {
        "input_pt": str(input_pt),
        "num_samples": len(samples),
        "num_failed": sum(0 if c["ok"] else 1 for c in checks),
        "ok": len(errors) == 0,
        "checks": checks,
        "errors": errors,
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(output_json)
    print(f"ok={report['ok']} num_samples={len(samples)} num_failed={report['num_failed']}")
    if errors:
        for e in errors[:20]:
            print("ERR", e)


if __name__ == "__main__":
    main()
