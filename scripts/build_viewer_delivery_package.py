#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def copy_viewer_dir(viewer_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for name in viewer_dir.iterdir():
        if name.name == "images_bank":
            continue
        if name.is_file():
            shutil.copy2(name, dst_dir / name.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained viewer delivery package with local image cache.")
    parser.add_argument("--viewer-dirs", nargs="+", required=True)
    parser.add_argument("--request-bank-roots", nargs="+", required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--title", default="Viewer Delivery Package")
    args = parser.parse_args()

    package_dir = args.package_dir
    if package_dir.exists():
        shutil.rmtree(package_dir)
    ensure_dir(package_dir)
    images_bank_dir = package_dir / "images_bank"
    shared_cache_dir = package_dir / "shared_frame_cache"
    ensure_dir(images_bank_dir)
    ensure_dir(shared_cache_dir)

    viewer_infos: list[dict[str, str]] = []
    copied_targets: set[tuple[str, str]] = set()
    copied_samples = 0

    request_bank_roots = [Path(x) for x in args.request_bank_roots]

    def resolve_sample_image(stem: str, filename: str) -> Path:
        for root in request_bank_roots:
            candidate = root / "images" / stem / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve {stem}/{filename} in any request bank root")

    for viewer_dir_str in args.viewer_dirs:
        viewer_dir = Path(viewer_dir_str)
        dst_viewer_dir = package_dir / viewer_dir.name
        copy_viewer_dir(viewer_dir, dst_viewer_dir)
        safe_symlink(package_dir / "images_bank", dst_viewer_dir / "images_bank")

        summary_path = viewer_dir / "summary.json"
        viewer_infos.append(
            {
                "viewer_name": viewer_dir.name,
                "summary_file": str(dst_viewer_dir / "summary.json"),
            }
        )

        data_json_candidates = sorted(dst_viewer_dir.glob("*_model_viewer_data.json"))
        if not data_json_candidates:
            raise RuntimeError(f"No viewer data JSON found in {dst_viewer_dir}")
        data_json_path = data_json_candidates[0]
        data = json.loads(data_json_path.read_text())

        for sample in data["samples"]:
            for cam_id_str, rel_img in sample["images"].items():
                rel_path = Path(rel_img)
                if len(rel_path.parts) < 3:
                    continue
                stem = rel_path.parts[1]
                filename = rel_path.parts[2]
                src_img = resolve_sample_image(stem, filename)
                resolved = src_img.resolve()
                semantic = resolved.parent.name
                cache_dst = shared_cache_dir / semantic / resolved.name
                ensure_dir(cache_dst.parent)
                target_key = (semantic, resolved.name)
                if target_key not in copied_targets:
                    shutil.copy2(resolved, cache_dst)
                    copied_targets.add(target_key)
                package_sample_dir = images_bank_dir / stem
                ensure_dir(package_sample_dir)
                safe_symlink(cache_dst, package_sample_dir / filename)
            copied_samples += 1

    readme = f"""# {args.title}

This package contains self-contained HTML viewers and the required image assets.

## Included viewers
"""
    for info in viewer_infos:
        readme += f"- `{info['viewer_name']}`\n"

    readme += f"""

## Layout
- `images_bank/`: sample image paths used by the HTML viewers
- `shared_frame_cache/`: deduplicated PNG assets
- one folder per viewer with:
  - `*_model_timeline_viewer.html`
  - `*_model_viewer_data.json`
  - `*_model_timeline_overview.png`
  - `summary.json`

## How to open
Serve this package directory over HTTP and open the target HTML file in a browser.

## Stats
- viewer count: {len(viewer_infos)}
- sample count touched: {copied_samples}
- unique cached PNGs: {len(copied_targets)}
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    if args.tar_path.exists():
        args.tar_path.unlink()
    ensure_dir(args.tar_path.parent)
    with tarfile.open(args.tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name, recursive=True)

    print(
        json.dumps(
            {
                "package_dir": str(package_dir),
                "tar_path": str(args.tar_path),
                "viewer_count": len(viewer_infos),
                "unique_png_count": len(copied_targets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
