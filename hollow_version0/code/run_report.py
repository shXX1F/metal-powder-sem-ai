#!/usr/bin/env python3
"""Packaged entrypoint for hollow-powder inference.

Examples:

    python code/run_report.py --input /path/to/image.png
    python code/run_report.py --input /path/to/image_folder --name sample_001

Outputs are written under:

    ans/data/back/<run_name>/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANS_ROOT = Path(__file__).resolve().parents[1]
GIVE_ROOT = ANS_ROOT / "data" / "give"
BACK_ROOT = ANS_ROOT / "data" / "back"


def safe_name(text: str) -> str:
    keep = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("._")
    return name or "run"


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def prepare_input(input_path: Path, run_name: str, copy_input: bool = True) -> Path:
    """Normalize a single image or image folder into ans/data/give/<run_name>."""
    give_dir = GIVE_ROOT / run_name
    if give_dir.exists():
        shutil.rmtree(give_dir)
    give_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {input_path}")
        target = give_dir / input_path.name
        if copy_input:
            shutil.copy2(input_path, target)
        else:
            target.symlink_to(input_path)
        return give_dir

    if input_path.is_dir():
        imgs = image_files(input_path)
        if not imgs:
            raise RuntimeError(f"No images found in {input_path}")
        for img in imgs:
            target = give_dir / img.name
            if copy_input:
                shutil.copy2(img, target)
            else:
                target.symlink_to(img)
        return give_dir

    raise FileNotFoundError(input_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image file or image folder.")
    parser.add_argument("--name", default=None, help="Output run name under data/back/.")
    parser.add_argument("--hollow-only", action="store_true", help="Skip particle inference.")
    parser.add_argument("--no-copy", action="store_true", help="Symlink input images into data/give instead of copying.")
    parser.add_argument("--reset", action="store_true", help="Overwrite data/back/<name> if it exists.")
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--hollow-full-score", type=float, default=0.2)
    parser.add_argument("--hollow-tile-score", type=float, default=0.3)
    parser.add_argument("--hollow-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--hollow-full-edge", type=float, default=0.4)
    parser.add_argument("--hollow-tile-edge", type=float, default=0.55)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    run_name = safe_name(args.name or input_path.stem)
    give_dir = prepare_input(input_path, run_name, copy_input=not args.no_copy)
    output_dir = BACK_ROOT / run_name
    if output_dir.exists() and args.reset:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ANS_ROOT / "code" / "mytools" / "run_hollow_report.py"),
        "--input_dir",
        str(give_dir),
        "--output_dir",
        str(output_dir),
        "--hollow_full_score",
        str(args.hollow_full_score),
        "--hollow_tile_score",
        str(args.hollow_tile_score),
        "--hollow_ratio_threshold",
        str(args.hollow_ratio_threshold),
        "--hollow_edge_white_ratio_threshold",
        str(args.hollow_tile_edge),
        "--hollow_full_edge_white_ratio_threshold",
        str(args.hollow_full_edge),
        "--hollow_tile_edge_white_ratio_threshold",
        str(args.hollow_tile_edge),
        "--image_batch_size",
        str(args.image_batch_size),
        "--torch_threads",
        str(args.torch_threads),
    ]
    if args.hollow_only:
        cmd.append("--hollow_only")

    print("give_dir=", give_dir)
    print("output_dir=", output_dir)
    print("command=", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ANS_ROOT / "code"), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
