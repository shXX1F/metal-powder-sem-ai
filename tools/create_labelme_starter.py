from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPS_DIRS = [Path(r"D:\CodexDeps\sem_runtime"), Path(r"D:\CodexDeps\sem_labelme"), PROJECT_ROOT / ".deps"]
for deps_dir in reversed(DEPS_DIRS):
    if deps_dir.exists():
        sys.path.insert(0, str(deps_dir))
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from metal_powder_sem_ai.preprocess import imwrite_unicode, preprocess_sem_image
from metal_powder_sem_ai.segment import ClassicalParticleSegmenter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成一批可人工修正的 LabelMe 初始颗粒标注。"
    )
    parser.add_argument("--image-root", default="图像SEM数据")
    parser.add_argument("--output-dir", default="data/labelme_starter")
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.12)
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--peak-min-distance", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    text = "__".join(relative.with_suffix("").parts)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text)
    return text


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def group_by_batch(paths: Iterable[Path], root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) >= 2:
            key = "/".join(relative.parts[:2])
        else:
            key = relative.parts[0]
        groups.setdefault(key, []).append(path)
    return groups


def split_groups(groups: dict[str, list[Path]], seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    train_end = max(1, round(n * 0.70))
    val_end = max(train_end + 1, round(n * 0.85)) if n >= 3 else train_end
    split_keys = {
        "train": keys[:train_end],
        "val": keys[train_end:val_end],
        "test": keys[val_end:],
    }
    if not split_keys["test"] and n >= 3:
        split_keys["test"] = [split_keys["val"].pop()]
    return {
        split: [path for key in split_keys[split] for path in groups[key]]
        for split in ["train", "val", "test"]
    }


def cap_split_counts(split_paths: dict[str, list[Path]], max_images: int, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    train_count = max(1, int(max_images * 0.70))
    val_count = max(1, int(max_images * 0.15)) if max_images >= 3 else 0
    test_count = max(0, max_images - train_count - val_count)
    if max_images >= 3 and test_count == 0:
        train_count = max(1, train_count - 1)
        test_count = 1
    targets = {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }
    selected: dict[str, list[Path]] = {}
    for split, paths in split_paths.items():
        paths = list(paths)
        rng.shuffle(paths)
        selected[split] = sorted(paths[: targets[split]])
    return selected


def contour_to_points(contour: np.ndarray, max_points: int = 80) -> list[list[float]]:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(0.8, 0.006 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    points = approx.reshape(-1, 2)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return [[float(x), float(y)] for x, y in points]


def write_labelme_json(
    json_path: Path,
    image_path_text: str,
    image_shape: tuple[int, int],
    polygons: list[list[list[float]]],
) -> None:
    height, width = image_shape
    shapes = [
        {
            "label": "particle",
            "points": polygon,
            "group_id": None,
            "description": "auto-generated; please review",
            "shape_type": "polygon",
            "flags": {"auto_label": True},
        }
        for polygon in polygons
        if len(polygon) >= 3
    ]
    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path_text,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = image_files(image_root)
    groups = group_by_batch(all_images, image_root)
    split_paths = cap_split_counts(split_groups(groups, args.seed), args.max_images, args.seed)
    segmenter = ClassicalParticleSegmenter(
        min_area_px=args.min_area_px,
        peak_min_distance=args.peak_min_distance,
    )

    manifest_rows = ["split,source_path,image_path,labelme_json,auto_particles"]
    total = 0
    for split, paths in split_paths.items():
        image_out_dir = output_dir / split / "images"
        label_out_dir = output_dir / split / "labelme"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for source_path in paths:
            pre = preprocess_sem_image(
                source_path,
                crop_bottom_fraction=args.crop_bottom_fraction,
            )
            instances = segmenter.segment(pre.binary)
            polygons = [contour_to_points(instance.contour) for instance in instances]

            name = safe_name(source_path, image_root)
            image_name = f"{name}.png"
            json_name = f"{name}.json"
            image_path = image_out_dir / image_name
            json_path = label_out_dir / json_name

            imwrite_unicode(image_path, pre.image)
            write_labelme_json(
                json_path=json_path,
                image_path_text=f"../images/{image_name}",
                image_shape=pre.image.shape[:2],
                polygons=polygons,
            )
            write_labelme_json(
                json_path=image_out_dir / json_name,
                image_path_text=image_name,
                image_shape=pre.image.shape[:2],
                polygons=polygons,
            )
            manifest_rows.append(
                ",".join(
                    [
                        split,
                        str(source_path).replace("\\", "/"),
                        str(image_path).replace("\\", "/"),
                        str(json_path).replace("\\", "/"),
                        str(len(polygons)),
                    ]
                )
            )
            total += 1

    (output_dir / "manifest.csv").write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="utf-8-sig",
    )
    print(f"生成 {total} 张初始标注，输出目录: {output_dir}")
    print("下一步：用 LabelMe 打开 output/split/images 中的图片，逐张修正 polygon。")


if __name__ == "__main__":
    main()


