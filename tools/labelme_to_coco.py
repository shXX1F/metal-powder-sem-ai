from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
DEFAULT_PARTICLE_LABELS = {
    "particle",
    "hollow_particle",
    "agglomerate_particle",
    "spherical_particle",
    "颗粒",
    "空心粉",
    "团聚颗粒",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 LabelMe polygon 标注转换成 Mask R-CNN 可用的 COCO 实例分割 JSON。"
    )
    parser.add_argument("--labelme-dir", required=True, help="LabelMe JSON 所在目录")
    parser.add_argument("--image-dir", required=True, help="训练图片目录")
    parser.add_argument("--output", required=True, help="输出 COCO annotations.json")
    parser.add_argument(
        "--particle-labels",
        default=",".join(sorted(DEFAULT_PARTICLE_LABELS)),
        help="按颗粒处理的 LabelMe 标签名，逗号分隔",
    )
    return parser.parse_args()


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def rectangle_to_polygon(points: list[list[float]]) -> list[list[float]]:
    (x1, y1), (x2, y2) = points
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def normalize_shape_points(shape: dict[str, Any]) -> list[list[float]]:
    points = [[float(x), float(y)] for x, y in shape.get("points", [])]
    if shape.get("shape_type") == "rectangle" and len(points) == 2:
        return rectangle_to_polygon(points)
    return points


def find_image_path(json_path: Path, image_dir: Path, image_path_text: str | None) -> Path:
    candidates: list[Path] = []
    if image_path_text:
        image_path = Path(image_path_text)
        candidates.extend(
            [
                image_path,
                json_path.parent / image_path,
                image_dir / image_path,
            ]
        )

    candidates.extend(image_dir / f"{json_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 {json_path.name} 对应的图片。")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def relative_file_name(image_path: Path, image_dir: Path) -> str:
    try:
        return str(image_path.resolve().relative_to(image_dir.resolve())).replace("\\", "/")
    except ValueError:
        return image_path.name


def main() -> None:
    args = parse_args()
    labelme_dir = Path(args.labelme_dir)
    image_dir = Path(args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    particle_labels = {
        label.strip()
        for label in args.particle_labels.split(",")
        if label.strip()
    }

    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "particle", "supercategory": "particle"}],
    }
    ann_id = 1

    for image_id, json_path in enumerate(sorted(labelme_dir.rglob("*.json")), start=1):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = find_image_path(json_path, image_dir, data.get("imagePath"))
        width, height = image_size(image_path)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_file_name(image_path, image_dir),
                "width": width,
                "height": height,
            }
        )

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label.lower().startswith("ignore") or label not in particle_labels:
                continue
            points = normalize_shape_points(shape)
            if len(points) < 3:
                continue
            area = polygon_area(points)
            if area <= 1:
                continue
            flat_segmentation = [coord for point in points for coord in point]
            if any(math.isnan(value) for value in flat_segmentation):
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [flat_segmentation],
                    "bbox": polygon_bbox(points),
                    "area": area,
                    "iscrowd": 0,
                    "attributes": {
                        "source_label": label,
                        "group_id": shape.get("group_id"),
                        **dict(shape.get("flags", {})),
                    },
                }
            )
            ann_id += 1

    output.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"写入 COCO 标注: {output}，"
        f"{len(coco['images'])} 张图，{len(coco['annotations'])} 个颗粒实例。"
    )


if __name__ == "__main__":
    main()
