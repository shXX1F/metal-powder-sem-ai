from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training-ready COCO boost set from GH tiled LabelMe data and HX full LabelMe data.",
    )
    parser.add_argument(
        "--gh-root",
        default="data/GH3536-SA0120724061801-H1",
        help="Root containing GH LabelMe JSON/images.",
    )
    parser.add_argument(
        "--hx-root",
        default="data/HX-ZA4720724102201",
        help="Root containing HX LabelMe JSON/images.",
    )
    parser.add_argument(
        "--image-root",
        default="data",
        help="Image root passed to training. COCO file_name values are relative to this root.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/boost_round1_gh_hx",
        help="Output directory for annotations, manifest, and generated HX tiles.",
    )
    parser.add_argument("--hx-columns", type=int, default=8)
    parser.add_argument("--hx-overlap", type=int, default=40)
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--particle-labels", default="particle")
    parser.add_argument(
        "--include-gh-whole",
        action="store_true",
        help="Also include GH whole images. Off by default to avoid huge-mask training memory.",
    )
    parser.add_argument(
        "--include-hx-whole",
        action="store_true",
        help="Also include HX whole images in addition to generated HX tiles.",
    )
    parser.add_argument("--overwrite", action="store_true")
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


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        return None
    return data


def find_image_path(json_path: Path, image_root: Path, image_path_text: str | None) -> Path:
    candidates: list[Path] = []
    if image_path_text:
        image_path = Path(image_path_text)
        candidates.extend(
            [
                image_path,
                json_path.parent / image_path,
                image_root / image_path,
            ]
        )
    candidates.extend(json_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    candidates.extend(image_root / f"{json_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {json_path}")


def relative_to_image_root(path: Path, image_root: Path) -> str:
    try:
        return path.resolve().relative_to(image_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def add_image(coco: dict[str, Any], file_name: str, width: int, height: int) -> int:
    image_id = len(coco["images"]) + 1
    coco["images"].append(
        {
            "id": image_id,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
        }
    )
    return image_id


def add_annotation(
    coco: dict[str, Any],
    image_id: int,
    points: list[list[float]],
    min_area: float,
    source_label: str,
) -> bool:
    if len(points) < 3:
        return False
    area = polygon_area(points)
    if area < min_area:
        return False
    flat = [float(coord) for point in points for coord in point]
    if any(math.isnan(value) for value in flat):
        return False
    ann_id = len(coco["annotations"]) + 1
    coco["annotations"].append(
        {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [flat],
            "bbox": polygon_bbox(points),
            "area": area,
            "iscrowd": 0,
            "attributes": {"source_label": source_label},
        }
    )
    return True


def iter_labelme_json(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for json_path in sorted(root.rglob("*.json")):
        data = load_json(json_path)
        if data is None:
            continue
        yield json_path, data


def add_gh_tiles(
    coco: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    gh_root: Path,
    image_root: Path,
    particle_labels: set[str],
    min_area: float,
    include_whole: bool,
) -> None:
    for json_path, data in iter_labelme_json(gh_root):
        is_tile = "__col" in json_path.stem
        if not is_tile and not include_whole:
            continue
        image_path = find_image_path(json_path, image_root, data.get("imagePath"))
        with Image.open(image_path) as image:
            width, height = image.size
        image_id = add_image(
            coco,
            relative_to_image_root(image_path, image_root),
            width,
            height,
        )
        before = len(coco["annotations"])
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label not in particle_labels:
                continue
            add_annotation(
                coco,
                image_id,
                normalize_shape_points(shape),
                min_area=min_area,
                source_label=label,
            )
        manifest_rows.append(
            {
                "dataset": "GH3536-SA0120724061801-H1",
                "source_json": str(json_path),
                "image_file": coco["images"][-1]["file_name"],
                "mode": "existing_tile" if is_tile else "whole",
                "annotations": len(coco["annotations"]) - before,
            }
        )


def add_labelme_whole_images(
    coco: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    dataset_name: str,
    root: Path,
    image_root: Path,
    particle_labels: set[str],
    min_area: float,
) -> None:
    for json_path, data in iter_labelme_json(root):
        if "__col" in json_path.stem:
            continue
        image_path = find_image_path(json_path, image_root, data.get("imagePath"))
        with Image.open(image_path) as image:
            width, height = image.size
        image_id = add_image(
            coco,
            relative_to_image_root(image_path, image_root),
            width,
            height,
        )
        before = len(coco["annotations"])
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if label not in particle_labels:
                continue
            add_annotation(
                coco,
                image_id,
                normalize_shape_points(shape),
                min_area=min_area,
                source_label=label,
            )
        manifest_rows.append(
            {
                "dataset": dataset_name,
                "source_json": str(json_path),
                "image_file": coco["images"][-1]["file_name"],
                "mode": "whole",
                "annotations": len(coco["annotations"]) - before,
            }
        )


def column_spans(width: int, columns: int, overlap: int) -> list[tuple[int, int]]:
    step = int(math.ceil(width / columns))
    spans: list[tuple[int, int]] = []
    for index in range(columns):
        x1 = min(width - 1, index * step)
        x2 = min(width, x1 + step + overlap)
        if index == columns - 1:
            x2 = width
        if x2 > x1:
            spans.append((x1, x2))
    return spans


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = approx.reshape(-1, 2).astype(float).tolist()
        polygons.append(points)
    return polygons


def clipped_shape_polygons(
    shape: dict[str, Any],
    x1: int,
    x2: int,
    height: int,
) -> list[list[list[float]]]:
    points = normalize_shape_points(shape)
    if len(points) < 3:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if max(xs) < x1 or min(xs) >= x2 or max(ys) < 0 or min(ys) >= height:
        return []

    tile_width = x2 - x1
    mask = np.zeros((height, tile_width), dtype=np.uint8)
    shifted = np.asarray([[x - x1, y] for x, y in points], dtype=np.float32)
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 1)
    return mask_to_polygons(mask)


def add_hx_as_tiles(
    coco: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    hx_root: Path,
    image_root: Path,
    output_dir: Path,
    particle_labels: set[str],
    columns: int,
    overlap: int,
    min_area: float,
) -> None:
    tile_dir = output_dir / "hx_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    for json_path, data in iter_labelme_json(hx_root):
        image_path = find_image_path(json_path, image_root, data.get("imagePath"))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for col_index, (x1, x2) in enumerate(column_spans(width, columns, overlap), start=1):
                tile_name = (
                    f"{image_path.stem}__col{col_index}of{columns}_"
                    f"x{x1:04d}-{x2 - 1:04d}_overlap{overlap}.jpg"
                )
                tile_path = tile_dir / tile_name
                image.crop((x1, 0, x2, height)).save(tile_path, quality=95)

                image_id = add_image(
                    coco,
                    relative_to_image_root(tile_path, image_root),
                    x2 - x1,
                    height,
                )
                before = len(coco["annotations"])
                for shape in data.get("shapes", []):
                    label = str(shape.get("label", "")).strip()
                    if label not in particle_labels:
                        continue
                    for polygon in clipped_shape_polygons(shape, x1, x2, height):
                        add_annotation(
                            coco,
                            image_id,
                            polygon,
                            min_area=min_area,
                            source_label=label,
                        )
                manifest_rows.append(
                    {
                        "dataset": "HX-ZA4720724102201",
                        "source_json": str(json_path),
                        "image_file": coco["images"][-1]["file_name"],
                        "mode": "generated_tile",
                        "annotations": len(coco["annotations"]) - before,
                    }
                )


def main() -> None:
    args = parse_args()
    gh_root = Path(args.gh_root)
    hx_root = Path(args.hx_root)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists. Use --overwrite to rebuild it.")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    manifest_rows: list[dict[str, Any]] = []

    add_gh_tiles(
        coco,
        manifest_rows,
        gh_root=gh_root,
        image_root=image_root,
        particle_labels=particle_labels,
        min_area=args.min_area,
        include_whole=bool(args.include_gh_whole),
    )
    add_hx_as_tiles(
        coco,
        manifest_rows,
        hx_root=hx_root,
        image_root=image_root,
        output_dir=output_dir,
        particle_labels=particle_labels,
        columns=args.hx_columns,
        overlap=args.hx_overlap,
        min_area=args.min_area,
    )
    if args.include_hx_whole:
        add_labelme_whole_images(
            coco,
            manifest_rows,
            dataset_name="HX-ZA4720724102201",
            root=hx_root,
            image_root=image_root,
            particle_labels=particle_labels,
            min_area=args.min_area,
        )

    ann_path = output_dir / "annotations.json"
    ann_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "source_json", "image_file", "mode", "annotations"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Boost round 1 GH/HX",
                "",
                "Training image root: `data`",
                "Training annotation: `data/boost_round1_gh_hx/annotations.json`",
                "",
                "- GH: existing column tiles only; whole images are skipped by default.",
                "- HX: full images are split into column tiles and polygons are clipped to each tile.",
                "- Fixed evaluation images are not included.",
                "",
                f"Images: {len(coco['images'])}",
                f"Annotations: {len(coco['annotations'])}",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "annotations": str(ann_path),
                "images": len(coco["images"]),
                "instances": len(coco["annotations"]),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
