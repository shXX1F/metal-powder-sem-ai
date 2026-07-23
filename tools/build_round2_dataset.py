from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PARTICLE_LABELS = {"particle"}

H6_TRAIN_GROUPS = {"56-4", "56-5", "56-7", "56-8", "56-9", "56-10", "56-13"}
H6_VAL_GROUPS = {"56-3", "56-12", "56-14"}
H6_TEST_GROUPS = {"56-2"}

H1_TRAIN_GROUPS = {
    "92-2",
    "92-4",
    "92-5",
    "92-7",
    "92-8",
    "92-9",
    "92-10",
    "92-13",
    "92-14",
}
H1_VAL_GROUPS = {"92-3", "92-12"}
HX_VAL_GROUPS = {"114-3", "114-7"}

SOURCE_SELECTED = "selected_60_moved"
SOURCE_H1 = "GH3536_SA_H1"
SOURCE_HX = "HX_ZA"
SOURCE_H6 = "GH3536_CA_H6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the leakage-safe round-2 train/val/test dataset.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="data/round2_balanced_v1")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def empty_coco() -> dict[str, Any]:
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "particle", "supercategory": "particle"}],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_labelme(path: Path) -> dict[str, Any] | None:
    try:
        data = load_json(path)
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        return None
    return data


def iter_labelme(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(root.rglob("*.json")):
        data = load_labelme(path)
        if data is not None:
            yield path, data


def find_labelme_image(json_path: Path, data: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    image_text = data.get("imagePath")
    if image_text:
        image_path = Path(str(image_text))
        candidates.extend([image_path, json_path.parent / image_path])
    candidates.extend(json_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {json_path}")


def find_coco_image(dataset_root: Path, ann_path: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = [
        dataset_root / relative,
        ann_path.parent / relative,
        ann_path.parent / "images" / relative.name,
        dataset_root / "images" / relative.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [
        path
        for path in dataset_root.rglob(relative.name)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot resolve COCO image {file_name!r} under {dataset_root}; matches={len(matches)}"
    )


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        twice_area += x1 * y2 - x2 * y1
    return abs(twice_area) / 2.0


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_points(shape: dict[str, Any]) -> list[list[float]]:
    points = [[float(x), float(y)] for x, y in shape.get("points", [])]
    if shape.get("shape_type") == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return points


def labelme_annotations(data: dict[str, Any], min_area: float) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label not in PARTICLE_LABELS:
            continue
        points = normalize_points(shape)
        if len(points) < 3:
            continue
        area = polygon_area(points)
        flat = [coordinate for point in points for coordinate in point]
        if area < min_area or any(not math.isfinite(value) for value in flat):
            continue
        annotations.append(
            {
                "category_id": 1,
                "segmentation": [flat],
                "bbox": polygon_bbox(points),
                "area": area,
                "iscrowd": 0,
            }
        )
    return annotations


class DatasetWriter:
    def __init__(
        self,
        split_dir: Path,
        fixed_hashes: set[str],
        min_area: float,
    ) -> None:
        self.split_dir = split_dir
        self.image_dir = split_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_hashes = fixed_hashes
        self.min_area = min_area
        self.coco = empty_coco()
        self.manifest: list[dict[str, Any]] = []
        self._used_names: set[str] = set()

    def _copy_image(self, source_path: Path, target_name: str) -> tuple[Path, str]:
        source_hash = sha256(source_path)
        if source_hash in self.fixed_hashes:
            raise RuntimeError(
                f"Fixed-evaluation leakage blocked: {source_path} has hash {source_hash}"
            )
        target_name = safe_name(target_name)
        if target_name in self._used_names:
            raise RuntimeError(f"Duplicate output image name: {target_name}")
        self._used_names.add(target_name)
        target_path = self.image_dir / target_name
        shutil.copy2(source_path, target_path)
        return target_path, source_hash

    def add(
        self,
        source_path: Path,
        target_name: str,
        width: int,
        height: int,
        annotations: list[dict[str, Any]],
        source_dataset: str,
        source_group: str,
        source_annotation: str,
    ) -> None:
        target_path, source_hash = self._copy_image(source_path, target_name)
        image_id = len(self.coco["images"]) + 1
        self.coco["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{target_path.name}",
                "width": int(width),
                "height": int(height),
                "source_dataset": source_dataset,
                "source_group": source_group,
            }
        )
        before = len(self.coco["annotations"])
        for annotation in annotations:
            item = dict(annotation)
            item["id"] = len(self.coco["annotations"]) + 1
            item["image_id"] = image_id
            item["category_id"] = 1
            item.setdefault("iscrowd", 0)
            self.coco["annotations"].append(item)
        self.manifest.append(
            {
                "split": self.split_dir.name,
                "source_dataset": source_dataset,
                "source_group": source_group,
                "source_image": str(source_path),
                "source_annotation": source_annotation,
                "output_image": f"images/{target_path.name}",
                "sha256": source_hash,
                "instances": len(self.coco["annotations"]) - before,
            }
        )

    def add_labelme(
        self,
        json_path: Path,
        data: dict[str, Any],
        prefix: str,
        source_dataset: str,
        source_group: str,
    ) -> None:
        image_path = find_labelme_image(json_path, data)
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = labelme_annotations(data, self.min_area)
        self.add(
            source_path=image_path,
            target_name=f"{prefix}__{image_path.name}",
            width=width,
            height=height,
            annotations=annotations,
            source_dataset=source_dataset,
            source_group=source_group,
            source_annotation=str(json_path),
        )

    def write(self) -> None:
        annotation_path = self.split_dir / "annotations.json"
        annotation_path.write_text(
            json.dumps(self.coco, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest_path = self.split_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
            fieldnames = [
                "split",
                "source_dataset",
                "source_group",
                "source_image",
                "source_annotation",
                "output_image",
                "sha256",
                "instances",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.manifest)


def add_selected_coco(
    writer: DatasetWriter,
    dataset_root: Path,
    annotation_path: Path,
    prefix: str,
) -> None:
    coco = load_json(annotation_path)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    for image_info in sorted(coco.get("images", []), key=lambda item: int(item["id"])):
        old_id = int(image_info["id"])
        source_path = find_coco_image(
            dataset_root,
            annotation_path,
            str(image_info["file_name"]),
        )
        copied_annotations: list[dict[str, Any]] = []
        for annotation in annotations_by_image.get(old_id, []):
            copied_annotations.append(
                {
                    "category_id": 1,
                    "segmentation": annotation.get("segmentation", []),
                    "bbox": annotation.get("bbox", [0, 0, 0, 0]),
                    "area": float(annotation.get("area", 0.0)),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
        group = Path(str(image_info["file_name"])).stem
        writer.add(
            source_path=source_path,
            target_name=f"{prefix}__{source_path.name}",
            width=int(image_info["width"]),
            height=int(image_info["height"]),
            annotations=copied_annotations,
            source_dataset=SOURCE_SELECTED,
            source_group=group,
            source_annotation=str(annotation_path),
        )


def group_from_stem(stem: str, prefix: str) -> str | None:
    match = re.match(rf"^({re.escape(prefix)}-\d+)", stem)
    return match.group(1) if match else None


def add_h1(train: DatasetWriter, val: DatasetWriter, root: Path) -> None:
    for json_path, data in iter_labelme(root):
        group = group_from_stem(json_path.stem, "92")
        if group is None:
            continue
        is_tile = "__col" in json_path.stem
        if group in H1_VAL_GROUPS:
            if is_tile:
                val.add_labelme(json_path, data, "h1", SOURCE_H1, group)
            continue
        if group not in H1_TRAIN_GROUPS:
            continue
        if group == "92-2":
            if not is_tile:
                train.add_labelme(json_path, data, "h1", SOURCE_H1, group)
        elif is_tile:
            train.add_labelme(json_path, data, "h1", SOURCE_H1, group)


def add_hx(train: DatasetWriter, val: DatasetWriter, root: Path) -> None:
    for json_path, data in iter_labelme(root):
        group = group_from_stem(json_path.stem, "114")
        if group is None:
            continue
        target = val if group in HX_VAL_GROUPS else train
        target.add_labelme(json_path, data, "hx", SOURCE_HX, group)


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    final_position = length - tile_size
    if not positions or positions[-1] != final_position:
        positions.append(final_position)
    return sorted(set(positions))


def mask_segmentations(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 0.5, True)
        if len(approx) < 3:
            continue
        segmentations.append(approx.reshape(-1, 2).astype(float).flatten().tolist())
    return segmentations


def add_h6_dense_tiles(
    writer: DatasetWriter,
    json_path: Path,
    data: dict[str, Any],
    tile_size: int,
    overlap: int,
    min_visible_ratio: float,
) -> None:
    image_path = find_labelme_image(json_path, data)
    with Image.open(image_path) as source_image:
        source_image = source_image.convert("RGB")
        width, height = source_image.size
        spans = [
            (x, y, min(width, x + tile_size), min(height, y + tile_size))
            for y in tile_positions(height, tile_size, overlap)
            for x in tile_positions(width, tile_size, overlap)
        ]

        annotations_by_tile: list[list[dict[str, Any]]] = [[] for _ in spans]
        for shape in data.get("shapes", []):
            if str(shape.get("label", "")).strip() not in PARTICLE_LABELS:
                continue
            points = normalize_points(shape)
            if len(points) < 3:
                continue
            full_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(
                full_mask,
                [np.round(np.asarray(points, dtype=np.float32)).astype(np.int32)],
                1,
            )
            full_area = int(full_mask.sum())
            if full_area < writer.min_area:
                continue

            best_index = -1
            best_area = 0
            best_mask: np.ndarray | None = None
            for tile_index, (x1, y1, x2, y2) in enumerate(spans):
                tile_mask = full_mask[y1:y2, x1:x2]
                visible_area = int(tile_mask.sum())
                if visible_area > best_area:
                    best_index = tile_index
                    best_area = visible_area
                    best_mask = tile_mask.copy()
            if best_index < 0 or best_mask is None:
                continue
            if best_area / full_area < min_visible_ratio:
                continue
            segmentations = mask_segmentations(best_mask)
            if not segmentations:
                continue
            ys, xs = np.where(best_mask > 0)
            bbox = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ]
            annotations_by_tile[best_index].append(
                {
                    "category_id": 1,
                    "segmentation": segmentations,
                    "bbox": bbox,
                    "area": float(best_area),
                    "iscrowd": 0,
                }
            )

        for tile_index, (x1, y1, x2, y2) in enumerate(spans, start=1):
            tile_name = (
                f"h6__56-4__tile{tile_index:02d}_"
                f"x{x1:04d}-{x2 - 1:04d}_y{y1:04d}-{y2 - 1:04d}.png"
            )
            temporary_path = writer.split_dir / tile_name
            source_image.crop((x1, y1, x2, y2)).save(temporary_path)
            try:
                writer.add(
                    source_path=temporary_path,
                    target_name=tile_name,
                    width=x2 - x1,
                    height=y2 - y1,
                    annotations=annotations_by_tile[tile_index - 1],
                    source_dataset=SOURCE_H6,
                    source_group="56-4",
                    source_annotation=str(json_path),
                )
            finally:
                temporary_path.unlink(missing_ok=True)


def add_h6(
    train: DatasetWriter,
    val: DatasetWriter,
    root: Path,
    tile_size: int,
    overlap: int,
    min_visible_ratio: float,
) -> None:
    dense_source: tuple[Path, dict[str, Any]] | None = None
    for json_path, data in iter_labelme(root):
        group = group_from_stem(json_path.stem, "56")
        if group is None or group in H6_TEST_GROUPS:
            continue
        if group == "56-4":
            dense_source = (json_path, data)
            continue
        if group in H6_TRAIN_GROUPS:
            train.add_labelme(json_path, data, "h6", SOURCE_H6, group)
        elif group in H6_VAL_GROUPS:
            val.add_labelme(json_path, data, "h6", SOURCE_H6, group)
    if dense_source is None:
        raise RuntimeError("Cannot find the H6 dense source 56-4.json")
    add_h6_dense_tiles(
        train,
        dense_source[0],
        dense_source[1],
        tile_size=tile_size,
        overlap=overlap,
        min_visible_ratio=min_visible_ratio,
    )


def copy_fixed_test(fixed_root: Path, test_dir: Path) -> list[dict[str, Any]]:
    image_dir = test_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for image_path in sorted((fixed_root / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = image_dir / image_path.name
        shutil.copy2(image_path, target)
        manifest_rows.append(
            {
                "split": "test",
                "source_dataset": "fixed_eval_20",
                "source_group": image_path.stem,
                "source_image": str(image_path),
                "source_annotation": "reference_particles.csv",
                "output_image": f"images/{target.name}",
                "sha256": sha256(target),
                "instances": "reference",
            }
        )
    for name in (
        "manifest.csv",
        "reference_particles.csv",
        "exclude_from_training.txt",
        "README.md",
    ):
        source = fixed_root / name
        if source.exists():
            shutil.copy2(source, test_dir / name)
    with (test_dir / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows


def assert_no_hash_overlap(split_manifests: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, rows in split_manifests.items():
        for row in rows:
            owners[str(row["sha256"])].append((split, str(row["output_image"])))
    overlaps = {
        digest: entries
        for digest, entries in owners.items()
        if len({split for split, _ in entries}) > 1
    }
    if overlaps:
        details = "\n".join(f"{digest}: {entries}" for digest, entries in overlaps.items())
        raise RuntimeError(f"Image leakage across splits:\n{details}")


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_images = Counter(str(row["source_dataset"]) for row in rows)
    source_instances: Counter[str] = Counter()
    for row in rows:
        try:
            source_instances[str(row["source_dataset"])] += int(row["instances"])
        except (TypeError, ValueError):
            continue
    return {
        "images": len(rows),
        "instances": sum(source_instances.values()),
        "images_by_source": dict(sorted(source_images.items())),
        "instances_by_source": dict(sorted(source_instances.items())),
    }


def write_root_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# Round 2 balanced dataset",
            "",
            "- `train/`: COCO training set with source metadata.",
            "- `val/`: independent COCO validation set grouped by original SEM view.",
            "- `test/`: frozen 20-image statistical evaluation set; never train on it.",
            "- `dataset_summary.json`: exact image and instance counts.",
            "",
            "The H6 image `56-2.tiff` is excluded because it is byte-identical to",
            "`fixed_eval_20/images/11_56-2.tiff`.",
            "",
            "Recommended source weights:",
            "",
            "`selected_60_moved=0.40,GH3536_CA_H6=0.30,GH3536_SA_H1=0.20,HX_ZA=0.10`",
            "",
            f"Summary: `{json.dumps(summary, ensure_ascii=False)}`",
            "",
        ]
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_root = data_root / "fixed_eval_20"
    fixed_hashes = {
        sha256(path)
        for path in (fixed_root / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    }

    train = DatasetWriter(output_dir / "train", fixed_hashes, args.min_area)
    val = DatasetWriter(output_dir / "val", fixed_hashes, args.min_area)

    selected_root = data_root / "selected_60_moved"
    add_selected_coco(
        train,
        selected_root,
        selected_root / "trainval_annotations.json",
        "selected",
    )
    add_selected_coco(
        val,
        selected_root,
        selected_root / "test" / "annotations.json",
        "selected",
    )
    add_h1(train, val, data_root / "GH3536-SA0120724061801-H1")
    add_hx(train, val, data_root / "HX-ZA4720724102201")
    add_h6(
        train,
        val,
        data_root / "GH3536-CA0121223091101-H6",
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        min_visible_ratio=args.min_visible_ratio,
    )

    train.write()
    val.write()
    test_rows = copy_fixed_test(fixed_root, output_dir / "test")
    split_manifests = {
        "train": train.manifest,
        "val": val.manifest,
        "test": test_rows,
    }
    assert_no_hash_overlap(split_manifests)

    summary = {
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_visible_ratio": args.min_visible_ratio,
        "train": split_summary(train.manifest),
        "val": split_summary(val.manifest),
        "test": split_summary(test_rows),
        "leakage_check": "passed",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_root_readme(output_dir, summary)
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
