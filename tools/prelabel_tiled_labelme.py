from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from metal_powder_sem_ai.preprocess import imread_unicode, imwrite_unicode, preprocess_sem_image
from metal_powder_sem_ai.segment import ClassicalParticleSegmenter, MaskRCNNSegmenter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Candidate:
    contour: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: float
    edge_clearance: float
    tile_xyxy: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LabelMe starter annotations by tiled prelabel + stitching."
    )
    parser.add_argument(
        "--image-root",
        default="data/selected_60_moved/val/images",
        help="Directory containing images to prelabel.",
    )
    parser.add_argument(
        "--labelme-dir",
        default=None,
        help="Where to write LabelMe JSON files. Defaults to --image-root.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=896,
        help="Square tile size in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Minimum preferred overlap between neighboring tiles in pixels.",
    )
    parser.add_argument(
        "--crop-bottom-fraction",
        type=float,
        default=0.09,
        help="Bottom fraction to exclude from detection for SEM info bars.",
    )
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--peak-min-distance", type=int, default=12)
    parser.add_argument("--fill-small-holes-px", type=int, default=64)
    parser.add_argument(
        "--segmenter",
        choices=["classical", "maskrcnn"],
        default="classical",
    )
    parser.add_argument("--weights", default=None, help="Mask R-CNN weights for --segmenter maskrcnn.")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--simplify-epsilon-ratio", type=float, default=0.006)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.25)
    parser.add_argument("--nms-overlap-threshold", type=float, default=0.70)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing LabelMe JSON files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose LabelMe JSON already exists.",
    )
    parser.add_argument(
        "--preview-dir",
        default="runs/tiled_val_prelabel/overlays",
        help="Directory for contour overlay preview PNGs. Empty string disables previews.",
    )
    parser.add_argument(
        "--summary",
        default="runs/tiled_val_prelabel/summary.csv",
        help="CSV summary path. Empty string disables summary.",
    )
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    count = max(2, math.ceil((length - overlap) / stride))
    starts = np.linspace(0, length - tile_size, count)
    return sorted({int(round(value)) for value in starts})


def ownership_intervals(starts: Sequence[int], length: int, tile_size: int) -> list[tuple[float, float]]:
    ends = [min(length, start + tile_size) for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    intervals: list[tuple[float, float]] = []
    for idx, center in enumerate(centers):
        left = 0.0 if idx == 0 else (centers[idx - 1] + center) / 2.0
        right = float(length) if idx == len(centers) - 1 else (center + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def contour_bbox_xyxy(contour: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(np.round(contour).astype(np.int32))
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))
    return x1, y1, x2, y2


def contour_centroid(contour: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def tile_edge_clearance(
    bbox_xyxy: tuple[int, int, int, int],
    tile_xyxy: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    tx1, ty1, tx2, ty2 = tile_xyxy
    return float(min(x1 - tx1, y1 - ty1, tx2 - x2, ty2 - y2))


def build_segmenter(args: argparse.Namespace):
    if args.segmenter == "classical":
        return ClassicalParticleSegmenter(
            min_area_px=args.min_area_px,
            peak_min_distance=args.peak_min_distance,
            fill_small_holes_px=args.fill_small_holes_px,
        )
    if not args.weights:
        raise ValueError("--weights is required when --segmenter maskrcnn")
    return MaskRCNNSegmenter(
        weights_path=args.weights,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
    )


def collect_candidates(image: np.ndarray, args: argparse.Namespace) -> tuple[list[Candidate], int, int]:
    height, width = image.shape[:2]
    active_height = int(round(height * (1.0 - args.crop_bottom_fraction)))
    active_height = max(1, min(height, active_height))
    active = image[:active_height, :]

    x_starts = axis_starts(width, args.tile_size, args.overlap)
    y_starts = axis_starts(active_height, args.tile_size, args.overlap)
    x_ownership = ownership_intervals(x_starts, width, args.tile_size)
    y_ownership = ownership_intervals(y_starts, active_height, args.tile_size)
    segmenter = build_segmenter(args)

    candidates: list[Candidate] = []
    for y_idx, y1 in enumerate(y_starts):
        y2 = min(active_height, y1 + args.tile_size)
        core_y1, core_y2 = y_ownership[y_idx]
        for x_idx, x1 in enumerate(x_starts):
            x2 = min(width, x1 + args.tile_size)
            core_x1, core_x2 = x_ownership[x_idx]
            tile = active[y1:y2, x1:x2]

            if args.segmenter == "classical":
                pre = preprocess_sem_image(tile, crop_bottom_fraction=0.0)
                instances = segmenter.segment(pre.binary)
            else:
                instances = segmenter.segment(tile)

            tile_xyxy = (x1, y1, x2, y2)
            for instance in instances:
                contour = instance.contour.astype(np.float32).copy()
                contour[:, 0, 0] += x1
                contour[:, 0, 1] += y1
                area = float(cv2.contourArea(contour))
                if area < args.min_area_px:
                    continue
                bbox = contour_bbox_xyxy(contour, width=width, height=height)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                cx, cy = contour_centroid(contour, bbox)
                if not (core_x1 <= cx < core_x2 and core_y1 <= cy < core_y2):
                    continue
                candidates.append(
                    Candidate(
                        contour=contour,
                        bbox_xyxy=bbox,
                        area=area,
                        edge_clearance=tile_edge_clearance(bbox, tile_xyxy),
                        tile_xyxy=tile_xyxy,
                    )
                )
    return candidates, len(x_starts), len(y_starts)


def bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_overlap_metrics(a: Candidate, b: Candidate) -> tuple[float, float]:
    intersection_bbox = bbox_intersection(a.bbox_xyxy, b.bbox_xyxy)
    if intersection_bbox is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = intersection_bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0, 0.0

    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    pts_a = np.round(a.contour.reshape(-1, 2) - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    pts_b = np.round(b.contour.reshape(-1, 2) - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask_a, [pts_a], 1)
    cv2.fillPoly(mask_b, [pts_b], 1)
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0, 0.0
    area_a = max(1, int(round(a.area)))
    area_b = max(1, int(round(b.area)))
    union = area_a + area_b - inter
    iou = inter / max(1, union)
    overlap_smaller = inter / max(1, min(area_a, area_b))
    return float(iou), float(overlap_smaller)


def dedupe_candidates(candidates: Sequence[Candidate], args: argparse.Namespace) -> list[Candidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.edge_clearance, item.area),
        reverse=True,
    )
    kept: list[Candidate] = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            intersection_bbox = bbox_intersection(candidate.bbox_xyxy, existing.bbox_xyxy)
            if intersection_bbox is None:
                continue
            ix1, iy1, ix2, iy2 = intersection_bbox
            bbox_inter_area = (ix2 - ix1) * (iy2 - iy1)
            if bbox_inter_area / max(1.0, min(candidate.area, existing.area)) < 0.05:
                continue
            iou, overlap_smaller = mask_overlap_metrics(candidate, existing)
            if iou >= args.nms_iou_threshold or overlap_smaller >= args.nms_overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))


def contour_to_polygon(
    contour: np.ndarray,
    width: int,
    height: int,
    max_points: int,
    epsilon_ratio: float,
) -> list[list[float]]:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(0.8, epsilon_ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def write_labelme_json(
    json_path: Path,
    image_path: Path,
    image_shape: tuple[int, int],
    candidates: Sequence[Candidate],
    args: argparse.Namespace,
) -> None:
    height, width = image_shape
    shapes = []
    for candidate in candidates:
        polygon = contour_to_polygon(
            candidate.contour,
            width=width,
            height=height,
            max_points=args.max_points,
            epsilon_ratio=args.simplify_epsilon_ratio,
        )
        if len(polygon) < 3:
            continue
        shapes.append(
            {
                "label": "particle",
                "points": polygon,
                "group_id": None,
                "description": "auto-generated by tiled prelabel; please review",
                "shape_type": "polygon",
                "flags": {
                    "auto_label": True,
                    "tiled_prelabel": True,
                },
            }
        )

    data = {
        "version": "5.10.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_preview(image: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
    preview = image.copy()
    for candidate in candidates:
        contour = np.round(candidate.contour).astype(np.int32)
        cv2.drawContours(preview, [contour], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    return preview


def write_summary(summary_path: Path, rows: Iterable[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "width",
        "height",
        "tiles_x",
        "tiles_y",
        "raw_candidates",
        "final_shapes",
        "json_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    labelme_dir = Path(args.labelme_dir) if args.labelme_dir else image_root
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    summary_path = Path(args.summary) if args.summary else None

    images = image_files(image_root)
    if not images:
        raise FileNotFoundError(f"No images found in {image_root}")

    rows: list[dict[str, object]] = []
    total_shapes = 0
    for image_path in images:
        json_path = labelme_dir / f"{image_path.stem}.json"
        if json_path.exists() and args.skip_existing:
            print(f"{image_path.name}: skipped existing JSON", flush=True)
            continue
        if json_path.exists() and not args.overwrite:
            raise FileExistsError(f"{json_path} exists. Pass --overwrite to replace it.")

        image = imread_unicode(image_path)
        candidates, tiles_x, tiles_y = collect_candidates(image, args)
        stitched = dedupe_candidates(candidates, args)
        write_labelme_json(
            json_path=json_path,
            image_path=image_path,
            image_shape=image.shape[:2],
            candidates=stitched,
            args=args,
        )

        if preview_dir is not None:
            preview = draw_preview(image, stitched)
            imwrite_unicode(preview_dir / f"{image_path.stem}.preview.png", preview)

        total_shapes += len(stitched)
        rows.append(
            {
                "image": image_path.name,
                "width": image.shape[1],
                "height": image.shape[0],
                "tiles_x": tiles_x,
                "tiles_y": tiles_y,
                "raw_candidates": len(candidates),
                "final_shapes": len(stitched),
                "json_path": str(json_path),
            }
        )
        print(f"{image_path.name}: {len(candidates)} raw -> {len(stitched)} stitched", flush=True)

    if summary_path is not None:
        write_summary(summary_path, rows)
    print(f"Wrote {len(images)} LabelMe JSON files with {total_shapes} shapes.")


if __name__ == "__main__":
    main()
