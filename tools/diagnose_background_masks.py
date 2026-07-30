from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metal_powder_sem_ai.postprocess import _background_support_metrics
from metal_powder_sem_ai.preprocess import imwrite_unicode, preprocess_sem_image
from metal_powder_sem_ai.segment import (
    InstanceMask,
    MaskRCNNSegmenter,
    merge_instance_groups,
    paint_instance_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Mask R-CNN instances that may cover SEM background."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.62)
    parser.add_argument("--mask-threshold", type=float, default=0.68)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument("--max-detections", type=int, default=10000)
    parser.add_argument("--max-inference-side", type=int, default=1280)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.12)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def merged_instances(
    image_bgr: np.ndarray,
    segmenter: MaskRCNNSegmenter,
    tile_size: int,
    tile_overlap: int,
) -> list[InstanceMask]:
    whole = segmenter.segment(image_bgr)
    tiled = segmenter.segment_tiled(
        image_bgr,
        tile_size=tile_size,
        overlap=tile_overlap,
        ownership_filter=True,
    )
    return merge_instance_groups(
        whole,
        tiled,
        protect_parent_fragments=True,
        protected_parent_diameter_ratio=1.6,
        protected_parent_center_distance_factor=0.0,
    )


def instance_row(
    instance: InstanceMask,
    image_bgr: np.ndarray,
    gray: np.ndarray,
) -> dict[str, object]:
    canvas = np.zeros(gray.shape, dtype=np.uint8)
    paint_instance_mask(canvas, instance, value=1)
    mask = canvas > 0
    inside = gray[mask].astype(np.float32)
    area = int(mask.sum())
    x1, y1, x2, y2 = [int(value) for value in instance.bbox_xyxy]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    contour_area = float(cv2.contourArea(instance.contour))
    perimeter = float(cv2.arcLength(instance.contour, True))
    circularity = (
        4.0 * math.pi * contour_area / (perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )
    support = _background_support_metrics(instance, gray=gray, ring_width_px=6) or {}
    return {
        "particle_id": int(instance.particle_id),
        "source": instance.source,
        "score": float(instance.score),
        "bbox_xyxy": ",".join(str(value) for value in instance.bbox_xyxy),
        "tile_xyxy": (
            ",".join(str(value) for value in instance.tile_xyxy)
            if instance.tile_xyxy
            else ""
        ),
        "internal_tile_edge_sides": ",".join(
            instance.internal_tile_edge_sides or ()
        ),
        "area_px": area,
        "bbox_area_px": bbox_area,
        "bbox_fill_ratio": area / bbox_area,
        "contour_area_px": contour_area,
        "circularity": circularity,
        "inside_min": float(np.min(inside)) if inside.size else "",
        "inside_p10": float(np.percentile(inside, 10)) if inside.size else "",
        "inside_median": float(np.median(inside)) if inside.size else "",
        "inside_p90": float(np.percentile(inside, 90)) if inside.size else "",
        "inside_max": float(np.max(inside)) if inside.size else "",
        **support,
    }


def draw_diagnostic(
    image_bgr: np.ndarray,
    instances: Sequence[InstanceMask],
) -> np.ndarray:
    output = image_bgr.copy()
    for instance in instances:
        x1, y1, x2, y2 = [int(value) for value in instance.bbox_xyxy]
        color = (0, 0, 255) if instance.source == "whole" else (255, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        label = (
            f"#{instance.particle_id} {instance.source} "
            f"{'/'.join(instance.internal_tile_edge_sides or ())}"
        ).strip()
        cv2.putText(
            output,
            label,
            (x1 + 3, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["particle_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    pre = preprocess_sem_image(
        image,
        crop_bottom_fraction=float(args.crop_bottom_fraction),
        blur_ksize=int(args.blur_ksize),
    )
    device = None if args.device == "auto" else args.device
    segmenter = MaskRCNNSegmenter(
        weights_path=str(args.weights),
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        max_detections=int(args.max_detections),
        max_inference_side=int(args.max_inference_side),
        device=device,
    )
    instances = merged_instances(
        pre.image,
        segmenter,
        tile_size=int(args.tile_size),
        tile_overlap=int(args.tile_overlap),
    )
    gray = cv2.cvtColor(pre.image, cv2.COLOR_BGR2GRAY)
    rows = [instance_row(instance, pre.image, gray) for instance in instances]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mask_diagnostics.csv", rows)
    imwrite_unicode(
        args.output_dir / "mask_diagnostics.png",
        draw_diagnostic(pre.image, instances),
    )
    summary = {
        "image": str(args.image),
        "weights": str(args.weights),
        "instance_count": len(instances),
        "whole_count": sum(row["source"] == "whole" for row in rows),
        "tiled_count": sum(row["source"] == "tiled" for row in rows),
        "internal_tile_edge_count": sum(
            bool(row["internal_tile_edge_sides"]) for row in rows
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
