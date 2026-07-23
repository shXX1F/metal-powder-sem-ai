from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metal_powder_sem_ai.preprocess import imread_unicode, imwrite_unicode
from metal_powder_sem_ai.train import coco_ann_to_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render side-by-side previews for checking COCO annotation coverage."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory relative to which COCO file_name values are resolved.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-per-source", type=int, default=2)
    parser.add_argument(
        "--image-id",
        type=int,
        action="append",
        default=[],
        help="Render an explicit COCO image id; may be repeated.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def select_images(
    images: List[Dict], annotation_counts: Counter, top_per_source: int, image_ids: List[int]
) -> List[Dict]:
    by_id = {int(image["id"]): image for image in images}
    selected: List[Dict] = []
    seen = set()

    for image_id in image_ids:
        if image_id in by_id and image_id not in seen:
            selected.append(by_id[image_id])
            seen.add(image_id)

    if top_per_source <= 0:
        return selected

    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for image in images:
        by_source[str(image.get("source_dataset", "default"))].append(image)
    for source in sorted(by_source):
        ranked = sorted(
            by_source[source],
            key=lambda item: annotation_counts[int(item["id"])],
            reverse=True,
        )
        for image in ranked[:top_per_source]:
            image_id = int(image["id"])
            if image_id not in seen:
                selected.append(image)
                seen.add(image_id)
    return selected


def render_one(image: np.ndarray, annotations: List[Dict]) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    overlay = image.copy()
    union = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        mask = coco_ann_to_mask(annotation, height=height, width=width)
        if not np.any(mask):
            continue
        union |= mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)

    preview = np.concatenate([image, overlay], axis=1)
    coverage = float(union.mean())
    return preview, coverage


def main() -> None:
    args = parse_args()
    with args.annotations.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = list(coco.get("images", []))
    annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    counts = Counter(
        {image_id: len(items) for image_id, items in annotations_by_image.items()}
    )
    selected = select_images(images, counts, args.top_per_source, args.image_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for image_info in selected:
        image_id = int(image_info["id"])
        image_path = args.image_root / str(image_info["file_name"])
        image = imread_unicode(image_path)
        annotations = annotations_by_image.get(image_id, [])
        preview, coverage = render_one(image, annotations)
        source = str(image_info.get("source_dataset", "default"))
        output_name = f"{safe_name(source)}__id{image_id}__{safe_name(image_path.stem)}.png"
        output_path = args.output_dir / output_name
        imwrite_unicode(output_path, preview)
        summary_rows.append(
            {
                "source_dataset": source,
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "annotation_count": len(annotations),
                "annotated_area_fraction": coverage,
                "preview_path": str(output_path),
            }
        )
        print(
            f"rendered {source} image_id={image_id} "
            f"annotations={len(annotations)} coverage={coverage:.4f}"
        )

    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
