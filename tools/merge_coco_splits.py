from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split COCO annotation files without copying images.",
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def merge_coco_splits(base_dir: Path, splits: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": None,
    }
    next_image_id = 1
    next_ann_id = 1

    for split in splits:
        split = split.strip()
        if not split:
            continue
        ann_path = base_dir / split / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(ann_path)

        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        if merged["categories"] is None:
            merged["categories"] = coco.get("categories", [])

        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            old_image_id = int(image["id"])
            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = (
                Path(split) / "images" / str(image["file_name"])
            ).as_posix()
            merged["images"].append(new_image)
            image_id_map[old_image_id] = next_image_id
            next_image_id += 1

        for ann in coco.get("annotations", []):
            old_image_id = int(ann["image_id"])
            if old_image_id not in image_id_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[old_image_id]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

    if merged["categories"] is None:
        merged["categories"] = []
    return merged


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    merged = merge_coco_splits(base_dir, splits)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {output}: "
        f"{len(merged['images'])} images, "
        f"{len(merged['annotations'])} annotations."
    )


if __name__ == "__main__":
    main()
