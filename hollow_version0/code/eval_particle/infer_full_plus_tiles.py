"""particle 整图 + 6 张重叠小图的融合评估脚本。

设计目标：

1. 整图推理负责完整上下文，作为主结果。
2. 重叠 tile 推理负责补充小颗粒或整图漏检。
3. tile 结果只有在和已有结果交集极小时才加入，避免同一颗粒被重复圈。
4. 所有预测最终都映射回原图坐标，再统一计算 mask IoU 指标和预览图。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


DEFAULT_LABEL = "particle"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


@dataclass
class Prediction:
    mask: np.ndarray
    score: float
    source: str
    tile_index: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    area: int = 0


def natural_key(text):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_image_for_stem(src_dir: str | Path, stem: str) -> Path | None:
    src_dir = Path(src_dir)
    for ext in IMAGE_EXTS:
        path = src_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def image_ids(src_dir: str | Path) -> list[str]:
    src_dir = Path(src_dir)
    stems = []
    for json_path in sorted(src_dir.glob("*.json"), key=lambda p: natural_key(p.name)):
        if find_image_for_stem(src_dir, json_path.stem) is not None:
            stems.append(json_path.stem)
    return stems


def polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=np.float32)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def bbox_from_points(points) -> list[float]:
    arr = np.asarray(points, dtype=np.float32)
    x_min = float(arr[:, 0].min())
    y_min = float(arr[:, 1].min())
    x_max = float(arr[:, 0].max())
    y_max = float(arr[:, 1].max())
    return [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)]


def polygon_to_mask(points, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon([tuple(p) for p in points], outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def labelme_to_coco(src_dir: Path, ids: list[str], out_json: Path, split_dir: Path, label_name: str) -> dict:
    images = []
    annotations = []
    ann_id = 1
    split_dir.mkdir(parents=True, exist_ok=True)
    for image_index, stem in enumerate(ids, start=1):
        img_path = find_image_for_stem(src_dir, stem)
        if img_path is None:
            continue
        json_path = src_dir / f"{stem}.json"
        obj = read_json(json_path)
        width = int(obj["imageWidth"])
        height = int(obj["imageHeight"])
        images.append({"id": image_index, "file_name": img_path.name, "width": width, "height": height})
        shutil.copy2(img_path, split_dir / img_path.name)
        shutil.copy2(json_path, split_dir / json_path.name)
        for shape in obj.get("shapes", []):
            if shape.get("label") != label_name:
                continue
            points = shape.get("points", [])
            area = polygon_area(points)
            if area <= 1:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": [[coord for point in points for coord in point]],
                    "area": area,
                    "bbox": bbox_from_points(points),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": label_name, "supercategory": label_name}],
    }
    write_json(out_json, coco)
    return coco


def parse_anchor_sizes(text: str):
    values = tuple(int(part) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("anchor_sizes must contain at least one integer")
    return tuple((value,) for value in values)


def build_model(model_min_size: int, model_max_size: int, anchor_sizes: str, detections_per_img: int):
    parsed_anchor_sizes = parse_anchor_sizes(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=parsed_anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(parsed_anchor_sizes),
    )
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=model_min_size,
        max_size=model_max_size,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=detections_per_img,
    )
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_gt_masks(src_dir: Path, stem: str, label_name: str) -> tuple[list[np.ndarray], tuple[int, int]]:
    obj = read_json(src_dir / f"{stem}.json")
    width = int(obj["imageWidth"])
    height = int(obj["imageHeight"])
    masks = []
    for shape in obj.get("shapes", []):
        if shape.get("label") != label_name:
            continue
        mask = polygon_to_mask(shape.get("points", []), width, height)
        if mask.sum() > 0:
            masks.append(mask.astype(bool))
    return masks, (width, height)


def make_tiles(width: int, height: int, cols: int, rows: int, tile_w: int, tile_h: int) -> list[tuple[int, int, int, int]]:
    """生成固定数量的重叠 tile，首尾贴住原图边界，中间均匀分布。"""
    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)
    xs = [0] if cols <= 1 else [round(i * (width - tile_w) / (cols - 1)) for i in range(cols)]
    ys = [0] if rows <= 1 else [round(i * (height - tile_h) / (rows - 1)) for i in range(rows)]
    tiles = []
    seen = set()
    for y in ys:
        for x in xs:
            tile = (int(x), int(y), int(x + tile_w), int(y + tile_h))
            if tile not in seen:
                seen.add(tile)
                tiles.append(tile)
    return tiles


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_prediction(mask: np.ndarray, score: float, source: str, tile_index: int | None = None) -> Prediction:
    mask = mask.astype(bool)
    bbox = mask_bbox(mask)
    area = int(mask.sum())
    return Prediction(mask=mask, score=float(score), source=source, tile_index=tile_index, bbox=bbox, area=area)


def bboxes_intersect(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def cropped_mask_iou(a: np.ndarray, b: np.ndarray, bbox_a, bbox_b) -> float:
    x1 = min(bbox_a[0], bbox_b[0])
    y1 = min(bbox_a[1], bbox_b[1])
    x2 = max(bbox_a[2], bbox_b[2])
    y2 = max(bbox_a[3], bbox_b[3])
    a_crop = a[y1:y2, x1:x2]
    b_crop = b[y1:y2, x1:x2]
    inter = np.logical_and(a_crop, b_crop).sum()
    union = np.logical_or(a_crop, b_crop).sum()
    return float(inter / union) if union else 0.0


def cropped_overlap_ratio(candidate: np.ndarray, existing: np.ndarray, bbox_candidate, bbox_existing, candidate_area: int) -> float:
    """只在 bbox 交叠区域内计算 candidate 被 existing 覆盖的面积占比。"""
    if candidate_area <= 0:
        return 1.0
    x1 = max(bbox_candidate[0], bbox_existing[0])
    y1 = max(bbox_candidate[1], bbox_existing[1])
    x2 = min(bbox_candidate[2], bbox_existing[2])
    y2 = min(bbox_candidate[3], bbox_existing[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = np.logical_and(candidate[y1:y2, x1:x2], existing[y1:y2, x1:x2]).sum()
    return float(inter / candidate_area)


def max_overlap_with_existing(candidate: np.ndarray, existing_preds: list[Prediction]) -> tuple[float, float]:
    """返回 candidate 与已有预测的最大交集占比和最大 mask IoU。"""
    bbox_c = mask_bbox(candidate)
    if bbox_c is None:
        return 1.0, 1.0
    area_c = int(candidate.sum())
    max_intersection_ratio = 0.0
    max_iou = 0.0
    for pred in existing_preds:
        bbox_e = pred.bbox
        if bbox_e is None or not bboxes_intersect(bbox_c, bbox_e):
            continue
        max_intersection_ratio = max(
            max_intersection_ratio,
            cropped_overlap_ratio(candidate, pred.mask, bbox_c, bbox_e, area_c),
        )
        max_iou = max(max_iou, cropped_mask_iou(candidate, pred.mask, bbox_c, bbox_e))
    return max_intersection_ratio, max_iou


def bbox_touches_image_boundary(bbox, image_width: int, image_height: int, margin: int) -> bool:
    """Return True when a tile-added object touches or nearly touches the full image edge.

    This filter is only applied to tile predictions after they are pasted back
    into full-image coordinates. It removes the entire particle when the object
    is cut by the original image boundary.
    """
    margin = max(0, int(margin))
    return (
        bbox[0] <= margin
        or bbox[1] <= margin
        or bbox[2] >= image_width - margin
        or bbox[3] >= image_height - margin
    )


def bbox_inside_image_safe_region(mask: np.ndarray, image_width: int, image_height: int, margin: int) -> bool:
    """Filter tile predictions that touch the original full-image boundary.

    Tile boundaries are handled separately by overlapping crops. This function
    only removes tile-added particles that are clipped by the outer boundary of
    the original image.
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return False
    return not bbox_touches_image_boundary(bbox, image_width, image_height, margin)


def bbox_inside_tile_safe_region(mask: np.ndarray, tile: tuple[int, int, int, int], margin: int) -> bool:
    """Keep the previous near-tile-boundary safety band without hard-removing every tile-edge touch."""
    bbox = mask_bbox(mask)
    if bbox is None:
        return False
    if margin <= 0:
        return True
    x0, y0, x1, y1 = tile
    return bbox[0] > x0 + margin and bbox[1] > y0 + margin and bbox[2] < x1 - margin and bbox[3] < y1 - margin


@torch.no_grad()
def predict_on_image(model, device, image: Image.Image, score_threshold: float, mask_threshold: float) -> list[Prediction]:
    tensor = image_to_tensor(image).to(device)
    output = model([tensor])[0]
    scores = output["scores"].detach().cpu().numpy()
    keep = scores >= score_threshold
    masks = output["masks"].detach().cpu()[keep, 0].numpy() >= mask_threshold
    kept_scores = scores[keep]
    return [make_prediction(m, float(s), "full") for m, s in zip(masks, kept_scores)]


@torch.no_grad()
def predict_on_images(
    model,
    device,
    images: list[Image.Image],
    score_thresholds: list[float],
    mask_threshold: float,
    sources: list[str],
) -> list[list[Prediction]]:
    """Run one batched Mask R-CNN forward for the full image plus all tile views."""
    tensors = [image_to_tensor(image).to(device) for image in images]
    outputs = model(tensors)
    all_predictions = []
    for output, score_threshold, source in zip(outputs, score_thresholds, sources):
        scores = output["scores"].detach().cpu().numpy()
        keep = scores >= score_threshold
        masks = output["masks"].detach().cpu()[keep, 0].numpy() >= mask_threshold
        kept_scores = scores[keep]
        all_predictions.append([make_prediction(m, float(s), source) for m, s in zip(masks, kept_scores)])
    return all_predictions


def paste_tile_prediction(pred: Prediction, tile: tuple[int, int, int, int], full_shape: tuple[int, int]) -> Prediction:
    x0, y0, x1, y1 = tile
    full_h, full_w = full_shape
    tile_h = y1 - y0
    tile_w = x1 - x0
    mask = pred.mask
    if mask.shape != (tile_h, tile_w):
        mask = cv2.resize(mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    full_mask = np.zeros((full_h, full_w), dtype=bool)
    full_mask[y0:y1, x0:x1] = mask
    return make_prediction(full_mask, pred.score, "tile", pred.tile_index)


def match_masks(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_threshold: float = 0.5):
    gt_bboxes = [mask_bbox(mask) for mask in gt_masks]
    pred_bboxes = [mask_bbox(mask) for mask in pred_masks]
    used_gt = set()
    used_pred = set()
    candidates = []
    for gi, gt in enumerate(gt_masks):
        if gt_bboxes[gi] is None:
            continue
        for pi, pred in enumerate(pred_masks):
            if pred_bboxes[pi] is None or not bboxes_intersect(gt_bboxes[gi], pred_bboxes[pi]):
                continue
            iou = cropped_mask_iou(gt, pred, gt_bboxes[gi], pred_bboxes[pi])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    pairs = []
    for iou, gi, pi in sorted(candidates, reverse=True):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi, iou))
    tp = len(pairs)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    mean_iou = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
    return tp, fp, fn, mean_iou


def draw_mask_contours(image: np.ndarray, masks: list[np.ndarray], color, thickness: int = 2) -> None:
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, thickness)


def write_preview(
    src_path: Path,
    gt_masks: list[np.ndarray],
    preds: list[Prediction],
    row: dict,
    out_path: Path,
) -> None:
    image = imread_unicode(src_path)
    if image is None:
        return
    full_masks = [p.mask for p in preds if p.source == "full"]
    tile_masks = [p.mask for p in preds if p.source == "tile"]
    preview = image.copy()
    draw_mask_contours(preview, gt_masks, (0, 220, 0), 2)
    draw_mask_contours(preview, full_masks, (0, 0, 255), 2)
    draw_mask_contours(preview, tile_masks, (255, 0, 0), 2)
    text = (
        f"GT {row['gt_count']} Pred {row['pred_count']} "
        f"Full {row['full_pred_count']} TileAdd {row['tile_added_count']} "
        f"TP {row['tp_iou50']} FP {row['fp_iou50']} FN {row['fn_iou50']}"
    )
    cv2.rectangle(preview, (8, 8), (min(preview.shape[1] - 1, 1150), 44), (255, 255, 255), -1)
    cv2.putText(preview, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, preview)


def run(args) -> dict:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and args.reset_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco").mkdir(exist_ok=True)
    split_dir = out_dir / "split_test_labelme"
    preview_dir = out_dir / args.preview_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    ids = image_ids(src_dir)
    if getattr(args, "max_images", 0) and args.max_images > 0:
        ids = ids[: args.max_images]
    write_json(out_dir / "split.json", {"test": ids, "note": "full image + 6 overlapping tile fusion"})
    (out_dir / "test.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    coco = labelme_to_coco(src_dir, ids, out_dir / "coco" / "test.json", split_dir, args.label)
    print(f"test images={len(coco['images'])}, anns={len(coco['annotations'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_min_size, args.model_max_size, args.anchor_sizes, args.detections_per_img)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print(f"device={device}, cuda={torch.cuda.is_available()}, torch_threads={torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    rows = []
    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "full": 0, "tile_added": 0, "tile_seen": 0}
    image_ious = []

    for index, stem in enumerate(ids):
        img_path = find_image_for_stem(src_dir, stem)
        if img_path is None:
            continue
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        gt_masks, _ = load_gt_masks(src_dir, stem, args.label)

        tiles = make_tiles(width, height, args.tile_cols, args.tile_rows, args.tile_w, args.tile_h)
        tile_images = [image.crop(tile) for tile in tiles]
        view_predictions = predict_on_images(
            model,
            device,
            [image] + tile_images,
            [args.full_score_threshold] + [args.tile_score_threshold] * len(tile_images),
            args.mask_threshold,
            ["full"] + ["tile"] * len(tile_images),
        )

        full_preds = view_predictions[0]
        for pred in full_preds:
            pred.source = "full"
        final_preds = list(full_preds)
        tile_seen = 0
        tile_added = 0

        for tile_index, (tile, tile_preds) in enumerate(zip(tiles, view_predictions[1:])):
            for pred in tile_preds:
                pred.tile_index = tile_index
                pasted = paste_tile_prediction(pred, tile, (height, width))
                tile_seen += 1
                if not bbox_inside_image_safe_region(pasted.mask, width, height, args.image_boundary_margin):
                    continue
                if not bbox_inside_tile_safe_region(pasted.mask, tile, args.tile_boundary_margin):
                    continue
                max_intersection, max_iou = max_overlap_with_existing(pasted.mask, final_preds)
                if max_intersection <= args.tile_max_intersection_ratio and max_iou <= args.tile_max_iou:
                    final_preds.append(pasted)
                    tile_added += 1

        pred_masks = [p.mask for p in final_preds]
        tp, fp, fn, mean_iou = match_masks(gt_masks, pred_masks)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        count_error = abs(len(pred_masks) - len(gt_masks)) / len(gt_masks) if gt_masks else 0.0
        row = {
            "image": img_path.name,
            "gt_count": len(gt_masks),
            "pred_count": len(pred_masks),
            "full_pred_count": len(full_preds),
            "tile_seen_count": tile_seen,
            "tile_added_count": tile_added,
            "tp_iou50": tp,
            "fp_iou50": fp,
            "fn_iou50": fn,
            "precision_iou50": precision,
            "recall_iou50": recall,
            "f1_iou50": f1,
            "mean_matched_iou": mean_iou,
            "count_error": count_error,
        }
        rows.append(row)
        totals["gt"] += len(gt_masks)
        totals["pred"] += len(pred_masks)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["full"] += len(full_preds)
        totals["tile_seen"] += tile_seen
        totals["tile_added"] += tile_added
        if mean_iou:
            image_ious.append(mean_iou)
        if args.preview_limit < 0 or index < args.preview_limit:
            write_preview(img_path, gt_masks, final_preds, row, preview_dir / f"{stem}_示意.png")
        print(
            f"{stem}: gt={len(gt_masks)} pred={len(pred_masks)} "
            f"full={len(full_preds)} tile_add={tile_added} tp={tp} fp={fp} fn={fn}",
            flush=True,
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        "strategy": "full_plus_6_overlapping_tiles",
        "full_score_threshold": args.full_score_threshold,
        "tile_score_threshold": args.tile_score_threshold,
        "mask_threshold": args.mask_threshold,
        "tile_cols": args.tile_cols,
        "tile_rows": args.tile_rows,
        "tile_w": args.tile_w,
        "tile_h": args.tile_h,
        "image_boundary_margin": args.image_boundary_margin,
        "tile_boundary_margin": args.tile_boundary_margin,
        "tile_max_intersection_ratio": args.tile_max_intersection_ratio,
        "tile_max_iou": args.tile_max_iou,
        "gt_total": totals["gt"],
        "pred_total": totals["pred"],
        "full_pred_total": totals["full"],
        "tile_seen_total": totals["tile_seen"],
        "tile_added_total": totals["tile_added"],
        "tp_iou50": totals["tp"],
        "fp_iou50": totals["fp"],
        "fn_iou50": totals["fn"],
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "mean_image_matched_iou": float(np.mean(image_ious)) if image_ious else 0.0,
        "mean_count_error": float(np.mean([r["count_error"] for r in rows])) if rows else 0.0,
    }

    with (out_dir / "test_metrics_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "image",
            "gt_count",
            "pred_count",
            "full_pred_count",
            "tile_seen_count",
            "tile_added_count",
            "tp_iou50",
            "fp_iou50",
            "fn_iou50",
            "precision_iou50",
            "recall_iou50",
            "f1_iou50",
            "mean_matched_iou",
            "count_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "test_metrics_summary.json", summary)
    write_json(
        out_dir / "run_info.json",
        {
            "src_dir": str(src_dir),
            "model": str(Path(args.model).resolve()),
            "label": args.label,
            "preview_dir": str(preview_dir),
            "device": str(device),
            "model_min_size": args.model_min_size,
            "model_max_size": args.model_max_size,
            "anchor_sizes": args.anchor_sizes,
            "summary": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"preview dir: {preview_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--model_min_size", type=int, default=512)
    parser.add_argument("--model_max_size", type=int, default=1024)
    parser.add_argument("--anchor_sizes", default="8,16,32,64,128")
    parser.add_argument("--detections_per_img", type=int, default=900)
    parser.add_argument("--full_score_threshold", type=float, default=0.5)
    parser.add_argument("--tile_score_threshold", type=float, default=0.2)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--tile_cols", type=int, default=3)
    parser.add_argument("--tile_rows", type=int, default=2)
    parser.add_argument("--tile_w", type=int, default=1024)
    parser.add_argument("--tile_h", type=int, default=960)
    parser.add_argument("--tile_margin", type=int, default=64, help="兼容旧命令；当前不用于过滤。")
    parser.add_argument("--image_boundary_margin", type=int, default=4)
    parser.add_argument("--tile_boundary_margin", type=int, default=4)
    parser.add_argument("--edge_score_threshold", type=float, default=0.5)
    parser.add_argument("--tile_max_intersection_ratio", type=float, default=0.05)
    parser.add_argument("--tile_max_iou", type=float, default=0.2)
    parser.add_argument("--torch_threads", type=int, default=8)
    parser.add_argument("--preview_limit", type=int, default=-1)
    parser.add_argument("--max_images", type=int, default=0, help="Only process the first N images; 0 means all images.")
    parser.add_argument("--preview_dir", default="preview_images")
    parser.add_argument("--reset_out_dir", action="store_true")
    parser.add_argument("--no_export", action="store_true", help="保留兼容参数；本脚本当前不导出 LabelMe。")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
