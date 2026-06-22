from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class InstanceMask:
    particle_id: int
    mask: np.ndarray
    contour: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    score: float = 1.0


@dataclass
class TileCandidate:
    mask: np.ndarray
    contour: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    edge_clearance: float
    tile_xyxy: Tuple[int, int, int, int]


def mask_to_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_mask_to_bbox(mask: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return (mask[y1:y2, x1:x2] > 0).astype(np.uint8)


def mask_is_local(instance: InstanceMask) -> bool:
    x1, y1, x2, y2 = instance.bbox_xyxy
    return instance.mask.shape[:2] == (max(0, y2 - y1), max(0, x2 - x1))


def paint_instance_mask(
    canvas: np.ndarray,
    instance: InstanceMask,
    value: bool | int = True,
) -> None:
    if instance.mask.shape[:2] == canvas.shape[:2]:
        canvas[instance.mask > 0] = value
        return
    x1, y1, x2, y2 = instance.bbox_xyxy
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(canvas.shape[1], int(x2))
    y2 = min(canvas.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    local = instance.mask[: y2 - y1, : x2 - x1] > 0
    canvas[y1:y2, x1:x2][local] = value


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    count = max(2, math.ceil((length - overlap) / stride))
    starts = np.linspace(0, length - tile_size, count)
    return sorted({int(round(value)) for value in starts})


def ownership_intervals(
    starts: Sequence[int],
    length: int,
    tile_size: int,
) -> list[tuple[float, float]]:
    ends = [min(length, start + tile_size) for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    intervals: list[tuple[float, float]] = []
    for idx, center in enumerate(centers):
        left = 0.0 if idx == 0 else (centers[idx - 1] + center) / 2.0
        right = float(length) if idx == len(centers) - 1 else (center + centers[idx + 1]) / 2.0
        intervals.append((left, right))
    return intervals


def contour_centroid(
    contour: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_intersection(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def tile_edge_clearance(
    bbox_xyxy: Tuple[int, int, int, int],
    tile_xyxy: Tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    tx1, ty1, tx2, ty2 = tile_xyxy
    return float(min(x1 - tx1, y1 - ty1, tx2 - x2, ty2 - y2))


def mask_overlap_metrics(a: TileCandidate, b: TileCandidate) -> Tuple[float, float]:
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
    pts_a = np.round(
        a.contour.reshape(-1, 2) - np.array([x1, y1], dtype=np.float32)
    ).astype(np.int32)
    pts_b = np.round(
        b.contour.reshape(-1, 2) - np.array([x1, y1], dtype=np.float32)
    ).astype(np.int32)
    cv2.fillPoly(mask_a, [pts_a], 1)
    cv2.fillPoly(mask_b, [pts_b], 1)
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0, 0.0
    area_a = max(1, int(round(cv2.contourArea(a.contour))))
    area_b = max(1, int(round(cv2.contourArea(b.contour))))
    union = area_a + area_b - inter
    iou = inter / max(1, union)
    overlap_smaller = inter / max(1, min(area_a, area_b))
    return float(iou), float(overlap_smaller)


def dedupe_tile_candidates(
    candidates: Sequence[TileCandidate],
    iou_threshold: float = 0.25,
    overlap_threshold: float = 0.70,
) -> list[TileCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.edge_clearance, item.score, cv2.contourArea(item.contour)),
        reverse=True,
    )
    kept: list[TileCandidate] = []
    for candidate in ordered:
        duplicate = False
        candidate_area = max(1.0, float(cv2.contourArea(candidate.contour)))
        for existing in kept:
            intersection_bbox = bbox_intersection(candidate.bbox_xyxy, existing.bbox_xyxy)
            if intersection_bbox is None:
                continue
            ix1, iy1, ix2, iy2 = intersection_bbox
            bbox_inter_area = float((ix2 - ix1) * (iy2 - iy1))
            existing_area = max(1.0, float(cv2.contourArea(existing.contour)))
            if bbox_inter_area / max(1.0, min(candidate_area, existing_area)) < 0.05:
                continue
            iou, overlap_smaller = mask_overlap_metrics(candidate, existing)
            if iou >= iou_threshold or overlap_smaller >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))


def instances_from_label_image(
    labels: np.ndarray,
    min_area_px: int = 30,
) -> List[InstanceMask]:
    instances: List[InstanceMask] = []
    next_id = 1
    for label_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label_id
        area = int(mask.sum())
        if area < min_area_px:
            continue
        contour = mask_to_contour(mask)
        if contour is None or cv2.contourArea(contour) < min_area_px:
            continue
        bbox = bbox_from_mask(mask)
        instances.append(
            InstanceMask(
                particle_id=next_id,
                mask=crop_mask_to_bbox(mask, bbox),
                contour=contour,
                bbox_xyxy=bbox,
                score=1.0,
            )
        )
        next_id += 1
    return instances


def renumber_instances(instances: Sequence[InstanceMask]) -> List[InstanceMask]:
    return [
        InstanceMask(
            particle_id=idx,
            mask=instance.mask,
            contour=instance.contour,
            bbox_xyxy=instance.bbox_xyxy,
            score=float(instance.score),
        )
        for idx, instance in enumerate(instances, start=1)
    ]


def merge_instance_groups(
    *groups: Sequence[InstanceMask],
    iou_threshold: float = 0.25,
    overlap_threshold: float = 0.70,
) -> List[InstanceMask]:
    """Merge multiple inference passes and remove duplicated particles.

    Earlier groups have priority when two masks overlap strongly. This lets a
    whole-image pass keep stable large-particle masks, while a tiled pass fills
    in additional small particles that were missed.
    """
    candidates: list[TileCandidate] = []
    for group_idx, group in enumerate(groups):
        priority = float(len(groups) - group_idx)
        for instance in group:
            candidates.append(
                TileCandidate(
                    mask=instance.mask.astype(np.uint8),
                    contour=instance.contour,
                    bbox_xyxy=instance.bbox_xyxy,
                    score=float(instance.score),
                    edge_clearance=priority,
                    tile_xyxy=instance.bbox_xyxy,
                )
            )

    kept = dedupe_tile_candidates(
        candidates,
        iou_threshold=iou_threshold,
        overlap_threshold=overlap_threshold,
    )
    instances: list[InstanceMask] = []
    for particle_id, candidate in enumerate(kept, start=1):
        instances.append(
            InstanceMask(
                particle_id=particle_id,
                mask=candidate.mask.astype(np.uint8),
                contour=np.round(candidate.contour).astype(np.int32),
                bbox_xyxy=candidate.bbox_xyxy,
                score=float(candidate.score),
            )
        )
    return instances


class ClassicalParticleSegmenter:
    """未训练模型时的可运行基线：Otsu 二值图 + distance transform + watershed。"""

    def __init__(
        self,
        min_area_px: int = 40,
        peak_min_distance: int = 8,
        fill_small_holes_px: int = 64,
    ) -> None:
        self.min_area_px = min_area_px
        self.peak_min_distance = peak_min_distance
        self.fill_small_holes_px = fill_small_holes_px

    def segment(self, binary: np.ndarray) -> List[InstanceMask]:
        foreground = binary > 0
        foreground = remove_small_objects_cv(foreground, self.min_area_px)
        foreground = fill_small_holes_cv(foreground, self.fill_small_holes_px)
        foreground_u8 = foreground.astype(np.uint8)

        distance = cv2.distanceTransform(foreground_u8, cv2.DIST_L2, 5)
        coordinates = local_maxima_coordinates(
            distance,
            min_distance=self.peak_min_distance,
            labels=foreground_u8,
        )

        markers = np.zeros(distance.shape, dtype=np.int32)
        if len(coordinates) == 0:
            markers, _ = ndi.label(foreground_u8)
        else:
            for idx, (row, col) in enumerate(coordinates, start=1):
                markers[row, col] = idx
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            markers = cv2.dilate(markers.astype(np.uint16), kernel, iterations=1).astype(np.int32)

        # watershed 将粘连颗粒按距离峰拆分，保证每个颗粒输出独立 mask。
        labels = watershed_cv(distance, markers, foreground)
        return instances_from_label_image(labels, min_area_px=self.min_area_px)


def remove_small_objects_cv(binary: np.ndarray, min_area_px: int) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)
    kept = np.zeros(binary_u8.shape, dtype=bool)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area_px:
            kept |= labels == label_id
    return kept


def fill_small_holes_cv(binary: np.ndarray, area_threshold: int) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    inv = (binary_u8 == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    filled = binary_u8.astype(bool)
    h, w = binary_u8.shape
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if (not touches_border) and area <= area_threshold:
            filled |= labels == label_id
    return filled


def local_maxima_coordinates(
    distance: np.ndarray,
    min_distance: int,
    labels: np.ndarray,
) -> np.ndarray:
    size = max(3, int(min_distance) * 2 + 1)
    max_filtered = ndi.maximum_filter(distance, size=size)
    peaks = (distance == max_filtered) & (distance > 0) & (labels > 0)
    peak_labels, peak_count = ndi.label(peaks)
    centers = ndi.center_of_mass(distance, peak_labels, range(1, peak_count + 1))
    coordinates = []
    for row, col in centers:
        if np.isnan(row) or np.isnan(col):
            continue
        coordinates.append((int(round(row)), int(round(col))))
    return np.asarray(coordinates, dtype=np.int32)


def watershed_cv(
    distance: np.ndarray,
    markers: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    markers_i32 = markers.astype(np.int32)
    markers_i32[~foreground] = 0
    distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
    image_3c = cv2.cvtColor(distance_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(image_3c, markers_i32)
    labels = markers_i32.copy()
    labels[labels < 0] = 0
    labels[~foreground] = 0
    return labels


class MaskRCNNSegmenter:
    """Mask R-CNN 推理封装；训练权重由 train.py 生成。"""

    def __init__(
        self,
        weights_path: str,
        num_classes: int = 2,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_detections: int = 500,
        max_inference_side: Optional[int] = 1280,
        device: Optional[str] = None,
    ) -> None:
        import torch

        from .train import build_mask_rcnn_model

        self.torch = torch
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.max_detections = int(max_detections)
        self.max_inference_side = int(max_inference_side or 0)
        self.model = build_mask_rcnn_model(
            num_classes=num_classes,
            pretrained=False,
            detections_per_img=self.max_detections,
        )
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image_bgr: np.ndarray) -> List[InstanceMask]:
        import torch

        original_h, original_w = image_bgr.shape[:2]
        inference_bgr = image_bgr
        scale = 1.0
        max_side = self.max_inference_side
        if max_side > 0 and max(original_h, original_w) > max_side:
            scale = max_side / float(max(original_h, original_w))
            resized_w = max(1, int(round(original_w * scale)))
            resized_h = max(1, int(round(original_h * scale)))
            inference_bgr = cv2.resize(
                image_bgr,
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            pred = self.model([tensor])[0]

        scores = pred.get("scores", torch.empty(0))
        masks = pred.get("masks", torch.empty(0))
        instances: List[InstanceMask] = []
        particle_id = 1
        for idx in range(len(scores)):
            score = float(scores[idx].detach().cpu().item())
            if float(score) < self.score_threshold:
                continue
            mask = masks[idx, 0].detach().cpu().numpy() >= self.mask_threshold
            if scale != 1.0:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            contour = mask_to_contour(mask)
            if contour is None:
                continue
            bbox = bbox_from_mask(mask)
            instances.append(
                InstanceMask(
                    particle_id=particle_id,
                    mask=crop_mask_to_bbox(mask, bbox),
                    contour=contour,
                    bbox_xyxy=bbox,
                    score=float(score),
                )
            )
            particle_id += 1
        return instances

    def segment_tiled(
        self,
        image_bgr: np.ndarray,
        tile_size: int = 896,
        overlap: int = 224,
        nms_iou_threshold: float = 0.25,
        nms_overlap_threshold: float = 0.70,
    ) -> List[InstanceMask]:
        height, width = image_bgr.shape[:2]
        tile_size = max(256, int(tile_size))
        overlap = max(0, min(int(overlap), tile_size - 1))
        x_starts = axis_starts(width, tile_size, overlap)
        y_starts = axis_starts(height, tile_size, overlap)
        x_ownership = ownership_intervals(x_starts, width, tile_size)
        y_ownership = ownership_intervals(y_starts, height, tile_size)

        candidates: list[TileCandidate] = []
        for y_idx, y1 in enumerate(y_starts):
            y2 = min(height, y1 + tile_size)
            core_y1, core_y2 = y_ownership[y_idx]
            for x_idx, x1 in enumerate(x_starts):
                x2 = min(width, x1 + tile_size)
                core_x1, core_x2 = x_ownership[x_idx]
                tile = image_bgr[y1:y2, x1:x2]
                tile_instances = self.segment(tile)
                tile_xyxy = (x1, y1, x2, y2)

                for instance in tile_instances:
                    contour = instance.contour.astype(np.float32).copy()
                    contour[:, 0, 0] += x1
                    contour[:, 0, 1] += y1
                    bbox = (
                        instance.bbox_xyxy[0] + x1,
                        instance.bbox_xyxy[1] + y1,
                        instance.bbox_xyxy[2] + x1,
                        instance.bbox_xyxy[3] + y1,
                    )
                    bbox = (
                        max(0, bbox[0]),
                        max(0, bbox[1]),
                        min(width, bbox[2]),
                        min(height, bbox[3]),
                    )
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    cx, cy = contour_centroid(contour, bbox)
                    if not (core_x1 <= cx < core_x2 and core_y1 <= cy < core_y2):
                        continue
                    candidates.append(
                        TileCandidate(
                            mask=instance.mask.astype(np.uint8),
                            contour=contour,
                            bbox_xyxy=bbox,
                            score=float(instance.score),
                            edge_clearance=tile_edge_clearance(bbox, tile_xyxy),
                            tile_xyxy=tile_xyxy,
                        )
                    )

        kept = dedupe_tile_candidates(
            candidates,
            iou_threshold=nms_iou_threshold,
            overlap_threshold=nms_overlap_threshold,
        )
        instances: list[InstanceMask] = []
        for particle_id, candidate in enumerate(kept, start=1):
            contour = np.round(candidate.contour).astype(np.int32)
            instances.append(
                InstanceMask(
                    particle_id=particle_id,
                    mask=candidate.mask.astype(np.uint8),
                    contour=contour,
                    bbox_xyxy=candidate.bbox_xyxy,
                    score=candidate.score,
                )
            )
        return instances


def build_segmenter(
    name: str,
    weights_path: Optional[str] = None,
    **kwargs,
):
    name = name.lower()
    if name == "classical":
        return ClassicalParticleSegmenter(**kwargs)
    if name in {"maskrcnn", "mask-r-cnn", "mask_rcnn"}:
        if not weights_path:
            raise ValueError("使用 Mask R-CNN 推理时必须提供 weights_path。")
        return MaskRCNNSegmenter(weights_path=weights_path, **kwargs)
    raise ValueError(f"未知分割器: {name}")
