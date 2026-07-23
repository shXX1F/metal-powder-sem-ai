from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .segment import InstanceMask, bbox_intersection, contour_centroid, renumber_instances


def _safe_diameter_px(area_px: float) -> float:
    return math.sqrt(4.0 * float(area_px) / math.pi) if area_px > 0 else 0.0


def _mask_in_region(
    instance: InstanceMask,
    region_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    rx1, ry1, rx2, ry2 = region_xyxy
    out = np.zeros((max(0, ry2 - ry1), max(0, rx2 - rx1)), dtype=bool)
    if out.size == 0:
        return out

    bx1, by1, bx2, by2 = [int(v) for v in instance.bbox_xyxy]
    ox1 = max(rx1, bx1)
    oy1 = max(ry1, by1)
    ox2 = min(rx2, bx2)
    oy2 = min(ry2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return out

    local_like = instance.mask.shape[:2] == (max(0, by2 - by1), max(0, bx2 - bx1))
    if local_like:
        src = instance.mask[oy1 - by1 : oy2 - by1, ox1 - bx1 : ox2 - bx1]
    else:
        src = instance.mask[oy1:oy2, ox1:ox2]

    out[oy1 - ry1 : oy2 - ry1, ox1 - rx1 : ox2 - rx1] = src > 0
    return out


def _expanded_bbox(bbox: Tuple[int, int, int, int], padding_px: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return x1 - padding_px, y1 - padding_px, x2 + padding_px, y2 + padding_px


def _overlap_smaller(child: InstanceMask, parent: InstanceMask) -> float:
    region = bbox_intersection(child.bbox_xyxy, parent.bbox_xyxy)
    if region is None:
        return 0.0
    child_mask = _mask_in_region(child, region)
    parent_mask = _mask_in_region(parent, region)
    child_area = int((child.mask > 0).sum())
    if child_area <= 0:
        return 0.0
    return float(np.logical_and(child_mask, parent_mask).sum() / child_area)


def _metrics(instance: InstanceMask) -> Dict[str, float]:
    area_px = float((instance.mask > 0).sum())
    diameter_px = _safe_diameter_px(area_px)
    cx, cy = contour_centroid(instance.contour, instance.bbox_xyxy)
    return {
        "area_px": area_px,
        "diameter_px": diameter_px,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def filter_false_surface_fragments(
    instances: Sequence[InstanceMask],
    pixel_size_um: Optional[float] = None,
    enabled: bool = True,
    max_child_diameter_um: float = 20.0,
    min_parent_diameter_um: float = 25.0,
    min_parent_child_diameter_ratio: float = 1.8,
    containment_overlap_threshold: float = 0.35,
    surface_contact_distance_factor: float = 0.0,
    contact_tolerance_px: int = 3,
) -> Tuple[List[InstanceMask], Dict[str, object]]:
    """Remove small masks that are likely surface fragments of larger particles.

    The rule is intentionally conservative: a small instance is removed only
    when it is much smaller than a nearby parent and either overlaps the parent
    mask substantially or has its centroid inside the parent's contour.
    """

    before_count = len(instances)
    info: Dict[str, object] = {
        "postprocess_enabled": bool(enabled),
        "postprocess_before_count": before_count,
        "postprocess_after_count": before_count,
        "postprocess_removed_count": 0,
        "postprocess_removed_particle_ids": "",
    }
    if not enabled or before_count <= 1:
        return list(instances), info

    pixel_size = float(pixel_size_um or 0.0)
    if pixel_size > 0:
        max_child_diameter_px = max_child_diameter_um / pixel_size
        min_parent_diameter_px = min_parent_diameter_um / pixel_size
    else:
        max_child_diameter_px = max_child_diameter_um
        min_parent_diameter_px = min_parent_diameter_um

    metrics = [_metrics(instance) for instance in instances]
    removed_indexes: set[int] = set()

    for child_idx, child in enumerate(instances):
        child_diameter = float(metrics[child_idx]["diameter_px"])
        if child_diameter <= 0 or child_diameter > max_child_diameter_px:
            continue

        child_center = (
            float(metrics[child_idx]["centroid_x"]),
            float(metrics[child_idx]["centroid_y"]),
        )
        child_bbox_expanded = _expanded_bbox(child.bbox_xyxy, int(contact_tolerance_px))

        for parent_idx, parent in enumerate(instances):
            if child_idx == parent_idx:
                continue

            parent_diameter = float(metrics[parent_idx]["diameter_px"])
            if parent_diameter < min_parent_diameter_px:
                continue
            if parent_diameter < child_diameter * float(min_parent_child_diameter_ratio):
                continue
            if bbox_intersection(child_bbox_expanded, parent.bbox_xyxy) is None:
                continue

            overlap_ratio = _overlap_smaller(child, parent)
            signed_distance = cv2.pointPolygonTest(parent.contour, child_center, True)
            inside_parent = signed_distance >= 0
            near_parent_surface = (
                surface_contact_distance_factor > 0
                and -signed_distance <= child_diameter * float(surface_contact_distance_factor)
            )
            if overlap_ratio >= containment_overlap_threshold or inside_parent or near_parent_surface:
                removed_indexes.add(child_idx)
                break

    kept = [instance for idx, instance in enumerate(instances) if idx not in removed_indexes]
    kept = renumber_instances(kept)
    removed_ids = [str(int(instances[idx].particle_id)) for idx in sorted(removed_indexes)]
    info.update(
        {
            "postprocess_after_count": len(kept),
            "postprocess_removed_count": len(removed_indexes),
            "postprocess_removed_particle_ids": ",".join(removed_ids),
        }
    )
    return kept, info
