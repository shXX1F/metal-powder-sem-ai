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


def _background_support_metrics(
    instance: InstanceMask,
    gray: np.ndarray,
    ring_width_px: int,
) -> Optional[Dict[str, float]]:
    height, width = gray.shape[:2]
    padding = max(2, int(ring_width_px))
    x1, y1, x2, y2 = _expanded_bbox(instance.bbox_xyxy, padding)
    region = (
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2),
    )
    rx1, ry1, rx2, ry2 = region
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    mask = _mask_in_region(instance, region)
    inside = gray[ry1:ry2, rx1:rx2][mask]
    if inside.size < 16:
        return None

    kernel_size = 2 * padding + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    ring = np.logical_and(dilated, np.logical_not(mask))
    ring_values = gray[ry1:ry2, rx1:rx2][ring]
    if ring_values.size < 16:
        return None

    inside_values = inside.astype(np.float32)
    ring_values = ring_values.astype(np.float32)
    background_level = float(np.median(ring_values))
    high_reference = float(
        np.percentile(np.concatenate([inside_values, ring_values]), 95)
    )
    dynamic_range = max(0.0, high_reference - background_level)
    support_threshold = background_level + max(5.0, 0.25 * dynamic_range)
    bright_fraction = float(np.mean(inside_values >= support_threshold))
    median_contrast = float(np.median(inside_values) - background_level)
    return {
        "background_level": background_level,
        "high_reference": high_reference,
        "dynamic_range": dynamic_range,
        "support_threshold": support_threshold,
        "bright_fraction": bright_fraction,
        "median_contrast": median_contrast,
    }


def _near_internal_tile_edge_sides(
    instance: InstanceMask,
    image_width: int,
    image_height: int,
    margin_fraction: float = 0.03,
    max_margin_px: int = 24,
) -> Tuple[str, ...]:
    """Return internal tile sides close to a tiled detection's mask boundary."""
    if instance.source != "tiled" or instance.tile_xyxy is None:
        return ()

    x1, y1, x2, y2 = [int(value) for value in instance.bbox_xyxy]
    tx1, ty1, tx2, ty2 = [int(value) for value in instance.tile_xyxy]
    tile_span = max(1, tx2 - tx1, ty2 - ty1)
    margin = max(
        3,
        min(int(max_margin_px), int(round(tile_span * float(margin_fraction)))),
    )
    sides: List[str] = []
    if tx1 > 0 and 0 <= x1 - tx1 <= margin:
        sides.append("left")
    if tx2 < int(image_width) and 0 <= tx2 - x2 <= margin:
        sides.append("right")
    if ty1 > 0 and 0 <= y1 - ty1 <= margin:
        sides.append("top")
    if ty2 < int(image_height) and 0 <= ty2 - y2 <= margin:
        sides.append("bottom")
    return tuple(sides)


def filter_false_background_masks(
    instances: Sequence[InstanceMask],
    image_bgr: np.ndarray,
    enabled: bool = True,
    ring_width_px: int = 6,
    min_dynamic_range: float = 20.0,
    max_background_bright_fraction: float = 0.35,
    min_particle_contrast: float = 8.0,
) -> Tuple[List[InstanceMask], Dict[str, object]]:
    """Remove background-like masks clipped by an internal inference tile edge.

    The filter deliberately requires both signals. A dark region alone is not
    removed, and a genuine bright particle crossing a tile boundary is kept.
    """

    before_count = len(instances)
    info: Dict[str, object] = {
        "background_filter_enabled": bool(enabled),
        "background_filter_before_count": before_count,
        "background_filter_after_count": before_count,
        "background_filter_removed_count": 0,
        "background_filter_removed_particle_ids": "",
        "background_filter_removed_details": [],
    }
    if not enabled or before_count == 0:
        return list(instances), info

    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kept: List[InstanceMask] = []
    removed_ids: List[str] = []
    removed_details: List[Dict[str, object]] = []
    for instance in instances:
        touched_sides = tuple(instance.internal_tile_edge_sides or ())
        if not touched_sides:
            touched_sides = _near_internal_tile_edge_sides(
                instance,
                image_width=gray.shape[1],
                image_height=gray.shape[0],
            )
        if not touched_sides:
            kept.append(instance)
            continue

        metrics = _background_support_metrics(
            instance,
            gray=gray,
            ring_width_px=ring_width_px,
        )
        if metrics is None:
            kept.append(instance)
            continue

        dynamic_range = float(metrics["dynamic_range"])
        bright_fraction = float(metrics["bright_fraction"])
        contrast_limit = max(
            float(min_particle_contrast),
            0.10 * dynamic_range,
        )
        background_like = (
            dynamic_range >= float(min_dynamic_range)
            and bright_fraction < float(max_background_bright_fraction)
            and float(metrics["median_contrast"]) < contrast_limit
        )
        if not background_like:
            kept.append(instance)
            continue

        particle_id = int(instance.particle_id)
        removed_ids.append(str(particle_id))
        removed_details.append(
            {
                "particle_id": particle_id,
                "source": instance.source,
                "internal_tile_edge_sides": ",".join(touched_sides),
                **metrics,
            }
        )

    kept = renumber_instances(kept)
    info.update(
        {
            "background_filter_after_count": len(kept),
            "background_filter_removed_count": len(removed_ids),
            "background_filter_removed_particle_ids": ",".join(removed_ids),
            "background_filter_removed_details": removed_details,
        }
    )
    return kept, info


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
