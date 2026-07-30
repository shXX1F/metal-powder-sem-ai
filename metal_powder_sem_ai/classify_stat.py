from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def gbt8170_round(value: float, ndigits: int = 2) -> float:
    """GB/T 8170 常用修约：四舍六入五留双，等价于 ROUND_HALF_EVEN。"""
    quant = Decimal("1").scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_EVEN))


DIAMETER_KEYS = {
    "equivalent": "equivalent_diameter_um",
    "major_axis": "major_axis_um",
    "feret_max": "feret_diameter_um",
    "bbox_max": "bbox_max_diameter_um",
    "statistical": "statistical_diameter_um",
}

ROUNDNESS_KEYS = {
    "contour": ("q_value_contour", "roundness_contour"),
    "crofton": ("q_value_crofton", "roundness_crofton"),
    "subpixel": ("q_value_subpixel", "roundness_subpixel"),
}

DEFAULT_ROUNDNESS_METHOD = "crofton"
DEFAULT_SPHERICAL_ROUNDNESS_METHOD = "crofton_open3_smooth3_blend"
DEFAULT_SPHERICAL_ROUNDNESS_THRESHOLD = 0.9490

SPHERICAL_ROUNDNESS_METHODS = {
    "report": (),
    "contour": (("roundness_contour", 1.0),),
    "crofton": (("roundness_crofton", 1.0),),
    "subpixel": (("roundness_subpixel", 1.0),),
    "crofton_open3": (("roundness_crofton_open3", 1.0),),
    "crofton_close3": (("roundness_crofton_close3", 1.0),),
    "crofton_smooth3": (("roundness_crofton_smooth3", 1.0),),
    "crofton_open3_smooth3_blend": (
        ("roundness_crofton_open3", 0.75),
        ("roundness_crofton_smooth3", 0.25),
    ),
}


def _spherical_roundness_value(
    feature: Dict,
    method: str,
    report_roundness_key: str,
) -> Tuple[float, bool]:
    """Return the classifier roundness and whether a compatibility fallback was used."""
    if method not in SPHERICAL_ROUNDNESS_METHODS:
        raise ValueError(f"Unknown spherical roundness method: {method}")

    components = SPHERICAL_ROUNDNESS_METHODS[method]
    if method == "report":
        value = feature.get(report_roundness_key, feature.get("roundness", 0.0))
        return float(value or 0.0), False

    weighted_value = 0.0
    for key, weight in components:
        value = feature.get(key)
        if value is None or not math.isfinite(float(value)):
            fallback = feature.get(
                report_roundness_key,
                feature.get("roundness", 0.0),
            )
            return float(fallback or 0.0), True
        weighted_value += float(weight) * float(value)
    return min(1.0, max(0.0, weighted_value)), False


def reference_percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Use the lower observed rank used by the supplied reference software."""
    valid = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if valid.size == 0:
        return None
    try:
        return float(np.percentile(valid, q, method="lower"))
    except TypeError:  # NumPy < 1.22
        return float(np.percentile(valid, q, interpolation="lower"))


def diameter_value(feature: Dict, method: str = "feret_max") -> float:
    if method not in DIAMETER_KEYS:
        raise ValueError(f"Unknown diameter method: {method}")
    key = DIAMETER_KEYS[method]
    value = feature.get(key)
    if value is None and method in {"feret_max", "bbox_max", "statistical"}:
        value = feature.get("equivalent_diameter_um", 0.0)
    return float(value or 0.0)


def summarize_size_distribution(
    features: Sequence[Dict],
    method: str = "feret_max",
    exclude_border: bool = False,
) -> Dict[str, object]:
    selected = [
        item
        for item in features
        if not exclude_border or not bool(item.get("touches_image_border", False))
    ]
    diameters = [diameter_value(item, method=method) for item in selected]
    diameters = [value for value in diameters if value > 0 and math.isfinite(value)]
    percentile_values = {
        label: reference_percentile(diameters, q)
        for label, q in (("d5", 5), ("d10", 10), ("d50", 50), ("d90", 90), ("d95", 95))
    }
    return {
        "diameter_method": method,
        "size_statistics_exclude_border": bool(exclude_border),
        "size_statistics_particle_count": len(diameters),
        "border_particle_count": sum(
            1 for item in features if bool(item.get("touches_image_border", False))
        ),
        "diameter_mean_um": float(np.mean(diameters)) if diameters else 0.0,
        "diameter_min_um": min(diameters) if diameters else 0.0,
        "diameter_max_um": max(diameters) if diameters else 0.0,
        **{f"diameter_{key}_um": float(value or 0.0) for key, value in percentile_values.items()},
    }


def _contact_by_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    tolerance_px: int = 3,
) -> bool:
    kernel_size = max(1, tolerance_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_a = cv2.dilate((mask_a > 0).astype(np.uint8), kernel, iterations=1) > 0
    return bool(np.any(dilated_a & (mask_b > 0)))


def _infer_pixel_size_um(features: Sequence[Dict]) -> Optional[float]:
    values: List[float] = []
    for feature in features:
        diameter_px = float(feature.get("equivalent_diameter_px", 0.0) or 0.0)
        diameter_um = float(feature.get("equivalent_diameter_um", 0.0) or 0.0)
        if diameter_px > 0 and diameter_um > 0:
            values.append(diameter_um / diameter_px)
    if not values:
        return None
    return float(np.median(values))


def _resolve_agglomerate_tolerance(
    features: Sequence[Dict],
    fallback_px: int,
    tolerance_um: Optional[float],
) -> Tuple[int, Optional[float], Optional[float], str]:
    """Resolve the contact tolerance and retain its calibration provenance."""
    pixel_size_um = _infer_pixel_size_um(features)
    if tolerance_um is not None and tolerance_um > 0 and pixel_size_um:
        tolerance_px = max(1, int(math.ceil(float(tolerance_um) / pixel_size_um)))
        return (
            tolerance_px,
            float(tolerance_px * pixel_size_um),
            float(pixel_size_um),
            "physical_um",
        )

    tolerance_px = max(1, int(round(fallback_px)))
    effective_um = (
        float(tolerance_px * pixel_size_um)
        if pixel_size_um is not None and pixel_size_um > 0
        else None
    )
    source = "fallback_px" if effective_um is not None else "fallback_px_no_scale"
    return tolerance_px, effective_um, pixel_size_um, source


def _mask_in_region(
    mask: np.ndarray,
    feature: Dict,
    region_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    rx1, ry1, rx2, ry2 = region_xyxy
    out = np.zeros((max(0, ry2 - ry1), max(0, rx2 - rx1)), dtype=bool)
    if out.size == 0:
        return out

    bx1 = int(feature.get("bbox_x1", 0))
    by1 = int(feature.get("bbox_y1", 0))
    bx2 = int(feature.get("bbox_x2", mask.shape[1]))
    by2 = int(feature.get("bbox_y2", mask.shape[0]))
    local_like = mask.shape[:2] == (max(0, by2 - by1), max(0, bx2 - bx1))

    ox1 = max(rx1, bx1)
    oy1 = max(ry1, by1)
    ox2 = min(rx2, bx2)
    oy2 = min(ry2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return out

    dst_y1 = oy1 - ry1
    dst_y2 = oy2 - ry1
    dst_x1 = ox1 - rx1
    dst_x2 = ox2 - rx1
    if local_like:
        src = mask[oy1 - by1 : oy2 - by1, ox1 - bx1 : ox2 - bx1]
    else:
        src = mask[oy1:oy2, ox1:ox2]
    out[dst_y1:dst_y2, dst_x1:dst_x2] = src > 0
    return out


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return (binary > 0) & (eroded == 0)


def _mask_pair_evidence(
    mask_a: np.ndarray,
    feature_a: Dict,
    mask_b: np.ndarray,
    feature_b: Dict,
    tolerance_px: int,
    pixel_size_um: Optional[float],
) -> Dict[str, object]:
    """Measure gap and strong-contact evidence inside a compact pair ROI."""
    ax1 = int(feature_a.get("bbox_x1", 0)) - tolerance_px
    ay1 = int(feature_a.get("bbox_y1", 0)) - tolerance_px
    ax2 = int(feature_a.get("bbox_x2", 0)) + tolerance_px
    ay2 = int(feature_a.get("bbox_y2", 0)) + tolerance_px
    bx1 = int(feature_b.get("bbox_x1", 0)) - tolerance_px
    by1 = int(feature_b.get("bbox_y1", 0)) - tolerance_px
    bx2 = int(feature_b.get("bbox_x2", 0)) + tolerance_px
    by2 = int(feature_b.get("bbox_y2", 0)) + tolerance_px
    region = (
        max(0, min(ax1, bx1)),
        max(0, min(ay1, by1)),
        max(ax2, bx2),
        max(ay2, by2),
    )
    region_a = _mask_in_region(mask_a, feature_a, region)
    region_b = _mask_in_region(mask_b, feature_b, region)
    area_a = int(np.count_nonzero(region_a))
    area_b = int(np.count_nonzero(region_b))
    if area_a == 0 or area_b == 0:
        return {
            "gap_px": float("inf"),
            "gap_um": None,
            "contact_ratio": 0.0,
            "overlap_ratio": 0.0,
            "center_distance_ratio": float("inf"),
        }

    boundary_a = _mask_boundary(region_a)
    boundary_b = _mask_boundary(region_b)
    distance_to_a = cv2.distanceTransform(
        (~region_a).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_b = cv2.distanceTransform(
        (~region_b).astype(np.uint8), cv2.DIST_L2, 3
    )
    gap_a = float(np.min(distance_to_b[boundary_a])) if np.any(boundary_a) else float("inf")
    gap_b = float(np.min(distance_to_a[boundary_b])) if np.any(boundary_b) else float("inf")
    gap_px = min(gap_a, gap_b)

    if area_a <= area_b:
        smaller_boundary = boundary_a
        distance_to_other = distance_to_b
    else:
        smaller_boundary = boundary_b
        distance_to_other = distance_to_a
    smaller_perimeter_px = int(np.count_nonzero(smaller_boundary))
    near_limit = float(max(0, tolerance_px)) + 0.5
    near_count = int(
        np.count_nonzero(smaller_boundary & (distance_to_other <= near_limit))
    )
    contact_ratio = (
        float(near_count / smaller_perimeter_px) if smaller_perimeter_px else 0.0
    )
    overlap_ratio = float(
        np.count_nonzero(region_a & region_b) / max(1, min(area_a, area_b))
    )

    dx = float(feature_a["centroid_x"]) - float(feature_b["centroid_x"])
    dy = float(feature_a["centroid_y"]) - float(feature_b["centroid_y"])
    center_distance = float(np.hypot(dx, dy))
    radius_sum = (
        float(feature_a["equivalent_diameter_px"])
        + float(feature_b["equivalent_diameter_px"])
    ) / 2.0
    center_distance_ratio = center_distance / radius_sum if radius_sum > 0 else float("inf")
    gap_um = float(gap_px * pixel_size_um) if pixel_size_um else None
    return {
        "gap_px": gap_px,
        "gap_um": gap_um,
        "contact_ratio": contact_ratio,
        "overlap_ratio": overlap_ratio,
        "center_distance_ratio": center_distance_ratio,
    }


def _contact_by_masks_in_bboxes(
    mask_a: np.ndarray,
    feature_a: Dict,
    mask_b: np.ndarray,
    feature_b: Dict,
    tolerance_px: int = 3,
) -> bool:
    ax1 = int(feature_a.get("bbox_x1", 0)) - tolerance_px
    ay1 = int(feature_a.get("bbox_y1", 0)) - tolerance_px
    ax2 = int(feature_a.get("bbox_x2", 0)) + tolerance_px
    ay2 = int(feature_a.get("bbox_y2", 0)) + tolerance_px
    bx1 = int(feature_b.get("bbox_x1", 0)) - tolerance_px
    by1 = int(feature_b.get("bbox_y1", 0)) - tolerance_px
    bx2 = int(feature_b.get("bbox_x2", 0)) + tolerance_px
    by2 = int(feature_b.get("bbox_y2", 0)) + tolerance_px
    rx1 = max(0, min(ax1, bx1))
    ry1 = max(0, min(ay1, by1))
    rx2 = max(ax2, bx2)
    ry2 = max(ay2, by2)
    region = (rx1, ry1, rx2, ry2)
    region_a = _mask_in_region(mask_a, feature_a, region)
    region_b = _mask_in_region(mask_b, feature_b, region)
    return _contact_by_masks(region_a, region_b, tolerance_px=tolerance_px)


def _contact_by_centers(
    feature_a: Dict,
    feature_b: Dict,
    tolerance_px: int = 3,
) -> bool:
    dx = float(feature_a["centroid_x"]) - float(feature_b["centroid_x"])
    dy = float(feature_a["centroid_y"]) - float(feature_b["centroid_y"])
    center_distance = float(np.hypot(dx, dy))
    radius_sum = (
        float(feature_a["equivalent_diameter_px"])
        + float(feature_b["equivalent_diameter_px"])
    ) / 2.0
    return center_distance <= radius_sum + tolerance_px


def detect_agglomerate_pairs(
    features: Sequence[Dict],
    masks: Optional[Sequence[np.ndarray]] = None,
    tolerance_px: int = 3,
    min_particle_diameter_px: float = 3.0,
    min_contact_ratio: float = 0.09,
    min_overlap_ratio: float = 0.03,
    pixel_size_um: Optional[float] = None,
) -> List[Tuple[int, int]]:
    evidence = detect_agglomerate_pair_evidence(
        features,
        masks=masks,
        tolerance_px=tolerance_px,
        min_particle_diameter_px=min_particle_diameter_px,
        min_contact_ratio=min_contact_ratio,
        min_overlap_ratio=min_overlap_ratio,
        pixel_size_um=pixel_size_um,
    )
    return [
        (int(item["particle_a"]), int(item["particle_b"]))
        for item in evidence
        if bool(item["accepted"])
    ]


def _candidate_pair_indices(
    features: Sequence[Dict], tolerance_px: int
) -> List[Tuple[int, int]]:
    """Use an x-axis sweep so dense images avoid a full O(N^2) pair scan."""
    ordered = sorted(
        range(len(features)), key=lambda idx: float(features[idx].get("bbox_x1", 0))
    )
    active: List[int] = []
    candidates: List[Tuple[int, int]] = []
    for current in ordered:
        current_left = float(features[current].get("bbox_x1", 0)) - tolerance_px
        active = [
            idx
            for idx in active
            if float(features[idx].get("bbox_x2", 0)) + tolerance_px >= current_left
        ]
        for other in active:
            if _expanded_bbox_overlaps(features[other], features[current], tolerance_px):
                candidates.append((other, current))
        active.append(current)
    return candidates


def detect_agglomerate_pair_evidence(
    features: Sequence[Dict],
    masks: Optional[Sequence[np.ndarray]] = None,
    tolerance_px: int = 3,
    min_particle_diameter_px: float = 3.0,
    min_contact_ratio: float = 0.09,
    min_overlap_ratio: float = 0.03,
    pixel_size_um: Optional[float] = None,
) -> List[Dict]:
    """Return auditable pair evidence instead of relying on distance alone."""
    if masks is not None and len(masks) != len(features):
        raise ValueError("masks and features must have the same length")

    diagnostics: List[Dict] = []
    min_particle_diameter_px = max(0.0, float(min_particle_diameter_px))
    min_contact_ratio = max(0.0, float(min_contact_ratio))
    min_overlap_ratio = max(0.0, float(min_overlap_ratio))
    for i, j in _candidate_pair_indices(features, tolerance_px):
        dia_i = float(features[i]["equivalent_diameter_px"])
        dia_j = float(features[j]["equivalent_diameter_px"])
        if dia_i <= 0 or dia_j <= 0 or min(dia_i, dia_j) < min_particle_diameter_px:
            continue

        if masks is not None:
            metrics = _mask_pair_evidence(
                masks[i],
                features[i],
                masks[j],
                features[j],
                tolerance_px=tolerance_px,
                pixel_size_um=pixel_size_um,
            )
            accepted_by_contact = (
                float(metrics["gap_px"]) <= float(tolerance_px) + 0.5
                and float(metrics["contact_ratio"]) >= min_contact_ratio
            )
            accepted_by_overlap = float(metrics["overlap_ratio"]) >= min_overlap_ratio
            accepted = accepted_by_contact or accepted_by_overlap
            if accepted_by_contact and accepted_by_overlap:
                reason = "contact_and_overlap"
            elif accepted_by_overlap:
                reason = "overlap"
            elif accepted_by_contact:
                reason = "contact_arc"
            else:
                reason = "weak_proximity"
        else:
            dx = float(features[i]["centroid_x"]) - float(features[j]["centroid_x"])
            dy = float(features[i]["centroid_y"]) - float(features[j]["centroid_y"])
            center_distance = float(np.hypot(dx, dy))
            radius_sum = (dia_i + dia_j) / 2.0
            gap_px = max(0.0, center_distance - radius_sum)
            metrics = {
                "gap_px": gap_px,
                "gap_um": float(gap_px * pixel_size_um) if pixel_size_um else None,
                "contact_ratio": 0.0,
                "overlap_ratio": 0.0,
                "center_distance_ratio": center_distance / radius_sum,
            }
            accepted = gap_px <= tolerance_px
            reason = "center_fallback" if accepted else "weak_proximity"

        diagnostics.append(
            {
                "particle_a": int(features[i]["particle_id"]),
                "particle_b": int(features[j]["particle_id"]),
                **metrics,
                "accepted": bool(accepted),
                "reason": reason,
            }
        )
    return diagnostics


def agglomerate_groups_from_pairs(
    pairs: Sequence[Tuple[int, int]],
    min_group_size: int = 3,
) -> List[Tuple[int, ...]]:
    adjacency: Dict[int, set[int]] = {}
    for left, right in pairs:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))

    groups: List[Tuple[int, ...]] = []
    visited: set[int] = set()
    for particle_id in sorted(adjacency):
        if particle_id in visited:
            continue
        stack = [particle_id]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency.get(current, set()) - component)

        visited |= component
        if len(component) >= min_group_size:
            groups.append(tuple(sorted(component)))
    return groups


def _expanded_bbox_overlaps(feature_a: Dict, feature_b: Dict, tolerance_px: int) -> bool:
    ax1 = float(feature_a.get("bbox_x1", 0)) - tolerance_px
    ay1 = float(feature_a.get("bbox_y1", 0)) - tolerance_px
    ax2 = float(feature_a.get("bbox_x2", 0)) + tolerance_px
    ay2 = float(feature_a.get("bbox_y2", 0)) + tolerance_px
    bx1 = float(feature_b.get("bbox_x1", 0)) - tolerance_px
    by1 = float(feature_b.get("bbox_y1", 0)) - tolerance_px
    bx2 = float(feature_b.get("bbox_x2", 0)) + tolerance_px
    by2 = float(feature_b.get("bbox_y2", 0)) + tolerance_px
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def classify_particles(
    features: Sequence[Dict],
    masks: Optional[Sequence[np.ndarray]] = None,
    hollow_threshold: float = 0.25,
    spherical_axis_ratio_threshold: float = 1.2,
    spherical_roundness_threshold: float = DEFAULT_SPHERICAL_ROUNDNESS_THRESHOLD,
    spherical_rule: str = "roundness",
    agglomerate_tolerance_px: int = 3,
    agglomerate_tolerance_um: Optional[float] = None,
    min_agglomerate_group_size: int = 3,
    agglomerate_min_contact_ratio: float = 0.09,
    agglomerate_min_overlap_ratio: float = 0.03,
    diameter_method: str = "feret_max",
    roundness_method: str = DEFAULT_ROUNDNESS_METHOD,
    spherical_roundness_method: str = DEFAULT_SPHERICAL_ROUNDNESS_METHOD,
    exclude_border_from_size_statistics: bool = False,
) -> Tuple[List[Dict], Dict]:
    if roundness_method not in ROUNDNESS_KEYS:
        raise ValueError(f"Unknown roundness method: {roundness_method}")
    if spherical_roundness_method not in SPHERICAL_ROUNDNESS_METHODS:
        raise ValueError(
            f"Unknown spherical roundness method: {spherical_roundness_method}"
        )
    if spherical_rule not in {"roundness", "axis_ratio", "hybrid"}:
        raise ValueError(f"Unknown spherical rule: {spherical_rule}")
    q_key, roundness_key = ROUNDNESS_KEYS[roundness_method]
    classified: List[Dict] = []
    for feature in features:
        item = dict(feature)
        item["q_value"] = float(item.get(q_key, item.get("q_value", 0.0)) or 0.0)
        item["roundness"] = float(
            item.get(roundness_key, item.get("roundness", 0.0)) or 0.0
        )
        item["roundness_method"] = roundness_method
        (
            item["spherical_roundness"],
            item["spherical_roundness_fallback"],
        ) = _spherical_roundness_value(
            item,
            method=spherical_roundness_method,
            report_roundness_key=roundness_key,
        )
        item["spherical_roundness_method"] = spherical_roundness_method
        item["statistical_diameter_um"] = diameter_value(item, method=diameter_method)
        item["statistical_diameter_method"] = diameter_method
        # 空心粉：孔隙面积 >= 颗粒总面积的 25%。
        item["is_hollow"] = float(item["hole_ratio"]) >= hollow_threshold
        # 球形颗粒默认按参考统计口径使用圆形度阈值；轴比规则保留为可选项。
        roundness_spherical = (
            float(item["spherical_roundness"])
            >= float(spherical_roundness_threshold)
        )
        axis_ratio_spherical = (
            float(item["axis_ratio"]) <= float(spherical_axis_ratio_threshold)
        )
        if spherical_rule == "roundness":
            item["is_spherical"] = roundness_spherical
        elif spherical_rule == "axis_ratio":
            item["is_spherical"] = axis_ratio_spherical
        else:
            item["is_spherical"] = roundness_spherical and axis_ratio_spherical
        item["spherical_rule"] = spherical_rule
        item["is_agglomerate"] = False
        item["agglomerate_group_id"] = ""
        item["agglomerate_group_size"] = 0
        item["agglomerate_contact_degree"] = 0
        item["agglomerate_max_contact_ratio"] = 0.0
        item["agglomerate_max_overlap_ratio"] = 0.0
        item["agglomerate_min_gap_um"] = None
        classified.append(item)

    (
        resolved_tolerance_px,
        effective_tolerance_um,
        inferred_pixel_size_um,
        tolerance_source,
    ) = _resolve_agglomerate_tolerance(
        classified,
        fallback_px=agglomerate_tolerance_px,
        tolerance_um=agglomerate_tolerance_um,
    )
    pair_evidence = detect_agglomerate_pair_evidence(
        classified,
        masks=masks,
        tolerance_px=resolved_tolerance_px,
        min_contact_ratio=agglomerate_min_contact_ratio,
        min_overlap_ratio=agglomerate_min_overlap_ratio,
        pixel_size_um=inferred_pixel_size_um,
    )
    pairs = [
        (int(item["particle_a"]), int(item["particle_b"]))
        for item in pair_evidence
        if bool(item["accepted"])
    ]
    groups = agglomerate_groups_from_pairs(
        pairs,
        min_group_size=min_agglomerate_group_size,
    )
    group_id_by_particle = {
        particle_id: group_idx
        for group_idx, group in enumerate(groups, start=1)
        for particle_id in group
    }
    group_size_by_particle = {
        particle_id: len(group)
        for group in groups
        for particle_id in group
    }
    item_by_id = {int(item["particle_id"]): item for item in classified}
    for item in classified:
        group_id = group_id_by_particle.get(int(item["particle_id"]))
        if group_id is not None:
            item["is_agglomerate"] = True
            item["agglomerate_group_id"] = group_id
            item["agglomerate_group_size"] = group_size_by_particle[int(item["particle_id"])]

    accepted_group_evidence: List[Dict] = []
    for evidence in pair_evidence:
        if not bool(evidence["accepted"]):
            continue
        left = int(evidence["particle_a"])
        right = int(evidence["particle_b"])
        if group_id_by_particle.get(left) != group_id_by_particle.get(right):
            continue
        if group_id_by_particle.get(left) is None:
            continue
        accepted_group_evidence.append(evidence)
        for particle_id in (left, right):
            item = item_by_id[particle_id]
            item["agglomerate_contact_degree"] += 1
            item["agglomerate_max_contact_ratio"] = max(
                float(item["agglomerate_max_contact_ratio"]),
                float(evidence["contact_ratio"]),
            )
            item["agglomerate_max_overlap_ratio"] = max(
                float(item["agglomerate_max_overlap_ratio"]),
                float(evidence["overlap_ratio"]),
            )
            gap_um = evidence.get("gap_um")
            if gap_um is not None and math.isfinite(float(gap_um)):
                previous_gap = item.get("agglomerate_min_gap_um")
                item["agglomerate_min_gap_um"] = (
                    float(gap_um)
                    if previous_gap is None
                    else min(float(previous_gap), float(gap_um))
                )

    group_details: List[Dict] = []
    for group_idx, group in enumerate(groups, start=1):
        group_set = set(group)
        edge_rows = [
            item
            for item in accepted_group_evidence
            if int(item["particle_a"]) in group_set and int(item["particle_b"]) in group_set
        ]
        gaps_um = [
            float(item["gap_um"])
            for item in edge_rows
            if item.get("gap_um") is not None and math.isfinite(float(item["gap_um"]))
        ]
        group_details.append(
            {
                "group_id": group_idx,
                "particle_ids": group,
                "particle_count": len(group),
                "strong_pair_count": len(edge_rows),
                "mean_contact_ratio": (
                    float(np.mean([float(item["contact_ratio"]) for item in edge_rows]))
                    if edge_rows
                    else 0.0
                ),
                "max_overlap_ratio": max(
                    [float(item["overlap_ratio"]) for item in edge_rows],
                    default=0.0,
                ),
                "min_gap_um": min(gaps_um) if gaps_um else None,
            }
        )

    total = len(classified)
    spherical_count = sum(1 for item in classified if item["is_spherical"])
    hollow_count = sum(1 for item in classified if item["is_hollow"])
    agglomerate_count = sum(1 for item in classified if item["is_agglomerate"])
    agglomerate_group_count = len(groups)
    total_area_um2 = sum(float(item.get("area_um2", 0.0)) for item in classified)
    agglomerate_area_um2 = sum(
        float(item.get("area_um2", 0.0))
        for item in classified
        if item["is_agglomerate"]
    )
    mean_q = gbt8170_round(
        (
            sum(float(item.get("q_value", 0.0)) for item in classified)
            / total
        )
        if total
        else 0.0,
        ndigits=4,
    )
    mean_roundness = gbt8170_round(
        (
            sum(float(item.get("roundness", 0.0)) for item in classified)
            / total
        )
        if total
        else 0.0,
        ndigits=4,
    )
    s_percent = gbt8170_round(
        (spherical_count / total * 100.0) if total else 0.0,
        ndigits=2,
    )
    hollow_percent = gbt8170_round(
        (hollow_count / total * 100.0) if total else 0.0,
        ndigits=2,
    )
    agglomerate_percent = gbt8170_round(
        (agglomerate_count / total * 100.0) if total else 0.0,
        ndigits=2,
    )
    agglomerate_area_percent = gbt8170_round(
        (agglomerate_area_um2 / total_area_um2 * 100.0) if total_area_um2 else 0.0,
        ndigits=2,
    )

    size_stats = summarize_size_distribution(
        classified,
        method=diameter_method,
        exclude_border=exclude_border_from_size_statistics,
    )
    stats = {
        "total_particles": total,
        "mean_sphericity_q": mean_q,
        "mean_sphericity_q_text": f"{mean_q:.4f}",
        "mean_roundness": mean_roundness,
        "mean_roundness_text": f"{mean_roundness:.4f}",
        "roundness_method": roundness_method,
        "spherical_rule": spherical_rule,
        "spherical_roundness_method": spherical_roundness_method,
        "spherical_roundness_threshold": float(spherical_roundness_threshold),
        "spherical_roundness_fallback_particles": sum(
            1 for item in classified if item["spherical_roundness_fallback"]
        ),
        "spherical_axis_ratio_threshold": float(spherical_axis_ratio_threshold),
        "spherical_particles": spherical_count,
        "hollow_particles": hollow_count,
        "agglomerate_particles": agglomerate_count,
        "agglomerate_group_count": agglomerate_group_count,
        "agglomerate_groups": groups,
        "agglomerate_group_details": group_details,
        "agglomerate_pairs": pairs,
        "agglomerate_pair_evidence": pair_evidence,
        "agglomerate_candidate_pair_count": len(pair_evidence),
        "agglomerate_accepted_pair_count": len(pairs),
        "agglomerate_group_pair_count": len(accepted_group_evidence),
        "agglomerate_tolerance_px": resolved_tolerance_px,
        "agglomerate_tolerance_um": agglomerate_tolerance_um,
        "agglomerate_effective_tolerance_um": effective_tolerance_um,
        "agglomerate_pixel_size_um": inferred_pixel_size_um,
        "agglomerate_tolerance_source": tolerance_source,
        "agglomerate_min_contact_ratio": float(agglomerate_min_contact_ratio),
        "agglomerate_min_overlap_ratio": float(agglomerate_min_overlap_ratio),
        "agglomerate_min_group_size": int(min_agglomerate_group_size),
        "agglomerate_definition": (
            "physical_gap_and_strong_contact_component"
            if masks is not None
            else "center_distance_fallback_component"
        ),
        "hollow_rate_percent": hollow_percent,
        "hollow_rate_text": f"{hollow_percent:.2f}%",
        "agglomerate_rate_percent": agglomerate_percent,
        "agglomerate_rate_text": f"{agglomerate_percent:.2f}%",
        "agglomerate_area_rate_percent": agglomerate_area_percent,
        "agglomerate_area_rate_text": f"{agglomerate_area_percent:.2f}%",
        # 球形率 S = n / N * 100%，按 GB/T 8170 保留两位小数。
        "sphericity_rate_s_percent": s_percent,
        "sphericity_rate_s_text": f"{s_percent:.2f}%",
        **size_stats,
    }
    for key in (
        "diameter_mean_um",
        "diameter_min_um",
        "diameter_d5_um",
        "diameter_d10_um",
        "diameter_d50_um",
        "diameter_d90_um",
        "diameter_d95_um",
        "diameter_max_um",
    ):
        stats[f"{key}_text"] = f"{float(stats.get(key, 0.0)):.4f}"
    return classified, stats


def format_stats(stats: Dict) -> str:
    return (
        f"总颗粒数: {stats['total_particles']}\n"
        f"平均球形度 Q: {stats['mean_sphericity_q_text']}\n"
        f"空心粉率 K: {stats['hollow_rate_text']} ({stats['hollow_particles']} / {stats['total_particles']})\n"
        f"团聚体数 N_agglom_group: {stats.get('agglomerate_group_count', 0)}\n"
        f"团聚率 P_agglom: {stats['agglomerate_rate_text']} ({stats['agglomerate_particles']} / {stats['total_particles']})\n"
        f"团聚面积率 P_area: {stats['agglomerate_area_rate_text']}\n"
        f"球形颗粒率 S: {stats['sphericity_rate_s_text']}\n"
    )
