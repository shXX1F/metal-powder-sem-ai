from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def gbt8170_round(value: float, ndigits: int = 2) -> float:
    """GB/T 8170 常用修约：四舍六入五留双，等价于 ROUND_HALF_EVEN。"""
    quant = Decimal("1").scaleb(-ndigits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_EVEN))


def _contact_by_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    tolerance_px: int = 3,
) -> bool:
    kernel_size = max(1, tolerance_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_a = cv2.dilate((mask_a > 0).astype(np.uint8), kernel, iterations=1) > 0
    return bool(np.any(dilated_a & (mask_b > 0)))


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
) -> List[Tuple[int, int]]:
    """团聚体判定：大球附着小球，且小球直径 > 大球直径 * 0.5。"""
    pairs: List[Tuple[int, int]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if not _expanded_bbox_overlaps(features[i], features[j], tolerance_px):
                continue
            dia_i = float(features[i]["equivalent_diameter_px"])
            dia_j = float(features[j]["equivalent_diameter_px"])
            if dia_i <= 0 or dia_j <= 0:
                continue
            big = max(dia_i, dia_j)
            small = min(dia_i, dia_j)
            diameter_rule = small > big * 0.5
            if not diameter_rule:
                continue

            if masks is not None:
                touching = _contact_by_masks_in_bboxes(
                    masks[i],
                    features[i],
                    masks[j],
                    features[j],
                    tolerance_px=tolerance_px,
                )
            else:
                touching = _contact_by_centers(
                    features[i],
                    features[j],
                    tolerance_px=tolerance_px,
                )
            if touching:
                pairs.append((int(features[i]["particle_id"]), int(features[j]["particle_id"])))
    return pairs


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
    agglomerate_tolerance_px: int = 3,
    min_agglomerate_group_size: int = 3,
) -> Tuple[List[Dict], Dict]:
    classified: List[Dict] = []
    for feature in features:
        item = dict(feature)
        # 空心粉：孔隙面积 >= 颗粒总面积的 25%。
        item["is_hollow"] = float(item["hole_ratio"]) >= hollow_threshold
        # 球形颗粒：长轴 / 短轴 <= 1.2。
        item["is_spherical"] = float(item["axis_ratio"]) <= spherical_axis_ratio_threshold
        item["is_agglomerate"] = False
        item["agglomerate_group_id"] = ""
        classified.append(item)

    pairs = detect_agglomerate_pairs(
        classified,
        masks=masks,
        tolerance_px=agglomerate_tolerance_px,
    )
    groups = agglomerate_groups_from_pairs(
        pairs,
        min_group_size=min_agglomerate_group_size,
    )
    group_id_by_particle = {
        particle_id: group_idx
        for group_idx, group in enumerate(groups, start=1)
        for particle_id in group
    }
    for item in classified:
        group_id = group_id_by_particle.get(int(item["particle_id"]))
        if group_id is not None:
            item["is_agglomerate"] = True
            item["agglomerate_group_id"] = group_id

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

    stats = {
        "total_particles": total,
        "mean_sphericity_q": mean_q,
        "mean_sphericity_q_text": f"{mean_q:.4f}",
        "spherical_particles": spherical_count,
        "hollow_particles": hollow_count,
        "agglomerate_particles": agglomerate_count,
        "agglomerate_group_count": agglomerate_group_count,
        "agglomerate_groups": groups,
        "agglomerate_pairs": pairs,
        "hollow_rate_percent": hollow_percent,
        "hollow_rate_text": f"{hollow_percent:.2f}%",
        "agglomerate_rate_percent": agglomerate_percent,
        "agglomerate_rate_text": f"{agglomerate_percent:.2f}%",
        "agglomerate_area_rate_percent": agglomerate_area_percent,
        "agglomerate_area_rate_text": f"{agglomerate_area_percent:.2f}%",
        # 球形率 S = n / N * 100%，按 GB/T 8170 保留两位小数。
        "sphericity_rate_s_percent": s_percent,
        "sphericity_rate_s_text": f"{s_percent:.2f}%",
    }
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
