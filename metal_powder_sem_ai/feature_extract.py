from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .segment import InstanceMask


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else default


def _axis_lengths_px(contour: np.ndarray) -> Tuple[float, float, str]:
    """用 fitEllipse 优先求长短轴；点数不足时退化为最小外接矩形。"""
    if len(contour) >= 5:
        # fitEllipse 返回椭圆的两个轴长，按长轴/短轴排序用于国标轴比判定。
        _, axes, _ = cv2.fitEllipse(contour)
        major = float(max(axes))
        minor = float(min(axes))
        return major, minor, "fitEllipse"

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    major = float(max(width, height))
    minor = float(min(width, height))
    return major, minor, "minAreaRect"


def _geometric_hole_area_px(mask: np.ndarray) -> int:
    """统计 mask 内部黑色连通域面积，即几何孔洞面积。"""
    mask_u8 = (mask > 0).astype(np.uint8)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=1)
    holes = flood == 0
    return int(holes[1:-1, 1:-1].sum())


def _grayscale_dark_pore_area_px(
    mask: np.ndarray,
    gray: Optional[np.ndarray],
    min_hole_area_px: int = 6,
) -> int:
    """在颗粒内部依据灰度寻找暗孔洞，适配实例 mask 已填充的情况。"""
    if gray is None:
        return 0

    mask_bool = mask > 0
    if int(mask_bool.sum()) < min_hole_area_px:
        return 0

    # 轻微腐蚀可减少颗粒边缘阴影被误认为孔洞。
    inner = cv2.erode(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    inner_bool = inner > 0
    values = gray[inner_bool]
    if values.size < min_hole_area_px:
        return 0

    otsu_threshold, _ = cv2.threshold(
        values.reshape(-1, 1).astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    dark = (gray < otsu_threshold) & inner_bool
    dark = _remove_small_components(dark, min_size=min_hole_area_px)
    return int(dark.sum())


def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)
    kept = np.zeros(binary_u8.shape, dtype=bool)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_size:
            kept |= labels == label_id
    return kept


def detect_hole_area_px(
    mask: np.ndarray,
    gray: Optional[np.ndarray] = None,
    min_hole_area_px: int = 6,
) -> int:
    geometry_area = _geometric_hole_area_px(mask)
    grayscale_area = _grayscale_dark_pore_area_px(
        mask,
        gray=gray,
        min_hole_area_px=min_hole_area_px,
    )
    return int(max(geometry_area, grayscale_area))


def _gray_for_instance(
    gray: Optional[np.ndarray],
    instance: InstanceMask,
) -> Optional[np.ndarray]:
    if gray is None:
        return None
    if instance.mask.shape[:2] == gray.shape[:2]:
        return gray
    x1, y1, x2, y2 = instance.bbox_xyxy
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(gray.shape[1], int(x2))
    y2 = min(gray.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]


def extract_particle_features(
    instance: InstanceMask,
    pixel_size_um: float,
    gray: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um 必须大于 0，例如 1 pixel = 0.1 um 则传入 0.1。")

    mask = instance.mask > 0
    contour = instance.contour
    area_px = int(mask.sum())
    area_um2 = float(area_px * pixel_size_um * pixel_size_um)

    perimeter_px = float(cv2.arcLength(contour, closed=True))
    perimeter_um = float(perimeter_px * pixel_size_um)

    # 球形度 Q：严格采用金相法公式 Q = 4*pi*A / P^2。
    q_value = _safe_divide(4.0 * math.pi * area_um2, perimeter_um * perimeter_um)

    major_px, minor_px, axis_method = _axis_lengths_px(contour)
    major_um = major_px * pixel_size_um
    minor_um = minor_px * pixel_size_um
    axis_ratio = _safe_divide(major_um, minor_um, default=float("inf"))

    hole_area_px = detect_hole_area_px(
        mask.astype(np.uint8),
        gray=_gray_for_instance(gray, instance),
    )
    hole_area_um2 = float(hole_area_px * pixel_size_um * pixel_size_um)
    hole_ratio = _safe_divide(hole_area_px, area_px)

    equivalent_diameter_px = math.sqrt(4.0 * area_px / math.pi) if area_px > 0 else 0.0
    equivalent_diameter_um = equivalent_diameter_px * pixel_size_um

    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        x1, y1, x2, y2 = instance.bbox_xyxy
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

    x1, y1, x2, y2 = instance.bbox_xyxy
    return {
        "particle_id": int(instance.particle_id),
        "score": float(instance.score),
        "area_px": area_px,
        "area_um2": area_um2,
        "perimeter_px": perimeter_px,
        "perimeter_um": perimeter_um,
        "q_value": q_value,
        "major_axis_px": major_px,
        "minor_axis_px": minor_px,
        "major_axis_um": major_um,
        "minor_axis_um": minor_um,
        "axis_ratio": axis_ratio,
        "axis_method": axis_method,
        "hole_area_px": hole_area_px,
        "hole_area_um2": hole_area_um2,
        "hole_ratio": hole_ratio,
        "equivalent_diameter_px": equivalent_diameter_px,
        "equivalent_diameter_um": equivalent_diameter_um,
        "centroid_x": cx,
        "centroid_y": cy,
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
        "bbox_x2": int(x2),
        "bbox_y2": int(y2),
    }


def extract_all_features(
    instances: Sequence[InstanceMask],
    pixel_size_um: float,
    gray: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    return [
        extract_particle_features(instance, pixel_size_um=pixel_size_um, gray=gray)
        for instance in instances
    ]
