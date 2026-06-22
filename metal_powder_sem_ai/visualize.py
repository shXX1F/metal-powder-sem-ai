from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .preprocess import imwrite_unicode
from .segment import InstanceMask


COLOR_BGR = {
    "spherical": (0, 180, 0),
    "non_spherical": (0, 0, 255),
    "agglomerate": (0, 220, 255),
}


def class_name_and_color(feature: Dict) -> Tuple[str, Tuple[int, int, int]]:
    # 当前 GUI 暂不展示空心粉识别，避免把表面孔洞误读为严格空心粉。
    if feature.get("is_agglomerate"):
        return "团聚体", COLOR_BGR["agglomerate"]
    if feature.get("is_spherical"):
        return "球形", COLOR_BGR["spherical"]
    return "非球形", COLOR_BGR["non_spherical"]


def draw_instances(
    image_bgr: np.ndarray,
    instances: Sequence[InstanceMask],
    features: Sequence[Dict],
    alpha: float = 0.30,
    draw_label: bool = True,
) -> np.ndarray:
    if len(instances) != len(features):
        raise ValueError("instances 与 features 数量必须一致。")

    output = image_bgr.copy()
    overlay = image_bgr.copy()

    for instance, feature in zip(instances, features):
        class_name, color = class_name_and_color(feature)
        mask = instance.mask > 0
        if mask.shape[:2] == overlay.shape[:2]:
            overlay[mask] = color
        else:
            x1, y1, x2, y2 = instance.bbox_xyxy
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(overlay.shape[1], int(x2))
            y2 = min(overlay.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                local = mask[: y2 - y1, : x2 - x1]
                overlay[y1:y2, x1:x2][local] = color
        cv2.drawContours(output, [instance.contour], -1, color, thickness=2)

        if draw_label:
            x1, y1, _, _ = instance.bbox_xyxy
            text = (
                f"#{int(feature['particle_id'])} {class_name} "
                f"Q={feature['q_value']:.3f} R={feature['axis_ratio']:.2f}"
            )
            y = max(12, int(y1) - 4)
            cv2.putText(
                output,
                text,
                (int(x1), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    legend = [
        ("球形", COLOR_BGR["spherical"]),
        ("非球形", COLOR_BGR["non_spherical"]),
        ("团聚体", COLOR_BGR["agglomerate"]),
    ]
    x, y = 12, 24
    for label, color in legend:
        cv2.rectangle(output, (x, y - 12), (x + 18, y + 4), color, thickness=-1)
        cv2.putText(
            output,
            label,
            (x + 26, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return output


def save_visualization(path: str | Path, image_bgr: np.ndarray) -> None:
    imwrite_unicode(path, image_bgr)
