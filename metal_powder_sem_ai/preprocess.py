from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


ImageLike = Union[str, Path, np.ndarray]


@dataclass
class PreprocessResult:
    image: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    crop_offset_xy: Tuple[int, int] = (0, 0)
    inverted: bool = False


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """支持 Windows 中文路径的 OpenCV 读图。"""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image


def imwrite_unicode(path: Union[str, Path], image: np.ndarray) -> None:
    """支持 Windows 中文路径的 OpenCV 写图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    encoded.tofile(str(path))


def load_image(image: ImageLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.copy()
    return imread_unicode(image, flags=flags)


def crop_sem_area(
    image: np.ndarray,
    crop_bottom_fraction: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """裁掉底部 SEM 信息栏/比例尺区域；默认不裁剪，比例尺建议手动输入。"""
    if crop_bottom_fraction <= 0:
        return image.copy(), (0, 0)
    if crop_bottom_fraction >= 0.5:
        raise ValueError("crop_bottom_fraction 建议小于 0.5，避免裁掉有效颗粒区域。")
    h = image.shape[0]
    keep_h = int(round(h * (1.0 - crop_bottom_fraction)))
    return image[:keep_h].copy(), (0, 0)


def preprocess_sem_image(
    image: ImageLike,
    crop_bottom_fraction: float = 0.0,
    blur_ksize: int = 5,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    invert: Optional[bool] = None,
    morph_kernel_size: int = 3,
) -> PreprocessResult:
    """SEM 图像预处理：高斯去噪、CLAHE 增强、Otsu 自适应阈值二值化。"""
    bgr = load_image(image, flags=cv2.IMREAD_COLOR)
    cropped, offset = crop_sem_area(bgr, crop_bottom_fraction=crop_bottom_fraction)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    denoised = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
    )
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_fraction = float(np.mean(binary == 255))
    should_invert = invert
    if should_invert is None:
        # SEM 金属粉末通常是“亮颗粒 + 暗背景”；若白色区域过大，说明阈值结果可能反了。
        should_invert = white_fraction > 0.60

    inverted = bool(should_invert)
    if should_invert:
        binary = cv2.bitwise_not(binary)

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return PreprocessResult(
        image=cropped,
        gray=gray,
        denoised=denoised,
        enhanced=enhanced,
        binary=binary,
        crop_offset_xy=offset,
        inverted=inverted,
    )


def save_preprocess_result(result: PreprocessResult, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(output_dir / "01_gray.png", result.gray)
    imwrite_unicode(output_dir / "02_denoised.png", result.denoised)
    imwrite_unicode(output_dir / "03_enhanced.png", result.enhanced)
    imwrite_unicode(output_dir / "04_binary.png", result.binary)

