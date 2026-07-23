#!/usr/bin/env python3
"""Run particle + hollow inference and report hollow-powder ratio.

Input is a plain image folder, for example:

    /root/code/empty_heart/expirement/give/xxx

Output is written to:

    /root/code/empty_heart/expirement/back/xxx

The tool intentionally treats the hollow model as a high-recall candidate
generator. A traditional image-processing filter then keeps only candidates
whose internal dark-area ratio reaches the requested threshold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from . import config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    import config  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RED = (0, 0, 255)
BLUE = (255, 0, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from a Windows path that may contain non-ASCII text."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Write an image to a Windows path that may contain non-ASCII text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def load_module(path: str | Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def natural_key(path: Path):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def image_files(src_dir: Path) -> list[Path]:
    return sorted((p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def default_output_dir(input_dir: Path) -> Path:
    give_root = Path("/root/code/empty_heart/expirement/give")
    back_root = Path("/root/code/empty_heart/expirement/back")
    try:
        rel = input_dir.resolve().relative_to(give_root)
        return back_root / rel
    except ValueError:
        return back_root / input_dir.name


def mask_contours(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def particle_mask_mean_gray(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean gray value inside one predicted particle mask.

    Pure black artifacts can be detected cheaply in image space: if the whole
    predicted particle region is dark on average, it is not a real white powder
    particle and should not enter the denominator.
    """
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray[mask_bool].mean())


def filter_dark_particle_predictions(
    image_bgr: np.ndarray,
    predictions: list,
    dark_mean_threshold: float,
) -> tuple[list, int]:
    """Remove particle predictions whose mask-average gray is in the black range.

    The threshold is an inclusive upper bound: mean_gray <= threshold means the
    candidate is treated as a pure-black false particle.
    """
    kept = []
    removed = 0
    for pred in predictions:
        mean_gray = particle_mask_mean_gray(image_bgr, pred.mask)
        setattr(pred, "mean_gray", mean_gray)
        if mean_gray <= dark_mean_threshold:
            removed += 1
            continue
        kept.append(pred)
    return kept, removed


def prediction_touches_image_boundary(pred, image_width: int, image_height: int, margin: int) -> bool:
    """Return True if a particle bbox touches the original image boundary."""
    bbox = getattr(pred, "bbox", None)
    if bbox is None:
        return True
    margin = max(0, int(margin))
    return (
        bbox[0] <= margin
        or bbox[1] <= margin
        or bbox[2] >= image_width - margin
        or bbox[3] >= image_height - margin
    )


def filter_particle_image_boundary_predictions(
    predictions: list,
    image_width: int,
    image_height: int,
    margin: int,
) -> tuple[list, int]:
    """Remove whole particle predictions clipped by the original image edge."""
    kept = []
    removed = 0
    for pred in predictions:
        if prediction_touches_image_boundary(pred, image_width, image_height, margin):
            removed += 1
            continue
        kept.append(pred)
    return kept, removed


def preview_tiles(width: int, height: int, cfg: config.InferenceConfig) -> list[tuple[int, int, int, int]]:
    tile_w = min(int(cfg.tile_w), width)
    tile_h = min(int(cfg.tile_h), height)
    xs = [0] if cfg.tile_cols <= 1 else [round(i * (width - tile_w) / (cfg.tile_cols - 1)) for i in range(cfg.tile_cols)]
    ys = [0] if cfg.tile_rows <= 1 else [round(i * (height - tile_h) / (cfg.tile_rows - 1)) for i in range(cfg.tile_rows)]
    return list(
        dict.fromkeys(
            (int(x), int(y), int(x + tile_w), int(y + tile_h))
            for y in ys
            for x in xs
        )
    )


def draw_tile_boundaries(image: np.ndarray, cfg: config.InferenceConfig) -> None:
    height, width = image.shape[:2]
    for x0, y0, x1, y1 in preview_tiles(width, height, cfg):
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), BLUE, 1)


def draw_text_box(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.58) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(x, image.shape[1] - tw - 6))
    y = max(th + baseline + 4, min(y, image.shape[0] - 4))
    cv2.rectangle(image, (x - 3, y - th - baseline - 4), (x + tw + 5, y + baseline + 3), BLACK, -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def build_model(eval_mod: ModuleType, model_path: Path, cfg: config.InferenceConfig, device: torch.device):
    model = eval_mod.build_model(
        cfg.model_min_size,
        cfg.model_max_size,
        cfg.anchor_sizes,
        cfg.detections_per_img,
    )
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_views(
    eval_mod: ModuleType,
    model,
    device: torch.device,
    images: list[Image.Image],
    score_thresholds: list[float],
    mask_threshold: float,
    sources: list[str],
) -> list[list]:
    """Predict views in memory-safe chunks while preserving their original order."""
    if not images:
        return []
    view_batch_size = 1 if device.type == "cpu" else len(images)
    predictions: list[list] = []
    for start in range(0, len(images), view_batch_size):
        end = start + view_batch_size
        predictions.extend(
            eval_mod.predict_on_images(
                model,
                device,
                images[start:end],
                score_thresholds[start:end],
                mask_threshold,
                sources[start:end],
            )
        )
    return predictions


@torch.no_grad()
def infer_one_image(
    eval_mod: ModuleType,
    model,
    device: torch.device,
    image: Image.Image,
    cfg: config.InferenceConfig,
) -> list:
    width, height = image.size
    tiles = eval_mod.make_tiles(width, height, cfg.tile_cols, cfg.tile_rows, cfg.tile_w, cfg.tile_h)
    tile_images = [image.crop(tile) for tile in tiles]
    view_predictions = predict_views(
        eval_mod,
        model,
        device,
        [image] + tile_images,
        [cfg.full_score_threshold] + [cfg.tile_score_threshold] * len(tile_images),
        cfg.mask_threshold,
        ["full"] + ["tile"] * len(tile_images),
    )

    full_preds = view_predictions[0]
    for pred in full_preds:
        pred.source = "full"
    final_preds = list(full_preds)

    for tile_index, (tile, tile_preds) in enumerate(zip(tiles, view_predictions[1:])):
        for pred in tile_preds:
            pred.tile_index = tile_index
            pasted = eval_mod.paste_tile_prediction(pred, tile, (height, width))
            if not eval_mod.bbox_inside_image_safe_region(
                pasted.mask, width, height, cfg.image_boundary_margin
            ):
                continue
            if not eval_mod.bbox_inside_tile_safe_region(pasted.mask, tile, cfg.tile_boundary_margin):
                continue
            max_intersection, max_iou = eval_mod.max_overlap_with_existing(pasted.mask, final_preds)
            if max_intersection <= cfg.tile_max_intersection_ratio and max_iou <= cfg.tile_max_iou:
                final_preds.append(pasted)
    return final_preds


@torch.no_grad()
def infer_image_batch(
    eval_mod: ModuleType,
    model,
    device: torch.device,
    images: list[tuple[Path, Image.Image]],
    cfg: config.InferenceConfig,
) -> dict[str, list]:
    """Infer several original images in one model forward over their full/tile views.

    Each source image contributes one full view plus `tile_cols * tile_rows` tile
    views. The detection model already accepts a list of variable-size tensors,
    so batching across images is the safest way to improve GPU utilization
    without launching competing CUDA processes.
    """
    if not images:
        return {}

    all_views: list[Image.Image] = []
    all_score_thresholds: list[float] = []
    all_sources: list[str] = []
    plans = []
    cursor = 0
    for image_path, image in images:
        width, height = image.size
        tiles = eval_mod.make_tiles(width, height, cfg.tile_cols, cfg.tile_rows, cfg.tile_w, cfg.tile_h)
        tile_images = [image.crop(tile) for tile in tiles]
        views = [image] + tile_images
        plans.append(
            {
                "name": image_path.name,
                "width": width,
                "height": height,
                "tiles": tiles,
                "start": cursor,
                "count": len(views),
            }
        )
        cursor += len(views)
        all_views.extend(views)
        all_score_thresholds.extend([cfg.full_score_threshold] + [cfg.tile_score_threshold] * len(tile_images))
        all_sources.extend(["full"] + ["tile"] * len(tile_images))

    view_predictions = predict_views(
        eval_mod,
        model,
        device,
        all_views,
        all_score_thresholds,
        cfg.mask_threshold,
        all_sources,
    )

    result: dict[str, list] = {}
    for plan in plans:
        preds_for_image = view_predictions[plan["start"] : plan["start"] + plan["count"]]
        full_preds = preds_for_image[0]
        for pred in full_preds:
            pred.source = "full"
        final_preds = list(full_preds)
        width = int(plan["width"])
        height = int(plan["height"])
        for tile_index, (tile, tile_preds) in enumerate(zip(plan["tiles"], preds_for_image[1:])):
            for pred in tile_preds:
                pred.tile_index = tile_index
                pasted = eval_mod.paste_tile_prediction(pred, tile, (height, width))
                if not eval_mod.bbox_inside_image_safe_region(
                    pasted.mask, width, height, cfg.image_boundary_margin
                ):
                    continue
                if not eval_mod.bbox_inside_tile_safe_region(pasted.mask, tile, cfg.tile_boundary_margin):
                    continue
                max_intersection, max_iou = eval_mod.max_overlap_with_existing(pasted.mask, final_preds)
                if max_intersection <= cfg.tile_max_intersection_ratio and max_iou <= cfg.tile_max_iou:
                    final_preds.append(pasted)
        result[str(plan["name"])] = final_preds
    return result


def with_model_size(cfg: config.InferenceConfig, model_min_size: int, model_max_size: int) -> config.InferenceConfig:
    values = asdict(cfg)
    values["model_min_size"] = int(model_min_size)
    values["model_max_size"] = int(model_max_size)
    return config.InferenceConfig(**values)


@torch.no_grad()
def infer_image_batch_split_models(
    eval_mod: ModuleType,
    full_model,
    tile_model,
    device: torch.device,
    images: list[tuple[Path, Image.Image]],
    cfg: config.InferenceConfig,
) -> dict[str, list]:
    """Infer full images and tile images with different model transform scales.

    This is mainly for particle: the full image can keep the training-like
    512/1024 scale while local tiles use a larger 1536/2048 transform to
    magnify small particles. Fusion remains identical to infer_image_batch.
    """
    if not images:
        return {}

    full_views = [image for _, image in images]
    full_predictions = predict_views(
        eval_mod,
        full_model,
        device,
        full_views,
        [cfg.full_score_threshold] * len(full_views),
        cfg.mask_threshold,
        ["full"] * len(full_views),
    )

    tile_views: list[Image.Image] = []
    plans = []
    cursor = 0
    for image_path, image in images:
        width, height = image.size
        tiles = eval_mod.make_tiles(width, height, cfg.tile_cols, cfg.tile_rows, cfg.tile_w, cfg.tile_h)
        crops = [image.crop(tile) for tile in tiles]
        plans.append(
            {
                "name": image_path.name,
                "width": width,
                "height": height,
                "tiles": tiles,
                "start": cursor,
                "count": len(crops),
            }
        )
        cursor += len(crops)
        tile_views.extend(crops)

    tile_predictions = predict_views(
        eval_mod,
        tile_model,
        device,
        tile_views,
        [cfg.tile_score_threshold] * len(tile_views),
        cfg.mask_threshold,
        ["tile"] * len(tile_views),
    )

    result: dict[str, list] = {}
    for image_index, plan in enumerate(plans):
        full_preds = full_predictions[image_index]
        for pred in full_preds:
            pred.source = "full"
        final_preds = list(full_preds)
        width = int(plan["width"])
        height = int(plan["height"])
        preds_for_tiles = tile_predictions[plan["start"] : plan["start"] + plan["count"]]
        for tile_index, (tile, tile_preds) in enumerate(zip(plan["tiles"], preds_for_tiles)):
            for pred in tile_preds:
                pred.tile_index = tile_index
                pasted = eval_mod.paste_tile_prediction(pred, tile, (height, width))
                if not eval_mod.bbox_inside_image_safe_region(
                    pasted.mask, width, height, cfg.image_boundary_margin
                ):
                    continue
                if not eval_mod.bbox_inside_tile_safe_region(pasted.mask, tile, cfg.tile_boundary_margin):
                    continue
                max_intersection, max_iou = eval_mod.max_overlap_with_existing(pasted.mask, final_preds)
                if max_intersection <= cfg.tile_max_intersection_ratio and max_iou <= cfg.tile_max_iou:
                    final_preds.append(pasted)
        result[str(plan["name"])] = final_preds
    return result


def batched(items: list[Path], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def set_filter_params(filter_mod: ModuleType, args) -> None:
    filter_mod.PARAMS["min_component_area_ratio"] = args.min_component_area_ratio
    filter_mod.PARAMS["white_threshold"] = args.hollow_white_threshold
    filter_mod.PARAMS["white_max_channel_diff"] = args.hollow_white_max_channel_diff
    filter_mod.PARAMS["boundary_strict_px"] = args.hollow_boundary_strict_px
    filter_mod.PARAMS["boundary_black_threshold"] = args.hollow_boundary_black_threshold
    filter_mod.PARAMS["edge_ring_px"] = args.hollow_edge_ring_px
    filter_mod.PARAMS["edge_white_ratio_threshold"] = args.hollow_edge_white_ratio_threshold
    filter_mod.PARAMS["full_edge_white_ratio_threshold"] = args.hollow_full_edge_white_ratio_threshold
    filter_mod.PARAMS["tile_edge_white_ratio_threshold"] = args.hollow_tile_edge_white_ratio_threshold


def load_existing_particle_counts(output_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Read per-image particle counts from a previous run's summary.json."""
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"--skip_particle needs existing {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = {}
    dark_filtered_counts = {}
    for row in data.get("per_image", []):
        image_name = row.get("image")
        if image_name:
            key = str(image_name)
            counts[key] = int(row.get("particle_count", 0))
            dark_filtered_counts[key] = int(row.get("particle_dark_filtered_count", 0))
    if not counts:
        raise RuntimeError(f"No per-image particle counts found in {summary_path}")
    return counts, dark_filtered_counts


def analyze_hollow_predictions(
    filter_mod: ModuleType,
    image_bgr: np.ndarray,
    predictions: Iterable,
    ratio_threshold: float,
    use_edge_white_filter: bool = True,
) -> tuple[list[dict], list[dict]]:
    kept = []
    all_items = []
    for pred in predictions:
        mask = pred.mask.astype(np.uint8)
        black_mask, raw_ratio, info = filter_mod.compute_black_mask(image_bgr, mask)
        corr_ratio, corr_info = filter_mod.compute_corrected_ratio(image_bgr, mask, black_mask)
        edge_white_ratio, edge_info = filter_mod.compute_edge_white_ratio(image_bgr, mask)
        info.update(corr_info)
        info.update(edge_info)
        source = getattr(pred, "source", "")
        default_edge_threshold = float(filter_mod.PARAMS.get("edge_white_ratio_threshold", 0.60))
        if source == "full":
            configured = filter_mod.PARAMS.get("full_edge_white_ratio_threshold", default_edge_threshold)
            edge_threshold = float(default_edge_threshold if configured is None else configured)
        elif source == "tile":
            configured = filter_mod.PARAMS.get("tile_edge_white_ratio_threshold", default_edge_threshold)
            edge_threshold = float(default_edge_threshold if configured is None else configured)
        else:
            edge_threshold = default_edge_threshold
        edge_white_pass = (not use_edge_white_filter) or (
            edge_white_ratio >= edge_threshold
        )
        item = {
            "mask": pred.mask.astype(bool),
            "black_mask": black_mask.astype(bool),
            "score": float(getattr(pred, "score", 0.0)),
            "source": source,
            "raw_ratio": float(corr_ratio),
            "corr_ratio": float(corr_ratio),
            "black_area": int(info.get("black_area", 0)),
            "body_area": int(info.get("body_area", 0)),
            "model_mask_area": int(info.get("model_mask_area", int(mask.sum()))),
            "edge_ring_px": int(info.get("edge_ring_px", 0)),
            "edge_ring_area": int(info.get("edge_ring_area", 0)),
            "edge_white_area": int(info.get("edge_white_area", 0)),
            "edge_white_ratio": float(edge_white_ratio),
            "edge_white_threshold": float(edge_threshold),
            "edge_white_pass": bool(edge_white_pass),
        }
        all_items.append(item)
        if corr_ratio >= ratio_threshold and edge_white_pass:
            kept.append(item)
    return kept, all_items


def draw_final_preview(image_path: Path, kept_hollows: list[dict], out_path: Path, ratio_threshold: float) -> None:
    image = imread_unicode(image_path)
    if image is None:
        return
    preview = image.copy()
    overlay = preview.copy()
    for item in kept_hollows:
        overlay[item["black_mask"]] = RED
    preview = cv2.addWeighted(overlay, 0.35, preview, 0.65, 0)

    for item in kept_hollows:
        mask = item["mask"].astype(np.uint8)
        contours = mask_contours(mask)
        cv2.drawContours(preview, contours, -1, RED, 3)
        bbox = cv2.boundingRect(mask)
        x, y, w, h = bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), RED, 2)
        draw_text_box(preview, f"ratio={item['corr_ratio']:.3f}", x, max(18, y - 8), 0.55)

    draw_text_box(
        preview,
        f"hollow powder: corr_ratio >= {ratio_threshold:.2f}, count={len(kept_hollows)}",
        12,
        28,
        0.62,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, preview)


def draw_particle_preview(image_path: Path, particle_preds: list, out_path: Path, cfg: config.InferenceConfig) -> None:
    image = imread_unicode(image_path)
    if image is None:
        return
    preview = image.copy()
    draw_tile_boundaries(preview, cfg)
    for pred in particle_preds:
        color = BLUE if getattr(pred, "source", "") == "tile" else RED
        cv2.drawContours(preview, mask_contours(pred.mask.astype(np.uint8)), -1, color, 2)
    draw_text_box(preview, f"particle count={len(particle_preds)}", 12, 28, 0.62)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, preview)


def draw_hollow_preview(
    image_path: Path,
    kept_hollows: list[dict],
    out_path: Path,
    ratio_threshold: float,
    cfg: config.InferenceConfig,
) -> None:
    image = imread_unicode(image_path)
    if image is None:
        return
    preview = image.copy()
    overlay = preview.copy()
    for item in kept_hollows:
        overlay[item["black_mask"]] = RED
    preview = cv2.addWeighted(overlay, 0.28, preview, 0.72, 0)
    draw_tile_boundaries(preview, cfg)

    for item in kept_hollows:
        mask = item["mask"].astype(np.uint8)
        color = BLUE if item.get("source") == "tile" else RED
        cv2.drawContours(preview, mask_contours(mask), -1, color, 1)
        x, y, w, h = cv2.boundingRect(mask)
        draw_text_box(preview, f"br={item['corr_ratio']:.3f}", x, max(18, y - 8), 0.52)
    draw_text_box(
        preview,
        f"hollow candidates: corr_ratio >= {ratio_threshold:.2f}, count={len(kept_hollows)}",
        12,
        28,
        0.62,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, preview)


def draw_pic_final_preview(
    image_path: Path,
    particle_preds: list,
    kept_hollows: list[dict],
    out_path: Path,
    particle_count: int,
    hollow_powder_count: int,
) -> None:
    """Draw the final presentation preview with particle denominator and hollow numerator."""
    image = imread_unicode(image_path)
    if image is None:
        return
    preview = image.copy()

    # Particle denominator: one uniform blue style, regardless of full/tile source.
    for pred in particle_preds:
        cv2.drawContours(preview, mask_contours(pred.mask.astype(np.uint8)), -1, BLUE, 2)

    # Hollow powder numerator: red overlay/contour and black-ratio label, same idea as pic/.
    overlay = preview.copy()
    for item in kept_hollows:
        overlay[item["black_mask"]] = RED
    preview = cv2.addWeighted(overlay, 0.30, preview, 0.70, 0)

    for item in kept_hollows:
        mask = item["mask"].astype(np.uint8)
        cv2.drawContours(preview, mask_contours(mask), -1, RED, 3)
        x, y, w, h = cv2.boundingRect(mask)
        cv2.rectangle(preview, (x, y), (x + w, y + h), RED, 2)
        draw_text_box(preview, f"br={item['corr_ratio']:.3f}", x, max(18, y - 8), 0.55)

    ratio = hollow_powder_count / particle_count if particle_count else 0.0
    draw_text_box(
        preview,
        f"particle={particle_count}  hollow={hollow_powder_count}  rate={ratio:.6f}",
        12,
        28,
        0.62,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, preview)


def write_summary_txt(path: Path, summary: dict, per_image: list[dict]) -> None:
    lines = [
        f"Input: {summary['input_dir']}",
        f"Output: {summary['output_dir']}",
        "",
        f"Images: {summary['image_count']}",
        f"Particle count: {summary['particle_count']}",
        f"Particle boundary-filtered count: {summary.get('particle_boundary_filtered_count', 0)}",
        f"Particle dark-filtered count: {summary.get('particle_dark_filtered_count', 0)}",
        f"Hollow candidate count: {summary['hollow_candidate_count']}",
        f"Hollow powder count: {summary['hollow_powder_count']}",
        f"Hollow powder ratio: {summary['hollow_powder_count']} / {summary['particle_count']} = {summary['hollow_powder_ratio']:.8f}",
        "",
        f"Rule: corrected_black_ratio >= {summary['hollow_ratio_threshold']}",
        "",
        "Per image:",
    ]
    for row in per_image:
        lines.append(
            f"{row['image']}\tparticle={row['particle_count']}\t"
            f"particle_dark_filtered={row.get('particle_dark_filtered_count', 0)}\t"
            f"hollow_candidates={row['hollow_candidate_count']}\t"
            f"hollow_powder={row['hollow_powder_count']}\t"
            f"ratio={row['hollow_powder_ratio']:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--reset", action="store_true", help="Delete output_dir before running.")
    parser.add_argument("--max_images", type=int, default=0, help="Debug mode: process only first N images.")
    parser.add_argument("--image_batch_size", type=int, default=4, help="Number of source images per GPU batch.")
    parser.add_argument("--torch_threads", type=int, default=8)
    parser.add_argument(
        "--skip_particle",
        action="store_true",
        help="Reuse particle counts/previews from an existing output_dir and only rerun hollow inference/reporting.",
    )
    parser.add_argument(
        "--hollow_only",
        action="store_true",
        help="Run only hollow inference/filtering; do not load or run the particle model.",
    )

    parser.add_argument("--particle_model", default=config.PARTICLE_MODEL)
    parser.add_argument("--hollow_model", default=config.HOLLOW_MODEL)
    parser.add_argument("--eval_particle_script", default=config.EVAL_PARTICLE_SCRIPT)
    parser.add_argument("--hollow_filter_script", default=config.HOLLOW_FILTER_SCRIPT)

    parser.add_argument("--hollow_ratio_threshold", type=float, default=config.HOLLOW_RATIO_THRESHOLD)
    parser.add_argument("--hollow_preview_threshold", type=float, default=0.10)
    parser.add_argument("--hollow_erode_px", type=int, default=0, help="Deprecated; boundary retreat is adaptive now.")
    parser.add_argument("--hollow_white_threshold", type=float, default=150.0)
    parser.add_argument("--hollow_white_max_channel_diff", type=float, default=50.0)
    parser.add_argument("--hollow_boundary_strict_px", type=int, default=3)
    parser.add_argument("--hollow_boundary_black_threshold", type=float, default=100.0)
    parser.add_argument("--hollow_edge_ring_px", type=int, default=5)
    parser.add_argument("--hollow_edge_white_ratio_threshold", type=float, default=0.60)
    parser.add_argument(
        "--hollow_full_edge_white_ratio_threshold",
        type=float,
        default=0.40,
        help="White-shell ratio threshold for full-image hollow candidates.",
    )
    parser.add_argument(
        "--hollow_tile_edge_white_ratio_threshold",
        type=float,
        default=None,
        help="White-shell ratio threshold for tile hollow candidates; defaults to --hollow_edge_white_ratio_threshold.",
    )
    parser.add_argument("--disable_hollow_edge_white_filter", action="store_true")
    parser.add_argument("--min_component_area_ratio", type=float, default=0.002)

    parser.add_argument("--particle_full_score", type=float, default=config.PARTICLE_INFERENCE.full_score_threshold)
    parser.add_argument("--particle_tile_score", type=float, default=config.PARTICLE_INFERENCE.tile_score_threshold)
    parser.add_argument(
        "--particle_dark_mean_threshold",
        type=float,
        default=100.0,
        help="Remove particle masks whose average grayscale is <= this value.",
    )
    parser.add_argument(
        "--disable_particle_dark_filter",
        action="store_true",
        help="Keep pure-black particle candidates instead of applying the gray-mean CV filter.",
    )
    parser.add_argument("--hollow_full_score", type=float, default=config.HOLLOW_INFERENCE.full_score_threshold)
    parser.add_argument("--hollow_tile_score", type=float, default=config.HOLLOW_INFERENCE.tile_score_threshold)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(input_dir)
    if args.skip_particle and args.reset:
        raise ValueError("--skip_particle cannot be combined with --reset because reset removes previous particle counts.")
    if args.hollow_only and args.skip_particle:
        raise ValueError("--hollow_only and --skip_particle are mutually exclusive.")
    if output_dir.exists() and args.reset:
        shutil.rmtree(output_dir)
    pic_dir = output_dir / "pic"
    pic_final_dir = output_dir / "pic_final"
    particle_pic_dir = output_dir / "particle"
    hollow_pic_dir = output_dir / "hollow"
    work_dir = output_dir / "work"
    pic_dir.mkdir(parents=True, exist_ok=True)
    pic_final_dir.mkdir(parents=True, exist_ok=True)
    particle_pic_dir.mkdir(parents=True, exist_ok=True)
    hollow_pic_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_mod = load_module(args.eval_particle_script, "empty_heart_eval_particle")
    filter_mod = load_module(args.hollow_filter_script, "empty_heart_hollow_filter")
    set_filter_params(filter_mod, args)

    particle_cfg = config.PARTICLE_INFERENCE
    hollow_cfg = config.HOLLOW_INFERENCE
    particle_cfg = config.InferenceConfig(**{**asdict(particle_cfg), "full_score_threshold": args.particle_full_score, "tile_score_threshold": args.particle_tile_score})
    hollow_cfg = config.InferenceConfig(**{**asdict(hollow_cfg), "full_score_threshold": args.hollow_full_score, "tile_score_threshold": args.hollow_tile_score})

    images = image_files(input_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"device={device}, images={len(images)}")
    if args.hollow_only:
        particle_counts = {path.name: 0 for path in images}
        particle_predictions_by_image = {}
        particle_dark_filtered_counts = {}
        particle_boundary_filtered_counts = {}
        print("hollow_only=True, skipped particle model/inference")
    elif args.skip_particle:
        particle_counts, particle_dark_filtered_counts = load_existing_particle_counts(output_dir)
        particle_predictions_by_image = {}
        particle_boundary_filtered_counts = {}
        print(f"skip_particle=True, reused particle counts from {output_dir / 'summary.json'}")
    else:
        print(f"particle_model={args.particle_model}")
        particle_model = build_model(eval_mod, Path(args.particle_model), particle_cfg, device)
        particle_tile_model = None
        if particle_cfg.tile_model_min_size is not None and particle_cfg.tile_model_max_size is not None:
            particle_tile_cfg = with_model_size(
                particle_cfg,
                particle_cfg.tile_model_min_size,
                particle_cfg.tile_model_max_size,
            )
            particle_tile_model = build_model(eval_mod, Path(args.particle_model), particle_tile_cfg, device)
            print(
                "particle_split_scale="
                f"full({particle_cfg.model_min_size},{particle_cfg.model_max_size}) "
                f"tile({particle_tile_cfg.model_min_size},{particle_tile_cfg.model_max_size})",
                flush=True,
            )
        particle_counts: dict[str, int] = {}
        particle_predictions_by_image: dict[str, list] = {}
        particle_dark_filtered_counts: dict[str, int] = {}
        particle_boundary_filtered_counts: dict[str, int] = {}
        for batch_index, image_batch in enumerate(batched(images, args.image_batch_size), start=1):
            opened = [(path, Image.open(path).convert("RGB")) for path in image_batch]
            if particle_tile_model is None:
                batch_preds = infer_image_batch(eval_mod, particle_model, device, opened, particle_cfg)
            else:
                batch_preds = infer_image_batch_split_models(
                    eval_mod, particle_model, particle_tile_model, device, opened, particle_cfg
                )
            view_count = sum(
                1 + len(eval_mod.make_tiles(image.width, image.height, particle_cfg.tile_cols, particle_cfg.tile_rows, particle_cfg.tile_w, particle_cfg.tile_h))
                for _, image in opened
            )
            print(
                f"particle batch {batch_index}: images={len(image_batch)} "
                f"views={view_count}",
                flush=True,
            )
            for image_path in image_batch:
                preds = batch_preds.get(image_path.name, [])
                with Image.open(image_path) as im_for_size:
                    image_width, image_height = im_for_size.size
                preds, removed_boundary = filter_particle_image_boundary_predictions(
                    preds,
                    image_width,
                    image_height,
                    particle_cfg.image_boundary_margin,
                )
                removed_dark = 0
                if not args.disable_particle_dark_filter:
                    image_bgr = imread_unicode(image_path)
                    if image_bgr is not None:
                        preds, removed_dark = filter_dark_particle_predictions(
                            image_bgr,
                            preds,
                            args.particle_dark_mean_threshold,
                        )
                particle_counts[image_path.name] = len(preds)
                particle_predictions_by_image[image_path.name] = preds
                particle_dark_filtered_counts[image_path.name] = removed_dark
                particle_boundary_filtered_counts[image_path.name] = removed_boundary
                draw_particle_preview(image_path, preds, particle_pic_dir / image_path.name, particle_cfg)
                print(
                    f"  particle {image_path.name}: {len(preds)}"
                    f" boundary_filtered={removed_boundary}"
                    f" dark_filtered={removed_dark}",
                    flush=True,
                )
        del particle_model
        if particle_tile_model is not None:
            del particle_tile_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"hollow_model={args.hollow_model}")
    hollow_model = build_model(eval_mod, Path(args.hollow_model), hollow_cfg, device)
    if (args.skip_particle or args.hollow_only) and "particle_dark_filtered_counts" not in locals():
        particle_dark_filtered_counts = {}
    if (args.skip_particle or args.hollow_only) and "particle_boundary_filtered_counts" not in locals():
        particle_boundary_filtered_counts = {}
    if (args.skip_particle or args.hollow_only) and "particle_predictions_by_image" not in locals():
        particle_predictions_by_image = {}

    per_image = []
    all_candidate_details = {}
    total_particles = 0
    total_particle_dark_filtered = 0
    total_particle_boundary_filtered = 0
    total_hollow_candidates = 0
    total_hollow_powder = 0

    for batch_index, image_batch in enumerate(batched(images, args.image_batch_size), start=1):
        opened = [(path, Image.open(path).convert("RGB")) for path in image_batch]
        batch_preds = infer_image_batch(eval_mod, hollow_model, device, opened, hollow_cfg)
        view_count = sum(
            1 + len(eval_mod.make_tiles(image.width, image.height, hollow_cfg.tile_cols, hollow_cfg.tile_rows, hollow_cfg.tile_w, hollow_cfg.tile_h))
            for _, image in opened
        )
        print(
            f"hollow batch {batch_index}: images={len(image_batch)} "
            f"views={view_count}",
            flush=True,
        )
        for image_path in image_batch:
            image_bgr = imread_unicode(image_path)
            hollow_preds = batch_preds.get(image_path.name, [])
            kept_hollows, all_hollows = analyze_hollow_predictions(
                filter_mod,
                image_bgr,
                hollow_preds,
                args.hollow_ratio_threshold,
                not args.disable_hollow_edge_white_filter,
            )
            preview_hollows = [
                item
                for item in all_hollows
                if item["corr_ratio"] >= args.hollow_preview_threshold and item.get("edge_white_pass", True)
            ]
            draw_hollow_preview(
                image_path,
                preview_hollows,
                hollow_pic_dir / image_path.name,
                args.hollow_preview_threshold,
                hollow_cfg,
            )
            out_image = pic_dir / image_path.name
            draw_final_preview(image_path, kept_hollows, out_image, args.hollow_ratio_threshold)

            particle_count = particle_counts.get(image_path.name, 0)
            if image_path.name in particle_predictions_by_image:
                draw_pic_final_preview(
                    image_path,
                    particle_predictions_by_image[image_path.name],
                    kept_hollows,
                    pic_final_dir / image_path.name,
                    particle_count,
                    len(kept_hollows),
                )
            particle_dark_filtered_count = particle_dark_filtered_counts.get(image_path.name, 0)
            particle_boundary_filtered_count = particle_boundary_filtered_counts.get(image_path.name, 0)
            hollow_candidate_count = len(hollow_preds)
            hollow_powder_count = len(kept_hollows)
            total_particles += particle_count
            total_particle_dark_filtered += particle_dark_filtered_count
            total_particle_boundary_filtered += particle_boundary_filtered_count
            total_hollow_candidates += hollow_candidate_count
            total_hollow_powder += hollow_powder_count
            row = {
                "image": image_path.name,
                "particle_count": particle_count,
                "particle_boundary_filtered_count": particle_boundary_filtered_count,
                "particle_dark_filtered_count": particle_dark_filtered_count,
                "hollow_candidate_count": hollow_candidate_count,
                "hollow_powder_count": hollow_powder_count,
                "hollow_powder_ratio": hollow_powder_count / particle_count if particle_count else 0.0,
            }
            per_image.append(row)
            all_candidate_details[image_path.name] = [
                {
                    "score": item["score"],
                    "source": item["source"],
                    "raw_ratio": item["raw_ratio"],
                    "corr_ratio": item["corr_ratio"],
                    "black_area": item["black_area"],
                    "body_area": item["body_area"],
                    "model_mask_area": item["model_mask_area"],
                    "edge_ring_px": item.get("edge_ring_px", 0),
                    "edge_white_ratio": item.get("edge_white_ratio", 0.0),
                    "edge_white_pass": item.get("edge_white_pass", True),
                    "kept": item["corr_ratio"] >= args.hollow_ratio_threshold and item.get("edge_white_pass", True),
                }
                for item in all_hollows
            ]
            print(
                f"  hollow {image_path.name}: candidates={hollow_candidate_count} "
                f"powder={hollow_powder_count}",
                flush=True,
            )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "hollow_only": args.hollow_only,
        "particle_count": total_particles,
        "particle_boundary_filtered_count": total_particle_boundary_filtered,
        "particle_image_boundary_margin": particle_cfg.image_boundary_margin,
        "particle_dark_filtered_count": total_particle_dark_filtered,
        "particle_dark_filter_enabled": (not args.disable_particle_dark_filter) and (not args.hollow_only),
        "particle_dark_mean_threshold": args.particle_dark_mean_threshold,
        "hollow_candidate_count": total_hollow_candidates,
        "hollow_powder_count": total_hollow_powder,
        "hollow_powder_ratio": total_hollow_powder / total_particles if total_particles else 0.0,
        "hollow_ratio_threshold": args.hollow_ratio_threshold,
        "hollow_preview_threshold": args.hollow_preview_threshold,
        "hollow_edge_white_filter_enabled": not args.disable_hollow_edge_white_filter,
        "hollow_edge_ring_px": args.hollow_edge_ring_px,
        "hollow_edge_white_ratio_threshold": args.hollow_edge_white_ratio_threshold,
        "hollow_full_edge_white_ratio_threshold": args.hollow_full_edge_white_ratio_threshold,
        "hollow_tile_edge_white_ratio_threshold": args.hollow_tile_edge_white_ratio_threshold
        if args.hollow_tile_edge_white_ratio_threshold is not None
        else args.hollow_edge_white_ratio_threshold,
        "particle_model": str(Path(args.particle_model).resolve()),
        "hollow_model": str(Path(args.hollow_model).resolve()),
        "particle_inference": asdict(particle_cfg),
        "hollow_inference": asdict(hollow_cfg),
    }
    write_summary_txt(output_dir / "summary.txt", summary, per_image)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "per_image": per_image}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work_dir / "hollow_candidate_details.json").write_text(
        json.dumps(all_candidate_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pic_dir={pic_dir}")
    print(f"pic_final_dir={pic_final_dir}")
    print(f"summary={output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
