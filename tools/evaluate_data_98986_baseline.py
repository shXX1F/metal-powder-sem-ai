from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metal_powder_sem_ai.classify_stat import (
    DEFAULT_SPHERICAL_ROUNDNESS_METHOD,
    DEFAULT_SPHERICAL_ROUNDNESS_THRESHOLD,
    SPHERICAL_ROUNDNESS_METHODS,
    classify_particles,
    diameter_value,
    reference_percentile,
)
from metal_powder_sem_ai.feature_extract import extract_all_features
from metal_powder_sem_ai.postprocess import (
    filter_false_background_masks,
    filter_false_surface_fragments,
)
from metal_powder_sem_ai.preprocess import imwrite_unicode, preprocess_sem_image
from metal_powder_sem_ai.segment import (
    ClassicalParticleSegmenter,
    InstanceMask,
    MaskRCNNSegmenter,
    merge_instance_groups,
)
from metal_powder_sem_ai.visualize import draw_instances


IMAGE_SUFFIXES = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]

SUMMARY_COLUMNS = [
    "source_kind",
    "workbook_path",
    "sheet_name",
    "batch_id",
    "image_path",
    "inference_mode",
    "diameter_method",
    "roundness_method",
    "spherical_roundness_method",
    "system_spherical_roundness_threshold",
    "reference_spherical_roundness_threshold",
    "score_threshold",
    "mask_threshold",
    "tile_size",
    "tile_overlap",
    "tile_ownership_filter",
    "pixel_size_um",
    "pixel_size_source",
    "system_total_particles",
    "reference_total_particles",
    "postprocess_removed_count",
    "postprocess_background_removed_count",
    "postprocess_surface_removed_count",
    "postprocess_before_count",
    "postprocess_after_count",
    "particle_count_error",
    "particle_count_error_percent",
    "system_mean_roundness",
    "reference_mean_roundness",
    "mean_roundness_error",
    "system_mean_sphericity_q",
    "reference_mean_sphericity_q",
    "mean_sphericity_q_error",
    "system_diameter_mean_um",
    "reference_diameter_mean_um",
    "diameter_mean_error_um",
    "system_d50_um",
    "reference_d50_um",
    "d50_error_um",
    "system_d10_um",
    "reference_d10_um",
    "system_d90_um",
    "reference_d90_um",
    "system_spherical_rate_percent",
    "reference_spherical_rate_percent",
    "spherical_rate_error_percent_points",
    "system_agglomerate_rate_percent",
    "system_agglomerate_area_rate_percent",
    "reference_particle_area_ratio_percent",
    "system_particle_area_ratio_percent",
    "elapsed_seconds",
    "status",
    "message",
]

BIN_COLUMNS = [
    "source_kind",
    "workbook_path",
    "sheet_name",
    "batch_id",
    "image_path",
    "inference_mode",
    "diameter_bin_um",
    "system_count",
    "reference_count",
    "count_error",
    "count_error_percent",
]

PARTICLE_COLUMNS = [
    "source_kind",
    "workbook_path",
    "sheet_name",
    "batch_id",
    "image_path",
    "inference_mode",
    "pixel_size_um",
    "particle_id",
    "score",
    "area_px",
    "area_um2",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "major_axis_px",
    "major_axis_um",
    "feret_diameter_px",
    "feret_diameter_um",
    "bbox_max_diameter_px",
    "bbox_max_diameter_um",
    "statistical_diameter_um",
    "statistical_diameter_method",
    "perimeter_contour_px",
    "perimeter_crofton_px",
    "perimeter_crofton2_px",
    "perimeter_crofton_close3_px",
    "perimeter_crofton_open3_px",
    "perimeter_crofton_smooth3_px",
    "perimeter_subpixel_px",
    "q_value",
    "roundness",
    "q_value_contour",
    "roundness_contour",
    "q_value_crofton",
    "roundness_crofton",
    "q_value_crofton2",
    "roundness_crofton2",
    "q_value_crofton_close3",
    "roundness_crofton_close3",
    "q_value_crofton_open3",
    "roundness_crofton_open3",
    "q_value_crofton_smooth3",
    "roundness_crofton_smooth3",
    "q_value_subpixel",
    "roundness_subpixel",
    "spherical_roundness",
    "spherical_roundness_method",
    "spherical_roundness_fallback",
    "axis_ratio",
    "is_spherical",
    "is_agglomerate",
    "touches_image_border",
    "valid_for_size_statistics",
    "centroid_x",
    "centroid_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

AGGLOMERATE_EVIDENCE_COLUMNS = [
    "source_kind",
    "batch_id",
    "image_path",
    "inference_mode",
    "pixel_size_um",
    "requested_tolerance_um",
    "effective_tolerance_um",
    "tolerance_px",
    "min_contact_ratio",
    "min_overlap_ratio",
    "particle_a",
    "particle_b",
    "gap_px",
    "gap_um",
    "contact_ratio",
    "overlap_ratio",
    "center_distance_ratio",
    "accepted",
    "reason",
    "manual_truth",
]

MODE_SUMMARY_COLUMNS = [
    "inference_mode",
    "evaluated_images",
    "system_total_particles",
    "reference_total_particles",
    "postprocess_removed_total",
    "total_particle_error",
    "total_particle_error_percent",
    "mean_abs_particle_count_error",
    "mean_abs_roundness_error",
    "mean_abs_sphericity_q_error",
    "mean_abs_spherical_rate_error_percent_points",
    "mean_abs_d50_error_um",
    "mean_elapsed_seconds",
]


@dataclass
class EvaluationItem:
    summary: Dict[str, str]
    particles: List[Dict[str, str]]
    image_path: Path

    @property
    def key(self) -> Tuple[str, str, str]:
        return (
            self.summary.get("workbook_path", ""),
            self.summary.get("sheet_name", ""),
            self.summary.get("batch_id", ""),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current SEM particle recognizer and compare its output "
            "with data_98986 reference roundness results."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_98986") / "data",
        help="Root directory that contains data_98986 images and Excel results.",
    )
    parser.add_argument(
        "--fixed-eval-dir",
        type=Path,
        default=None,
        help=(
            "Frozen evaluation set directory created by tools/create_fixed_eval_set.py. "
            "When set, manifest.csv and reference_particles.csv under this directory "
            "are used instead of resolving images from data_98986/data."
        ),
    )
    parser.add_argument(
        "--reference-summary-csv",
        type=Path,
        default=Path("outputs")
        / "data_98986_roundness"
        / "data_98986_image_summary.csv",
        help="CSV generated by tools/extract_data_98986_roundness.py.",
    )
    parser.add_argument(
        "--reference-particle-csv",
        type=Path,
        default=Path("outputs")
        / "data_98986_roundness"
        / "data_98986_particle_roundness.csv",
        help="Particle CSV generated by tools/extract_data_98986_roundness.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "data_98986_evaluation",
        help="Directory for evaluation CSV files.",
    )
    parser.add_argument(
        "--segmenter",
        choices=["maskrcnn", "classical"],
        default="maskrcnn",
        help="Segmentation method used for evaluation.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Mask R-CNN weight path. Required when --segmenter maskrcnn unless --dry-run is used.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument(
        "--diameter-method",
        choices=["equivalent", "major_axis", "feret_max", "bbox_max"],
        default="feret_max",
        help="Particle diameter definition used by mean, percentiles and bins.",
    )
    parser.add_argument(
        "--roundness-method",
        choices=["contour", "crofton", "subpixel"],
        default="crofton",
        help="Perimeter estimator used by Q and roundness C.",
    )
    parser.add_argument(
        "--spherical-roundness-threshold",
        type=float,
        default=DEFAULT_SPHERICAL_ROUNDNESS_THRESHOLD,
        help="Roundness C threshold used to classify a particle as spherical.",
    )
    parser.add_argument(
        "--spherical-roundness-method",
        choices=sorted(SPHERICAL_ROUNDNESS_METHODS),
        default=DEFAULT_SPHERICAL_ROUNDNESS_METHOD,
        help=(
            "Roundness feature used only for spherical classification. "
            "The reported mean Q/C still follows --roundness-method."
        ),
    )
    parser.add_argument(
        "--reference-spherical-roundness-threshold",
        type=float,
        default=0.90,
        help=(
            "Fixed reference-software roundness threshold. This must remain "
            "independent from the system threshold being calibrated."
        ),
    )
    parser.add_argument("--max-detections", type=int, default=10000)
    parser.add_argument("--max-inference-side", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=int, default=192)
    parser.add_argument(
        "--tile-ownership-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only detections owned by each tile core. Disable to retain all "
            "overlap detections before mask-IoU deduplication."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=["whole", "tiled", "merged", "all"],
        default=None,
        help=(
            "Mask R-CNN inference mode. whole=whole image, tiled=tiled image, "
            "merged=whole+tiled merged, all=run and compare all three modes."
        ),
    )
    parser.add_argument(
        "--merge-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run whole-image and tiled Mask R-CNN inference, then merge results.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Inference device. CPU matches the current GUI comprehensive preset.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Manual calibration: 1 pixel = X um.",
    )
    parser.add_argument(
        "--infer-pixel-size-from-field-area",
        action="store_true",
        help=(
            "When field_area_um2 exists in reference CSV, estimate pixel size "
            "from reference field area and cropped image size."
        ),
    )
    parser.add_argument("--agglomerate-tolerance-um", type=float, default=0.30)
    parser.add_argument("--min-agglomerate-group-size", type=int, default=3)
    parser.add_argument("--agglomerate-min-contact-ratio", type=float, default=0.09)
    parser.add_argument("--agglomerate-min-overlap-ratio", type=float, default=0.03)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.12)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--peak-min-distance", type=int, default=12)
    parser.add_argument(
        "--postprocess-background-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove background-like masks that are clipped by an internal "
            "inference tile edge."
        ),
    )
    parser.add_argument(
        "--postprocess-fragments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter small masks that are likely surface fragments of larger particles.",
    )
    parser.add_argument(
        "--protect-parent-fragments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In merged mode, keep earlier whole-image parent masks over later small tiled fragments.",
    )
    parser.add_argument("--protected-parent-diameter-ratio", type=float, default=1.6)
    parser.add_argument("--protected-parent-center-distance-factor", type=float, default=0.0)
    parser.add_argument("--postprocess-max-child-diameter-um", type=float, default=20.0)
    parser.add_argument("--postprocess-min-parent-diameter-um", type=float, default=25.0)
    parser.add_argument("--postprocess-parent-child-ratio", type=float, default=1.8)
    parser.add_argument("--postprocess-surface-distance-factor", type=float, default=0.0)
    parser.add_argument(
        "--diameter-bins-um",
        default="0,5,10,15,20,30,50,inf",
        help="Comma-separated diameter bins used for small-particle error analysis.",
    )
    parser.add_argument(
        "--source-kind",
        action="append",
        choices=["result-return", "roundness-summary"],
        help="Filter reference rows by source kind. Can be specified multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N matched images.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save annotated images for inspected samples.",
    )
    parser.add_argument(
        "--save-agglomerate-evidence",
        action="store_true",
        help=(
            "Write one reviewable *.agglomerate_pairs.csv file per evaluated image. "
            "Fill its manual_truth column before threshold calibration."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only match reference rows to images; do not run segmentation.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def agglomerate_evidence_rows(
    base: Dict[str, object],
    pixel_size_um: float,
    stats: Dict[str, object],
) -> List[Dict[str, object]]:
    metadata = {
        "source_kind": base.get("source_kind", ""),
        "batch_id": base.get("batch_id", ""),
        "image_path": base.get("image_path", ""),
        "inference_mode": base.get("inference_mode", ""),
        "pixel_size_um": pixel_size_um,
        "requested_tolerance_um": stats.get("agglomerate_tolerance_um", ""),
        "effective_tolerance_um": stats.get("agglomerate_effective_tolerance_um", ""),
        "tolerance_px": stats.get("agglomerate_tolerance_px", ""),
        "min_contact_ratio": stats.get("agglomerate_min_contact_ratio", ""),
        "min_overlap_ratio": stats.get("agglomerate_min_overlap_ratio", ""),
    }
    rows: List[Dict[str, object]] = []
    for evidence in stats.get("agglomerate_pair_evidence", []):
        if not isinstance(evidence, dict):
            continue
        rows.append({**metadata, **evidence, "manual_truth": ""})
    return rows


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: object) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("workbook_path", ""),
        row.get("sheet_name", ""),
        row.get("batch_id", ""),
    )


def is_aggregate_summary(row: Dict[str, str]) -> bool:
    workbook_name = Path(row.get("workbook_path", "")).name
    if "统计" in workbook_name:
        return True
    return False


def normalize_workbook_path(data_root: Path, workbook_path: str) -> Path:
    parts = [part for part in workbook_path.replace("\\", "/").split("/") if part]
    return data_root.joinpath(*parts)


def image_name_candidates(batch_id: str, sheet_name: str) -> List[str]:
    stems = []
    for value in [batch_id, Path(sheet_name).stem, sheet_name]:
        value = str(value or "").strip()
        if value and value not in stems:
            stems.append(value)
    names = []
    for stem in stems:
        for suffix in IMAGE_SUFFIXES:
            names.append(stem + suffix)
    return names


def resolve_image_path(row: Dict[str, str], data_root: Path) -> Optional[Path]:
    workbook = normalize_workbook_path(data_root, row.get("workbook_path", ""))
    directory = workbook.parent
    batch_id = row.get("batch_id", "")
    sheet_name = row.get("sheet_name", "")

    for name in image_name_candidates(batch_id, sheet_name):
        candidate = directory / name
        if candidate.is_file():
            return candidate

    stems = {Path(name).stem for name in image_name_candidates(batch_id, sheet_name)}
    for candidate in sorted(directory.glob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = candidate.stem
        if stem.endswith("-0"):
            continue
        if "分布图" in str(candidate) or "截图" in stem:
            continue
        if stem in stems:
            return candidate

    for stem in stems:
        for suffix in IMAGE_SUFFIXES:
            matches = sorted(data_root.rglob(stem + suffix))
            matches = [
                item
                for item in matches
                if item.is_file()
                and not item.stem.endswith("-0")
                and "分布图" not in str(item)
                and "截图" not in item.stem
            ]
            if matches:
                return matches[0]
    return None


def load_reference_items(
    summary_csv: Path,
    particle_csv: Path,
    data_root: Path,
    source_kinds: Optional[Sequence[str]],
) -> Tuple[List[EvaluationItem], List[Dict[str, object]]]:
    summary_rows = read_csv_rows(summary_csv)
    particle_rows = read_csv_rows(particle_csv) if particle_csv.exists() else []

    particles_by_key: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in particle_rows:
        particles_by_key[row_key(row)].append(row)

    items: List[EvaluationItem] = []
    skipped: List[Dict[str, object]] = []
    allowed = set(source_kinds or [])
    for row in summary_rows:
        if allowed and row.get("source_kind", "") not in allowed:
            continue
        if is_aggregate_summary(row):
            skipped.append({**row, "status": "skipped", "message": "aggregate summary workbook"})
            continue
        image_path = resolve_image_path(row, data_root)
        if image_path is None:
            skipped.append({**row, "status": "skipped", "message": "image not found"})
            continue
        items.append(
            EvaluationItem(
                summary=row,
                particles=particles_by_key.get(row_key(row), []),
                image_path=image_path,
            )
        )
    return items, skipped


def load_fixed_eval_items(fixed_eval_dir: Path) -> Tuple[List[EvaluationItem], List[Dict[str, object]]]:
    manifest_path = fixed_eval_dir / "manifest.csv"
    particle_path = fixed_eval_dir / "reference_particles.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not particle_path.exists():
        raise FileNotFoundError(particle_path)

    manifest_rows = read_csv_rows(manifest_path)
    particle_rows = read_csv_rows(particle_path)
    particles_by_key: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in particle_rows:
        particles_by_key[row_key(row)].append(row)

    items: List[EvaluationItem] = []
    skipped: List[Dict[str, object]] = []
    for row in manifest_rows:
        frozen_image = row.get("frozen_image_path", "")
        image_path = fixed_eval_dir / frozen_image
        if not image_path.exists():
            skipped.append({**row, "status": "skipped", "message": "frozen image not found"})
            continue
        items.append(
            EvaluationItem(
                summary=row,
                particles=particles_by_key.get(row_key(row), []),
                image_path=image_path,
            )
        )
    return items, skipped


def parse_bins(text: str) -> List[float]:
    values: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        values.append(float("inf") if chunk in {"inf", "infinity"} else float(chunk))
    if len(values) < 2:
        raise ValueError("--diameter-bins-um must contain at least two values.")
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise ValueError("--diameter-bins-um must be strictly increasing.")
    return values


def bin_label(left: float, right: float) -> str:
    if math.isinf(right):
        return f">={left:g}"
    return f"{left:g}-{right:g}"


def count_by_bins(values: Iterable[float], bins: Sequence[float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    labels = [bin_label(bins[idx], bins[idx + 1]) for idx in range(len(bins) - 1)]
    for label in labels:
        counts[label] = 0
    for value in values:
        if value is None or not math.isfinite(float(value)):
            continue
        value = float(value)
        for idx in range(len(bins) - 1):
            left = bins[idx]
            right = bins[idx + 1]
            if left <= value < right or (math.isinf(right) and value >= left):
                counts[labels[idx]] += 1
                break
    return counts


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return reference_percentile(values, q)


def mean(values: Iterable[float]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def error_value(system_value: Optional[float], reference_value: Optional[float]) -> str:
    if system_value is None or reference_value is None:
        return ""
    return f"{system_value - reference_value:.6g}"


def error_percent(system_value: Optional[float], reference_value: Optional[float]) -> str:
    if system_value is None or reference_value is None or reference_value == 0:
        return ""
    return f"{(system_value - reference_value) / reference_value * 100.0:.6g}"


def infer_pixel_size_um(
    item: EvaluationItem,
    image_shape: Tuple[int, int, int],
    crop_bottom_fraction: float,
    manual_pixel_size_um: Optional[float],
    infer_from_field_area: bool,
) -> Tuple[float, str]:
    if manual_pixel_size_um is not None and manual_pixel_size_um > 0:
        return float(manual_pixel_size_um), "manual"

    if infer_from_field_area:
        field_area = to_float(item.summary.get("field_area_um2"))
        if field_area and field_area > 0:
            height, width = image_shape[:2]
            cropped_height = int(round(height * (1.0 - crop_bottom_fraction)))
            cropped_height = max(1, min(height, cropped_height))
            pixel_area = float(width * cropped_height)
            if pixel_area > 0:
                return float(math.sqrt(field_area / pixel_area)), "reference_field_area"

    return 1.0, "default_1um_per_px"


def build_segmenter(args: argparse.Namespace):
    if args.segmenter == "classical":
        return ClassicalParticleSegmenter(
            min_area_px=int(args.min_area_px),
            peak_min_distance=int(args.peak_min_distance),
        )
    if args.weights is None:
        raise ValueError("--weights is required when --segmenter maskrcnn.")
    device = None if args.device == "auto" else args.device
    return MaskRCNNSegmenter(
        weights_path=str(args.weights),
        score_threshold=float(args.score_threshold),
        mask_threshold=float(args.mask_threshold),
        max_detections=int(args.max_detections),
        max_inference_side=int(args.max_inference_side),
        device=device,
    )


def selected_inference_modes(args: argparse.Namespace) -> List[str]:
    if args.segmenter == "classical":
        return ["classical"]
    if args.inference_mode == "all":
        return ["whole", "tiled", "merged"]
    if args.inference_mode:
        return [args.inference_mode]
    return ["merged" if args.merge_inference else "tiled"]


def segment_image(
    image_bgr: np.ndarray,
    segmenter,
    args: argparse.Namespace,
) -> List[InstanceMask]:
    mode = args.inference_mode or ("merged" if args.merge_inference else "tiled")
    if mode == "whole":
        return segmenter.segment(image_bgr)
    if mode == "tiled":
        return segmenter.segment_tiled(
            image_bgr,
            tile_size=int(args.tile_size),
            overlap=int(args.tile_overlap),
            ownership_filter=bool(args.tile_ownership_filter),
        )
    if mode == "merged":
        normal_instances = segmenter.segment(image_bgr)
        tiled_instances = segmenter.segment_tiled(
            image_bgr,
            tile_size=int(args.tile_size),
            overlap=int(args.tile_overlap),
            ownership_filter=bool(args.tile_ownership_filter),
        )
        return merge_instance_groups(
            normal_instances,
            tiled_instances,
            protect_parent_fragments=bool(args.protect_parent_fragments),
            protected_parent_diameter_ratio=float(args.protected_parent_diameter_ratio),
            protected_parent_center_distance_factor=float(args.protected_parent_center_distance_factor),
        )
    raise ValueError(f"Unknown inference mode: {mode}")


def reference_metric_from_particles(
    item: EvaluationItem,
    particle_column: str,
    summary_column: str,
    reducer,
) -> Optional[float]:
    particle_values = [to_float(row.get(particle_column)) for row in item.particles]
    particle_values = [value for value in particle_values if value is not None]
    if particle_values:
        return reducer(particle_values)
    return to_float(item.summary.get(summary_column))


def evaluate_item(
    item: EvaluationItem,
    segmenter,
    args: argparse.Namespace,
    bins: Sequence[float],
    output_dir: Path,
    data_root: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    start = time.perf_counter()
    base = {
        "source_kind": item.summary.get("source_kind", ""),
        "workbook_path": item.summary.get("workbook_path", ""),
        "sheet_name": item.summary.get("sheet_name", ""),
        "batch_id": item.summary.get("batch_id", ""),
        "image_path": safe_rel(item.image_path, data_root),
        "inference_mode": "classical"
        if args.segmenter == "classical"
        else (args.inference_mode or ("merged" if args.merge_inference else "tiled")),
        "diameter_method": str(args.diameter_method),
        "roundness_method": str(args.roundness_method),
        "spherical_roundness_method": str(args.spherical_roundness_method),
        "system_spherical_roundness_threshold": float(
            args.spherical_roundness_threshold
        ),
        "reference_spherical_roundness_threshold": float(
            args.reference_spherical_roundness_threshold
        ),
        "score_threshold": float(args.score_threshold),
        "mask_threshold": float(args.mask_threshold),
        "tile_size": int(args.tile_size),
        "tile_overlap": int(args.tile_overlap),
        "tile_ownership_filter": bool(args.tile_ownership_filter),
    }

    image_bgr = cv2.imdecode(np.fromfile(str(item.image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {item.image_path}")

    pixel_size_um, pixel_size_source = infer_pixel_size_um(
        item,
        image_bgr.shape,
        crop_bottom_fraction=float(args.crop_bottom_fraction),
        manual_pixel_size_um=args.pixel_size_um,
        infer_from_field_area=bool(args.infer_pixel_size_from_field_area),
    )
    pre = preprocess_sem_image(
        image_bgr,
        crop_bottom_fraction=args.crop_bottom_fraction,
        blur_ksize=args.blur_ksize,
    )

    if args.segmenter == "classical":
        instances = segmenter.segment(pre.binary)
    else:
        instances = segment_image(pre.image, segmenter, args)
        instances, background_filter_info = filter_false_background_masks(
            instances,
            image_bgr=pre.image,
            enabled=bool(args.postprocess_background_artifacts),
        )
        instances, surface_filter_info = filter_false_surface_fragments(
            instances,
            pixel_size_um=pixel_size_um,
            enabled=bool(args.postprocess_fragments),
            max_child_diameter_um=float(args.postprocess_max_child_diameter_um),
            min_parent_diameter_um=float(args.postprocess_min_parent_diameter_um),
            min_parent_child_diameter_ratio=float(args.postprocess_parent_child_ratio),
            surface_contact_distance_factor=float(args.postprocess_surface_distance_factor),
        )
        postprocess_info = {
            **background_filter_info,
            **surface_filter_info,
            "postprocess_removed_count": (
                int(background_filter_info["background_filter_removed_count"])
                + int(surface_filter_info["postprocess_removed_count"])
            ),
            "postprocess_background_removed_count": int(
                background_filter_info["background_filter_removed_count"]
            ),
            "postprocess_surface_removed_count": int(
                surface_filter_info["postprocess_removed_count"]
            ),
            "postprocess_before_count": int(
                background_filter_info["background_filter_before_count"]
            ),
            "postprocess_after_count": int(
                surface_filter_info["postprocess_after_count"]
            ),
        }
    if args.segmenter == "classical":
        postprocess_info = {
            "postprocess_removed_count": 0,
            "postprocess_background_removed_count": 0,
            "postprocess_surface_removed_count": 0,
            "postprocess_before_count": len(instances),
            "postprocess_after_count": len(instances),
        }
    features = extract_all_features(
        instances,
        pixel_size_um=pixel_size_um,
        gray=pre.enhanced,
    )
    masks = [instance.mask for instance in instances]
    classified, stats = classify_particles(
        features,
        masks=masks,
        diameter_method=str(args.diameter_method),
        roundness_method=str(args.roundness_method),
        spherical_roundness_method=str(args.spherical_roundness_method),
        spherical_roundness_threshold=float(args.spherical_roundness_threshold),
        spherical_rule="roundness",
        agglomerate_tolerance_um=float(args.agglomerate_tolerance_um),
        min_agglomerate_group_size=int(args.min_agglomerate_group_size),
        agglomerate_min_contact_ratio=float(args.agglomerate_min_contact_ratio),
        agglomerate_min_overlap_ratio=float(args.agglomerate_min_overlap_ratio),
    )

    diameters = [
        diameter_value(row, method=str(args.diameter_method)) for row in classified
    ]
    roundness_values = [float(row.get("roundness", 0.0)) for row in classified]
    q_values = [float(row.get("q_value", 0.0)) for row in classified]
    system_diameter_mean = mean(diameters)
    system_mean_roundness = mean(roundness_values)
    system_mean_q = mean(q_values)
    system_d10 = percentile(diameters, 10)
    system_d50 = percentile(diameters, 50)
    system_d90 = percentile(diameters, 90)

    reference_total = to_int(item.summary.get("particle_count"))
    if reference_total is None:
        reference_total = len(item.particles)
    reference_mean_roundness = reference_metric_from_particles(
        item,
        particle_column="source_roundness",
        summary_column="source_mean_roundness",
        reducer=mean,
    )
    reference_roundness_values = [
        value
        for value in (to_float(row.get("source_roundness")) for row in item.particles)
        if value is not None
    ]
    reference_q_values = []
    for row in item.particles:
        value = to_float(row.get("source_q_value"))
        if value is None:
            roundness = to_float(row.get("source_roundness"))
            value = roundness * roundness if roundness is not None else None
        if value is not None:
            reference_q_values.append(value)
    reference_mean_q = mean(reference_q_values)
    reference_spherical_rate = (
        100.0
        * sum(
            value >= float(args.reference_spherical_roundness_threshold)
            for value in reference_roundness_values
        )
        / len(reference_roundness_values)
        if reference_roundness_values
        else None
    )
    reference_diameter_values = [
        value
        for value in (to_float(row.get("diameter_um")) for row in item.particles)
        if value is not None
    ]
    reference_diameter_mean = (
        mean(reference_diameter_values)
        if reference_diameter_values
        else to_float(item.summary.get("diameter_mean_um"))
    )
    reference_d10 = (
        percentile(reference_diameter_values, 10)
        if reference_diameter_values
        else to_float(item.summary.get("d10_um"))
    )
    reference_d50 = (
        percentile(reference_diameter_values, 50)
        if reference_diameter_values
        else to_float(item.summary.get("d50_um"))
    )
    reference_d90 = (
        percentile(reference_diameter_values, 90)
        if reference_diameter_values
        else to_float(item.summary.get("d90_um"))
    )
    reference_area_ratio = to_float(item.summary.get("particle_area_ratio_percent"))
    field_area = to_float(item.summary.get("field_area_um2"))
    system_area = sum(float(row.get("area_um2", 0.0)) for row in classified)
    system_area_ratio = (system_area / field_area * 100.0) if field_area else None

    elapsed = time.perf_counter() - start
    summary_row: Dict[str, object] = {
        **base,
        "pixel_size_um": f"{pixel_size_um:.8g}",
        "pixel_size_source": pixel_size_source,
        "system_total_particles": int(stats["total_particles"]),
        "reference_total_particles": reference_total,
        "postprocess_removed_count": int(postprocess_info.get("postprocess_removed_count", 0) or 0),
        "postprocess_background_removed_count": int(
            postprocess_info.get("postprocess_background_removed_count", 0) or 0
        ),
        "postprocess_surface_removed_count": int(
            postprocess_info.get("postprocess_surface_removed_count", 0) or 0
        ),
        "postprocess_before_count": int(postprocess_info.get("postprocess_before_count", len(instances)) or 0),
        "postprocess_after_count": int(postprocess_info.get("postprocess_after_count", len(instances)) or 0),
        "particle_count_error": int(stats["total_particles"]) - int(reference_total or 0),
        "particle_count_error_percent": error_percent(
            float(stats["total_particles"]),
            float(reference_total) if reference_total is not None else None,
        ),
        "system_mean_roundness": f"{system_mean_roundness:.6g}" if system_mean_roundness is not None else "",
        "reference_mean_roundness": (
            f"{reference_mean_roundness:.6g}" if reference_mean_roundness is not None else ""
        ),
        "mean_roundness_error": error_value(system_mean_roundness, reference_mean_roundness),
        "system_mean_sphericity_q": (
            f"{system_mean_q:.6g}" if system_mean_q is not None else ""
        ),
        "reference_mean_sphericity_q": (
            f"{reference_mean_q:.6g}" if reference_mean_q is not None else ""
        ),
        "mean_sphericity_q_error": error_value(system_mean_q, reference_mean_q),
        "system_diameter_mean_um": (
            f"{system_diameter_mean:.6g}" if system_diameter_mean is not None else ""
        ),
        "reference_diameter_mean_um": (
            f"{reference_diameter_mean:.6g}" if reference_diameter_mean is not None else ""
        ),
        "diameter_mean_error_um": error_value(system_diameter_mean, reference_diameter_mean),
        "system_d50_um": f"{system_d50:.6g}" if system_d50 is not None else "",
        "reference_d50_um": f"{reference_d50:.6g}" if reference_d50 is not None else "",
        "d50_error_um": error_value(system_d50, reference_d50),
        "system_d10_um": f"{system_d10:.6g}" if system_d10 is not None else "",
        "reference_d10_um": f"{reference_d10:.6g}" if reference_d10 is not None else "",
        "system_d90_um": f"{system_d90:.6g}" if system_d90 is not None else "",
        "reference_d90_um": f"{reference_d90:.6g}" if reference_d90 is not None else "",
        "system_spherical_rate_percent": stats.get("sphericity_rate_s_percent", ""),
        "reference_spherical_rate_percent": (
            f"{reference_spherical_rate:.6g}" if reference_spherical_rate is not None else ""
        ),
        "spherical_rate_error_percent_points": error_value(
            float(stats.get("sphericity_rate_s_percent", 0.0)),
            reference_spherical_rate,
        ),
        "system_agglomerate_rate_percent": stats.get("agglomerate_rate_percent", ""),
        "system_agglomerate_area_rate_percent": stats.get("agglomerate_area_rate_percent", ""),
        "reference_particle_area_ratio_percent": (
            f"{reference_area_ratio:.6g}" if reference_area_ratio is not None else ""
        ),
        "system_particle_area_ratio_percent": (
            f"{system_area_ratio:.6g}" if system_area_ratio is not None else ""
        ),
        "elapsed_seconds": f"{elapsed:.3f}",
        "status": "ok",
        "message": "",
    }

    system_bin_counts = count_by_bins(diameters, bins)
    reference_bin_counts = count_by_bins(reference_diameter_values, bins)
    bin_rows: List[Dict[str, object]] = []
    for label in system_bin_counts:
        system_count = system_bin_counts[label]
        reference_count = reference_bin_counts.get(label, 0)
        bin_rows.append(
            {
                **base,
                "diameter_bin_um": label,
                "system_count": system_count,
                "reference_count": reference_count,
                "count_error": system_count - reference_count,
                "count_error_percent": error_percent(float(system_count), float(reference_count)),
            }
        )

    particle_rows: List[Dict[str, object]] = []
    for row in classified:
        particle_rows.append({**base, "pixel_size_um": pixel_size_um, **row})

    if args.save_agglomerate_evidence:
        evidence_dir = output_dir / "agglomerate_evidence"
        evidence_name = str(
            item.summary.get("batch_id") or Path(item.image_path).stem
        ).strip()
        evidence_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in evidence_name
        )
        mode_name = str(base["inference_mode"])
        evidence_path = (
            evidence_dir / f"{evidence_name}_{mode_name}.agglomerate_pairs.csv"
        )
        write_csv(
            evidence_path,
            agglomerate_evidence_rows(base, pixel_size_um, stats),
            AGGLOMERATE_EVIDENCE_COLUMNS,
        )

    if args.save_visualizations:
        visual_dir = output_dir / "visualizations"
        visual_name = item.summary.get("batch_id") or Path(item.image_path).stem
        mode_name = base["inference_mode"]
        visual_path = visual_dir / f"{visual_name}_{mode_name}.png"
        visualized = draw_instances(pre.image, instances, classified)
        imwrite_unicode(visual_path, visualized)

    return summary_row, bin_rows, particle_rows


def mode_summary_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("status") == "ok":
            grouped[str(row.get("inference_mode", ""))].append(row)

    rows: List[Dict[str, object]] = []
    for mode, items in sorted(grouped.items()):
        system_total = sum(int(float(item.get("system_total_particles", 0) or 0)) for item in items)
        reference_total = sum(int(float(item.get("reference_total_particles", 0) or 0)) for item in items)
        removed_total = sum(int(float(item.get("postprocess_removed_count", 0) or 0)) for item in items)
        count_errors = [
            abs(float(item.get("particle_count_error", 0) or 0))
            for item in items
            if item.get("particle_count_error") not in {"", None}
        ]
        roundness_errors = [
            abs(float(item.get("mean_roundness_error", 0) or 0))
            for item in items
            if item.get("mean_roundness_error") not in {"", None}
        ]
        q_errors = [
            abs(float(item.get("mean_sphericity_q_error", 0) or 0))
            for item in items
            if item.get("mean_sphericity_q_error") not in {"", None}
        ]
        spherical_rate_errors = [
            abs(float(item.get("spherical_rate_error_percent_points", 0) or 0))
            for item in items
            if item.get("spherical_rate_error_percent_points") not in {"", None}
        ]
        d50_errors = [
            abs(float(item.get("d50_error_um", 0) or 0))
            for item in items
            if item.get("d50_error_um") not in {"", None}
        ]
        elapsed_values = [
            float(item.get("elapsed_seconds", 0) or 0)
            for item in items
            if item.get("elapsed_seconds") not in {"", None}
        ]
        total_error = system_total - reference_total
        rows.append(
            {
                "inference_mode": mode,
                "evaluated_images": len(items),
                "system_total_particles": system_total,
                "reference_total_particles": reference_total,
                "postprocess_removed_total": removed_total,
                "total_particle_error": total_error,
                "total_particle_error_percent": error_percent(
                    float(system_total),
                    float(reference_total) if reference_total else None,
                ),
                "mean_abs_particle_count_error": (
                    f"{sum(count_errors) / len(count_errors):.6g}" if count_errors else ""
                ),
                "mean_abs_roundness_error": (
                    f"{sum(roundness_errors) / len(roundness_errors):.6g}" if roundness_errors else ""
                ),
                "mean_abs_sphericity_q_error": (
                    f"{sum(q_errors) / len(q_errors):.6g}" if q_errors else ""
                ),
                "mean_abs_spherical_rate_error_percent_points": (
                    f"{sum(spherical_rate_errors) / len(spherical_rate_errors):.6g}"
                    if spherical_rate_errors
                    else ""
                ),
                "mean_abs_d50_error_um": (
                    f"{sum(d50_errors) / len(d50_errors):.6g}" if d50_errors else ""
                ),
                "mean_elapsed_seconds": (
                    f"{sum(elapsed_values) / len(elapsed_values):.6g}" if elapsed_values else ""
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    bins = parse_bins(args.diameter_bins_um)

    if args.fixed_eval_dir is not None:
        fixed_eval_dir = args.fixed_eval_dir.resolve()
        data_root = fixed_eval_dir
        items, skipped = load_fixed_eval_items(fixed_eval_dir)
    else:
        data_root = args.data_root.resolve()
        items, skipped = load_reference_items(
            summary_csv=args.reference_summary_csv,
            particle_csv=args.reference_particle_csv,
            data_root=data_root,
            source_kinds=args.source_kind,
        )
    if args.limit is not None:
        items = items[: max(0, int(args.limit))]

    output_dir.mkdir(parents=True, exist_ok=True)
    matched_rows = [
        {
            **item.summary,
            "image_path": safe_rel(item.image_path, data_root),
            "reference_particle_rows": len(item.particles),
            "status": "matched",
            "message": "",
        }
        for item in items
    ]
    write_csv(
        output_dir / "matched_reference_images.csv",
        matched_rows + skipped,
        [
            "source_kind",
            "workbook_path",
            "sheet_name",
            "batch_id",
            "particle_count",
            "source_mean_roundness",
            "diameter_mean_um",
            "d50_um",
            "image_path",
            "reference_particle_rows",
            "status",
            "message",
        ],
    )

    print(f"matched images: {len(items)}")
    print(f"skipped reference rows: {len(skipped)}")
    print(f"matched list: {output_dir / 'matched_reference_images.csv'}")
    if args.dry_run:
        return

    segmenter = build_segmenter(args)
    modes = selected_inference_modes(args)
    print(f"inference modes: {', '.join(modes)}")
    summary_rows: List[Dict[str, object]] = []
    bin_rows_all: List[Dict[str, object]] = []
    particle_rows_all: List[Dict[str, object]] = []

    for index, item in enumerate(items, start=1):
        name = item.summary.get("batch_id") or Path(item.image_path).stem
        for mode in modes:
            mode_args = argparse.Namespace(**vars(args))
            mode_args.inference_mode = None if mode == "classical" else mode
            label = f"{name} [{mode}]" if len(modes) > 1 else name
            print(f"[{index}/{len(items)}] evaluating {label}")
            try:
                summary_row, bin_rows, particle_rows = evaluate_item(
                    item=item,
                    segmenter=segmenter,
                    args=mode_args,
                    bins=bins,
                    output_dir=output_dir,
                    data_root=data_root,
                )
            except Exception as exc:
                summary_row = {
                    "source_kind": item.summary.get("source_kind", ""),
                    "workbook_path": item.summary.get("workbook_path", ""),
                    "sheet_name": item.summary.get("sheet_name", ""),
                    "batch_id": item.summary.get("batch_id", ""),
                    "image_path": safe_rel(item.image_path, data_root),
                    "inference_mode": mode,
                    "status": "error",
                    "message": str(exc),
                }
                bin_rows = []
                particle_rows = []
                print(f"  error: {exc}")
            summary_rows.append(summary_row)
            bin_rows_all.extend(bin_rows)
            particle_rows_all.extend(particle_rows)

    write_csv(output_dir / "evaluation_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "diameter_bin_comparison.csv", bin_rows_all, BIN_COLUMNS)
    write_csv(output_dir / "system_particle_features.csv", particle_rows_all, PARTICLE_COLUMNS)
    mode_rows = mode_summary_rows(summary_rows)
    write_csv(output_dir / "mode_summary.csv", mode_rows, MODE_SUMMARY_COLUMNS)

    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    count_errors = [abs(float(row["particle_count_error"])) for row in ok_rows if row.get("particle_count_error") != ""]
    roundness_errors = [
        abs(float(row["mean_roundness_error"]))
        for row in ok_rows
        if row.get("mean_roundness_error") not in {"", None}
    ]
    print(f"evaluated images: {len(ok_rows)}")
    if count_errors:
        print(f"mean absolute particle-count error: {sum(count_errors) / len(count_errors):.3f}")
    if roundness_errors:
        print(f"mean absolute roundness error: {sum(roundness_errors) / len(roundness_errors):.4f}")
    print(f"summary: {output_dir / 'evaluation_summary.csv'}")
    print(f"mode summary: {output_dir / 'mode_summary.csv'}")
    print(f"diameter bins: {output_dir / 'diameter_bin_comparison.csv'}")
    print(f"system particles: {output_dir / 'system_particle_features.csv'}")
    if args.save_agglomerate_evidence:
        print(f"agglomerate evidence: {output_dir / 'agglomerate_evidence'}")


if __name__ == "__main__":
    main()
