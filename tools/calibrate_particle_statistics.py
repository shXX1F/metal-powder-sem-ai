from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


DIAMETER_CANDIDATES = {
    "equivalent": "equivalent_diameter_um",
    "major_axis": "major_axis_um",
    "feret_max": "feret_diameter_um",
    "bbox_max": "bbox_max_diameter_um",
}

ROUNDNESS_CANDIDATES = {
    "contour": "roundness_contour",
    "crofton": "roundness_crofton",
    "subpixel": "roundness_subpixel",
}

Q_CANDIDATES = {
    "contour": "q_value_contour",
    "crofton": "q_value_crofton",
    "subpixel": "q_value_subpixel",
}

KEY_COLUMNS = ("workbook_path", "sheet_name", "batch_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match fixed-evaluation particles and calibrate measurement definitions."
    )
    parser.add_argument("--system-particle-csv", type=Path, required=True)
    parser.add_argument("--reference-particle-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-match-distance-px", type=float, default=5.0)
    parser.add_argument("--max-match-distance-px", type=float, default=40.0)
    parser.add_argument("--diameter-gate-factor", type=float, default=0.60)
    parser.add_argument("--diameter-bins-um", default="0,5,10,15,20,30,50,inf")
    parser.add_argument("--reference-spherical-roundness-threshold", type=float, default=0.90)
    parser.add_argument("--system-threshold-min", type=float, default=0.75)
    parser.add_argument("--system-threshold-max", type=float, default=0.98)
    parser.add_argument("--system-threshold-step", type=float, default=0.0025)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return tuple(row.get(column, "") for column in KEY_COLUMNS)


def number(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def reference_q_value(row: Dict[str, str]) -> float:
    value = number(row, "source_q_value")
    if math.isfinite(value):
        return value
    roundness = number(row, "source_roundness")
    return roundness * roundness if math.isfinite(roundness) else float("nan")


def rate_percent(values: Sequence[float], threshold: float) -> float:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return float("nan")
    return 100.0 * sum(value >= threshold for value in valid) / len(valid)


def binary_metrics(reference: Sequence[bool], predicted: Sequence[bool]) -> Dict[str, float]:
    tp = sum(target and result for target, result in zip(reference, predicted))
    tn = sum(not target and not result for target, result in zip(reference, predicted))
    fp = sum(not target and result for target, result in zip(reference, predicted))
    fn = sum(target and not result for target, result in zip(reference, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + tn + fp + fn
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
    }


def finite_values(rows: Iterable[Dict[str, str]], key: str) -> List[float]:
    values = [number(row, key) for row in rows]
    return [value for value in values if math.isfinite(value)]


def lower_percentile(values: Sequence[float], q: float) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan")
    try:
        return float(np.percentile(array, q, method="lower"))
    except TypeError:
        return float(np.percentile(array, q, interpolation="lower"))


def parse_bins(text: str) -> List[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower()
        values.append(float("inf") if chunk in {"inf", "infinity"} else float(chunk))
    if len(values) < 2 or any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Diameter bins must be strictly increasing.")
    return values


def bin_counts(values: Sequence[float], bins: Sequence[float]) -> np.ndarray:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return np.histogram(finite, bins=np.asarray(bins, dtype=np.float64))[0]


def group_rows(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row_key(row)].append(row)
    return grouped


def infer_pixel_size(rows: Sequence[Dict[str, str]]) -> float:
    direct = finite_values(rows, "pixel_size_um")
    if direct:
        return float(np.median(direct))
    ratios = []
    for row in rows:
        diameter_px = number(row, "equivalent_diameter_px")
        diameter_um = number(row, "equivalent_diameter_um")
        if diameter_px > 0 and diameter_um > 0:
            ratios.append(diameter_um / diameter_px)
    return float(np.median(ratios)) if ratios else 1.0


def match_image(
    system_rows: Sequence[Dict[str, str]],
    reference_rows: Sequence[Dict[str, str]],
    min_distance_px: float,
    max_distance_px: float,
    diameter_gate_factor: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    system_xy = np.asarray(
        [[number(row, "centroid_x"), number(row, "centroid_y")] for row in system_rows],
        dtype=np.float64,
    )
    reference_xy = np.asarray(
        [[number(row, "centroid_x"), number(row, "centroid_y")] for row in reference_rows],
        dtype=np.float64,
    )
    if system_xy.size == 0 or reference_xy.size == 0:
        return [], list(range(len(system_rows))), list(range(len(reference_rows)))

    distances = np.linalg.norm(system_xy[:, None, :] - reference_xy[None, :, :], axis=2)
    system_indices, reference_indices = linear_sum_assignment(distances)
    pixel_size_um = infer_pixel_size(system_rows)
    matches: List[Tuple[int, int, float]] = []
    matched_system: set[int] = set()
    matched_reference: set[int] = set()
    for system_index, reference_index in zip(system_indices, reference_indices):
        reference_diameter_px = number(reference_rows[reference_index], "diameter_um") / max(
            pixel_size_um, 1e-9
        )
        gate = float(
            np.clip(
                reference_diameter_px * diameter_gate_factor,
                min_distance_px,
                max_distance_px,
            )
        )
        distance = float(distances[system_index, reference_index])
        if distance <= gate:
            matches.append((int(system_index), int(reference_index), distance))
            matched_system.add(int(system_index))
            matched_reference.add(int(reference_index))
    unmatched_system = [index for index in range(len(system_rows)) if index not in matched_system]
    unmatched_reference = [index for index in range(len(reference_rows)) if index not in matched_reference]
    return matches, unmatched_system, unmatched_reference


def main() -> None:
    args = parse_args()
    system_rows = read_csv(args.system_particle_csv)
    reference_rows = read_csv(args.reference_particle_csv)
    system_groups = group_rows(system_rows)
    reference_groups = group_rows(reference_rows)
    bins = parse_bins(args.diameter_bins_um)

    match_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    unmatched_rows: List[Dict[str, object]] = []
    for key in sorted(reference_groups):
        system = system_groups.get(key, [])
        reference = reference_groups[key]
        matches, unmatched_system, unmatched_reference = match_image(
            system,
            reference,
            min_distance_px=float(args.min_match_distance_px),
            max_distance_px=float(args.max_match_distance_px),
            diameter_gate_factor=float(args.diameter_gate_factor),
        )
        common = dict(zip(KEY_COLUMNS, key))
        unmatched_rows.append(
            {
                **common,
                "system_count": len(system),
                "reference_count": len(reference),
                "matched_count": len(matches),
                "unmatched_system_count": len(unmatched_system),
                "unmatched_reference_count": len(unmatched_reference),
            }
        )

        for system_index, reference_index, distance in matches:
            predicted = system[system_index]
            target = reference[reference_index]
            target_roundness = number(target, "source_roundness")
            target_q = reference_q_value(target)
            row: Dict[str, object] = {
                **common,
                "system_particle_id": predicted.get("particle_id", ""),
                "reference_particle_id": target.get("particle_id", ""),
                "match_distance_px": distance,
                "score": predicted.get("score", ""),
                "touches_image_border": predicted.get("touches_image_border", ""),
                "reference_diameter_um": number(target, "diameter_um"),
                "reference_roundness": target_roundness,
                "reference_q_value": target_q,
                "reference_is_spherical": target_roundness
                >= float(args.reference_spherical_roundness_threshold),
            }
            for method, column in DIAMETER_CANDIDATES.items():
                value = number(predicted, column)
                row[f"diameter_{method}_um"] = value
                row[f"diameter_{method}_error_um"] = value - number(target, "diameter_um")
            for method, column in ROUNDNESS_CANDIDATES.items():
                value = number(predicted, column)
                row[f"roundness_{method}"] = value
                row[f"roundness_{method}_error"] = value - target_roundness
            for method, column in Q_CANDIDATES.items():
                value = number(predicted, column)
                row[f"sphericity_q_{method}"] = value
                row[f"sphericity_q_{method}_error"] = value - target_q
            match_rows.append(row)

        reference_diameters = finite_values(reference, "diameter_um")
        reference_roundness = finite_values(reference, "source_roundness")
        reference_q = [reference_q_value(row) for row in reference]
        reference_q = [value for value in reference_q if math.isfinite(value)]
        for method, column in DIAMETER_CANDIDATES.items():
            predicted = finite_values(system, column)
            image_rows.append(
                {
                    **common,
                    "metric_family": "diameter",
                    "method": method,
                    "system_count": len(predicted),
                    "reference_count": len(reference_diameters),
                    "mean_error": float(np.mean(predicted)) - float(np.mean(reference_diameters))
                    if predicted and reference_diameters
                    else "",
                    "d50_error": lower_percentile(predicted, 50)
                    - lower_percentile(reference_diameters, 50)
                    if predicted and reference_diameters
                    else "",
                    "bin_absolute_error": int(
                        np.abs(bin_counts(predicted, bins) - bin_counts(reference_diameters, bins)).sum()
                    ),
                }
            )
        for method, column in ROUNDNESS_CANDIDATES.items():
            predicted = finite_values(system, column)
            image_rows.append(
                {
                    **common,
                    "metric_family": "roundness",
                    "method": method,
                    "system_count": len(predicted),
                    "reference_count": len(reference_roundness),
                    "mean_error": float(np.mean(predicted)) - float(np.mean(reference_roundness))
                    if predicted and reference_roundness
                    else "",
                    "d50_error": "",
                    "bin_absolute_error": "",
                }
            )
        for method, column in Q_CANDIDATES.items():
            predicted = finite_values(system, column)
            image_rows.append(
                {
                    **common,
                    "metric_family": "sphericity_q",
                    "method": method,
                    "system_count": len(predicted),
                    "reference_count": len(reference_q),
                    "mean_error": float(np.mean(predicted)) - float(np.mean(reference_q))
                    if predicted and reference_q
                    else "",
                    "d50_error": "",
                    "bin_absolute_error": "",
                }
            )

    calibration_rows: List[Dict[str, object]] = []
    for family, methods, error_suffix in (
        ("diameter", DIAMETER_CANDIDATES, "error_um"),
        ("roundness", ROUNDNESS_CANDIDATES, "error"),
        ("sphericity_q", Q_CANDIDATES, "error"),
    ):
        family_image_rows = [row for row in image_rows if row["metric_family"] == family]
        for method in methods:
            particle_errors = [
                float(row[f"{family}_{method}_{error_suffix}"])
                for row in match_rows
                if math.isfinite(float(row[f"{family}_{method}_{error_suffix}"]))
            ]
            method_image_rows = [row for row in family_image_rows if row["method"] == method]
            mean_errors = [abs(float(row["mean_error"])) for row in method_image_rows if row["mean_error"] != ""]
            d50_errors = [abs(float(row["d50_error"])) for row in method_image_rows if row["d50_error"] != ""]
            score = (
                float(np.mean(mean_errors))
                + (float(np.mean(d50_errors)) if d50_errors else 0.0)
                + 0.25 * float(np.mean(np.abs(particle_errors)))
            )
            calibration_rows.append(
                {
                    "metric_family": family,
                    "method": method,
                    "matched_particles": len(particle_errors),
                    "particle_mae": float(np.mean(np.abs(particle_errors))),
                    "particle_bias": float(np.mean(particle_errors)),
                    "image_mean_mae": float(np.mean(mean_errors)),
                    "image_d50_mae": float(np.mean(d50_errors)) if d50_errors else "",
                    "diameter_bin_absolute_error": sum(
                        int(row["bin_absolute_error"] or 0) for row in method_image_rows
                    )
                    if family == "diameter"
                    else "",
                    "selection_score": score,
                }
            )

    if args.system_threshold_step <= 0:
        raise ValueError("--system-threshold-step must be positive.")
    threshold_values = np.arange(
        float(args.system_threshold_min),
        float(args.system_threshold_max) + 0.5 * float(args.system_threshold_step),
        float(args.system_threshold_step),
    )
    spherical_threshold_rows: List[Dict[str, object]] = []
    reference_threshold = float(args.reference_spherical_roundness_threshold)
    reference_labels = [bool(row["reference_is_spherical"]) for row in match_rows]
    for method, column in ROUNDNESS_CANDIDATES.items():
        matched_values = [float(row[f"roundness_{method}"]) for row in match_rows]
        for threshold in threshold_values:
            predicted_labels = [value >= threshold for value in matched_values]
            metrics = binary_metrics(reference_labels, predicted_labels)
            image_rate_errors: List[float] = []
            image_rate_biases: List[float] = []
            for key, reference in reference_groups.items():
                predicted_values = finite_values(system_groups.get(key, []), column)
                target_values = finite_values(reference, "source_roundness")
                predicted_rate = rate_percent(predicted_values, float(threshold))
                reference_rate = rate_percent(target_values, reference_threshold)
                if math.isfinite(predicted_rate) and math.isfinite(reference_rate):
                    difference = predicted_rate - reference_rate
                    image_rate_errors.append(abs(difference))
                    image_rate_biases.append(difference)
            image_rate_mae = float(np.mean(image_rate_errors)) if image_rate_errors else float("inf")
            objective = (
                image_rate_mae
                + 15.0 * (1.0 - float(metrics["f1"]))
                + 10.0 * (1.0 - float(metrics["balanced_accuracy"]))
            )
            spherical_threshold_rows.append(
                {
                    "method": method,
                    "reference_roundness_threshold": reference_threshold,
                    "system_roundness_threshold": float(threshold),
                    "matched_particles": len(match_rows),
                    **metrics,
                    "image_spherical_rate_mae_percent_points": image_rate_mae,
                    "image_spherical_rate_bias_percent_points": float(np.mean(image_rate_biases))
                    if image_rate_biases
                    else "",
                    "selection_score": objective,
                    "recommended": False,
                }
            )

    best_spherical_threshold = min(
        spherical_threshold_rows,
        key=lambda row: float(row["selection_score"]),
    )
    best_spherical_threshold["recommended"] = True

    for family in {row["metric_family"] for row in calibration_rows}:
        family_rows = [row for row in calibration_rows if row["metric_family"] == family]
        best = min(family_rows, key=lambda row: float(row["selection_score"]))
        for row in family_rows:
            row["recommended"] = row is best

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "particle_matches.csv", match_rows)
    write_csv(args.output_dir / "image_method_comparison.csv", image_rows)
    write_csv(args.output_dir / "matching_summary.csv", unmatched_rows)
    write_csv(args.output_dir / "calibration_summary.csv", calibration_rows)
    write_csv(args.output_dir / "sphericity_rate_thresholds.csv", spherical_threshold_rows)

    print(f"matched particles: {len(match_rows)}")
    for row in calibration_rows:
        marker = "*" if row.get("recommended") else " "
        print(
            f"{marker} {row['metric_family']}/{row['method']}: "
            f"particle MAE={float(row['particle_mae']):.4f}, "
            f"image mean MAE={float(row['image_mean_mae']):.4f}, "
            f"score={float(row['selection_score']):.4f}"
        )
    print(
        "* spherical-rate threshold: "
        f"method={best_spherical_threshold['method']}, "
        f"system C>={float(best_spherical_threshold['system_roundness_threshold']):.4f}, "
        f"image rate MAE={float(best_spherical_threshold['image_spherical_rate_mae_percent_points']):.3f} pp, "
        f"F1={float(best_spherical_threshold['f1']):.4f}"
    )
    print(f"summary: {args.output_dir / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
