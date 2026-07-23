from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


GROUP_COLUMNS = ("workbook_path", "sheet_name", "batch_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-image-out validation of measurement calibration."
    )
    parser.add_argument("--particle-matches", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return tuple(row.get(column, "") for column in GROUP_COLUMNS)


def finite_pairs(
    rows: Sequence[Dict[str, str]], predicted_key: str, reference_key: str
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, str]]]:
    predicted: List[float] = []
    reference: List[float] = []
    groups: List[Tuple[str, str, str]] = []
    for row in rows:
        try:
            x = float(row[predicted_key])
            y = float(row[reference_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            predicted.append(x)
            reference.append(y)
            groups.append(group_key(row))
    return np.asarray(predicted), np.asarray(reference), groups


def fit_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Tuple[str, str, str]],
    degree: int,
) -> np.ndarray:
    counts = Counter(groups)
    weights = np.asarray([1.0 / math.sqrt(counts[group]) for group in groups])
    return np.polyfit(x, y, degree, w=weights)


def clip_predictions(values: np.ndarray, family: str) -> np.ndarray:
    if family in {"roundness", "sphericity_q"}:
        return np.clip(values, 0.0, 1.0)
    return np.maximum(values, 0.0)


def summarize_errors(
    predicted: np.ndarray,
    reference: np.ndarray,
    groups: Sequence[Tuple[str, str, str]],
) -> Tuple[float, float]:
    particle_mae = float(np.mean(np.abs(predicted - reference)))
    image_errors = []
    for group in sorted(set(groups)):
        indices = np.asarray([value == group for value in groups], dtype=bool)
        image_errors.append(abs(float(np.mean(predicted[indices]) - np.mean(reference[indices]))))
    return particle_mae, float(np.mean(image_errors))


def cross_validate(
    rows: Sequence[Dict[str, str]],
    family: str,
    predicted_key: str,
    reference_key: str,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str, str]], np.ndarray]:
    x, y, groups = finite_pairs(rows, predicted_key, reference_key)
    output = np.empty_like(x)
    for held_out in sorted(set(groups)):
        test = np.asarray([group == held_out for group in groups], dtype=bool)
        train = ~test
        coefficients = fit_polynomial(
            x[train], y[train], [group for index, group in enumerate(groups) if train[index]], degree
        )
        output[test] = np.polyval(coefficients, x[test])
    output = clip_predictions(output, family)
    all_coefficients = fit_polynomial(x, y, groups, degree)
    return x, y, output, groups, all_coefficients


def main() -> None:
    args = parse_args()
    rows = read_csv(args.particle_matches)
    configurations = (
        ("roundness", "roundness_crofton", "reference_roundness"),
        ("sphericity_q", "sphericity_q_crofton", "reference_q_value"),
        ("diameter", "diameter_bbox_max_um", "reference_diameter_um"),
    )
    output_rows: List[Dict[str, object]] = []
    for family, predicted_key, reference_key in configurations:
        x, y, groups = finite_pairs(rows, predicted_key, reference_key)
        raw_particle_mae, raw_image_mae = summarize_errors(x, y, groups)
        output_rows.append(
            {
                "metric_family": family,
                "method": "raw",
                "particle_mae": raw_particle_mae,
                "image_mean_mae": raw_image_mae,
                "coefficient_2": "",
                "coefficient_1": 1.0,
                "coefficient_0": 0.0,
            }
        )
        for degree, method in ((1, "affine_loio"), (2, "quadratic_loio")):
            _, reference, calibrated, calibrated_groups, coefficients = cross_validate(
                rows, family, predicted_key, reference_key, degree
            )
            particle_mae, image_mae = summarize_errors(
                calibrated, reference, calibrated_groups
            )
            padded = np.pad(coefficients, (3 - len(coefficients), 0))
            output_rows.append(
                {
                    "metric_family": family,
                    "method": method,
                    "particle_mae": particle_mae,
                    "image_mean_mae": image_mae,
                    "coefficient_2": float(padded[-3]),
                    "coefficient_1": float(padded[-2]),
                    "coefficient_0": float(padded[-1]),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(output_rows[0])
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            f"{row['metric_family']}/{row['method']}: "
            f"particle MAE={row['particle_mae']:.6f}, image MAE={row['image_mean_mae']:.6f}, "
            f"coeff=({row['coefficient_2']}, {row['coefficient_1']}, {row['coefficient_0']})"
        )


if __name__ == "__main__":
    main()
