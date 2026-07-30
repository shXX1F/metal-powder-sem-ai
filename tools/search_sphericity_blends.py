from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_PAIRS = (
    ("roundness_crofton", "roundness_crofton_open3"),
    ("roundness_crofton", "roundness_crofton_close3"),
    ("roundness_crofton", "roundness_crofton_smooth3"),
    ("roundness_crofton_open3", "roundness_crofton_smooth3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search interpretable two-method blends for spherical-particle "
            "classification with image-level leave-one-out validation."
        )
    )
    parser.add_argument("--system-particle-csv", type=Path, required=True)
    parser.add_argument("--evaluation-summary-csv", type=Path, required=True)
    parser.add_argument("--particle-matches-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-threshold", type=float, default=0.90)
    parser.add_argument("--threshold-min", type=float, default=0.80)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.0005)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def threshold_values(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.asarray([start + index * step for index in range(count + 1)])


def blend_value(row: Dict[str, str], left: str, right: str, alpha: float) -> float | None:
    left_value = finite_float(row.get(left))
    right_value = finite_float(row.get(right))
    if left_value is None or right_value is None:
        return None
    return alpha * left_value + (1.0 - alpha) * right_value


def f1_score(reference: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(np.count_nonzero(reference & predicted))
    fp = int(np.count_nonzero(~reference & predicted))
    fn = int(np.count_nonzero(reference & ~predicted))
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else 0.0


def evaluate_blend(
    system_rows: Sequence[Dict[str, str]],
    reference_rates: Dict[str, float],
    match_rows: Sequence[Dict[str, str]],
    left: str,
    right: str,
    alpha: float,
    thresholds: np.ndarray,
    reference_threshold: float,
) -> Dict[str, object]:
    image_values: Dict[str, List[float]] = {}
    for row in system_rows:
        batch_id = row.get("batch_id", "").strip()
        value = blend_value(row, left, right, alpha)
        if batch_id in reference_rates and value is not None:
            image_values.setdefault(batch_id, []).append(value)

    image_ids = sorted(set(reference_rates) & set(image_values))
    rate_errors = np.empty((len(image_ids), len(thresholds)), dtype=np.float64)
    signed_errors = np.empty_like(rate_errors)
    for image_index, batch_id in enumerate(image_ids):
        values = np.sort(np.asarray(image_values[batch_id], dtype=np.float64))
        predicted_rates = (
            (len(values) - np.searchsorted(values, thresholds, side="left"))
            / len(values)
            * 100.0
        )
        signed = predicted_rates - float(reference_rates[batch_id])
        signed_errors[image_index] = signed
        rate_errors[image_index] = np.abs(signed)

    full_mae = np.mean(rate_errors, axis=0)
    full_bias = np.mean(signed_errors, axis=0)
    best_index = min(
        range(len(thresholds)),
        key=lambda index: (
            float(full_mae[index]),
            abs(float(full_bias[index])),
        ),
    )
    best_threshold = float(thresholds[best_index])

    held_errors: List[float] = []
    selected_thresholds: List[float] = []
    error_sums = np.sum(rate_errors, axis=0)
    signed_error_sums = np.sum(signed_errors, axis=0)
    for image_index in range(len(image_ids)):
        train_count = max(1, len(image_ids) - 1)
        train_mae = (error_sums - rate_errors[image_index]) / train_count
        train_bias = (
            signed_error_sums - signed_errors[image_index]
        ) / train_count
        selected_index = min(
            range(len(thresholds)),
            key=lambda index: (
                float(train_mae[index]),
                abs(float(train_bias[index])),
            ),
        )
        held_errors.append(float(rate_errors[image_index, selected_index]))
        selected_thresholds.append(float(thresholds[selected_index]))

    matched_values: List[float] = []
    matched_reference: List[bool] = []
    for row in match_rows:
        value = blend_value(row, left, right, alpha)
        reference_value = finite_float(row.get("reference_roundness"))
        if value is None or reference_value is None:
            continue
        matched_values.append(value)
        matched_reference.append(reference_value >= reference_threshold)
    matched_array = np.asarray(matched_values, dtype=np.float64)
    reference_array = np.asarray(matched_reference, dtype=bool)
    predicted_array = matched_array >= best_threshold

    return {
        "left_method": left,
        "right_method": right,
        "left_weight_alpha": alpha,
        "recommended_threshold": best_threshold,
        "full_image_rate_mae_percent_points": float(full_mae[best_index]),
        "full_image_rate_bias_percent_points": float(full_bias[best_index]),
        "loio_image_rate_mae_percent_points": float(np.mean(held_errors)),
        "loio_threshold_min": min(selected_thresholds),
        "loio_threshold_max": max(selected_thresholds),
        "matched_particle_f1": f1_score(reference_array, predicted_array),
        "evaluated_images": len(image_ids),
        "matched_particles": len(matched_values),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    system_rows = read_csv(args.system_particle_csv)
    evaluation_rows = read_csv(args.evaluation_summary_csv)
    match_rows = read_csv(args.particle_matches_csv)
    reference_rates = {
        row["batch_id"].strip(): float(row["reference_spherical_rate_percent"])
        for row in evaluation_rows
        if row.get("batch_id") and row.get("reference_spherical_rate_percent")
    }
    thresholds = threshold_values(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    alpha_count = int(round(1.0 / float(args.alpha_step)))
    alphas = [index * float(args.alpha_step) for index in range(alpha_count + 1)]
    rows = [
        evaluate_blend(
            system_rows,
            reference_rates,
            match_rows,
            left=left,
            right=right,
            alpha=alpha,
            thresholds=thresholds,
            reference_threshold=float(args.reference_threshold),
        )
        for left, right in DEFAULT_PAIRS
        for alpha in alphas
    ]
    rows.sort(
        key=lambda row: (
            float(row["loio_image_rate_mae_percent_points"]),
            float(row["full_image_rate_mae_percent_points"]),
            -float(row["matched_particle_f1"]),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sphericity_blend_ranking.csv", rows)
    (args.output_dir / "sphericity_blend_best.json").write_text(
        json.dumps(rows[0], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
