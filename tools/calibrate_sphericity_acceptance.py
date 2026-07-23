from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the roundness threshold against official image-level "
            "spherical-rate references and validate it by leave-one-image-out."
        )
    )
    parser.add_argument("--system-particle-csv", type=Path, required=True)
    parser.add_argument("--evaluation-summary-csv", type=Path, required=True)
    parser.add_argument("--particle-matches-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roundness-column", default="roundness_crofton")
    parser.add_argument("--reference-threshold", type=float, default=0.90)
    parser.add_argument("--threshold-min", type=float, default=0.84)
    parser.add_argument("--threshold-max", type=float, default=0.96)
    parser.add_argument("--threshold-step", type=float, default=0.0025)
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


def threshold_values(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def image_references(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for row in rows:
        batch_id = row.get("batch_id", "").strip()
        value = finite_float(row.get("reference_spherical_rate_percent"))
        if batch_id and value is not None:
            output[batch_id] = value
    return output


def system_roundness_by_image(
    rows: Sequence[Dict[str, str]], column: str
) -> Dict[str, List[float]]:
    output: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        batch_id = row.get("batch_id", "").strip()
        value = finite_float(row.get(column))
        if batch_id and value is not None:
            output[batch_id].append(value)
    return dict(output)


def image_rate_errors(
    roundness_by_image: Dict[str, List[float]],
    references: Dict[str, float],
    threshold: float,
    image_ids: Iterable[str] | None = None,
) -> Tuple[float, float, int]:
    selected = set(image_ids) if image_ids is not None else set(references)
    errors: List[float] = []
    for batch_id in sorted(selected):
        values = roundness_by_image.get(batch_id, [])
        if not values or batch_id not in references:
            continue
        predicted = sum(value >= threshold for value in values) / len(values) * 100.0
        errors.append(predicted - references[batch_id])
    if not errors:
        return math.nan, math.nan, 0
    return (
        sum(abs(error) for error in errors) / len(errors),
        sum(errors) / len(errors),
        len(errors),
    )


def matched_f1(
    rows: Sequence[Dict[str, str]],
    roundness_column: str,
    system_threshold: float,
    reference_threshold: float,
    image_ids: Iterable[str] | None = None,
) -> Tuple[float, int, int, int, int]:
    selected = set(image_ids) if image_ids is not None else None
    tp = fp = fn = tn = 0
    for row in rows:
        batch_id = row.get("batch_id", "").strip()
        if selected is not None and batch_id not in selected:
            continue
        predicted_value = finite_float(row.get(roundness_column))
        reference_value = finite_float(row.get("reference_roundness"))
        if predicted_value is None or reference_value is None:
            continue
        predicted = predicted_value >= system_threshold
        reference = reference_value >= reference_threshold
        if predicted and reference:
            tp += 1
        elif predicted:
            fp += 1
        elif reference:
            fn += 1
        else:
            tn += 1
    denominator = 2 * tp + fp + fn
    return ((2 * tp / denominator) if denominator else 0.0, tp, tn, fp, fn)


def choose_threshold(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return min(
        rows,
        key=lambda row: (
            float(row["image_rate_mae_percent_points"]),
            -float(row["matched_particle_f1"]),
            abs(float(row["image_rate_bias_percent_points"])),
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    system_rows = read_csv(args.system_particle_csv)
    evaluation_rows = read_csv(args.evaluation_summary_csv)
    match_rows = read_csv(args.particle_matches_csv) if args.particle_matches_csv else []
    references = image_references(evaluation_rows)
    roundness_by_image = system_roundness_by_image(system_rows, args.roundness_column)
    image_ids = sorted(set(references) & set(roundness_by_image))
    if not image_ids:
        raise RuntimeError("No common images between system particles and references")

    sweep_rows: List[Dict[str, object]] = []
    for threshold in threshold_values(
        args.threshold_min, args.threshold_max, args.threshold_step
    ):
        mae, bias, count = image_rate_errors(
            roundness_by_image, references, threshold, image_ids
        )
        f1, tp, tn, fp, fn = matched_f1(
            match_rows,
            args.roundness_column,
            threshold,
            args.reference_threshold,
            image_ids,
        )
        sweep_rows.append(
            {
                "system_roundness_threshold": threshold,
                "evaluated_images": count,
                "image_rate_mae_percent_points": mae,
                "image_rate_bias_percent_points": bias,
                "matched_particle_f1": f1,
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            }
        )
    best = choose_threshold(sweep_rows)

    loio_rows: List[Dict[str, object]] = []
    for held_out in image_ids:
        train_ids = [batch_id for batch_id in image_ids if batch_id != held_out]
        train_candidates: List[Dict[str, object]] = []
        for row in sweep_rows:
            threshold = float(row["system_roundness_threshold"])
            mae, bias, count = image_rate_errors(
                roundness_by_image, references, threshold, train_ids
            )
            f1, *_ = matched_f1(
                match_rows,
                args.roundness_column,
                threshold,
                args.reference_threshold,
                train_ids,
            )
            train_candidates.append(
                {
                    "system_roundness_threshold": threshold,
                    "image_rate_mae_percent_points": mae,
                    "image_rate_bias_percent_points": bias,
                    "matched_particle_f1": f1,
                    "evaluated_images": count,
                }
            )
        selected = choose_threshold(train_candidates)
        threshold = float(selected["system_roundness_threshold"])
        held_mae, held_bias, _ = image_rate_errors(
            roundness_by_image, references, threshold, [held_out]
        )
        loio_rows.append(
            {
                "held_out_batch_id": held_out,
                "selected_threshold": threshold,
                "held_out_absolute_error_percent_points": held_mae,
                "held_out_signed_error_percent_points": held_bias,
            }
        )

    selected_counts = Counter(float(row["selected_threshold"]) for row in loio_rows)
    loio_mae = sum(
        float(row["held_out_absolute_error_percent_points"]) for row in loio_rows
    ) / len(loio_rows)
    summary = {
        "roundness_column": args.roundness_column,
        "reference_threshold": args.reference_threshold,
        "recommended_system_threshold": float(best["system_roundness_threshold"]),
        "full_data_image_rate_mae_percent_points": float(
            best["image_rate_mae_percent_points"]
        ),
        "full_data_image_rate_bias_percent_points": float(
            best["image_rate_bias_percent_points"]
        ),
        "matched_particle_f1": float(best["matched_particle_f1"]),
        "loio_image_rate_mae_percent_points": loio_mae,
        "loio_selected_threshold_counts": {
            str(key): value for key, value in sorted(selected_counts.items())
        },
        "evaluated_images": len(image_ids),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "official_rate_threshold_sweep.csv", sweep_rows)
    write_csv(args.output_dir / "official_rate_loio.csv", loio_rows)
    (args.output_dir / "official_rate_calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
