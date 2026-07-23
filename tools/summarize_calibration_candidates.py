from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank fixed-set calibration candidates.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def values(rows: Iterable[Dict[str, str]], key: str, absolute: bool = False) -> List[float]:
    result: List[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(abs(value) if absolute else value)
    return result


def mean(items: Sequence[float]) -> float:
    return sum(items) / len(items) if items else float("nan")


def recommended_threshold(path: Path) -> Dict[str, str]:
    rows = read_csv(path)
    for row in rows:
        if str(row.get("recommended", "")).lower() == "true":
            return row
    return min(rows, key=lambda row: float(row["selection_score"]))


def summarize(candidate_dir: Path) -> Dict[str, object]:
    summary = read_csv(candidate_dir / "evaluation_summary.csv")
    bins = read_csv(candidate_dir / "diameter_bin_comparison.csv")
    matching = read_csv(candidate_dir / "calibration" / "matching_summary.csv")
    threshold = recommended_threshold(
        candidate_dir / "calibration" / "sphericity_rate_thresholds.csv"
    )

    count_mae = mean(values(summary, "particle_count_error", absolute=True))
    roundness_mae = mean(values(summary, "mean_roundness_error", absolute=True))
    q_mae = mean(values(summary, "mean_sphericity_q_error", absolute=True))
    d50_mae = mean(values(summary, "d50_error_um", absolute=True))
    diameter_mean_mae = mean(values(summary, "diameter_mean_error_um", absolute=True))
    spherical_rate_mae = float(threshold["image_spherical_rate_mae_percent_points"])
    reference_total = sum(int(float(row.get("reference_total_particles", 0) or 0)) for row in summary)
    bin_absolute_error = sum(abs(int(float(row.get("count_error", 0) or 0))) for row in bins)
    unmatched_reference = sum(
        int(float(row.get("unmatched_reference_count", 0) or 0)) for row in matching
    )
    unmatched_system = sum(
        int(float(row.get("unmatched_system_count", 0) or 0)) for row in matching
    )

    composite = (
        0.25 * q_mae / 0.02
        + 0.25 * roundness_mae / 0.015
        + 0.20 * spherical_rate_mae / 5.0
        + 0.10 * count_mae / 10.0
        + 0.08 * d50_mae / 2.0
        + 0.05 * diameter_mean_mae / 2.0
        + 0.07 * (bin_absolute_error / max(reference_total, 1)) / 0.20
    )
    return {
        "candidate": candidate_dir.name,
        "composite_score_lower_is_better": composite,
        "mean_sphericity_q_mae": q_mae,
        "mean_roundness_c_mae": roundness_mae,
        "spherical_rate_s_mae_percent_points": spherical_rate_mae,
        "spherical_rate_threshold": float(threshold["system_roundness_threshold"]),
        "spherical_rate_f1": float(threshold["f1"]),
        "particle_count_mae": count_mae,
        "d50_mae_um": d50_mae,
        "diameter_mean_mae_um": diameter_mean_mae,
        "diameter_bin_absolute_error": bin_absolute_error,
        "unmatched_reference_particles": unmatched_reference,
        "unmatched_system_particles": unmatched_system,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for candidate_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        required = candidate_dir / "calibration" / "sphericity_rate_thresholds.csv"
        if required.exists():
            rows.append(summarize(candidate_dir))
    rows.sort(key=lambda row: float(row["composite_score_lower_is_better"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["rank", *[key for key in rows[0] if key != "rank"]] if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"#{row['rank']} {row['candidate']}: score={row['composite_score_lower_is_better']:.4f}, "
            f"Q={row['mean_sphericity_q_mae']:.4f}, C={row['mean_roundness_c_mae']:.4f}, "
            f"S={row['spherical_rate_s_mae_percent_points']:.3f}pp, "
            f"count={row['particle_count_mae']:.3f}, D50={row['d50_mae_um']:.3f}um"
        )


if __name__ == "__main__":
    main()
