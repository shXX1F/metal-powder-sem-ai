from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_COLUMNS = [
    "count_mae",
    "roundness_mae",
    "d50_mae_um",
    "diameter_mean_mae_um",
    "diameter_bin_abs_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-set evaluation directories into one checkpoint table."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Discover evaluation directories recursively below --eval-root.",
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        default=[],
        help="Additional run as label=/path/to/evaluation_directory.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def to_float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except (TypeError, ValueError):
        return math.nan


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_run(label: str, run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "evaluation_summary.csv"
    bin_path = run_dir / "diameter_bin_comparison.csv"
    rows = [row for row in read_rows(summary_path) if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"No successful evaluation rows in {summary_path}")

    aggregate_bins: Dict[str, List[int]] = {}
    for row in read_rows(bin_path):
        values = aggregate_bins.setdefault(row["diameter_bin_um"], [0, 0])
        values[0] += int(float(row.get("system_count") or 0))
        values[1] += int(float(row.get("reference_count") or 0))

    system_total = sum(int(float(row["system_total_particles"])) for row in rows)
    reference_total = sum(int(float(row["reference_total_particles"])) for row in rows)
    result: Dict[str, object] = {
        "label": label,
        "run_dir": str(run_dir),
        "images": len(rows),
        "system_total": system_total,
        "reference_total": reference_total,
        "count_bias_total": system_total - reference_total,
        "count_mae": mean(abs(to_float(row["particle_count_error"])) for row in rows),
        "roundness_mae": mean(
            abs(to_float(row["mean_roundness_error"])) for row in rows
        ),
        "d50_mae_um": mean(abs(to_float(row["d50_error_um"])) for row in rows),
        "diameter_mean_mae_um": mean(
            abs(to_float(row["diameter_mean_error_um"])) for row in rows
        ),
        "diameter_bin_abs_error": sum(
            abs(system - reference) for system, reference in aggregate_bins.values()
        ),
    }
    for bin_name, (system, reference) in aggregate_bins.items():
        result[f"bin_{bin_name}_error"] = system - reference
    return result


def parse_extra_runs(values: List[str]) -> List[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --extra-run value: {value!r}")
        label, path = value.split("=", 1)
        result.append((label.strip(), Path(path.strip())))
    return result


def add_metric_ranks(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        row["rank_sum"] = 0
    for metric in METRIC_COLUMNS:
        ordered = sorted(rows, key=lambda row: float(row[metric]))
        for rank, row in enumerate(ordered, start=1):
            row[f"rank_{metric}"] = rank
            row["rank_sum"] = int(row["rank_sum"]) + rank


def main() -> None:
    args = parse_args()
    if args.recursive:
        run_dirs = [
            (str(path.parent.relative_to(args.eval_root)).replace("\\", "/"), path.parent)
            for path in sorted(args.eval_root.rglob("evaluation_summary.csv"))
        ]
    else:
        run_dirs = [
            (path.name, path)
            for path in sorted(args.eval_root.iterdir())
            if (path / "evaluation_summary.csv").is_file()
        ]
    run_dirs.extend(parse_extra_runs(args.extra_run))
    rows = [summarize_run(label, path) for label, path in run_dirs]
    if not rows:
        raise ValueError(f"No completed evaluation runs found under {args.eval_root}")

    add_metric_ranks(rows)
    rows.sort(key=lambda row: (int(row["rank_sum"]), float(row["count_mae"])))
    output_csv = args.output_csv or args.eval_root / "checkpoint_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "label,count_mae,roundness_mae,d50_mae_um,diameter_mean_mae_um,"
        "diameter_bin_abs_error,count_bias_total,rank_sum"
    )
    for row in rows:
        print(
            f"{row['label']},{float(row['count_mae']):.4f},"
            f"{float(row['roundness_mae']):.6f},{float(row['d50_mae_um']):.4f},"
            f"{float(row['diameter_mean_mae_um']):.4f},"
            f"{int(row['diameter_bin_abs_error'])},{int(row['count_bias_total'])},"
            f"{int(row['rank_sum'])}"
        )
    print(f"summary: {output_csv}")


if __name__ == "__main__":
    main()
