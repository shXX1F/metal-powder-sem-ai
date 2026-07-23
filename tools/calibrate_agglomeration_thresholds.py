from __future__ import annotations

import argparse
import csv
from glob import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TRUE_VALUES = {"1", "true", "yes", "y", "agglomerate", "positive", "团聚", "是"}
FALSE_VALUES = {"0", "false", "no", "n", "normal", "negative", "非团聚", "否"}


@dataclass(frozen=True)
class PairTruth:
    source: str
    gap_um: float
    contact_ratio: float
    overlap_ratio: float
    truth: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据人工复核的团聚颗粒对证据，校准物理间隙和强接触阈值"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="一个或多个 report.agglomerate_pairs.csv 文件或包含这些文件的目录",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/agglomeration_calibration",
        help="校准结果目录",
    )
    parser.add_argument(
        "--tolerance-grid",
        default="0.10,0.20,0.30,0.50,0.80,1.00",
        help="候选物理间隙阈值(um)",
    )
    parser.add_argument(
        "--contact-grid",
        default="0.05,0.08,0.10,0.12,0.15,0.20,0.25",
        help="候选接触弧占比阈值",
    )
    parser.add_argument(
        "--overlap-grid",
        default="0.01,0.02,0.03,0.05,0.08,0.10",
        help="候选掩膜重叠占比阈值",
    )
    parser.add_argument(
        "--min-labeled-pairs",
        type=int,
        default=20,
        help="允许开始校准的最少人工复核颗粒对数量",
    )
    return parser.parse_args()


def parse_grid(text: str) -> List[float]:
    values = sorted({float(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid threshold grid: {text}")
    return values


def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.agglomerate_pairs.csv")))
        elif path.is_file():
            paths.append(path)
        else:
            paths.extend(Path(match) for match in sorted(glob(raw, recursive=True)))
    unique = {path.resolve() for path in paths}
    return sorted(unique)


def parse_truth(value: object) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_truth(paths: Iterable[Path]) -> List[PairTruth]:
    rows: List[PairTruth] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                truth = parse_truth(row.get("manual_truth"))
                gap_um = finite_float(row.get("gap_um"))
                contact_ratio = finite_float(row.get("contact_ratio"))
                overlap_ratio = finite_float(row.get("overlap_ratio"))
                if (
                    truth is None
                    or gap_um is None
                    or contact_ratio is None
                    or overlap_ratio is None
                ):
                    continue
                rows.append(
                    PairTruth(
                        source=path.name,
                        gap_um=gap_um,
                        contact_ratio=contact_ratio,
                        overlap_ratio=overlap_ratio,
                        truth=truth,
                    )
                )
    return rows


def predict(
    row: PairTruth,
    tolerance_um: float,
    min_contact_ratio: float,
    min_overlap_ratio: float,
) -> bool:
    contact_match = (
        row.gap_um <= tolerance_um and row.contact_ratio >= min_contact_ratio
    )
    overlap_match = row.overlap_ratio >= min_overlap_ratio
    return contact_match or overlap_match


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    rows: Sequence[PairTruth],
    tolerance_um: float,
    min_contact_ratio: float,
    min_overlap_ratio: float,
) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = predict(
            row,
            tolerance_um=tolerance_um,
            min_contact_ratio=min_contact_ratio,
            min_overlap_ratio=min_overlap_ratio,
        )
        if predicted and row.truth:
            tp += 1
        elif predicted:
            fp += 1
        elif row.truth:
            fn += 1
        else:
            tn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0
    objective = 0.60 * f1 + 0.40 * balanced_accuracy
    return {
        "tolerance_um": tolerance_um,
        "min_contact_ratio": min_contact_ratio,
        "min_overlap_ratio": min_overlap_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "objective": objective,
    }


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError("No *.agglomerate_pairs.csv files were found")
    rows = load_truth(paths)
    if len(rows) < int(args.min_labeled_pairs):
        raise ValueError(
            f"Only {len(rows)} labeled pairs found; at least "
            f"{int(args.min_labeled_pairs)} are required"
        )
    positive_count = sum(1 for row in rows if row.truth)
    negative_count = len(rows) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both positive and negative manual_truth labels are required")

    results = [
        evaluate(rows, tolerance_um, contact_ratio, overlap_ratio)
        for tolerance_um in parse_grid(args.tolerance_grid)
        for contact_ratio in parse_grid(args.contact_grid)
        for overlap_ratio in parse_grid(args.overlap_grid)
    ]
    results.sort(
        key=lambda item: (
            -float(item["objective"]),
            -float(item["f1"]),
            int(item["fp"]),
            float(item["tolerance_um"]),
        )
    )
    best = results[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = output_dir / "agglomeration_threshold_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "labeled_pairs": len(rows),
        "positive_pairs": positive_count,
        "negative_pairs": negative_count,
        "source_files": [str(path) for path in paths],
        "objective": "0.60 * pair_f1 + 0.40 * balanced_accuracy",
        "best": best,
        "recommended_arguments": (
            f"--agglomerate-tolerance-um {best['tolerance_um']} "
            f"--agglomerate-min-contact-ratio {best['min_contact_ratio']} "
            f"--agglomerate-min-overlap-ratio {best['min_overlap_ratio']}"
        ),
        "note": (
            "This calibrates pair evidence only. Keep min group size aligned with "
            "the accepted project definition, currently 3 particles."
        ),
    }
    summary_path = output_dir / "agglomeration_calibration.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Ranking: {ranking_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
