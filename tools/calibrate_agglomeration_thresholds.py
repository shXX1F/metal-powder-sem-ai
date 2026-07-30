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
    gap_px: float | None = None
    pixel_size_um: float | None = None
    evidence_tolerance_px: int | None = None
    group: str = ""
    review_key: str = ""


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
    parser.add_argument(
        "--min-contact-profile-exact-fraction",
        type=float,
        default=0.95,
        help=(
            "Only recommend thresholds whose pixel tolerance matches the radius "
            "used to measure contact_ratio for at least this fraction of rows."
        ),
    )
    parser.add_argument("--current-tolerance-um", type=float, default=0.30)
    parser.add_argument("--current-contact-ratio", type=float, default=0.09)
    parser.add_argument("--current-overlap-ratio", type=float, default=0.03)
    parser.add_argument(
        "--max-precision-drop",
        type=float,
        default=0.005,
        help="Maximum allowed pair-precision drop from the current thresholds.",
    )
    parser.add_argument(
        "--plateau-policy",
        choices=["conservative", "midpoint", "aggressive"],
        default="conservative",
        help=(
            "Tie-break policy for thresholds with identical confusion matrices. "
            "Conservative uses the highest contact threshold to limit graph "
            "bridging; midpoint selects the center of the stable plateau."
        ),
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


def _normalize_source_row(value: object) -> str:
    text = str(value or "").strip()
    try:
        parsed = float(text)
    except ValueError:
        return text
    if math.isfinite(parsed) and parsed.is_integer():
        return str(int(parsed))
    return text


def _review_key(row: dict, path: Path, row_number: int) -> str:
    source_csv = str(
        row.get("source_csv")
        or row.get("image_path")
        or path.name
    ).strip()
    source_name = Path(source_csv.replace("\\", "/")).name.casefold()
    source_row = _normalize_source_row(row.get("source_row"))
    if source_row:
        return f"{source_name}#{source_row}"
    return f"{path.resolve()}#{row_number}"


def load_truth(paths: Iterable[Path]) -> tuple[List[PairTruth], dict]:
    rows_by_key: dict[str, PairTruth] = {}
    duplicate_pairs_removed = 0
    ignored_rows = 0
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                truth = parse_truth(row.get("manual_truth"))
                gap_um = finite_float(row.get("gap_um"))
                gap_px = finite_float(row.get("gap_px"))
                pixel_size_um = finite_float(row.get("pixel_size_um"))
                evidence_tolerance = finite_float(row.get("tolerance_px"))
                contact_ratio = finite_float(row.get("contact_ratio"))
                overlap_ratio = finite_float(row.get("overlap_ratio"))
                if (
                    truth is None
                    or gap_um is None
                    or contact_ratio is None
                    or overlap_ratio is None
                ):
                    ignored_rows += 1
                    continue
                review_key = _review_key(row, path, row_number)
                pair = PairTruth(
                    source=path.name,
                    gap_um=gap_um,
                    contact_ratio=contact_ratio,
                    overlap_ratio=overlap_ratio,
                    truth=truth,
                    gap_px=gap_px,
                    pixel_size_um=pixel_size_um,
                    evidence_tolerance_px=(
                        int(round(evidence_tolerance))
                        if evidence_tolerance is not None
                        else None
                    ),
                    group=str(
                        row.get("image_path")
                        or row.get("source_csv")
                        or path.name
                    ),
                    review_key=review_key,
                )
                previous = rows_by_key.get(review_key)
                if previous is not None:
                    if previous.truth != pair.truth:
                        raise ValueError(
                            "Conflicting manual_truth labels for "
                            f"{review_key}: {previous.truth} vs {pair.truth}"
                        )
                    duplicate_pairs_removed += 1
                    continue
                rows_by_key[review_key] = pair
    return list(rows_by_key.values()), {
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "ignored_unlabeled_or_invalid_rows": ignored_rows,
    }


def resolve_tolerance_px(row: PairTruth, tolerance_um: float) -> int | None:
    if (
        row.gap_px is None
        or row.pixel_size_um is None
        or row.pixel_size_um <= 0
    ):
        return None
    return max(1, int(math.ceil(float(tolerance_um) / row.pixel_size_um)))


def predict(
    row: PairTruth,
    tolerance_um: float,
    min_contact_ratio: float,
    min_overlap_ratio: float,
) -> bool:
    tolerance_px = resolve_tolerance_px(row, tolerance_um)
    if tolerance_px is not None and row.gap_px is not None:
        gap_match = row.gap_px <= float(tolerance_px) + 0.5
    else:
        gap_match = row.gap_um <= tolerance_um
    contact_match = gap_match and row.contact_ratio >= min_contact_ratio
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
    profile_known = profile_exact = 0
    group_counts: dict[str, list[int]] = {}
    for row in rows:
        tolerance_px = resolve_tolerance_px(row, tolerance_um)
        if tolerance_px is not None and row.evidence_tolerance_px is not None:
            profile_known += 1
            if tolerance_px == row.evidence_tolerance_px:
                profile_exact += 1
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
        group = row.group or row.source
        counts = group_counts.setdefault(group, [0, 0])
        counts[1] += 1
        counts[0] += int(predicted == row.truth)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0
    objective = 0.60 * f1 + 0.40 * balanced_accuracy
    group_accuracies = [
        safe_divide(correct, total) for correct, total in group_counts.values()
    ]
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
        "group_accuracy_mean": (
            sum(group_accuracies) / len(group_accuracies)
            if group_accuracies
            else 0.0
        ),
        "group_accuracy_min": min(group_accuracies) if group_accuracies else 0.0,
        "contact_profile_exact_pairs": profile_exact,
        "contact_profile_known_pairs": profile_known,
        "contact_profile_exact_fraction": (
            safe_divide(profile_exact, profile_known) if profile_known else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError("No *.agglomerate_pairs.csv files were found")
    rows, load_stats = load_truth(paths)
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
    current = evaluate(
        rows,
        tolerance_um=float(args.current_tolerance_um),
        min_contact_ratio=float(args.current_contact_ratio),
        min_overlap_ratio=float(args.current_overlap_ratio),
    )
    min_exact_fraction = min(
        1.0, max(0.0, float(args.min_contact_profile_exact_fraction))
    )
    max_precision_drop = max(0.0, float(args.max_precision_drop))
    precision_floor = max(0.0, float(current["precision"]) - max_precision_drop)
    for result in results:
        result["contact_profile_eligible"] = (
            float(result["contact_profile_exact_fraction"]) >= min_exact_fraction
        )
        result["precision_eligible"] = (
            float(result["precision"]) >= precision_floor
        )
        result["eligible_for_recommendation"] = (
            bool(result["contact_profile_eligible"])
            and bool(result["precision_eligible"])
        )
    results.sort(
        key=lambda item: (
            not bool(item["eligible_for_recommendation"]),
            -float(item["objective"]),
            -float(item["f1"]),
            int(item["fp"]),
            float(item["tolerance_um"]),
        )
    )
    eligible = [
        result for result in results if bool(result["eligible_for_recommendation"])
    ]
    if not eligible:
        raise ValueError(
            "No threshold candidate satisfies the exact-contact and precision "
            "constraints."
        )
    leading = eligible[0]
    leading_confusion = tuple(
        int(leading[key]) for key in ("tp", "fp", "tn", "fn")
    )
    stable_plateau = [
        result
        for result in eligible
        if tuple(int(result[key]) for key in ("tp", "fp", "tn", "fn"))
        == leading_confusion
    ]
    contact_midpoint = (
        min(float(result["min_contact_ratio"]) for result in stable_plateau)
        + max(float(result["min_contact_ratio"]) for result in stable_plateau)
    ) / 2.0
    if args.plateau_policy == "conservative":
        contact_key = lambda item: -float(item["min_contact_ratio"])
    elif args.plateau_policy == "aggressive":
        contact_key = lambda item: float(item["min_contact_ratio"])
    else:
        contact_key = lambda item: abs(
            float(item["min_contact_ratio"]) - contact_midpoint
        )
    best = min(
        stable_plateau,
        key=lambda item: (
            abs(float(item["tolerance_um"]) - float(args.current_tolerance_um)),
            abs(float(item["min_overlap_ratio"]) - float(args.current_overlap_ratio)),
            contact_key(item),
            float(item["min_contact_ratio"]),
        ),
    )
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
        **load_stats,
        "source_files": [str(path) for path in paths],
        "objective": "0.60 * pair_f1 + 0.40 * balanced_accuracy",
        "production_equivalent_gap_test": (
            "gap_px <= ceil(tolerance_um / pixel_size_um) + 0.5"
        ),
        "min_contact_profile_exact_fraction": min_exact_fraction,
        "max_precision_drop": max_precision_drop,
        "precision_floor": precision_floor,
        "stable_plateau_candidates": len(stable_plateau),
        "stable_contact_midpoint": contact_midpoint,
        "plateau_policy": args.plateau_policy,
        "current": current,
        "best": best,
        "improvement": {
            "precision": best["precision"] - current["precision"],
            "recall": best["recall"] - current["recall"],
            "f1": best["f1"] - current["f1"],
            "balanced_accuracy": (
                best["balanced_accuracy"] - current["balanced_accuracy"]
            ),
            "objective": best["objective"] - current["objective"],
        },
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
