from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manageable, image-stratified review CSV from "
            "*.agglomerate_pairs.csv evidence files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Evidence CSV files or directories containing them.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Combined review CSV with an empty manual_truth column.",
    )
    parser.add_argument("--accepted-target", type=int, default=80)
    parser.add_argument("--rejected-target", type=int, default=80)
    parser.add_argument("--max-per-file-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def find_evidence_files(inputs: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = item.resolve()
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.agglomerate_pairs.csv")))
        elif path.is_file():
            paths.append(path)
    return sorted({path.resolve() for path in paths})


def is_accepted(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def read_evidence(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for source_row, row in enumerate(csv.DictReader(handle), start=2):
                copied = dict(row)
                copied["source_csv"] = path.name
                copied["source_row"] = str(source_row)
                copied["review_bucket"] = (
                    "currently_accepted"
                    if is_accepted(row.get("accepted"))
                    else "currently_rejected"
                )
                copied["manual_truth"] = ""
                rows.append(copied)
    return rows


def stratified_sample(
    rows: Sequence[Dict[str, str]],
    *,
    target: int,
    max_per_file: int,
    seed: int,
) -> List[Dict[str, str]]:
    if target <= 0:
        return []
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["source_csv"]].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: List[Dict[str, str]] = []
    selected_ids = set()
    per_file = defaultdict(int)
    ordered_files = sorted(buckets)
    while len(selected) < target:
        progressed = False
        for source_csv in ordered_files:
            bucket = buckets[source_csv]
            if not bucket or per_file[source_csv] >= max_per_file:
                continue
            row = bucket.pop()
            selected.append(row)
            selected_ids.add((row["source_csv"], row["source_row"]))
            per_file[source_csv] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break

    if len(selected) < target:
        remaining = [
            row
            for row in rows
            if (row["source_csv"], row["source_row"]) not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])
    return selected


def output_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    preferred = [
        "source_csv",
        "source_row",
        "review_bucket",
        "source_kind",
        "batch_id",
        "image_path",
        "inference_mode",
        "pixel_size_um",
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
    available = {key for row in rows for key in row}
    columns = [column for column in preferred if column in available]
    columns.extend(sorted(available - set(columns)))
    return columns


def main() -> None:
    args = parse_args()
    files = find_evidence_files(args.inputs)
    if not files:
        raise FileNotFoundError("No *.agglomerate_pairs.csv files were found")
    rows = read_evidence(files)
    accepted = [row for row in rows if is_accepted(row.get("accepted"))]
    rejected = [row for row in rows if not is_accepted(row.get("accepted"))]
    selected = stratified_sample(
        accepted,
        target=int(args.accepted_target),
        max_per_file=int(args.max_per_file_per_class),
        seed=int(args.seed),
    )
    selected.extend(
        stratified_sample(
            rejected,
            target=int(args.rejected_target),
            max_per_file=int(args.max_per_file_per_class),
            seed=int(args.seed) + 1,
        )
    )
    selected.sort(
        key=lambda row: (
            str(row.get("batch_id", "")),
            str(row.get("review_bucket", "")),
            int(row.get("source_row", 0) or 0),
        )
    )

    output_path = args.output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = output_columns(selected)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "evidence_files": len(files),
        "available_pairs": len(rows),
        "available_accepted": len(accepted),
        "available_rejected": len(rejected),
        "selected_pairs": len(selected),
        "selected_accepted": sum(is_accepted(row.get("accepted")) for row in selected),
        "selected_rejected": sum(
            not is_accepted(row.get("accepted")) for row in selected
        ),
        "seed": int(args.seed),
        "output_csv": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
