from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


MANIFEST_COLUMNS = [
    "eval_id",
    "batch_id",
    "source_kind",
    "source_image_path",
    "frozen_image_path",
    "workbook_path",
    "sheet_name",
    "particle_count",
    "source_mean_roundness",
    "diameter_mean_um",
    "d50_um",
    "reference_particle_rows",
    "status",
    "message",
]


README_TEXT = """# fixed_eval_20

This directory is the frozen evaluation set for model comparison.

Rules:
- Do not add these images to train/val datasets.
- Use this set only for evaluation after code or model changes.
- Keep file names and manifest rows stable so metrics stay comparable.

Main files:
- images/: frozen copies of the original SEM images.
- manifest.csv: image-level reference values and source paths.
- reference_particles.csv: particle-level reference values from data_98986.
- exclude_from_training.txt: source paths that must be excluded from future training sets.

Recommended evaluation command:

```powershell
D:\\anaconda3x\\envs\\sdsd_torch\\python.exe tools\\evaluate_data_98986_baseline.py --weights runs\\maskrcnn_particle_last.pth --source-kind result-return --limit 20 --infer-pixel-size-from-field-area --output-dir outputs\\fixed_eval_20_eval --device cuda --inference-mode merged
```
"""


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalize_relative_path(path_text: str) -> Path:
    parts = [part for part in path_text.replace("\\", "/").split("/") if part]
    return Path(*parts)


def particle_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("workbook_path", ""),
        row.get("sheet_name", ""),
        row.get("batch_id", ""),
    )


def safe_image_name(index: int, row: Dict[str, str], source_path: Path) -> str:
    batch_id = row.get("batch_id", "").strip() or source_path.stem
    suffix = source_path.suffix.lower() or ".png"
    return f"{index:02d}_{batch_id}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the frozen 20-image evaluation set.")
    parser.add_argument(
        "--matched-csv",
        type=Path,
        default=Path("outputs")
        / "data_98986_parent_protect_inside_20"
        / "matched_reference_images.csv",
    )
    parser.add_argument(
        "--particle-csv",
        type=Path,
        default=Path("outputs")
        / "data_98986_roundness"
        / "data_98986_particle_roundness.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_98986") / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "fixed_eval_20",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing fixed evaluation directory before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    matched_rows = [
        row
        for row in read_csv(args.matched_csv)
        if row.get("status") == "matched" and row.get("image_path")
    ][: max(0, int(args.limit))]
    if len(matched_rows) != int(args.limit):
        raise ValueError(f"Expected {args.limit} matched rows, got {len(matched_rows)}")

    particles = read_csv(args.particle_csv) if args.particle_csv.exists() else []
    wanted_keys = {particle_key(row) for row in matched_rows}
    reference_particles = [row for row in particles if particle_key(row) in wanted_keys]

    manifest_rows: List[Dict[str, object]] = []
    excluded_paths: List[str] = []
    for index, row in enumerate(matched_rows, start=1):
        relative_source = normalize_relative_path(row["image_path"])
        source_path = data_root / relative_source
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        frozen_name = safe_image_name(index, row, source_path)
        frozen_path = image_dir / frozen_name
        if not frozen_path.exists() or frozen_path.stat().st_size != source_path.stat().st_size:
            shutil.copy2(source_path, frozen_path)

        excluded_paths.append(str(relative_source))
        manifest_rows.append(
            {
                "eval_id": f"eval_{index:02d}",
                "batch_id": row.get("batch_id", ""),
                "source_kind": row.get("source_kind", ""),
                "source_image_path": str(relative_source),
                "frozen_image_path": str(frozen_path.relative_to(output_dir)),
                "workbook_path": row.get("workbook_path", ""),
                "sheet_name": row.get("sheet_name", ""),
                "particle_count": row.get("particle_count", ""),
                "source_mean_roundness": row.get("source_mean_roundness", ""),
                "diameter_mean_um": row.get("diameter_mean_um", ""),
                "d50_um": row.get("d50_um", ""),
                "reference_particle_rows": row.get("reference_particle_rows", ""),
                "status": "fixed",
                "message": "reserved for evaluation; exclude from training",
            }
        )

    write_csv(output_dir / "manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    if reference_particles:
        write_csv(output_dir / "reference_particles.csv", reference_particles, list(reference_particles[0].keys()))

    (output_dir / "exclude_from_training.txt").write_text(
        "\n".join(excluded_paths) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(README_TEXT, encoding="utf-8")

    print(f"fixed images: {len(manifest_rows)}")
    print(f"reference particle rows: {len(reference_particles)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
