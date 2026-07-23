from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import BadZipFile, ZipFile


SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

TEXT = {
    "batch": "\u6279\u53f7",
    "name": "\u540d\u79f0",
    "value_um": "\u503c/\u5fae\u7c73",
    "id": "\u5e8f\u53f7",
    "particle_count": "\u9897\u7c92\u6570\u91cf",
    "particle_total_area": "\u9897\u7c92\u603b\u9762\u79ef",
    "field_area": "\u89c6\u573a\u9762\u79ef",
    "particle_area_ratio": "\u9897\u7c92\u9762\u79ef\u5360\u6bd4",
    "mean_area": "\u5e73\u5747\u9762\u79ef",
    "max_area": "\u6700\u5927\u9762\u79ef",
    "roundness": "\u5706\u5f62\u5ea6",
    "diameter_mean": "\u7c92\u5f84\u5e73\u5747\u503c",
    "perimeter": "\u5468\u957f",
    "area": "\u9762\u79ef",
    "position": "\u4f4d\u7f6e",
    "diameter": "\u7c92\u5f84(\u5fae\u7c73)",
    "major_axis": "\u957f\u5f84(\u5fae\u7c73)",
    "minor_axis": "\u77ed\u5f84(\u5fae\u7c73)",
    "axis_ratio": "\u957f\u5bbd\u6bd4",
}

SUMMARY_MAP = {
    TEXT["particle_count"]: "particle_count",
    TEXT["particle_total_area"]: "particle_total_area_um2",
    TEXT["field_area"]: "field_area_um2",
    TEXT["particle_area_ratio"]: "particle_area_ratio_percent",
    TEXT["mean_area"]: "mean_area_um2",
    TEXT["max_area"]: "max_area_um2",
    TEXT["roundness"]: "source_mean_roundness",
    TEXT["diameter_mean"]: "diameter_mean_um",
    "Dmin": "dmin_um",
    "D5": "d5_um",
    "D10": "d10_um",
    "D50": "d50_um",
    "D90": "d90_um",
    "D95": "d95_um",
    "Dmax": "dmax_um",
}

PARTICLE_MAP = {
    TEXT["perimeter"]: "perimeter_um",
    TEXT["area"]: "area_um2",
    TEXT["roundness"]: "source_roundness",
    TEXT["position"]: "position",
    TEXT["diameter"]: "diameter_um",
    TEXT["major_axis"]: "major_axis_um",
    TEXT["minor_axis"]: "minor_axis_um",
    TEXT["axis_ratio"]: "axis_ratio",
}

SUMMARY_COLUMNS = [
    "source_kind",
    "workbook_path",
    "sheet_name",
    "batch_id",
    "particle_count",
    "particle_total_area_um2",
    "field_area_um2",
    "particle_area_ratio_percent",
    "mean_area_um2",
    "max_area_um2",
    "source_mean_roundness",
    "diameter_mean_um",
    "dmin_um",
    "d5_um",
    "d10_um",
    "d50_um",
    "d90_um",
    "d95_um",
    "dmax_um",
    "particle_rows",
]

PARTICLE_COLUMNS = [
    "source_kind",
    "workbook_path",
    "sheet_name",
    "batch_id",
    "particle_id",
    "perimeter_um",
    "area_um2",
    "source_roundness",
    "source_q_value",
    "formula_roundness",
    "formula_q_value",
    "position",
    "centroid_x",
    "centroid_y",
    "diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "axis_ratio",
]


def col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: object) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    if abs(number - round(number)) > 1e-6:
        return None
    return int(round(number))


def parse_position(value: str) -> Tuple[Optional[float], Optional[float]]:
    match = re.search(r"X\s*:\s*([0-9.]+)\s*,\s*Y\s*:\s*([0-9.]+)", value or "")
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def shared_strings(zip_file: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("a:si", SPREADSHEET_NS):
        values.append("".join(text.text or "" for text in item.findall(".//a:t", SPREADSHEET_NS)))
    return values


def workbook_sheet_names(zip_file: ZipFile) -> List[str]:
    if "xl/workbook.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/workbook.xml"))
    names = []
    for sheet in root.findall(".//a:sheet", SPREADSHEET_NS):
        names.append(sheet.attrib.get("name", ""))
    return names


def cell_value(cell: ET.Element, strings: Sequence[str]) -> str:
    value_node = cell.find("a:v", SPREADSHEET_NS)
    if value_node is None:
        return ""
    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        index = int(raw) if raw.isdigit() else -1
        return strings[index] if 0 <= index < len(strings) else raw
    return raw


def read_sheet_rows(zip_file: ZipFile, sheet_path: str, strings: Sequence[str]) -> List[List[str]]:
    root = ET.fromstring(zip_file.read(sheet_path))
    rows: List[List[str]] = []
    for row in root.findall(".//a:row", SPREADSHEET_NS):
        values: Dict[int, str] = {}
        max_col = 0
        for cell in row.findall("a:c", SPREADSHEET_NS):
            idx = col_index(cell.attrib.get("r", ""))
            if idx <= 0:
                continue
            values[idx] = cell_value(cell, strings)
            max_col = max(max_col, idx)
        if max_col:
            rows.append([values.get(i, "") for i in range(1, max_col + 1)])
        else:
            rows.append([])
    return rows


def pad(row: Sequence[str], length: int) -> List[str]:
    return list(row) + [""] * max(0, length - len(row))


def parse_summary(rows: Sequence[Sequence[str]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if rows:
        first = pad(rows[0], 2)
        if first[0] == TEXT["batch"]:
            result["batch_id"] = first[1]

    names_row = next((pad(row, 30) for row in rows if row and row[0] == TEXT["name"]), None)
    values_row = next((pad(row, 30) for row in rows if row and row[0] == TEXT["value_um"]), None)
    if names_row and values_row:
        for name, value in zip(names_row[1:], values_row[1:]):
            column = SUMMARY_MAP.get(name)
            if column:
                result[column] = to_float(value) if column != "particle_count" else to_int(value)
    return result


def find_particle_header(rows: Sequence[Sequence[str]]) -> Optional[int]:
    for index, row in enumerate(rows):
        if not row:
            continue
        if row[0] == TEXT["id"] and TEXT["roundness"] in row and TEXT["position"] in row:
            return index
    return None


def parse_particles(rows: Sequence[Sequence[str]]) -> List[Dict[str, object]]:
    header_index = find_particle_header(rows)
    if header_index is None:
        return []
    header = list(rows[header_index])
    index_to_column = {idx: PARTICLE_MAP[name] for idx, name in enumerate(header) if name in PARTICLE_MAP}
    particles: List[Dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        padded = pad(row, len(header))
        particle_id = to_int(padded[0] if padded else "")
        if particle_id is None:
            break
        item: Dict[str, object] = {"particle_id": particle_id}
        for idx, column in index_to_column.items():
            raw = padded[idx]
            if column == "position":
                item[column] = raw
                cx, cy = parse_position(raw)
                item["centroid_x"] = cx
                item["centroid_y"] = cy
            else:
                item[column] = to_float(raw)

        source_roundness = to_float(item.get("source_roundness"))
        if source_roundness is not None:
            item["source_q_value"] = source_roundness * source_roundness

        area = to_float(item.get("area_um2"))
        perimeter = to_float(item.get("perimeter_um"))
        if area is not None and perimeter and perimeter > 0:
            q_value = 4.0 * math.pi * area / (perimeter * perimeter)
            item["formula_q_value"] = q_value
            item["formula_roundness"] = math.sqrt(max(0.0, q_value))
        particles.append(item)
    return particles


def source_kind(path: Path) -> str:
    parts = set(path.parts)
    if "result-return" in parts:
        return "result-return"
    if "\u7c89\u672b\u5706\u5ea6\u6c47\u603b" in parts:
        return "roundness-summary"
    return "unknown"


def iter_workbooks(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("*.xlsx"))


def parse_workbook(path: Path, root_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    particle_rows: List[Dict[str, object]] = []
    relative_path = str(path.relative_to(root_dir))
    kind = source_kind(path)

    try:
        with ZipFile(path) as zip_file:
            strings = shared_strings(zip_file)
            names = workbook_sheet_names(zip_file)
            sheets = sorted(
                [
                    name
                    for name in zip_file.namelist()
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                ],
                key=lambda item: int(re.search(r"sheet(\d+)\.xml$", item).group(1)),
            )
            for sheet_index, sheet_path in enumerate(sheets):
                sheet_name = names[sheet_index] if sheet_index < len(names) else Path(sheet_path).stem
                rows = read_sheet_rows(zip_file, sheet_path, strings)
                summary = parse_summary(rows)
                particles = parse_particles(rows)
                base = {
                    "source_kind": kind,
                    "workbook_path": relative_path,
                    "sheet_name": sheet_name,
                    "batch_id": summary.get("batch_id", ""),
                }
                summary_record = {**base, **summary, "particle_rows": len(particles)}
                summary_rows.append(summary_record)
                for particle in particles:
                    particle_rows.append({**base, **particle})
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        summary_rows.append(
            {
                "source_kind": kind,
                "workbook_path": relative_path,
                "sheet_name": "",
                "batch_id": "",
                "particle_rows": 0,
                "error": str(exc),
            }
        )
    return summary_rows, particle_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract data_98986 roundness Excel results into CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_98986") / "data",
        help="Directory containing result-return and roundness summary folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "data_98986_roundness",
        help="Directory for generated CSV files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_rows: List[Dict[str, object]] = []
    particle_rows: List[Dict[str, object]] = []

    for workbook in iter_workbooks(input_dir):
        summaries, particles = parse_workbook(workbook, input_dir)
        summary_rows.extend(summaries)
        particle_rows.extend(particles)

    write_csv(output_dir / "data_98986_image_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "data_98986_particle_roundness.csv", particle_rows, PARTICLE_COLUMNS)
    print(f"workbooks: {len(list(iter_workbooks(input_dir)))}")
    print(f"summary rows: {len(summary_rows)}")
    print(f"particle rows: {len(particle_rows)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
