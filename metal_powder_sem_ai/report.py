from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence


PARTICLE_COLUMNS = [
    "particle_id",
    "area_um2",
    "perimeter_um",
    "q_value",
    "major_axis_um",
    "minor_axis_um",
    "axis_ratio",
    "hole_area_um2",
    "hole_ratio",
    "equivalent_diameter_um",
    "is_spherical",
    "is_hollow",
    "is_agglomerate",
    "agglomerate_group_id",
    "score",
    "centroid_x",
    "centroid_y",
]

RENAME_MAP = {
    "particle_id": "颗粒ID",
    "area_um2": "面积A(um^2)",
    "perimeter_um": "周长P(um)",
    "q_value": "球形度Q",
    "major_axis_um": "长轴(um)",
    "minor_axis_um": "短轴(um)",
    "axis_ratio": "轴比Lmajor/Lminor",
    "hole_area_um2": "孔洞面积(um^2)",
    "hole_ratio": "孔洞占比",
    "equivalent_diameter_um": "等效直径(um)",
    "is_spherical": "是否球形颗粒",
    "is_hollow": "是否空心粉",
    "is_agglomerate": "是否团聚体",
    "agglomerate_group_id": "团聚体Group ID",
    "score": "分割置信度",
    "centroid_x": "中心X(px)",
    "centroid_y": "中心Y(px)",
}


def _particle_rows(features: Sequence[Dict]):
    keep_columns = [col for col in PARTICLE_COLUMNS if any(col in item for item in features)]
    headers = [RENAME_MAP.get(col, col) for col in keep_columns]
    rows = [[item.get(col, "") for col in keep_columns] for item in features]
    return headers, rows


def export_excel_report(
    output_path: str | Path,
    features: Sequence[Dict],
    stats: Dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers, rows = _particle_rows(features)
    stats_headers = [
        "总颗粒数N",
        "平均球形度Q",
        "空心粉数n_hollow",
        "空心粉率K",
        "团聚体数N_agglom_group",
        "团聚颗粒数N_agglom_particle",
        "团聚率P_agglom",
        "团聚面积率P_area",
        "球形颗粒数",
        "球形颗粒率S",
        "团聚体group",
        "团聚体pair",
    ]
    stats_row = [
        stats["total_particles"],
        stats["mean_sphericity_q_text"],
        stats["hollow_particles"],
        stats["hollow_rate_text"],
        stats.get("agglomerate_group_count", 0),
        stats["agglomerate_particles"],
        stats["agglomerate_rate_text"],
        stats["agglomerate_area_rate_text"],
        stats["spherical_particles"],
        stats["sphericity_rate_s_text"],
        str(stats.get("agglomerate_groups", [])),
        str(stats.get("agglomerate_pairs", [])),
    ]

    try:
        from openpyxl import Workbook

        workbook = Workbook()
        stats_sheet = workbook.active
        stats_sheet.title = "统计结果"
        stats_sheet.append(stats_headers)
        stats_sheet.append(stats_row)

        particle_sheet = workbook.create_sheet("颗粒级特征")
        particle_sheet.append(headers)
        for row in rows:
            particle_sheet.append(row)

        workbook.save(output_path)
    except Exception:
        # Excel writer unavailable时，至少保证CSV报告可用。
        output_path.with_suffix(".stats.csv").write_text(
            ",".join(stats_headers) + "\n" + ",".join(str(v) for v in stats_row) + "\n",
            encoding="utf-8-sig",
        )

    csv_path = output_path.with_suffix(".particles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
