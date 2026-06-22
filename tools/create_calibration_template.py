from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 SEM 图像目录，生成比例尺标定表模板。"
    )
    parser.add_argument("--image-root", default="图像SEM数据", help="SEM 图像根目录")
    parser.add_argument(
        "--output",
        default="data/sem_image_calibration.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--crop-bottom-fraction",
        type=float,
        default=0.12,
        help="GUI/推理时裁掉底部 SEM 信息栏的比例",
    )
    return parser.parse_args()


def image_size(path: Path) -> tuple[int | str, int | str]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return "", ""


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        relative = path.relative_to(image_root)
        parts = relative.parts
        width, height = image_size(path)
        rows.append(
            {
                "image_path": str(path).replace("\\", "/"),
                "material": parts[0] if len(parts) > 0 else "",
                "batch": parts[1] if len(parts) > 1 else "",
                "file_name": path.name,
                "width_px": width,
                "height_px": height,
                "scale_um": "",
                "scale_bar_px": "",
                "pixel_size_um": "",
                "crop_bottom_fraction": args.crop_bottom_fraction,
                "split": "",
                "note": "",
            }
        )

    fieldnames = [
        "image_path",
        "material",
        "batch",
        "file_name",
        "width_px",
        "height_px",
        "scale_um",
        "scale_bar_px",
        "pixel_size_um",
        "crop_bottom_fraction",
        "split",
        "note",
    ]
    with open(output, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"写入 {len(rows)} 张图像的标定模板: {output}")
    print("下一步：打开 CSV，填写 scale_um 和 scale_bar_px，并计算 pixel_size_um = scale_um / scale_bar_px。")


if __name__ == "__main__":
    main()
