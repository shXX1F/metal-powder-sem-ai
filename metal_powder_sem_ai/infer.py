from __future__ import annotations

import argparse
from pathlib import Path

from .classify_stat import classify_particles, format_stats
from .feature_extract import extract_all_features
from .preprocess import (
    imread_unicode,
    preprocess_sem_image,
    save_preprocess_result,
)
from .report import export_excel_report
from .segment import ClassicalParticleSegmenter, MaskRCNNSegmenter
from .visualize import draw_instances, save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="金属粉末 SEM 图像 AI 识别推理")
    parser.add_argument("--image", required=True, help="输入 SEM 图像路径")
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        required=True,
        help="比例尺标定：1 pixel = X um，例如 0.1",
    )
    parser.add_argument("--output-dir", default="runs/sem_infer", help="输出目录")
    parser.add_argument(
        "--segmenter",
        choices=["classical", "maskrcnn"],
        default="classical",
        help="分割器：classical 用于无权重快速跑通，maskrcnn 用于训练后推理",
    )
    parser.add_argument("--weights", default=None, help="Mask R-CNN 权重路径")
    parser.add_argument(
        "--crop-bottom-fraction",
        type=float,
        default=0.0,
        help="裁掉底部 SEM 信息栏比例，例如 0.12；默认不裁剪",
    )
    parser.add_argument("--min-area-px", type=int, default=80)
    parser.add_argument("--peak-min-distance", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre = preprocess_sem_image(
        args.image,
        crop_bottom_fraction=args.crop_bottom_fraction,
    )
    save_preprocess_result(pre, output_dir / "preprocess")

    if args.segmenter == "classical":
        segmenter = ClassicalParticleSegmenter(
            min_area_px=args.min_area_px,
            peak_min_distance=args.peak_min_distance,
        )
        instances = segmenter.segment(pre.binary)
    else:
        if not args.weights:
            raise ValueError("segmenter=maskrcnn 时必须提供 --weights")
        segmenter = MaskRCNNSegmenter(
            weights_path=args.weights,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
        )
        instances = segmenter.segment(pre.image)

    features = extract_all_features(
        instances,
        pixel_size_um=args.pixel_size_um,
        gray=pre.enhanced,
    )
    masks = [instance.mask for instance in instances]
    classified, stats = classify_particles(features, masks=masks)

    visualized = draw_instances(pre.image, instances, classified)
    save_visualization(output_dir / "visualized.png", visualized)
    export_excel_report(output_dir / "report.xlsx", classified, stats)

    print(format_stats(stats))
    print(f"可视化输出: {output_dir / 'visualized.png'}")
    print(f"Excel 报告: {output_dir / 'report.xlsx'}")


if __name__ == "__main__":
    main()
