from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .preprocess import imread_unicode


def build_mask_rcnn_model(
    num_classes: int = 2,
    pretrained: bool = True,
    detections_per_img: int = 500,
    rpn_batch_size_per_image: int = 256,
    rpn_positive_fraction: float = 0.5,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_train: int = 2000,
):
    """构建 Mask R-CNN，num_classes=背景+颗粒。"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model_kwargs = {
        "rpn_batch_size_per_image": int(rpn_batch_size_per_image),
        "rpn_positive_fraction": float(rpn_positive_fraction),
        "box_batch_size_per_image": int(box_batch_size_per_image),
        "box_positive_fraction": float(box_positive_fraction),
        "rpn_pre_nms_top_n_train": int(rpn_pre_nms_top_n_train),
        "rpn_post_nms_top_n_train": int(rpn_post_nms_top_n_train),
    }
    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            **model_kwargs,
        )
    except (ImportError, TypeError):
        model = maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained,
            **model_kwargs,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    try:
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer=256,
            num_classes=num_classes,
        )
    except TypeError:
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes,
        )
    model.roi_heads.detections_per_img = int(detections_per_img)
    return model


class CocoParticleDataset:
    """COCO 实例分割数据集：类别建议统一标注为 particle。"""

    def __init__(
        self,
        image_dir: str | Path,
        ann_file: str | Path,
        augmenter: Optional["DetectionAugmenter"] = None,
        min_mask_area: float = 16.0,
        min_box_size: float = 3.0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.augmenter = augmenter
        self.min_mask_area = float(min_mask_area)
        self.min_box_size = float(min_box_size)
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
        self.source_names = [
            str(image.get("source_dataset", "default")) for image in self.images
        ]
        self.annotations_by_image: Dict[int, List[Dict]] = {
            int(image["id"]): [] for image in self.images
        }
        for ann in coco.get("annotations", []):
            self.annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch

        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_dir / image_info["file_name"]
        image_bgr = imread_unicode(image_path)
        image_rgb = image_bgr[:, :, ::-1].copy()
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        anns = self.annotations_by_image.get(image_id, [])
        height = int(image_info.get("height", image.shape[1]))
        width = int(image_info.get("width", image.shape[2]))

        boxes: List[List[float]] = []
        masks: List[np.ndarray] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            mask = coco_ann_to_mask(ann, height=height, width=width)
            mask_area = float(mask.sum())
            if mask_area < self.min_mask_area:
                continue
            x, y, w, h = bbox_from_binary_mask(mask)
            if w < self.min_box_size or h < self.min_box_size:
                continue
            boxes.append([x, y, x + w, y + h])
            masks.append(mask.astype(np.uint8))
            labels.append(1)
            areas.append(mask_area)
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros(
                (0, image.shape[1], image.shape[2]),
                dtype=torch.uint8,
            )
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }
        if self.augmenter is not None:
            image, target = self.augmenter(image, target)
        return image, target


def bbox_from_binary_mask(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return [x1, y1, x2 - x1, y2 - y1]


def coco_ann_to_mask(ann: Dict, height: int, width: int) -> np.ndarray:
    """将 COCO polygon 标注转成二值 mask；Windows 下无需 pycocotools。"""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
        for polygon in polygons:
            if len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(segmentation)
            return (decoded > 0).astype(np.uint8)
        except Exception as exc:
            raise RuntimeError(
                "当前 COCO 标注使用 RLE mask，但环境中不能解析 RLE。"
                "请从 CVAT/LabelMe 导出 polygon 格式，或安装 pycocotools。"
            ) from exc

    if "bbox" in ann:
        x, y, w, h = [int(round(v)) for v in ann["bbox"]]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + max(0, w))
        y2 = min(height, y + max(0, h))
        mask[y1:y2, x1:x2] = 1
    return mask


def collate_fn(batch):
    return tuple(zip(*batch))


def _boxes_from_torch_masks(masks):
    import torch

    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    boxes = []
    for mask in masks:
        positions = torch.nonzero(mask > 0, as_tuple=False)
        if positions.numel() == 0:
            boxes.append(torch.zeros(4, dtype=torch.float32, device=masks.device))
            continue
        y1, x1 = positions.min(dim=0).values
        y2, x2 = positions.max(dim=0).values + 1
        boxes.append(torch.stack([x1, y1, x2, y2]).to(torch.float32))
    return torch.stack(boxes)


class DetectionAugmenter:
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        intensity_prob: float = 0.8,
        brightness: float = 0.12,
        contrast: float = 0.15,
        noise_std: float = 0.01,
        rotate90_prob: float = 0.5,
        random_crop_prob: float = 0.0,
        random_crop_size: int = 640,
        crop_min_retained_fraction: float = 0.25,
        min_mask_area: float = 16.0,
        min_box_size: float = 3.0,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.intensity_prob = intensity_prob
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.rotate90_prob = rotate90_prob
        self.random_crop_prob = random_crop_prob
        self.random_crop_size = int(random_crop_size)
        self.crop_min_retained_fraction = float(crop_min_retained_fraction)
        self.min_mask_area = float(min_mask_area)
        self.min_box_size = float(min_box_size)

    def _random_crop(self, image, target):
        import torch

        _, height, width = image.shape
        crop_h = min(self.random_crop_size, height)
        crop_w = min(self.random_crop_size, width)
        if crop_h >= height and crop_w >= width:
            return image, target

        masks = target["masks"]
        attempts = 10 if masks.numel() else 1
        original_areas = (
            masks.flatten(1).sum(dim=1).to(torch.float32)
            if masks.numel()
            else torch.zeros((0,), dtype=torch.float32, device=masks.device)
        )
        for _ in range(attempts):
            max_y = height - crop_h
            max_x = width - crop_w
            y1 = int(torch.randint(max_y + 1, ()).item()) if max_y > 0 else 0
            x1 = int(torch.randint(max_x + 1, ()).item()) if max_x > 0 else 0
            cropped_masks = masks[:, y1 : y1 + crop_h, x1 : x1 + crop_w]
            if not masks.numel():
                target["masks"] = cropped_masks
                return image[:, y1 : y1 + crop_h, x1 : x1 + crop_w], target

            cropped_areas = cropped_masks.flatten(1).sum(dim=1).to(torch.float32)
            boxes = _boxes_from_torch_masks(cropped_masks)
            box_widths = boxes[:, 2] - boxes[:, 0]
            box_heights = boxes[:, 3] - boxes[:, 1]
            retained = cropped_areas / torch.clamp(original_areas, min=1.0)
            keep = (
                (cropped_areas >= self.min_mask_area)
                & (box_widths >= self.min_box_size)
                & (box_heights >= self.min_box_size)
                & (retained >= self.crop_min_retained_fraction)
            )
            if not torch.any(keep):
                continue

            target["masks"] = cropped_masks[keep]
            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]
            target["area"] = cropped_areas[keep]
            target["iscrowd"] = target["iscrowd"][keep]
            return image[:, y1 : y1 + crop_h, x1 : x1 + crop_w], target

        return image, target

    def __call__(self, image, target):
        import torch

        if (
            self.random_crop_prob > 0
            and self.random_crop_size > 0
            and torch.rand(()) < self.random_crop_prob
        ):
            image, target = self._random_crop(image, target)

        _, height, width = image.shape
        if self.hflip_prob > 0 and torch.rand(()) < self.hflip_prob:
            image = torch.flip(image, dims=[2])
            if target["masks"].numel():
                target["masks"] = torch.flip(target["masks"], dims=[2])
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
                target["boxes"] = boxes

        if self.vflip_prob > 0 and torch.rand(()) < self.vflip_prob:
            image = torch.flip(image, dims=[1])
            if target["masks"].numel():
                target["masks"] = torch.flip(target["masks"], dims=[1])
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                y1 = boxes[:, 1].clone()
                y2 = boxes[:, 3].clone()
                boxes[:, 1] = height - y2
                boxes[:, 3] = height - y1
                target["boxes"] = boxes

        if self.rotate90_prob > 0 and torch.rand(()) < self.rotate90_prob:
            turns = int(torch.randint(1, 4, ()).item())
            image = torch.rot90(image, k=turns, dims=[1, 2])
            if target["masks"].numel():
                target["masks"] = torch.rot90(
                    target["masks"],
                    k=turns,
                    dims=[1, 2],
                )
                target["boxes"] = _boxes_from_torch_masks(target["masks"])

        if self.intensity_prob > 0 and torch.rand(()) < self.intensity_prob:
            contrast = 1.0 + torch.empty((), device=image.device).uniform_(
                -self.contrast,
                self.contrast,
            )
            brightness = torch.empty((), device=image.device).uniform_(
                -self.brightness,
                self.brightness,
            )
            image = torch.clamp(image * contrast + brightness, 0.0, 1.0)

        if self.noise_std > 0:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)

        return image, target


def _move_targets_to_device(targets: Sequence[Dict], device: torch.device):
    return [{k: v.to(device) for k, v in target.items()} for target in targets]


def compute_detection_loss(model, loader, device: torch.device) -> float:
    import torch

    was_training = model.training
    model.train()
    losses = []
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = _move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())
            losses.append(float(loss.item()))
    model.train(was_training)
    return float(np.mean(losses)) if losses else 0.0


def _torch_load_checkpoint(checkpoint_path: Path, device):
    import torch

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_model_state_dict(checkpoint) -> Dict:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return _strip_module_prefix(state_dict)
        return _strip_module_prefix(checkpoint)
    raise TypeError("Unsupported checkpoint format. Expected a dict-like state dict.")


def _strip_module_prefix(state_dict: Dict) -> Dict:
    keys = list(state_dict.keys())
    if keys and all(isinstance(key, str) and key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _checkpoint_epoch(checkpoint) -> int:
    if isinstance(checkpoint, dict):
        try:
            return int(checkpoint.get("epoch", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _checkpoint_history(checkpoint) -> List[Dict]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("history"), list):
        return list(checkpoint["history"])
    return []


def _parse_source_weights(text: Optional[str]) -> Dict[str, float]:
    if not text:
        return {}
    weights: Dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "--source-weights must use source=value pairs separated by commas"
            )
        source, value_text = part.split("=", 1)
        source = source.strip()
        value = float(value_text)
        if not source or value <= 0:
            raise ValueError(f"Invalid source weight: {part!r}")
        weights[source] = value
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--source-weights must contain at least one positive weight")
    return {source: value / total for source, value in weights.items()}


def _checkpoint_best_val_loss(checkpoint, history: Sequence[Dict]) -> float:
    if isinstance(checkpoint, dict):
        try:
            value = float(checkpoint.get("best_val_loss", float("inf")))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    candidates = []
    for row in history:
        value = row.get("val_loss")
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                candidates.append(numeric)
    return min(candidates) if candidates else float("inf")


def train_maskrcnn(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not 0.0 <= args.random_crop_prob <= 1.0:
        raise ValueError("--random-crop-prob must be between 0 and 1")
    if args.random_crop_size <= 0:
        raise ValueError("--random-crop-size must be positive")
    if not 0.0 <= args.crop_min_retained_fraction <= 1.0:
        raise ValueError("--crop-min-retained-fraction must be between 0 and 1")
    if args.rpn_batch_size_per_image <= 0:
        raise ValueError("--rpn-batch-size-per-image must be positive")
    if args.box_batch_size_per_image <= 0:
        raise ValueError("--box-batch-size-per-image must be positive")
    if not 0.0 < args.rpn_positive_fraction <= 1.0:
        raise ValueError("--rpn-positive-fraction must be in (0, 1]")
    if not 0.0 < args.box_positive_fraction <= 1.0:
        raise ValueError("--box-positive-fraction must be in (0, 1]")
    if args.rpn_pre_nms_top_n_train <= 0 or args.rpn_post_nms_top_n_train <= 0:
        raise ValueError("RPN train proposal limits must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    augmenter = (
        DetectionAugmenter(
            hflip_prob=args.hflip_prob,
            vflip_prob=args.vflip_prob,
            intensity_prob=args.intensity_prob,
            brightness=args.brightness,
            contrast=args.contrast,
            noise_std=args.noise_std,
            rotate90_prob=args.rotate90_prob,
            random_crop_prob=args.random_crop_prob,
            random_crop_size=args.random_crop_size,
            crop_min_retained_fraction=args.crop_min_retained_fraction,
            min_mask_area=args.min_mask_area,
            min_box_size=args.min_box_size,
        )
        if args.augment
        else None
    )

    train_ds = CocoParticleDataset(
        args.train_images,
        args.train_ann,
        augmenter=augmenter,
        min_mask_area=args.min_mask_area,
        min_box_size=args.min_box_size,
    )
    val_ds = (
        CocoParticleDataset(
            args.val_images,
            args.val_ann,
            min_mask_area=args.min_mask_area,
            min_box_size=args.min_box_size,
        )
        if args.val_ann
        else None
    )

    source_weights = _parse_source_weights(args.source_weights)
    source_counts = Counter(train_ds.source_names)
    train_sampler = None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    if source_weights:
        missing = sorted(set(source_counts) - set(source_weights))
        unknown = sorted(set(source_weights) - set(source_counts))
        if missing or unknown:
            raise ValueError(
                "Source-weight mismatch: "
                f"missing={missing}, unknown={unknown}, available={sorted(source_counts)}"
            )
        sample_weights = [
            source_weights[source] / source_counts[source]
            for source in train_ds.source_names
        ]
        samples_per_epoch = args.samples_per_epoch or len(train_ds)
        try:
            train_sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=samples_per_epoch,
                replacement=True,
                generator=loader_generator,
            )
        except TypeError:
            train_sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=samples_per_epoch,
                replacement=True,
            )
        print(
            json.dumps(
                {
                    "source_counts": dict(sorted(source_counts.items())),
                    "source_target_probabilities": source_weights,
                    "samples_per_epoch": samples_per_epoch,
                },
                ensure_ascii=False,
            )
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        if val_ds is not None
        else None
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resume_checkpoint = None
    resume_epoch = 0

    model = build_mask_rcnn_model(
        num_classes=args.num_classes,
        pretrained=not bool(args.resume_from),
        detections_per_img=args.detections_per_img,
        rpn_batch_size_per_image=args.rpn_batch_size_per_image,
        rpn_positive_fraction=args.rpn_positive_fraction,
        box_batch_size_per_image=args.box_batch_size_per_image,
        box_positive_fraction=args.box_positive_fraction,
        rpn_pre_nms_top_n_train=args.rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
    )
    model.to(device)
    model_sampling = {
        "rpn_batch_size_per_image": args.rpn_batch_size_per_image,
        "rpn_positive_fraction": args.rpn_positive_fraction,
        "rpn_max_positive_per_image": int(
            args.rpn_batch_size_per_image * args.rpn_positive_fraction
        ),
        "box_batch_size_per_image": args.box_batch_size_per_image,
        "box_positive_fraction": args.box_positive_fraction,
        "box_max_positive_per_image": int(
            args.box_batch_size_per_image * args.box_positive_fraction
        ),
        "rpn_pre_nms_top_n_train": args.rpn_pre_nms_top_n_train,
        "rpn_post_nms_top_n_train": args.rpn_post_nms_top_n_train,
    }
    print(json.dumps({"model_sampling": model_sampling}, ensure_ascii=False))

    if args.resume_from:
        resume_path = Path(args.resume_from)
        resume_checkpoint = _torch_load_checkpoint(resume_path, device)
        model.load_state_dict(_extract_model_state_dict(resume_checkpoint))
        resume_epoch = _checkpoint_epoch(resume_checkpoint)
        print(
            json.dumps(
                {
                    "resume_from": str(resume_path),
                    "resume_epoch": resume_epoch,
                    "mode": "model_weights_loaded",
                },
                ensure_ascii=False,
            )
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    if args.resume_optimizer and isinstance(resume_checkpoint, dict):
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        else:
            print(json.dumps({"warning": "resume checkpoint has no optimizer state"}))
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        else:
            print(json.dumps({"warning": "resume checkpoint has no scheduler state"}))

    history: List[Dict] = _checkpoint_history(resume_checkpoint)
    best_val_loss = _checkpoint_best_val_loss(resume_checkpoint, history)
    if isinstance(resume_checkpoint, dict) and args.val_ann:
        previous_val_ann = resume_checkpoint.get("val_ann")
        if previous_val_ann and str(previous_val_ann) != str(args.val_ann):
            best_val_loss = float("inf")
    start_epoch = resume_epoch + 1
    end_epoch = resume_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_losses = []
        skipped_nonfinite_batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = _move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            if not torch.isfinite(losses):
                skipped_nonfinite_batches += 1
                loss_values = {
                    key: float(value.detach().cpu().item())
                    for key, value in loss_dict.items()
                }
                image_ids = [
                    int(target["image_id"].detach().cpu().flatten()[0])
                    for target in targets
                ]
                print(
                    json.dumps(
                        {
                            "warning": "non_finite_loss_skip_batch",
                            "epoch": epoch,
                            "image_ids": image_ids,
                            "losses": loss_values,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            losses.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params,
                    max_norm=args.clip_grad_norm,
                )
                grad_norm_value = (
                    float(grad_norm.detach().cpu().item())
                    if hasattr(grad_norm, "detach")
                    else float(grad_norm)
                )
                if not np.isfinite(grad_norm_value):
                    skipped_nonfinite_batches += 1
                    image_ids = [
                        int(target["image_id"].detach().cpu().flatten()[0])
                        for target in targets
                    ]
                    print(
                        json.dumps(
                            {
                                "warning": "non_finite_gradient_skip_batch",
                                "epoch": epoch,
                                "image_ids": image_ids,
                                "grad_norm": grad_norm_value,
                            },
                            ensure_ascii=False,
                        )
                    )
                    optimizer.zero_grad()
                    continue
            optimizer.step()
            epoch_losses.append(float(losses.item()))

        scheduler.step()
        if not epoch_losses:
            raise RuntimeError(
                f"All batches produced non-finite loss at epoch {epoch}. "
                "Try lowering --lr or increasing --min-mask-area/--min-box-size."
            )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = compute_detection_loss(model, val_loader, device) if val_loader else None
        is_best = bool(
            val_loss is not None
            and np.isfinite(val_loss)
            and float(val_loss) < best_val_loss
        )
        if is_best:
            best_val_loss = float(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "skipped_nonfinite_batches": skipped_nonfinite_batches,
            "is_best": is_best,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "num_classes": args.num_classes,
            "train_images": str(args.train_images),
            "train_ann": str(args.train_ann),
            "val_images": str(args.val_images) if args.val_images else None,
            "val_ann": str(args.val_ann) if args.val_ann else None,
            "source_weights": source_weights,
            "model_sampling": model_sampling,
            "augmentation": {
                "enabled": bool(args.augment),
                "random_crop_prob": args.random_crop_prob,
                "random_crop_size": args.random_crop_size,
                "crop_min_retained_fraction": args.crop_min_retained_fraction,
            },
            "seed": args.seed,
        }
        torch.save(checkpoint, output_dir / "maskrcnn_particle_last.pth")
        if is_best:
            torch.save(checkpoint, output_dir / "maskrcnn_particle_best_val.pth")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                checkpoint,
                output_dir / f"maskrcnn_particle_epoch_{epoch:03d}.pth",
            )

    with open(output_dir / "loss_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "skipped_nonfinite_batches",
                "is_best",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        val_losses = [row["val_loss"] for row in history]
        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="train_loss")
        if any(value is not None for value in val_losses):
            ax.plot(epochs, val_losses, label="val_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass


DEFAULT_FEATURE_COLUMNS = [
    "area_um2",
    "perimeter_um",
    "q_value",
    "major_axis_um",
    "minor_axis_um",
    "axis_ratio",
    "hole_ratio",
    "equivalent_diameter_um",
]


def train_xgboost_classifier(args: argparse.Namespace) -> None:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from xgboost import XGBClassifier

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_csv)
    feature_cols = args.feature_cols.split(",") if args.feature_cols else DEFAULT_FEATURE_COLUMNS
    label_cols = args.label_cols.split(",")

    X = df[feature_cols].astype(float)
    y = df[label_cols].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y[label_cols[0]] if len(y[label_cols[0]].unique()) > 1 else None,
    )

    base_model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
    )
    if len(label_cols) > 1:
        model = MultiOutputClassifier(base_model)
    else:
        model = base_model

    model.fit(X_train, y_train if len(label_cols) > 1 else y_train[label_cols[0]])
    pred = model.predict(X_val)
    y_true = y_val.values if len(label_cols) > 1 else y_val[label_cols[0]].values
    acc = float(accuracy_score(y_true, pred))
    report = classification_report(y_true, pred, zero_division=0)

    with open(output_dir / "xgb_geometry_classifier.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "label_cols": label_cols}, f)

    metrics = {"accuracy": acc, "classification_report": report}
    (output_dir / "xgb_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="金属粉末 SEM 分割/分类训练")
    subparsers = parser.add_subparsers(dest="task", required=True)

    seg = subparsers.add_parser("segment", help="训练 Mask R-CNN 实例分割模型")
    seg.add_argument("--train-images", required=True)
    seg.add_argument("--train-ann", required=True)
    seg.add_argument("--val-images", default=None)
    seg.add_argument("--val-ann", default=None)
    seg.add_argument("--output-dir", default="runs/train_maskrcnn")
    seg.add_argument("--epochs", type=int, default=20)
    seg.add_argument("--batch-size", type=int, default=2)
    seg.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size. Keep at 1 for dense SEM masks.",
    )
    seg.add_argument("--num-workers", type=int, default=0)
    seg.add_argument("--seed", type=int, default=42)
    seg.add_argument("--lr", type=float, default=0.001)
    seg.add_argument("--lr-step-size", type=int, default=10)
    seg.add_argument("--lr-gamma", type=float, default=0.5)
    seg.add_argument("--weight-decay", type=float, default=0.0005)
    seg.add_argument("--num-classes", type=int, default=2)
    seg.add_argument(
        "--detections-per-img",
        type=int,
        default=1000,
        help="Maximum detections retained per image during inference.",
    )
    seg.add_argument(
        "--rpn-batch-size-per-image",
        type=int,
        default=256,
        help="Number of RPN anchors sampled per training image.",
    )
    seg.add_argument(
        "--rpn-positive-fraction",
        type=float,
        default=0.5,
        help="Maximum positive fraction in the sampled RPN anchors.",
    )
    seg.add_argument(
        "--box-batch-size-per-image",
        type=int,
        default=512,
        help="Number of ROI proposals sampled per training image.",
    )
    seg.add_argument(
        "--box-positive-fraction",
        type=float,
        default=0.25,
        help="Maximum positive fraction in the sampled ROI proposals.",
    )
    seg.add_argument(
        "--rpn-pre-nms-top-n-train",
        type=int,
        default=2000,
        help="RPN proposals retained before NMS during training.",
    )
    seg.add_argument(
        "--rpn-post-nms-top-n-train",
        type=int,
        default=2000,
        help="RPN proposals retained after NMS during training.",
    )
    seg.add_argument("--device", default=None)
    seg.add_argument(
        "--source-weights",
        default=None,
        help=(
            "Target sampling probabilities, for example "
            "selected_60_moved=0.40,GH3536_CA_H6=0.30,"
            "GH3536_SA_H1=0.20,HX_ZA=0.10"
        ),
    )
    seg.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Number of weighted samples drawn per epoch; defaults to dataset size.",
    )
    seg.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also save a numbered checkpoint every N epochs; 0 disables it.",
    )
    seg.add_argument(
        "--min-mask-area",
        type=float,
        default=16.0,
        help="Skip training masks smaller than this many pixels.",
    )
    seg.add_argument(
        "--min-box-size",
        type=float,
        default=3.0,
        help="Skip training boxes whose width or height is smaller than this many pixels.",
    )
    seg.add_argument(
        "--clip-grad-norm",
        type=float,
        default=5.0,
        help="Clip gradient norm during Mask R-CNN training. Use 0 to disable.",
    )
    seg.add_argument(
        "--resume-from",
        default=None,
        help="Path to a previous maskrcnn_particle_last.pth checkpoint for fine-tuning.",
    )
    seg.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore optimizer and scheduler states from --resume-from.",
    )
    seg.add_argument("--augment", action="store_true")
    seg.add_argument("--hflip-prob", type=float, default=0.5)
    seg.add_argument("--vflip-prob", type=float, default=0.5)
    seg.add_argument("--intensity-prob", type=float, default=0.8)
    seg.add_argument("--brightness", type=float, default=0.12)
    seg.add_argument("--contrast", type=float, default=0.15)
    seg.add_argument("--noise-std", type=float, default=0.01)
    seg.add_argument("--rotate90-prob", type=float, default=0.5)
    seg.add_argument(
        "--random-crop-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of training on a random image crop. Use 0.5-0.75 for "
            "dense small-particle SEM images while retaining some full images."
        ),
    )
    seg.add_argument(
        "--random-crop-size",
        type=int,
        default=640,
        help="Square crop size in source-image pixels.",
    )
    seg.add_argument(
        "--crop-min-retained-fraction",
        type=float,
        default=0.25,
        help="Minimum fraction of an instance mask retained by a random crop.",
    )

    clf = subparsers.add_parser("classifier", help="训练 XGBoost 几何特征分类器")
    clf.add_argument("--feature-csv", required=True)
    clf.add_argument("--output-dir", default="runs/train_xgb")
    clf.add_argument(
        "--label-cols",
        default="is_spherical,is_hollow,is_agglomerate",
        help="标签列，逗号分隔",
    )
    clf.add_argument("--feature-cols", default=None, help="特征列，逗号分隔")
    clf.add_argument("--val-ratio", type=float, default=0.2)
    clf.add_argument("--n-estimators", type=int, default=300)
    clf.add_argument("--max-depth", type=int, default=4)
    clf.add_argument("--learning-rate", type=float, default=0.05)
    clf.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "segment":
        train_maskrcnn(args)
    elif args.task == "classifier":
        train_xgboost_classifier(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
