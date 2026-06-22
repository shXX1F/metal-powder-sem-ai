from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .preprocess import imread_unicode


def build_mask_rcnn_model(
    num_classes: int = 2,
    pretrained: bool = True,
    detections_per_img: int = 500,
):
    """构建 Mask R-CNN，num_classes=背景+颗粒。"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = maskrcnn_resnet50_fpn(pretrained=pretrained)

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


class DetectionAugmenter:
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        intensity_prob: float = 0.8,
        brightness: float = 0.12,
        contrast: float = 0.15,
        noise_std: float = 0.01,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.intensity_prob = intensity_prob
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image, target):
        import torch

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


def train_maskrcnn(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmenter = (
        DetectionAugmenter(
            hflip_prob=args.hflip_prob,
            vflip_prob=args.vflip_prob,
            intensity_prob=args.intensity_prob,
            brightness=args.brightness,
            contrast=args.contrast,
            noise_std=args.noise_std,
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
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
    )
    model.to(device)

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
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "skipped_nonfinite_batches": skipped_nonfinite_batches,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "num_classes": args.num_classes,
                "train_images": str(args.train_images),
                "train_ann": str(args.train_ann),
                "val_images": str(args.val_images) if args.val_images else None,
                "val_ann": str(args.val_ann) if args.val_ann else None,
            },
            output_dir / "maskrcnn_particle_last.pth",
        )

    with open(output_dir / "loss_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "skipped_nonfinite_batches",
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
    seg.add_argument("--num-workers", type=int, default=0)
    seg.add_argument("--lr", type=float, default=0.001)
    seg.add_argument("--lr-step-size", type=int, default=10)
    seg.add_argument("--lr-gamma", type=float, default=0.5)
    seg.add_argument("--weight-decay", type=float, default=0.0005)
    seg.add_argument("--num-classes", type=int, default=2)
    seg.add_argument("--device", default=None)
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
