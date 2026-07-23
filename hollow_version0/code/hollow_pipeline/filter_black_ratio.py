#!/usr/bin/env python3
"""Filter hollow candidates by internal dark-area ratio and redraw previews.

The Mask R-CNN hollow model is intentionally high-recall. This post-process keeps
a candidate only when its predicted particle region contains enough dark interior
area. The dark area uses a dual-threshold rule: strict black cores prove a pore,
and connected looser dark regions expand the pore boundary.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RESULT_DIR = None
PRED_DIR = None
GT_DIR = None
OUT_DIR = None
LABEL = 'hollow'

PARAMS = {
    'white_threshold': 150,
    'white_max_channel_diff': 50.0,
    'boundary_strict_px': 3,
    'boundary_black_threshold': 100.0,
    'std_k': 1.0,
    'abs_black_max': 120,
    'morph_kernel': 3,
    'min_component_area_px': 0,
    'min_component_area_ratio': 0.0,
    'black_ratio_thresholds': [0.05, 0.10, 0.15, 0.20, 0.25],
    'selected_threshold': 0.10,
    'ratio_mode': 'black_over_black_plus_white',
    'iou_threshold': 0.5,
    'edge_ring_px': 5,
    'edge_white_ratio_threshold': 0.60,
}


HOLLOW_COLOR = (255, 0, 255)  # BGR magenta
BLACK_COLOR = (0, 0, 255)     # red overlay for internal black area
GT_COLOR = (0, 220, 0)        # green
TEXT_COLOR = (255, 255, 255)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.png', image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def shape_points(shape):
    pts = shape.get('points') or []
    return np.array(pts, dtype=np.float32)


def polygon_mask(shape, h, w):
    pts = shape_points(shape)
    mask = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def mask_to_contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compute_white_mask(image_bgr, threshold):
    channel_min = image_bgr.min(axis=2)
    channel_max = image_bgr.max(axis=2)
    return (channel_min >= float(threshold)) & ((channel_max - channel_min) <= float(PARAMS['white_max_channel_diff']))


def compute_boundary_band(obj_mask, px):
    obj_mask = obj_mask.astype(np.uint8)
    px = int(px)
    if px <= 0 or int(obj_mask.sum()) <= 0:
        return np.zeros_like(obj_mask, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(obj_mask, kernel, iterations=px)
    return (obj_mask > 0) & (inner == 0)


def compute_edge_white_ratio(image_bgr, obj_mask, ring_px=None):
    """Compute white-pixel ratio in the inner edge ring of one hollow mask.

    A valid hollow powder should still have a visible white shell near its outer
    contour. Whole dark fragments and hallucinated dark regions usually fail
    this constraint because their mask edge is not white enough.
    """
    if ring_px is None:
        ring_px = PARAMS.get('edge_ring_px', 5)
    ring = compute_boundary_band(obj_mask, int(ring_px))
    ring_area = int(ring.sum())
    if ring_area <= 0:
        return 0.0, {'edge_ring_px': int(ring_px), 'edge_ring_area': 0, 'edge_white_area': 0, 'edge_white_ratio': 0.0}
    white_pixels = compute_white_mask(image_bgr, float(PARAMS['white_threshold']))
    white_area = int((white_pixels & ring).sum())
    ratio = white_area / ring_area
    return ratio, {
        'edge_ring_px': int(ring_px),
        'edge_ring_area': ring_area,
        'edge_white_area': white_area,
        'edge_white_ratio': ratio,
    }


def remove_dark_boundary_band(image_bgr, obj_mask):
    """Return the original mask; boundary cleanup is intentionally disabled."""
    obj_mask = obj_mask.astype(np.uint8)
    return obj_mask, np.zeros_like(obj_mask)


def compute_black_mask(image_bgr, obj_mask):
    obj_area = int(obj_mask.sum())
    if obj_area <= 0:
        return np.zeros_like(obj_mask), 0.0, {'reason': 'empty_mask'}
    valid_mask, removed_boundary = remove_dark_boundary_band(image_bgr, obj_mask)
    valid_area = int(valid_mask.sum())

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[valid_mask > 0]
    if vals.size == 0:
        return np.zeros_like(obj_mask), 0.0, {'reason': 'empty_inner'}
    mean = float(vals.mean())
    std = float(vals.std())
    white_threshold = float(PARAMS['white_threshold'])
    white_pixels = compute_white_mask(image_bgr, white_threshold)
    boundary_band = compute_boundary_band(valid_mask, PARAMS.get('boundary_strict_px', 0))
    boundary_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    strict_black = boundary_gray < float(PARAMS.get('boundary_black_threshold', 100.0))
    non_white = (~white_pixels) & (valid_mask > 0)
    filtered = (non_white & ((~boundary_band) | strict_black)).astype(np.uint8)
    comp_areas = []

    black_area = int(filtered.sum())
    ratio = black_area / valid_area if valid_area else 0.0
    info = {
        'area': obj_area,
        'valid_area': valid_area,
        'removed_boundary_area': int(removed_boundary.sum()),
        'mean': mean,
        'std': std,
        'white_threshold': white_threshold,
        'boundary_strict_px': int(PARAMS.get('boundary_strict_px', 0)),
        'boundary_black_threshold': float(PARAMS.get('boundary_black_threshold', 100.0)),
        'black_area': black_area,
        'black_ratio': ratio,
        'components': comp_areas,
        'min_component_area': 0,
        'method': 'not_all_bgr_channels_above_white_threshold',
    }
    return filtered, ratio, info

def compute_corrected_ratio(image_bgr, obj_mask, black_mask):
    obj_area = int(obj_mask.sum())
    black_area = int(black_mask.sum())
    if obj_area <= 0:
        return 0.0, {'corrected_area': 0, 'body_area': 0, 'black_area': black_area, 'body_threshold': 0.0, 'raw_ratio': 0.0}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[obj_mask > 0]
    if vals.size == 0:
        return 0.0, {'corrected_area': black_area, 'body_area': 0, 'black_area': black_area, 'body_threshold': 0.0, 'raw_ratio': black_area / obj_area}
    valid_mask, removed_boundary = remove_dark_boundary_band(image_bgr, obj_mask)
    body_threshold = float(PARAMS['white_threshold'])
    white_pixels = compute_white_mask(image_bgr, body_threshold)
    body = (white_pixels & (valid_mask > 0) & (black_mask == 0)).astype(np.uint8)
    body_area = int(body.sum())
    corrected_area = max(black_area + body_area, black_area)
    corr_ratio = black_area / corrected_area if corrected_area else 0.0
    return corr_ratio, {
        'corrected_area': corrected_area,
        'body_area': body_area,
        'black_area': black_area,
        'body_threshold': float(body_threshold),
        'removed_boundary_area': int(removed_boundary.sum()),
        'valid_area': int(valid_mask.sum()),
        'raw_ratio': corr_ratio,
        'model_mask_area': obj_area,
    }

def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def eval_masks(pred_masks, gt_masks):
    matched_gt = set()
    tp = 0
    fp = 0
    matched_ious = []
    for pm in pred_masks:
        best_i = -1
        best = 0.0
        for j, gm in enumerate(gt_masks):
            if j in matched_gt:
                continue
            v = iou(pm, gm)
            if v > best:
                best = v
                best_i = j
        if best_i >= 0 and best >= PARAMS['iou_threshold']:
            tp += 1
            matched_gt.add(best_i)
            matched_ious.append(best)
        else:
            fp += 1
    fn = len(gt_masks) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou': float(np.mean(matched_ious)) if matched_ious else 0.0}


def draw_preview(image_bgr, kept_items, gt_masks, out_path, summary_text):
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    for item in kept_items:
        black = item['black_mask']
        overlay[black > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for gm in gt_masks:
        cv2.drawContours(canvas, mask_to_contours(gm), -1, GT_COLOR, 2)
    for item in kept_items:
        cv2.drawContours(canvas, mask_to_contours(item['mask']), -1, HOLLOW_COLOR, 2)
        x, y, w, h = cv2.boundingRect(item['mask'].astype(np.uint8))
        label = f"raw={item.get('raw_ratio', item['ratio']):.3f} corr={item.get('corr_ratio', item['ratio']):.3f}"
        tx, ty = x, max(18, y - 6)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(canvas, (tx - 2, ty - th - baseline - 3), (tx + tw + 4, ty + baseline + 2), (0, 0, 0), -1)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (6, 6), (760, 34), (0, 0, 0), -1)
    cv2.putText(canvas, summary_text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out_path, canvas)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir', required=True, help='Inference result dir containing split_test_labelme/.')
    parser.add_argument('--gt_dir', required=True, help='Ground-truth LabelMe folder for evaluation.')
    parser.add_argument('--out_preview_dir', default='preview_images_1', help='Preview output dir, relative to result_dir unless absolute.')
    parser.add_argument('--label', default='hollow')
    parser.add_argument('--selected_threshold', type=float, default=0.10)
    parser.add_argument('--thresholds', default='0.05,0.10,0.15,0.20,0.25')
    parser.add_argument('--white_threshold', type=float, default=150.0)
    parser.add_argument('--white_max_channel_diff', type=float, default=50.0)
    parser.add_argument('--boundary_strict_px', type=int, default=3)
    parser.add_argument('--boundary_black_threshold', type=float, default=100.0)
    parser.add_argument('--min_component_area_ratio', type=float, default=0.002)
    return parser.parse_args()


def main():
    global RESULT_DIR, PRED_DIR, GT_DIR, OUT_DIR, LABEL
    args = parse_args()
    RESULT_DIR = Path(args.result_dir)
    PRED_DIR = RESULT_DIR / 'split_test_labelme'
    GT_DIR = Path(args.gt_dir)
    out_arg = Path(args.out_preview_dir)
    OUT_DIR = out_arg if out_arg.is_absolute() else RESULT_DIR / out_arg
    LABEL = args.label
    PARAMS['selected_threshold'] = args.selected_threshold
    PARAMS['black_ratio_thresholds'] = [float(x) for x in args.thresholds.split(',') if x.strip()]
    PARAMS['white_threshold'] = args.white_threshold
    PARAMS['white_max_channel_diff'] = args.white_max_channel_diff
    PARAMS['boundary_strict_px'] = args.boundary_strict_px
    PARAMS['boundary_black_threshold'] = args.boundary_black_threshold
    PARAMS['min_component_area_ratio'] = args.min_component_area_ratio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob('*'):
        if old.is_file():
            old.unlink()

    per_threshold_preds = {t: [] for t in PARAMS['black_ratio_thresholds']}
    per_threshold_gts = {t: [] for t in PARAMS['black_ratio_thresholds']}
    selected = PARAMS['selected_threshold']
    per_image = []

    for pred_json in sorted(PRED_DIR.glob('*.json'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        stem = pred_json.stem
        image_path = PRED_DIR / (stem + '.jpg')
        if not image_path.exists():
            image_path = PRED_DIR / (stem + '.png')
        image = imread_unicode(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        pred_data = read_json(pred_json)
        gt_data = read_json(GT_DIR / pred_json.name)

        pred_shapes = [s for s in pred_data.get('shapes', []) if str(s.get('label', '')).strip().lower() == LABEL]
        gt_shapes = [s for s in gt_data.get('shapes', []) if str(s.get('label', '')).strip().lower() == LABEL]
        gt_masks = [polygon_mask(s, h, w).astype(bool) for s in gt_shapes]

        candidates = []
        for s in pred_shapes:
            m = polygon_mask(s, h, w)
            black, raw_ratio, info = compute_black_mask(image, m)
            corr_ratio, corr_info = compute_corrected_ratio(image, m, black)
            info.update(corr_info)
            candidates.append({'mask': m.astype(bool), 'black_mask': black.astype(bool), 'ratio': corr_ratio, 'raw_ratio': raw_ratio, 'corr_ratio': corr_ratio, 'info': info})

        for t in PARAMS['black_ratio_thresholds']:
            kept = [c['mask'] for c in candidates if c['ratio'] >= t]
            per_threshold_preds[t].extend(kept)
            per_threshold_gts[t].extend(gt_masks)

        kept_items = [c for c in candidates if c['ratio'] >= selected]
        m = eval_masks([c['mask'] for c in kept_items], gt_masks)
        per_image.append({'image': pred_json.name, 'gt': len(gt_masks), 'raw_pred': len(candidates), 'kept': len(kept_items), **m})
        txt = f"black_ratio>={selected:.2f} kept={len(kept_items)}/{len(candidates)} gt={len(gt_masks)} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        draw_preview(image, kept_items, gt_masks, OUT_DIR / f'{stem}_示意.png', txt)

    metrics = {}
    for t in PARAMS['black_ratio_thresholds']:
        metrics[f'{t:.2f}'] = eval_masks(per_threshold_preds[t], per_threshold_gts[t]) | {
            'pred_total': len(per_threshold_preds[t]),
            'gt_total': len(per_threshold_gts[t]),
            'black_ratio_threshold': t,
        }
    report = {'params': PARAMS, 'metrics_by_threshold': metrics, 'selected_threshold': selected, 'per_image_selected': per_image, 'preview_dir': str(OUT_DIR)}
    (RESULT_DIR / 'black_area_filter_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'preview_dir': str(OUT_DIR), 'metrics_by_threshold': metrics}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
