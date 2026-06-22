from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from metal_powder_sem_ai.classify_stat import classify_particles
from metal_powder_sem_ai.feature_extract import extract_all_features
from metal_powder_sem_ai.preprocess import imwrite_unicode, preprocess_sem_image
from metal_powder_sem_ai.report import export_excel_report
from metal_powder_sem_ai.segment import (
    ClassicalParticleSegmenter,
    MaskRCNNSegmenter,
    merge_instance_groups,
)
from metal_powder_sem_ai.visualize import draw_instances, save_visualization


WEIGHT_SEARCH_DIRS = ["runs", "models", "weights", "outputs"]
DEFAULT_MASKRCNN_WEIGHT = "runs/train_maskrcnn_trainval/maskrcnn_particle_last.pth"


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析上传图像，请确认文件为 jpg/png/tif 等常见图像格式。")
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detect_scale_bar_px(
    image_bgr: np.ndarray,
    search_bottom_fraction: float = 0.24,
) -> Optional[dict]:
    """Detect the horizontal SEM scale bar in the bottom information strip."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    height, width = image_bgr.shape[:2]
    y0 = int(round(height * (1.0 - search_bottom_fraction)))
    crop = image_bgr[y0:height].copy()
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 140)

    min_line_length = max(24, int(width * 0.035))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=6,
    )
    if lines is None:
        return None

    candidates = []
    crop_h = crop.shape[0]
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > max(2, int(round(crop_h * 0.01))):
            continue
        if dx < min_line_length:
            continue
        # Exclude frame borders and very long UI separator lines.
        if dx > int(width * 0.50):
            continue
        if x1 >= width - 4 or x2 >= width - 4:
            continue
        center_x = (x1 + x2) / 2.0
        if center_x > width * 0.70:
            continue
        if y1 < int(crop_h * 0.35) and y2 < int(crop_h * 0.35):
            continue
        x_left, x_right = sorted((x1, x2))
        y_mid = int(round((y1 + y2) / 2))
        preferred_center = width * 0.35
        score = dx + y_mid * 0.08 - abs(center_x - preferred_center) * 0.02
        candidates.append((score, dx, x_left, x_right, y_mid))

    if not candidates:
        return None

    _, length_px, x_left, x_right, y_mid = max(candidates, key=lambda item: item[0])
    preview = crop.copy()
    cv2.line(preview, (x_left, y_mid), (x_right, y_mid), (0, 0, 255), 3)
    cv2.rectangle(
        preview,
        (max(0, x_left - 4), max(0, y_mid - 12)),
        (min(width - 1, x_right + 4), min(crop_h - 1, y_mid + 12)),
        (0, 255, 0),
        2,
    )
    return {
        "length_px": float(length_px),
        "bbox": (x_left, y0 + y_mid, x_right, y0 + y_mid),
        "preview_bgr": preview,
    }


def format_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def find_weight_files() -> list[Path]:
    candidates: dict[Path, Path] = {}
    for path in [*Path(".").glob("*.pth"), *Path(".").glob("*.pt")]:
        if path.is_file():
            candidates[path.resolve()] = path
    for root_text in WEIGHT_SEARCH_DIRS:
        root = Path(root_text)
        if not root.exists():
            continue
        for suffix in ("*.pth", "*.pt"):
            for path in root.rglob(suffix):
                if path.is_file():
                    candidates[path.resolve()] = path
    return sorted(
        candidates.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def save_uploaded_weight(uploaded_weight) -> Path:
    output_dir = Path("runs/gui_weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_weight.name).name
    output_path = output_dir / safe_name
    data = uploaded_weight.getvalue()
    if not output_path.exists() or output_path.stat().st_size != len(data):
        output_path.write_bytes(data)
    return output_path


def open_native_weight_picker() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        st.sidebar.warning(f"本机文件选择器不可用，请手动输入路径：{exc}")
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        file_path = filedialog.askopenfilename(
            parent=root,
            title="选择 Mask R-CNN 权重文件",
            initialdir=str(Path.cwd()),
            filetypes=[
                ("PyTorch weights", "*.pth *.pt"),
                ("All files", "*.*"),
            ],
        )
        return str(Path(file_path)) if file_path else None
    except Exception as exc:
        st.sidebar.warning(f"打开本机文件选择器失败，请手动输入路径：{exc}")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def segment_with_maskrcnn_auto_retry(
    image_bgr: np.ndarray,
    weights_path: str,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    max_detections: int = 500,
    max_inference_side: int = 1280,
    tiled_inference: bool = False,
    tile_size: int = 896,
    tile_overlap: int = 224,
    device: Optional[str] = None,
) -> tuple[list, dict]:
    segmenter = load_maskrcnn_segmenter(
        weights_path=weights_path,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        max_detections=max_detections,
        max_inference_side=max_inference_side,
        device=device,
    )
    tried_scores = [float(score_threshold)]
    fallback_to_cpu = False
    try:
        instances = _run_maskrcnn_segmenter(
            segmenter,
            image_bgr,
            tiled_inference=tiled_inference,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    except RuntimeError as exc:
        if _is_cuda_oom(exc) and device != "cpu":
            del segmenter
            load_maskrcnn_segmenter.clear()
            _clear_cuda_cache()
            segmenter = load_maskrcnn_segmenter(
                weights_path=weights_path,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                max_detections=max_detections,
                max_inference_side=max_inference_side,
                device="cpu",
            )
            instances = _run_maskrcnn_segmenter(
                segmenter,
                image_bgr,
                tiled_inference=tiled_inference,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            fallback_to_cpu = True
        else:
            raise
    used_score_threshold = float(score_threshold)

    if len(instances) == 0:
        for retry_score in [0.35, 0.20, 0.10]:
            if retry_score >= score_threshold:
                continue
            segmenter.score_threshold = retry_score
            tried_scores.append(retry_score)
            instances = _run_maskrcnn_segmenter(
                segmenter,
                image_bgr,
                tiled_inference=tiled_inference,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            used_score_threshold = retry_score
            if len(instances) > 0:
                break

        segmenter.score_threshold = score_threshold

    diagnostics = {
        "used_score_threshold": used_score_threshold,
        "tried_scores": tried_scores,
        "auto_retry_used": used_score_threshold != float(score_threshold),
        "max_detections": int(max_detections),
        "max_inference_side": int(max_inference_side),
        "tiled_inference": bool(tiled_inference),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "device": segmenter.device,
        "fallback_to_cpu": fallback_to_cpu,
    }
    return instances, diagnostics


def _run_maskrcnn_segmenter(
    segmenter: MaskRCNNSegmenter,
    image_bgr: np.ndarray,
    tiled_inference: bool,
    tile_size: int,
    tile_overlap: int,
):
    if tiled_inference:
        return segmenter.segment_tiled(
            image_bgr,
            tile_size=int(tile_size),
            overlap=int(tile_overlap),
        )
    return segmenter.segment(image_bgr)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "outofmemoryerror" in message


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def maskrcnn_preset_config(preset_label: str) -> dict:
    if preset_label.startswith("综合识别"):
        return {
            "score_threshold": 0.30,
            "mask_threshold": 0.50,
            "max_detections": 10000,
            "max_inference_side": 1280,
            "tiled_inference": True,
            "tile_size": 640,
            "tile_overlap": 192,
            "inference_device": "cpu",
            "merge_inference": True,
        }
    return {
        "score_threshold": 0.50,
        "mask_threshold": 0.50,
        "max_detections": 300,
        "max_inference_side": 1280,
        "tiled_inference": False,
        "tile_size": 896,
        "tile_overlap": 224,
        "inference_device": None,
        "merge_inference": False,
    }


def enrich_display_stats(features, stats: dict) -> dict:
    """给旧 Streamlit 进程/旧统计模块补齐新版 GUI 需要的展示字段。"""
    display_stats = dict(stats)
    total = int(display_stats.get("total_particles", len(features)))
    hollow_count = int(
        display_stats.get(
            "hollow_particles",
            sum(1 for item in features if item.get("is_hollow")),
        )
    )
    agglomerate_count = int(
        display_stats.get(
            "agglomerate_particles",
            sum(1 for item in features if item.get("is_agglomerate")),
        )
    )

    q_values = [float(item.get("q_value", 0.0)) for item in features]
    mean_q = round((sum(q_values) / len(q_values)) if q_values else 0.0, 4)
    hollow_rate = round((hollow_count / total * 100.0) if total else 0.0, 2)
    agglomerate_rate = round((agglomerate_count / total * 100.0) if total else 0.0, 2)

    total_area = sum(float(item.get("area_um2", 0.0)) for item in features)
    agglomerate_area = sum(
        float(item.get("area_um2", 0.0))
        for item in features
        if item.get("is_agglomerate")
    )
    agglomerate_area_rate = round(
        (agglomerate_area / total_area * 100.0) if total_area else 0.0,
        2,
    )

    display_stats.setdefault("mean_sphericity_q", mean_q)
    display_stats.setdefault("mean_sphericity_q_text", f"{mean_q:.4f}")
    display_stats.setdefault("hollow_rate_percent", hollow_rate)
    display_stats.setdefault("hollow_rate_text", f"{hollow_rate:.2f}%")
    display_stats.setdefault("agglomerate_rate_percent", agglomerate_rate)
    display_stats.setdefault("agglomerate_rate_text", f"{agglomerate_rate:.2f}%")
    display_stats.setdefault("agglomerate_area_rate_percent", agglomerate_area_rate)
    display_stats.setdefault("agglomerate_area_rate_text", f"{agglomerate_area_rate:.2f}%")
    return display_stats


@st.cache_resource(show_spinner=False)
def load_maskrcnn_segmenter(
    weights_path: str,
    score_threshold: float,
    mask_threshold: float,
    max_detections: int,
    max_inference_side: int,
    device: Optional[str],
) -> MaskRCNNSegmenter:
    return MaskRCNNSegmenter(
        weights_path=weights_path,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        max_detections=max_detections,
        max_inference_side=max_inference_side,
        device=device,
    )


def get_pixel_size_um(image_bgr: Optional[np.ndarray] = None) -> float:
    st.sidebar.subheader("比例尺标定")
    st.sidebar.caption(
        "必须按当前图片的 SEM 标尺填写。同一倍率、同一导出尺寸的一批图通常可以复用同一个值。"
    )
    mode = st.sidebar.radio(
        "输入方式",
        ["自动检测标尺像素长度（推荐）", "手动输入标尺长度", "直接输入 pixel_size_um"],
        horizontal=False,
    )

    if mode == "直接输入 pixel_size_um":
        st.sidebar.caption("如果已经知道 1 pixel 等于多少微米，直接填这里。默认 0.1 只是示例值。")
        return st.sidebar.number_input(
            "1 pixel = X um",
            min_value=0.000001,
            value=0.100000,
            step=0.010000,
            format="%.6f",
        )

    if mode == "自动检测标尺像素长度（推荐）":
        scale_um = st.sidebar.number_input(
            "图中标尺真实长度 (um)",
            min_value=0.000001,
            value=30.0,
            step=10.0,
            format="%.6f",
            help="例如图片标尺写着 30 um，就填 30。当前版本自动检测横线像素长度，但不做OCR识别文字。",
        )
        detected_px = None
        if image_bgr is None:
            st.sidebar.info("上传 SEM 图像后会自动检测底部标尺横线长度。")
        else:
            detection = detect_scale_bar_px(image_bgr)
            if detection is None:
                st.sidebar.warning("没有稳定识别到标尺横线，请改用手动输入标尺长度。")
            else:
                detected_px = float(detection["length_px"])
                st.sidebar.success(f"自动识别标尺长度：{detected_px:.1f} px")
                st.sidebar.image(
                    bgr_to_rgb(detection["preview_bgr"]),
                    caption="红线为自动识别到的标尺横线",
                    use_container_width=True,
                )

        scale_px = st.sidebar.number_input(
            "标尺长度 (px，可手动修正)",
            min_value=1.0,
            value=float(detected_px or 300.0),
            step=1.0,
            format="%.2f",
            help="如果红线没有完全覆盖标尺，请手动改成真实像素长度。",
        )
        pixel_size_um = scale_um / scale_px
        st.sidebar.success(f"换算结果：1 pixel = {pixel_size_um:.6f} um")
        return float(pixel_size_um)

    scale_um = st.sidebar.number_input(
        "图中标尺真实长度 (um)",
        min_value=0.000001,
        value=30.0,
        step=10.0,
        format="%.6f",
        help="例如图片标尺写着 30 um，就填 30。",
    )
    scale_px = st.sidebar.number_input(
        "量到的标尺长度 (px)",
        min_value=1.0,
        value=300.0,
        step=10.0,
        format="%.2f",
        help="用 ImageJ/Fiji、LabelMe 或截图测量工具量出标尺横线有多少像素。",
    )
    pixel_size_um = scale_um / scale_px
    st.sidebar.success(f"换算结果：1 pixel = {pixel_size_um:.6f} um")
    return float(pixel_size_um)


def build_output_dir(base_dir: str = "runs/gui") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def choose_maskrcnn_weights() -> Optional[str]:
    st.sidebar.markdown("**Mask R-CNN 权重**")
    mode_options = ["本机选择 .pth（不限大小）", "手动输入路径"]
    mode = st.sidebar.radio(
        "权重选择方式",
        mode_options,
        index=0,
        horizontal=False,
    )

    weights_path: Optional[str] = None
    if mode == "本机选择 .pth（不限大小）":
        st.sidebar.caption(
            "直接选择电脑上的权重文件，不经过网页上传，所以不受文件大小限制。"
        )
        if st.sidebar.button("打开本机文件选择器", use_container_width=True):
            selected_path = open_native_weight_picker()
            if selected_path:
                st.session_state["maskrcnn_weights_path"] = selected_path
        weights_path = st.sidebar.text_input(
            "当前权重路径",
            value=st.session_state.get("maskrcnn_weights_path", DEFAULT_MASKRCNN_WEIGHT),
            help="可以点击上面的按钮选择，也可以直接粘贴 .pth / .pt 路径。",
        ).strip()
        if weights_path:
            st.session_state["maskrcnn_weights_path"] = weights_path

    else:
        weights_path = st.sidebar.text_input(
            "Mask R-CNN 权重路径",
            value=st.session_state.get("maskrcnn_weights_path", DEFAULT_MASKRCNN_WEIGHT),
            help="例如：runs/train_maskrcnn_trainval/maskrcnn_particle_last.pth",
        ).strip()
        if weights_path:
            st.session_state["maskrcnn_weights_path"] = weights_path

    if weights_path:
        path = Path(weights_path)
        if path.exists():
            st.sidebar.success(f"权重文件已找到：{format_file_size(path.stat().st_size)}")
            lowered = weights_path.lower()
            if "smoke" in lowered or "synthetic" in lowered:
                st.sidebar.warning(
                    "这个权重文件名像是测试/冒烟训练权重，不一定是火山正式训练结果。"
                    "正式识别建议使用火山训练得到的 maskrcnn_particle_last.pth。"
                )
        else:
            st.sidebar.error("当前路径找不到权重文件。请确认 .pth 是否已下载到本机。")
    return weights_path


def render_color_legend() -> None:
    legend_html = """
    <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;
                margin:8px 0 4px 0; font-size:15px; line-height:1.6;">
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; display:inline-block; border-radius:3px;
                     background:#00b400; border:1px solid #111;"></span>
        绿色：球形颗粒
      </span>
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; display:inline-block; border-radius:3px;
                     background:#ffdc00; border:1px solid #111;"></span>
        黄色：团聚体
      </span>
      <span style="display:inline-flex; align-items:center; gap:6px;">
        <span style="width:14px; height:14px; display:inline-block; border-radius:3px;
                     background:#ff0000; border:1px solid #111;"></span>
        红色：非球形颗粒
      </span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


def render_feature_table(features, max_rows: int = 200) -> None:
    columns = [
        ("particle_id", "ID"),
        ("q_value", "Q值"),
        ("axis_ratio", "轴比"),
        ("area_um2", "面积(um²)"),
        ("is_spherical", "球形"),
        ("is_agglomerate", "团聚体"),
    ]
    rows = features[:max_rows]
    header_html = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_parts = []
    for item in rows:
        cells = []
        for key, _ in columns:
            value = item.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    table_html = f"""
    <div style="max-height: 420px; overflow: auto; border: 1px solid #e5e7eb;">
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead style="position: sticky; top: 0; background: #f8fafc;">
          <tr>{header_html}</tr>
        </thead>
        <tbody>{''.join(body_parts)}</tbody>
      </table>
    </div>
    <style>
      th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
      th {{ font-weight: 700; }}
    </style>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    if len(features) > max_rows:
        st.caption(f"页面预览前 {max_rows} 行，完整颗粒级数据请下载 CSV 或 Excel。")


def apply_white_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --app-bg: #ffffff;
            --app-text: #000000;
            --app-border: #d1d5db;
            --app-muted-bg: #f8fafc;
          }

          html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--app-bg) !important;
            color: var(--app-text) !important;
          }

          [data-testid="stHeader"],
          [data-testid="stToolbar"],
          [data-testid="stSidebar"],
          [data-testid="stSidebarContent"] {
            background-color: var(--app-bg) !important;
            color: var(--app-text) !important;
          }

          h1, h2, h3, h4, h5, h6, p, span, label, div,
          [data-testid="stMarkdownContainer"],
          [data-testid="stMetric"],
          [data-testid="stMetric"] * {
            color: var(--app-text) !important;
          }

          input, textarea, select,
          [data-baseweb="input"],
          [data-baseweb="select"],
          [data-baseweb="select"] > div,
          [data-baseweb="textarea"],
          [data-baseweb="base-input"],
          [data-baseweb="base-input"] > div {
            background-color: #ffffff !important;
            color: var(--app-text) !important;
            border-color: var(--app-border) !important;
          }

          [data-baseweb="select"] *,
          [data-baseweb="input"] *,
          [data-baseweb="base-input"] * {
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
          }

          [data-baseweb="popover"],
          [data-baseweb="popover"] *,
          [role="listbox"],
          [role="listbox"] * {
            background-color: #ffffff !important;
            color: var(--app-text) !important;
          }

          [role="option"],
          [role="option"] *,
          [role="option"][aria-selected="true"],
          [role="option"][aria-selected="true"] *,
          [role="option"]:hover,
          [role="option"]:hover *,
          [data-baseweb="menu"] *,
          [data-baseweb="popover"] li,
          [data-baseweb="popover"] li *,
          [data-baseweb="select"] [aria-selected="true"],
          [data-baseweb="select"] [aria-selected="true"] * {
            background-color: #ffffff !important;
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
          }

          [role="option"][aria-selected="true"],
          [role="option"]:hover,
          [data-baseweb="popover"] li:hover {
            outline: 1px solid var(--app-border) !important;
          }

          [data-baseweb="menu"] li,
          [data-baseweb="menu"] li *,
          [data-baseweb="menu"] li:hover,
          [data-baseweb="menu"] li:hover *,
          [data-baseweb="menu"] li:focus,
          [data-baseweb="menu"] li:focus *,
          [data-baseweb="menu"] li:active,
          [data-baseweb="menu"] li:active *,
          [data-baseweb="menu"] li[aria-selected="true"],
          [data-baseweb="menu"] li[aria-selected="true"] *,
          [data-baseweb="menu"] li[data-highlighted="true"],
          [data-baseweb="menu"] li[data-highlighted="true"] *,
          [role="listbox"] li,
          [role="listbox"] li *,
          [role="listbox"] li:hover,
          [role="listbox"] li:hover *,
          [role="listbox"] li:focus,
          [role="listbox"] li:focus *,
          [role="listbox"] li:active,
          [role="listbox"] li:active *,
          [role="listbox"] [aria-selected="true"],
          [role="listbox"] [aria-selected="true"] *,
          [role="listbox"] [data-highlighted="true"],
          [role="listbox"] [data-highlighted="true"] * {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            box-shadow: none !important;
          }

          [data-baseweb="menu"] li::before,
          [data-baseweb="menu"] li::after,
          [role="listbox"] li::before,
          [role="listbox"] li::after,
          [role="option"]::before,
          [role="option"]::after {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
          }

          button,
          button[kind="primary"],
          button[kind="secondary"],
          [data-testid="stButton"] button,
          [data-testid="stDownloadButton"] button,
          [data-testid="stFileUploader"] button,
          [data-testid="baseButton-primary"],
          [data-testid="baseButton-secondary"] {
            background-color: #ffffff !important;
            color: var(--app-text) !important;
            border: 1px solid var(--app-border) !important;
            box-shadow: none !important;
          }

          button:hover,
          button[kind="primary"]:hover,
          button[kind="secondary"]:hover,
          [data-testid="stButton"] button:hover,
          [data-testid="stDownloadButton"] button:hover,
          [data-testid="stFileUploader"] button:hover,
          [data-testid="baseButton-primary"]:hover,
          [data-testid="baseButton-secondary"]:hover {
            background-color: var(--app-muted-bg) !important;
            color: var(--app-text) !important;
            border-color: #000000 !important;
          }

          [data-testid="stAlert"],
          [data-testid="stExpander"],
          [data-testid="stFileUploader"],
          [data-testid="stFileUploader"] section,
          [data-testid="stFileUploader"] section *,
          [data-testid="stFileUploaderDropzone"],
          [data-testid="stFileUploaderDropzone"] *,
          [data-testid="stDataFrame"] {
            background-color: var(--app-bg) !important;
            color: var(--app-text) !important;
          }

          [data-testid="stFileUploader"] section,
          [data-testid="stFileUploaderDropzone"] {
            border: 1px solid var(--app-border) !important;
            border-radius: 8px !important;
            padding: 18px !important;
            box-shadow: none !important;
          }

          table, thead, tbody, tr, th, td {
            color: var(--app-text) !important;
          }

          thead {
            background-color: var(--app-muted-bg) !important;
          }

          /* Keep Streamlit/BaseWeb menu and expander states light.
             BaseWeb injects hover/active backgrounds late, so these rules sit last. */
          div[data-baseweb="popover"],
          div[data-baseweb="popover"] *,
          div[data-baseweb="menu"],
          div[data-baseweb="menu"] *,
          div[role="listbox"],
          div[role="listbox"] *,
          ul[role="listbox"],
          ul[role="listbox"] *,
          li[role="option"],
          li[role="option"] *,
          div[role="option"],
          div[role="option"] * {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            box-shadow: none !important;
          }

          div[data-baseweb="popover"] *:hover,
          div[data-baseweb="popover"] *:focus,
          div[data-baseweb="popover"] *:active,
          div[data-baseweb="popover"] *[aria-selected="true"],
          div[data-baseweb="popover"] *[aria-current="true"],
          div[data-baseweb="popover"] *[data-highlighted="true"],
          div[data-baseweb="popover"] *[data-focus-visible-added],
          div[data-baseweb="menu"] *:hover,
          div[data-baseweb="menu"] *:focus,
          div[data-baseweb="menu"] *:active,
          div[data-baseweb="menu"] *[aria-selected="true"],
          div[data-baseweb="menu"] *[aria-current="true"],
          div[data-baseweb="menu"] *[data-highlighted="true"],
          div[data-baseweb="menu"] *[data-focus-visible-added],
          [role="option"]:hover,
          [role="option"]:focus,
          [role="option"]:active,
          [role="option"][aria-selected="true"],
          [role="option"][data-highlighted="true"] {
            background: #f8fafc !important;
            background-color: #f8fafc !important;
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            box-shadow: none !important;
          }

          div[data-baseweb="popover"] *::before,
          div[data-baseweb="popover"] *::after,
          div[data-baseweb="menu"] *::before,
          div[data-baseweb="menu"] *::after {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
          }

          [data-testid="stExpander"],
          [data-testid="stExpander"] *,
          [data-testid="stExpander"] details,
          [data-testid="stExpander"] details *,
          [data-testid="stExpander"] summary,
          [data-testid="stExpander"] summary *,
          [data-testid="stExpander"] button,
          [data-testid="stExpander"] button * {
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
          }

          [data-testid="stExpander"] summary,
          [data-testid="stExpander"] summary:hover,
          [data-testid="stExpander"] summary:focus,
          [data-testid="stExpander"] summary:active,
          [data-testid="stExpander"] details[open] summary,
          [data-testid="stExpander"] button,
          [data-testid="stExpander"] button:hover,
          [data-testid="stExpander"] button:focus,
          [data-testid="stExpander"] button:active,
          [data-testid="stExpander"] button[aria-expanded="true"] {
            background: #f8fafc !important;
            background-color: #f8fafc !important;
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            border-color: var(--app-border) !important;
            box-shadow: none !important;
            outline-color: var(--app-border) !important;
          }

          [data-testid="stNumberInput"] input {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            caret-color: #000000 !important;
          }

          [data-testid="stNumberInput"],
          [data-testid="stNumberInput"] * {
            --border-color: #000000 !important;
            --border: #000000 !important;
          }

          [data-testid="stNumberInputContainer"],
          [data-testid="stNumberInputContainer"]:hover,
          [data-testid="stNumberInputContainer"]:focus-within {
            border: 1px solid #000000 !important;
            border-color: #000000 !important;
            outline: 0 !important;
            box-shadow: none !important;
          }

          [data-testid="stNumberInputContainer"] > div,
          [data-testid="stNumberInputContainer"] button,
          [data-testid="stNumberInputContainer"] button:hover,
          [data-testid="stNumberInputContainer"] button:focus,
          [data-testid="stNumberInputContainer"] button:active {
            border-color: #000000 !important;
            box-shadow: none !important;
          }

          [data-testid="stNumberInput"] div[data-baseweb="input"],
          [data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
          [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-color: #000000 !important;
            outline-color: #000000 !important;
            box-shadow: none !important;
          }

          [data-testid="stNumberInput"] div[data-baseweb="input"] > div,
          [data-testid="stNumberInput"] div[data-baseweb="base-input"],
          [data-testid="stNumberInput"] div[data-baseweb="base-input"] > div {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-color: #000000 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="金属粉末 SEM 图像 AI 识别",
        page_icon="",
        layout="wide",
    )
    apply_white_theme()

    st.title("金属粉末 SEM 图像 AI 识别系统")

    uploaded_image = st.sidebar.file_uploader(
        "上传 SEM 图像",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    )

    image_bgr: Optional[np.ndarray] = None
    if uploaded_image is not None:
        try:
            image_bgr = decode_uploaded_image(uploaded_image)
        except Exception as exc:
            st.error(str(exc))
            return

    pixel_size_um = get_pixel_size_um(image_bgr)

    st.sidebar.subheader("预处理参数")
    crop_bottom_fraction = st.sidebar.slider(
        "裁掉底部信息栏比例",
        min_value=0.0,
        max_value=0.30,
        value=0.12,
        step=0.01,
        help="SEM 图底部有标尺、倍率、文字时建议 0.10-0.15；没有底部信息栏时设为 0。",
    )

    st.sidebar.subheader("分割方式")
    segmenter_label = st.sidebar.selectbox(
        "选择分割方式",
        ["训练模型 Mask R-CNN（推荐）", "传统阈值分割（无模型调试）"],
        index=0,
        help="已经有训练好的 .pth 时选 Mask R-CNN；没有权重或只想快速试流程时选传统阈值分割。",
    )
    segmenter_name = "maskrcnn" if segmenter_label.startswith("训练模型") else "classical"

    weights_path: Optional[str] = None
    score_threshold = 0.5
    mask_threshold = 0.5
    max_detections = 300
    max_inference_side = 1280
    tiled_inference = False
    tile_size = 896
    tile_overlap = 224
    merge_inference = False
    inference_device: Optional[str] = None
    min_area_px = 80
    peak_min_distance = 12
    blur_ksize = 5

    if segmenter_name == "maskrcnn":
        st.sidebar.info(
            "先选择权重，再选择识别场景即可。一般直接使用“综合识别”，系统会自动合并整图和切块结果。"
        )
        weights_path = choose_maskrcnn_weights()
        preset_label = st.sidebar.selectbox(
            "识别场景",
            [
                "综合识别（推荐，普通+密集合并）",
                "自定义专家参数",
            ],
            index=0,
            help="综合识别会自动合并普通整图和密集切块结果；需要细调时选自定义专家参数。",
        )
        preset = maskrcnn_preset_config(preset_label)
        score_threshold = preset["score_threshold"]
        mask_threshold = preset["mask_threshold"]
        max_detections = preset["max_detections"]
        max_inference_side = preset["max_inference_side"]
        tiled_inference = preset["tiled_inference"]
        tile_size = preset["tile_size"]
        tile_overlap = preset["tile_overlap"]
        inference_device = preset["inference_device"]
        merge_inference = preset["merge_inference"]

        if preset_label.startswith("自定义"):
            with st.sidebar.expander("专家参数", expanded=True):
                score_threshold = st.slider(
                    "检测置信度阈值",
                    0.0,
                    1.0,
                    float(score_threshold),
                    0.05,
                    help="越低越容易检出颗粒，但误检可能增加；越高越保守。",
                )
                mask_threshold = st.slider(
                    "Mask 阈值",
                    0.0,
                    1.0,
                    float(mask_threshold),
                    0.05,
                    help="一般保持 0.5。边界偏小可略降，边界偏大可略升。",
                )
                max_detections = st.number_input(
                    "最大识别颗粒数/块",
                    min_value=100,
                    max_value=20000,
                    value=int(max_detections),
                    step=100,
                    help="整图识别时是全图上限；切块识别时是每个切块的上限，整张图可累计到几千个。",
                )
                max_inference_side = st.number_input(
                    "最大推理边长(px)",
                    min_value=640,
                    max_value=2400,
                    value=int(max_inference_side),
                    step=160,
                    help="值越小越省内存但小颗粒可能更容易漏检；6GB 显卡或 CPU 内存不足时建议 960-1280。",
                )
                tiled_inference = st.checkbox(
                    "切块识别",
                    value=bool(tiled_inference),
                    help="颗粒很多、整体识别漏小颗粒时打开。会逐块识别再拼回整图，速度较慢但小颗粒更容易检出。",
                )
                if tiled_inference:
                    tile_size = st.select_slider(
                        "切块尺寸(px)",
                        options=[512, 640, 768, 896, 1024, 1280],
                        value=int(tile_size),
                        help="小球漏检时优先用 512-640；图像内存足够且大颗粒多时可用 896-1024。",
                    )
                    tile_overlap = st.select_slider(
                        "切块重叠(px)",
                        options=[128, 192, 224, 256, 320],
                        value=int(tile_overlap),
                        help="重叠越大，边界颗粒越不容易漏，但速度更慢。",
                    )
                merge_inference = st.checkbox(
                    "合并普通整图 + 密集切块结果",
                    value=bool(merge_inference),
                    help="先跑普通整图识别，再跑密集切块识别，并按重叠去重合并。适合大小颗粒同时存在的图。",
                )
                device_index = 0
                if inference_device == "cpu":
                    device_index = 1
                elif inference_device == "cuda":
                    device_index = 2
                device_label = st.selectbox(
                    "推理设备",
                    ["自动（优先GPU，显存不足切CPU）", "CPU（更稳但较慢）", "GPU CUDA（较快但吃显存）"],
                    index=device_index,
                    help="6GB 显卡遇到密集大图容易爆显存，可选 CPU。",
                )
                if device_label.startswith("CPU"):
                    inference_device = "cpu"
                elif device_label.startswith("GPU"):
                    inference_device = "cuda"
                else:
                    inference_device = None
        else:
            st.sidebar.caption(
                "当前预设："
                f"置信度 {score_threshold:.2f}，"
                f"{'切块识别' if tiled_inference else '整图识别'}，"
                f"{'每块最多' if tiled_inference else '最多'} {max_detections} 个颗粒。"
            )
    else:
        st.sidebar.info("传统阈值分割不需要 .pth，适合没模型时调试；正式识别建议用 Mask R-CNN。")
        with st.sidebar.expander("传统分割参数", expanded=True):
            blur_ksize = st.select_slider(
                "高斯核大小",
                options=[3, 5, 7, 9],
                value=5,
            )
            min_area_px = st.number_input("最小颗粒面积 (px)", 1, 100000, 80, 10)
            peak_min_distance = st.number_input("粘连拆分距离", 1, 1000, 12, 1)

    run_button = st.sidebar.button("开始识别", type="primary", use_container_width=True)

    if uploaded_image is None:
        st.info("请先在左侧上传 SEM 图像，然后输入比例尺参数。")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("原始图像")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)

    if not run_button:
        with right:
            st.subheader("结果预览")
            st.write("点击左侧“开始识别”后显示标注结果。")
        return

    try:
        segmentation_diagnostics = {}
        with st.spinner("正在预处理、分割并计算颗粒特征..."):
            pre = preprocess_sem_image(
                image_bgr,
                crop_bottom_fraction=crop_bottom_fraction,
                blur_ksize=blur_ksize,
            )

            if segmenter_name == "classical":
                segmenter = ClassicalParticleSegmenter(
                    min_area_px=int(min_area_px),
                    peak_min_distance=int(peak_min_distance),
                )
                instances = segmenter.segment(pre.binary)
            else:
                if not weights_path:
                    raise ValueError("请选择或输入 Mask R-CNN 权重路径。")
                if merge_inference:
                    normal_max_detections = min(int(max_detections), 1000)
                    normal_instances, normal_diagnostics = segment_with_maskrcnn_auto_retry(
                        image_bgr=pre.image,
                        weights_path=weights_path,
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        max_detections=normal_max_detections,
                        max_inference_side=int(max_inference_side),
                        tiled_inference=False,
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        device=inference_device,
                    )
                    load_maskrcnn_segmenter.clear()
                    _clear_cuda_cache()
                    dense_instances, dense_diagnostics = segment_with_maskrcnn_auto_retry(
                        image_bgr=pre.image,
                        weights_path=weights_path,
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        max_detections=int(max_detections),
                        max_inference_side=int(max_inference_side),
                        tiled_inference=True,
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        device=inference_device,
                    )
                    instances = merge_instance_groups(
                        normal_instances,
                        dense_instances,
                    )
                    segmentation_diagnostics = dict(dense_diagnostics)
                    segmentation_diagnostics.update(
                        {
                            "merge_inference": True,
                            "normal_instances": len(normal_instances),
                            "dense_instances": len(dense_instances),
                            "merged_instances": len(instances),
                            "merged_duplicates": max(
                                0,
                                len(normal_instances) + len(dense_instances) - len(instances),
                            ),
                        }
                    )
                    if normal_diagnostics.get("fallback_to_cpu") or dense_diagnostics.get("fallback_to_cpu"):
                        segmentation_diagnostics["fallback_to_cpu"] = True
                else:
                    instances, segmentation_diagnostics = segment_with_maskrcnn_auto_retry(
                        image_bgr=pre.image,
                        weights_path=weights_path,
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        max_detections=int(max_detections),
                        max_inference_side=int(max_inference_side),
                        tiled_inference=bool(tiled_inference),
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        device=inference_device,
                    )

            features = extract_all_features(
                instances,
                pixel_size_um=pixel_size_um,
                gray=pre.enhanced,
            )
            masks = [instance.mask for instance in instances]
            classified, stats = classify_particles(features, masks=masks)
            stats = enrich_display_stats(classified, stats)
            visualized = draw_instances(pre.image, instances, classified)

            output_dir = build_output_dir()
            image_path = output_dir / "uploaded_image.png"
            binary_path = output_dir / "binary.png"
            visual_path = output_dir / "visualized.png"
            report_path = output_dir / "report.xlsx"
            imwrite_unicode(image_path, image_bgr)
            imwrite_unicode(binary_path, pre.binary)
            save_visualization(visual_path, visualized)
            export_excel_report(report_path, classified, stats)

        with right:
            st.subheader("识别标注图")
            st.image(bgr_to_rgb(visualized), use_container_width=True)
            render_color_legend()

        if segmenter_name == "maskrcnn":
            if segmentation_diagnostics.get("auto_retry_used"):
                st.warning(
                    "默认检测置信度 0.50 没有识别到颗粒，系统已自动使用更宽松阈值 "
                    f"{segmentation_diagnostics['used_score_threshold']:.2f} 重新识别。"
                )
            if segmentation_diagnostics.get("fallback_to_cpu"):
                st.warning("GPU 显存不足，已自动切换到 CPU 完成识别；速度会慢一些。")
            if segmentation_diagnostics.get("tiled_inference"):
                st.info(
                    "已使用切块识别："
                    f"tile={segmentation_diagnostics['tile_size']} px，"
                    f"overlap={segmentation_diagnostics['tile_overlap']} px。"
                )
            if segmentation_diagnostics.get("merge_inference"):
                st.info(
                    "已合并普通整图识别与密集切块识别结果："
                    f"普通 {int(segmentation_diagnostics.get('normal_instances', 0))} 个，"
                    f"密集 {int(segmentation_diagnostics.get('dense_instances', 0))} 个，"
                    f"去重 {int(segmentation_diagnostics.get('merged_duplicates', 0))} 个，"
                    f"最终 {int(segmentation_diagnostics.get('merged_instances', len(instances)))} 个。"
                )
            max_detection_hint = int(segmentation_diagnostics.get("max_detections", 0) or 0)
            if (
                max_detection_hint
                and not segmentation_diagnostics.get("tiled_inference")
                and len(instances) >= max_detection_hint
            ):
                st.warning(
                    f"当前识别数量达到最大识别颗粒数 {max_detection_hint}，"
                    "密集图可能仍被截断。可将识别场景改为“自定义专家参数”后继续调高。"
                )
            if len(instances) == 0:
                st.error(
                    "当前权重在这张图上仍然没有识别到颗粒。请优先检查："
                    "1）是否选中了火山训练得到的 maskrcnn_particle_last.pth；"
                    "2）训练是否真的完成且不是空模型；"
                    "3）这张图和训练图的倍率、裁剪方式、图像风格是否一致。"
                )

        stats_cols = st.columns(3)
        stats_cols[0].metric("总颗粒数", stats["total_particles"])
        stats_cols[1].metric("平均球形度 Q", stats["mean_sphericity_q_text"])
        stats_cols[2].metric(
            "团聚率 P_agglom",
            stats["agglomerate_rate_text"],
            delta=f"{stats['agglomerate_particles']} / {stats['total_particles']}",
            delta_color="off",
        )

        st.subheader("补充统计")
        extra_cols = st.columns(3)
        extra_cols[0].metric("球形颗粒率 S", stats["sphericity_rate_s_text"])
        extra_cols[1].metric("球形颗粒数", stats["spherical_particles"])
        extra_cols[2].metric("团聚面积率 P_area", stats["agglomerate_area_rate_text"])

        st.subheader("颗粒级特征")
        render_feature_table(classified)

        download_cols = st.columns(3)
        with open(report_path, "rb") as f:
            download_cols[0].download_button(
                "下载 Excel 报告",
                data=f.read(),
                file_name="metal_powder_sem_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with open(visual_path, "rb") as f:
            download_cols[1].download_button(
                "下载标注图",
                data=f.read(),
                file_name="metal_powder_sem_visualized.png",
                mime="image/png",
                use_container_width=True,
            )
        with open(output_dir / "report.particles.csv", "rb") as f:
            download_cols[2].download_button(
                "下载颗粒 CSV",
                data=f.read(),
                file_name="metal_powder_particles.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("预处理结果"):
            col1, col2 = st.columns(2)
            col1.image(pre.gray, caption="灰度图", use_container_width=True)
            col2.image(pre.binary, caption="Otsu 二值图", use_container_width=True)
            st.caption(f"本次输出目录：{output_dir}")

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if (
            "cuda out of memory" in lowered
            or "defaultcpuallocator" in lowered
            or "not enough memory" in lowered
        ):
            st.error(
                "识别失败：内存不足。请把“识别场景”改成“自定义专家参数”，"
                "将推理设备选为 CPU，并把最大推理边长调到 960 或切块尺寸调到 512/640。"
            )
        else:
            st.error(f"识别失败：{exc}")


if __name__ == "__main__":
    main()
