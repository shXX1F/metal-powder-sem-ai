from __future__ import annotations

import io
import json
import locale
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from metal_powder_sem_ai.gui_preview import collect_uploaded_images

DEFAULT_PARTICLE_MODEL = Path("hollow_version0/model/particle/particle.pt")
DEFAULT_HOLLOW_MODEL = Path("hollow_version0/model/hollow/hollow.pt")
HOLLOW_RUNNER = Path("hollow_version0/code/mytools/run_hollow_report.py")


def _unique_path(folder: Path, file_name: str) -> Path:
    candidate = folder / file_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = folder / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1

def save_uploaded_inputs(uploaded_files: Iterable, input_dir: str | Path) -> list[Path]:
    """Save images and safely flatten image members from uploaded ZIP files."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for item in collect_uploaded_images(uploaded_files):
        target = _unique_path(input_dir, item.saved_name)
        target.write_bytes(item.data)
        saved.append(target)

    if not saved:
        raise ValueError("没有找到可识别的图片，请上传 jpg/png/tif/bmp 图片或包含这些图片的 ZIP。")
    return saved


def build_hollow_run_dir(base_dir: str | Path = "runs/hollow_gui") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(base_dir) / stamp
    (run_dir / "input").mkdir(parents=True, exist_ok=False)
    return run_dir


def run_hollow_pipeline(
    *,
    project_root: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    particle_model: str | Path,
    hollow_model: str | Path,
    image_batch_size: int = 1,
    torch_threads: int = 8,
    particle_full_score: float = 0.5,
    particle_tile_score: float = 0.2,
    hollow_full_score: float = 0.2,
    hollow_tile_score: float = 0.3,
    hollow_ratio_threshold: float = 0.25,
    hollow_preview_threshold: float = 0.10,
    hollow_white_threshold: float = 150.0,
    hollow_full_edge_threshold: float = 0.40,
    hollow_tile_edge_threshold: float = 0.55,
    particle_dark_mean_threshold: float = 100.0,
    hollow_only: bool = False,
    disable_particle_dark_filter: bool = False,
    disable_hollow_edge_filter: bool = False,
) -> str:
    """Run the original hollow-powder pipeline with GUI-selected inputs."""
    project_root = Path(project_root).resolve()
    runner = (project_root / HOLLOW_RUNNER).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"空心粉推理入口不存在：{runner}")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    particle_model = Path(particle_model).expanduser().resolve()
    hollow_model = Path(hollow_model).expanduser().resolve()
    if not hollow_model.is_file():
        raise FileNotFoundError(f"hollow 权重不存在：{hollow_model}")
    if not hollow_only and not particle_model.is_file():
        raise FileNotFoundError(f"particle 权重不存在：{particle_model}")

    command = [
        sys.executable,
        str(runner),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--particle_model",
        str(particle_model),
        "--hollow_model",
        str(hollow_model),
        "--image_batch_size",
        str(max(1, int(image_batch_size))),
        "--torch_threads",
        str(max(1, int(torch_threads))),
        "--particle_full_score",
        str(float(particle_full_score)),
        "--particle_tile_score",
        str(float(particle_tile_score)),
        "--particle_dark_mean_threshold",
        str(float(particle_dark_mean_threshold)),
        "--hollow_full_score",
        str(float(hollow_full_score)),
        "--hollow_tile_score",
        str(float(hollow_tile_score)),
        "--hollow_ratio_threshold",
        str(float(hollow_ratio_threshold)),
        "--hollow_preview_threshold",
        str(float(hollow_preview_threshold)),
        "--hollow_white_threshold",
        str(float(hollow_white_threshold)),
        "--hollow_full_edge_white_ratio_threshold",
        str(float(hollow_full_edge_threshold)),
        "--hollow_tile_edge_white_ratio_threshold",
        str(float(hollow_tile_edge_threshold)),
    ]
    if hollow_only:
        command.append("--hollow_only")
    if disable_particle_dark_filter:
        command.append("--disable_particle_dark_filter")
    if disable_hollow_edge_filter:
        command.append("--disable_hollow_edge_white_filter")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        check=False,
    )
    log_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    (output_dir / "run.log").write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        tail = log_text[-8000:] if log_text else "推理进程没有返回错误日志。"
        raise RuntimeError(f"空心粉推理失败（退出码 {completed.returncode}）：\n{tail}")
    return log_text


def load_hollow_result(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"结果摘要不存在：{summary_path}")
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(result.get("summary"), dict) or not isinstance(result.get("per_image"), list):
        raise ValueError(f"结果摘要格式无效：{summary_path}")
    return result


def build_result_zip(output_dir: str | Path) -> bytes:
    output_dir = Path(output_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    return buffer.getvalue()
