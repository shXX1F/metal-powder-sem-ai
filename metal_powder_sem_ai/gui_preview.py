from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class UploadedImageItem:
    """One directly uploaded image or one image extracted from an uploaded ZIP."""

    display_name: str
    saved_name: str
    data: bytes
    source_name: str


def safe_file_name(name: str, fallback: str = "image.png") -> str:
    """Return a flat, filesystem-safe upload name."""
    base_name = Path(str(name).replace("\\", "/")).name
    keep = [ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in base_name]
    cleaned = "".join(keep).strip("._")
    return (cleaned or fallback)[:180]


def uploaded_bytes(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    data = uploaded_file.read()
    return data if isinstance(data, bytes) else bytes(data)


def _unique_name(file_name: str, used_names: set[str]) -> str:
    candidate = file_name
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate.casefold())
    return candidate


def collect_uploaded_images(uploaded_files: Iterable) -> list[UploadedImageItem]:
    """Flatten image uploads and ZIP image members into previewable in-memory items."""
    items: list[UploadedImageItem] = []
    used_names: set[str] = set()

    for uploaded_file in uploaded_files:
        original_name = str(getattr(uploaded_file, "name", "upload"))
        upload_name = safe_file_name(original_name)
        suffix = Path(upload_name).suffix.lower()
        data = uploaded_bytes(uploaded_file)

        if suffix == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        normalized_member = info.filename.replace("\\", "/")
                        member_suffix = Path(normalized_member).suffix.lower()
                        if member_suffix not in IMAGE_EXTENSIONS:
                            continue
                        member_name = safe_file_name(normalized_member)
                        saved_name = _unique_name(member_name, used_names)
                        items.append(
                            UploadedImageItem(
                                display_name=normalized_member,
                                saved_name=saved_name,
                                data=archive.read(info),
                                source_name=original_name,
                            )
                        )
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                raise ValueError(f"无法读取 ZIP 文件“{original_name}”：{exc}") from exc
            continue

        if suffix not in IMAGE_EXTENSIONS:
            continue
        saved_name = _unique_name(upload_name, used_names)
        items.append(
            UploadedImageItem(
                display_name=original_name,
                saved_name=saved_name,
                data=data,
                source_name=original_name,
            )
        )

    return items
