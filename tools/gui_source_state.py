from __future__ import annotations

import hashlib
from pathlib import Path


def source_signature(project_root: Path) -> str:
    files = [project_root / "app_streamlit.py"]
    files.extend(
        sorted((project_root / "metal_powder_sem_ai").rglob("*.py"))
    )
    files.append(project_root / "tools" / "run_streamlit_local.py")

    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        relative_path = path.relative_to(project_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
