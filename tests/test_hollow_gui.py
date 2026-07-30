from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from metal_powder_sem_ai.hollow_gui import (
    _subprocess_environment,
    build_result_zip,
    load_hollow_result,
    save_uploaded_inputs,
)


class FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_subprocess_environment_preserves_gui_dependency_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dependency_dir = tmp_path / "deps"
    dependency_dir.mkdir()
    inherited_dir = tmp_path / "inherited"
    inherited_dir.mkdir()
    monkeypatch.syspath_prepend(str(dependency_dir))
    monkeypatch.setenv("PYTHONPATH", str(inherited_dir))

    env = _subprocess_environment(tmp_path)
    python_paths = env["PYTHONPATH"].split(os.pathsep)

    assert python_paths[0] == str(tmp_path)
    assert str(dependency_dir) in python_paths
    assert str(inherited_dir) in python_paths
    assert len(python_paths) == len(
        {os.path.normcase(os.path.abspath(path)) for path in python_paths}
    )
    assert env["PYTHONUTF8"] == "1"


def test_subprocess_environment_can_import_dependency_from_parent_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dependency_dir = tmp_path / "deps"
    package_dir = dependency_dir / "runtime_probe"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(dependency_dir))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runtime_probe; print(runtime_probe.VALUE)",
        ],
        cwd=str(tmp_path),
        env=_subprocess_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "42"


def test_save_uploaded_inputs_flattens_safe_image_members(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.png", b"png")
        archive.writestr("nested/sample.jpg", b"jpg")
        archive.writestr("nested/ignore.txt", b"text")

    saved = save_uploaded_inputs(
        [FakeUpload("images.zip", buffer.getvalue())],
        tmp_path / "input",
    )

    assert [path.name for path in saved] == ["outside.png", "sample.jpg"]
    assert all(path.parent == tmp_path / "input" for path in saved)
    assert not (tmp_path / "outside.png").exists()


def test_save_uploaded_inputs_rejects_upload_without_images(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"no image")

    try:
        save_uploaded_inputs([FakeUpload("notes.zip", buffer.getvalue())], tmp_path / "input")
    except ValueError as exc:
        assert "没有找到可识别的图片" in str(exc)
    else:
        raise AssertionError("Expected upload without images to be rejected")


def test_result_summary_and_zip_round_trip(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    expected = {"summary": {"image_count": 1}, "per_image": [{"image": "a.png"}]}
    (output_dir / "summary.json").write_text(
        json.dumps(expected, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text("Images: 1\n", encoding="utf-8")

    assert load_hollow_result(output_dir) == expected

    with zipfile.ZipFile(io.BytesIO(build_result_zip(output_dir))) as archive:
        assert sorted(archive.namelist()) == ["summary.json", "summary.txt"]
