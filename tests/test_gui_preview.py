from __future__ import annotations

import io
import zipfile

import pytest

from metal_powder_sem_ai.gui_preview import collect_uploaded_images


class FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_collect_uploaded_images_includes_direct_and_zip_members() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("folder/inside.png", b"zip-image")
        archive.writestr("folder/readme.txt", b"skip")

    items = collect_uploaded_images(
        [
            FakeUpload("direct.jpg", b"direct-image"),
            FakeUpload("images.zip", archive_buffer.getvalue()),
        ]
    )

    assert [item.display_name for item in items] == [
        "direct.jpg",
        "folder/inside.png",
    ]
    assert [item.saved_name for item in items] == ["direct.jpg", "inside.png"]
    assert [item.source_name for item in items] == ["direct.jpg", "images.zip"]
    assert [item.data for item in items] == [b"direct-image", b"zip-image"]


def test_collect_uploaded_images_assigns_unique_result_names() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("one/sample.png", b"first")
        archive.writestr("two/sample.png", b"second")

    items = collect_uploaded_images(
        [
            FakeUpload("sample.png", b"direct"),
            FakeUpload("images.zip", archive_buffer.getvalue()),
        ]
    )

    assert [item.saved_name for item in items] == [
        "sample.png",
        "sample_2.png",
        "sample_3.png",
    ]


def test_collect_uploaded_images_rejects_invalid_zip() -> None:
    with pytest.raises(ValueError, match="无法读取 ZIP"):
        collect_uploaded_images([FakeUpload("broken.zip", b"not-a-zip")])
