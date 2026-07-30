from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from app_streamlit import uploaded_items_signature
from metal_powder_sem_ai.gui_preview import UploadedImageItem


APP_PATH = Path(__file__).resolve().parents[1] / "app_streamlit.py"


def test_all_download_buttons_do_not_rerun_the_page() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    download_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "download_button"
    ]

    assert download_calls
    for call in download_calls:
        on_click = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "on_click"),
            None,
        )
        assert isinstance(on_click, ast.Constant)
        assert on_click.value == "ignore"


def png_bytes(width: int = 48, height: int = 32, value: int = 160) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_sem_page_previews_multiple_uploaded_images() -> None:
    image_data = png_bytes()
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app.file_uploader[0].upload(
        "first.png", image_data, "image/png"
    ).upload(
        "second.png", image_data, "image/png"
    ).run()

    assert not app.exception
    assert any(element.label == "逐张查看（共 2 张）" for element in app.selectbox)
    captions = [element.value for element in app.caption]
    assert any("已从 2 个上传项目中载入 2 张图像" in text for text in captions)
    assert any("文件：first.png" in text and "48 × 32 px" in text for text in captions)


def test_hollow_result_keeps_original_next_to_result(
    tmp_path: Path,
) -> None:
    image_data = png_bytes(width=36, height=28, value=100)
    output_dir = tmp_path / "result"
    for folder in ("pic_final", "hollow", "particle"):
        target_dir = output_dir / folder
        target_dir.mkdir(parents=True)
        (target_dir / "sample.png").write_bytes(image_data)

    payload = {
        "summary": {
            "image_count": 1,
            "particle_count": 12,
            "hollow_candidate_count": 3,
            "hollow_powder_count": 2,
            "hollow_powder_ratio": 2 / 12,
            "hollow_ratio_threshold": 0.25,
        },
        "per_image": [
            {
                "image": "sample.png",
                "particle_count": 12,
                "particle_boundary_filtered_count": 0,
                "particle_dark_filtered_count": 0,
                "hollow_candidate_count": 3,
                "hollow_powder_count": 2,
                "hollow_powder_ratio": 2 / 12,
            }
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    item = UploadedImageItem(
        display_name="sample.png",
        saved_name="sample.png",
        data=image_data,
        source_name="sample.png",
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio[0].set_value("空心粉图像识别").run()
    app.file_uploader[0].upload("sample.png", image_data, "image/png").run()
    app.session_state["hollow_last_output_dir"] = str(output_dir)
    app.session_state["hollow_last_upload_signature"] = (
        uploaded_items_signature([item])
    )
    app.run()

    assert not app.exception
    subheaders = [element.value for element in app.subheader]
    assert "原图与识别结果" in subheaders
    assert "待识别原图" in subheaders
    assert "空心粉识别结果" in subheaders
