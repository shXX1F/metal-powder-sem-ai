#!/usr/bin/env python3
"""Tiny standard-library web UI for the packaged inference app."""

from __future__ import annotations

import argparse
import cgi
import html
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANS_ROOT = Path(__file__).resolve().parents[1]
BACK_ROOT = ANS_ROOT / "data" / "back"
UPLOAD_ROOT = ANS_ROOT / "data" / "uploads"
VIEW_DIRS = [
    ("pic_final", "最终图"),
    ("hollow", "hollow"),
    ("particle", "particle"),
]


def safe_name(text: str) -> str:
    keep = []
    for ch in text.strip():
        keep.append(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_")
    return ("".join(keep).strip("._") or "run")[:80]


def latest_runs() -> list[Path]:
    if not BACK_ROOT.exists():
        return []
    return sorted([p for p in BACK_ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:20]


def result_href(path: Path) -> str:
    rel = html.escape(str(path.relative_to(BACK_ROOT)))
    return f"/result/{rel}"


def image_map(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {p.name: p for p in sorted(folder.iterdir(), key=lambda p: p.name) if p.suffix.lower() in IMAGE_EXTS}


class Handler(BaseHTTPRequestHandler):
    server_version = "HollowPowderHTTP/0.1"

    def send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/result/"):
            self.serve_result_file(parsed.path[len("/result/") :])
            return
        self.render_index()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/run":
            self.send_error(404)
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        upload = form["file"] if "file" in form else None
        if upload is None or not getattr(upload, "filename", ""):
            self.send_html("<p>没有收到文件。</p><p><a href='/'>返回</a></p>", 400)
            return

        run_name = safe_name(form.getfirst("name") or Path(upload.filename).stem)
        hollow_only = form.getfirst("hollow_only") == "on"
        reset = True
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        upload_dir = UPLOAD_ROOT / run_name
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True)

        uploaded_path = upload_dir / Path(upload.filename).name
        with uploaded_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        input_path = uploaded_path
        if uploaded_path.suffix.lower() == ".zip":
            extract_dir = upload_dir / "unzipped"
            extract_dir.mkdir()
            with zipfile.ZipFile(uploaded_path) as zf:
                for info in zf.infolist():
                    name = info.filename.replace("\\", "/")
                    parts = [p for p in Path(name).parts if p not in ("", ".", "/")]
                    if not parts:
                        continue
                    out = (extract_dir.joinpath(*parts)).resolve()
                    if not str(out).startswith(str(extract_dir.resolve())):
                        raise RuntimeError(f"unsafe zip path: {info.filename}")
                    if info.is_dir() or name.endswith("/"):
                        out.mkdir(parents=True, exist_ok=True)
                    else:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            input_path = extract_dir

        cmd = [
            sys.executable,
            str(ANS_ROOT / "code" / "run_report.py"),
            "--input",
            str(input_path),
            "--name",
            run_name,
            "--reset",
        ]
        if hollow_only:
            cmd.append("--hollow-only")

        log_path = upload_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(ANS_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            text = html.escape(log_path.read_text(encoding="utf-8", errors="replace")[-6000:])
            self.send_html(f"<h2>推理失败</h2><pre>{text}</pre><p><a href='/'>返回</a></p>", 500)
            return
        self.send_response(303)
        self.send_header("Location", f"/?run={run_name}")
        self.end_headers()

    def serve_result_file(self, rel: str) -> None:
        target = (BACK_ROOT / rel).resolve()
        if not str(target).startswith(str(BACK_ROOT.resolve())) or not target.is_file():
            self.send_error(404)
            return
        mime = "image/jpeg" if target.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        if target.suffix.lower() == ".txt":
            mime = "text/plain; charset=utf-8"
        elif target.suffix.lower() == ".json":
            mime = "application/json"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def render_index(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        selected = safe_name(query.get("run", [""])[0]) if query.get("run") else None
        runs = latest_runs()
        run = BACK_ROOT / selected if selected else (runs[0] if runs else None)
        summary = ""
        gallery = ""
        if run and run.exists():
            summary_path = run / "summary.txt"
            if summary_path.exists():
                summary = f"<pre>{html.escape(summary_path.read_text(encoding='utf-8', errors='replace'))}</pre>"
            maps = {dirname: image_map(run / dirname) for dirname, _label in VIEW_DIRS}
            names = sorted(set().union(*(paths.keys() for paths in maps.values())))[:24]
            rows = []
            for name in names:
                cells = []
                for dirname, label in VIEW_DIRS:
                    path = maps[dirname].get(name)
                    if path:
                        href = result_href(path)
                        cells.append(
                            f"<div class='view-card'><div class='view-title'>{html.escape(label)}</div>"
                            f"<a href='{href}' target='_blank'><img src='{href}' alt='{html.escape(label)} {html.escape(name)}'></a></div>"
                        )
                    else:
                        cells.append(
                            f"<div class='view-card missing'><div class='view-title'>{html.escape(label)}</div>"
                            f"<div class='placeholder'>暂无图片</div></div>"
                        )
                rows.append(f"<article class='image-row'><h3>{html.escape(name)}</h3><div class='views'>{''.join(cells)}</div></article>")
            gallery = "".join(rows)
        links = "".join(f"<li><a href='/?run={html.escape(p.name)}'>{html.escape(p.name)}</a></li>" for p in runs)
        self.send_html(
            f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Hollow Powder Inference</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; background: #f6f7f8; color: #111; }}
form, section {{ background: white; border: 1px solid #ddd; padding: 18px; margin-bottom: 18px; }}
input, button {{ font-size: 15px; padding: 8px; }}
button {{ cursor: pointer; }}
.image-row {{ border-top: 1px solid #e5e5e5; padding: 14px 0 18px; }}
.image-row:first-child {{ border-top: 0; }}
.image-row h3 {{ margin: 0 0 10px; font-size: 15px; }}
.views {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }}
.view-card {{ min-width: 0; }}
.view-title {{ font-size: 13px; font-weight: 700; margin-bottom: 6px; color: #333; }}
.view-card img {{ width: 100%; border: 1px solid #ccc; background: #222; display: block; }}
.placeholder {{ display: grid; place-items: center; height: 160px; border: 1px dashed #ccc; color: #777; background: #fafafa; }}
pre {{ white-space: pre-wrap; max-height: 360px; overflow: auto; }}
@media (max-width: 760px) {{ .views {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Hollow Powder Inference</h1>
<form method="post" action="/run" enctype="multipart/form-data">
  <p><input type="file" name="file" required> 上传单张图片或 zip 图片文件夹</p>
  <p><input name="name" placeholder="任务名，可选"> <label><input type="checkbox" name="hollow_only"> 只跑 hollow</label></p>
  <p><button type="submit">开始推理</button></p>
</form>
<section>
<h2>最近结果</h2>
<ul>{links}</ul>
</section>
<section>
<h2>当前结果</h2>
{summary}
<div class="gallery">{gallery}</div>
</section>
</body></html>"""
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
