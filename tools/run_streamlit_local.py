from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if sys.stdout is None or sys.stderr is None:
        log_dir = project_root / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if sys.stdout is None:
            sys.stdout = open(
                log_dir / "streamlit_stdout.log",
                "a",
                encoding="utf-8",
                buffering=1,
            )
        if sys.stderr is None:
            sys.stderr = open(
                log_dir / "streamlit_stderr.log",
                "a",
                encoding="utf-8",
                buffering=1,
            )

    deps_dirs = [Path(r"D:\CodexDeps\sem_runtime"), Path(r"D:\CodexDeps\sem_labelme"), project_root / ".deps"]
    for deps_dir in reversed(deps_dirs):
        if deps_dir.exists():
            sys.path.insert(0, str(deps_dir))

    default_args = [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.maxUploadSize",
        "1024",
        "--server.headless",
        "true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats",
        "false",
    ]
    extra_args = sys.argv[1:] or default_args
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "app_streamlit.py"),
        *extra_args,
    ]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()


