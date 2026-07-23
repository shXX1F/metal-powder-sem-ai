from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def health_ok(port: int = 8501) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/_stcore/health",
            timeout=1,
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "gui.stop"
    supervisor_pid_file = run_dir / "gui_supervisor.pid"
    stop_file.write_text("stop\n", encoding="ascii")

    try:
        supervisor_pid = int(
            supervisor_pid_file.read_text(encoding="ascii").strip()
        )
    except (OSError, ValueError):
        supervisor_pid = None

    for _ in range(20):
        supervisor_running = bool(
            supervisor_pid and pid_alive(supervisor_pid)
        )
        if not supervisor_running and not health_ok():
            print("GUI 已停止。")
            return 0
        time.sleep(0.5)

    print("已发送停止请求，但服务仍在退出中；请稍后再次检查。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
