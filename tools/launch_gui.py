from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def health_ok(port: int, timeout: float = 1.0) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200 and response.read().strip().lower() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def tail_text(path: Path, max_chars: int = 5000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def local_network_address(port: int) -> str | None:
    try:
        address = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    if not address or address.startswith("127."):
        return None
    return f"http://{address}:{port}"


def start_supervisor(
    *,
    project_root: Path,
    host: str,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(project_root / "tools" / "gui_supervisor.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    kwargs = {
        "cwd": str(project_root),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(command, **kwargs)
    finally:
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the metal-powder GUI reliably.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "gui_server.log"
    supervisor_pid_file = run_dir / "gui_supervisor.pid"
    host_file = run_dir / "gui_host.txt"
    stop_file = run_dir / "gui.stop"
    local_url = f"http://127.0.0.1:{args.port}"

    if health_ok(args.port):
        try:
            current_host = host_file.read_text(encoding="ascii").strip()
        except OSError:
            current_host = ""
        needs_lan_restart = args.host == "0.0.0.0" and current_host == "127.0.0.1"
        if not needs_lan_restart:
            print(f"GUI 已经在运行：{local_url}")
            if not args.no_browser:
                webbrowser.open(local_url)
            return 0

        print("当前服务仅允许本机访问，正在切换为局域网模式...")
        stop_file.write_text("stop\n", encoding="ascii")
        for _ in range(40):
            stopping_pid = read_pid(supervisor_pid_file)
            if (
                not health_ok(args.port)
                and not (stopping_pid and pid_alive(stopping_pid))
            ):
                break
            time.sleep(0.5)
        stopping_pid = read_pid(supervisor_pid_file)
        if health_ok(args.port) or (stopping_pid and pid_alive(stopping_pid)):
            print("现有服务未能停止，请先运行 stop_gui.bat。")
            return 1

    existing_pid = read_pid(supervisor_pid_file)
    if existing_pid and pid_alive(existing_pid):
        print(f"检测到 GUI 正在启动（pid={existing_pid}），继续等待服务就绪...")
        supervisor = None
    else:
        supervisor_pid_file.unlink(missing_ok=True)
        print("正在后台启动 GUI 服务...")
        supervisor = start_supervisor(
            project_root=project_root,
            host=args.host,
            port=args.port,
            log_path=log_path,
        )
        supervisor_pid_file.write_text(str(supervisor.pid), encoding="ascii")

    deadline = time.monotonic() + max(10, args.timeout)
    while time.monotonic() < deadline:
        if health_ok(args.port):
            print(f"GUI 已就绪：{local_url}")
            if args.host == "0.0.0.0":
                network_url = local_network_address(args.port)
                if network_url:
                    print(f"局域网地址：{network_url}")
            print(f"运行日志：{log_path}")
            if not args.no_browser:
                webbrowser.open(local_url)
            return 0
        if supervisor is not None and supervisor.poll() is not None:
            break
        time.sleep(1)

    print("GUI 未能在规定时间内启动。最新日志：")
    print(tail_text(log_path) or "没有生成日志。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
