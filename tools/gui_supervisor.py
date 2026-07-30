from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from gui_source_state import source_signature


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def health_ok(port: int, timeout: float = 3.0) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200 and response.read().strip().lower() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def terminate_process(process: subprocess.Popen, timeout: float = 12.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_streamlit_command(project_root: Path, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        str(project_root / "tools" / "run_streamlit_local.py"),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        "1024",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.runOnSave",
        "false",
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the Streamlit GUI process healthy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--health-failures", type=int, default=6)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "gui.stop"
    child_pid_file = run_dir / "gui_streamlit.pid"
    supervisor_pid_file = run_dir / "gui_supervisor.pid"
    host_file = run_dir / "gui_host.txt"
    signature_file = run_dir / "gui_source_signature.txt"
    stop_file.unlink(missing_ok=True)
    supervisor_pid_file.write_text(str(os.getpid()), encoding="ascii")
    host_file.write_text(args.host, encoding="ascii")
    signature_file.write_text(
        source_signature(project_root),
        encoding="ascii",
    )

    stop_requested = False
    child: subprocess.Popen | None = None

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    short_failure_count = 0
    command = build_streamlit_command(project_root, args.host, args.port)
    log(f"Supervisor started, pid={os.getpid()}, host={args.host}, port={args.port}")

    try:
        while not stop_requested and not stop_file.exists():
            started_at = time.monotonic()
            log("Starting Streamlit...")
            child = subprocess.Popen(
                command,
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
            )
            child_pid_file.write_text(str(child.pid), encoding="ascii")
            log(f"Streamlit started, pid={child.pid}")

            consecutive_health_failures = 0
            startup_grace_until = time.monotonic() + 60.0
            while child.poll() is None:
                if stop_requested or stop_file.exists():
                    log("Stop requested.")
                    terminate_process(child)
                    break

                time.sleep(3)
                if time.monotonic() < startup_grace_until:
                    continue
                if health_ok(args.port):
                    consecutive_health_failures = 0
                else:
                    consecutive_health_failures += 1
                    log(
                        "Health check failed "
                        f"({consecutive_health_failures}/{args.health_failures})."
                    )
                    if consecutive_health_failures >= args.health_failures:
                        log("Streamlit is unhealthy; restarting it.")
                        terminate_process(child)
                        break

            if stop_requested or stop_file.exists():
                break

            exit_code = child.poll()
            runtime_seconds = time.monotonic() - started_at
            log(f"Streamlit exited with code {exit_code} after {runtime_seconds:.1f}s.")
            if runtime_seconds < 15:
                short_failure_count += 1
            else:
                short_failure_count = 0
            if short_failure_count >= 5:
                log("Stopped after five consecutive startup failures; inspect this log.")
                return 1
            time.sleep(3)
    finally:
        if child is not None and child.poll() is None:
            terminate_process(child)
        child_pid_file.unlink(missing_ok=True)
        supervisor_pid_file.unlink(missing_ok=True)
        host_file.unlink(missing_ok=True)
        signature_file.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)
        log("Supervisor stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
