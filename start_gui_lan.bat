@echo off
setlocal EnableExtensions
title Metal Powder GUI LAN Launcher

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    echo Project: %~dp0
    pause
    exit /b 1
)

set "LOCAL_PYTHON=%~dp0.runtime\Scripts\python.exe"
if exist "%LOCAL_PYTHON%" (
    set "PYTHON_EXE=%LOCAL_PYTHON%"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found.
        echo Install the full release package or configure Python first.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

if not exist "%~dp0tools\launch_gui.py" (
    echo ERROR: tools\launch_gui.py is missing.
    pause
    exit /b 1
)

echo Starting Metal Powder GUI in LAN mode...
"%PYTHON_EXE%" "%~dp0tools\launch_gui.py" --host 0.0.0.0 --port 8501 %*
if errorlevel 1 (
    echo.
    echo ERROR: GUI startup failed.
    echo Log: %~dp0runs\gui_server.log
    pause
    exit /b 1
)
echo.
echo LAN access also requires Windows Firewall to allow TCP port 8501.
pause

endlocal
