@echo off
setlocal EnableExtensions
title Stop Metal Powder GUI

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
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
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0tools\stop_gui.py"
if errorlevel 1 pause

endlocal
