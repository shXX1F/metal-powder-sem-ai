@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "APP_PORT=8501"
set "APP_URL=http://127.0.0.1:%APP_PORT%"
set "LOCAL_PYTHON=%~dp0.runtime\Scripts\python.exe"
set "CONDA_PYTHON=D:\anaconda3x\envs\sdsd_torch\python.exe"

if exist "%LOCAL_PYTHON%" (
    set "PYTHON_EXE=%LOCAL_PYTHON%"
) else if exist "%CONDA_PYTHON%" (
    set "PYTHON_EXE=%CONDA_PYTHON%"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Could not find bundled Python or system Python.
        echo If you are a user, please download the full release zip that contains the .runtime folder.
        echo If you are a developer, run build_release.ps1 first.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo Starting Metal Powder SEM GUI...
echo Local address: %APP_URL%
start "" "%APP_URL%"

"%PYTHON_EXE%" tools\run_streamlit_local.py ^
  --server.address 127.0.0.1 ^
  --server.port %APP_PORT% ^
  --server.maxUploadSize 1024 ^
  --server.headless true ^
  --global.developmentMode false ^
  --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo The GUI failed to start. Please check whether all files are present.
    pause
    exit /b 1
)

endlocal
