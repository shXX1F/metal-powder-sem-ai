$ErrorActionPreference = "Stop"

$AppName = "MetalPowderSEM_GUI"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistRoot = Join-Path $ProjectRoot "dist"
$ReleaseDir = Join-Path $DistRoot $AppName
$RuntimePython = Join-Path $ReleaseDir ".runtime\Scripts\python.exe"
$ZipPath = Join-Path $DistRoot "$AppName.zip"

Write-Host "Building portable release: $AppName"

if (Test-Path $ReleaseDir) {
    Remove-Item -Recurse -Force $ReleaseDir
}

New-Item -ItemType Directory -Force $ReleaseDir | Out-Null

Copy-Item (Join-Path $ProjectRoot "app_streamlit.py") $ReleaseDir
Copy-Item (Join-Path $ProjectRoot "requirements.txt") $ReleaseDir
Copy-Item (Join-Path $ProjectRoot "start_gui.bat") $ReleaseDir
Copy-Item (Join-Path $ProjectRoot "RELEASE_README.md") $ReleaseDir
Copy-Item (Join-Path $ProjectRoot "metal_powder_sem_ai") (Join-Path $ReleaseDir "metal_powder_sem_ai") -Recurse

python -m venv (Join-Path $ReleaseDir ".runtime")
& $RuntimePython -m pip install --upgrade pip
& $RuntimePython -m pip install -r (Join-Path $ReleaseDir "requirements.txt")

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Compress-Archive -Path (Join-Path $ReleaseDir "*") -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Done."
Write-Host "Release folder: $ReleaseDir"
Write-Host "Release zip:    $ZipPath"
Write-Host ""
Write-Host "Send the zip file to users. They only need to unzip it and double-click start_gui.bat."
