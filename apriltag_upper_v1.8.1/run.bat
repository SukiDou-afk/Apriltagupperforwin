@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" apriltag_upper.py
) else (
    echo Local .venv was not found.
    echo Trying the current Python environment instead...
    python apriltag_upper.py
)

if errorlevel 1 (
    echo.
    echo Program exited with an error.
    pause
)
