@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo AprilTag Upper Computer v0.8 - Setup
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 goto :no_python

echo Python found:
python --version
if errorlevel 1 goto :no_python

echo.
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating local virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 goto :venv_fail
) else (
    echo [1/3] Existing .venv found.
)

echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_fail

echo [3/3] Installing packages ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_fail

echo.
echo ============================================
echo Setup completed successfully.
echo Double-click run.bat to start the program.
echo ============================================
pause
exit /b 0

:no_python
echo.
echo ERROR: Python was not found in PATH.
echo Open Anaconda Prompt in this folder and run install_current_env.bat,
echo or add Python to PATH.
pause
exit /b 1

:venv_fail
echo.
echo ERROR: Could not create .venv.
echo Since you are using Anaconda, try install_current_env.bat instead.
pause
exit /b 1

:install_fail
echo.
echo ERROR: Package installation failed.
echo Copy the last 20 lines of this window and send them to ChatGPT.
pause
exit /b 1
