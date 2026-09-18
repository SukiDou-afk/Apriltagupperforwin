@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo AprilTag Upper Computer v0.8 - Current Python Setup
echo ============================================
echo This installs packages into the Python environment currently active.
echo If your prompt shows ^(base^), this means the Anaconda base environment.
echo.

where python >nul 2>nul
if errorlevel 1 goto :no_python
python --version
if errorlevel 1 goto :no_python

echo.
echo [1/2] Upgrading pip ...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [2/2] Installing packages ...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Setup completed.
echo Run: python apriltag_upper.py
pause
exit /b 0

:no_python
echo ERROR: Python was not found.
pause
exit /b 1

:fail
echo.
echo ERROR: Package installation failed.
echo Copy the last 20 lines and send them to ChatGPT.
pause
exit /b 1
