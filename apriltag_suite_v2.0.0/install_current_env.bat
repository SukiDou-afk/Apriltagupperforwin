@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; print(sys.version); print(sys.executable)"
python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,8) else 1)"
if not errorlevel 1 goto py38
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,14) else 1)"
if not errorlevel 1 goto py314
echo Use Python 3.8 or 3.14, or follow README for another version.
goto fail
:py38
python -m pip install -r requirements-py38.txt
goto package
:py314
python -m pip install -r requirements-py314.txt
:package
if errorlevel 1 goto fail
python -m pip install --no-deps .
if errorlevel 1 goto fail
echo Installation completed. Run run.bat in this environment.
pause
exit /b 0
:fail
echo Installation failed. Read the error above.
pause
exit /b 1
