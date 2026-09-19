@echo off
setlocal
cd /d "%~dp0"
python apriltag_upper.py
if errorlevel 1 pause
