@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u workpc\run_full_doe.py
echo.
echo Finished or interrupted. Completed conditions are saved in workpc_results.
pause
