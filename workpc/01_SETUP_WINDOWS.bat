@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Python 3.12 is needed. Ask IT to provide it, then run this file again.
  pause
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install -r requirements-tested.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" workpc\run_full_doe.py --check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" workpc\run_full_doe.py --smoke-test
if errorlevel 1 goto failed
echo Setup and tiny plumbing check finished. Full DOE has NOT started.
pause
exit /b 0
:failed
echo Setup/check failed. Read the error above before starting the full DOE.
pause
exit /b 1
