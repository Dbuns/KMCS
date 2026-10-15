@echo off
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" workpc\run_full_doe.py --analyse-only
echo Open workpc_results\corrected\analysis\REPORT.md for the report and top views.
pause
