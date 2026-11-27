@echo off
REM Double-click: EXP-020, natural stop first + 2% sizing: how many signals are skipped (run once).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine lb-natural --source mt5
echo.
echo Open the newest reports\lb_natural_mt5_... folder and read report.md
pause
