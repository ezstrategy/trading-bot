@echo off
REM Double-click: EXP-018, 3-part scale-out exit (T1 / T2 / T3) on the gold London breakout (run once).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine lb-scaleout --source mt5
echo.
echo Open the newest reports\lb_scaleout_mt5_... folder and read report.md
pause
