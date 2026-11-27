@echo off
REM Double-click: EXP-016, Phase 2 research on the London Asian-range breakout (run once).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine lb-phase2 --source mt5
echo.
echo Open the newest reports\lb_phase2_mt5_... folder and read report.md
pause
