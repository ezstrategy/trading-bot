@echo off
REM Double-click: EXP-015, one-shot test of the London breakout on untouched 2015-2019 data.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine london-oos --source mt5
echo.
echo Open the newest reports\london_oos_mt5_... folder and read report.md
pause
