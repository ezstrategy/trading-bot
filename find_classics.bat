@echo off
REM Double-click: backtest well-known public gold strategies since 2020 with a $0.08 spread. Signals only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine classics --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\classics_mt5_... folder and read report.md
pause
