@echo off
REM Double-click: backtest your entry (sweep of a 30-minute low/high + reversal candle) on gold since 2020.
REM Spread $0.08. Signals only, never trades.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine entries --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\entries_mt5_... folder and read report.md
pause
