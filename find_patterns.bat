@echo off
REM Double-click: find repeatable gold patterns per market type (bullish / bearish / sideways)
REM in your MT5 history, then test them on newer data they never saw. Signals only, never trades.
REM Needs MetaTrader 5 open and logged in (it downloads the last 365 days first).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine download --days 365
python -m quant_engine patterns --source mt5
echo.
echo Open the newest reports\patterns_mt5_... folder and read report.md
pause
