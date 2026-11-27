@echo off
REM Double-click: EXP-021, reproduce the long-only D1 breakout results (Strategy A + B) on the MT5 M5 history.
REM Run once. First refresh the broker spec (swap!) with: python -m quant_engine mt5-instrument --profile my_broker
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine breakout-backtest %*
echo.
echo Open the newest reports\breakout\... folder and read report.md (or the Breakout page of the dashboard).
pause
