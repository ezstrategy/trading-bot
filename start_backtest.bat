@echo off
REM Double-click: download free Dukascopy XAUUSD history (last 180 days) and run the backtest.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine dukascopy --days 180
python -m quant_engine backtest --source dukascopy
pause
