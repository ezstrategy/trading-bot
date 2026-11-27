@echo off
REM Double-click: EXP-017, the unchanged London breakout on silver, EURUSD, GBPUSD, USDJPY (run once).
REM First download their history:  python -m quant_engine --instrument EURUSD download --timeframes M5,H4 --start 2015-01-01
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine lb-symbols
echo.
echo Open the newest reports\lb_symbols_... folder and read report.md
pause
