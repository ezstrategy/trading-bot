@echo off
REM Double-click: describe how gold behaves in bullish / bearish / sideways / uncertain markets (development data).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine study --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\study_mt5_... folder and read report.md
pause
