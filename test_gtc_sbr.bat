@echo off
REM Double-click: test the GTC-SBR strategy (session breakout and retest) on gold since 2020.
REM Experiments A-F, development / validation / out-of-sample split, cost stress, audit report. Signals only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine gtc --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\gtc_mt5_... folder and read report.md
pause
