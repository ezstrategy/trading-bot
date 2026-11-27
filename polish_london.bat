@echo off
REM Double-click: test changes to the London breakout on gold since 2020 ($0.08 spread),
REM pick the best on older data and check it on unseen data. Signals only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine london --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\london_mt5_... folder and read report.md
pause
