@echo off
REM Double-click: London breakout versions aimed at 50%+ wins (small targets, TP1 half close, early break-even).
REM Chosen on older data, checked on unseen data, $0.08 spread. Signals only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine london --source mt5 --start 2020-01-01 --grid high-win --min-win 50
echo.
echo Open the newest reports\london_mt5_... folder and read report.md
pause
