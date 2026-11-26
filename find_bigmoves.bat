@echo off
REM Double-click: find 30-50 pip gold moves within 30 minutes since 2020, what came before them,
REM and test a simple support/resistance + price action + liquidity strategy. Signals only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine bigmoves --source mt5 --start 2020-01-01
echo.
echo Open the newest reports\bigmoves_mt5_... folder and read report.md
pause
