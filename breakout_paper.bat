@echo off
REM Double-click: paper signals for the long-only D1 breakout strategies (stored + dashboard only, nothing is
REM sent to Telegram and nothing is ever traded). Keep the window open; Ctrl+C stops it.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine breakout-run
pause
