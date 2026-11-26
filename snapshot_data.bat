@echo off
REM Freeze the MT5 gold candles you already downloaded (M1, M5, M15) as new, never-modified dataset versions.
REM Nothing is downloaded, changed or deleted. Signals and research only.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine data-snapshot --timeframes M1,M5,M15
echo.
python -m quant_engine data-list
pause
