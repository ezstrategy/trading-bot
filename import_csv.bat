@echo off
REM Import a CSV of gold candles: drag the CSV file onto this file, then answer the two questions.
REM The original file is copied unchanged; a new dataset version is created with a quality report.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
set FILE=%~1
if "%FILE%"=="" set /p FILE=Path of the CSV file: 
set /p TF=Timeframe (M1, M5 or M15): 
set /p TZ=Timezone of the times (UTC, mt5_server, +02:00 ...) [mt5_server]: 
if "%TZ%"=="" set TZ=mt5_server
python -m quant_engine data-import "%FILE%" --timeframe %TF% --tz %TZ% --source-name "%~nx1"
pause
