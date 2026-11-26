@echo off
REM Quant Lab, step 1 on your PC: download gold history from MT5 (M1, M5, M15 since 2020) and freeze it as
REM never-modified dataset versions, then read the brokers XAUUSD specification into a profile.
REM Reads data only. Never places, changes or closes a trade.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine mt5-import --timeframes M1,M5,M15 --start 2020-01-01
python -m quant_engine mt5-instrument --profile my_broker
python -m quant_engine data-list
pause
