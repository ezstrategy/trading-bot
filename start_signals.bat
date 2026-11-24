@echo off
REM Double-click: start the live XAUUSD signal engine (signals to Telegram, never places trades).
REM Needs MT5 open and logged in, and TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m quant_engine telegram-test
python -m quant_engine run
pause
