@echo off
REM Double-click to open the XAUUSD Quant Engine dashboard (signals only).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -c "import fastapi, uvicorn, markdown" 2>nul || python -m pip install -r requirements.txt
python -m quant_engine dashboard
pause
