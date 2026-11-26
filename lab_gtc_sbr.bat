@echo off
REM Quant Lab, step 2: GTC-SBR bullish + bearish on development data, then validation data, then the
REM walk-forward / sensitivity / cost-stress check. The untouched test period is NOT used here.
REM Every run is registered (python -m quant_engine lab-list) and can be audited (lab-audit LAB-xxxx).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set PYTHONPATH=src
set P=gtc_sbr_2020
python -m quant_engine lab-run --programme %P% --part development --title "GTC-SBR v1.0.0 development" --hypothesis "A London / New York opening-range breakout that is retested and agrees with the M15 trend continues far enough to pay 1.5R after costs"
python -m quant_engine lab-run --programme %P% --part validation --title "GTC-SBR v1.0.0 validation"
python -m quant_engine lab-validate --programme %P% --neighbours "{\"breakout_atr\": [0.05, 0.1, 0.15], \"retest_tol_atr\": [0.1, 0.15, 0.2], \"sl_buffer_atr\": [0.15, 0.2, 0.25], \"time_exit_bars\": [9, 12, 15]}"
python -m quant_engine lab-list
echo.
echo Open the dashboard (start_dashboard.bat), tab "Quant Lab", to see the equity curves, trades and reports.
pause
