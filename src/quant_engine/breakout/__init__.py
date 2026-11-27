"""Long-only XAUUSD daily breakout module (Strategy A: previous month high, Strategy B: 20-day channel).

Signals only. The package is independent from the London breakout engine and the Quant Lab
plug-ins; it shares the instrument specification (:mod:`quant_engine.instruments`), the MT5
data layer and the web app. See ``docs/BREAKOUT_ARCHITECTURE.md``.
"""
