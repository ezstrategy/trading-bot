"""Synthetic XAUUSD-like M1 history for backtest tests (no real data needed).

A seeded random walk with regime drift (up, down, flat blocks), higher
volatility in the London/New York kill zones and a weekend gap, so every
code path of the entry model and backtester gets exercised. It says nothing
about how the strategy performs on real gold prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.data.dukascopy import resample
from quant_engine.timeframes import Timeframe

TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4)


def synthetic_m1(start: str = "2026-06-01", days: int = 60, seed: int = 7,
                 price: float = 2400.0, block_days: int = 10) -> pd.DataFrame:
    """Seeded M1 bars (UTC) with drift that switches every ``block_days`` days:
    up, down, flat, repeated. Weekend (Fri 21:00 - Sun 22:00 UTC) is closed."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 1440, freq="1min")
    dow, hour = t.dayofweek, t.hour
    open_ = ~(((dow == 4) & (hour >= 21)) | (dow == 5) | ((dow == 6) & (hour < 22)))
    t = t[open_]
    n = len(t)
    block = ((t - t[0]).days // block_days) % 3
    drift = np.select([block == 0, block == 1], [0.012, -0.012], 0.0)
    hrs = t.hour.to_numpy()
    vol = np.where(((hrs >= 7) & (hrs < 10)) | ((hrs >= 12) & (hrs < 16)), 0.55, 0.25)
    # occasional bursts make displacement candles and sweeps
    burst = rng.random(n) < 0.004
    steps = drift + vol * rng.standard_normal(n) + burst * rng.choice([-1, 1], n) * 2.5
    close = price + np.cumsum(steps)
    open_px = np.concatenate([[price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.15, n))
    high = np.maximum(open_px, close) + wick
    low = np.minimum(open_px, close) - np.abs(rng.normal(0, 0.15, n))
    return pd.DataFrame({"time": t, "open": open_px.round(2), "high": high.round(2),
                         "low": low.round(2), "close": close.round(2),
                         "tick_volume": 50.0, "spread": 30.0, "real_volume": 0.0})


def synthetic_bars(**kw) -> dict[Timeframe, pd.DataFrame]:
    """M5/M15/H1/H4 bars built from :func:`synthetic_m1`."""
    m1 = synthetic_m1(**kw)
    return {tf: resample(m1, tf) for tf in TFS}
