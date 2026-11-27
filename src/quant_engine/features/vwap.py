"""Daily VWAP with standard-deviation bands (resets at the trading-day start, 17:00 New York).

Uses tick volume (MT5 has no real volume for gold CFDs). The value at candle ``i`` uses
candles up to and including ``i``, so it is known at that candle's close (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SessionsConfig
from .sessions import trading_day


def daily_vwap(bars: pd.DataFrame, cfg: SessionsConfig) -> tuple[np.ndarray, np.ndarray]:
    """(vwap, sd) per candle: typical price (H+L+C)/3 weighted by tick volume, and the
    volume-weighted standard deviation of typical price around it, both since the day started."""
    tp = ((bars["high"] + bars["low"] + bars["close"]) / 3.0).to_numpy(dtype="float64")
    vol = bars["tick_volume"].to_numpy(dtype="float64") if "tick_volume" in bars else np.ones(len(bars))
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)    # GUESS: a candle with no volume counts as 1
    day = trading_day(bars["time"], cfg)
    g = pd.Series(day).factorize()[0]
    df = pd.DataFrame({"g": g, "pv": tp * vol, "v": vol, "p2v": tp * tp * vol})
    cs = df.groupby("g")[["pv", "v", "p2v"]].cumsum()
    vwap = (cs["pv"] / cs["v"]).to_numpy()
    var = np.maximum(cs["p2v"].to_numpy() / cs["v"].to_numpy() - vwap ** 2, 0.0)
    return vwap, np.sqrt(var)
