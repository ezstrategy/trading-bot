"""Small builders for hand-made candle sequences."""
from __future__ import annotations

import pandas as pd

from quant_engine.timeframes import Timeframe


def candles(ohlc: list[tuple[float, float, float, float]], start: str = "2026-11-23 08:00",
            tf: Timeframe = Timeframe.M5, spread: float = 20.0) -> pd.DataFrame:
    """Bars from (open, high, low, close) tuples, consecutive from ``start`` (UTC)."""
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(ohlc), freq=tf.pandas_freq)
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    df["tick_volume"], df["spread"], df["real_volume"] = 100.0, spread, 0.0
    return df


def from_closes(closes: list[float], wick: float = 0.2, **kw) -> pd.DataFrame:
    """Bars whose open is the previous close. Each candle has a wick of ``wick`` beyond
    its close only (so turning points make strict fractal highs/lows)."""
    rows, prev = [], closes[0]
    for c in closes:
        hi = c + wick if c >= prev else prev
        lo = c - wick if c <= prev else prev
        rows.append((prev, hi, lo, c))
        prev = c
    return candles(rows, **kw)


def highs_lows(highs: list[float], lows: list[float] | None = None, **kw) -> pd.DataFrame:
    """Bars with given highs (and lows); open/close sit inside."""
    lows = lows or [h - 1 for h in highs]
    rows = [(l + 0.25 * (h - l), h, l, l + 0.75 * (h - l)) for h, l in zip(highs, lows)]
    return candles(rows, **kw)
