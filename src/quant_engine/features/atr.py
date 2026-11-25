"""Average True Range (Wilder)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(high-low, |high-prev close|, |low-prev close|); first candle uses high-low."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.fillna(h - l)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR. The first ``period - 1`` values are NaN (not enough history).

    The value on candle *i* uses candles <= *i*, so it is known at that candle's close.
    """
    tr = true_range(bars).to_numpy(dtype="float64")
    out = np.full(len(tr), np.nan)
    if len(tr) >= period:
        out[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=bars.index, name=f"atr{period}")
