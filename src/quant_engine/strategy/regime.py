"""Market regime labels used to break backtest results down.

Two independent views, both known at the signal time (closed candles only):

* ``trend_regime``: bullish when the last closed H4 candle is above its
  EMA(50) and the EMA rose over the last 10 H4 candles by at least
  0.5 x H4 ATR(14); bearish for the mirror; neutral otherwise (including a
  flat EMA). This does not use the strategy's own logic.
  GUESS: the 0.5 ATR slope threshold; it is configurable in ``backtest:``.
* ``h4_structure``: the strategy's H4 structure state (bullish / bearish / ranging).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.atr import atr
from ..timeframes import Timeframe

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"


def trend_regime_series(h4: pd.DataFrame, ema_period: int = 50, slope_bars: int = 10,
                        min_slope_atr: float = 0.5, atr_period: int = 14) -> pd.Series:
    """Regime per H4 candle (known at that candle's close; uses candles <= it only)."""
    c = h4["close"].astype("float64")
    ema = c.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    slope = ema - ema.shift(slope_bars)
    need = min_slope_atr * atr(h4, atr_period)
    out = np.where((c > ema) & (slope >= need), BULLISH,
                   np.where((c < ema) & (slope <= -need), BEARISH, NEUTRAL))
    out = np.where(ema.isna() | slope.isna() | need.isna(), NEUTRAL, out)
    return pd.Series(out, index=h4.index, name="trend_regime")


def regime_at(h4: pd.DataFrame, regimes: pd.Series, when: pd.Timestamp) -> str:
    """Regime of the last H4 candle closed at ``when``."""
    closes = pd.DatetimeIndex(h4["time"]) + Timeframe.H4.delta
    k = int(closes.searchsorted(when, side="right")) - 1
    return str(regimes.iat[k]) if k >= 0 else NEUTRAL
