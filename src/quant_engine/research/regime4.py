"""Four-state market regime on H4 (rulebook rule 4): bullish, bearish, sideways, uncertain.

Known at each H4 candle's close, from candles up to and including it (no look-ahead). Map it to
lower timeframes with :func:`quant_engine.research.events.regime_per_bar`, which only uses H4
candles that have closed.

* **bullish**: close above EMA(50), and the EMA rose by at least ``trend_slope_atr`` x ATR over
  ``slope_bars`` candles, on each of the last ``confirm_bars`` closes.
* **bearish**: the mirror.
* **sideways**: the EMA is flat (|change| below ``flat_slope_atr`` x ATR) and the last
  ``range_bars`` candles span at most ``range_max_atr`` x ATR.
* **uncertain**: everything else (transitions, choppy trends, wide flat markets). No engine trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Regime4Config
from ..features.atr import atr

BULL, BEAR, SIDE, UNCERTAIN = "bullish", "bearish", "sideways", "uncertain"
STATES = (BULL, BEAR, SIDE, UNCERTAIN)


def regime4_series(h4: pd.DataFrame, cfg: Regime4Config | None = None) -> pd.Series:
    """Regime label per H4 candle."""
    cfg = cfg or Regime4Config()
    c = h4["close"].astype("float64")
    a = atr(h4, 14)
    ema = c.ewm(span=cfg.ema_period, adjust=False, min_periods=cfg.ema_period).mean()
    slope = (ema - ema.shift(cfg.slope_bars)) / a
    up = ((c > ema) & (slope >= cfg.trend_slope_atr)).astype(float)
    dn = ((c < ema) & (slope <= -cfg.trend_slope_atr)).astype(float)
    k = cfg.confirm_bars
    bull = up.rolling(k, min_periods=k).sum() == k
    bear = dn.rolling(k, min_periods=k).sum() == k
    span = (h4["high"].rolling(cfg.range_bars, min_periods=cfg.range_bars).max()
            - h4["low"].rolling(cfg.range_bars, min_periods=cfg.range_bars).min()) / a
    side = (slope.abs() < cfg.flat_slope_atr) & (span <= cfg.range_max_atr)
    out = np.where(bull, BULL, np.where(bear, BEAR, np.where(side, SIDE, UNCERTAIN)))
    out = np.where(ema.isna() | slope.isna() | a.isna(), UNCERTAIN, out)
    return pd.Series(out, index=h4.index, name="regime4")
