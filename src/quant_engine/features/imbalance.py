"""Displacement candles, fair value gaps (FVG) and order blocks (OB)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeframes import Timeframe

FVG_COLUMNS = ["c3_idx", "time", "known_at", "direction", "bottom", "top", "ce", "size",
               "valid_size", "invalidated_at"]


def displacement(bars: pd.DataFrame, atr: pd.Series, min_body_atr: float = 1.2,
                 min_body_ratio: float = 0.6) -> pd.Series:
    """+1 / -1 for a bullish / bearish displacement candle, 0 otherwise.

    Displacement: |close - open| >= min_body_atr x ATR and body >= min_body_ratio x range.
    Uses the ATR of the same candle (known at its close).
    """
    o, h, l, c = (bars[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    body = np.abs(c - o)
    rng = h - l
    a = atr.to_numpy(dtype="float64")
    ok = (body >= min_body_atr * a) & (body >= min_body_ratio * np.where(rng > 0, rng, np.inf)) & ~np.isnan(a)
    return pd.Series(np.where(ok, np.sign(c - o), 0).astype(int), index=bars.index, name="displacement")


def _first_close_beyond(c: np.ndarray, start: int, level: float, direction: int,
                        chunk: int = 256) -> int | None:
    """Index of the first close below ``level`` (direction +1) / above it (-1) from ``start``."""
    for a in range(start, len(c), chunk):
        seg = c[a:a + chunk]
        hit = np.flatnonzero(seg < level if direction == 1 else seg > level)
        if len(hit):
            return a + int(hit[0])
    return None


def find_fvgs(bars: pd.DataFrame, tf: Timeframe, atr: pd.Series, point: float,
              min_size_atr: float = 0.25, min_size_spreads: float = 3.0) -> pd.DataFrame:
    """All fair value gaps, with their size check and invalidation time.

    Bullish FVG on candles (c1, c2, c3): low(c3) > high(c1); gap = [high(c1), low(c3)].
    Bearish: high(c3) < low(c1); gap = [high(c3), low(c1)].
    ``valid_size``: gap >= min_size_atr x ATR(c3) and >= min_size_spreads x spread(c3).
    ``invalidated_at``: close time of the first later candle that CLOSES beyond the
    gap's far side (below bottom for bullish, above top for bearish), else NaT.
    ``known_at`` is the close of c3.
    """
    n = len(bars)
    if n < 3:
        return pd.DataFrame(columns=FVG_COLUMNS)
    h, l, c = (bars[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
    spread = bars["spread"].to_numpy(dtype="float64") * point if "spread" in bars else np.zeros(n)
    a = atr.to_numpy(dtype="float64")
    t = pd.DatetimeIndex(bars["time"])
    rows = []
    for m in range(2, n):
        for direction, bottom, top in ((1, h[m - 2], l[m]), (-1, h[m], l[m - 2])):
            if top <= bottom:
                continue
            size = top - bottom
            valid = (not np.isnan(a[m]) and size >= min_size_atr * a[m]
                     and size >= min_size_spreads * spread[m])
            k = _first_close_beyond(c, m + 1, bottom if direction == 1 else top, direction)
            inv = t[k] + tf.delta if k is not None else pd.NaT
            rows.append((m, t[m], t[m] + tf.delta, direction, bottom, top, (bottom + top) / 2, size,
                         bool(valid), inv))
    return pd.DataFrame(rows, columns=FVG_COLUMNS)


def order_block(bars: pd.DataFrame, leg_start_idx: int, direction: int,
                max_lookback: int = 10) -> tuple[float, float, int] | None:
    """Order block before a displacement leg that starts at ``leg_start_idx``.

    Bullish: the last down-close candle (close < open) before the leg; zone = [low, open].
    Bearish: the last up-close candle before the leg; zone = [open, high].
    Returns (zone_low, zone_high, candle_idx) or None.
    """
    o, h, l, c = (bars[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    for i in range(leg_start_idx - 1, max(leg_start_idx - 1 - max_lookback, -1), -1):
        if direction == 1 and c[i] < o[i]:
            return float(l[i]), float(o[i]), i
        if direction == -1 and c[i] > o[i]:
            return float(o[i]), float(h[i]), i
    return None
