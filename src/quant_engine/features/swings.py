"""Confirmed fractal swing points.

Candle *i* is a swing high when its high is strictly greater than the highs
of the L candles before AND the L candles after it (swing low: mirror).
The swing is only known once candle *i + L* has closed, so every swing
carries ``confirmed_idx`` (= i + L) and ``known_at`` (close time of that
candle). Nothing may use a swing before ``known_at``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeframes import Timeframe

SWING_COLUMNS = ["idx", "time", "kind", "price", "confirmed_idx", "known_at"]


def _strict_extreme(values: np.ndarray, length: int, high: bool) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n < 2 * length + 1:
        return out
    for i in range(length, n - length):
        v = values[i]
        left = values[i - length:i]
        right = values[i + 1:i + 1 + length]
        if high:
            out[i] = v > left.max() and v > right.max()
        else:
            out[i] = v < left.min() and v < right.min()
    return out


def find_swings(bars: pd.DataFrame, tf: Timeframe, length: int) -> pd.DataFrame:
    """All confirmed swings in ``bars``, sorted by the time they became known.

    Returns columns: idx (row of the swing candle), time (its open time, UTC),
    kind ('high'/'low'), price, confirmed_idx, known_at (UTC).
    """
    if bars.empty:
        return pd.DataFrame(columns=SWING_COLUMNS)
    t = pd.DatetimeIndex(bars["time"])
    rows = []
    for kind, col, is_high in (("high", "high", True), ("low", "low", False)):
        mask = _strict_extreme(bars[col].to_numpy(dtype="float64"), length, is_high)
        for i in np.flatnonzero(mask):
            rows.append((int(i), t[i], kind, float(bars[col].iat[i]), int(i + length),
                         t[i + length] + tf.delta))
    df = pd.DataFrame(rows, columns=SWING_COLUMNS)
    return df.sort_values(["confirmed_idx", "idx"], kind="stable").reset_index(drop=True)


def swings_known_at(swings: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """Swings that are confirmed (known) at instant ``when``."""
    return swings[swings["known_at"] <= when]
