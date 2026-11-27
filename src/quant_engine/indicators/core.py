"""EMA, EMA slope, ATR, opening range, session labels and the EMA trend state."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.atr import atr as _atr
from ..research.metrics import SESSION_ORDER, session_of as _session_of


def ema(close: pd.Series | np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (span ``period``, seeded at the first value, NaN for the first
    ``period - 1`` candles)."""
    s = pd.Series(np.asarray(close, dtype="float64"))
    return s.ewm(span=period, adjust=False, min_periods=period).mean().to_numpy()


def ema_slope(values: np.ndarray, bars: int) -> np.ndarray:
    """Change of a series over ``bars`` candles (value[i] - value[i - bars]); NaN where unknown."""
    v = np.asarray(values, dtype="float64")
    out = np.full(len(v), np.nan)
    if bars < len(v):
        out[bars:] = v[bars:] - v[:-bars]
    return out


def atr(bars: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR (see :func:`quant_engine.features.atr.atr`)."""
    return _atr(bars, period).to_numpy(dtype="float64")


def trend_state(close: np.ndarray, fast: np.ndarray, slow: np.ndarray, slope_bars: int) -> np.ndarray:
    """``bullish`` when fast > slow, close > fast and both EMAs rose over ``slope_bars`` candles;
    ``bearish`` for the mirror; ``sideways`` otherwise (including warm-up)."""
    c = np.asarray(close, dtype="float64")
    sf, ss = ema_slope(fast, slope_bars), ema_slope(slow, slope_bars)
    with np.errstate(invalid="ignore"):
        bull = (fast > slow) & (c > fast) & (sf > 0) & (ss > 0)
        bear = (fast < slow) & (c < fast) & (sf < 0) & (ss < 0)
    return np.where(bull, "bullish", np.where(bear, "bearish", "sideways")).astype(object)


def opening_range(times: pd.DatetimeIndex, high: np.ndarray, low: np.ndarray, tz: str, start_minute: int,
                  minutes: int, tf_minutes: int = 5) -> pd.DataFrame:
    """Opening ranges: one row per local trading date whose window is complete in the data.

    The window is ``[start_minute, start_minute + minutes)`` in local time ``tz`` (so daylight
    saving is handled). Columns: ``date``, ``first_idx``, ``last_idx``, ``high``, ``low``,
    ``complete`` (all ``minutes / tf_minutes`` candles present and contiguous). The range is
    known at the close of ``last_idx``.
    """
    t = pd.DatetimeIndex(times)
    loc = t.tz_convert(tz)
    lmin = np.asarray(loc.hour * 60 + loc.minute)
    ldate = np.asarray(loc.normalize().tz_localize(None))
    inside = (lmin >= start_minute) & (lmin < start_minute + minutes)
    need = minutes // tf_minutes
    rows = []
    for d in np.unique(ldate[inside]):
        idx = np.flatnonzero(inside & (ldate == d))
        contiguous = len(idx) == need and (t[idx[-1]] - t[idx[0]]) == pd.Timedelta(minutes=tf_minutes * (need - 1))
        rows.append({"date": pd.Timestamp(d), "first_idx": int(idx[0]), "last_idx": int(idx[-1]),
                     "high": float(np.max(high[idx])), "low": float(np.min(low[idx])), "complete": bool(contiguous)})
    return pd.DataFrame(rows, columns=["date", "first_idx", "last_idx", "high", "low", "complete"])


def session_of(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Rulebook rule-11 session per UTC time: asia, london, london_ny_overlap, new_york, off
    (New York clock, so daylight saving is handled)."""
    return _session_of(times)


SESSIONS = SESSION_ORDER
