"""Closed-candle rules (the main guard against look-ahead bias).

A candle that opened at ``t`` on timeframe ``tf`` is only final at
``t + tf``. Anything that decides a signal must only see candles whose
close time is <= the decision time.
"""
from __future__ import annotations

import pandas as pd

from ..timeframes import Timeframe


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def is_closed(open_time: pd.Timestamp, tf: Timeframe, now_utc: pd.Timestamp) -> bool:
    """True if the candle opened at ``open_time`` has fully closed by ``now_utc``."""
    return _utc(open_time) + tf.delta <= _utc(now_utc)


def closed_bars(bars: pd.DataFrame, tf: Timeframe, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Return only candles that have closed by ``now_utc`` (drops the forming candle)."""
    if bars.empty:
        return bars
    mask = bars["time"] + tf.delta <= _utc(now_utc)
    return bars.loc[mask].reset_index(drop=True)


def last_closed_open_time(tf: Timeframe, now_utc: pd.Timestamp) -> pd.Timestamp:
    """Open time of the most recent candle on a UTC-aligned grid that is closed at ``now_utc``.

    Note: H4 candles are aligned to the broker's server midnight, not UTC,
    so use :func:`closed_bars` on real data; this helper is for UTC-aligned
    timeframes (M1-H1) and scheduling.
    """
    now = _utc(now_utc)
    return now.floor(tf.pandas_freq) - tf.delta


def as_of(bars: pd.DataFrame, tf: Timeframe, decision_time: pd.Timestamp) -> pd.DataFrame:
    """Higher-timeframe view that is safe to use at ``decision_time``.

    Use this when combining timeframes (e.g. reading H1 context while
    deciding on an M5 candle that closes at ``decision_time``): only H1
    candles already closed at that moment are returned.
    """
    return closed_bars(bars, tf, decision_time)
