"""D1 candles built from M5 without look-ahead.

A trading day ends at the configured boundary (default 17:00 New York, DST-correct: the FX/metals
day most MT5 brokers use for their server D1 candles). Day *D* covers ``(D-1) 17:00 NY`` to
``D 17:00 NY``; Sunday-evening candles belong to Monday. A D1 candle is **final only at its
official close time** (``close_time``), even on an early-close holiday, because before that time
nobody can know that no more candles will come. Every decision in this module happens at a D1
``close_time`` and uses only D1 candles whose ``close_time`` is <= that moment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"

D1_COLUMNS = ["day", "open_time", "close_time", "open", "high", "low", "close", "n_bars", "spread_median",
              "complete"]


def trading_day(times: pd.DatetimeIndex | pd.Series, boundary: str = "new_york_17") -> pd.DatetimeIndex:
    """Trading-day date (tz-naive midnight) of each UTC candle open time."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        raise ValueError("times must be tz-aware UTC")
    if boundary == "new_york_17":
        return (t.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)).normalize()
    if boundary == "utc_midnight":
        return t.tz_convert("UTC").tz_localize(None).normalize()
    raise ValueError(f"unknown day boundary {boundary!r}")


def day_close_time(days: pd.DatetimeIndex | pd.Series, boundary: str = "new_york_17") -> pd.DatetimeIndex:
    """Official UTC close time of each trading day."""
    d = pd.DatetimeIndex(days)
    if boundary == "new_york_17":
        return (d + pd.Timedelta(hours=17)).tz_localize(NY).tz_convert("UTC")
    if boundary == "utc_midnight":
        return (d + pd.Timedelta(days=1)).tz_localize("UTC")
    raise ValueError(f"unknown day boundary {boundary!r}")


def build_daily(m5: pd.DataFrame, boundary: str = "new_york_17", min_bars: int = 100,
                point: float = 0.01, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate M5 candles (UTC open ``time``) into D1 candles.

    Only days whose ``close_time`` is <= ``as_of`` are returned (default: the close of the last
    M5 candle), so the forming day is never included. ``complete`` is False when the day has fewer
    than ``min_bars`` M5 candles (missing data; such a day is still a session, but no signal may be
    generated from it). ``spread_median`` is in price units (``point`` x MT5 spread points).
    """
    if m5.empty:
        return pd.DataFrame(columns=D1_COLUMNS)
    t = pd.DatetimeIndex(m5["time"])
    day = trading_day(t, boundary)
    df = pd.DataFrame({"day": day, "time": t, "open": m5["open"].to_numpy(float), "high": m5["high"].to_numpy(float),
                       "low": m5["low"].to_numpy(float), "close": m5["close"].to_numpy(float),
                       "spread": (m5["spread"].to_numpy(float) * point) if "spread" in m5 else np.nan})
    g = df.groupby("day", sort=True)
    out = pd.DataFrame({"open_time": g["time"].first(), "open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(), "n_bars": g["time"].size(),
                        "spread_median": g["spread"].median()})
    out.index.name = "day"
    out = out.reset_index()
    out["close_time"] = day_close_time(out["day"], boundary)
    out["complete"] = out["n_bars"] >= min_bars
    end = pd.Timestamp(as_of) if as_of is not None else t[-1] + pd.Timedelta(minutes=5)
    out = out[out["close_time"] <= end].reset_index(drop=True)
    return out[D1_COLUMNS]


def bar_sessions(m5_times: pd.DatetimeIndex, d1: pd.DataFrame, boundary: str = "new_york_17") -> np.ndarray:
    """Session ordinal (row of ``d1``) of every M5 candle; candles of days not in ``d1`` (the
    forming day) get ``len(d1)`` and later ordinals in order."""
    day = trading_day(m5_times, boundary)
    days = pd.DatetimeIndex(d1["day"])
    pos = days.get_indexer(day)
    if (pos < 0).any():                      # days after the last final D1 (forming day)
        extra = pd.DatetimeIndex(sorted(set(day[pos < 0])))
        pos = np.where(pos < 0, len(days) + extra.get_indexer(day), pos)
    return pos.astype("int64")


def session_number(day: pd.Timestamp) -> int:
    """Stable business-day number of a trading day (for distances between breakouts across runs)."""
    return int(np.busday_count(np.datetime64("2000-01-03"), np.datetime64(pd.Timestamp(day).date())))
