"""Synthetic M5 data for breakout tests (shapes only; never used for reported results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def session_times(day: pd.Timestamp) -> pd.DatetimeIndex:
    """M5 open times (UTC) of the NY-17:00 trading day ``day``: (day-1) 18:00 NY to day 16:55 NY."""
    start = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(NY)
    end = (pd.Timestamp(day) + pd.Timedelta(hours=17)).tz_localize(NY)
    return pd.date_range(start, end, freq="5min", inclusive="left").tz_convert("UTC")


def trading_days(start: str, n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, periods=n))


def m5_from_daily(days: list[pd.Timestamp], ohlc: list[tuple[float, float, float, float]],
                  spread_points: int = 10, path: list[list[float]] | None = None) -> pd.DataFrame:
    """M5 candles whose D1 aggregates are exactly ``ohlc`` (path open -> low -> high -> close, or
    open -> high -> low -> close for down days). ``path`` may give explicit way points per day."""
    rows = []
    for k, (d, (o, h, l, c)) in enumerate(zip(days, ohlc)):
        t = session_times(d)
        n = len(t)
        pts = path[k] if path and path[k] else ([o, l, h, c] if c >= o else [o, h, l, c])
        xs = np.linspace(0, n, len(pts))
        line = np.interp(np.arange(n + 1), xs, pts)
        opens, closes = line[:-1], line[1:]
        hi = np.maximum(opens, closes)
        lo = np.minimum(opens, closes)
        hi = np.minimum(hi, h)
        lo = np.maximum(lo, l)
        for j in range(n):
            rows.append((t[j], opens[j], hi[j], lo[j], closes[j]))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["tick_volume"] = 100
    df["spread"] = spread_points
    df["real_volume"] = 0
    return df


def flat_days(start: str, n: int, base: float = 2000.0, rng: float = 10.0, drift: float = 0.0,
              seed: int = 1) -> tuple[list[pd.Timestamp], list[tuple[float, float, float, float]]]:
    """Quiet random days around ``base``."""
    r = np.random.default_rng(seed)
    days = trading_days(start, n)
    out, px = [], base
    for _ in days:
        o = px
        c = o + r.normal(drift, rng / 4)
        h = max(o, c) + abs(r.normal(rng / 3, 1))
        l = min(o, c) - abs(r.normal(rng / 3, 1))
        out.append((o, h, l, c))
        px = c
    return days, out
