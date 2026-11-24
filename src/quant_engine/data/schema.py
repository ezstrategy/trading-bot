"""Column layout of bar and tick frames used across the engine.

Bars: one row per candle, ``time`` = candle OPEN time in UTC (tz-aware).
Ticks: one row per quote, ``time`` = quote time in UTC with millisecond precision.
"""
from __future__ import annotations

import pandas as pd

BAR_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
TICK_COLUMNS = ["time", "bid", "ask", "last", "volume", "flags"]


def empty_bars() -> pd.DataFrame:
    """An empty bar frame with the right columns and dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS[1:]})
    df.insert(0, "time", pd.Series(dtype="datetime64[ns, UTC]"))
    return df


def empty_ticks() -> pd.DataFrame:
    """An empty tick frame with the right columns and dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in TICK_COLUMNS[1:]})
    df.insert(0, "time", pd.Series(dtype="datetime64[ns, UTC]"))
    return df
