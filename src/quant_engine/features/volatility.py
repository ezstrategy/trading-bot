"""Volatility and spread filters (Strategy Spec v1, "News and volatility filters")."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SessionsConfig, VolatilityConfig
from .sessions import ny_minutes, trading_day

NORMAL, DEAD, CHAOS, UNKNOWN = "normal", "dead", "chaos", "unknown"


def atr_time_of_day_ratio(m5: pd.DataFrame, atr_m5: pd.Series, days: int = 20,
                          sessions: SessionsConfig | None = None) -> pd.Series:
    """ATR(14, M5) divided by its median at the same New York time of day over
    the previous ``days`` trading days (the current day is excluded)."""
    sessions = sessions or SessionsConfig()
    t = pd.DatetimeIndex(m5["time"])
    df = pd.DataFrame({"slot": ny_minutes(t), "td": trading_day(t, sessions),
                       "atr": atr_m5.to_numpy(dtype="float64")}, index=m5.index)
    med = (df.groupby("slot", group_keys=False)["atr"]
           .apply(lambda s: s.shift(1).rolling(days, min_periods=max(5, days // 2)).median()))
    return (df["atr"] / med.reindex(df.index)).rename("atr_tod_ratio")


def volatility_regime(ratio: pd.Series, cfg: VolatilityConfig) -> pd.Series:
    """'dead' below dead_ratio, 'chaos' above chaos_ratio, else 'normal' ('unknown' without history)."""
    r = ratio.to_numpy(dtype="float64")
    out = np.where(np.isnan(r), UNKNOWN, np.where(r < cfg.dead_ratio, DEAD,
                                                   np.where(r > cfg.chaos_ratio, CHAOS, NORMAL)))
    return pd.Series(out, index=ratio.index, name="vol_regime")


def spread_ratio(m5: pd.DataFrame, days: int = 20) -> pd.Series:
    """Candle spread divided by the median spread of the previous ``days`` calendar days."""
    t = pd.DatetimeIndex(m5["time"])
    s = pd.Series(m5["spread"].to_numpy(dtype="float64"), index=t)
    med = s.shift(1).rolling(f"{days}D", min_periods=50).median()
    return pd.Series((s / med).to_numpy(), index=m5.index, name="spread_ratio")


def spread_ok(ratio: float, stop_distance: float | None, spread_price: float, cfg: VolatilityConfig) -> bool:
    """Spread gate: spread <= max_spread_ratio x median, and stop >= min_stop_spreads x spread."""
    if not np.isnan(ratio) and ratio > cfg.max_spread_ratio:
        return False
    if stop_distance is not None and stop_distance < cfg.min_stop_spreads * spread_price:
        return False
    return True
