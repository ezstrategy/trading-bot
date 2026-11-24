"""Sessions, kill zones and trading days in New York time.

The strategy defines its windows in New York local time, so US daylight
saving moves them automatically in UTC (London kill zone = 07:00-10:00 UTC
in winter, 06:00-09:00 UTC in summer). A candle belongs to a window when
its OPEN time is inside ``[start, end)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SessionsConfig, Window

NY = "America/New_York"

ASIA, LONDON_KZ, NEWYORK_KZ, LONDON_CLOSE, ROLLOVER, OFF = (
    "asia", "london_kz", "newyork_kz", "london_close", "rollover", "off")
KILL_ZONES = (LONDON_KZ, NEWYORK_KZ)


def ny_minutes(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Minutes after New York midnight for each UTC timestamp."""
    ny = pd.DatetimeIndex(times).tz_convert(NY)
    return np.asarray(ny.hour * 60 + ny.minute)


def _in(m: np.ndarray, w: Window) -> np.ndarray:
    return (m >= w.start) & (m < w.end) if w.start <= w.end else (m >= w.start) | (m < w.end)


def session_labels(times: pd.Series | pd.DatetimeIndex, cfg: SessionsConfig) -> np.ndarray:
    """Session name per timestamp: asia, london_kz, newyork_kz, london_close, rollover or off."""
    m = ny_minutes(times)
    out = np.full(len(m), OFF, dtype=object)
    for name, w in ((ASIA, cfg.asia), (LONDON_KZ, cfg.london_kz), (NEWYORK_KZ, cfg.newyork_kz),
                    (LONDON_CLOSE, cfg.london_close), (ROLLOVER, cfg.rollover)):
        out[_in(m, w)] = name
    return out


def in_kill_zone(times: pd.Series | pd.DatetimeIndex, cfg: SessionsConfig) -> np.ndarray:
    """True when the timestamp is inside the London or New York kill zone."""
    m = ny_minutes(times)
    return _in(m, cfg.london_kz) | _in(m, cfg.newyork_kz)


def in_kill_zone_or_grace(times: pd.Series | pd.DatetimeIndex, cfg: SessionsConfig) -> np.ndarray:
    """Kill zone plus the grace period after it (CHoCH and entry may complete then)."""
    m = ny_minutes(times)
    g = cfg.kill_zone_grace_minutes
    lz = Window(start=cfg.london_kz.start, end=cfg.london_kz.end + g)
    nz = Window(start=cfg.newyork_kz.start, end=cfg.newyork_kz.end + g)
    return _in(m, lz) | _in(m, nz)


def trading_day(times: pd.Series | pd.DatetimeIndex, cfg: SessionsConfig) -> pd.DatetimeIndex:
    """Trading date of each timestamp. A day runs from 17:00 to 17:00 New York time,
    labelled with the calendar date it ends on (Monday 17:00 NY starts Tuesday)."""
    ny = pd.DatetimeIndex(times).tz_convert(NY).tz_localize(None)
    shift = pd.Timedelta(minutes=24 * 60 - cfg.trading_day_start)
    return (ny + shift).normalize()


def new_signals_blocked(times: pd.Series | pd.DatetimeIndex, cfg: SessionsConfig) -> np.ndarray:
    """True on Friday after the cutoff (New York time) and on Sunday (the weekly open)."""
    ny = pd.DatetimeIndex(times).tz_convert(NY)
    m = np.asarray(ny.hour * 60 + ny.minute)
    wd = np.asarray(ny.weekday)
    return ((wd == 4) & (m >= cfg.friday_cutoff)) | (wd == 6)
