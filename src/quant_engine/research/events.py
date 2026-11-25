"""Market events that may start a repeatable move, found on closed M5 candles.

Every event on candle ``j`` is known at that candle's close (``time[j] + 5 min``)
and only uses candles ``<= j`` plus higher-timeframe candles already closed then.
Events (all distances in M5 ATR(14)):

* ``hour HH UTC``: the clock reaches HH:00 UTC (the candle closing at HH:00).
* ``swept Nh low / high``: the candle trades below the lowest low (above the highest
  high) of the previous N hours and closes back inside: a liquidity sweep.
* ``swept prev-day low / high``: same against the previous trading day (17:00 New York).
* ``broke Nh high / low``: first close above the N-hour high (below the low).
* ``impulse up / down``: a large, full-bodied candle.
* ``big move up / down``: price moved a lot in the last hour (first candle that qualifies).
* ``range top / bottom touch``: inside a 4-hour range of normal width, price reaches the
  top (bottom) 10% without breaking it.

Each event has a ``natural`` direction (+1 for the bullish-looking ones), so a BUY after an
up event is "continuation" and a SELL after it is "reversal".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ResearchConfig, SessionsConfig
from ..features.atr import atr as atr_fn
from ..features.sessions import new_signals_blocked, session_labels, trading_day
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe


@dataclass
class EventTable:
    """Per-M5-candle inputs for mining (all arrays have one row per M5 candle)."""

    times: pd.DatetimeIndex        # open times, UTC
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray             # in price
    atr: np.ndarray
    regime: np.ndarray             # H4 trend regime known at the candle close
    session: np.ndarray            # New York session label of the candle
    blocked: np.ndarray            # Friday after the cutoff / Sunday: no new signals
    events: dict[str, np.ndarray]  # event name -> bool mask
    natural: dict[str, int]        # event name -> +1 bullish-looking, -1 bearish-looking
    hour_events: frozenset[str]    # events that already are a time of day (no session split)


def regime_per_bar(m5_times: pd.DatetimeIndex, h4: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
    """H4 regime of the last H4 candle closed at each M5 candle's close (no look-ahead)."""
    closes = pd.DatetimeIndex(h4["time"]) + Timeframe.H4.delta
    decision = m5_times + Timeframe.M5.delta
    k = closes.searchsorted(decision, side="right") - 1
    reg = regimes.to_numpy(dtype=object)
    return np.where(k >= 0, reg[np.clip(k, 0, None)], "neutral").astype(object)


def _prior_extreme(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Max/min of the previous ``n`` values (excluding the current one)."""
    s = pd.Series(x).shift(1).rolling(n, min_periods=n)
    return (s.max() if kind == "max" else s.min()).to_numpy()


def _first(mask: np.ndarray) -> np.ndarray:
    """True only where ``mask`` turns on (not on the following candles)."""
    prev = np.concatenate([[False], mask[:-1]])
    return mask & ~prev


def build_events(m5: pd.DataFrame, h4: pd.DataFrame, rc: ResearchConfig, sessions: SessionsConfig,
                 point: float, assumed_spread_points: float, regime_kw: dict | None = None) -> EventTable:
    """Detect every event on ``m5`` (bars with open times in UTC). ``regime_kw`` are the
    :func:`trend_regime_series` settings (default: the backtest's)."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    sp_pts = m5["spread"].to_numpy(dtype="float64") if "spread" in m5 else np.full(len(c), np.nan)
    sp_pts = np.where(np.isnan(sp_pts) | (sp_pts <= 0), assumed_spread_points, sp_pts)
    a = atr_fn(m5, 14).to_numpy(dtype="float64")
    regimes = trend_regime_series(h4, **(regime_kw or {}))
    ev: dict[str, np.ndarray] = {}
    nat: dict[str, int] = {}
    valid = ~np.isnan(a) & (a > 0)

    # time of day: the candle that closes at HH:00 UTC
    close_hour = (t + Timeframe.M5.delta)
    on_hour = np.asarray(close_hour.minute == 0)
    hrs = np.asarray(close_hour.hour)
    hour_names = []
    for hh in range(24):
        name = f"hour {hh:02d}:00 UTC"
        ev[name] = on_hour & (hrs == hh) & valid
        nat[name] = 1
        hour_names.append(name)

    # liquidity sweeps and closed breakouts of the last N hours
    for n in rc.sweep_lookbacks:
        hrs_lbl = f"{n * 5 / 60:g}h"
        lo, hi = _prior_extreme(l, n, "min"), _prior_extreme(h, n, "max")
        ev[f"swept {hrs_lbl} low"] = valid & (l < lo) & (c > lo)
        nat[f"swept {hrs_lbl} low"] = 1
        ev[f"swept {hrs_lbl} high"] = valid & (h > hi) & (c < hi)
        nat[f"swept {hrs_lbl} high"] = -1
        ev[f"broke {hrs_lbl} high"] = valid & _first(c > hi)
        nat[f"broke {hrs_lbl} high"] = 1
        ev[f"broke {hrs_lbl} low"] = valid & _first(c < lo)
        nat[f"broke {hrs_lbl} low"] = -1

    # previous trading day high / low (known once that day has closed)
    day = trading_day(t, sessions)
    df = pd.DataFrame({"day": day, "h": h, "l": l})
    daily = df.groupby("day").agg(h=("h", "max"), l=("l", "min"))
    prev = daily.shift(1)
    pdh = prev["h"].reindex(day).to_numpy()
    pdl = prev["l"].reindex(day).to_numpy()
    # first sweep of the day only: no earlier candle today went beyond the level
    low_so_far = df.groupby("day")["l"].cummin().shift(1).to_numpy()
    high_so_far = df.groupby("day")["h"].cummax().shift(1).to_numpy()
    new_day = np.concatenate([[True], day[1:] != day[:-1]])
    low_so_far = np.where(new_day, np.inf, low_so_far)
    high_so_far = np.where(new_day, -np.inf, high_so_far)
    ev["swept prev-day low"] = valid & (l < pdl) & (c > pdl) & (low_so_far >= pdl)
    nat["swept prev-day low"] = 1
    ev["swept prev-day high"] = valid & (h > pdh) & (c < pdh) & (high_so_far <= pdh)
    nat["swept prev-day high"] = -1

    # impulse candles
    body, rng = c - o, h - l
    full = np.divide(np.abs(body), rng, out=np.zeros_like(body), where=rng > 0) >= rc.impulse_body_ratio
    ev["impulse up"] = valid & full & (body >= rc.impulse_body_atr * a)
    nat["impulse up"] = 1
    ev["impulse down"] = valid & full & (-body >= rc.impulse_body_atr * a)
    nat["impulse down"] = -1

    # big move over the last hour (first candle that qualifies)
    move = c - pd.Series(c).shift(rc.move_bars).to_numpy()
    ev["big move up"] = valid & _first(move >= rc.move_atr * a)
    nat["big move up"] = 1
    ev["big move down"] = valid & _first(move <= -rc.move_atr * a)
    nat["big move down"] = -1

    # range edges (touch without a break)
    rhi, rlo = _prior_extreme(h, rc.range_bars, "max"), _prior_extreme(l, rc.range_bars, "min")
    width = rhi - rlo
    ok = valid & (width >= rc.range_min_atr * a) & (width <= rc.range_max_atr * a)
    ev["range top touch"] = ok & (h >= rhi - rc.range_edge * width) & (h <= rhi)
    nat["range top touch"] = -1
    ev["range bottom touch"] = ok & (l <= rlo + rc.range_edge * width) & (l >= rlo)
    nat["range bottom touch"] = 1

    for k in ev:
        ev[k] = np.nan_to_num(ev[k], nan=False).astype(bool)
    return EventTable(times=t, o=o, h=h, l=l, c=c, spread=sp_pts * point, atr=a,
                      regime=regime_per_bar(t, h4, regimes), session=session_labels(t, sessions),
                      blocked=new_signals_blocked(t, sessions), events=ev, natural=nat,
                      hour_events=frozenset(hour_names))
