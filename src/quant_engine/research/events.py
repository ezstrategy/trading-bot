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
* Major levels, round numbers ($50 / $100), pin bars at them and news shocks: see
  :func:`_levels_and_news`.

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
    tf: Timeframe = Timeframe.M5   # timeframe of the candles (entry timeframe)


def regime_per_bar(m5_times: pd.DatetimeIndex, h4: pd.DataFrame, regimes: pd.Series,
                   tf: Timeframe = Timeframe.M5) -> np.ndarray:
    """H4 regime of the last H4 candle closed at each candle's close (no look-ahead)."""
    closes = pd.DatetimeIndex(h4["time"]) + Timeframe.H4.delta
    decision = m5_times + tf.delta
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
                 point: float, assumed_spread_points: float, regime_kw: dict | None = None,
                 tf: Timeframe = Timeframe.M5) -> EventTable:
    """Detect every event on ``m5`` (bars with open times in UTC; despite the name they can be
    H1 or H4 candles when ``tf`` says so). ``regime_kw`` are the :func:`trend_regime_series`
    settings (default: the backtest's). Lookbacks in ``rc`` are counted in ``tf`` candles."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    sp_pts = m5["spread"].to_numpy(dtype="float64") if "spread" in m5 else np.full(len(c), np.nan)
    sp_pts = np.where(np.isnan(sp_pts) | (sp_pts <= 0), assumed_spread_points, sp_pts)
    if rc.fixed_spread_points is not None:   # same cost in every year (broker spreads changed a lot)
        sp_pts = np.full(len(c), float(rc.fixed_spread_points))
    a = atr_fn(m5, 14).to_numpy(dtype="float64")
    regimes = trend_regime_series(h4, **(regime_kw or {}))
    ev: dict[str, np.ndarray] = {}
    nat: dict[str, int] = {}
    valid = ~np.isnan(a) & (a > 0)

    # time of day: the candle that closes at HH:00 UTC
    close_hour = (t + tf.delta)
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
        hrs_lbl = f"{n * tf.minutes / 60:g}h"
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

    _levels_and_news(ev, nat, t, o, h, l, c, a, valid, rc, tf)
    _price_action(ev, nat, o, h, l, c, a, valid, rc, tf)

    for k in ev:
        ev[k] = np.nan_to_num(ev[k], nan=False).astype(bool)
    return EventTable(times=t, o=o, h=h, l=l, c=c, spread=sp_pts * point, atr=a,
                      regime=regime_per_bar(t, h4, regimes, tf), session=session_labels(t, sessions),
                      blocked=new_signals_blocked(t, sessions), events=ev, natural=nat,
                      hour_events=frozenset(hour_names), tf=tf)


def _pins(o, h, l, c, a, rc: ResearchConfig) -> tuple[np.ndarray, np.ndarray]:
    """(bullish pin, bearish pin): long lower / upper rejection wick on a candle >= pin_min_atr."""
    rng = h - l
    big = rng >= rc.pin_min_atr * a
    safe = np.where(rng > 0, rng, np.nan)
    lower = (np.minimum(o, c) - l) / safe
    upper = (h - np.maximum(o, c)) / safe
    return big & (lower >= rc.pin_wick_ratio), big & (upper >= rc.pin_wick_ratio)


def _levels_and_news(ev: dict, nat: dict, t: pd.DatetimeIndex, o, h, l, c, a, valid, rc: ResearchConfig,
                     tf: Timeframe = Timeframe.M5) -> None:
    """Major support / resistance, round numbers, price action at them, and news shocks.

    * ``round N support / resistance``: the candle wicks through a multiple of N USD and
      closes back on the side it opened (a rejection of the round number).
    * ``round N break up / down``: the close crosses a multiple of N USD.
    * ``bullish / bearish pin at round``: pin bar whose wick touches a round number.
    * ``swept / broke Nd high / low``: the N-day high or low (major S/R) is swept and
      rejected, or closed through. ``bullish / bearish pin at Nd low / high``: a pin bar
      within ``level_touch_atr`` of that level.
    * ``news spike up / down``: the candle containing a US release time (New York clock)
      is >= ``news_spike_atr`` x the previous ATR. ``NFP spike up / down``: the same on the
      first Friday of the month at 08:30 New York.
    """
    bull_pin, bear_pin = _pins(o, h, l, c, a, rc)
    smallest = min(rc.round_steps)
    for step in rc.round_steps:
        s = f"{step:g}"
        lvl_hi = np.floor(h / step) * step          # highest round level at or below the high
        lvl_lo = np.ceil(l / step) * step           # lowest round level at or above the low
        res = valid & (lvl_hi > np.maximum(o, c)) & (lvl_hi <= h)
        sup = valid & (lvl_lo < np.minimum(o, c)) & (lvl_lo >= l)
        ev[f"round {s} resistance"], nat[f"round {s} resistance"] = res, -1
        ev[f"round {s} support"], nat[f"round {s} support"] = sup, 1
        prev_c = np.concatenate([[np.nan], c[:-1]])
        up = valid & (np.floor(c / step) > np.floor(prev_c / step))
        dn = valid & (np.floor(c / step) < np.floor(prev_c / step))
        ev[f"round {s} break up"], nat[f"round {s} break up"] = up, 1
        ev[f"round {s} break down"], nat[f"round {s} break down"] = dn, -1
        if step == smallest:
            touch_lo = np.ceil(l / step) * step <= np.minimum(o, c)
            touch_hi = np.floor(h / step) * step >= np.maximum(o, c)
            ev["bullish pin at round"], nat["bullish pin at round"] = valid & bull_pin & touch_lo, 1
            ev["bearish pin at round"], nat["bearish pin at round"] = valid & bear_pin & touch_hi, -1

    bars_per_day = 1440 // tf.minutes
    for days in rc.level_days:
        n = days * bars_per_day
        lo, hi = _prior_extreme(l, n, "min"), _prior_extreme(h, n, "max")
        d = f"{days}d"
        ev[f"swept {d} low"], nat[f"swept {d} low"] = valid & (l < lo) & (c > lo), 1
        ev[f"swept {d} high"], nat[f"swept {d} high"] = valid & (h > hi) & (c < hi), -1
        ev[f"broke {d} high"], nat[f"broke {d} high"] = valid & _first(c > hi), 1
        ev[f"broke {d} low"], nat[f"broke {d} low"] = valid & _first(c < lo), -1
        near = rc.level_touch_atr * a
        ev[f"bullish pin at {d} low"] = valid & bull_pin & (l <= lo + near) & (c > lo)
        nat[f"bullish pin at {d} low"] = 1
        ev[f"bearish pin at {d} high"] = valid & bear_pin & (h >= hi - near) & (c < hi)
        nat[f"bearish pin at {d} high"] = -1

    ny = t.tz_convert("America/New_York")
    mins = np.asarray(ny.hour * 60 + ny.minute)
    release = np.zeros(len(c), dtype=bool)
    for hhmm in rc.news_times_ny:
        hh, mm = (int(x) for x in hhmm.split(":"))
        rel = hh * 60 + mm
        release |= (mins <= rel) & (rel < mins + tf.minutes)
    prev_atr = np.concatenate([[np.nan], a[:-1]])
    shock = valid & release & ((h - l) >= rc.news_spike_atr * prev_atr)
    ev["news spike up"], nat["news spike up"] = shock & (c > o), 1
    ev["news spike down"], nat["news spike down"] = shock & (c < o), -1
    nfp = valid & (np.asarray(ny.weekday) == 4) & (np.asarray(ny.day) <= 7) & (mins <= 510) & (510 < mins + tf.minutes)
    ev["NFP spike up"], nat["NFP spike up"] = nfp & (c > o), 1
    ev["NFP spike down"], nat["NFP spike down"] = nfp & (c < o), -1


def _price_action(ev: dict, nat: dict, o, h, l, c, a, valid, rc: ResearchConfig,
                  tf: Timeframe = Timeframe.M5) -> None:
    """Candle patterns, anywhere and at key levels.

    * ``bullish / bearish pin``: long rejection wick (see :func:`_pins`).
    * ``bullish / bearish engulfing``: the candle's body swallows the previous opposite body and
      is at least ``pin_min_atr`` / 2 ATR long.
    * ``inside bar break up / down``: the previous candle sat inside the one before (the mother
      bar) and this candle closes above / below the mother bar.
    * ``... at support / resistance``: the same with the candle's low (high) within
      ``level_touch_atr`` ATR of the shortest ``level_days`` low (high) or touching a round number.
    """
    bull_pin, bear_pin = _pins(o, h, l, c, a, rc)
    po, pc = np.concatenate([[np.nan], o[:-1]]), np.concatenate([[np.nan], c[:-1]])
    body = np.abs(c - o)
    big = body >= 0.5 * rc.pin_min_atr * a
    bull_eng = big & (c > o) & (pc < po) & (o <= pc) & (c >= po)
    bear_eng = big & (c < o) & (pc > po) & (o >= pc) & (c <= po)
    h1, l1 = np.concatenate([[np.nan], h[:-1]]), np.concatenate([[np.nan], l[:-1]])
    h2, l2 = np.concatenate([[np.nan] * 2, h[:-2]]), np.concatenate([[np.nan] * 2, l[:-2]])
    inside_prev = (h1 <= h2) & (l1 >= l2)
    ib_up, ib_dn = inside_prev & (c > h2), inside_prev & (c < l2)

    n = min(rc.level_days) * (1440 // tf.minutes)
    lo, hi = _prior_extreme(l, n, "min"), _prior_extreme(h, n, "max")
    near = rc.level_touch_atr * a
    step = min(rc.round_steps)
    at_sup = (l <= lo + near) | (np.ceil(l / step) * step <= np.minimum(o, c))
    at_res = (h >= hi - near) | (np.floor(h / step) * step >= np.maximum(o, c))

    for name, mask, d in (("bullish pin", bull_pin, 1), ("bearish pin", bear_pin, -1),
                          ("bullish engulfing", bull_eng, 1), ("bearish engulfing", bear_eng, -1),
                          ("inside bar break up", ib_up, 1), ("inside bar break down", ib_dn, -1)):
        ev[name], nat[name] = valid & mask, d
    for name, mask, d in (("bullish pin at support", bull_pin & at_sup, 1),
                          ("bearish pin at resistance", bear_pin & at_res, -1),
                          ("bullish engulfing at support", bull_eng & at_sup, 1),
                          ("bearish engulfing at resistance", bear_eng & at_res, -1),
                          ("inside bar break up at support", ib_up & at_sup, 1),
                          ("inside bar break down at resistance", ib_dn & at_res, -1)):
        ev[name], nat[name] = valid & mask, d


PRICE_ACTION_FAMILIES = ("price action", "price action at levels")


def event_family(name: str) -> str:
    """Group an event name into an idea family (for the report)."""
    if name.startswith("hour "):
        return "time of day"
    if "news" in name or "NFP" in name:
        return "news shocks"
    if any(k in name for k in ("pin", "engulfing", "inside bar")):
        return "price action at levels" if " at " in name else "price action"
    if name.startswith("round "):
        return "round numbers"
    if name.endswith(("d high", "d low")) or "prev-day" in name:
        return "major support / resistance"
    if name.startswith(("impulse", "big move")):
        return "momentum"
    if name.startswith("range "):
        return "range edges"
    return "intraday sweeps / breakouts"
