"""Life of one signal on closed M5 candles: fill, targets, stops, expiry.

The same function drives the backtest and the live follow-up messages, so a
signal is managed identically in both. Rules (conservative on purpose):

* Bars are BID prices. A buy limit fills when the ask (bid + spread) reaches
  the entry; buy stops/targets are hit on the bid. Sells mirror this, so the
  spread is paid on every trade.
* The order is live from the candle after the signal for ``expiry_candles``.
  It is cancelled if price reaches TP1 first or closes beyond the sweep extreme.
* On the fill candle only the stop is checked. On later candles the stop is
  checked before any target.
* TP1 closes 50% and moves the stop to entry + spread. With a TP3, 25% closes
  at TP2 and 25% at TP3; otherwise the other 50% closes at TP2.
* Time stop: no TP1 within ``time_stop_candles`` after the fill -> close.
  Session end: anything open ``session_close_hours`` after the kill zone ends is closed
  (or at ``Signal.close_by`` when the model sets one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..features.sessions import NY
from ..timeframes import Timeframe
from .entry import Signal

# event kinds
FILLED, TP1, TP2, TP3 = "filled", "tp1", "tp2", "tp3"
STOP, BREAKEVEN, TIME_STOP, SESSION_END = "stop", "breakeven", "time_stop", "session_end"
EXPIRED, CANCEL_TP1, CANCEL_SWEEP = "expired", "cancelled_tp1", "cancelled_sweep"
FINAL = {STOP, BREAKEVEN, TIME_STOP, SESSION_END, EXPIRED, CANCEL_TP1, CANCEL_SWEEP}


@dataclass
class TradeEvent:
    """Something that happened to a signal, known at the close of candle ``idx``."""

    idx: int
    time: pd.Timestamp        # close time of that candle (UTC)
    kind: str
    price: float | None
    r: float                  # realised R of the whole position so far


@dataclass
class TradeResult:
    outcome: str
    filled: bool
    fill_idx: int | None
    exit_idx: int             # last candle looked at
    r: float
    is_open: bool             # still pending or running at the end of the data
    events: list[TradeEvent] = field(default_factory=list)


def kill_zone_end(t: pd.Timestamp, kz: str, cfg: StrategyConfig) -> pd.Timestamp:
    """UTC end of the kill zone a signal at ``t`` belongs to (New York wall clock)."""
    w = cfg.sessions.london_kz if kz == "london_kz" else cfg.sessions.newyork_kz
    ny = t.tz_convert(NY)
    end = ny.normalize().tz_localize(None) + pd.Timedelta(minutes=w.end)
    end = end.tz_localize(NY, ambiguous=False, nonexistent="shift_forward")
    if end < ny - pd.Timedelta(hours=12):
        end += pd.Timedelta(days=1)
    return end.tz_convert("UTC")


def simulate(s: Signal, t: pd.DatetimeIndex, h: np.ndarray, l: np.ndarray, c: np.ndarray,
             cfg: StrategyConfig, j0: int | None = None) -> TradeResult:
    """Follow signal ``s`` (sent at the close of candle ``j0``, default ``s.m5_idx``)
    over the M5 candles ``t/h/l/c`` (BID, open times UTC)."""
    e, d, n, sp = cfg.entry, s.direction, len(c), s.spread
    j0 = s.m5_idx if j0 is None else j0
    events: list[TradeEvent] = []

    def close_time(k: int) -> pd.Timestamp:
        return t[k] + Timeframe.M5.delta

    def ev(k, kind, px, r):
        events.append(TradeEvent(k, close_time(k), kind, None if px is None else float(px), round(float(r), 4)))

    # ---- pending order
    last = min(j0 + e.expiry_candles, n - 1)
    fill = None
    for k in range(j0 + 1, last + 1):
        if (l[k] + sp <= s.entry) if d == 1 else (h[k] >= s.entry):
            fill = k
            break
        if (h[k] >= s.tp1) if d == 1 else (l[k] + sp <= s.tp1):
            ev(k, CANCEL_TP1, None, 0.0)
            return TradeResult("cancelled: TP1 before fill", False, None, k, 0.0, False, events)
        if (c[k] < s.sweep_extreme) if d == 1 else (c[k] > s.sweep_extreme):
            ev(k, CANCEL_SWEEP, None, 0.0)
            return TradeResult("cancelled: closed beyond sweep", False, None, k, 0.0, False, events)
    if fill is None:
        if j0 + e.expiry_candles <= n - 1:
            ev(last, EXPIRED, None, 0.0)
            return TradeResult("expired: not filled in 30 min", False, None, last, 0.0, False, events)
        return TradeResult("pending", False, None, n - 1, 0.0, True, events)

    # ---- position
    ev(fill, FILLED, s.entry, 0.0)
    stop, size, realised = s.stop_loss, 1.0, 0.0
    session_end = (s.close_by if s.close_by is not None
                   else kill_zone_end(s.time, s.kill_zone, cfg) + pd.Timedelta(hours=e.session_close_hours))
    targets = [(s.tp1, 0.5, TP1)] + ([(s.tp2, 0.25, TP2), (s.tp3, 0.25, TP3)] if s.tp3 is not None
                                     else [(s.tp2, 0.5, TP2)])

    def stop_hit(k):
        return (l[k] <= stop) if d == 1 else (h[k] + sp >= stop)

    def tp_hit(k, px):
        return (h[k] >= px) if d == 1 else (l[k] + sp <= px)

    def exit_px(k):            # market close: bid for longs, ask for shorts
        return c[k] + (sp if d == -1 else 0)

    if stop_hit(fill):
        ev(fill, STOP, stop, -1.0)
        return TradeResult("stop loss", True, fill, fill, -1.0, False, events)
    nxt, k = 0, fill
    while True:
        k += 1
        if k >= n:
            r = realised + size * s.r_multiple(exit_px(n - 1))
            return TradeResult("open at end of data", True, fill, n - 1, r, True, events)
        if stop_hit(k):
            realised += size * (stop - s.entry) * d / s.risk
            kind = BREAKEVEN if nxt else STOP
            ev(k, kind, stop, realised)
            return TradeResult("breakeven after TP1" if nxt else "stop loss", True, fill, k, realised, False, events)
        while nxt < len(targets) and tp_hit(k, targets[nxt][0]):
            px, frac, kind = targets[nxt]
            realised += frac * s.r_multiple(px)
            size -= frac
            if nxt == 0:
                stop = s.entry + d * sp          # breakeven: entry plus spread
            ev(k, kind, px, realised)
            nxt += 1
        if nxt == len(targets):
            return TradeResult(f"TP{len(targets)} hit", True, fill, k, realised, False, events)
        if nxt == 0 and k - fill >= (s.time_stop_candles or e.time_stop_candles):
            r = realised + size * s.r_multiple(exit_px(k))
            ev(k, TIME_STOP, exit_px(k), r)
            return TradeResult("time stop (no TP1 in time)", True, fill, k, r, False, events)
        if close_time(k) >= session_end:
            r = realised + size * s.r_multiple(exit_px(k))
            ev(k, SESSION_END, exit_px(k), r)
            return TradeResult("session end close", True, fill, k, r, False, events)
