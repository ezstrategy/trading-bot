"""London breakout signal mode (``strategy.mode: london``).

The rules are the tested research version, unchanged: EXP-016 "original" (= EXP-015 trend version)
with the EXP-020 natural stop (owner rule 2026-11-27):

* Asian range: high and low of the M5 candles from 20:00 to 01:00 New York time.
* At 01:00 NY two stop orders are possible: buy above the high, sell below the low (0.02% of
  price beyond the range). With the H4 trend filter only the trend side is allowed (neutral H4:
  both sides, one cancels the other). The first breakout decides the day: if it is on a side
  that is not allowed, there is no trade that day.
* Stop: the other side of the range (natural stop, never tightened). Target 3R. Stop to
  breakeven once price has gone 1R in favour. Anything still open is closed at 16:00 NY.

One function, :func:`replay_day`, follows a trading day candle by candle and returns every event
with its time. The live engine sends the events it has not sent yet; :func:`london_trades`
replays history with the same function (so the backtest and the live signals cannot differ),
and a test checks it against the research code (``research.lb_phase2.lb_trades``).

No look-ahead: a day is replayed only on closed M5 candles, and the H4 trend is the one of the
last H4 candle closed at the end of the range (01:00 NY).
SIGNALS ONLY: nothing here places, changes or closes orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import LondonConfig, SessionsConfig
from ..features.sessions import ny_minutes, trading_day
from ..timeframes import Timeframe
from .regime import BEARISH, BULLISH, trend_regime_series

# event kinds (in the order they can happen)
SETUP, NO_SETUP, TRIGGERED, BREAKEVEN = "setup", "no_setup", "triggered", "breakeven"
TARGET, STOP, BE_STOP, SESSION_CLOSE, CANCELLED = "target", "stop", "breakeven_stop", "session_close", "cancelled"
EXITS = (TARGET, STOP, BE_STOP, SESSION_CLOSE)


@dataclass
class Level:
    """One side of the day's setup (what a trader would place as a stop order)."""

    side: int            # +1 buy stop, -1 sell stop
    entry: float         # stop-order level
    stop: float          # natural stop (other side of the range)
    target: float        # target_r from the level

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Event:
    kind: str
    time: pd.Timestamp   # close time of the candle where it happened (when it is known)
    price: float | None = None
    r: float | None = None
    note: str = ""


@dataclass
class LondonDay:
    """Everything known about one trading day at the time of the last candle given."""

    day: pd.Timestamp
    setup_time: pd.Timestamp | None = None          # close of the last range candle (01:00 NY)
    close_time: pd.Timestamp | None = None          # 16:00 NY
    range_high: float = np.nan
    range_low: float = np.nan
    trend: str = "neutral"
    allowed: list[int] = field(default_factory=list)
    levels: dict[int, Level] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    side: int = 0                                   # side that triggered (0 = none yet)
    entry: float = np.nan                           # fill price (can be beyond the level on a gap)
    initial_stop: float = np.nan                    # natural stop at the fill (before breakeven)
    stop: float = np.nan                            # current stop (entry after breakeven)
    target: float = np.nan
    entry_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float = np.nan
    r: float | None = None

    @property
    def finished(self) -> bool:
        return any(e.kind in EXITS + (CANCELLED, NO_SETUP) for e in self.events)

    @property
    def is_open(self) -> bool:
        return self.side != 0 and not self.finished


def trend_per_candle(m5_times: pd.DatetimeIndex, h4: pd.DataFrame, bt_cfg) -> np.ndarray:
    """H4 trend (bullish / bearish / neutral) of the last H4 candle closed at each M5 candle's close."""
    reg = trend_regime_series(h4, ema_period=bt_cfg.regime_ema_period, slope_bars=bt_cfg.regime_slope_bars,
                              min_slope_atr=bt_cfg.regime_min_slope_atr).to_numpy(dtype=object)
    closes = pd.DatetimeIndex(h4["time"]) + Timeframe.H4.delta
    k = closes.searchsorted(m5_times + Timeframe.M5.delta, side="right") - 1
    return np.where(k >= 0, reg[np.clip(k, 0, None)], "neutral").astype(object)


def replay_day(t: pd.DatetimeIndex, o, h, l, c, idx: np.ndarray, trend: np.ndarray, sp: float,
               p: LondonConfig, day: pd.Timestamp, last_known: pd.Timestamp | None = None) -> LondonDay:
    """Follow one trading day (candle indices ``idx``) and return its setup and events.

    ``last_known``: decision time (close of the newest closed candle). Candles closing after it are
    never read; without it every candle in ``idx`` is used (history replay).
    """
    if last_known is not None:
        idx = idx[t[idx] + Timeframe.M5.delta <= last_known]
    mins = ny_minutes(t[idx])
    rs, re_, cl = p.minutes(p.range_start_ny), p.minutes(p.range_end_ny), p.minutes(p.close_ny)
    in_range = (mins >= rs) | (mins < re_) if rs > re_ else (mins >= rs) & (mins < re_)
    active = (mins >= re_) & (mins < cl)
    out = LondonDay(day=day)
    r_idx, a_idx = idx[in_range], idx[active]
    setup_at, out.close_time = _ny_time(day, re_), _ny_time(day, cl)
    if last_known is None:
        if not len(a_idx):                   # history: a day without London candles is not traded
            return out
    elif last_known < setup_at:              # live: the range is still forming
        return out
    out.setup_time = setup_at                # close of the last Asian-range candle
    if len(r_idx) < p.min_range_candles:
        out.events.append(Event(NO_SETUP, out.setup_time, note=f"only {len(r_idx)} Asian-range candles"))
        return out
    hi, lo = float(h[r_idx].max()), float(l[r_idx].min())
    px = float(c[r_idx[-1]])
    out.range_high, out.range_low = hi, lo
    out.trend = str(trend[r_idx[-1]])
    if p.trend_filter:
        out.allowed = [1] if out.trend == BULLISH else [-1] if out.trend == BEARISH else [1, -1]
    else:
        out.allowed = [1, -1]
    buf = p.buffer_pct / 100.0 * px
    buy_lvl, sell_lvl = hi + buf, lo - buf
    for s in out.allowed:
        lvl = buy_lvl if s == 1 else sell_lvl
        stop = _stop(s, lvl, hi, lo, sp, p)
        out.levels[s] = Level(s, lvl, stop, lvl + s * p.target_r * abs(lvl - stop))
    out.events.append(Event(SETUP, out.setup_time))

    trig = None
    for k, i in enumerate(a_idx):
        up, dn = h[i] + sp >= buy_lvl, l[i] <= sell_lvl
        if not (up or dn):
            continue
        if up and dn:                        # both in one candle: the side nearer the open
            up = abs(buy_lvl - o[i] - sp) <= abs(o[i] - sell_lvl)
        s = 1 if up else -1
        when = t[i] + Timeframe.M5.delta
        if s not in out.allowed:
            out.events.append(Event(CANCELLED, when, note=f"price broke {'above' if s == 1 else 'below'} the range "
                                                         f"first (against the H4 trend): no trade today"))
            return out
        entry = max(buy_lvl, o[i] + sp) if s == 1 else min(sell_lvl, o[i])
        stop = _stop(s, entry, hi, lo, sp, p)
        if abs(entry - stop) <= 0 or (entry - stop) * s <= 0:
            out.events.append(Event(CANCELLED, when, note="invalid stop distance"))
            return out
        trig = k
        out.side, out.entry, out.stop, out.initial_stop = s, entry, stop, stop
        out.target = entry + s * p.target_r * abs(entry - stop)
        out.entry_time = t[i]
        out.events.append(Event(TRIGGERED, when, entry))
        break
    if trig is None:
        if _window_over(last_known, out.close_time):
            out.events.append(Event(CANCELLED, out.close_time, note="no breakout before the London close"))
        return out
    _follow(out, t, o, h, l, c, a_idx[trig:], sp, p, last_known)
    return out


def _ny_time(day: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """UTC time of ``minutes`` after New York midnight on trading date ``day`` (DST-correct)."""
    local = pd.Timestamp(day).tz_localize(None).normalize() + pd.Timedelta(minutes=minutes)
    return local.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def _stop(side: int, entry: float, hi: float, lo: float, sp: float, p: LondonConfig) -> float:
    nat = lo if side == 1 else hi + sp
    if p.stop_cap_pct is None:
        return nat
    cap = p.stop_cap_pct / 100.0
    return max(nat, entry * (1 - cap)) if side == 1 else min(nat, entry * (1 + cap))


def _window_over(last_known: pd.Timestamp | None, close_time: pd.Timestamp) -> bool:
    """True when the day's active window has ended (so 'no breakout' or 'session close' is final)."""
    return last_known is None or last_known >= close_time


def _follow(d: LondonDay, t, o, h, l, c, bars: np.ndarray, sp: float, p: LondonConfig,
            last_known: pd.Timestamp | None) -> None:
    """Walk the triggered trade (stop before target; target not on the fill candle; breakeven at be_r)."""
    s, entry, stop, tgt = d.side, d.entry, d.stop, d.target
    risk, i0 = abs(entry - stop), bars[0]
    state = STOP
    for i in bars:
        when = t[i] + Timeframe.M5.delta
        if s == 1:
            if l[i] <= stop:
                return _exit(d, state, when, min(o[i], stop) if i > i0 else stop, t[i])
            if i > i0 and h[i] >= tgt:
                return _exit(d, TARGET, when, max(o[i], tgt), t[i])
            moved = p.breakeven_r is not None and h[i] - entry >= p.breakeven_r * risk and stop < entry
        else:
            if h[i] + sp >= stop:
                return _exit(d, state, when, max(o[i] + sp, stop) if i > i0 else stop, t[i])
            if i > i0 and l[i] + sp <= tgt:
                return _exit(d, TARGET, when, min(o[i] + sp, tgt), t[i])
            moved = p.breakeven_r is not None and entry - (l[i] + sp) >= p.breakeven_r * risk and stop > entry
        if moved:
            stop, state = entry, BE_STOP
            d.stop = stop
            d.events.append(Event(BREAKEVEN, when, entry))
    if _window_over(last_known, d.close_time):
        last = bars[-1]
        _exit(d, SESSION_CLOSE, t[last] + Timeframe.M5.delta, c[last] + (sp if s == -1 else 0.0), t[last])


def _exit(d: LondonDay, kind: str, when: pd.Timestamp, price: float, bar_time: pd.Timestamp) -> None:
    d.r = d.side * (price - d.entry) / abs(d.entry - d.initial_stop)
    d.exit_time, d.exit_price = bar_time, price
    d.events.append(Event(kind, when, price, d.r))


def day_groups(t: pd.DatetimeIndex, sessions: SessionsConfig) -> list[tuple[pd.Timestamp, np.ndarray]]:
    """(trading day, candle indices) for each 17:00-17:00 New York day."""
    day = trading_day(t, sessions)
    codes, uniq = pd.Series(day).factorize()
    bounds = np.r_[0, np.nonzero(np.diff(codes))[0] + 1, len(codes)]
    return [(pd.Timestamp(uniq[codes[a]]), np.arange(a, b)) for a, b in zip(bounds[:-1], bounds[1:])]


def _arrays(m5: pd.DataFrame):
    t = pd.DatetimeIndex(m5["time"])
    return (t, *(m5[k].to_numpy(float) for k in ("open", "high", "low", "close")))


def current_day(m5: pd.DataFrame, h4: pd.DataFrame, strat, sp: float, now: pd.Timestamp) -> LondonDay | None:
    """The newest trading day as known at ``now`` (closed candles only), or None without data."""
    m5 = m5[m5["time"] + Timeframe.M5.delta <= now]
    if m5.empty:
        return None
    t, o, h, l, c = _arrays(m5)
    groups = day_groups(t, strat.sessions)
    day, idx = groups[-1]
    trend = trend_per_candle(t, h4[h4["time"] + Timeframe.H4.delta <= now], strat.backtest)
    return replay_day(t, o, h, l, c, idx, trend, sp, strat.london, day, last_known=now)


def days_known_at(m5: pd.DataFrame, h4: pd.DataFrame, strat, sp: float, now: pd.Timestamp) -> list[LondonDay]:
    """Every trading day in ``m5`` replayed with only the candles closed at ``now`` (oldest first)."""
    m5 = m5[m5["time"] + Timeframe.M5.delta <= now]
    if m5.empty:
        return []
    t, o, h, l, c = _arrays(m5)
    trend = trend_per_candle(t, h4[h4["time"] + Timeframe.H4.delta <= now], strat.backtest)
    return [replay_day(t, o, h, l, c, idx, trend, sp, strat.london, day, last_known=now)
            for day, idx in day_groups(t, strat.sessions)]


def london_trades(m5: pd.DataFrame, h4: pd.DataFrame, strat, sp: float) -> pd.DataFrame:
    """Every finished trade in the history, replayed with the live code (one row per traded day)."""
    t, o, h, l, c = _arrays(m5)
    trend = trend_per_candle(t, h4, strat.backtest)
    rows = []
    for day, idx in day_groups(t, strat.sessions):
        d = replay_day(t, o, h, l, c, idx, trend, sp, strat.london, day)
        if d.side and d.r is not None:
            rows.append({"day": day, "entry_time": d.entry_time, "exit_time": d.exit_time,
                         "side": "BUY" if d.side == 1 else "SELL", "entry": d.entry, "stop": d.initial_stop,
                         "exit": d.exit_price, "risk": abs(d.entry - d.initial_stop), "r": d.r,
                         "exit_reason": d.events[-1].kind, "trend": d.trend,
                         "range_high": d.range_high, "range_low": d.range_low})
    return pd.DataFrame(rows)
