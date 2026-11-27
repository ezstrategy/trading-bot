"""Shared signal-generation interface: replay one strategy over the data, D1 close by D1 close.

The same function serves the backtest (whole history) and live / paper mode (recent history up
to "now"): it builds nothing from the future because D1 candles are only those final at the data
end (:func:`daily.build_daily`) and each decision uses rows up to the current D1 close.

For every D1 close the strategy's :class:`~.state.BreakoutStateMachine` is updated. A breakout
runs the signal checks; when they pass, the trade is simulated on M5 from the next M5 open
(:func:`trade.simulate_long`). Signals are identified by ``strategy:version:breakout-day`` so the
same breakout can never produce two signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BreakoutConfig
from .daily import bar_sessions, build_daily
from .rules import BreakoutStrategy, daily_atr
from .state import ACTIVE, SIGNAL, BreakoutStateMachine
from .trade import NO_FILL, OPEN, M5Arrays, TradeCosts, TradeResult, simulate_long

GENERATED, BLOCKED, IGNORED = "signal", "blocked", "ignored"


@dataclass
class BreakoutEvent:
    """One breakout of one strategy (a signal, a breakout that failed a check, or one ignored)."""
    kind: str
    key: str
    strategy_id: str
    strategy_name: str
    version: str
    rule_hash: str
    side: str
    day: pd.Timestamp
    session: int
    signal_time: pd.Timestamp
    level: float
    level_note: str
    d1_close: float
    d1_high: float
    atr: float
    checks: list[dict] = field(default_factory=list)
    trade: TradeResult | None = None
    explanation: str = ""
    snapshot: dict = field(default_factory=dict)

    @property
    def failed_checks(self) -> list[str]:
        return [c["detail"] for c in self.checks if c["ok"] is False]

    def to_row(self) -> dict:
        t = self.trade
        row = {"kind": self.kind, "key": self.key, "strategy_id": self.strategy_id, "version": self.version,
               "side": self.side, "day": self.day.strftime("%Y-%m-%d"), "signal_time": self.signal_time,
               "level": self.level, "d1_close": self.d1_close, "atr": self.atr,
               "reason": "; ".join(self.failed_checks) or (t.note if t else "")}
        if t is not None:
            row.update({k: getattr(t, k) for k in ("status", "entry_time", "entry_bid", "entry", "spread",
                                                   "spread_from_data", "stop", "target", "risk", "exit_time",
                                                   "exit", "nights", "swap_nights_weighted", "r_price", "r_swap",
                                                   "r_commission", "r", "r_gross", "r_spread", "r_slippage",
                                                   "swap_included", "mae_r", "mfe_r", "sessions_held")})
        return row


@dataclass
class ScanResult:
    strategy: BreakoutStrategy
    d1: pd.DataFrame
    levels: np.ndarray
    atr: np.ndarray
    events: list[BreakoutEvent]
    machine: BreakoutStateMachine
    as_of: pd.Timestamp

    @property
    def signals(self) -> list[BreakoutEvent]:
        return [e for e in self.events if e.kind == GENERATED]

    def current(self) -> dict:
        """State of the strategy at ``as_of`` for the dashboard."""
        last = len(self.d1) - 1
        lvl = float(self.levels[last]) if last >= 0 and np.isfinite(self.levels[last]) else None
        nxt = None
        if last >= 0:   # level the forming D1 candle will be compared with
            nxt = next_level(self.strategy, self.d1, self.levels)
        close = float(self.d1["close"].iat[last]) if last >= 0 else None
        return {"strategy_id": self.strategy.id, "name": self.strategy.name, "version": self.strategy.version,
                "state": self.machine.state, "state_label": self.machine.label, "armed": self.machine.armed,
                "last_d1_day": None if last < 0 else self.d1["day"].iat[last].strftime("%Y-%m-%d"),
                "last_d1_close": close, "last_level": lvl, "next_level": nxt,
                "distance_to_next_level": None if (nxt is None or close is None) else round(nxt - close, 2),
                "atr": None if last < 0 or not np.isfinite(self.atr[last]) else round(float(self.atr[last]), 2),
                "as_of": self.as_of.isoformat(),
                "history": [{"time": w.isoformat(), "from": a, "to": b, "why": y}
                            for w, a, b, y in self.machine.history[-12:]]}


def next_level(s: BreakoutStrategy, d1: pd.DataFrame, levels: np.ndarray) -> float | None:
    """Level for the D1 candle after the last final one (what the forming day must close above)."""
    if d1.empty:
        return None
    if s.id == "prev_month_high":
        last_day = pd.Timestamp(d1["day"].iat[-1])
        nxt_day = last_day + pd.offsets.BDay(1)
        if nxt_day.month == last_day.month:
            v = levels[-1]
            return float(v) if np.isfinite(v) else None
        m = d1[pd.DatetimeIndex(d1["day"]).to_period("M") == last_day.to_period("M")]
        return float(m["high"].max())
    n = s.params.channel_days
    if len(d1) < n:
        return None
    return float(d1["high"].iloc[-n:].max())


def prepare(m5: pd.DataFrame, cfg: BreakoutConfig, point: float = 0.01,
            as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, M5Arrays]:
    """D1 candles final at ``as_of`` and the M5 arrays with session ordinals."""
    dc = cfg.daily
    if as_of is not None:
        m5 = m5[pd.DatetimeIndex(m5["time"]) + pd.Timedelta(minutes=5) <= pd.Timestamp(as_of)]
    m5 = m5.reset_index(drop=True)
    d1 = build_daily(m5, dc.day_boundary, dc.min_m5_bars_per_day, point, as_of)
    sess = bar_sessions(pd.DatetimeIndex(m5["time"]), d1, dc.day_boundary)
    from .daily import trading_day
    days = pd.DatetimeIndex(d1["day"])
    extra = pd.DatetimeIndex(sorted(set(trading_day(pd.DatetimeIndex(m5["time"]), dc.day_boundary)) - set(days)))
    return d1, M5Arrays.build(m5, sess, days.append(extra), point)


def scan(s: BreakoutStrategy, d1: pd.DataFrame, m: M5Arrays, cfg: BreakoutConfig, costs: TradeCosts,
         as_of: pd.Timestamp | None = None, snapshot_days: int = 25) -> ScanResult:
    """Replay strategy ``s`` over the final D1 candles ``d1`` (see module docstring)."""
    p = s.params
    levels, notes = s.levels(d1, cfg)
    atr = daily_atr(d1, p.atr_period)
    sm = BreakoutStateMachine(p.reset_mode)
    events: list[BreakoutEvent] = []
    active: TradeResult | None = None
    end = pd.Timestamp(as_of) if as_of is not None else (
        m.t[-1] + pd.Timedelta(minutes=5) if len(m.t) else pd.Timestamp.now(tz="UTC"))
    days, closes, highs = pd.DatetimeIndex(d1["day"]), d1["close"].to_numpy(float), d1["high"].to_numpy(float)
    ctimes, complete = pd.DatetimeIndex(d1["close_time"]), d1["complete"].to_numpy(bool)

    def advance(t: pd.Timestamp) -> None:
        nonlocal active
        if active is None:
            return
        if sm.state == SIGNAL and active.entry_time is not None and active.entry_time <= t:
            sm.entry_filled(active.entry_time)
        if active.exit_time is not None and active.exit_time <= t:
            if sm.state == SIGNAL:
                sm.entry_filled(active.entry_time)
            sm.position_closed(active.exit_time, active.status)
            active = None

    for i in range(len(d1)):
        t = ctimes[i]
        advance(t)
        lvl = float(levels[i])
        if not sm.on_d1_close(t, days[i], float(closes[i]), lvl):
            if sm.history and sm.history[-1][0] == t and sm.history[-1][3].startswith("breakout ignored"):
                events.append(_event(IGNORED, s, d1, i, lvl, notes[i], atr, snapshot_days,
                                     [{"check": "no conflicting position", "ok": False,
                                       "detail": "strategy already has a pending/active position"}]))
            continue
        checks = _checks(d1, i, lvl, atr[i], complete[i], cfg)
        ev = _event(GENERATED, s, d1, i, lvl, notes[i], atr, snapshot_days, checks)
        if any(c["ok"] is False for c in checks):
            ev.kind = BLOCKED
            sm.checks_failed(t, "; ".join(ev.failed_checks))
            events.append(ev)
            continue
        tr = simulate_long(m, t, float(atr[i]), p.stop_atr_mult, p.target_r, p.max_sessions, costs,
                           cfg.validation.max_spread)
        ev.trade = tr
        if tr.status == NO_FILL and tr.entry_idx >= 0:          # checks only possible at the entry candle
            ev.kind = BLOCKED
            ev.checks.append({"check": "entry candle checks", "ok": False, "detail": tr.note})
            sm.checks_failed(t, tr.note)
        else:
            ok = tr.status != NO_FILL
            ev.checks.append({"check": "spread within maximum", "ok": True if ok else None,
                              "detail": f"spread {tr.spread:.2f} <= {cfg.validation.max_spread:.2f}" if ok
                              else "checked when the entry candle opens"})
            ev.checks.append({"check": "stop and target on the correct side", "ok": True if ok else None,
                              "detail": f"stop {tr.stop:.2f} < entry {tr.entry:.2f} < target {tr.target:.2f}" if ok
                              else "checked when the entry candle opens"})
            sm.signal_generated(t)
            active = tr
        ev.explanation = explain(s, ev, cfg)
        events.append(ev)
    advance(end)
    if active is not None and active.status == OPEN and active.entry_time is not None and sm.state == SIGNAL:
        sm.entry_filled(active.entry_time)
    return ScanResult(s, d1, levels, atr, events, sm, end)


def _checks(d1: pd.DataFrame, i: int, lvl: float, atr: float, complete: bool, cfg: BreakoutConfig) -> list[dict]:
    n = int(d1["n_bars"].iat[i])
    return [
        {"check": "breakout candle closed", "ok": True,
         "detail": f"D1 {d1['day'].iat[i]:%Y-%m-%d} final at {d1['close_time'].iat[i]:%Y-%m-%d %H:%M} UTC"},
        {"check": "market data valid", "ok": bool(complete),
         "detail": f"{n} M5 candles in the breakout day" + ("" if complete else
                   f" (fewer than {cfg.daily.min_m5_bars_per_day}: missing data)")},
        {"check": "ATR available and > 0", "ok": bool(np.isfinite(atr) and atr > 0),
         "detail": f"ATR {atr:.2f}" if np.isfinite(atr) else "ATR not available yet"},
        {"check": "no duplicate signal", "ok": True,
         "detail": "one signal per strategy and breakout (state machine disarmed until reset)"},
    ]


def _event(kind: str, s: BreakoutStrategy, d1: pd.DataFrame, i: int, lvl: float, note: str, atr: np.ndarray,
           snapshot_days: int, checks: list[dict]) -> BreakoutEvent:
    day = pd.Timestamp(d1["day"].iat[i])
    lo = max(0, i - snapshot_days + 1)
    snap = d1.iloc[lo: i + 1][["day", "open", "high", "low", "close", "n_bars", "spread_median"]].copy()
    snap["day"] = snap["day"].dt.strftime("%Y-%m-%d")
    return BreakoutEvent(kind, f"{s.id}:{s.version}:{day:%Y-%m-%d}", s.id, s.name, s.version, s.params.rule_hash(),
                         "BUY", day, i, pd.Timestamp(d1["close_time"].iat[i]), lvl, str(note),
                         float(d1["close"].iat[i]), float(d1["high"].iat[i]), float(atr[i]), checks,
                         snapshot={"d1": snap.round(3).to_dict(orient="records"), "level": lvl,
                                   "level_rule": s.level_name, "atr": float(atr[i]), "params": s.describe()["params"]})


def explain(s: BreakoutStrategy, ev: BreakoutEvent, cfg: BreakoutConfig) -> str:
    """Plain-language reason for a signal."""
    p, t = s.params, ev.trade
    where = ("the highest high of " + (ev.day.to_period("M") - 1).strftime("%B %Y")
             if s.id == "prev_month_high" else f"the highest high of the previous {p.channel_days} D1 candles")
    txt = (f"{s.name}: the D1 candle of {ev.day:%a %d %b %Y} closed at {ev.d1_close:.2f}, "
           f"{ev.d1_close - ev.level:.2f} above {where} ({ev.level:.2f}). The candle was final at "
           f"{ev.signal_time:%Y-%m-%d %H:%M} UTC; the strategy was armed ({s.reset_rules}). "
           f"D1 ATR({p.atr_period}) = {ev.atr:.2f}, so the stop is {p.stop_atr_mult:g} x ATR = "
           f"{p.stop_atr_mult * ev.atr:.2f} below entry and the target {p.target_r:g}R above it. ")
    if t is not None and t.entry_time is not None and t.status != NO_FILL:
        txt += (f"Entry at the next M5 open {t.entry_time:%Y-%m-%d %H:%M} UTC: bid {t.entry_bid:.2f} + spread "
                f"{t.spread:.2f} + slippage = {t.entry:.2f}; stop {t.stop:.2f}; target {t.target:.2f}. ")
    else:
        txt += "Entry will be the open of the next M5 candle (not available yet). "
    txt += f"If neither is hit, the trade closes at the end of trading session {p.max_sessions}."
    return txt
