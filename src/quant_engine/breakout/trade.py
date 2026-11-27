"""Simulation of one breakout trade on M5 candles (shared by the backtest and paper tracking).

Price convention: candles are BID prices (MT5 charts). A BUY fills at the ask = bid open +
spread, plus slippage. Stops and targets of a long are triggered by the bid. Rules:

* entry at the open of the first M5 candle at or after the breakout D1 close;
* stop = entry - ``stop_atr_mult`` x ATR, target = entry + ``target_r`` x (entry - stop);
* each candle from the entry candle on: if its low reaches the stop the trade is stopped
  (at the stop, or at the open when the candle gaps below it), else if its high reaches the
  target it takes profit at the target. **Stop and target in the same M5 candle = stop first**;
* if neither happened, the trade is closed at the close of the last M5 candle of session
  ``max_sessions`` (the entry session is session 1);
* exits pay slippage (targets too: conservative); swap is charged for every rollover crossed
  (triple on ``swap_triple_weekday``), commission per lot round turn.

Results are in R (multiples of the initial risk ``entry - stop``), so they do not depend on the
lot size; :class:`TradeCosts` converts money costs per lot into R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

OPEN, TP, SL, EXPIRED, NO_FILL = "open", "tp_hit", "sl_hit", "expired", "no_fill"


@dataclass(frozen=True)
class TradeCosts:
    """Cost assumptions for one simulation (all recorded in results)."""
    spread_floor: float = 0.08            # price units
    slippage: float = 0.02                # price units per fill
    commission_per_lot: float = 0.0       # account currency, round turn
    money_per_price_unit_lot: float = 100.0   # account currency per 1.0 price move per 1.00 lot
    swap_long_points: float | None = None     # per lot per night, MT5 points (negative = cost)
    point: float = 0.01
    swap_triple_weekday: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeResult:
    """Outcome of one simulated trade."""
    status: str
    entry_idx: int = -1
    entry_time: pd.Timestamp | None = None
    entry_bid: float = float("nan")
    entry: float = float("nan")
    spread: float = float("nan")
    spread_from_data: bool = False
    stop: float = float("nan")
    target: float = float("nan")
    risk: float = float("nan")
    exit_idx: int = -1
    exit_time: pd.Timestamp | None = None
    exit: float = float("nan")
    nights: int = 0
    swap_nights_weighted: int = 0
    r_price: float = float("nan")          # (exit - entry) / risk: spread and slippage included
    r_swap: float = 0.0
    r_commission: float = 0.0
    r: float = float("nan")                # net R: price + swap - commission
    r_gross: float = float("nan")          # before spread, slippage, swap and commission
    r_spread: float = 0.0
    r_slippage: float = 0.0
    swap_included: bool = False
    mae_r: float = 0.0
    mfe_r: float = 0.0
    sessions_held: int = 0
    note: str = ""

    @property
    def is_final(self) -> bool:
        return self.status in (TP, SL, EXPIRED)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("entry_time", "exit_time"):
            d[k] = None if d[k] is None else pd.Timestamp(d[k]).isoformat()
        return d


@dataclass
class M5Arrays:
    """M5 candles as arrays plus each candle's session ordinal and the sessions' dates."""
    t: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray        # price units, NaN where unknown
    session: np.ndarray       # session ordinal per candle
    session_days: pd.DatetimeIndex   # trading-day date per ordinal (final D1 days, then forming days)

    @classmethod
    def build(cls, m5: pd.DataFrame, session: np.ndarray, session_days: pd.DatetimeIndex,
              point: float) -> "M5Arrays":
        sp = m5["spread"].to_numpy(float) * point if "spread" in m5 else np.full(len(m5), np.nan)
        return cls(pd.DatetimeIndex(m5["time"]), *(m5[k].to_numpy(float, copy=True) for k in ("open", "high", "low", "close")),
                   sp, np.asarray(session, dtype="int64"), pd.DatetimeIndex(session_days))


def entry_index(m: M5Arrays, after: pd.Timestamp) -> int:
    """First M5 candle opening at or after ``after`` (-1 if none yet)."""
    i = int(m.t.searchsorted(pd.Timestamp(after), side="left"))
    return i if i < len(m.t) else -1


def simulate_long(m: M5Arrays, after: pd.Timestamp, atr: float, stop_atr_mult: float, target_r: float,
                  max_sessions: int, costs: TradeCosts, max_spread: float | None = None,
                  side: int = 1) -> TradeResult:
    """Simulate one BUY entered at the first M5 open at/after ``after`` (see module docstring)."""
    if side != 1:
        raise NotImplementedError("SELL strategies are disabled in this release")
    e = entry_index(m, after)
    if e < 0:
        return TradeResult(NO_FILL, note="no M5 candle after the breakout yet")
    raw_sp = m.spread[e]
    from_data = bool(np.isfinite(raw_sp))
    sp = max(raw_sp, costs.spread_floor) if from_data else costs.spread_floor
    res = TradeResult(OPEN, entry_idx=e, entry_time=m.t[e], entry_bid=float(m.o[e]), spread=float(sp),
                      spread_from_data=from_data)
    if max_spread is not None and sp > max_spread:
        res.status, res.note = NO_FILL, f"spread {sp:.2f} above the maximum {max_spread:.2f} at entry"
        return res
    slip = costs.slippage
    entry = float(m.o[e] + sp + slip)
    risk = float(stop_atr_mult * atr)
    stop, target = entry - risk, entry + target_r * risk
    res.entry, res.stop, res.target, res.risk = entry, stop, target, risk
    if not (np.isfinite(risk) and risk > 0 and stop < entry < target):
        res.status, res.note = NO_FILL, "stop/target not on the correct side of entry"
        return res
    last_session = int(m.session[e]) + max_sessions - 1
    n = len(m.t)
    mae = mfe = 0.0
    for k in range(e, n):
        if m.session[k] > last_session:                 # session limit passed: close at previous candle
            _close(res, k - 1, float(m.c[k - 1]) - slip, EXPIRED, m, costs, mae, mfe)
            return res
        lo, hi = m.l[k], m.h[k]
        mae, mfe = max(mae, entry - lo), max(mfe, hi - entry)
        if lo <= stop:
            px = min(float(m.o[k]), stop) if k > e else stop
            _close(res, k, px - slip, SL, m, costs, mae, mfe,
                   "stop and target in the same M5 candle: stop counted first" if hi >= target else "")
            return res
        if hi >= target:
            _close(res, k, target - slip, TP, m, costs, mae, mfe)
            return res
    res.mae_r, res.mfe_r = mae / risk, mfe / risk
    res.sessions_held = int(m.session[n - 1] - m.session[e] + 1)
    res.note = "still open at the end of the data"
    return res


def _close(res: TradeResult, k: int, px: float, status: str, m: M5Arrays, costs: TradeCosts,
           mae: float, mfe: float, note: str = "") -> None:
    e, risk = res.entry_idx, res.risk
    res.status, res.exit_idx, res.exit_time, res.exit = status, k, m.t[k] + pd.Timedelta(minutes=5), float(px)
    s0, s1 = int(m.session[e]), int(m.session[k])
    res.sessions_held = s1 - s0 + 1
    res.nights = s1 - s0
    days = m.session_days
    res.swap_nights_weighted = int(sum(3 if (s < len(days) and days[s].weekday() == costs.swap_triple_weekday) else 1
                                       for s in range(s0, s1)))
    res.r_price = (px - res.entry) / risk
    if costs.swap_long_points is not None:
        res.swap_included = True
        res.r_swap = costs.swap_long_points * costs.point * res.swap_nights_weighted / risk
    res.r_commission = costs.commission_per_lot / (risk * costs.money_per_price_unit_lot) if costs.commission_per_lot else 0.0
    res.r = res.r_price + res.r_swap - res.r_commission
    res.r_spread = res.spread / risk
    res.r_slippage = 2 * costs.slippage / risk
    res.r_gross = res.r_price + res.r_spread + res.r_slippage
    res.mae_r, res.mfe_r = mae / risk, mfe / risk
    if note:
        res.note = note
