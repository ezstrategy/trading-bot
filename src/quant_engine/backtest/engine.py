"""Chronological, event-driven backtest engine (Quant Lab module 5).

For every M5 candle ``i`` in time order:

1. Orders queued at the close of candle ``i - 1`` are filled at the **open of candle i** (next-bar
   execution). Buys pay the ask (bid + spread) and slippage; sells get the bid minus slippage.
   If candle ``i`` does not directly follow ``i - 1`` (a data gap) the order is cancelled.
2. Open positions are checked against candle ``i``: stop and target, using the M1 candles inside
   it when they exist; otherwise, if both could have been touched, the **stop is assumed first**.
   A candle that opens beyond the stop fills at its open. Shorts are stopped / take profit on the
   ask (bid + spread). Then time exits are applied at the candle's close.
3. At the close of candle ``i`` the H4 regime known at that moment is updated and each strategy's
   ``on_bar`` is called. Intents are rejected when the strategy is not allowed in the regime, when
   the strategy (or the account, ``max_concurrent``) already has a position, or when an identical
   intent was already given on this candle.

Position size is ``risk_pct`` of current equity at the filled stop distance, rounded down to the
broker's lot step (:class:`~quant_engine.instruments.InstrumentSpec`). Commission is charged per
lot. Every trade records gross (before spread, slippage and commission) and net results.
Strategies never see fills or equity, so they cannot adapt to results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..instruments import InstrumentSpec
from ..logging_setup import get_logger
from ..research.regime4 import regime4_series
from ..timeframes import Timeframe
from .market import BASE, MarketData, MarketView
from .mtf import align

log = get_logger(__name__)


@dataclass
class ExecutionConfig:
    """Execution and cost assumptions of one run (all recorded with the results)."""

    risk_pct: float = 0.5
    initial_equity: float = 10_000.0
    spread_floor: float | None = None          # price units; default = instrument typical_spread
    spread_multiplier: float = 1.0             # cost stress
    slippage_points: float | None = None       # per fill; default = instrument execution.slippage_points
    slippage_on_targets: bool = True           # conservative: limit exits also slip
    commission_per_lot: float | None = None    # round turn; default = instrument commission_per_lot
    use_m1: bool = True                        # resolve intrabar order with M1 when available
    max_concurrent: int = 1                    # open positions across all strategies
    regime_config: dict | None = None          # Regime4Config overrides

    def resolved(self, spec: InstrumentSpec) -> dict:
        """Values actually used, with instrument defaults filled in."""
        d = asdict(self)
        d["spread_floor"] = (spec.typical_spread or 0.0) if self.spread_floor is None else self.spread_floor
        d["slippage_points"] = spec.execution.slippage_points if self.slippage_points is None else self.slippage_points
        d["commission_per_lot"] = spec.commission_per_lot if self.commission_per_lot is None else self.commission_per_lot
        d["fill"] = spec.execution.fill
        d["intrabar"] = "m1_then_stop_first" if self.use_m1 else "stop_first"
        d["price_side"] = spec.execution.price_side
        return d


@dataclass
class _Pending:
    intent: object
    signal_idx: int


@dataclass
class _Position:
    strategy_id: str
    side: int
    signal_idx: int
    fill_idx: int
    entry: float
    entry_raw: float
    stop: float
    target: float | None
    time_exit_bars: int | None
    lots: float
    risk_price: float
    risk_money: float
    equity_before: float
    session: str
    reason: dict
    regime: str
    spread_entry: float
    held: int = 0
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`."""

    trades: pd.DataFrame
    equity: pd.DataFrame
    signals: pd.DataFrame
    skips: dict[str, dict[str, int]]
    execution: dict
    strategies: list[dict]
    data: dict = field(default_factory=dict)


TRADE_COLUMNS = ["trade_id", "strategy_id", "strategy_version", "side", "session", "regime", "signal_time",
                 "entry_time", "entry", "stop", "target", "exit_time", "exit", "exit_reason", "intrabar",
                 "bars_held", "lots", "risk_price", "risk_money", "equity_before", "equity_after",
                 "gross_pnl", "spread_cost", "slippage_cost", "commission", "net_pnl", "r_gross", "r_after_spread",
                 "r", "mae_r", "mfe_r", "spread_assumed", "reason"]


def _regimes(data: MarketData, cfg: ExecutionConfig) -> np.ndarray:
    if not data.has(Timeframe.H4):
        return np.full(len(data), "unknown", dtype=object)
    from ..config import Regime4Config
    h4 = data.frame(Timeframe.H4)
    reg = regime4_series(h4.df, Regime4Config(**(cfg.regime_config or {}))).to_numpy(dtype=object)
    return align(reg, data.closed_htf_index(Timeframe.H4), fill="uncertain")


def run_backtest(data: MarketData, strategies: list, spec: InstrumentSpec,
                 cfg: ExecutionConfig | None = None, trade_from: pd.Timestamp | None = None,
                 trade_until: pd.Timestamp | None = None) -> BacktestResult:
    """Simulate ``strategies`` over ``data``.

    ``trade_from`` / ``trade_until`` limit which candles may *generate signals* (UTC decision times
    in ``[trade_from, trade_until)``); earlier candles still warm up the indicators and later ones
    let open trades finish. This is how development / validation / test parts are run.
    """
    cfg = cfg or ExecutionConfig()
    ex = cfg.resolved(spec)
    b = data.base
    n = len(b)
    pt = spec.tick_size if spec.tick_size else data.point
    slip = ex["slippage_points"] * pt
    slip_t = slip if cfg.slippage_on_targets else 0.0
    comm_lot = ex["commission_per_lot"]
    mpu1 = spec.money_per_price_unit(1.0)
    raw_sp = b.spread_pts * data.point
    assumed = ~np.isfinite(raw_sp)
    sp = np.maximum(np.where(assumed, ex["spread_floor"], raw_sp), ex["spread_floor"]) * cfg.spread_multiplier
    o, h, l, c, t = b.o, b.h, b.l, b.c, b.t
    step = BASE.delta
    contiguous = np.r_[False, np.diff(t.as_unit("ns").asi8) == step.value]
    m1a, m1b = data.m1_slices() if cfg.use_m1 else (np.zeros(n, "int64"), np.zeros(n, "int64"))
    m1 = data.frame(Timeframe.M1) if cfg.use_m1 and data.has(Timeframe.M1) else None
    regime = _regimes(data, cfg)
    decision = t + step
    allow_from = pd.Timestamp(trade_from) if trade_from is not None else None
    allow_until = pd.Timestamp(trade_until) if trade_until is not None else None

    for s in strategies:
        s.prepare(data)
    by_id = {s.meta.id: s for s in strategies}
    if len(by_id) != len(strategies):
        raise ValueError("each strategy may be used once per run")
    skips: dict[str, dict[str, int]] = {sid: {} for sid in by_id}

    def skip(sid: str, why: str) -> None:
        skips[sid][why] = skips[sid].get(why, 0) + 1

    equity = cfg.initial_equity
    pending: list[_Pending] = []
    open_pos: list[_Position] = []
    trades: list[dict] = []
    signals: list[dict] = []
    eq_rows = [{"time": t[0], "equity": equity, "trade_id": None}] if n else []

    def close(p: _Position, k: int, px: float, reason: str, how: str, when: pd.Timestamp, exit_slip: float) -> None:
        """Book the trade. ``px`` is the filled exit price (spread and slippage included)."""
        nonlocal equity
        side = p.side
        net_px = side * (px - p.entry) * mpu1 * p.lots
        comm = comm_lot * p.lots
        net = net_px - comm
        spread_cost = (p.spread_entry if side == 1 else sp[k]) * mpu1 * p.lots   # buys pay at entry, sells at exit
        slip_cost = (slip + exit_slip) * mpu1 * p.lots
        gross = net + spread_cost + slip_cost + comm
        equity_before = equity
        equity += net
        tid = len(trades) + 1
        trades.append({
            "trade_id": tid, "strategy_id": p.strategy_id, "strategy_version": by_id[p.strategy_id].meta.version,
            "side": "BUY" if side == 1 else "SELL", "session": p.session, "regime": p.regime,
            "signal_time": decision[p.signal_idx], "entry_time": t[p.fill_idx], "entry": round(p.entry, 5),
            "stop": round(p.stop, 5), "target": None if p.target is None else round(p.target, 5),
            "exit_time": when, "exit": round(px, 5), "exit_reason": reason, "intrabar": how, "bars_held": p.held,
            "lots": p.lots, "risk_price": p.risk_price, "risk_money": p.risk_money, "equity_before": equity_before,
            "equity_after": equity, "gross_pnl": gross, "spread_cost": spread_cost, "slippage_cost": slip_cost,
            "commission": comm, "net_pnl": net, "r_gross": gross / p.risk_money,
            "r_after_spread": (gross - spread_cost) / p.risk_money, "r": net / p.risk_money,
            "mae_r": p.mae / p.risk_price, "mfe_r": p.mfe / p.risk_price, "spread_assumed": bool(assumed[p.fill_idx]),
            "reason": p.reason})
        eq_rows.append({"time": when, "equity": equity, "trade_id": tid})

    def check_bar(p: _Position, k: int):
        """Exit on candle k as (price, reason, intrabar method, exit time, exit slippage), or None."""
        side, stop, tp = p.side, p.stop, p.target
        both_m5 = _touch(side, stop, tp, o[k], h[k], l[k], sp[k]) == 2
        use_m1 = m1 is not None and m1b[k] > m1a[k]
        if use_m1:
            subs = [(m1.o[j], m1.h[j], m1.l[j], m1.t[j] + Timeframe.M1.delta) for j in range(m1a[k], m1b[k])]
        else:
            subs = [(o[k], h[k], l[k], t[k] + step)]
        for (bo, bh, bl, bclose) in subs:
            hit_stop = (bl <= stop) if side == 1 else (bh + sp[k] >= stop)
            hit_tp = tp is not None and ((bh >= tp) if side == 1 else (bl + sp[k] <= tp))
            p.mae = max(p.mae, (p.entry - bl) if side == 1 else (bh + sp[k] - p.entry))
            p.mfe = max(p.mfe, (bh - p.entry) if side == 1 else (p.entry - bl - sp[k]))
            if hit_stop:
                if use_m1:
                    how = "m1_stop_first" if hit_tp else ("m1" if both_m5 else "single")
                else:
                    how = "stop_first" if both_m5 else "single"
                px = (min(bo, stop) - slip) if side == 1 else (max(bo + sp[k], stop) + slip)
                return px, "stop", how, bclose, slip
            if hit_tp:
                how = "m1" if (use_m1 and both_m5) else "single"
                px = (max(bo, tp) - slip_t) if side == 1 else (min(bo + sp[k], tp) + slip_t)
                return px, "target", how, bclose, slip_t
        return None

    for i in range(n):
        # 1. fills at the open of candle i
        if pending:
            queued, pending = pending, []
            for q in queued:
                it, sid = q.intent, q.intent.strategy_id
                if not contiguous[i]:
                    skip(sid, "data gap before fill")
                    continue
                side = it.side
                entry = o[i] + sp[i] + slip if side == 1 else o[i] - slip
                entry_raw = o[i]
                risk_price = (entry - it.stop) * side
                if risk_price <= 0:
                    skip(sid, "stop on the wrong side of the fill")
                    continue
                if it.risk_bounds and not (it.risk_bounds[0] <= risk_price <= it.risk_bounds[1]):
                    skip(sid, "stop size outside limits")
                    continue
                lots = spec.lots_for_risk(equity, cfg.risk_pct, risk_price)
                if lots <= 0:
                    skip(sid, "lot below broker minimum for this risk")
                    continue
                target = it.target if it.target is not None else (
                    entry + side * it.target_r * risk_price if it.target_r is not None else None)
                open_pos.append(_Position(sid, side, q.signal_idx, i, entry, entry_raw, it.stop, target,
                                          it.time_exit_bars, lots, risk_price, lots * risk_price * mpu1, equity,
                                          it.session, it.reason, str(regime[q.signal_idx]), sp[i]))
        # 2. manage positions on candle i
        still = []
        for p in open_pos:
            res = check_bar(p, i)
            p.held += 1
            if res is not None:
                close(p, i, *res)
                continue
            if p.time_exit_bars is not None and p.held >= p.time_exit_bars:
                px = (c[i] - slip) if p.side == 1 else (c[i] + sp[i] + slip)
                close(p, i, px, "time", "close", t[i] + step, slip)
                continue
            still.append(p)
        open_pos = still
        # 3. decisions at the close of candle i
        if (allow_from is not None and decision[i] < allow_from) or (allow_until is not None and decision[i] >= allow_until):
            continue
        view = MarketView(data, i, str(regime[i]))
        seen: set = set()
        for s in strategies:
            sid = s.meta.id
            for it in s.on_bar(view) or []:
                it.strategy_id = sid
                key = (sid, it.side, round(it.stop, 6), it.target, it.target_r)
                row = {"time": decision[i], "strategy_id": sid, "side": "BUY" if it.side == 1 else "SELL",
                       "stop": it.stop, "target": it.target, "target_r": it.target_r, "session": it.session,
                       "regime": str(regime[i]), "outcome": "queued"}
                allowed = s.allowed_regimes()
                if key in seen:
                    row["outcome"] = "duplicate"
                elif allowed is not None and str(regime[i]) not in allowed:
                    row["outcome"] = "blocked: regime"
                elif any(p.strategy_id == sid for p in open_pos) or any(q.intent.strategy_id == sid for q in pending):
                    row["outcome"] = "skipped: strategy already in a trade"
                elif len(open_pos) + len(pending) >= cfg.max_concurrent:
                    row["outcome"] = "skipped: max concurrent positions"
                else:
                    pending.append(_Pending(it, i))
                seen.add(key)
                if row["outcome"] != "queued":
                    skip(sid, row["outcome"])
                signals.append(row)
    for p in open_pos:                        # data ended with a position open
        close(p, n - 1, (c[-1] - slip) if p.side == 1 else (c[-1] + sp[-1] + slip), "end_of_data", "close",
              t[-1] + step, slip)
    for q in pending:
        skip(q.intent.strategy_id, "no candle left to fill")

    for s in strategies:
        for k, v in s.skips.items():
            skips[s.meta.id]["rule: " + k] = v
    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    eq = pd.DataFrame(eq_rows, columns=["time", "equity", "trade_id"])
    sig = pd.DataFrame(signals, columns=["time", "strategy_id", "side", "stop", "target", "target_r", "session",
                                         "regime", "outcome"])
    info = {"dataset_ids": data.dataset_ids, "first_utc": str(t[0]) if n else None, "last_utc": str(t[-1]) if n else None,
            "m5_candles": n, "m1_available": m1 is not None, "spread_assumed_share": float(assumed.mean()) if n else None,
            "trade_from": None if allow_from is None else str(allow_from),
            "trade_until": None if allow_until is None else str(allow_until)}
    log.info("backtest_done", extra={"trades": len(tr), "strategies": list(by_id)})
    return BacktestResult(tr, eq, sig, skips, ex, [s.describe() for s in strategies], info)


def _touch(side: int, stop: float, tp: float | None, o: float, h: float, l: float, sp: float) -> int:
    """How many of stop / target the whole candle reaches (0, 1 or 2)."""
    hs = (l <= stop) if side == 1 else (h + sp >= stop)
    ht = tp is not None and ((h >= tp) if side == 1 else (l + sp <= tp))
    return int(hs) + int(ht)
