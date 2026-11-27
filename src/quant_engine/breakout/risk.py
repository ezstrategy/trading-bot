"""Shared risk-management engine for breakout signals (live, paper and the sized backtest).

Position size comes from the broker's symbol specification (contract size, tick size, tick value,
lot limits; :class:`~quant_engine.instruments.InstrumentSpec`), never from an assumed point value.
Rules, in order:

1. the requested risk must be one of ``allowed_risk_pct`` (0.25 / 0.5 / 1 by default);
2. one active position per strategy;
3. signals of different strategies whose breakout candles are within
   ``correlation_window_sessions`` of each other are ONE correlated exposure: together they never
   risk more than ``per_trade_pct`` (``primary_only``: the primary strategy keeps it and the other
   is cancelled; ``split``: the budget is shared);
4. all open and pending breakout positions together stay within ``max_combined_open_pct``;
5. lots are rounded DOWN to the lot step; if even the minimum lot would exceed the risk, the signal
   is skipped (the stop is never moved and the risk never raised to make it fit).

No martingale, grid, averaging down or lot escalation: the size depends only on equity, the stop
distance and the fixed risk percentage, never on previous results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..instruments import InstrumentSpec
from .config import RiskSettings


@dataclass
class Exposure:
    """An open or pending breakout position as the risk engine sees it."""
    strategy_id: str
    session: int               # session ordinal of the breakout D1 candle
    risk_pct: float            # actual % of equity at risk (after rounding the lots)
    signal_key: str = ""


@dataclass
class RiskCandidate:
    strategy_id: str
    session: int
    entry: float
    stop: float
    extra_cost_per_unit: float = 0.0    # price-unit costs not inside entry/stop (exit slippage)
    signal_key: str = ""


@dataclass
class RiskDecision:
    approved: bool
    lots: float = 0.0
    risk_pct: float = 0.0               # actual % of equity at risk with these lots
    risk_money: float = 0.0
    budget_pct: float = 0.0             # % the engine allowed before rounding
    reason: str = ""
    correlated_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RiskEngine:
    """Applies :class:`RiskSettings` to candidates against the current exposures."""

    def __init__(self, settings: RiskSettings, spec: InstrumentSpec) -> None:
        self.s, self.spec = settings, spec

    def loss_per_lot(self, entry: float, stop: float, extra_cost_per_unit: float = 0.0) -> float:
        """Account-currency loss of 1.00 lot at the stop, costs included."""
        return ((entry - stop) + max(extra_cost_per_unit, 0.0)) * self.spec.money_per_price_unit(1.0) \
            + self.spec.commission_per_lot

    def assess(self, c: RiskCandidate, open_exp: list[Exposure], equity: float,
               risk_pct: float | None = None, same_batch: list[RiskCandidate] | None = None) -> RiskDecision:
        """Decide one candidate. ``same_batch`` = other candidates generated at the same moment
        (used by the ``split`` policy to share the budget)."""
        s = self.s
        want = s.per_trade_pct if risk_pct is None else risk_pct
        if want not in s.allowed_risk_pct:
            return RiskDecision(False, reason=f"risk {want}% is not one of the allowed levels {s.allowed_risk_pct}")
        if equity <= 0:
            return RiskDecision(False, reason="no account equity")
        if c.entry - c.stop <= 0:
            return RiskDecision(False, reason="stop is not below entry")
        if s.one_position_per_strategy and any(x.strategy_id == c.strategy_id for x in open_exp):
            return RiskDecision(False, reason="strategy already has an active position")
        corr = [x for x in open_exp if x.strategy_id != c.strategy_id
                and abs(x.session - c.session) <= s.correlation_window_sessions]
        corr_ids = sorted({x.strategy_id for x in corr})
        budget = want
        if corr:
            if s.correlated_policy == "primary_only":
                return RiskDecision(False, reason=f"correlated with {', '.join(corr_ids)} on the same breakout: "
                                                  "risk not doubled", correlated_with=corr_ids)
            budget = min(want, s.per_trade_pct - sum(x.risk_pct for x in corr))
        peers = [p for p in (same_batch or []) if p.strategy_id != c.strategy_id
                 and abs(p.session - c.session) <= s.correlation_window_sessions]
        if peers and s.correlated_policy == "split":
            budget = min(budget, s.per_trade_pct / (len(peers) + 1))
            corr_ids = sorted(set(corr_ids) | {p.strategy_id for p in peers})
        budget = min(budget, s.max_combined_open_pct - sum(x.risk_pct for x in open_exp))
        if budget <= 1e-9:
            return RiskDecision(False, reason="combined open risk limit reached "
                                              f"({s.max_combined_open_pct}% of equity)", correlated_with=corr_ids)
        per_lot = self.loss_per_lot(c.entry, c.stop, c.extra_cost_per_unit)
        lots = self.spec.round_lot(equity * budget / 100.0 / per_lot)
        if lots <= 0:
            min_pct = 100 * self.spec.min_lot * per_lot / equity
            if s.skip_if_min_lot_over:
                return RiskDecision(False, budget_pct=budget, correlated_with=corr_ids,
                                    reason=f"skipped: minimum lot {self.spec.min_lot:g} would risk {min_pct:.2f}% "
                                           f"(limit {budget:.2f}%)")
            lots = self.spec.min_lot
        money = lots * per_lot
        return RiskDecision(True, lots, 100 * money / equity, money, budget, "ok", corr_ids)

    def batch(self, cands: list[RiskCandidate], open_exp: list[Exposure], equity: float,
              risk_pct: float | None = None) -> list[RiskDecision]:
        """Decide candidates generated at the same moment, primary strategies first."""
        order = {sid: i for i, sid in enumerate(self.s.primary_order)}
        idx = sorted(range(len(cands)), key=lambda i: order.get(cands[i].strategy_id, 99))
        exp = list(open_exp)
        out: list[RiskDecision | None] = [None] * len(cands)
        for i in idx:
            d = self.assess(cands[i], exp, equity, risk_pct, same_batch=cands)
            out[i] = d
            if d.approved:
                exp.append(Exposure(cands[i].strategy_id, cands[i].session, d.risk_pct, cands[i].signal_key))
        return out  # type: ignore[return-value]
