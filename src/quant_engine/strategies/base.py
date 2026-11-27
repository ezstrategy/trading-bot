"""Strategy plug-in contract.

A strategy is a class with :class:`StrategyMeta` (unique id, version, direction, rules text)
and a pydantic ``Params`` model. The engine calls :meth:`Strategy.prepare` once with all market
data, then :meth:`Strategy.on_bar` at every M5 close. ``on_bar`` returns :class:`OrderIntent`
objects; it never fills orders, sizes positions or touches equity: that is the engine's job, so
strategy logic stays out of the backtester and every strategy is simulated the same way.

Changing any rule of a strategy means a new ``version``.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from ..backtest.market import MarketData, MarketView


@dataclass(frozen=True)
class StrategyMeta:
    """Identity and description of a strategy."""

    id: str
    version: str
    name: str
    direction: str                      # "long", "short" or "both"
    family: str = ""                    # strategies that share a hypothesis (test-set reuse is tracked per family)
    entry_rules: str = ""
    exit_rules: str = ""
    stop_rules: str = ""
    target_rules: str = ""
    session_rules: str = ""
    regime_rules: str = ""
    risk_rules: str = "0.5% of equity per trade (engine), lots rounded down to the broker step"


@dataclass
class OrderIntent:
    """A market order to open at the next candle's open.

    Exactly one of ``target`` (price) or ``target_r`` (multiple of the filled risk) should be set,
    or neither for no target. ``risk_bounds`` = (min, max) allowed stop distance in price at the
    fill; outside it the order is not opened (recorded as skipped).
    """

    side: int                           # +1 buy, -1 sell
    stop: float
    target: float | None = None
    target_r: float | None = None
    time_exit_bars: int | None = None   # close at the close of the N-th candle held (fill candle = 1)
    risk_bounds: tuple[float, float] | None = None
    session: str = ""
    reason: dict = field(default_factory=dict)
    strategy_id: str = ""               # set by the engine


class StrategyParams(BaseModel):
    """Base for strategy parameters: unknown keys are rejected."""
    model_config = ConfigDict(extra="forbid")


class Strategy(ABC):
    """Base class of every strategy plug-in."""

    meta: ClassVar[StrategyMeta]
    Params: ClassVar[type[StrategyParams]] = StrategyParams

    def __init__(self, params: dict | None = None) -> None:
        self.params = self.Params(**(params or {}))
        self.skips: dict[str, int] = {}

    # ------------------------------------------------------------- contract
    @abstractmethod
    def prepare(self, data: MarketData) -> None:
        """Precompute causal indicators / setups for all candles (called once)."""

    @abstractmethod
    def on_bar(self, view: MarketView) -> list[OrderIntent]:
        """Intents decided at the close of base candle ``view.i``."""

    def allowed_regimes(self) -> set[str] | None:
        """H4 regimes this strategy may trade in (None = any); enforced by the engine."""
        return None

    # ------------------------------------------------------------- helpers
    def skip(self, why: str) -> None:
        """Count a setup that was rejected by the strategy's own rules."""
        self.skips[why] = self.skips.get(why, 0) + 1

    def param_dict(self) -> dict:
        """Parameters as plain JSON values."""
        return json.loads(self.params.model_dump_json())

    def fingerprint(self) -> str:
        """Hash of id, version and parameters (identifies an exact rule set)."""
        s = json.dumps({"id": self.meta.id, "version": self.meta.version, "params": self.param_dict()}, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def describe(self) -> dict:
        """Everything a reader needs to know about the rule set."""
        m = self.meta
        return {"id": m.id, "version": m.version, "name": m.name, "direction": m.direction, "family": m.family,
                "entry": m.entry_rules, "exit": m.exit_rules, "stop_loss": m.stop_rules,
                "take_profit": m.target_rules, "sessions": m.session_rules, "regime": m.regime_rules,
                "risk": m.risk_rules, "params": self.param_dict(), "fingerprint": self.fingerprint(),
                "allowed_regimes": sorted(self.allowed_regimes()) if self.allowed_regimes() else "any"}
