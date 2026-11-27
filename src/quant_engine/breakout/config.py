"""Settings of the breakout module (``config/breakout.yaml``), validated with pydantic."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DailyConfig(_Strict):
    """How D1 candles are built from M5."""
    day_boundary: Literal["new_york_17", "utc_midnight"] = "new_york_17"
    min_m5_bars_per_day: int = Field(100, ge=1)
    min_days_in_month: int = Field(15, ge=1)


class StrategyParams(_Strict):
    """Parameters shared by both breakout strategies."""
    enabled: bool = True
    version: str = "1.0.0"
    atr_period: int = Field(14, ge=1)
    stop_atr_mult: float = Field(1.5, gt=0)
    target_r: float = Field(2.0, gt=0)
    max_sessions: int = Field(5, ge=1)
    reset_mode: Literal["new_month", "close_below_level", "new_month_or_close_below"] = "close_below_level"
    channel_days: int = Field(20, ge=2)        # used by the channel strategy only

    def rule_hash(self) -> str:
        """Short hash of every rule parameter (identifies an exact rule set)."""
        d = self.model_dump(exclude={"enabled"})
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class StrategiesConfig(_Strict):
    prev_month_high: StrategyParams = Field(default_factory=lambda: StrategyParams(reset_mode="new_month"))
    channel_20d: StrategyParams = Field(default_factory=StrategyParams)

    @model_validator(mode="after")
    def _channel_reset(self) -> "StrategiesConfig":
        if self.channel_20d.reset_mode != "close_below_level":
            raise ValueError("channel_20d.reset_mode must be close_below_level (its level changes every day)")
        return self


class ValidationConfig(_Strict):
    max_spread: float = Field(1.0, gt=0)
    max_data_age_minutes: float = Field(30, gt=0)
    max_signal_age_hours: float = Field(20, gt=0)


class RiskSettings(_Strict):
    account_equity_fallback: float = Field(500, gt=0)
    per_trade_pct: float = 0.5
    allowed_risk_pct: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0])
    max_combined_open_pct: float = Field(1.0, gt=0)
    one_position_per_strategy: bool = True
    correlation_window_sessions: int = Field(3, ge=0)
    correlated_policy: Literal["primary_only", "split"] = "primary_only"
    primary_order: list[str] = Field(default_factory=lambda: ["prev_month_high", "channel_20d"])
    skip_if_min_lot_over: bool = True

    @model_validator(mode="after")
    def _risk_allowed(self) -> "RiskSettings":
        if self.per_trade_pct not in self.allowed_risk_pct:
            raise ValueError(f"per_trade_pct {self.per_trade_pct} is not one of allowed_risk_pct {self.allowed_risk_pct}")
        if any(p <= 0 or p > 1.0 for p in self.allowed_risk_pct):
            raise ValueError("allowed_risk_pct values must be in (0, 1]")
        if self.per_trade_pct > self.max_combined_open_pct:
            raise ValueError("per_trade_pct is above max_combined_open_pct")
        return self


class CostSettings(_Strict):
    instrument_profile: str = ""
    spread_floor: float = Field(0.08, ge=0)
    slippage_points: float = Field(2.0, ge=0)
    commission_per_lot: float = Field(0.0, ge=0)
    swap_long_points: float | None = None
    swap_triple_weekday: int = Field(2, ge=0, le=6)
    swap_stress_points: float = -50.0


class Period(_Strict):
    name: str
    start: str | None = None
    end: str | None = None


class WalkForward(_Strict):
    train_years: int = Field(3, ge=1)
    test_years: int = Field(1, ge=1)
    candidates: list[dict] = Field(default_factory=list)


class Neighbourhood(_Strict):
    stop_atr_mult: list[float] = Field(default_factory=list)
    target_r: list[float] = Field(default_factory=list)
    max_sessions: list[int] = Field(default_factory=list)


class BacktestSettings(_Strict):
    main_window_start: str = "2023-01-01"
    periods: list[Period] = Field(default_factory=list)
    train_test_split: str = "2023-01-01"
    walk_forward: WalkForward = Field(default_factory=WalkForward)
    equity_accounts: list[float] = Field(default_factory=lambda: [500.0, 10000.0])
    neighbourhood: Neighbourhood = Field(default_factory=Neighbourhood)


class StoreSettings(_Strict):
    path: str = "data/breakout/signals.sqlite"


class BreakoutConfig(_Strict):
    """Whole ``breakout.yaml``."""
    mode: Literal["off", "paper", "live"] = "paper"
    live_review_approved_by: str = ""
    daily: DailyConfig = Field(default_factory=DailyConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    reports_dir: str = "reports/breakout"

    @field_validator("live_review_approved_by")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()

    @model_validator(mode="after")
    def _review_gate(self) -> "BreakoutConfig":
        # Owner spec section 10: live signal generation only after the review of the reproduction.
        if self.mode == "live" and not self.live_review_approved_by:
            raise ValueError("mode: live needs live_review_approved_by (the owner's review of the backtest "
                             "reproduction, unit tests and paper mode); use mode: paper until then")
        return self

    def strategy_params(self, strategy_id: str) -> StrategyParams:
        """Parameters of one strategy by id."""
        try:
            return getattr(self.strategies, strategy_id)
        except AttributeError as exc:
            raise KeyError(f"unknown breakout strategy {strategy_id!r}") from exc


def load_breakout_config(config_dir: str | Path | None = None) -> BreakoutConfig:
    """Read ``breakout.yaml`` (defaults when the file is missing)."""
    root = Path(config_dir or os.getenv("QE_CONFIG_DIR", "config"))
    p = root / "breakout.yaml"
    if not p.exists():
        return BreakoutConfig()
    try:
        return BreakoutConfig(**(yaml.safe_load(p.read_text(encoding="utf-8")) or {}))
    except Exception as exc:
        raise ConfigError(f"Invalid {p}: {exc}") from exc
