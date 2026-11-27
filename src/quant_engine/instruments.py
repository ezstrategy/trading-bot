"""Broker-independent instrument specifications (Quant Lab module 2).

Profiles live in ``config/instruments.yaml``. Each holds what a backtest or a signal needs to turn
prices into money and lots: contract size, lot limits and step, tick size and value, digits,
currencies, commission and execution assumptions. Values the owner has not confirmed are listed
by :meth:`InstrumentSpec.guesses` so reports can say so.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .config import ConfigError

SPEC_FIELDS = ("contract_size", "min_lot", "max_lot", "lot_step", "tick_size", "tick_value", "digits",
               "account_currency", "profit_currency", "commission_per_lot", "typical_spread")


class ExecutionAssumptions(BaseModel):
    """How simulated fills are made."""
    fill: Literal["next_bar_open"] = "next_bar_open"
    slippage_points: float = Field(2.0, ge=0)
    intrabar: Literal["m1_then_stop_first", "stop_first"] = "m1_then_stop_first"
    price_side: Literal["bid", "mid", "ask"] = "bid"


class InstrumentSpec(BaseModel):
    """One broker's specification of one instrument."""
    name: str = ""
    symbol: str
    broker_symbol: str = ""
    broker: str = "unknown"
    contract_size: float = Field(gt=0)
    min_lot: float = Field(gt=0)
    max_lot: float = Field(gt=0)
    lot_step: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    digits: int = Field(ge=0, le=10)
    account_currency: str = "USD"
    profit_currency: str = "USD"
    commission_per_lot: float = Field(0.0, ge=0)
    typical_spread: float | None = Field(None, ge=0)
    swap_long_points: float | None = None
    swap_short_points: float | None = None
    execution: ExecutionAssumptions = Field(default_factory=ExecutionAssumptions)
    confirmed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "InstrumentSpec":
        if self.min_lot > self.max_lot:
            raise ValueError("min_lot is above max_lot")
        bad = [c for c in self.confirmed if c not in SPEC_FIELDS and c not in self.model_fields]
        if bad:
            raise ValueError(f"unknown confirmed field(s) {bad}")
        return self

    # ----------------------------------------------------------------- maths
    @property
    def point(self) -> float:
        """Price of one point (10^-digits)."""
        return 10.0 ** -self.digits

    def money_per_price_unit(self, lots: float) -> float:
        """Account-currency P/L of a 1.0 price move for ``lots`` (tick value / tick size)."""
        return lots * self.tick_value / self.tick_size

    def pnl(self, side: int, entry: float, exit_: float, lots: float) -> float:
        """Gross P/L in account currency (``side`` +1 buy, -1 sell), before commission."""
        return side * (exit_ - entry) * self.money_per_price_unit(lots)

    def commission(self, lots: float) -> float:
        """Round-turn commission for ``lots``."""
        return lots * self.commission_per_lot

    def round_lot(self, lots: float) -> float:
        """Round DOWN to the lot step and cap at ``max_lot``; 0 if below ``min_lot``."""
        if not math.isfinite(lots) or lots <= 0:
            return 0.0
        steps = math.floor(lots / self.lot_step + 1e-9)
        out = min(round(steps * self.lot_step, 8), self.max_lot)
        return out if out >= self.min_lot - 1e-12 else 0.0

    def lots_for_risk(self, equity: float, risk_pct: float, stop_distance: float) -> float:
        """Lots that lose at most ``risk_pct`` of ``equity`` at a stop ``stop_distance`` away (0 if
        even the minimum lot would risk more)."""
        if stop_distance <= 0:
            return 0.0
        return self.round_lot(equity * risk_pct / 100.0 / self.money_per_price_unit(1.0) / stop_distance)

    def consistency_warnings(self) -> list[str]:
        """Plain-language warnings about values that do not fit together."""
        w = []
        if self.profit_currency == self.account_currency:
            expect = self.contract_size * self.tick_size
            if abs(expect - self.tick_value) > 1e-9 * max(1.0, expect):
                w.append(f"tick_value {self.tick_value} differs from contract_size x tick_size = {expect:g} "
                         "although profit and account currency are the same")
        else:
            w.append(f"account currency {self.account_currency} differs from profit currency "
                     f"{self.profit_currency}: tick_value changes with the exchange rate")
        if abs(round(self.tick_size / self.point) * self.point - self.tick_size) > 1e-12:
            w.append(f"tick_size {self.tick_size} is not a multiple of the point {self.point:g}")
        return w

    def guesses(self) -> list[str]:
        """Specification fields the owner has not confirmed."""
        return [f for f in SPEC_FIELDS if f not in self.confirmed]

    @classmethod
    def from_symbol_info(cls, info: Any, account_currency: str, name: str = "", broker: str = "",
                         base: "InstrumentSpec | None" = None) -> "InstrumentSpec":
        """Build a spec from an MT5 ``symbol_info`` record (read-only data; nothing is traded).

        Fields MT5 does not report (commission, execution assumptions), or that this record lacks,
        are copied from ``base`` and stay unconfirmed.
        """
        b = base.model_dump() if base else {}
        mt5_fields = {"contract_size": "trade_contract_size", "min_lot": "volume_min", "max_lot": "volume_max",
                      "lot_step": "volume_step", "tick_size": "trade_tick_size", "tick_value": "trade_tick_value",
                      "digits": "digits", "profit_currency": "currency_profit"}
        got = {}
        for ours, theirs in mt5_fields.items():
            v = getattr(info, theirs, None)
            if v is not None and v != "":
                got[ours] = v if ours == "profit_currency" else (int(v) if ours == "digits" else float(v))
        if account_currency:
            got["account_currency"] = account_currency
        # swap per lot per night, only when the broker quotes it in points (SYMBOL_SWAP_MODE_POINTS = 1)
        if getattr(info, "swap_mode", None) == 1:
            for ours, theirs in (("swap_long_points", "swap_long"), ("swap_short_points", "swap_short")):
                v = getattr(info, theirs, None)
                if v is not None:
                    got[ours] = float(v)
        confirmed = sorted(set(b.get("confirmed", [])) | set(got))
        return cls(**{**b, **got, "name": name or b.get("name", ""), "broker": broker or b.get("broker", "unknown"),
                      "symbol": b.get("symbol", getattr(info, "name", "")),
                      "broker_symbol": getattr(info, "name", b.get("broker_symbol", "")), "confirmed": confirmed})


class InstrumentBook(BaseModel):
    """All profiles from ``instruments.yaml``."""
    default_profile: str
    profiles: dict[str, InstrumentSpec]

    @model_validator(mode="after")
    def _default_exists(self) -> "InstrumentBook":
        if self.default_profile not in self.profiles:
            raise ValueError(f"default_profile {self.default_profile!r} is not in profiles")
        for k, p in self.profiles.items():
            p.name = p.name or k
        return self

    def get(self, name: str | None = None) -> InstrumentSpec:
        """A profile by name (default profile when ``name`` is empty)."""
        key = name or self.default_profile
        if key not in self.profiles:
            raise ConfigError(f"No instrument profile {key!r}; known: {sorted(self.profiles)}")
        return self.profiles[key]


def load_instruments(config_dir: str | Path | None = None) -> InstrumentBook:
    """Read and validate ``instruments.yaml`` plus any ``instruments.<name>.yaml`` files (profiles
    written by ``mt5-instrument``); a later file cannot redefine an existing profile name."""
    import os
    root = Path(config_dir or os.getenv("QE_CONFIG_DIR", "config"))
    p = root / "instruments.yaml"
    if not p.exists():
        raise ConfigError(f"Missing config file: {p}")
    try:
        book = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for extra in sorted(root.glob("instruments.*.yaml")):
            more = (yaml.safe_load(extra.read_text(encoding="utf-8")) or {}).get("profiles", {})
            clash = set(more) & set(book.get("profiles", {}))
            if clash:
                raise ValueError(f"{extra.name} redefines profile(s) {sorted(clash)}")
            book.setdefault("profiles", {}).update(more)
        return InstrumentBook(**book)
    except Exception as exc:
        raise ConfigError(f"Invalid instrument profiles: {exc}") from exc


def write_profile(spec: InstrumentSpec, config_dir: str | Path, note: str = "") -> Path:
    """Save one profile as ``config/instruments.<name>.yaml`` (refuses to overwrite)."""
    root = Path(config_dir)
    p = root / f"instruments.{spec.name}.yaml"
    if p.exists():
        raise FileExistsError(f"{p} exists; choose another --profile name (profiles are not overwritten)")
    body = {"profiles": {spec.name: spec.model_dump(exclude={"name"})}}
    head = f"# Written by `quant_engine mt5-instrument` {note}\n# Fields in `confirmed` came from MT5; the rest are copied guesses.\n"
    p.write_text(head + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return p
