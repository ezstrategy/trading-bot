"""Broker symbol discovery and canonical-symbol mapping."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..config import SymbolMapping
from ..logging_setup import get_logger

log = get_logger(__name__)


class SymbolResolutionError(LookupError):
    """The canonical instrument could not be mapped to exactly one broker symbol."""


@dataclass(frozen=True)
class BrokerSymbol:
    """The subset of MT5 symbol_info the engine relies on."""

    name: str
    description: str
    digits: int
    point: float
    contract_size: float
    volume_min: float
    volume_step: float
    currency_profit: str
    trade_mode: int
    visible: bool

    @classmethod
    def from_info(cls, info: Any) -> "BrokerSymbol":
        """Build from an MT5 SymbolInfo named tuple (or any object with the same attrs)."""
        g = lambda k, d=None: getattr(info, k, d)  # noqa: E731
        return cls(name=g("name"), description=g("description", ""), digits=int(g("digits", 0)),
                   point=float(g("point", 0.0)), contract_size=float(g("trade_contract_size", 0.0)),
                   volume_min=float(g("volume_min", 0.0)), volume_step=float(g("volume_step", 0.0)),
                   currency_profit=g("currency_profit", ""), trade_mode=int(g("trade_mode", 0)),
                   visible=bool(g("visible", False)))


def discover_symbols(client: Any, group: str | None = None) -> list[BrokerSymbol]:
    """List every symbol the broker offers (optionally filtered by an MT5 group mask like '*XAU*')."""
    raw = client.call("symbols_get", group=group) if group else client.call("symbols_get")
    symbols = [BrokerSymbol.from_info(s) for s in raw or ()]
    log.info("symbols_discovered", extra={"count": len(symbols), "group": group})
    return symbols


def resolve_symbol(canonical: str, mapping: SymbolMapping, available: list[str]) -> str:
    """Map a canonical instrument (e.g. XAUUSD) to the broker's symbol name.

    Order: explicit ``broker_symbol`` -> first matching ``candidates`` entry
    (case-insensitive) -> ``pattern`` regex, which must match exactly one.

    Raises:
        SymbolResolutionError: nothing matches, or the regex is ambiguous.
    """
    by_lower = {s.lower(): s for s in available}
    if mapping.broker_symbol:
        hit = by_lower.get(mapping.broker_symbol.lower())
        if not hit:
            raise SymbolResolutionError(
                f"{canonical}: configured broker_symbol {mapping.broker_symbol!r} not offered by broker")
        return hit
    for cand in mapping.candidates:
        if cand.lower() in by_lower:
            return by_lower[cand.lower()]
    if mapping.pattern:
        rx = re.compile(mapping.pattern, re.IGNORECASE)
        hits = sorted(s for s in available if rx.match(s))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SymbolResolutionError(
                f"{canonical}: several broker symbols match ({', '.join(hits)}); "
                f"set broker_symbol in config/symbols.yaml")
    raise SymbolResolutionError(f"{canonical}: no matching broker symbol found")


def prepare_symbol(client: Any, canonical: str, mapping: SymbolMapping) -> BrokerSymbol:
    """Resolve the broker symbol, add it to Market Watch and return its info."""
    names = [s.name for s in discover_symbols(client)]
    name = resolve_symbol(canonical, mapping, names)
    if not client.call("symbol_select", name, True):
        raise SymbolResolutionError(f"could not add {name} to Market Watch")
    info = BrokerSymbol.from_info(client.call("symbol_info", name))
    log.info("symbol_resolved", extra={"canonical": canonical, "broker_symbol": name,
                                       "digits": info.digits, "contract_size": info.contract_size})
    return info
