"""Registry of strategy plug-ins by id."""
from __future__ import annotations

from .base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: add a strategy under ``cls.meta.id`` (ids must be unique)."""
    sid = cls.meta.id
    if sid in _REGISTRY and _REGISTRY[sid] is not cls:
        raise ValueError(f"strategy id {sid!r} is already registered")
    _REGISTRY[sid] = cls
    return cls


def get(strategy_id: str) -> type[Strategy]:
    """Strategy class by id."""
    if strategy_id not in _REGISTRY:
        raise KeyError(f"unknown strategy {strategy_id!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[strategy_id]


def create(strategy_id: str, params: dict | None = None) -> Strategy:
    """New strategy instance with parameters."""
    return get(strategy_id)(params)


def available() -> list[str]:
    """Registered ids."""
    return sorted(_REGISTRY)
