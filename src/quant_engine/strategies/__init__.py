"""Strategy plug-ins (Quant Lab module 4). Import this package to register the built-in ones."""
from .base import OrderIntent, Strategy, StrategyMeta
from .registry import available, create, get, register

from . import gtc_sbr  # noqa: E402,F401  (registers gtc_sbr_bull / gtc_sbr_bear)

__all__ = ["OrderIntent", "Strategy", "StrategyMeta", "available", "create", "get", "register"]
