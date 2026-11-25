"""Timeframe definitions shared by the whole engine."""
from __future__ import annotations

from enum import Enum

import pandas as pd


class Timeframe(str, Enum):
    """Supported chart timeframes."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"

    @property
    def minutes(self) -> int:
        """Length of one candle in minutes."""
        return {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}[self.value]

    @property
    def delta(self) -> pd.Timedelta:
        """Length of one candle as a Timedelta."""
        return pd.Timedelta(minutes=self.minutes)

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias, e.g. '5min' or '4h'."""
        return f"{self.minutes}min" if self.minutes < 60 else f"{self.minutes // 60}h"

    def mt5_constant(self, mt5: object) -> int:
        """Return the MetaTrader5 TIMEFRAME_* constant for this timeframe."""
        name = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
                "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4"}[self.value]
        return int(getattr(mt5, name))

    @classmethod
    def parse(cls, value: "str | Timeframe") -> "Timeframe":
        """Parse 'm5', 'M5' or a Timeframe into a Timeframe."""
        if isinstance(value, Timeframe):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported timeframe {value!r}; use one of "
                             f"{', '.join(t.value for t in cls)}") from exc
