"""A fake market-data collector replaying synthetic candles up to a moving clock."""
from __future__ import annotations

import pandas as pd

from quant_engine.data.candles import closed_bars
from quant_engine.mt5.symbols import BrokerSymbol
from quant_engine.timeframes import Timeframe

SYMBOL = BrokerSymbol(name="XAUUSD.r", description="Gold", digits=2, point=0.01, contract_size=100.0,
                      volume_min=0.01, volume_step=0.01, currency_profit="USD", trade_mode=4, visible=True)


class ReplayCollector:
    """``bars(tf, start, end)`` like :class:`MarketDataCollector`, from stored frames, as of ``clock.now``."""

    def __init__(self, bars: dict[Timeframe, pd.DataFrame], clock) -> None:
        self.data, self.clock = bars, clock
        self.calls = 0

    def bars(self, tf, start, end, closed_only=True):
        self.calls += 1
        b = self.data[tf]
        b = b[(b["time"] >= start) & (b["time"] < end)]
        return closed_bars(b, tf, self.clock.now).reset_index(drop=True)


class Clock:
    def __init__(self, t: str) -> None:
        self.now = pd.Timestamp(t, tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.now


class Recorder:
    """Notifier that records messages."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, int | None]] = []

    def send(self, text, reply_to=None):
        self.sent.append((text, reply_to))
        return len(self.sent)
