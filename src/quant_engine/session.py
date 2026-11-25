"""Wire config, MT5 client, server clock, symbol and collector together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .config import AppConfig
from .data.collector import MarketDataCollector
from .logging_setup import get_logger
from .mt5.client import MT5Client
from .mt5.server_time import ServerClock
from .mt5.symbols import BrokerSymbol, prepare_symbol

log = get_logger(__name__)


@dataclass
class EngineSession:
    """Everything needed to read market data for the configured instrument."""

    cfg: AppConfig
    client: MT5Client
    clock: ServerClock
    symbol: BrokerSymbol
    collector: MarketDataCollector

    def close(self) -> None:
        self.client.disconnect()

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_session(cfg: AppConfig, mt5_module: Any | None = None,
                 now: Callable[[], pd.Timestamp] | None = None, **client_kwargs: Any) -> EngineSession:
    """Connect to MT5, resolve the instrument, calibrate the server clock.

    Args:
        cfg: loaded configuration.
        mt5_module: injectable MetaTrader5 module (tests).
        now: injectable UTC clock (tests); defaults to the system clock.
    """
    now = now or (lambda: pd.Timestamp.now(tz="UTC"))
    client = MT5Client(cfg.mt5.connection, cfg.credentials, mt5_module=mt5_module, **client_kwargs)
    client.connect()
    try:
        canonical = cfg.engine.instrument
        sym = prepare_symbol(client, canonical, cfg.mapping(canonical))
        clock = ServerClock.from_config(cfg.mt5.server_time)
        collector = MarketDataCollector(client, sym.name, clock, now=now)
        if cfg.mt5.server_time.mode == "auto":
            epoch = collector.last_tick_server_epoch()
            if epoch is not None:
                clock.calibrate(epoch, cfg.mt5.server_time, now_utc=now().timestamp())
        log.info("session_ready", extra={"symbol": sym.name, "clock_mode": clock.mode})
        return EngineSession(cfg, client, clock, sym, collector)
    except Exception:
        client.disconnect()
        raise
