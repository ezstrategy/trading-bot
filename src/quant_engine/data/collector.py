"""Pull bars and ticks from MT5 and normalise them to UTC."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from ..mt5.server_time import ServerClock
from ..timeframes import Timeframe
from .candles import closed_bars
from .schema import BAR_COLUMNS, TICK_COLUMNS, empty_bars, empty_ticks

log = get_logger(__name__)


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class MarketDataCollector:
    """Read-only market data access for one broker symbol.

    Args:
        client: a connected :class:`~quant_engine.mt5.MT5Client`.
        symbol: the broker's symbol name (already resolved).
        clock: converts server time to UTC.
        now: injectable clock returning the current UTC time (tests).
    """

    def __init__(self, client: Any, symbol: str, clock: ServerClock,
                 now: Callable[[], pd.Timestamp] = _now) -> None:
        self.client, self.symbol, self.clock, self.now = client, symbol, clock, now

    # ---------------------------------------------------------------- bars
    def _rates_to_frame(self, rates: Any) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            return empty_bars()
        raw = pd.DataFrame(np.asarray(rates))
        df = pd.DataFrame({"time": self.clock.to_utc(raw["time"].to_numpy(), unit="s")})
        for col in BAR_COLUMNS[1:]:
            df[col] = raw[col].to_numpy(dtype="float64") if col in raw else np.nan
        return df.sort_values("time", kind="stable").reset_index(drop=True)

    def bars(self, tf: Timeframe, start_utc: pd.Timestamp, end_utc: pd.Timestamp,
             closed_only: bool = True) -> pd.DataFrame:
        """Candles with UTC open time in ``[start_utc, end_utc)``.

        The request window is padded by one day each side (to absorb any
        offset error) and then cut exactly in UTC. With ``closed_only`` the
        still-forming candle is removed.
        """
        start, end = _utc(start_utc), _utc(end_utc)
        rates = self.client.call(
            "copy_rates_range", self.symbol, tf.mt5_constant(self.client.mt5),
            self.clock.to_server(start - pd.Timedelta(days=1)),
            self.clock.to_server(end + pd.Timedelta(days=1)), allow_none=True)
        df = self._rates_to_frame(rates)
        df = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
        if closed_only:
            df = closed_bars(df, tf, self.now())
        log.debug("bars_fetched", extra={"symbol": self.symbol, "timeframe": tf.value,
                                         "start": str(start), "end": str(end), "rows": len(df)})
        return df

    def latest_bars(self, tf: Timeframe, count: int) -> pd.DataFrame:
        """The last ``count`` CLOSED candles (the forming candle is never included)."""
        rates = self.client.call("copy_rates_from_pos", self.symbol, tf.mt5_constant(self.client.mt5),
                                 0, count + 1, allow_none=True)
        df = closed_bars(self._rates_to_frame(rates), tf, self.now())
        return df.tail(count).reset_index(drop=True)

    # --------------------------------------------------------------- ticks
    def ticks(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
        """All bid/ask ticks with UTC time in ``[start_utc, end_utc)``."""
        start, end = _utc(start_utc), _utc(end_utc)
        mt5 = self.client.mt5
        raw = self.client.call("copy_ticks_range", self.symbol,
                               self.clock.to_server(start - pd.Timedelta(hours=1)),
                               self.clock.to_server(end + pd.Timedelta(hours=1)),
                               int(mt5.COPY_TICKS_ALL), allow_none=True)
        if raw is None or len(raw) == 0:
            return empty_ticks()
        r = pd.DataFrame(np.asarray(raw))
        df = pd.DataFrame({"time": self.clock.to_utc(r["time_msc"].to_numpy(), unit="ms")})
        for col in TICK_COLUMNS[1:]:
            df[col] = r[col].to_numpy(dtype="float64") if col in r else np.nan
        df = df[(df["time"] >= start) & (df["time"] < end)]
        return df.sort_values("time", kind="stable").reset_index(drop=True)

    def last_tick_server_epoch(self) -> float | None:
        """Server-clock epoch of the latest tick (used to calibrate the clock)."""
        tick = self.client.call("symbol_info_tick", self.symbol, allow_none=True)
        return float(tick.time) if tick is not None else None
