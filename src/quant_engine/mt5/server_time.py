"""Convert MT5 broker-server timestamps to true UTC.

MT5 reports bar and tick times as epoch seconds of the broker's *server
clock* (for most brokers GMT+2 in winter and GMT+3 in summer). Treating
those numbers as UTC shifts every candle by 2-3 hours, which breaks
session logic (London/New York) and daily levels. :class:`ServerClock`
does the conversion in both directions.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import ServerTimeConfig
from ..logging_setup import get_logger

log = get_logger(__name__)
NY = "America/New_York"


@dataclass
class ServerClock:
    """Broker server clock model.

    Attributes:
        mode: "ny_close" (UTC+7h vs New York, DST aware), "fixed" or "timezone".
        fixed_offset_hours: used when mode == "fixed".
        timezone: IANA zone used when mode == "timezone".
    """

    mode: str = "ny_close"
    fixed_offset_hours: float = 2.0
    timezone: str = "Etc/UTC"
    detected_offset_hours: float | None = field(default=None, compare=False)

    # ------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg: ServerTimeConfig) -> "ServerClock":
        """Build a clock from config. ``auto`` starts on the fallback mode
        until :meth:`calibrate` is called."""
        mode = cfg.fallback_mode if cfg.mode == "auto" else cfg.mode
        return cls(mode=mode, fixed_offset_hours=cfg.fixed_offset_hours, timezone=cfg.timezone)

    # --------------------------------------------------------- conversion
    def _zone(self) -> str | None:
        return NY if self.mode == "ny_close" else self.timezone if self.mode == "timezone" else None

    def to_utc(self, server_epoch: Iterable[float] | np.ndarray | pd.Series,
               unit: str = "s") -> pd.DatetimeIndex:
        """Convert server-clock epoch values (seconds or ms) to a UTC DatetimeIndex."""
        wall = pd.to_datetime(np.asarray(server_epoch), unit=unit)  # naive server wall time
        if self.mode == "fixed":
            return pd.DatetimeIndex(wall - pd.Timedelta(hours=self.fixed_offset_hours)).tz_localize("UTC")
        if self.mode == "ny_close":
            wall = wall - pd.Timedelta(hours=7)  # server 00:00 == New York 17:00
        local = pd.DatetimeIndex(wall).tz_localize(
            self._zone(), ambiguous=np.zeros(len(wall), dtype=bool), nonexistent="shift_forward")
        return local.tz_convert("UTC")

    def offset_at(self, utc: pd.Timestamp) -> pd.Timedelta:
        """Server-minus-UTC offset in effect at a UTC instant."""
        utc = pd.Timestamp(utc).tz_convert("UTC") if pd.Timestamp(utc).tzinfo else pd.Timestamp(utc, tz="UTC")
        if self.mode == "fixed":
            return pd.Timedelta(hours=self.fixed_offset_hours)
        off = utc.tz_convert(self._zone()).utcoffset()
        return pd.Timedelta(off) + (pd.Timedelta(hours=7) if self.mode == "ny_close" else pd.Timedelta(0))

    def to_server(self, utc: pd.Timestamp | datetime) -> datetime:
        """Convert a UTC instant to the datetime MT5 expects in range requests.

        The MetaTrader5 package turns a datetime into epoch seconds, and
        compares that with its server-clock bar times. So we return the
        server wall time labelled as UTC.
        """
        ts = pd.Timestamp(utc)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        wall = (ts + self.offset_at(ts)).tz_localize(None)
        return wall.to_pydatetime().replace(tzinfo=timezone.utc)

    # ---------------------------------------------------------- calibrate
    def calibrate(self, tick_server_epoch: float, cfg: ServerTimeConfig,
                  now_utc: float | None = None) -> bool:
        """Detect the broker offset from a fresh tick (``mode: auto``).

        The raw difference between the tick's server time and real UTC is
        rounded to ``auto_round_minutes``. If the leftover (the tick's age)
        is larger than ``auto_max_tick_age_seconds`` the market is probably
        closed and the clock keeps its current model.

        When the detected offset equals what ny_close predicts right now,
        ny_close is kept because it also handles past DST changes. Otherwise
        the clock switches to a fixed offset and logs a warning.

        Returns:
            True if the offset was measured, False if the tick was too old.
        """
        now = _time.time() if now_utc is None else now_utc
        diff = tick_server_epoch - now
        step = cfg.auto_round_minutes * 60
        offset = round(diff / step) * step
        age = offset - diff
        if abs(age) > cfg.auto_max_tick_age_seconds:
            log.warning("server_offset_not_measured",
                        extra={"reason": "stale tick (market closed?)", "tick_age_s": round(age),
                               "using_mode": self.mode})
            return False
        hours = offset / 3600
        self.detected_offset_hours = hours
        now_ts = pd.Timestamp(now, unit="s", tz="UTC")
        ny = ServerClock(mode="ny_close").offset_at(now_ts)
        if pd.Timedelta(hours=hours) == ny:
            self.mode = "ny_close"
        elif self.mode != "fixed" or self.fixed_offset_hours != hours:
            log.warning("server_offset_not_ny_close",
                        extra={"detected_hours": hours, "ny_close_hours": ny.total_seconds() / 3600,
                               "note": "using a fixed offset; history across DST changes may be "
                                       "shifted by 1h - set server_time.mode explicitly"})
            self.mode, self.fixed_offset_hours = "fixed", hours
        log.info("server_offset_detected", extra={"offset_hours": hours, "mode": self.mode})
        return True
