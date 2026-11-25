"""A simulated MetaTrader5 module for tests.

It behaves like the real package where the engine depends on it:
* bar/tick ``time`` fields are epoch seconds of the BROKER SERVER clock
  (New York close: GMT+2 winter, GMT+3 summer);
* range requests take datetimes whose epoch is server time;
* the newest candle is still forming at ``now``;
* the market is closed at weekends and during a daily break;
* connections can be dropped and initialisation can fail on demand.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_engine.mt5.server_time import ServerClock

TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15 = 1, 5, 15
TIMEFRAME_H1, TIMEFRAME_H4 = 16385, 16388
COPY_TICKS_ALL = -1
_TF_MIN = {1: 1, 5: 5, 15: 15, 16385: 60, 16388: 240}

NOW_UTC = pd.Timestamp("2026-11-24 10:02:30", tz="UTC")   # a Tuesday, market open
HISTORY_START_UTC = pd.Timestamp("2026-10-12", tz="UTC")  # spans the 1 Nov US DST change
CLOCK = ServerClock(mode="ny_close")

RATE_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
              ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
TICK_DTYPE = [("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"), ("volume", "<u8"),
              ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8")]


def market_open_utc(ts: pd.DatetimeIndex) -> np.ndarray:
    """Fake trading hours (UTC): closed Fri 21:00 - Sun 23:00 and 21:00-22:00 daily."""
    wd, h = ts.weekday, ts.hour
    weekend = ((wd == 4) & (h >= 21)) | (wd == 5) | ((wd == 6) & (h < 23))
    return ~(weekend | (h == 21))


@lru_cache(maxsize=None)
def _m1_utc() -> pd.DataFrame:
    idx = pd.date_range(HISTORY_START_UTC, NOW_UTC.floor("min"), freq="1min")
    idx = idx[market_open_utc(idx)]
    rng = np.random.default_rng(7)
    close = 2600 + np.cumsum(rng.normal(0, 0.4, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, len(idx))
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, len(idx))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(10, 200, len(idx))}, index=idx)


@lru_cache(maxsize=None)
def _rates(tf_minutes: int) -> np.ndarray:
    """Bars as the terminal returns them: grouped on the SERVER clock grid."""
    m1 = _m1_utc()
    # server wall clock = New York wall clock + 7h (NY 17:00 == server 00:00)
    server_wall = m1.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    key = server_wall.floor(f"{tf_minutes}min")
    g = m1.groupby(key)
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "tick_volume": g["tick_volume"].sum()})
    out = np.zeros(len(bars), dtype=RATE_DTYPE)
    out["time"] = (bars.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    for c in ("open", "high", "low", "close", "tick_volume"):
        out[c] = bars[c].to_numpy()
    out["spread"] = 20
    return out


class FakeMT5:
    """Module-like object; attributes mirror the MetaTrader5 package."""

    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15 = TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15
    TIMEFRAME_H1, TIMEFRAME_H4 = TIMEFRAME_H1, TIMEFRAME_H4
    COPY_TICKS_ALL = COPY_TICKS_ALL

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbol_names = symbols or ["EURUSD", "GBPUSD", "XAUUSD.r", "XAGUSD.r", "US30"]
        self.initialized = False
        self.link_up = True
        self.fail_init = 0          # number of initialize() calls that fail
        self.auth_fail = False
        self.drop_next_calls = 0    # data calls that fail with a lost connection
        self.init_kwargs: list[dict] = []
        self.selected: list[str] = []
        self.calls: list[str] = []
        self._err = (1, "Success")
        self.order_send_called = False

    # ----------------------------------------------------------- lifecycle
    def initialize(self, **kwargs):
        self.init_kwargs.append(kwargs)
        if self.auth_fail:
            self._err = (-6, "Terminal: Authorization failed")
            return False
        if self.fail_init > 0:
            self.fail_init -= 1
            self._err = (-10003, "IPC initialize failed")
            return False
        self.initialized, self.link_up = True, True
        self._err = (1, "Success")
        return True

    def shutdown(self):
        self.initialized = False
        return True

    def last_error(self):
        return self._err

    def terminal_info(self):
        if not self.initialized:
            return None
        return SimpleNamespace(connected=self.link_up, name="FakeTerminal")

    def account_info(self):
        return SimpleNamespace(login=123456, server="Fake-Demo", trade_mode=0, balance=10000.0, currency="USD") if self.initialized else None

    # ------------------------------------------------------------- symbols
    def symbols_get(self, group: str | None = None):
        names = self.symbol_names
        if group:
            needle = group.strip("*").upper()
            names = [n for n in names if needle in n.upper()]
        return tuple(SimpleNamespace(name=n, description=f"{n} desc", digits=2 if "XAU" in n else 5,
                                     point=0.01 if "XAU" in n else 1e-5, trade_contract_size=100.0,
                                     volume_min=0.01, volume_step=0.01, currency_profit="USD",
                                     trade_mode=4, visible=False) for n in names)

    def symbol_select(self, name, enable=True):
        if name not in self.symbol_names:
            return False
        self.selected.append(name)
        return True

    def symbol_info(self, name):
        hits = [s for s in self.symbols_get() if s.name == name]
        return hits[0] if hits else None

    def symbol_info_tick(self, name):
        server_now = int((NOW_UTC + CLOCK.offset_at(NOW_UTC)).tz_localize(None).value // 1_000_000_000)
        return SimpleNamespace(time=server_now - 1, bid=2650.0, ask=2650.2)

    # ---------------------------------------------------------------- data
    def _gate(self, fn: str) -> bool:
        self.calls.append(fn)
        if not self.initialized:
            self._err = (-10004, "No IPC connection")
            return False
        if self.drop_next_calls > 0:
            self.drop_next_calls -= 1
            self.link_up = False
            self._err = (-10004, "No IPC connection")
            return False
        self._err = (1, "Success")
        return True

    def copy_rates_range(self, symbol, timeframe, date_from: datetime, date_to: datetime):
        if not self._gate("copy_rates_range"):
            return None
        if symbol not in self.symbol_names:
            self._err = (-4, "Symbol not found")
            return None
        r = _rates(_TF_MIN[timeframe])
        a, b = int(date_from.timestamp()), int(date_to.timestamp())
        return r[(r["time"] >= a) & (r["time"] <= b)]

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if not self._gate("copy_rates_from_pos"):
            return None
        r = _rates(_TF_MIN[timeframe])
        end = len(r) - start_pos
        return r[max(0, end - count):end]

    def copy_ticks_range(self, symbol, date_from: datetime, date_to: datetime, flags):
        if not self._gate("copy_ticks_range"):
            return None
        a, b = int(date_from.timestamp()), int(date_to.timestamp())
        r = _rates(1)
        r = r[(r["time"] >= a) & (r["time"] <= b)]
        n = len(r) * 4
        out = np.zeros(n, dtype=TICK_DTYPE)
        base = np.repeat(r["time"] * 1000, 4) + np.tile([0, 15000, 30000, 45000], len(r))
        out["time_msc"] = base
        out["time"] = base // 1000
        px = np.repeat(r["close"], 4)
        out["bid"], out["ask"], out["flags"] = px, px + 0.2, 6
        return out

    # ------------------------------------------- must never be called
    def order_send(self, request):  # pragma: no cover - the guard blocks this
        self.order_send_called = True
        return SimpleNamespace(retcode=10009)

    def order_check(self, request):  # pragma: no cover
        self.order_send_called = True
        return None
