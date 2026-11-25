import numpy as np
import pandas as pd
import pytest

from quant_engine.timeframes import Timeframe

from .fake_mt5 import NOW_UTC, market_open_utc

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def test_bars_are_utc_and_on_grid(collector):
    df = collector.bars(Timeframe.M5, T("2026-11-23"), T("2026-11-24"))
    assert str(df["time"].dt.tz) == "UTC"
    assert (df["time"].dt.minute % 5 == 0).all() and (df["time"].dt.second == 0).all()
    assert df["time"].is_monotonic_increasing and not df["time"].duplicated().any()
    assert list(df.columns)[:5] == ["time", "open", "high", "low", "close"]


def test_range_is_exact_in_utc(collector):
    df = collector.bars(Timeframe.M15, T("2026-11-23 08:00"), T("2026-11-23 12:00"))
    assert df["time"].iat[0] == T("2026-11-23 08:00") and df["time"].iat[-1] == T("2026-11-23 11:45")
    assert len(df) == 16


def test_bars_match_true_utc_minutes(collector):
    """The 08:00 UTC M5 candle must be built from the 08:00-08:04 UTC minutes."""
    m1 = collector.bars(Timeframe.M1, T("2026-11-23 08:00"), T("2026-11-23 08:05"))
    m5 = collector.bars(Timeframe.M5, T("2026-11-23 08:00"), T("2026-11-23 08:05"))
    assert len(m1) == 5 and len(m5) == 1
    assert m5["open"].iat[0] == m1["open"].iat[0] and m5["close"].iat[0] == m1["close"].iat[-1]
    assert m5["high"].iat[0] == m1["high"].max() and m5["low"].iat[0] == m1["low"].min()


def test_forming_candle_never_returned(collector):
    df = collector.bars(Timeframe.M5, T("2026-11-24 09:00"), T("2026-11-25"))
    assert df["time"].iat[-1] == T("2026-11-24 09:55")  # 10:00 candle forms until 10:05
    assert (df["time"] + Timeframe.M5.delta <= NOW_UTC).all()
    raw = collector.bars(Timeframe.M5, T("2026-11-24 09:00"), T("2026-11-25"), closed_only=False)
    assert raw["time"].iat[-1] == T("2026-11-24 10:00")


@pytest.mark.parametrize("tf,last", [(Timeframe.M5, "09:55"), (Timeframe.M15, "09:45"),
                                     (Timeframe.H1, "09:00"), (Timeframe.H4, "06:00")])
def test_latest_bars_only_closed(collector, tf, last):
    df = collector.latest_bars(tf, 5)
    assert len(df) == 5 and df["time"].iat[-1] == T(f"2026-11-24 {last}")


def test_h4_follows_server_midnight(collector):
    """Winter: server midnight is 22:00 UTC, so H4 candles open at 22, 02, 06 ... UTC."""
    df = collector.bars(Timeframe.H4, T("2026-11-17"), T("2026-11-21"))
    assert set(df["time"].dt.hour) <= {22, 2, 6, 10, 14, 18}


def test_no_gap_or_overlap_across_dst_change(collector):
    """US DST ended 1 Nov 2026; UTC hourly bars must stay continuous around it."""
    df = collector.bars(Timeframe.H1, T("2026-10-29"), T("2026-11-04"))
    t = pd.DatetimeIndex(df["time"])
    assert not t.duplicated().any()
    expected = pd.date_range(T("2026-10-29"), T("2026-11-04"), freq="1h", inclusive="left")
    expected = expected[market_open_utc(expected)]
    assert t.equals(pd.DatetimeIndex(expected))


def test_ticks_bid_ask_utc(collector):
    t = collector.ticks(T("2026-11-23 08:00"), T("2026-11-23 08:10"))
    assert len(t) == 40 and str(t["time"].dt.tz) == "UTC"
    assert t["time"].iat[0] == T("2026-11-23 08:00") and t["time"].iat[-1] < T("2026-11-23 08:10")
    assert (t["ask"] >= t["bid"]).all() and t["time"].is_monotonic_increasing


def test_empty_range(collector):
    df = collector.bars(Timeframe.M5, T("2026-11-21 06:00"), T("2026-11-21 12:00"))  # Saturday
    assert df.empty and list(df.columns)[0] == "time"
    assert collector.ticks(T("2026-11-21 06:00"), T("2026-11-21 07:00")).empty


def test_last_tick_epoch(collector):
    e = collector.last_tick_server_epoch()
    assert collector.clock.to_utc([e])[0] == T("2026-11-24 10:02:29")
