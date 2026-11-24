import numpy as np
import pandas as pd
import pytest

from quant_engine.config import QualityConfig
from quant_engine.data.quality import is_market_closed, validate_bars, validate_ticks
from quant_engine.timeframes import Timeframe

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
Q = QualityConfig()


def make(times, tf="5min"):
    t = pd.DatetimeIndex(times) if not isinstance(times, str) else None
    n = len(t)
    c = 2600 + np.arange(n) * 0.1
    return pd.DataFrame({"time": t, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c + 0.1,
                         "tick_volume": 10.0, "spread": 20.0, "real_volume": 0.0})


def grid(a, b, freq="5min"):
    return pd.date_range(T(a), T(b), freq=freq, inclusive="left")


@pytest.mark.parametrize("tf", list(Timeframe))
def test_clean_broker_data_passes(collector, tf):
    df = collector.bars(tf, T("2026-10-13"), T("2026-11-24"))
    rep = validate_bars(df, tf, Q)
    assert rep.ok, rep.counts()
    assert rep.counts().get("missing", 0) == 0


def test_duplicate():
    t = grid("2026-11-23 08:00", "2026-11-23 09:00")
    rep = validate_bars(make(t.append(t[[3]]).sort_values()), Timeframe.M5, Q)
    assert rep.counts()["duplicate"] == 1 and not rep.ok


def test_misaligned():
    t = list(grid("2026-11-23 08:00", "2026-11-23 09:00"))
    t[5] = t[5] + pd.Timedelta(minutes=2)
    rep = validate_bars(make(t), Timeframe.M5, Q)
    assert rep.counts()["misaligned"] == 1


def test_invalid_ohlc_nan_nonpositive():
    df = make(grid("2026-11-23 08:00", "2026-11-23 09:00"))
    df.loc[2, "high"] = df.loc[2, "open"] - 1      # high below open
    df.loc[4, "low"] = df.loc[4, "close"] + 1      # low above close
    df.loc[6, "close"] = np.nan
    df.loc[8, ["open", "low"]] = [0.0, -1.0]
    c = validate_bars(df, Timeframe.M5, Q).counts()
    assert c["invalid_ohlc"] == 2 and c["nan"] == 1 and c["non_positive"] == 1


def test_missing_candles_flagged():
    t = grid("2026-11-23 08:00", "2026-11-23 10:00").delete(slice(5, 9))
    rep = validate_bars(make(t), Timeframe.M5, Q)
    miss = [i for i in rep.issues if i.kind == "missing"]
    assert len(miss) == 1 and miss[0].detail.startswith("4 M5") and miss[0].time.startswith("2026-11-23 08:25")


def test_weekend_and_daily_break_not_flagged():
    t = grid("2026-11-20 18:00", "2026-11-23 23:00")
    t = t[[not is_market_closed(x, Q) for x in t]]
    rep = validate_bars(make(t), Timeframe.M5, Q)
    assert rep.counts().get("missing", 0) == 0


def test_spike_warning_does_not_fail():
    df = make(grid("2026-11-23 08:00", "2026-11-23 12:00"))
    df.loc[40, "high"] += 50
    rep = validate_bars(df, Timeframe.M5, Q)
    assert rep.counts()["spike"] == 1 and rep.ok


def test_unsorted_and_not_utc():
    t = grid("2026-11-23 08:00", "2026-11-23 09:00")
    rep = validate_bars(make(t[::-1]), Timeframe.M5, Q)
    assert "unsorted" in rep.counts()
    rep = validate_bars(make(t.tz_convert("Asia/Kolkata")), Timeframe.M5, Q)
    assert "not_utc" in rep.counts()


def test_empty():
    rep = validate_bars(make([]), Timeframe.M5, Q)
    assert rep.ok and rep.counts() == {"empty": 1}


@pytest.mark.parametrize("ts,closed", [
    ("2026-11-20 20:30", True),   # Friday after close
    ("2026-11-21 12:00", True),   # Saturday
    ("2026-11-22 22:30", True),   # Sunday before open
    ("2026-11-22 23:05", False),  # Sunday after open
    ("2026-11-24 21:15", True),   # daily break
    ("2026-11-24 10:00", False),
])
def test_market_hours(ts, closed):
    assert is_market_closed(T(ts), Q) is closed


def test_report_json():
    d = validate_bars(make(grid("2026-11-23 08:00", "2026-11-23 09:00")), Timeframe.M5, Q).to_dict()
    assert d["ok"] and d["rows"] == 12 and d["first"].startswith("2026-11-23 08:00")


def test_tick_checks(collector):
    good = collector.ticks(T("2026-11-23 08:00"), T("2026-11-23 09:00"))
    assert validate_ticks(good, Q).ok
    bad = good.copy()
    bad.loc[0, "ask"] = bad.loc[0, "bid"] - 1
    bad.loc[1, "bid"] = 0
    bad.loc[2, "ask"] = bad.loc[2, "bid"] + 10
    bad = pd.concat([bad, bad.iloc[[5]]], ignore_index=True)
    c = validate_ticks(bad, Q).counts()
    assert c["crossed"] == 1 and c["non_positive"] == 1 and c["wide_spread"] == 2
    assert c["duplicate"] == 1 and c["unsorted"] == 1
