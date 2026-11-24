import pandas as pd

from quant_engine.data.candles import as_of, closed_bars, is_closed, last_closed_open_time
from quant_engine.timeframes import Timeframe

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def bars(times):
    return pd.DataFrame({"time": [T(t) for t in times], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})


def test_is_closed_boundary():
    assert not is_closed(T("2026-11-24 10:00"), Timeframe.M5, T("2026-11-24 10:04:59"))
    assert is_closed(T("2026-11-24 10:00"), Timeframe.M5, T("2026-11-24 10:05:00"))


def test_forming_candle_removed():
    b = bars(["2026-11-24 09:55", "2026-11-24 10:00"])
    out = closed_bars(b, Timeframe.M5, T("2026-11-24 10:02:30"))
    assert list(out["time"]) == [T("2026-11-24 09:55")]


def test_htf_as_of_has_no_lookahead():
    h1 = bars(["2026-11-24 08:00", "2026-11-24 09:00", "2026-11-24 10:00"])
    # Deciding on the M5 candle 10:00-10:05: the 10:00 H1 candle is still forming.
    assert as_of(h1, Timeframe.H1, T("2026-11-24 10:05"))["time"].iat[-1] == T("2026-11-24 09:00")
    assert as_of(h1, Timeframe.H1, T("2026-11-24 11:00"))["time"].iat[-1] == T("2026-11-24 10:00")


def test_last_closed_open_time():
    assert last_closed_open_time(Timeframe.M5, T("2026-11-24 10:07")) == T("2026-11-24 10:00")
    assert last_closed_open_time(Timeframe.M5, T("2026-11-24 10:05")) == T("2026-11-24 10:00")
