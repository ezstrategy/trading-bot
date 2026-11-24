import pandas as pd

from quant_engine.data.storage import ParquetStore
from quant_engine.timeframes import Timeframe

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def test_bar_round_trip_keeps_utc(tmp_path, collector):
    store = ParquetStore(tmp_path)
    df = collector.bars(Timeframe.M5, T("2026-11-23"), T("2026-11-24"))
    store.write_bars("XAUUSD", Timeframe.M5, df)
    back = store.read_bars("xauusd", Timeframe.M5)
    assert len(back) == len(df) and str(back["time"].dt.tz) == "UTC"
    assert (back["time"].values == df["time"].values).all()
    assert store.bars_path("XAUUSD", Timeframe.M5).name == "M5.parquet"


def test_upsert_overwrites_and_dedupes(tmp_path, collector):
    store = ParquetStore(tmp_path)
    a = collector.bars(Timeframe.H1, T("2026-11-23 00:00"), T("2026-11-23 12:00"))
    b = collector.bars(Timeframe.H1, T("2026-11-23 06:00"), T("2026-11-23 18:00"))
    b.loc[0, "close"] = 9999.0
    store.write_bars("XAUUSD", Timeframe.H1, a)
    n = store.write_bars("XAUUSD", Timeframe.H1, b)
    back = store.read_bars("XAUUSD", Timeframe.H1)
    assert n == len(back) == 18 and back["time"].is_unique and back["time"].is_monotonic_increasing
    assert back.loc[back["time"] == T("2026-11-23 06:00"), "close"].iat[0] == 9999.0
    assert len(store.read_bars("XAUUSD", Timeframe.H1, T("2026-11-23 10:00"), T("2026-11-23 12:00"))) == 2
    assert store.last_bar_time("XAUUSD", Timeframe.H1) == T("2026-11-23 17:00")


def test_ticks_split_by_day(tmp_path, collector):
    store = ParquetStore(tmp_path)
    t = collector.ticks(T("2026-11-23 23:50"), T("2026-11-24 00:10"))
    store.write_ticks("XAUUSD", t)
    store.write_ticks("XAUUSD", t)  # idempotent
    assert store.ticks_path("XAUUSD", T("2026-11-23")).exists()
    assert store.ticks_path("XAUUSD", T("2026-11-24")).exists()
    back = store.read_ticks("XAUUSD", T("2026-11-23 23:50"), T("2026-11-24 00:10"))
    assert len(back) == len(t) == 80


def test_missing_files_return_empty(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.read_bars("XAUUSD", Timeframe.M5).empty
    assert store.last_bar_time("XAUUSD", Timeframe.M5) is None
    assert store.read_ticks("XAUUSD", T("2026-11-23"), T("2026-11-24")).empty
