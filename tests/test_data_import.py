"""Quant Lab Phase 2: CSV import, UTC normalisation and the versioned dataset catalog.

Fixtures in tests/fixtures/data are small hand-written files whose correct answers are known:

* xauusd_m5_mt5_export.csv: MT5 "Bars" export (tab separated, CRLF, server clock GMT+2 winter /
  GMT+3 summer). Server 02:00 on 2024-01-08 and server 03:00 on 2024-07-08 are both 00:00 UTC.
* xauusd_m5_generic_issues.csv: 11 data rows with one out-of-order row, one duplicate time,
  one impossible candle (high below open), one bad price (line 9), one bad date (line 11)
  and missing candles.
* xauusd_m1_utc.csv: semicolon separated, ISO times with Z, spread column.
* xauusd_m15_noheader.csv: MT5 no-header layout, times at +02:00.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.config import QualityConfig
from quant_engine.data.catalog import DataQualityError, DatasetCatalog, DatasetNotFound, sha256_file
from quant_engine.data.csv_import import check_timeframe, parse_csv
from quant_engine.data.quality import validate_bars
from quant_engine.mt5.server_time import ServerClock
from quant_engine.timeframes import Timeframe

FIX = Path(__file__).resolve().parent / "fixtures" / "data"
MT5 = FIX / "xauusd_m5_mt5_export.csv"
ISSUES = FIX / "xauusd_m5_generic_issues.csv"
M1 = FIX / "xauusd_m1_utc.csv"
M15 = FIX / "xauusd_m15_noheader.csv"
Q = QualityConfig(price_bounds=(100, 20000))
T0 = pd.Timestamp("2026-11-26 12:00", tz="UTC")


@pytest.fixture
def cat(tmp_path) -> DatasetCatalog:
    return DatasetCatalog(tmp_path / "catalog", Q)


# ----------------------------------------------------------------- parsing / UTC
def test_mt5_export_server_time_to_utc_winter_and_summer():
    r = parse_csv(MT5, source_tz="mt5_server", clock=ServerClock(mode="ny_close"))
    assert r.layout == "mt5_export" and r.rows_in_file == 24 and not r.unparseable
    t = pd.DatetimeIndex(r.bars["time"])
    assert str(t.tz) == "UTC"
    assert t[0] == pd.Timestamp("2024-01-08 00:00", tz="UTC")      # server 02:00, GMT+2
    assert t[12] == pd.Timestamp("2024-07-08 00:00", tz="UTC")     # server 03:00, GMT+3
    assert r.columns["tick_volume"] == "<TICKVOL>" and r.columns["real_volume"] == "<VOL>"
    assert r.bars["spread"].tolist()[:3] == [8.0, 9.0, 10.0]       # points, exactly as in the file
    assert r.bars["open"].iat[0] == 2030.00 and r.bars["tick_volume"].iat[0] == 100


def test_mt5_server_needs_a_clock():
    with pytest.raises(ValueError, match="ServerClock"):
        parse_csv(MT5, source_tz="mt5_server")


def test_fixed_offset_no_header_m15():
    r = parse_csv(M15, source_tz="+02:00")
    assert r.layout == "mt5_noheader" and r.inferred_minutes == 15
    assert pd.DatetimeIndex(r.bars["time"])[0] == pd.Timestamp("2024-01-08 00:00", tz="UTC")
    assert r.bars["spread"].isna().all()                            # no spread column -> nothing invented


def test_iso_z_times_and_semicolons_m1():
    r = parse_csv(M1)
    assert r.inferred_minutes == 1 and r.spread_in_file and len(r.bars) == 6
    assert pd.DatetimeIndex(r.bars["time"])[-1] == pd.Timestamp("2024-01-08 00:05", tz="UTC")


def test_iana_zone_with_dst_and_epoch_seconds():
    text = "time,open,high,low,close\n2024-07-01 09:00,1,2,0.5,1.5\n2024-01-02 09:00,1,2,0.5,1.5\n"
    t = pd.DatetimeIndex(parse_csv(text=text, source_tz="Europe/London").bars["time"])
    assert t[0] == pd.Timestamp("2024-07-01 08:00", tz="UTC") and t[1] == pd.Timestamp("2024-01-02 09:00", tz="UTC")
    epoch = int(pd.Timestamp("2024-01-08 00:05", tz="UTC").timestamp())
    r = parse_csv(text=f"timestamp,open,high,low,close\n{epoch},1,2,0.5,1.5\n")
    assert r.bars["time"].iat[0] == pd.Timestamp("2024-01-08 00:05", tz="UTC")


def test_ambiguous_local_time_is_reported_not_guessed():
    # 01:30 on 2024-10-27 happens twice in London (clocks go back): the row is reported.
    text = "time,open,high,low,close\n2024-10-27 01:30,1,2,0.5,1.5\n2024-10-27 03:00,1,2,0.5,1.5\n"
    r = parse_csv(text=text, source_tz="Europe/London")
    assert len(r.bars) == 1 and r.unparseable == [(2, "unparseable or non-existent time")]


def test_unparseable_rows_are_listed_with_line_numbers():
    r = parse_csv(ISSUES)
    assert r.rows_in_file == 11 and len(r.bars) == 9
    assert r.unparseable == [(9, "unparseable price"), (11, "unparseable or non-existent time")]


def test_declared_timeframe_must_match_the_file():
    with pytest.raises(ValueError, match="median candle spacing is 1 min"):
        check_timeframe(parse_csv(M1), Timeframe.M5)
    check_timeframe(parse_csv(M1), Timeframe.M1)


# ----------------------------------------------------------------- catalog
def test_import_preserves_original_and_records_source(cat):
    res = cat.import_csv(MT5, "xauusd", "M5", source_tz="mt5_server", clock=ServerClock(), source_name="broker X demo",
                         now=T0)
    assert res.created and res.quality_ok
    f = cat.folder(res.version_id)
    assert (f / "original.csv").read_bytes() == MT5.read_bytes()            # byte-identical copy
    m = res.manifest
    assert m["symbol"] == "XAUUSD" and m["timeframe"] == "M5" and m["rows"] == 24
    assert m["source"]["kind"] == "csv" and m["source"]["name"] == "broker X demo"
    assert m["source"]["sha256"] == sha256_file(MT5)
    assert m["settings"]["source_tz"] == "mt5_server" and m["settings"]["server_clock"]["mode"] == "ny_close"
    assert m["spread_in_source"] and m["time_convention"] == "UTC, candle open time"
    assert res.version_id.startswith("XAUUSD_M5_20261126T120000Z_")
    for p in f.iterdir():                                                    # frozen
        assert not os.stat(p).st_mode & stat.S_IWUSR
    assert cat.verify(res.version_id) == {"original_ok": True, "bars_ok": True, "ok": True}


def test_reimport_same_settings_reuses_version_other_settings_make_new(cat):
    a = cat.import_csv(M15, "XAUUSD", "M15", source_tz="+02:00", now=T0)
    b = cat.import_csv(M15, "XAUUSD", "M15", source_tz="+02:00", now=T0 + pd.Timedelta(hours=1))
    assert not b.created and b.version_id == a.version_id
    c = cat.import_csv(M15, "XAUUSD", "M15", source_tz="UTC", now=T0 + pd.Timedelta(hours=2))
    assert c.created and c.version_id != a.version_id
    assert len(cat.list()) == 2
    # the first version is untouched by the second import
    assert cat.load(a.version_id)["time"].iat[0] == pd.Timestamp("2024-01-08 00:00", tz="UTC")
    assert cat.load(c.version_id)["time"].iat[0] == pd.Timestamp("2024-01-08 02:00", tz="UTC")
    assert cat.latest("XAUUSD", "M15") == c.version_id


def test_problems_are_reported_and_block_research_use(cat):
    res = cat.import_csv(ISSUES, "XAUUSD", "M5", now=T0)
    q = res.manifest["quality"]
    assert not res.quality_ok
    assert q["blocking"] == {"duplicate": 1, "invalid_ohlc": 1, "unparseable_row": 2}
    assert q["counts"]["reordered"] == 1 and q["counts"]["no_spread"] == 1 and q["counts"]["missing"] >= 1
    with pytest.raises(DataQualityError):
        cat.load(res.version_id)
    bars = cat.load(res.version_id, allow_issues=True)
    # nothing dropped or patched: both 00:20 rows kept (file order), the bad candle kept as is
    assert (bars["time"] == pd.Timestamp("2024-01-08 00:20", tz="UTC")).sum() == 2
    assert bars.loc[bars["time"] == pd.Timestamp("2024-01-08 00:20", tz="UTC"), "close"].tolist() == [2030.50, 2030.55]
    bad = bars[bars["time"] == pd.Timestamp("2024-01-08 00:25", tz="UTC")].iloc[0]
    assert (bad["high"], bad["low"]) == (2030.30, 2030.40)
    assert bars["time"].is_monotonic_increasing                      # only the order changed
    assert bars["spread"].isna().all()
    assert not pd.Timestamp("2024-01-08 00:40", tz="UTC") in set(bars["time"])   # missing stays missing
    assert cat.latest("XAUUSD", "M5") is None and cat.latest("XAUUSD", "M5", usable_only=False) == res.version_id
    full = cat.quality_report(res.version_id)
    assert any(i["detail"].startswith("line 9") for i in full["issues"])


def test_verify_detects_a_changed_file(cat):
    res = cat.import_csv(M1, "XAUUSD", "M1", now=T0)
    f = cat.folder(res.version_id) / "bars.parquet"
    os.chmod(f, stat.S_IWUSR | stat.S_IRUSR)
    df = pd.read_parquet(f)
    df.loc[0, "close"] += 1.0
    df.to_parquet(f, index=False)
    assert cat.verify(res.version_id) == {"original_ok": True, "bars_ok": False, "ok": False}


def test_unknown_version(cat):
    with pytest.raises(DatasetNotFound):
        cat.manifest("nope")


def test_import_frame_snapshot_of_mt5_store(cat):
    bars = parse_csv(MT5, source_tz="mt5_server", clock=ServerClock()).bars
    a = cat.import_frame(bars, "XAUUSD", "M5", source_kind="mt5_store", source_name="store", now=T0)
    assert a.created and a.manifest["source"]["stored_as"] == "original.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(cat.folder(a.version_id) / "original.parquet"), bars)
    b = cat.import_frame(bars, "XAUUSD", "M5", source_kind="mt5_store", source_name="store", now=T0 + pd.Timedelta(1, "h"))
    assert not b.created
    more = pd.concat([bars, bars.tail(1).assign(time=bars["time"].iat[-1] + pd.Timedelta(minutes=5))], ignore_index=True)
    c = cat.import_frame(more, "XAUUSD", "M5", source_kind="mt5_store", source_name="store", now=T0 + pd.Timedelta(2, "h"))
    assert c.created and c.manifest["rows"] == 25 and cat.manifest(a.version_id)["rows"] == 24
    naive = bars.assign(time=bars["time"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="UTC"):
        cat.import_frame(naive, "XAUUSD", "M5", source_kind="x", source_name="x")


# ----------------------------------------------------------------- new quality checks
def _bars(n=60):
    t = pd.date_range("2024-01-08", periods=n, freq="5min", tz="UTC")
    o = 2000 + np.arange(n) * 0.1
    return pd.DataFrame({"time": t, "open": o, "high": o + 0.5, "low": o - 0.5, "close": o + 0.1,
                         "tick_volume": 1.0, "spread": 8.0, "real_volume": 0.0})


def test_gap_warning_and_price_bounds():
    df = _bars()
    assert validate_bars(df, Timeframe.M5, Q).counts() == {}
    df.loc[40, ["open", "high", "low", "close"]] = [2100.0, 2100.5, 2099.5, 2100.1]
    c = validate_bars(df, Timeframe.M5, Q).counts()
    assert c["gap"] == 2 and "out_of_bounds" not in c                    # into and out of the jump
    df.loc[10, ["open", "high", "low", "close"]] = [20.0, 20.5, 19.5, 20.1]   # wrong scaling
    rep = validate_bars(df, Timeframe.M5, Q)
    assert rep.counts()["out_of_bounds"] == 1 and not rep.ok
    assert validate_bars(df, Timeframe.M5, QualityConfig(max_gap_atr=0)).counts().get("gap") is None


# ----------------------------------------------------------------- CLI
@pytest.fixture
def run(config_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return lambda *a: cli.run(["--config", str(config_dir), *a])


def test_cli_import_list_show_verify(run, capsys):
    assert run("data-import", str(MT5), "--timeframe", "M5", "--tz", "mt5_server", "--source-name", "fixture") == 0
    out = capsys.readouterr().out
    vid = json.loads(out[out.index("{\n"):out.index("\n}") + 2])["version_id"]
    assert run("data-list") == 0 and vid in capsys.readouterr().out
    assert run("data-show", vid) == 0 and '"sha256"' in capsys.readouterr().out
    assert run("data-verify", vid) == 0 and '"ok": true' in capsys.readouterr().out
    assert run("data-import", str(ISSUES), "--timeframe", "M5") == 1
    assert "BLOCKING" in capsys.readouterr().out
    assert run("data-import", str(M1), "--timeframe", "M5") == 1
    assert "median candle spacing" in capsys.readouterr().err
    assert run("data-show", "nope") == 1


def test_cli_snapshot_of_downloaded_bars(run, cfg, capsys):
    from quant_engine.data.storage import ParquetStore
    bars = parse_csv(MT5, source_tz="mt5_server", clock=ServerClock()).bars
    ParquetStore(cfg.data.storage.root).write_bars("XAUUSD", Timeframe.M5, bars)
    assert run("data-snapshot", "--timeframes", "M1,M5") == 0
    out = capsys.readouterr().out
    assert "M1: nothing downloaded" in out and '"new_version": true' in out
    assert run("data-snapshot", "--timeframes", "M5") == 0 and '"new_version": false' in capsys.readouterr().out
