import json

import pandas as pd

from quant_engine.data.downloader import HistoricalDownloader, write_report
from quant_engine.data.storage import ParquetStore
from quant_engine.timeframes import Timeframe

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def make(cfg, collector, tmp_path):
    return HistoricalDownloader(collector, ParquetStore(tmp_path), "XAUUSD", cfg.data)


def test_chunked_download_all_timeframes(cfg, collector, tmp_path, fake):
    dl = make(cfg, collector, tmp_path)
    for tf in Timeframe:
        r = dl.download_bars(tf, T("2026-10-20"), T("2026-11-24 10:02"))
        direct = collector.bars(tf, T("2026-10-20"), T("2026-11-24 10:02"))
        assert r.rows_stored == len(direct), tf
        assert r.quality.ok, (tf, r.quality.counts())
    # M1 uses 7-day chunks over ~35 days
    assert make(cfg, collector, tmp_path / "x").download_bars(Timeframe.M1, T("2026-10-20"), T("2026-11-24")).chunks == 5


def test_resume_only_fetches_new_data(cfg, collector, tmp_path, fake):
    dl = make(cfg, collector, tmp_path)
    dl.download_bars(Timeframe.M5, T("2026-11-01"), T("2026-11-20"))
    r = dl.download_bars(Timeframe.M5, T("2026-11-01"), T("2026-11-24"))
    assert pd.Timestamp(r.start) == T("2026-11-19 23:40")  # last stored candle (23:55) minus 3 bars
    assert r.rows_stored == len(collector.bars(Timeframe.M5, T("2026-11-01"), T("2026-11-24")))


def test_no_resume(cfg, collector, tmp_path):
    dl = make(cfg, collector, tmp_path)
    dl.download_bars(Timeframe.H1, T("2026-11-01"), T("2026-11-10"))
    r = dl.download_bars(Timeframe.H1, T("2026-11-01"), T("2026-11-20"), resume=False)
    assert pd.Timestamp(r.start) == T("2026-11-01")


def test_tick_download_and_report(cfg, collector, tmp_path):
    dl = make(cfg, collector, tmp_path)
    r = dl.download_ticks(T("2026-11-23 06:00"), T("2026-11-23 20:00"))
    assert r.chunks == 3 and r.rows_stored == 14 * 60 * 4 and r.quality.ok
    path = write_report([r], tmp_path / "reports" / "r.json")
    data = json.loads(path.read_text())
    assert data[0]["dataset"] == "XAUUSD ticks" and data[0]["quality"]["ok"]
