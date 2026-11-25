"""Historical download utility: chunked, resumable, validated."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..logging_setup import get_logger
from ..timeframes import Timeframe
from .collector import MarketDataCollector
from .quality import QualityReport, validate_bars, validate_ticks
from .storage import ParquetStore

log = get_logger(__name__)


@dataclass
class DownloadResult:
    """Summary of one download job."""

    dataset: str
    start: str
    end: str
    chunks: int
    rows_fetched: int
    rows_stored: int
    quality: QualityReport


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class HistoricalDownloader:
    """Downloads bars and ticks in chunks, stores them and validates the result.

    Args:
        collector: data source for one broker symbol.
        store: parquet storage.
        canonical: canonical instrument name used for file paths (e.g. XAUUSD).
        cfg: data settings (chunk sizes and quality thresholds).
    """

    def __init__(self, collector: MarketDataCollector, store: ParquetStore, canonical: str,
                 cfg: DataConfig) -> None:
        self.collector, self.store, self.canonical, self.cfg = collector, store, canonical, cfg

    def download_bars(self, tf: Timeframe, start: pd.Timestamp, end: pd.Timestamp,
                      resume: bool = True) -> DownloadResult:
        """Download candles with UTC open time in ``[start, end)``.

        With ``resume`` the job restarts a few candles before the last stored
        candle instead of re-downloading everything (stored rows still win
        only if not re-fetched; fresh rows overwrite them).
        """
        start, end = _utc(start), _utc(end)
        if resume:
            last = self.store.last_bar_time(self.canonical, tf)
            if last is not None and start < last < end:
                start = last - 3 * tf.delta
                log.info("download_resume", extra={"timeframe": tf.value, "from": str(start)})
        chunk = pd.Timedelta(days=self.cfg.download.chunk_days.get(tf, 30))
        cur, chunks, fetched = start, 0, 0
        while cur < end:
            nxt = min(cur + chunk, end)
            df = self.collector.bars(tf, cur, nxt, closed_only=True)
            stored = self.store.write_bars(self.canonical, tf, df)
            chunks += 1
            fetched += len(df)
            log.info("bars_chunk", extra={"timeframe": tf.value, "from": str(cur), "to": str(nxt),
                                          "rows": len(df), "stored_total": stored})
            cur = nxt
        data = self.store.read_bars(self.canonical, tf)
        rep = validate_bars(data, tf, self.cfg.quality, dataset=f"{self.canonical} {tf.value}")
        log.info("bars_download_done", extra={"timeframe": tf.value, "rows_stored": len(data),
                                              "quality_ok": rep.ok, "issues": rep.counts()})
        return DownloadResult(f"{self.canonical} {tf.value}", str(start), str(end), chunks, fetched,
                              len(data), rep)

    def download_ticks(self, start: pd.Timestamp, end: pd.Timestamp) -> DownloadResult:
        """Download bid/ask ticks in ``[start, end)`` into daily parquet files."""
        start, end = _utc(start), _utc(end)
        step = pd.Timedelta(hours=self.cfg.download.tick_chunk_hours)
        cur, chunks, fetched = start, 0, 0
        while cur < end:
            nxt = min(cur + step, end)
            df = self.collector.ticks(cur, nxt)
            self.store.write_ticks(self.canonical, df)
            chunks += 1
            fetched += len(df)
            log.info("ticks_chunk", extra={"from": str(cur), "to": str(nxt), "rows": len(df)})
            cur = nxt
        data = self.store.read_ticks(self.canonical, start, end)
        rep = validate_ticks(data, self.cfg.quality, dataset=f"{self.canonical} ticks")
        return DownloadResult(f"{self.canonical} ticks", str(start), str(end), chunks, fetched,
                              len(data), rep)


def write_report(results: list[DownloadResult], path: str | Path) -> Path:
    """Save a JSON quality/download report and return its path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"dataset": r.dataset, "start": r.start, "end": r.end, "chunks": r.chunks,
                "rows_fetched": r.rows_fetched, "rows_stored": r.rows_stored,
                "quality": r.quality.to_dict()} for r in results]
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p
