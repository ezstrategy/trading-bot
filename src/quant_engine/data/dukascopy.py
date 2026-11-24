"""Import Dukascopy historical candles (free, true spot XAUUSD, timestamps in UTC).

Dukascopy serves one LZMA-compressed ``.bi5`` file per day of M1 candles:

    https://datafeed.dukascopy.com/datafeed/XAUUSD/YYYY/MM/DD/BID_candles_min_1.bi5

where **MM is zero-based** (January = 00). Each record is 24 bytes, big-endian:
``int32 seconds-from-day-start, int32 open, int32 close, int32 low, int32 high,
float32 volume``; prices are integers scaled by ``price_scale`` (1000 for XAUUSD).

These are BID candles. The engine's bar schema has a ``spread`` column; it is
filled from ``assumed_spread_points`` because the candle files carry no spread.
"""
from __future__ import annotations

import lzma
import struct
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..timeframes import Timeframe
from .schema import BAR_COLUMNS, empty_bars

RECORD = struct.Struct(">5if")
NY = "America/New_York"


def day_url(symbol: str, day: date, side: str = "BID") -> str:
    """Dukascopy URL for one day of M1 candles (month is zero-based)."""
    return (f"https://datafeed.dukascopy.com/datafeed/{symbol.upper()}/{day.year:04d}/"
            f"{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5")


def parse_day(raw: bytes, day: date, price_scale: float = 1000.0,
              assumed_spread_points: float = 30.0) -> pd.DataFrame:
    """Decode one day's ``.bi5`` payload into engine-schema M1 bars (UTC).

    Flat candles with zero volume (Dukascopy fills closed-market minutes that
    way) are dropped, so market closures show up as gaps, as they do in MT5.
    """
    if not raw:
        return empty_bars()
    data = lzma.decompress(raw)
    n = len(data) // RECORD.size
    rec = np.frombuffer(data[: n * RECORD.size], dtype=np.dtype([
        ("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    base = pd.Timestamp(day, tz="UTC")
    df = pd.DataFrame({
        "time": base + pd.to_timedelta(rec["t"].astype("int64"), unit="s"),
        "open": rec["o"] / price_scale, "high": rec["h"] / price_scale,
        "low": rec["l"] / price_scale, "close": rec["c"] / price_scale,
        "tick_volume": rec["v"].astype("float64"), "spread": float(assumed_spread_points),
        "real_volume": 0.0,
    })
    flat = (df["tick_volume"] <= 0) & (df["high"] == df["low"])
    return df.loc[~flat, BAR_COLUMNS].reset_index(drop=True)


def load_folder(folder: str | Path, price_scale: float = 1000.0,
                assumed_spread_points: float = 30.0) -> pd.DataFrame:
    """Parse every ``YYYY-MM-DD.bi5`` file in ``folder`` into one M1 frame."""
    parts = []
    for p in sorted(Path(folder).glob("*.bi5")):
        day = date.fromisoformat(p.stem)
        parts.append(parse_day(p.read_bytes(), day, price_scale, assumed_spread_points))
    if not parts:
        return empty_bars()
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def resample(m1: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """Build higher-timeframe candles from M1 the way an MT5 broker on New York
    close time does: M5-H1 on the UTC grid, H4 on the server-midnight grid
    (17:00 New York), so H4 opens at 22:00/02:00/... UTC in winter."""
    if m1.empty:
        return empty_bars()
    t = pd.DatetimeIndex(m1["time"])
    if tf == Timeframe.H4:
        wall = t.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7)   # server wall clock
        key_wall = wall.floor("4h")
        offset = wall - t.tz_localize(None)                                  # per-row server offset
        key = (key_wall - offset).tz_localize("UTC")
    else:
        key = t.floor(tf.pandas_freq)
    g = m1.groupby(key.to_numpy(), sort=True)
    out = pd.DataFrame({
        "time": pd.DatetimeIndex(g["time"].first().index).tz_localize("UTC")
        if pd.DatetimeIndex(g["time"].first().index).tz is None else g["time"].first().index,
        "open": g["open"].first().to_numpy(), "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(), "close": g["close"].last().to_numpy(),
        "tick_volume": g["tick_volume"].sum().to_numpy(), "spread": g["spread"].max().to_numpy(),
        "real_volume": g["real_volume"].sum().to_numpy(),
    })
    return out.reset_index(drop=True)


def download_days(symbol: str, start: date, end: date, folder: str | Path,
                  fetch=None, retries: int = 3, backoff: float = 2.0) -> dict[str, int]:
    """Save each day's M1 file in ``[start, end)`` as ``folder/YYYY-MM-DD.bi5``.

    Days already on disk are skipped (resume). Saturdays are skipped (gold is
    closed). An empty file is saved for days Dukascopy has no data for, so they
    are not fetched again. ``fetch(url) -> bytes`` is injectable for tests.
    Returns counts: downloaded, skipped, empty, failed.
    """
    import time
    import urllib.error
    import urllib.request

    from ..logging_setup import get_logger
    log = get_logger(__name__)

    def _default_fetch(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "quant-engine"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            raise

    fetch = fetch or _default_fetch
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "empty": 0, "failed": 0}
    for day in pd.date_range(start, end, freq="D", inclusive="left").date:
        path = out / f"{day.isoformat()}.bi5"
        if path.exists() or day.weekday() == 5:
            counts["skipped"] += 1
            continue
        url = day_url(symbol, day)
        for attempt in range(retries):
            try:
                raw = fetch(url)
                break
            except Exception as exc:  # network errors: retry with backoff
                log.warning("dukascopy_retry", extra={"day": str(day), "attempt": attempt + 1, "error": str(exc)})
                time.sleep(backoff * (attempt + 1))
        else:
            counts["failed"] += 1
            log.error("dukascopy_failed", extra={"day": str(day)})
            continue
        path.write_bytes(raw)
        counts["empty" if not raw else "downloaded"] += 1
    return counts
