"""Parquet storage for bars and ticks.

Layout::

    data/raw/XAUUSD/M5.parquet
    data/raw/XAUUSD/ticks/2026-11-24.parquet

Writes are upserts: new rows replace stored rows with the same time, so
re-downloading a range is always safe. Files are written atomically.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..timeframes import Timeframe
from .schema import empty_bars, empty_ticks


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class ParquetStore:
    """Bar and tick storage rooted at ``root`` (e.g. ``data``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ---------------------------------------------------------------- bars
    def bars_path(self, canonical: str, tf: Timeframe) -> Path:
        return self.root / "raw" / canonical.upper() / f"{tf.value}.parquet"

    def read_bars(self, canonical: str, tf: Timeframe, start: pd.Timestamp | None = None,
                  end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Stored bars, optionally limited to UTC ``[start, end)``."""
        p = self.bars_path(canonical, tf)
        if not p.exists():
            return empty_bars()
        df = pd.read_parquet(p)
        if start is not None:
            df = df[df["time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["time"] < pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def write_bars(self, canonical: str, tf: Timeframe, new: pd.DataFrame) -> int:
        """Upsert bars; returns the total number of stored rows."""
        if new.empty:
            return len(self.read_bars(canonical, tf))
        old = self.read_bars(canonical, tf)
        df = new if old.empty else pd.concat([old, new], ignore_index=True)
        df = (df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))
        _atomic_write(df, self.bars_path(canonical, tf))
        return len(df)

    def last_bar_time(self, canonical: str, tf: Timeframe) -> pd.Timestamp | None:
        df = self.read_bars(canonical, tf)
        return None if df.empty else df["time"].iat[-1]

    # --------------------------------------------------------------- ticks
    def ticks_path(self, canonical: str, day: pd.Timestamp) -> Path:
        return self.root / "raw" / canonical.upper() / "ticks" / f"{pd.Timestamp(day):%Y-%m-%d}.parquet"

    def write_ticks(self, canonical: str, ticks: pd.DataFrame) -> int:
        """Upsert ticks into one file per UTC day; returns rows written."""
        if ticks.empty:
            return 0
        for day, part in ticks.groupby(ticks["time"].dt.floor("D")):
            p = self.ticks_path(canonical, day)
            old = pd.read_parquet(p) if p.exists() else empty_ticks()
            df = part if old.empty else pd.concat([old, part], ignore_index=True)
            df = df.drop_duplicates(keep="last").sort_values("time", kind="stable").reset_index(drop=True)
            _atomic_write(df, p)
        return len(ticks)

    def read_ticks(self, canonical: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Stored ticks in UTC ``[start, end)``."""
        days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end), freq="D")
        parts = [pd.read_parquet(p) for d in days if (p := self.ticks_path(canonical, d)).exists()]
        if not parts:
            return empty_ticks()
        df = pd.concat(parts, ignore_index=True)
        return df[(df["time"] >= pd.Timestamp(start)) & (df["time"] < pd.Timestamp(end))].reset_index(drop=True)
