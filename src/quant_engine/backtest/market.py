"""Market data for the backtester and the view a strategy gets at each decision time.

:class:`MarketData` holds the candles of every timeframe (UTC open times, BID prices) as NumPy
arrays. The base timeframe is M5: the engine steps through M5 candles and a decision is made at
each candle's close. :meth:`MarketData.truncate` cuts every timeframe at a moment, keeping only
candles closed by then; the look-ahead harness (:mod:`.lookahead`) uses it to prove a strategy
gives the same signals with and without future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..timeframes import Timeframe
from .mtf import closed_index

BASE = Timeframe.M5


@dataclass
class Frame:
    """Arrays of one timeframe."""

    tf: Timeframe
    df: pd.DataFrame
    t: pd.DatetimeIndex = field(init=False)
    o: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)
    l: np.ndarray = field(init=False)
    c: np.ndarray = field(init=False)
    spread_pts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.df = self.df.reset_index(drop=True)
        self.t = pd.DatetimeIndex(self.df["time"])
        if self.t.tz is None or str(self.t.tz) != "UTC":
            raise ValueError(f"{self.tf.value} times must be tz-aware UTC")
        if not self.t.is_monotonic_increasing or self.t.has_duplicates:
            raise ValueError(f"{self.tf.value} candles must be unique and in time order (check the data version)")
        self.o, self.h, self.l, self.c = (self.df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
        self.spread_pts = (self.df["spread"].to_numpy(dtype="float64") if "spread" in self.df
                           else np.full(len(self.df), np.nan))

    def __len__(self) -> int:
        return len(self.t)

    @property
    def close_times(self) -> pd.DatetimeIndex:
        """When each candle becomes final."""
        return self.t + self.tf.delta


class MarketData:
    """All timeframes of one instrument for one backtest.

    Args:
        bars: frames per timeframe; M5 is required, M1 (intrabar), M15 and H4 are optional.
        point: price of one point (0.01 for XAUUSD with 2 digits).
        dataset_ids: dataset versions the frames came from (recorded in results).
    """

    def __init__(self, bars: dict[Timeframe, pd.DataFrame], point: float = 0.01,
                 dataset_ids: dict[str, str] | None = None) -> None:
        if BASE not in bars or bars[BASE].empty:
            raise ValueError("M5 candles are required")
        self.frames = {tf: Frame(tf, df) for tf, df in bars.items() if df is not None and not df.empty}
        self.point = point
        self.dataset_ids = dict(dataset_ids or {})
        self.base = self.frames[BASE]
        self._m1_index: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.base)

    def has(self, tf: Timeframe) -> bool:
        """Whether a timeframe is loaded."""
        return tf in self.frames

    def frame(self, tf: Timeframe) -> Frame:
        """One timeframe (KeyError if not loaded)."""
        return self.frames[tf]

    @property
    def decision_times(self) -> pd.DatetimeIndex:
        """Close time of each base candle (when the engine asks strategies for signals)."""
        return self.base.close_times

    def closed_htf_index(self, tf: Timeframe) -> np.ndarray:
        """For each base candle, the last ``tf`` candle closed at that base candle's close."""
        return closed_index(self.frames[tf].t, tf, self.decision_times)

    def m1_slices(self) -> tuple[np.ndarray, np.ndarray]:
        """For each base candle, ``[start, stop)`` positions of the M1 candles inside it
        (start == stop when there is no M1 data for it)."""
        if self._m1_index is None:
            if Timeframe.M1 not in self.frames:
                z = np.zeros(len(self.base), dtype="int64")
                self._m1_index = (z, z)
            else:
                m1t = self.frames[Timeframe.M1].t
                a = m1t.searchsorted(self.base.t, side="left")
                b = m1t.searchsorted(self.base.t + BASE.delta, side="left")
                self._m1_index = (a.astype("int64"), b.astype("int64"))
        return self._m1_index

    def truncate(self, end: pd.Timestamp) -> "MarketData":
        """Copy holding only candles that have closed by ``end`` (UTC), on every timeframe."""
        end = pd.Timestamp(end)
        bars = {tf: f.df[f.close_times <= end] for tf, f in self.frames.items()}
        return MarketData(bars, self.point, self.dataset_ids)

    def between(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> "MarketData":
        """Copy limited to candles opening in UTC ``[start, end)`` (warm-up is the caller's job)."""
        def cut(f: Frame) -> pd.DataFrame:
            m = np.ones(len(f), dtype=bool)
            if start is not None:
                m &= f.t >= pd.Timestamp(start)
            if end is not None:
                m &= f.t < pd.Timestamp(end)
            return f.df[m]
        return MarketData({tf: cut(f) for tf, f in self.frames.items()}, self.point, self.dataset_ids)


@dataclass
class MarketView:
    """What a strategy may look at when the base candle ``i`` has just closed.

    ``time`` is the decision time (close of candle ``i``). Everything a strategy computes in
    ``prepare`` must be causal; the look-ahead harness checks that.
    """

    data: MarketData
    i: int
    regime: str

    @property
    def time(self) -> pd.Timestamp:
        """Decision time (UTC)."""
        return self.data.base.t[self.i] + BASE.delta

    def closes(self, n: int) -> np.ndarray:
        """The last ``n`` closed base-candle closes."""
        return self.data.base.c[max(0, self.i - n + 1): self.i + 1]
