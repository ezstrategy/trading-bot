"""Multi-timeframe alignment that never looks ahead.

A higher-timeframe candle opened at ``t`` is usable only from ``t + timeframe``. For each
decision time these helpers return the index of the newest higher-timeframe candle that has
closed by then (or -1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeframes import Timeframe


def closed_index(htf_times: pd.DatetimeIndex | pd.Series, htf: Timeframe,
                 decision_times: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Index of the last ``htf`` candle with close time <= each decision time (-1 if none)."""
    closes = pd.DatetimeIndex(htf_times) + htf.delta
    return closes.searchsorted(pd.DatetimeIndex(decision_times), side="right").astype("int64") - 1


def align(values: np.ndarray, idx: np.ndarray, fill: object = np.nan) -> np.ndarray:
    """``values[idx]`` with ``fill`` where ``idx`` is -1."""
    v = np.asarray(values)
    out = v[np.clip(idx, 0, None)] if len(v) else np.full(len(idx), fill, dtype=object)
    if v.dtype.kind in "fc":
        out = out.astype("float64")
        out[idx < 0] = fill
        return out
    out = out.astype(object)
    out[idx < 0] = fill
    return out
