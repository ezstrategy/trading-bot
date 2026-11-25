"""What would have happened after each M5 candle: a fixed-rule trade.

Rules (the same ones :class:`~quant_engine.research.model.PatternModel` sends as signals,
so the mined numbers and the backtest agree):

* Decision at the close of candle ``j``; enter at the open of candle ``j+1``.
  Bars are BID prices: a buy pays the spread on entry, a sell pays it on exit.
* Stop ``sl`` x ATR(14, M5) and target ``tp`` x ATR away from the entry.
* Candles ``j+1 .. j+horizon`` are checked; the stop is checked before the target on
  every candle (conservative). Not done by then: closed at the last candle's close.
* Trades that would span a gap longer than 3 hours (weekend, outage) are left out.

Result: R per candle (``nan`` when no trade is possible) and the index of the exit candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeframes import Timeframe

MAX_GAP = pd.Timedelta(hours=3)


def trade_outcomes(times: pd.DatetimeIndex, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                   spread: np.ndarray, atr: np.ndarray, direction: int, sl_atr: float, tp_atr: float,
                   horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised outcome of a trade started after every candle (see module docs)."""
    n = len(c)
    r = np.full(n, np.nan)
    exit_idx = np.full(n, -1, dtype=np.int64)
    m = n - horizon - 1
    if m <= 0:
        return r, exit_idx
    j = np.arange(m)
    risk = sl_atr * atr[j]
    sp = spread[j]
    entry = o[j + 1] + (sp if direction == 1 else 0.0)
    stop = entry - direction * risk
    tp = entry + direction * tp_atr * atr[j]
    tv = times.to_numpy()
    span = tv[j + horizon] - tv[j]
    ok = (~np.isnan(risk)) & (risk > 0) & (span <= (horizon * Timeframe.M5.delta + MAX_GAP).to_timedelta64())
    done = ~ok
    res = np.full(m, np.nan)
    ex = np.full(m, -1, dtype=np.int64)
    win_r = tp_atr / sl_atr
    for k in range(1, horizon + 1):
        idx = j + k
        if direction == 1:
            hit_stop = l[idx] <= stop
            hit_tp = h[idx] >= tp
        else:
            hit_stop = h[idx] + sp >= stop
            hit_tp = l[idx] + sp <= tp
        s = ~done & hit_stop
        res[s], ex[s] = -1.0, idx[s]
        done |= s
        w = ~done & hit_tp
        res[w], ex[w] = win_r, idx[w]
        done |= w
    last = j + horizon
    rest = ~done
    exit_px = c[last] + (sp if direction == -1 else 0.0)
    res[rest] = ((exit_px - entry) * direction / risk)[rest]
    ex[rest] = last[rest]
    res[~ok] = np.nan
    ex[~ok] = -1
    r[:m], exit_idx[:m] = res, ex
    return r, exit_idx
