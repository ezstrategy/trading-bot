"""Look-ahead / repainting harness.

A strategy is causal when the intents it gives at the close of candle ``k`` are the same whether
or not it has seen any later data. :func:`check_no_lookahead` re-runs ``prepare`` + ``on_bar``
on data truncated at several decision times and compares every intent up to each cut with the
full-data run. Any difference is look-ahead (or repainting) and is reported.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .market import MarketData, MarketView


def _intents(strategy, data: MarketData, upto: int | None = None) -> list[tuple]:
    strategy.prepare(data)
    out = []
    last = len(data) if upto is None else min(upto + 1, len(data))
    for i in range(last):
        for it in strategy.on_bar(MarketView(data, i, "n/a")) or []:
            out.append((str(data.base.t[i]), it.side, round(float(it.stop), 6),
                        None if it.target is None else round(float(it.target), 6), it.target_r,
                        it.time_exit_bars, None if it.risk_bounds is None else tuple(round(x, 6) for x in it.risk_bounds)))
    return out


def check_no_lookahead(make_strategy: Callable[[], object], data: MarketData, cuts: int = 8,
                       seed: int = 0, cut_times: list[pd.Timestamp] | None = None) -> dict:
    """Compare full-data intents with intents computed on data cut at ``cuts`` decision times.

    Returns ``{"ok": bool, "cuts": [...], "differences": [...]}``. Cut points are spread over the
    candles where the full run gave intents (plus random ones), so the check is not vacuous.
    """
    full = _intents(make_strategy(), data)
    times = data.decision_times
    if cut_times is None:
        rng = np.random.default_rng(seed)
        sig_idx = sorted({int(times.searchsorted(pd.Timestamp(x[0]) + pd.Timedelta(minutes=5))) for x in full})
        pick = list(rng.choice(sig_idx, size=min(len(sig_idx), cuts // 2), replace=False)) if sig_idx else []
        pick += list(rng.integers(len(data) // 4, len(data), size=cuts - len(pick)))
        cut_times = [times[min(int(k), len(times) - 1)] for k in sorted(set(pick))]
    diffs, done = [], []
    for ct in cut_times:
        cut_data = data.truncate(ct)
        if not len(cut_data):
            continue
        got = _intents(make_strategy(), cut_data)
        want = [x for x in full if pd.Timestamp(x[0]) + pd.Timedelta(minutes=5) <= ct]
        done.append(str(ct))
        if got != want:
            extra = [x for x in got if x not in want][:5]
            missing = [x for x in want if x not in got][:5]
            diffs.append({"cut": str(ct), "only_with_cut_data": extra, "only_with_full_data": missing})
    return {"ok": not diffs, "cuts": done, "intents_full": len(full), "differences": diffs}
