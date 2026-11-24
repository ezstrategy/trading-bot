"""Break of structure (BOS), change of character (CHoCH) and structure state.

Rules (Strategy Spec v1, "Market structure"):
* Structure is judged on candle CLOSES; a wick through a level is not a break.
* A bullish break is the first close above the most recent confirmed swing
  high (bearish: first close below the most recent confirmed swing low).
  Each swing level can be broken once.
* CHoCH = a break against the current trend (the direction of the previous
  break). Other breaks are BOS.
* State per candle:
  - bullish: last break was up and no close since below the swing low that
    started that leg;
  - bearish: mirror;
  - ranging: no break yet, the leg origin was violated, or the last two
    breaks went in opposite directions within the last ``ranging_lookback``
    candles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..timeframes import Timeframe
from .swings import find_swings

BULLISH, BEARISH, RANGING = "bullish", "bearish", "ranging"
UP, DOWN = 1, -1
EVENT_COLUMNS = ["idx", "time", "known_at", "kind", "direction", "level", "swing_idx", "leg_origin"]


@dataclass
class StructureResult:
    """Structure analysis of one timeframe.

    Attributes:
        swings: confirmed swings (see :func:`find_swings`).
        events: one row per break: idx/time of the breaking candle, known_at
            (its close), kind ('BOS'/'CHoCH'), direction (+1/-1), level broken,
            swing_idx (candle that made the level), leg_origin (opposite swing
            that started the leg, may be NaN).
        state: per-candle state ('bullish'/'bearish'/'ranging') known at that
            candle's close.
    """

    swings: pd.DataFrame
    events: pd.DataFrame
    state: pd.Series


def analyze_structure(bars: pd.DataFrame, tf: Timeframe, length: int,
                      ranging_lookback: int = 10) -> StructureResult:
    """Run the BOS/CHoCH state machine over ``bars`` (oldest first)."""
    swings = find_swings(bars, tf, length)
    n = len(bars)
    close = bars["close"].to_numpy(dtype="float64")
    t = pd.DatetimeIndex(bars["time"]) if n else pd.DatetimeIndex([], tz="UTC")
    by_confirm: dict[int, list[tuple[str, float, int]]] = {}
    for r in swings.itertuples(index=False):
        by_confirm.setdefault(int(r.confirmed_idx), []).append((r.kind, float(r.price), int(r.idx)))

    hi: tuple[float, int] | None = None      # (price, idx) of most recent confirmed swing high
    lo: tuple[float, int] | None = None
    hi_broken = lo_broken = True
    trend = 0
    origin = np.nan                          # leg origin level for the current state
    origin_violated = False
    events: list[tuple] = []
    state = np.full(n, RANGING, dtype=object)

    for j in range(n):
        for kind, price, idx in by_confirm.get(j, ()):
            if kind == "high":
                hi, hi_broken = (price, idx), False
            else:
                lo, lo_broken = (price, idx), False
        c = close[j]
        direction = 0
        if hi is not None and not hi_broken and c > hi[0]:
            direction, level, sidx = UP, hi[0], hi[1]
            hi_broken = True
            leg_origin = lo[0] if lo is not None else np.nan
        elif lo is not None and not lo_broken and c < lo[0]:
            direction, level, sidx = DOWN, lo[0], lo[1]
            lo_broken = True
            leg_origin = hi[0] if hi is not None else np.nan
        if direction:
            kind = "CHoCH" if trend and direction != trend else "BOS"
            events.append((j, t[j], t[j] + tf.delta, kind, direction, level, sidx, leg_origin))
            trend, origin, origin_violated = direction, leg_origin, False
        elif trend == UP and not np.isnan(origin) and c < origin:
            origin_violated = True
        elif trend == DOWN and not np.isnan(origin) and c > origin:
            origin_violated = True

        if trend == 0 or origin_violated:
            state[j] = RANGING
        else:
            state[j] = BULLISH if trend == UP else BEARISH
            if len(events) >= 2:
                prev, last = events[-2], events[-1]
                if prev[4] != last[4] and j - prev[0] < ranging_lookback:
                    state[j] = RANGING

    ev = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return StructureResult(swings, ev, pd.Series(state, index=bars.index, name="state"))
