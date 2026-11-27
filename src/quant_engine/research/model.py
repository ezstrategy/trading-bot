"""Turn mined patterns into signals the existing backtester and live engine can use.

A pattern fires on the close of an M5 candle when its event happens in its H4 market
(and session). The signal is a limit order at that close (fills on the next candle in
practice), stop ``sl_atr`` x ATR away, a single target ``tp_atr`` x ATR away (TP1 = TP2,
so the whole position closes there), and a time stop after ``horizon`` candles, matching
the rules the pattern was measured with (:mod:`.outcomes`). Signals only; nothing is traded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..features.context import MarketFeatures
from ..features.news import NewsCalendar
from ..strategy.entry import EntryModel, Signal
from ..timeframes import Timeframe
from .events import build_events
from .outcomes import SWING_MAX_GAP

PATTERN_KEYS = ("event", "regime", "session", "direction", "sl_atr", "tp_atr", "horizon")


def regime_kwargs(cfg: StrategyConfig) -> dict:
    """The backtest's H4 regime settings, as keyword arguments for ``trend_regime_series``."""
    b = cfg.backtest
    return {"ema_period": b.regime_ema_period, "slope_bars": b.regime_slope_bars,
            "min_slope_atr": b.regime_min_slope_atr}


def load_patterns(path: str | Path, verdicts: list[str] | None = None) -> list[dict]:
    """Patterns saved by ``python -m quant_engine patterns`` (optionally only some verdicts)."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"No pattern file at {p}. Run: python -m quant_engine patterns")
    data = json.loads(p.read_text(encoding="utf-8"))
    pats = data.get("patterns", [])
    if verdicts is not None:
        pats = [x for x in pats if x.get("verdict") in verdicts]
    return pats


class PatternModel(EntryModel):
    """Signals from a list of patterns (dicts with :data:`PATTERN_KEYS`), first match wins."""

    name = "patterns"

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig, news: NewsCalendar | None = None,
                 patterns: list[dict] | None = None, start: pd.Timestamp | None = None,
                 timeframe: str | Timeframe = Timeframe.M5) -> None:
        """``timeframe``: candles the patterns were found on (M5, H1 or H4). Signals are still
        sent on the M5 candle that closes together with that candle and managed on M5."""
        super().__init__(features, cfg, news)
        self.tf = tf = Timeframe.parse(timeframe)
        self.patterns = [{k: p[k] for k in PATTERN_KEYS} | {"label": p.get("description", p["event"]),
                                                            "exit_units": p.get("exit_units", "atr")}
                         for p in (patterns or [])]
        self.et = build_events(features.bars[tf], features.bars[Timeframe.H4], cfg.research.for_timeframe(tf.value),
                               cfg.sessions, features.point, cfg.backtest.assumed_spread_points,
                               regime_kwargs(cfg), tf)
        n = len(self.et.c)
        self.start_idx = 0 if start is None else int(self.et.times.searchsorted(start))
        # M5 candle closing at the same moment as each tf candle (or the last one before it)
        m5_close = features.times[Timeframe.M5] + Timeframe.M5.delta
        tf_close = self.et.times + tf.delta
        self.to_m5 = m5_close.searchsorted(tf_close, side="right") - 1
        self.from_m5 = {int(m): k for k, m in enumerate(self.to_m5) if m >= 0}
        self.masks = []
        for p in self.patterns:
            m = self.et.events.get(p["event"], np.zeros(n, dtype=bool)) & (self.et.regime == p["regime"])
            if p["session"] != "any":
                m = m & (self.et.session == p["session"])
            self.masks.append(m & ~self.et.blocked)

    def candidates(self) -> list[int]:
        if not self.masks:
            return []
        any_m = np.logical_or.reduce(self.masks)
        return sorted({int(self.to_m5[k]) for k in np.nonzero(any_m)[0]
                       if k >= self.start_idx and self.to_m5[k] >= 0})

    def evaluate(self, j: int) -> Signal | None:
        t = self.decision_time(j)
        if self.news is not None and self.news.blackout(t) is not None:
            return self._reject("1 news blackout")
        k = self.from_m5.get(j)
        if k is None:
            return None
        for p, m in zip(self.patterns, self.masks):
            if not m[k]:
                continue
            a, sp, d = float(self.et.atr[k]), float(self.et.spread[k]), int(p["direction"])
            if p.get("exit_units") == "usd":      # stop / target given in dollars
                a = 1.0
            if not a > 0:
                return self._reject("no ATR yet")
            entry = float(self.et.c[k]) + (sp if d == 1 else 0.0)
            sl = entry - d * p["sl_atr"] * a
            tp = entry + d * p["tp_atr"] * a
            hz = int(p["horizon"]) * self.tf.minutes // 5      # in M5 candles
            self._reject(f"signal: {p['event']} / {p['regime']}")
            return Signal(time=t, m5_idx=j, direction=d, entry=entry, stop_loss=sl, tp1=tp, tp2=tp, tp3=None,
                          score=70, grade="A", sweep_idx=j, sweep_extreme=sl, kill_zone=str(self.et.session[k]),
                          spread=sp, reasons=[p["label"]], strategy="patterns",
                          time_stop_candles=max(hz - 1, 1),
                          close_by=t + (hz + self.cfg.entry.expiry_candles + 2) * Timeframe.M5.delta
                          + (SWING_MAX_GAP if self.tf.minutes >= 60 else pd.Timedelta(0)))
        return None
