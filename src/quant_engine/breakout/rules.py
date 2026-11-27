"""Breakout levels of the two strategies, computed from completed D1 candles only.

Both functions return one value per D1 row: the level that row's close is compared with. The
value on row *d* uses rows before *d* only (Strategy A: the previous calendar month; Strategy B:
the previous ``n`` rows), so it is known before candle *d* closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..features.atr import atr as wilder_atr
from .config import BreakoutConfig, StrategyParams


def prev_month_high(d1: pd.DataFrame, min_days: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """(level, note) per D1 row: highest high of the previous calendar month (by trading day).

    The current, incomplete month is never used. The level is NaN when the previous month has
    fewer than ``min_days`` complete D1 candles, or is the first month of the data and does not
    start within its first 4 calendar days (it would be a partial month).
    """
    n = len(d1)
    lvl, note = np.full(n, np.nan), np.full(n, "", dtype=object)
    if n == 0:
        return lvl, note
    days = pd.DatetimeIndex(d1["day"])
    month = days.to_period("M")
    stats = pd.DataFrame({"m": month, "h": d1["high"].to_numpy(float), "ok": d1["complete"].to_numpy(bool),
                          "dom": days.day}).groupby("m").agg(high=("h", "max"), ok=("ok", "sum"), first_dom=("dom", "min"))
    first_month = month[0]
    for i, m in enumerate(month):
        pm = m - 1
        if pm not in stats.index:
            note[i] = "no data for the previous month"
            continue
        row = stats.loc[pm]
        if pm == first_month and row["first_dom"] > 4:
            note[i] = "previous month is the partial first month of the data"
        elif row["ok"] < min_days:
            note[i] = f"previous month has only {int(row['ok'])} complete D1 candles"
        else:
            lvl[i] = float(row["high"])
    return lvl, note


def channel_high(d1: pd.DataFrame, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """(level, note) per D1 row: highest high of the previous ``n`` D1 rows (row itself excluded)."""
    h = pd.Series(d1["high"].to_numpy(float))
    lvl = h.shift(1).rolling(n, min_periods=n).max().to_numpy()
    note = np.where(np.isnan(lvl), f"fewer than {n} previous D1 candles", "").astype(object)
    return lvl, note


def daily_atr(d1: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR of the D1 candles; the value on row *d* includes candle *d* (known at its close)."""
    if d1.empty:
        return np.array([], dtype=float)
    return wilder_atr(d1[["high", "low", "close"]].reset_index(drop=True), period).to_numpy()


@dataclass(frozen=True)
class BreakoutStrategy:
    """Identity, rules text and level function of one breakout strategy.

    ``side`` is +1 (BUY). SELL strategies are disabled in this release; a future SELL strategy is
    a separate entry here with side -1 and its own tests (the trade simulator refuses side -1 now).
    """

    id: str
    name: str
    priority: str
    side: int
    params: StrategyParams
    level_fn: Callable[[pd.DataFrame, "BreakoutConfig", StrategyParams], tuple[np.ndarray, np.ndarray]]
    level_name: str
    entry_rules: str
    stop_rules: str
    target_rules: str
    exit_rules: str
    reset_rules: str

    @property
    def version(self) -> str:
        return self.params.version

    def levels(self, d1: pd.DataFrame, cfg: BreakoutConfig) -> tuple[np.ndarray, np.ndarray]:
        """Breakout level per D1 row and why it is missing where it is NaN."""
        return self.level_fn(d1, cfg, self.params)

    def describe(self) -> dict:
        p = self.params
        return {"id": self.id, "version": self.version, "name": self.name, "priority": self.priority,
                "direction": "BUY" if self.side == 1 else "SELL", "level": self.level_name,
                "entry": self.entry_rules, "stop_loss": self.stop_rules, "take_profit": self.target_rules,
                "exit": self.exit_rules, "reset": self.reset_rules, "params": p.model_dump(exclude={"enabled"}),
                "rule_hash": p.rule_hash()}


_RESET_TEXT = {
    "new_month": "one breakout per calendar month: the strategy re-arms on the first D1 candle of each new month",
    "close_below_level": "re-arms after a completed D1 candle closes at or below the breakout level",
    "new_month_or_close_below": "re-arms on a new month or after a D1 close at or below the level",
}


def _common(p: StrategyParams) -> dict:
    return dict(
        entry_rules="BUY at the open of the next available M5 candle after the breakout D1 candle has closed "
                    "(ask = bid open + spread, plus slippage)",
        stop_rules=f"entry - {p.stop_atr_mult:g} x D1 ATR({p.atr_period}) (ATR includes the breakout candle)",
        target_rules=f"entry + {p.target_r:g} x initial risk",
        exit_rules=f"stop or target, else close at the end of trading session {p.max_sessions} "
                   "(the entry session is session 1); if both are touched in one M5 candle the stop counts first",
        reset_rules=_RESET_TEXT[p.reset_mode])


def strategies(cfg: BreakoutConfig, only_enabled: bool = True) -> list[BreakoutStrategy]:
    """The configured breakout strategies (A first: it is the primary candidate)."""
    a, b = cfg.strategies.prev_month_high, cfg.strategies.channel_20d
    out = [
        BreakoutStrategy("prev_month_high", "Previous Month High Breakout", "primary", 1, a,
                         lambda d1, c, p: prev_month_high(d1, c.daily.min_days_in_month),
                         "highest high of the previous completed calendar month", **_common(a)),
        BreakoutStrategy("channel_20d", f"{b.channel_days}-Day Daily Channel Breakout", "secondary", 1, b,
                         lambda d1, c, p: channel_high(d1, p.channel_days),
                         f"highest high of the previous {b.channel_days} completed D1 candles", **_common(b)),
    ]
    return [s for s in out if s.params.enabled or not only_enabled]


def with_params(s: BreakoutStrategy, **changes) -> BreakoutStrategy:
    """Copy of a strategy with changed parameters (for the neighbourhood table and walk-forward)."""
    from dataclasses import replace
    p = s.params.model_copy(update=changes)
    return replace(s, params=p, **_common(p))
