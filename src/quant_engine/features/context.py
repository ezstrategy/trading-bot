"""Multi-timeframe market context with no look-ahead.

:class:`MarketFeatures` computes every feature once over the full history
(fast), and :meth:`MarketFeatures.snapshot` returns what was knowable at a
decision time *t*: only candles closed by *t* on each timeframe, swings
confirmed by *t*, pools not yet taken by *t*, sweeps whose reclaim closed
by *t*. Phase 3 (entry model and signals) reads these snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..timeframes import Timeframe
from . import liquidity as liq
from .atr import atr
from .imbalance import displacement, find_fvgs
from .sessions import (in_kill_zone, in_kill_zone_or_grace, new_signals_blocked, session_labels,
                       trading_day)
from .structure import BEARISH, BULLISH, RANGING, StructureResult, analyze_structure
from .volatility import atr_time_of_day_ratio, spread_ratio, volatility_regime

BULL, BEAR, NONE = "BULL", "BEAR", "NONE"
STRUCTURE_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4)


def htf_bias(h4_state: str, h1_state: str) -> tuple[str, int]:
    """Bias table from the spec: returns (BULL/BEAR/NONE, HTF score points 20/10/0)."""
    if h1_state == BULLISH and h4_state in (BULLISH, RANGING):
        return BULL, 20 if h4_state == BULLISH else 10
    if h1_state == BEARISH and h4_state in (BEARISH, RANGING):
        return BEAR, 20 if h4_state == BEARISH else 10
    return NONE, 0


def m15_allows(m15_state: str, direction: int) -> bool:
    """M15 confirmation gate: the M15 state must not oppose the trade."""
    return m15_state != (BEARISH if direction == 1 else BULLISH)


@dataclass
class DealingRange:
    """H1 dealing range from the last confirmed swing low to the last confirmed swing high."""

    low: float
    high: float

    @property
    def equilibrium(self) -> float:
        return (self.low + self.high) / 2

    def zone(self, price: float) -> str:
        """'discount' below equilibrium, 'premium' above, 'equilibrium' exactly at it."""
        eq = self.equilibrium
        return "discount" if price < eq else "premium" if price > eq else "equilibrium"


@dataclass
class MarketSnapshot:
    """Everything known at ``decision_time`` (the close of the last M5 candle)."""

    decision_time: pd.Timestamp
    m5_idx: int
    m5_bar: pd.Series
    atr: dict[Timeframe, float]
    state: dict[Timeframe, str]
    bias: str
    htf_points: int
    m15_allows_long: bool
    m15_allows_short: bool
    session: str
    in_kill_zone: bool
    in_kill_zone_or_grace: bool
    new_signals_blocked: bool
    dealing_range: DealingRange | None
    pools: pd.DataFrame
    recent_sweeps: list[liq.Sweep]
    recent_fvgs: pd.DataFrame
    m5_events: pd.DataFrame
    vol_ratio: float
    vol_regime: str
    spread_ratio: float
    notes: list[str] = field(default_factory=list)


def _last_closed_idx(times: pd.DatetimeIndex, tf: Timeframe, when: pd.Timestamp) -> int:
    """Row of the last candle whose close (open + tf) is <= ``when``; -1 if none."""
    return int((times + tf.delta).searchsorted(when, side="right")) - 1


class MarketFeatures:
    """Precomputed features for one instrument.

    Args:
        bars: closed candles per timeframe (engine bar schema, UTC). M5, M15,
            H1 and H4 are required.
        cfg: strategy settings.
        point: price of one tick (MT5 ``symbol_info.point``, 0.01 for most XAUUSD feeds).
    """

    def __init__(self, bars: dict[Timeframe, pd.DataFrame], cfg: StrategyConfig, point: float) -> None:
        missing = [tf.value for tf in STRUCTURE_TFS if tf not in bars]
        if missing:
            raise ValueError(f"missing timeframes: {missing}")
        self.cfg, self.point = cfg, point
        self.bars = {tf: bars[tf].reset_index(drop=True) for tf in STRUCTURE_TFS}
        self.times = {tf: pd.DatetimeIndex(b["time"]) for tf, b in self.bars.items()}
        self.atr = {tf: atr(b, cfg.atr_period) for tf, b in self.bars.items()}
        self.structure: dict[Timeframe, StructureResult] = {
            tf: analyze_structure(b, tf, cfg.swing_length(tf), cfg.structure.ranging_lookback)
            for tf, b in self.bars.items()}

        m5, t5 = self.bars[Timeframe.M5], self.times[Timeframe.M5]
        s = cfg.sessions
        self.session = session_labels(t5, s)
        self.kill_zone = in_kill_zone(t5, s)
        self.kill_zone_grace = in_kill_zone_or_grace(t5, s)
        self.blocked = new_signals_blocked(t5, s)
        self.displacement = displacement(m5, self.atr[Timeframe.M5], cfg.displacement.min_body_atr,
                                         cfg.displacement.min_body_ratio)
        self.fvgs = find_fvgs(m5, Timeframe.M5, self.atr[Timeframe.M5], point,
                              cfg.fvg.min_size_atr, cfg.fvg.min_size_spreads)
        self.vol_ratio = atr_time_of_day_ratio(m5, self.atr[Timeframe.M5], cfg.volatility.median_days, s)
        self.vol_regime = volatility_regime(self.vol_ratio, cfg.volatility)
        self.spread_ratio = (spread_ratio(m5, cfg.volatility.median_days) if "spread" in m5
                             else pd.Series(np.nan, index=m5.index))
        self.pools = self._build_pools()
        self.pools["day"] = trading_day(self.pools["source_time"], s) if len(self.pools) else []
        self.sweeps = liq.detect_sweeps(m5, self.pools, self.atr[Timeframe.M5], point,
                                        cfg.liquidity, s)

    # ----------------------------------------------------------------- pools
    def _atr_known_at(self, tf: Timeframe, when: pd.Series) -> pd.Series:
        """ATR of ``tf`` from the last candle closed at each instant in ``when``."""
        closes = self.times[tf] + tf.delta
        idx = closes.searchsorted(pd.DatetimeIndex(when), side="right") - 1
        vals = self.atr[tf].to_numpy()
        return pd.Series(np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan), index=when.index)

    def _build_pools(self) -> pd.DataFrame:
        m5 = self.bars[Timeframe.M5]
        L = self.cfg.liquidity
        rows = liq.session_pools(m5, self.cfg.sessions)
        rows += liq.swing_pools(self.structure[Timeframe.H1].swings, "h1_swing", liq.HTF)
        m5_sw = self.structure[Timeframe.M5].swings
        rows += liq.swing_pools(m5_sw, "m5_swing", liq.MINOR)
        both = pd.concat([m5_sw, self.structure[Timeframe.M15].swings], ignore_index=True)
        both = both.sort_values("known_at", kind="stable").reset_index(drop=True)
        tol = L.equal_level_atr_m15 * self._atr_known_at(Timeframe.M15, both["known_at"])
        rows += liq.equal_level_pools(both, tol, pd.Timedelta(days=L.swing_pool_lookback_days))
        return liq.build_pools(m5, rows)

    def _pools_at(self, t: pd.Timestamp) -> pd.DataFrame:
        """Live pools at ``t``: today's Asia/London ranges, the previous trading day's
        high/low, and swing/equal-level pools from the lookback window."""
        p = liq.live_pools(self.pools, t)
        today = trading_day(pd.DatetimeIndex([t - pd.Timedelta(microseconds=1)]), self.cfg.sessions)[0]
        prev_days = self.pools.loc[(self.pools["kind"] == "previous_day") & (self.pools["day"] < today), "day"]
        prev = prev_days.max() if len(prev_days) else None
        lookback = t - pd.Timedelta(days=self.cfg.liquidity.swing_pool_lookback_days)
        keep = (((p["tier"] == liq.SESSION) & (p["day"] == today))
                | ((p["kind"] == "previous_day") & (p["day"] == prev))
                | (p["kind"].isin(["h1_swing", "m5_swing", "equal"]) & (p["known_at"] >= lookback)))
        p = p[keep].assign(_rank=lambda d: d["tier"].map(liq.TIER_RANK))
        p = p.sort_values(["_rank", "known_at"], kind="stable").drop_duplicates(["side", "price"], keep="first")
        return p.drop(columns="_rank").sort_values("price", kind="stable")

    # -------------------------------------------------------------- snapshot
    def snapshot(self, decision_time: pd.Timestamp) -> MarketSnapshot:
        """What was knowable at ``decision_time`` (normally the close of an M5 candle)."""
        t = pd.Timestamp(decision_time)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        idx = {tf: _last_closed_idx(self.times[tf], tf, t) for tf in STRUCTURE_TFS}
        j = idx[Timeframe.M5]
        if j < 0:
            raise ValueError(f"no closed M5 candle at {t}")
        notes: list[str] = []
        state, atr_now = {}, {}
        for tf in STRUCTURE_TFS:
            k = idx[tf]
            state[tf] = self.structure[tf].state.iat[k] if k >= 0 else RANGING
            atr_now[tf] = float(self.atr[tf].iat[k]) if k >= 0 else float("nan")
            if k < 0:
                notes.append(f"no closed {tf.value} candle yet")
        bias, pts = htf_bias(state[Timeframe.H4], state[Timeframe.H1])

        h1_sw = self.structure[Timeframe.H1].swings
        known = h1_sw[h1_sw["known_at"] <= t]
        hi, lo = known[known["kind"] == "high"], known[known["kind"] == "low"]
        dr = DealingRange(float(lo["price"].iat[-1]), float(hi["price"].iat[-1])) if len(hi) and len(lo) else None

        pools = self._pools_at(t)

        window = self.cfg.liquidity.sweep_valid_candles
        sweeps = [s for s in self.sweeps if s.known_at <= t and s.sweep_idx > j - window]
        f = self.fvgs
        fvgs = f[(f["known_at"] <= t) & (f["c3_idx"] > j - 2 * window)
                 & (f["invalidated_at"].isna() | (f["invalidated_at"] > t))]
        ev = self.structure[Timeframe.M5].events
        m5_events = ev[ev["known_at"] <= t].tail(5)

        return MarketSnapshot(
            decision_time=t, m5_idx=j, m5_bar=self.bars[Timeframe.M5].iloc[j], atr=atr_now, state=state,
            bias=bias, htf_points=pts,
            m15_allows_long=m15_allows(state[Timeframe.M15], 1),
            m15_allows_short=m15_allows(state[Timeframe.M15], -1),
            session=str(self.session[j]), in_kill_zone=bool(self.kill_zone[j]),
            in_kill_zone_or_grace=bool(self.kill_zone_grace[j]),
            new_signals_blocked=bool(self.blocked[j]), dealing_range=dr,
            pools=pools.reset_index(drop=True), recent_sweeps=sweeps,
            recent_fvgs=fvgs.reset_index(drop=True), m5_events=m5_events.reset_index(drop=True),
            vol_ratio=float(self.vol_ratio.iat[j]), vol_regime=str(self.vol_regime.iat[j]),
            spread_ratio=float(self.spread_ratio.iat[j]), notes=notes)
