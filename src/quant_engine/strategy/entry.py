"""Entry model and confidence score (Strategy Spec v1).

On every closed M5 candle that completes a change of character (CHoCH), the
model checks, in order:

1. no signal block (Friday after 10:00 NY, Sunday), no news blackout (when a
   calendar is given) and the candle is inside a kill zone or its 30-minute
   grace period;
2. H4/H1 bias agrees with the CHoCH direction, and M15 does not oppose it;
3. a valid liquidity sweep against the trade happened within the last 12 M5
   candles and price has not closed beyond the sweep extreme since;
4. the leg from the sweep to the CHoCH contains a displacement candle and a
   valid fair value gap (c2 inside the leg, not invalidated);
5. the entry (FVG/OB overlap edge, else FVG midpoint) is in the leg's
   discount (longs) / premium (shorts);
6. volatility is not dead or chaotic and the spread is normal;
7. stop beyond the sweep extreme passes the 0.5-2.5 ATR check;
8. TP2 (nearest untaken liquidity beyond TP1) is at least 2R away.

A setup passing every gate is scored 0-100 and graded A+ / A / B.
Everything is decided from :class:`MarketFeatures` data known at the
candle's close, so there is no look-ahead.

GUESS (spec ambiguities, see the Phase 3 report):
* Several valid sweeps: the highest-tier pool wins, then the most recent.
* Several FVGs: only those whose entry is in the leg's discount/premium are
  eligible; the one closest to the leg's 50% level is used.
* TP1 = nearest confirmed M5 swing at least 1R away, else exactly 1R.
* TP2 = nearest pool (previous day, Asia, London, H1 swing) beyond TP1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..features import liquidity as liq
from ..features.context import BEAR, BULL, MarketFeatures
from ..features.imbalance import order_block
from ..features.news import NewsCalendar
from ..features.sessions import KILL_ZONES
from ..features.volatility import CHAOS, DEAD
from ..timeframes import Timeframe

TARGET_POOLS = ("previous_day", "asia", "london", "h1_swing")
SWEEP_POINTS = {liq.HTF: 15, liq.SESSION: 15, liq.ENGINEERED: 12, liq.MINOR: 5}


@dataclass
class Signal:
    """A trade idea as it would be posted to Telegram."""

    time: pd.Timestamp            # close of the CHoCH candle (when the signal is sent)
    m5_idx: int
    direction: int                # +1 buy, -1 sell
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float | None
    score: int
    grade: str
    sweep_idx: int
    sweep_extreme: float
    kill_zone: str
    spread: float
    factors: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    strategy: str = "v1"          # which model produced it (v1, bullish, bearish, sideways)
    time_stop_candles: int | None = None   # mode-specific time stop (None = entry.time_stop_candles)
    close_by: pd.Timestamp | None = None   # close anything still open then (None = kill zone end + session_close_hours)

    @property
    def side(self) -> str:
        return "BUY" if self.direction == 1 else "SELL"

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    def r_multiple(self, price: float) -> float:
        """Distance from entry to ``price`` in R, positive in the trade's favour."""
        return (price - self.entry) * self.direction / self.risk


class EntryModel:
    """Strategy Spec v1 (sweep + reversal with the HTF bias). ``rejections`` counts why
    candidates failed. The helper methods are shared with the regime modes in :mod:`.modes`."""

    name = "v1"

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig, news: "NewsCalendar | None" = None) -> None:
        self.f, self.cfg, self.news = features, cfg, news
        self.m5 = features.bars[Timeframe.M5]
        self.o, self.h, self.l, self.c = (self.m5[k].to_numpy(dtype="float64")
                                          for k in ("open", "high", "low", "close"))
        self.atr = features.atr[Timeframe.M5].to_numpy(dtype="float64")
        self.disp = features.displacement.to_numpy()
        ev = features.structure[Timeframe.M5].events
        self.choch = {int(r.idx): int(r.direction) for r in ev[ev["kind"] == "CHoCH"].itertuples()}
        sw = features.structure[Timeframe.M5].swings
        self.m5_swings = sw
        self.spread_pts = self.m5["spread"].to_numpy(dtype="float64") if "spread" in self.m5 else None
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------- helpers
    def _reject(self, why: str) -> None:
        self.rejections[why] = self.rejections.get(why, 0) + 1

    def spread_price(self, j: int) -> float:
        pts = self.spread_pts[j] if self.spread_pts is not None else np.nan
        if not pts or np.isnan(pts):
            pts = self.cfg.backtest.assumed_spread_points
        return float(pts) * self.f.point

    def candidates(self) -> list[int]:
        """M5 rows that complete a CHoCH (the only candles worth evaluating)."""
        return sorted(self.choch)

    # ------------------------------------------------------ shared building blocks
    def decision_time(self, j: int) -> pd.Timestamp:
        return self.f.times[Timeframe.M5][j] + Timeframe.M5.delta

    def session_gates(self, j: int, t: pd.Timestamp) -> bool:
        """Gate 1: signal block, kill zone (+grace) and news blackout."""
        if self.f.blocked[j]:
            return self._reject("1 signal block (Fri after 10:00 NY / Sunday)")
        if not self.f.kill_zone_grace[j]:
            return self._reject("1 outside kill zone")
        if self.news is not None and self.news.blackout(t) is not None:
            return self._reject("1 news blackout")
        return True

    def intact_sweeps(self, snap, d: int, j: int) -> list:
        """Recent sweeps against direction ``d`` whose extreme no close has broken since."""
        out = []
        for s in snap.recent_sweeps:
            if s.direction != d or s.sweep_idx > j:
                continue
            closes = self.c[s.reclaim_idx + 1:j + 1]
            if not ((closes < s.extreme).any() if d == 1 else (closes > s.extreme).any()):
                out.append(s)
        return out

    def leg_entry(self, a: int, j: int, d: int, t: pd.Timestamp, extreme: float):
        """Gates 4-5 on the leg from candle ``a`` (sweep / pullback extreme) to ``j``:
        displacement, a valid FVG, and an entry in the leg's discount (longs) / premium
        (shorts). Returns (entry, overlap_with_OB, largest displacement body in ATR) or None."""
        leg = np.arange(a, j + 1)
        disp_idx = leg[self.disp[leg] == d]
        if not len(disp_idx):
            return self._reject("4 no displacement in the leg")
        body_atr = max(abs(self.c[k] - self.o[k]) / self.atr[k] for k in disp_idx)
        fv = self.f.fvgs
        fv = fv[(fv["direction"] == d) & fv["valid_size"] & (fv["c3_idx"] >= a + 1) & (fv["c3_idx"] <= j)
                & (fv["invalidated_at"].isna() | (fv["invalidated_at"] > t))]
        if fv.empty:
            return self._reject("4 no valid FVG in the leg")
        if d == 1:
            leg_hi, leg_lo = self.h[a:j + 1].max(), extreme
        else:
            leg_hi, leg_lo = extreme, self.l[a:j + 1].min()
        mid = (leg_hi + leg_lo) / 2
        ob = order_block(self.m5, int(disp_idx[0]), d)
        if ob is not None:
            closes = self.c[ob[2] + 1:j + 1]
            if (d == 1 and (closes < ob[0]).any()) or (d == -1 and (closes > ob[1]).any()):
                ob = None
        options = []
        for r in fv.itertuples():
            lo, hi = r.bottom, r.top
            overlap = ob is not None and min(hi, ob[1]) > max(lo, ob[0])
            if overlap:
                zlo, zhi = max(lo, ob[0]), min(hi, ob[1])
                entry = zhi if d == 1 else zlo
            else:
                entry = r.ce
            if (d == 1 and entry <= mid) or (d == -1 and entry >= mid):
                options.append((abs(entry - mid), entry, overlap))
        if not options:
            return self._reject("5 entry not in the leg's discount/premium")
        _, entry, overlap = min(options, key=lambda x: x[0])
        return float(entry), bool(overlap), float(body_atr)

    def market_ok(self, snap, j: int) -> float | None:
        """Gate 6: volatility and spread. Returns the spread in price, or None."""
        if snap.vol_regime in (DEAD, CHAOS):
            return self._reject(f"6 volatility {snap.vol_regime}")
        if not np.isnan(snap.spread_ratio) and snap.spread_ratio > self.cfg.volatility.max_spread_ratio:
            return self._reject("6 spread too wide")
        return self.spread_price(j)

    def stop_for(self, extreme: float, entry: float, d: int, j: int, spread: float):
        """Gate 7: stop beyond ``extreme`` with the buffer; returns (stop, risk) or None."""
        e = self.cfg.entry
        atr = self.atr[j]
        buffer = max(e.sl_buffer_atr * atr, e.sl_buffer_spreads * spread)
        sl = extreme - buffer if d == 1 else extreme + buffer
        risk = abs(entry - sl)
        if not (e.min_stop_atr * atr <= risk <= e.max_stop_atr * atr):
            return self._reject("7 stop distance outside 0.5-2.5 ATR")
        if risk < self.cfg.volatility.min_stop_spreads * spread:
            return self._reject("7 stop smaller than 5 spreads")
        return float(sl), float(risk)

    def first_target(self, entry: float, risk: float, d: int, t: pd.Timestamp) -> float:
        """TP1: nearest confirmed M5 swing at least ``tp1_min_r`` away, else exactly that."""
        sw = self.m5_swings
        sw = sw[(sw["known_at"] <= t) & (sw["kind"] == ("high" if d == 1 else "low"))]
        min_tp1 = entry + d * self.cfg.entry.tp1_min_r * risk
        beyond = sw[(sw["price"] - min_tp1) * d >= 0]["price"]
        return float(beyond.min() if d == 1 else beyond.max()) if len(beyond) else float(min_tp1)

    def pool_levels(self, snap, d: int, beyond: float, entry: float, kinds=TARGET_POOLS) -> list[float]:
        """Untaken liquidity on the target side, past ``beyond``, nearest first."""
        pools = snap.pools[snap.pools["kind"].isin(kinds) & (snap.pools["side"] == ("high" if d == 1 else "low"))]
        return sorted({float(p) for p in pools["price"] if (p - beyond) * d > 0}, key=lambda p: (p - entry) * d)

    def grade(self, score: int) -> str:
        e = self.cfg.entry
        return "A+" if score >= e.a_plus_score else "A" if score >= e.min_score else \
            "B" if score >= e.b_grade_score else "-"

    def session_points(self, a: int) -> tuple[str, int]:
        kz = str(self.f.session[a])
        return kz, 10 if kz == self.cfg.entry.preferred_kill_zone else 5 if kz in KILL_ZONES else 0

    # ------------------------------------------------------------ evaluate
    def evaluate(self, j: int) -> Signal | None:
        """Signal for the M5 candle at row ``j`` (decided at its close), or None."""
        d = self.choch.get(j)
        if d is None:
            return None
        e, f = self.cfg.entry, self.f
        t = self.decision_time(j)
        if not self.session_gates(j, t):
            return None
        snap = f.snapshot(t)
        if snap.bias != (BULL if d == 1 else BEAR):
            return self._reject("2 HTF bias does not agree")
        if not (snap.m15_allows_long if d == 1 else snap.m15_allows_short):
            return self._reject("2 M15 opposes")

        # 3. sweep against the trade, still intact
        valid = self.intact_sweeps(snap, d, j)
        if not valid:
            return self._reject("3 no valid sweep in last 12 candles")
        sweep = sorted(valid, key=lambda s: (liq.TIER_RANK[s.tier], -s.sweep_idx))[0]
        a, extreme = sweep.sweep_idx, sweep.extreme

        # 4-5. displacement, FVG and entry
        got = self.leg_entry(a, j, d, t, extreme)
        if not got:
            return None
        entry, overlap, body_atr = got
        # 6-7. market and stop
        spread = self.market_ok(snap, j)
        if spread is None:
            return None
        st = self.stop_for(extreme, entry, d, j, spread)
        if not st:
            return None
        sl, risk = st
        # 8. targets
        tp1 = self.first_target(entry, risk, d, t)
        levels = self.pool_levels(snap, d, tp1, entry)
        if not levels:
            return self._reject("8 no liquidity target beyond TP1")
        tp2 = levels[0]
        if (tp2 - entry) * d < e.tp2_min_r * risk:
            return self._reject("8 TP2 closer than 2R")
        tp3 = next((p for p in levels[1:] if (p - entry) * d >= e.tp3_min_r * risk), None)

        kz, session_pts = self.session_points(a)
        factors = {
            "htf_alignment": snap.htf_points,
            "liquidity_swept": SWEEP_POINTS[sweep.tier],
            "sweep_rejection": 10 if sweep.same_candle else 5,
            "displacement": 15 if body_atr >= 2.0 else 10 if body_atr >= 1.5 else 5,
            "entry_confluence": 10 if overlap else 5,
            "premium_discount": 10 if snap.dealing_range is not None and (
                (d == 1 and entry < snap.dealing_range.equilibrium)
                or (d == -1 and entry > snap.dealing_range.equilibrium)) else 0,
            "session_fit": session_pts,
            "reward_to_tp2": 10 if (tp2 - entry) * d >= 3 * risk else 5,
        }
        score = int(sum(factors.values()))
        grade = self.grade(score)
        if grade == "-":
            return self._reject("9 score below 50")
        reasons = [
            f"H4/H1 bias {snap.bias}, M15 {snap.state[Timeframe.M15]}",
            f"Swept {sweep.pool_kind} {'low' if d == 1 else 'high'} {sweep.level:.2f} "
            f"({'same-candle rejection' if sweep.same_candle else 'reclaimed'})",
            f"M5 CHoCH with displacement ({body_atr:.1f}x ATR)",
            f"Entry {'FVG + order block' if overlap else 'FVG midpoint'} {entry:.2f} in the leg's "
            f"{'discount' if d == 1 else 'premium'}",
            f"TP2 at {tp2:.2f} ({(tp2 - entry) * d / risk:.1f}R)",
        ]
        return Signal(t, j, d, float(entry), float(sl), float(tp1), float(tp2), tp3, score, grade,
                      a, float(extreme), kz, spread, factors, reasons, self.name)
