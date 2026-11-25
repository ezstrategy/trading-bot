"""Three regime strategies and the switch between them.

Gold behaves differently in each kind of market, so each mode uses its own
setup, trigger, entry, stop and exits (all on closed candles, no look-ahead):

* **Bullish: "buy the dip in an uptrend".** Uptrends in gold grind higher in
  steps with shallow, orderly pullbacks that hold above the last higher low and
  find buyers at dynamic support. Setup: pullback to the M15 EMA20/EMA50 zone
  or the H1 discount, higher low intact. Trigger: M5 break of structure UP
  (continuation, not reversal) with displacement and an FVG. Entry: FVG
  midpoint. Stop: below the pullback low. Exits: TP1 1R (50%), TP2 at the next
  buy-side liquidity or a 2R measured move, TP3 runner at 3R; a longer time
  stop because trends need time.

* **Bearish: "sell the corrective rally".** Sell-offs in gold are fast
  liquidations; counter-trend rallies are weak and usually run the buy stops
  above a recent high before rolling over. Setup: rally into the premium half
  of the H1 range that raids an M5 high (inducement). Trigger: M5 change of
  character DOWN with displacement. Entry: FVG midpoint (or OB overlap) in the
  rally's premium. Stop: above the raid high. Exits: TP1 1R (50%), TP2 must be
  real sell-side liquidity (previous day / session / H1 low) at >= 2R, no
  runner, short time stop because drops snap back quickly.

* **Sideways: "fade the false breakout".** Ranges hold their edges; breakouts
  without expansion fail and return to the middle. Setup: H4 neutral, an H1
  range 2-8 x ATR(H1) wide, volatility not expanding. Trigger: price sweeps a
  range edge and closes back inside with a rejection (no displacement needed;
  ranges reverse quietly). Entry: limit at the swept edge (the retest). Stop:
  beyond the sweep extreme. Exits: TP1 at the range middle (50%), TP2 just
  before the opposite edge, no runner, short time stop.

The switch (:class:`RegimeModel`) asks the H4 trend-regime detector
(:mod:`.regime`) which market it is at the candle's close and uses that mode
only. Each mode can also run alone on all markets (``require_regime=False``)
to check that it really does better in its own market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..features import liquidity as liq
from ..features.context import MarketFeatures, _last_closed_idx
from ..features.news import NewsCalendar
from ..features.sessions import KILL_ZONES
from ..features.structure import BEARISH as S_BEAR
from ..features.structure import BULLISH as S_BULL
from ..features.structure import RANGING as S_RANGE
from ..timeframes import Timeframe
from .entry import SWEEP_POINTS, EntryModel, Signal
from .regime import BEARISH, BULLISH, NEUTRAL, regime_at, trend_regime_series

SELL_SIDE_TARGETS = ("previous_day", "asia", "london", "h1_swing")
BUY_SIDE_TARGETS = ("previous_day", "asia", "london", "h1_swing")


def _ema(x: pd.Series, n: int) -> np.ndarray:
    return x.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


class _ModeBase(EntryModel):
    """Shared plumbing: regime lookup, M15 EMAs, per-mode config."""

    regime_name = ""

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig, news: NewsCalendar | None = None,
                 require_regime: bool = True) -> None:
        super().__init__(features, cfg, news)
        self.require_regime = require_regime
        self.mc = cfg.modes
        h4 = features.bars[Timeframe.H4]
        b = cfg.backtest
        self.h4 = h4
        self.regimes = trend_regime_series(h4, b.regime_ema_period, b.regime_slope_bars, b.regime_min_slope_atr)
        m15 = features.bars[Timeframe.M15]
        self.m15_t = features.times[Timeframe.M15]
        self.ema_fast = _ema(m15["close"].astype("float64"), self.mc.ema_fast)
        self.ema_slow = _ema(m15["close"].astype("float64"), self.mc.ema_slow)
        self.atr_m15 = features.atr[Timeframe.M15].to_numpy(dtype="float64")
        self.atr_h1 = features.atr[Timeframe.H1].to_numpy(dtype="float64")
        self.h1_t = features.times[Timeframe.H1]
        ev = features.structure[Timeframe.M5].events
        self.events = ev
        self.by_sweep_idx: dict[int, list] = {}
        self.by_reclaim_idx: dict[int, list] = {}
        for s in features.sweeps:
            self.by_sweep_idx.setdefault(int(s.sweep_idx), []).append(s)
            self.by_reclaim_idx.setdefault(int(s.reclaim_idx), []).append(s)

    def sweeps_between(self, lo: int, hi: int, t: pd.Timestamp) -> list:
        """Sweeps whose sweep candle is in [lo, hi] and that are known at ``t``."""
        return [s for k in range(max(lo, 0), hi + 1) for s in self.by_sweep_idx.get(k, ()) if s.known_at <= t]

    def regime(self, t: pd.Timestamp) -> str:
        return regime_at(self.h4, self.regimes, t)

    def regime_gate(self, t: pd.Timestamp) -> str | None:
        r = self.regime(t)
        if self.require_regime and r != self.regime_name:
            return self._reject(f"0 market is {r}, not {self.regime_name}")
        return r

    def m15_ema(self, t: pd.Timestamp) -> tuple[float, float, float, float]:
        """(EMA fast, EMA slow, EMA fast 3 candles earlier, ATR M15) of the last closed M15 candle."""
        k = _last_closed_idx(self.m15_t, Timeframe.M15, t)
        if k < 3:
            return (np.nan,) * 4
        return self.ema_fast[k], self.ema_slow[k], self.ema_fast[k - 3], self.atr_m15[k]

    def h1_atr(self, t: pd.Timestamp) -> float:
        k = _last_closed_idx(self.h1_t, Timeframe.H1, t)
        return float(self.atr_h1[k]) if k >= 0 else np.nan

    def last_swing(self, tf: Timeframe, kind: str, t: pd.Timestamp, min_idx: int = -1):
        sw = self.f.structure[tf].swings
        sw = sw[(sw["known_at"] <= t) & (sw["kind"] == kind) & (sw["idx"] > min_idx)]
        return sw.iloc[-1] if len(sw) else None

    def finish(self, t, j, d, entry, sl, tp1, tp2, tp3, factors, reasons, anchor_idx, extreme, spread,
               time_stop) -> Signal | None:
        score = int(sum(factors.values()))
        grade = self.grade(score)
        if grade == "-":
            return self._reject("9 score below 50")
        kz = str(self.f.session[anchor_idx])
        kz = kz if kz in KILL_ZONES else str(self.f.session[j])
        return Signal(t, j, d, float(entry), float(sl), float(tp1), float(tp2), tp3, score, grade, anchor_idx,
                      float(extreme), kz, spread, factors, reasons, self.name, time_stop)


# =============================================================== bullish
class BullishTrendModel(_ModeBase):
    """Buy the dip in an uptrend (see the module docstring)."""

    name, regime_name = "bullish", BULLISH

    def candidates(self) -> list[int]:
        ev = self.events
        return sorted(int(i) for i in ev[(ev["kind"] == "BOS") & (ev["direction"] == 1)]["idx"])

    def evaluate(self, j: int) -> Signal | None:
        ev = self.events
        if not ((ev["idx"] == j) & (ev["kind"] == "BOS") & (ev["direction"] == 1)).any():
            return None
        c, d = self.mc.bullish, 1
        t = self.decision_time(j)
        if not self.session_gates(j, t):
            return None
        r = self.regime_gate(t)
        if r is None:
            return None
        snap = self.f.snapshot(t)
        h4s, h1s = snap.state[Timeframe.H4], snap.state[Timeframe.H1]
        if S_BEAR in (h4s, h1s):
            return self._reject("2 H4/H1 structure bearish")
        ef, es, ef_prev, atr15 = self.m15_ema(t)
        if not (ef > es):
            return self._reject("2 M15 EMA20 not above EMA50")

        # 3. pullback: last confirmed M5 swing low in the recent window, higher low intact
        pl = self.last_swing(Timeframe.M5, "low", t, j - c.pullback_window)
        if pl is None:
            return self._reject("3 no pullback low")
        a = int(pl["idx"])
        prior_high = self.h[max(0, a - c.pullback_window):a + 1].max()
        depth_atr = (prior_high - pl["price"]) / self.atr[j]
        if depth_atr < c.min_pullback_atr:
            return self._reject("3 pullback too shallow")
        hl = self.last_swing(Timeframe.M15, "low", t)
        if hl is not None and pl["price"] < hl["price"]:
            return self._reject("3 pullback broke the M15 higher low")
        at_ema = pl["price"] <= ef + c.ema_touch_atr * atr15
        in_discount = snap.dealing_range is not None and pl["price"] < snap.dealing_range.equilibrium
        if not (at_ema or in_discount):
            return self._reject("3 pullback did not reach EMA zone or H1 discount")
        extreme = float(self.l[a:j + 1].min())

        # 4-5. continuation break with displacement and FVG
        got = self.leg_entry(a, j, d, t, extreme)
        if not got:
            return None
        entry, overlap, body_atr = got
        spread = self.market_ok(snap, j)
        if spread is None:
            return None
        st = self.stop_for(extreme, entry, d, j, spread)
        if not st:
            return None
        sl, risk = st
        tp1 = self.first_target(entry, risk, d, t)
        levels = self.pool_levels(snap, d, tp1, entry, BUY_SIDE_TARGETS)
        measured = entry + c.tp2_r * risk
        tp2 = levels[0] if levels and (levels[0] - entry) >= c.tp2_r * risk else measured
        tp3 = entry + c.tp3_r * risk if c.tp3_r and c.tp3_r * risk > tp2 - entry else None
        swept = any(s.direction == 1 for s in self.sweeps_between(a - 3, a + 3, t))

        factors = {
            "trend_strength": 20 if (h4s == S_BULL and h1s == S_BULL) else 10,
            "momentum": 10 if ef > ef_prev else 5,
            "pullback_location": 15 if (at_ema and in_discount) else 10,
            "pullback_liquidity": 10 if swept else 0,
            "displacement": 15 if body_atr >= 2.0 else 10 if body_atr >= 1.5 else 5,
            "entry_confluence": 10 if overlap else 5,
            "session_fit": self.session_points(j)[1],
            "reward_to_tp2": 10 if levels and (levels[0] - entry) >= c.tp2_r * risk else 5,
        }
        reasons = [
            f"Uptrend: H4 regime {r}, H4 {h4s} / H1 {h1s}, M15 EMA{self.mc.ema_fast} > EMA{self.mc.ema_slow}",
            f"Pullback {depth_atr:.1f}x ATR to {pl['price']:.2f}"
            + (" at the M15 EMA zone" if at_ema else "") + (" in H1 discount" if in_discount else "")
            + (", took sell stops" if swept else ""),
            f"M5 break of structure up with displacement ({body_atr:.1f}x ATR)",
            f"Buy {'FVG + order block' if overlap else 'FVG midpoint'} {entry:.2f}, stop under the pullback low",
            f"TP2 {tp2:.2f} ({(tp2 - entry) / risk:.1f}R)" + (f", runner TP3 {tp3:.2f}" if tp3 else ""),
        ]
        return self.finish(t, j, d, entry, sl, tp1, tp2, tp3, factors, reasons, a, extreme, spread,
                           c.time_stop_candles)


# =============================================================== bearish
class BearishRallyModel(_ModeBase):
    """Sell the corrective rally in a downtrend (see the module docstring)."""

    name, regime_name = "bearish", BEARISH

    def candidates(self) -> list[int]:
        return sorted(j for j, d in self.choch.items() if d == -1)

    def evaluate(self, j: int) -> Signal | None:
        if self.choch.get(j) != -1:
            return None
        c, d = self.mc.bearish, -1
        t = self.decision_time(j)
        if not self.session_gates(j, t):
            return None
        r = self.regime_gate(t)
        if r is None:
            return None
        snap = self.f.snapshot(t)
        h4s, h1s = snap.state[Timeframe.H4], snap.state[Timeframe.H1]
        if h1s != S_BEAR or h4s == S_BULL:
            return self._reject("2 H1 not bearish (or H4 bullish)")
        if not snap.m15_allows_short:
            return self._reject("2 M15 opposes")
        dr = snap.dealing_range
        if dr is None:
            return self._reject("3 no H1 range")

        # 3. the rally raided buy stops (an M5 high or better) in the premium half
        raids = [s for s in self.sweeps_between(j - c.rally_window, j, t) if s.direction == -1
                 and not (self.c[s.reclaim_idx + 1:j + 1] > s.extreme).any()]
        raids = [s for s in raids if s.extreme > dr.equilibrium]
        if not raids:
            return self._reject("3 no buy-stop raid in the H1 premium")
        raid = sorted(raids, key=lambda s: (liq.TIER_RANK[s.tier], -s.sweep_idx))[0]
        a, extreme = raid.sweep_idx, float(raid.extreme)
        depth = (extreme - dr.low) / max(dr.high - dr.low, 1e-9)

        got = self.leg_entry(a, j, d, t, extreme)
        if not got:
            return None
        entry, overlap, body_atr = got
        spread = self.market_ok(snap, j)
        if spread is None:
            return None
        st = self.stop_for(extreme, entry, d, j, spread)
        if not st:
            return None
        sl, risk = st
        tp1 = self.first_target(entry, risk, d, t)
        levels = self.pool_levels(snap, d, tp1, entry, SELL_SIDE_TARGETS)
        if not levels:
            return self._reject("8 no sell-side liquidity below TP1")
        tp2 = levels[0]
        if (entry - tp2) < c.tp2_min_r * risk:
            return self._reject("8 sell-side target closer than 2R")

        factors = {
            "trend_strength": 20 if h4s == S_BEAR else 10,
            "rally_location": 15 if depth >= 0.75 else 10,
            "buy_stops_raided": SWEEP_POINTS[raid.tier],
            "raid_rejection": 10 if raid.same_candle else 5,
            "displacement": 15 if body_atr >= 2.0 else 10 if body_atr >= 1.5 else 5,
            "entry_confluence": 5 if overlap else 0,
            "session_fit": self.session_points(a)[1],
            "reward_to_tp2": 10 if (entry - tp2) >= 3 * risk else 5,
        }
        reasons = [
            f"Downtrend: H4 regime {r}, H4 {h4s} / H1 {h1s}",
            f"Rally into H1 premium ({depth:.0%} of range) raided buy stops at {raid.level:.2f} ({raid.pool_kind})",
            f"M5 change of character down with displacement ({body_atr:.1f}x ATR)",
            f"Sell {'FVG + order block' if overlap else 'FVG midpoint'} {entry:.2f}, stop above the raid high",
            f"TP2 at sell-side liquidity {tp2:.2f} ({(entry - tp2) / risk:.1f}R), no runner",
        ]
        return self.finish(t, j, d, entry, sl, tp1, tp2, None, factors, reasons, a, extreme, spread,
                           c.time_stop_candles)


# ============================================================== sideways
class RangeFadeModel(_ModeBase):
    """Fade false breakouts of an H1 range (see the module docstring)."""

    name, regime_name = "sideways", NEUTRAL

    def candidates(self) -> list[int]:
        return sorted(self.by_reclaim_idx)

    def evaluate(self, j: int) -> Signal | None:
        sweeps = self.by_reclaim_idx.get(j, [])
        if not sweeps:
            return None
        c = self.mc.sideways
        t = self.decision_time(j)
        if not self.session_gates(j, t):
            return None
        r = self.regime_gate(t)
        if r is None:
            return None
        snap = self.f.snapshot(t)
        dr = snap.dealing_range
        atr_h1 = self.h1_atr(t)
        if dr is None or np.isnan(atr_h1):
            return self._reject("2 no H1 range")
        width = dr.high - dr.low
        if not (c.min_range_atr_h1 * atr_h1 <= width <= c.max_range_atr_h1 * atr_h1):
            return self._reject("2 H1 range too narrow or too wide")
        if not np.isnan(snap.vol_ratio) and snap.vol_ratio > c.max_vol_ratio:
            return self._reject("2 volatility expanding (breakout risk)")

        # 3. the sweep must be at a range edge
        edge = c.edge_zone * width
        best = None
        for s in sorted(sweeps, key=lambda s: liq.TIER_RANK[s.tier]):
            if (s.direction == 1 and abs(s.level - dr.low) <= edge) or \
                    (s.direction == -1 and abs(s.level - dr.high) <= edge):
                best = s
                break
        if best is None:
            return self._reject("3 sweep not at a range edge")
        d, a, extreme = best.direction, best.sweep_idx, float(best.extreme)

        # 4. rejection: closed back inside with a long wick or same candle
        k = a
        rng = max(self.h[k] - self.l[k], 1e-9)
        wick = (min(self.o[k], self.c[k]) - self.l[k]) / rng if d == 1 else (self.h[k] - max(self.o[k], self.c[k])) / rng
        if not (best.same_candle or wick >= c.min_wick):
            return self._reject("4 no rejection at the edge")

        # 5. entry: limit at the swept edge (retest)
        entry = float(best.level)
        if (self.c[j] - entry) * d <= 0 or not (dr.low < self.c[j] < dr.high):
            return self._reject("5 price is not back inside the range")
        spread = self.market_ok(snap, j)
        if spread is None:
            return None
        st = self.stop_for(extreme, entry, d, j, spread, widen_to_atr=c.min_stop_atr)
        if not st:
            return None
        sl, risk = st
        mid = dr.equilibrium
        if (mid - entry) * d < self.cfg.entry.tp1_min_r * risk:
            return self._reject("8 range middle closer than 1R")
        # TP1 banks half and moves the stop to breakeven; a far range middle (often 5R+ from a
        # tight edge stop) is rarely reached first, so it can be capped at tp1_max_r.
        tp1 = entry + d * min((mid - entry) * d, c.tp1_max_r * risk) if c.tp1_max_r > 0 else mid
        far = dr.high if d == 1 else dr.low
        tp2 = far - d * c.opposite_edge_buffer * width
        rr2 = (tp2 - entry) * d / risk
        if rr2 < c.tp2_min_r:
            return self._reject("8 opposite edge closer than the minimum R")
        h4s, h1s = snap.state[Timeframe.H4], snap.state[Timeframe.H1]
        at_true_edge = (d == 1 and extreme < dr.low) or (d == -1 and extreme > dr.high)
        factors = {
            "range_quality": 20 if (h4s == S_RANGE and h1s == S_RANGE) else 10 if S_RANGE in (h4s, h1s) else 5,
            "edge_precision": 15 if at_true_edge else 10,
            "liquidity_swept": SWEEP_POINTS[best.tier],
            "rejection": 15 if best.same_candle else 8,
            "compression": 10 if (np.isnan(snap.vol_ratio) or snap.vol_ratio <= 1.0) else 5,
            "session_fit": self.session_points(a)[1],
            "reward_to_tp2": 15 if rr2 >= 2.5 else 10 if rr2 >= 2.0 else 5,
        }
        reasons = [
            f"Sideways: H4 regime {r}, H1 range {dr.low:.2f}-{dr.high:.2f} ({width / atr_h1:.1f}x H1 ATR)",
            f"False breakout of the range {'low' if d == 1 else 'high'}: swept {best.pool_kind} {best.level:.2f}, "
            f"{'closed back inside on the same candle' if best.same_candle else 'rejected with a long wick'}",
            f"{'Buy' if d == 1 else 'Sell'} the retest of the edge {entry:.2f}, stop beyond the sweep",
            (f"TP1 {tp1:.2f} (range middle)" if tp1 == mid else f"TP1 {tp1:.2f} ({c.tp1_max_r:g}R, before the middle {mid:.2f})")
            + f", TP2 before the opposite edge {tp2:.2f} ({rr2:.1f}R)",
        ]
        return self.finish(t, j, d, entry, sl, tp1, tp2, None, factors, reasons, a, extreme, spread,
                           c.time_stop_candles)


# =============================================================== switch
class RegimeModel:
    """Picks the mode for the market at each candle: bullish / bearish / sideways."""

    name = "regime"

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig, news: NewsCalendar | None = None) -> None:
        self.models = {BULLISH: BullishTrendModel(features, cfg, news, require_regime=False),
                       BEARISH: BearishRallyModel(features, cfg, news, require_regime=False),
                       NEUTRAL: RangeFadeModel(features, cfg, news, require_regime=False)}
        self.router = self.models[BULLISH]
        self.f = features
        self._cand: dict[int, set] = {}
        for reg, m in self.models.items():
            for j in m.candidates():
                self._cand.setdefault(j, set()).add(reg)
        self._skipped: dict[str, int] = {}

    @property
    def rejections(self) -> dict[str, int]:
        out = dict(self._skipped)
        for reg, m in self.models.items():
            for k, v in m.rejections.items():
                out[f"{m.name}: {k}"] = out.get(f"{m.name}: {k}", 0) + v
        return out

    def candidates(self) -> list[int]:
        return sorted(self._cand)

    def evaluate(self, j: int) -> Signal | None:
        t = self.router.decision_time(j)
        reg = self.router.regime(t)
        if reg not in self._cand.get(j, ()):
            key = f"0 {reg} market: no {self.models[reg].name} setup candle"
            self._skipped[key] = self._skipped.get(key, 0) + 1
            return None
        return self.models[reg].evaluate(j)


MODES = ("v1", "regime", "bullish", "bearish", "sideways")


def build_model(mode: str, features: MarketFeatures, cfg: StrategyConfig, news: NewsCalendar | None = None,
                require_regime: bool = False):
    """Entry model for ``mode``. Single modes run on every market unless ``require_regime``."""
    if mode == "v1":
        return EntryModel(features, cfg, news)
    if mode == "regime":
        return RegimeModel(features, cfg, news)
    cls = {"bullish": BullishTrendModel, "bearish": BearishRallyModel, "sideways": RangeFadeModel}[mode]
    return cls(features, cfg, news, require_regime=require_regime)
