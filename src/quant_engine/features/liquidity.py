"""Liquidity pools and sweeps (Strategy Spec v1, "Liquidity and sweeps").

A *pool* is a price where stops rest: previous trading-day high/low, Asia
range high/low, London high/low, unswept H1 swings, equal highs/lows and
other M5 swings. Each pool has the time it became known (``known_at``) and
the time it was first traded through (``taken_at``), computed from M5
candles. At decision time *t* a pool is live when ``known_at <= t`` and it
has not been taken by *t*.

A *sweep* of a sell-side pool (lows) for a long:
1. the first M5 candle to trade below the pool goes beyond it by at least
   one tick and at most ``sweep_max_depth_atr`` x ATR(14, M5);
2. that candle closes back above the level, or one of the next
   ``sweep_reclaim_candles - 1`` candles does;
3. there are never two consecutive closes below the level before the reclaim;
4. the sweep candle opens inside a kill zone.
Buy-side is the mirror. A sweep is known at the close of the reclaim candle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LiquidityConfig, SessionsConfig
from ..timeframes import Timeframe
from .sessions import NY, in_kill_zone, ny_minutes, trading_day

HTF, SESSION, ENGINEERED, MINOR = "HTF", "session", "engineered", "minor"
TIER_RANK = {HTF: 0, SESSION: 1, ENGINEERED: 2, MINOR: 3}
POOL_COLUMNS = ["kind", "tier", "side", "price", "known_at", "source_time", "taken_at", "taken_idx"]


@dataclass(frozen=True)
class Sweep:
    """A confirmed liquidity sweep on M5."""

    pool_kind: str
    tier: str
    side: str            # 'low' = sell-side pool swept (bullish idea), 'high' = buy-side
    level: float
    sweep_idx: int       # M5 row of the candle that traded through
    sweep_time: pd.Timestamp
    reclaim_idx: int
    known_at: pd.Timestamp
    extreme: float       # lowest low (or highest high) from sweep to reclaim
    depth: float
    same_candle: bool    # closed back inside on the sweep candle

    @property
    def direction(self) -> int:
        """+1 if the sweep supports a long (sell-side taken), -1 for a short."""
        return 1 if self.side == "low" else -1


# --------------------------------------------------------------------- pools
def _taken(m5: pd.DataFrame, side: str, price: float, after: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    """First M5 candle opening at/after ``after`` that trades through ``price``."""
    t = pd.DatetimeIndex(m5["time"])
    start = int(t.searchsorted(after, side="left"))
    vals = m5["low" if side == "low" else "high"].to_numpy(dtype="float64")
    for a in range(start, len(vals), 512):
        seg = vals[a:a + 512]
        hit = np.flatnonzero(seg < price) if side == "low" else np.flatnonzero(seg > price)
        if len(hit):
            k = a + int(hit[0])
            return t[k], k
    return pd.NaT, -1


def _ny_instant(day_ny: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """UTC instant of ``minutes`` after midnight (New York) on a New York calendar date."""
    wall = pd.Timestamp(day_ny).normalize() + pd.Timedelta(minutes=minutes)
    return wall.tz_localize(NY, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def session_pools(m5: pd.DataFrame, cfg: SessionsConfig) -> list[tuple]:
    """Previous-day, Asia and London highs/lows built from M5 candles.

    A pool is only created once its window has fully ended *within the data*
    (the last candle's close is at or after the window end), and its
    ``known_at`` is the window end, so the result never depends on candles
    after the decision time.
    """
    if m5.empty:
        return []
    t = pd.DatetimeIndex(m5["time"])
    data_end = t[-1] + Timeframe.M5.delta
    df = pd.DataFrame({"td": trading_day(t, cfg), "m": ny_minutes(t), "high": m5["high"].to_numpy(),
                       "low": m5["low"].to_numpy(), "time": t})
    rows = []
    for day, d in df.groupby("td", sort=True):
        # The trading day labelled D ends at 17:00 New York on date D.
        day_end = _ny_instant(day, cfg.trading_day_start)
        if day_end <= data_end:
            rows.append(("previous_day", HTF, "high", d["high"].max(), day_end, d["time"].iat[0]))
            rows.append(("previous_day", HTF, "low", d["low"].min(), day_end, d["time"].iat[0]))
        for name, w in (("asia", cfg.asia), ("london", cfg.london_kz)):
            s = d[(d["m"] >= w.start) & (d["m"] < w.end)] if w.start < w.end else d.iloc[0:0]
            if not len(s):
                continue
            first_ny = s["time"].iat[0].tz_convert(NY)
            end = _ny_instant(first_ny.tz_localize(None).normalize(), w.end)
            if end <= data_end:
                rows.append((name, SESSION, "high", s["high"].max(), end, s["time"].iat[0]))
                rows.append((name, SESSION, "low", s["low"].min(), end, s["time"].iat[0]))
    return rows


def swing_pools(swings: pd.DataFrame, kind: str, tier: str) -> list[tuple]:
    """Turn confirmed swings into pools (known when the swing is confirmed)."""
    return [(kind, tier, r.kind, float(r.price), r.known_at, r.time) for r in swings.itertuples(index=False)]


def equal_level_pools(swings: pd.DataFrame, tolerance: pd.Series | float, max_gap: pd.Timedelta,
                      kind: str = "equal") -> list[tuple]:
    """Equal highs/lows: two confirmed swings of the same side within ``tolerance``.

    ``tolerance`` may be a float or a Series aligned to the swing rows (e.g.
    0.1 x ATR(14, M15) known when the second swing was confirmed). The pool
    sits at the more extreme of the two and is known when the later swing is.
    """
    rows = []
    tol = (tolerance if isinstance(tolerance, pd.Series)
           else pd.Series(float(tolerance), index=swings.index))
    min_sep = pd.Timedelta(minutes=15)
    for side in ("high", "low"):
        # The same extreme can be a swing on M5 and on M15: keep it once.
        s = swings[swings["kind"] == side].drop_duplicates("price", keep="first")
        recs = list(s.itertuples())
        for b in range(1, len(recs)):
            rb = recs[b]
            for a in range(b - 1, -1, -1):
                ra = recs[a]
                if rb.known_at - ra.known_at > max_gap:
                    break
                if abs(rb.time - ra.time) < min_sep:
                    continue  # the same price swing seen on two timeframes
                tb = tol.loc[rb.Index]
                if np.isnan(tb):
                    continue
                if abs(rb.price - ra.price) <= tb:
                    price = max(ra.price, rb.price) if side == "high" else min(ra.price, rb.price)
                    rows.append((kind, ENGINEERED, side, float(price), max(ra.known_at, rb.known_at), ra.time))
                    break
    return rows


def build_pools(m5: pd.DataFrame, rows: list[tuple]) -> pd.DataFrame:
    """Attach ``taken_at`` (first M5 candle trading through after the pool is known)."""
    out = []
    for kind, tier, side, price, known_at, source_time in rows:
        taken_at, k = _taken(m5, side, price, known_at)
        out.append((kind, tier, side, price, known_at, source_time, taken_at, k))
    df = pd.DataFrame(out, columns=POOL_COLUMNS)
    return df.sort_values("known_at", kind="stable").reset_index(drop=True)


def live_pools(pools: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """Pools known at ``when`` and not yet traded through by then."""
    known = pools["known_at"] <= when
    untaken = pools["taken_at"].isna() | (pools["taken_at"] + Timeframe.M5.delta > when)
    # A pool whose taking candle is still forming at ``when`` is treated as live.
    return pools[known & untaken]


# -------------------------------------------------------------------- sweeps
def detect_sweeps(m5: pd.DataFrame, pools: pd.DataFrame, atr_m5: pd.Series, tick: float,
                  liq: LiquidityConfig, sessions: SessionsConfig) -> list[Sweep]:
    """All sweeps of ``pools`` on M5, each known at its reclaim candle's close."""
    if m5.empty or pools.empty:
        return []
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    a = atr_m5.to_numpy(dtype="float64")
    kz = in_kill_zone(t, sessions)
    out: list[Sweep] = []
    for p in pools.itertuples(index=False):
        k = int(p.taken_idx)
        if k < 0 or not kz[k] or np.isnan(a[k]):
            continue
        low_side = p.side == "low"
        depth = (p.price - l[k]) if low_side else (h[k] - p.price)
        if depth < tick - 1e-12 or depth > liq.sweep_max_depth_atr * a[k]:
            continue
        reclaim = None
        prev_beyond = False
        for j in range(k, min(k + liq.sweep_reclaim_candles, len(c))):
            inside = c[j] > p.price if low_side else c[j] < p.price
            if inside:
                reclaim = j
                break
            if prev_beyond:      # two consecutive closes beyond = breakout
                break
            prev_beyond = True
        if reclaim is None:
            continue
        seg = slice(k, reclaim + 1)
        extreme = float(l[seg].min()) if low_side else float(h[seg].max())
        out.append(Sweep(p.kind, p.tier, p.side, float(p.price), k, t[k], reclaim,
                         t[reclaim] + Timeframe.M5.delta, extreme, float(depth), reclaim == k))
    # One sweep per candle and level: keep the highest-tier pool name.
    best: dict[tuple, Sweep] = {}
    for sw in out:
        key = (sw.side, sw.sweep_idx, round(sw.level, 8))
        if key not in best or TIER_RANK[sw.tier] < TIER_RANK[best[key].tier]:
            best[key] = sw
    return sorted(best.values(), key=lambda s: (s.known_at, s.sweep_idx))
