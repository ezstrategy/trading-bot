"""GTC-SBR (Gold Trend Continuation, Session Breakout and Retest) as two plug-ins.

``gtc_sbr_bull`` trades only upside breakouts and ``gtc_sbr_bear`` only downside ones. Both use
the owner's rules as coded in :mod:`quant_engine.research.gtc_sbr` (version 1.0.0 reproduces that
code; differences in how the engine handles data gaps and overlapping trades are listed in
``docs/QUANT_LAB.md``):

* Opening range: the first ``range_minutes`` after 08:00 London time and 08:20 New York time
  (daylight saving from the real time zones). It must be complete and contiguous.
* Breakout: the first completed M5 candle within ``breakout_window_bars`` that closes at least
  ``breakout_atr`` x ATR(14, M5) beyond the range. Only the session's first breakout counts.
* Trend (M15, completed candles only): EMA20 > EMA50, close > EMA20 and both EMAs higher than
  ``slope_bars`` M15 candles ago for bullish; the mirror for bearish.
* Retest: candle 3-8 after the breakout comes within ``retest_tol_atr`` x ATR of the broken
  boundary; confirmation = that candle if it closes beyond the boundary, else the next one.
* Filters: news windows (no historical calendar: fixed New York release slots), spread above
  ``spread_mult`` x its past median for that half-hour.
* Entry: market at the next candle's open (engine). Stop: beyond the retest-to-confirmation
  candles + ``sl_buffer_atr`` x ATR; stop distance must be within ``min_risk_atr``-``max_risk_atr``
  x ATR at the fill. Target ``target_r`` x the filled risk. Time exit after ``time_exit_bars``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from ..backtest.market import MarketData, MarketView
from ..features.atr import atr as atr_fn
from ..research.gtc_sbr import m15_trend, past_spread_median
from ..timeframes import Timeframe
from .base import OrderIntent, Strategy, StrategyMeta, StrategyParams
from .registry import register


class SessionSpec(StrategyParams):
    """One opening-range session."""
    name: str
    tz: str
    open_minute: int = Field(ge=0, lt=1440)


class GtcSbrParams(StrategyParams):
    """GTC-SBR parameters (defaults = the owner's spec; GUESS where the spec left a choice)."""
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    slope_bars: int = 3                     # GUESS
    trend_filter: bool = True
    sessions: list[SessionSpec] = Field(default_factory=lambda: [
        SessionSpec(name="london", tz="Europe/London", open_minute=8 * 60),
        SessionSpec(name="new_york", tz="America/New_York", open_minute=8 * 60 + 20)])
    range_minutes: int = 30
    breakout_atr: float = 0.1
    breakout_window_bars: int = 24          # GUESS
    retest_min: int = 3                     # GUESS
    retest_max: int = 8
    retest_tol_atr: float = 0.15
    sl_buffer_atr: float = 0.2
    min_risk_atr: float = 0.3               # GUESS
    max_risk_atr: float = 3.0               # GUESS
    target_r: float = 1.5
    time_exit_bars: int = 12
    news_ny_minutes: list[int] = Field(default_factory=lambda: [8 * 60 + 30, 10 * 60])   # GUESS: fixed slots
    news_before: int = 15
    news_after: int = 30
    spread_mult: float = 1.5
    spread_bucket_min: int = 30
    spread_window: int = 360
    spread_min_obs: int = 20
    assumed_spread_points: float = 8.0      # floor for the strategy's own ask estimate (owner: $0.08)
    regime_filter: list[str] | None = None  # e.g. ["bullish"]; None = as specified (M15 trend only)


_RULES = dict(
    family="gtc_sbr",
    entry_rules="Session opening-range breakout (close >= 0.1 ATR beyond), M15 EMA20/50 trend agrees, "
                "retest of the boundary on candle 3-8 after the breakout, confirmation close beyond it; "
                "market entry at the next M5 open.",
    exit_rules="Stop, target or time exit after 12 M5 candles; the stop is assumed first when both are inside one candle.",
    stop_rules="Beyond the retest-to-confirmation candles + 0.2 ATR; distance must be 0.3-3 ATR.",
    target_rules="1.5 x the filled risk.",
    session_rules="London (08:00 London + 30 min range) and New York (08:20 New York + 30 min range); "
                  "no entries 15 min before to 30 min after 08:30 and 10:00 New York.",
    regime_rules="M15 EMA trend must agree with the breakout; H4 regime recorded, not used unless regime_filter is set.",
)


class _GtcSbr(Strategy):
    Params = GtcSbrParams
    side: int = 0

    def allowed_regimes(self) -> set[str] | None:
        return set(self.params.regime_filter) if self.params.regime_filter else None

    def prepare(self, data: MarketData) -> None:
        p = self.params
        b = data.base
        t, o, h, l, c = b.t, b.o, b.h, b.l, b.c
        n = len(b)
        pd_ = {"ema_fast": p.ema_fast, "ema_slow": p.ema_slow, "slope_bars": p.slope_bars,
               "spread_bucket_min": p.spread_bucket_min, "spread_window": p.spread_window,
               "spread_min_obs": p.spread_min_obs}
        a = atr_fn(b.df, p.atr_period).to_numpy(float)
        raw_sp = b.spread_pts
        sp = np.maximum(np.nan_to_num(raw_sp, nan=p.assumed_spread_points), p.assumed_spread_points) * data.point
        med = past_spread_median(b.df, pd_)
        if data.has(Timeframe.M15):
            m15 = data.frame(Timeframe.M15)
            trend = m15_trend(m15.df, pd_)
            closes15 = m15.close_times
        else:
            trend, closes15 = np.array([], dtype=object), pd.DatetimeIndex([], tz="UTC")
        decision_ny = (t + Timeframe.M5.delta).tz_convert("America/New_York")
        ny_min = np.asarray(decision_ny.hour * 60 + decision_ny.minute)
        weekday = np.asarray(decision_ny.dayofweek) < 5
        step = Timeframe.M5.delta
        self.setups: dict[int, OrderIntent] = {}
        self.skips = {}
        events = []
        for sess in p.sessions:
            loc = t.tz_convert(sess.tz)
            lmin = np.asarray(loc.hour * 60 + loc.minute)
            ldate = np.asarray(loc.normalize().tz_localize(None))
            in_rng = (lmin >= sess.open_minute) & (lmin < sess.open_minute + p.range_minutes)
            for d in np.unique(ldate[in_rng]):
                ridx = np.nonzero(in_rng & (ldate == d))[0]
                events.append((t[ridx[0]], sess.name, ridx))
        events.sort(key=lambda e: e[0])
        nbars_rng = p.range_minutes // 5
        want = "bullish" if self.side == 1 else "bearish"
        for _, name, ridx in events:
            if len(ridx) != nbars_rng or t[ridx[-1]] - t[ridx[0]] != (nbars_rng - 1) * step:
                self.skip("no complete range")
                continue
            hi, lo = h[ridx].max(), l[ridx].min()
            start = ridx[-1] + 1
            brk, side = None, 0
            for i in range(start, min(start + p.breakout_window_bars, n)):
                if t[i] - t[ridx[-1]] != (i - ridx[-1]) * step or not a[i] > 0:
                    break
                if c[i] >= hi + p.breakout_atr * a[i]:
                    brk, side = i, 1
                    break
                if c[i] <= lo - p.breakout_atr * a[i]:
                    brk, side = i, -1
                    break
            if brk is None:
                self.skip("no breakout")
                continue
            if side != self.side:
                self.skip("first breakout in the other direction")
                continue
            k15 = int(closes15.searchsorted(t[brk] + step, side="right")) - 1
            tr = trend[k15] if k15 >= 0 else "sideways"
            if p.trend_filter and tr != want:
                self.skip("M15 trend does not agree")
                continue
            bound = hi if side == 1 else lo
            tol = p.retest_tol_atr * a[brk]
            k = None
            for i in range(brk + p.retest_min, min(brk + p.retest_max, n - 1) + 1):
                if (side == 1 and l[i] <= bound + tol) or (side == -1 and h[i] + sp[i] >= bound - tol):
                    k = i
                    break
            if k is None:
                self.skip("no retest")
                continue
            conf = k if (c[k] - bound) * side > 0 else (k + 1 if k + 1 < n and (c[k + 1] - bound) * side > 0 else None)
            if conf is None:
                self.skip("no confirmation")
                continue
            if weekday[conf] and any(-p.news_before <= ny_min[conf] - m <= p.news_after for m in p.news_ny_minutes):
                self.skip("news window")
                continue
            if np.isfinite(med[conf]) and np.isfinite(raw_sp[conf]) and raw_sp[conf] > p.spread_mult * med[conf]:
                self.skip("spread too wide")
                continue
            if side == 1:
                stop = l[k:conf + 1].min() - p.sl_buffer_atr * a[conf]
            else:
                stop = h[k:conf + 1].max() + sp[conf] + p.sl_buffer_atr * a[conf]
            self.setups[conf] = OrderIntent(
                side=side, stop=float(stop), target_r=p.target_r, time_exit_bars=p.time_exit_bars,
                risk_bounds=(p.min_risk_atr * a[conf], p.max_risk_atr * a[conf]), session=name,
                reason={"range_high": round(float(hi), 2), "range_low": round(float(lo), 2),
                        "breakout_time": str(t[brk]), "retest_time": str(t[k]), "trend_m15": str(tr),
                        "atr": round(float(a[conf]), 4)})

    def on_bar(self, view: MarketView) -> list[OrderIntent]:
        s = self.setups.get(view.i)
        if s is None:
            return []
        return [OrderIntent(**{**s.__dict__, "reason": dict(s.reason)})]


@register
class GtcSbrBull(_GtcSbr):
    """GTC-SBR, bullish breakouts only."""
    side = 1
    meta = StrategyMeta(id="gtc_sbr_bull", version="1.0.0", name="GTC-SBR bullish", direction="long", **_RULES)


@register
class GtcSbrBear(_GtcSbr):
    """GTC-SBR, bearish breakouts only."""
    side = -1
    meta = StrategyMeta(id="gtc_sbr_bear", version="1.0.0", name="GTC-SBR bearish", direction="short", **_RULES)
