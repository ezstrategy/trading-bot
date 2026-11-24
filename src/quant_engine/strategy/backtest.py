"""Event-driven backtest of the entry model on closed M5 candles.

Execution model (conservative on purpose):
* Bars are BID prices. A buy limit fills when the ask (bid + spread) reaches
  the entry; buy stops/targets are hit on the bid. Sells mirror this, so the
  spread is paid on every trade.
* The order is live from the candle after the signal for ``expiry_candles``.
  It is cancelled if price reaches TP1 first or closes beyond the sweep extreme.
* On the fill candle only the stop is checked (a target on that candle is
  ignored). On later candles the stop is checked before any target.
* TP1 closes 50% and moves the stop to entry + spread. With a TP3, 25% closes
  at TP2 and 25% at TP3; otherwise the other 50% closes at TP2.
* Time stop: no TP1 and no stop within ``time_stop_candles`` after the fill ->
  close at that candle's close. Session end: anything open
  ``session_close_hours`` after the kill zone ends is closed.
* One signal at a time. Grade B setups are logged but not traded.
* Risk locks (in R, 1R = ``risk_pct``): daily loss lock, weekly loss lock,
  two losses in a row stops for the day, max two signals per kill zone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..config import StrategyConfig
from ..features.context import MarketFeatures
from ..features.sessions import NY, trading_day
from ..timeframes import Timeframe
from .entry import EntryModel, Signal
from .regime import regime_at, trend_regime_series


@dataclass
class Trade:
    signal_time: pd.Timestamp
    side: str
    grade: str
    score: int
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float | None
    kill_zone: str
    outcome: str                  # filled outcome or why the order never filled
    filled: bool
    fill_time: pd.Timestamp | None
    exit_time: pd.Timestamp | None
    r: float                      # result in R (0 when not filled)
    trend_regime: str
    h4_structure: str
    bias: str


def _kz_end(t: pd.Timestamp, kz: str, cfg: StrategyConfig) -> pd.Timestamp:
    w = cfg.sessions.london_kz if kz == "london_kz" else cfg.sessions.newyork_kz
    ny = t.tz_convert(NY)
    end = ny.normalize().tz_localize(None) + pd.Timedelta(minutes=w.end)
    end = end.tz_localize(NY, ambiguous=False, nonexistent="shift_forward")
    if end < ny - pd.Timedelta(hours=12):
        end += pd.Timedelta(days=1)
    return end.tz_convert("UTC")


class Backtester:
    """Runs the entry model over all stored history."""

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig) -> None:
        self.f, self.cfg = features, cfg
        self.model = EntryModel(features, cfg)
        m5 = features.bars[Timeframe.M5]
        self.t = features.times[Timeframe.M5]
        self.h, self.l, self.c = (m5[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
        h4 = features.bars[Timeframe.H4]
        b = cfg.backtest
        self.h4 = h4
        self.regimes = trend_regime_series(h4, b.regime_ema_period, b.regime_slope_bars, b.regime_min_slope_atr)
        self.day = trading_day(self.t, cfg.sessions)
        self.signals: list[Signal] = []

    # ---------------------------------------------------------- simulation
    def _simulate(self, s: Signal) -> tuple[str, bool, int | None, int, float]:
        """Returns (outcome, filled, fill_idx, exit_idx, R)."""
        e, d, n = self.cfg.entry, s.direction, len(self.c)
        sp = s.spread
        h, l, c = self.h, self.l, self.c
        j0 = s.m5_idx
        fill = None
        for k in range(j0 + 1, min(j0 + 1 + e.expiry_candles, n)):
            touched = (l[k] + sp <= s.entry) if d == 1 else (h[k] >= s.entry)
            if touched:
                fill = k
                break
            if (h[k] >= s.tp1) if d == 1 else (l[k] + sp <= s.tp1):
                return "cancelled: TP1 before fill", False, None, k, 0.0
            if (c[k] < s.sweep_extreme) if d == 1 else (c[k] > s.sweep_extreme):
                return "cancelled: closed beyond sweep", False, None, k, 0.0
        else:
            return "expired: not filled in 30 min", False, None, min(j0 + e.expiry_candles, n - 1), 0.0
        if fill is None:
            return "expired: not filled in 30 min", False, None, n - 1, 0.0

        risk = s.risk
        stop = s.stop_loss
        size = 1.0
        realised = 0.0
        session_end = _kz_end(s.time, s.kill_zone, self.cfg) + pd.Timedelta(hours=e.session_close_hours)
        targets = [(s.tp1, 0.5)] + ([(s.tp2, 0.25), (s.tp3, 0.25)] if s.tp3 is not None else [(s.tp2, 0.5)])

        def stop_hit(k):
            return (l[k] <= stop) if d == 1 else (h[k] + sp >= stop)

        def tp_hit(k, px):
            return (h[k] >= px) if d == 1 else (l[k] + sp <= px)

        k = fill
        if stop_hit(k):
            return "stop loss", True, fill, k, -1.0
        nxt = 0                                   # index of the next target to reach
        while True:
            k += 1
            if k >= n:
                px = c[n - 1] + (sp if d == -1 else 0)
                return "open at end of data", True, fill, n - 1, realised + size * s.r_multiple(px)
            if stop_hit(k):
                realised += size * (stop - s.entry) * d / risk
                return ("breakeven after TP1" if nxt else "stop loss"), True, fill, k, realised
            while nxt < len(targets) and tp_hit(k, targets[nxt][0]):
                px, frac = targets[nxt]
                realised += frac * s.r_multiple(px)
                size -= frac
                if nxt == 0:
                    stop = s.entry + d * sp       # breakeven: entry plus spread
                nxt += 1
            if nxt == len(targets):
                return f"TP{len(targets)} hit", True, fill, k, realised
            if nxt == 0 and k - fill >= e.time_stop_candles:
                px = c[k] + (sp if d == -1 else 0)
                return "time stop (2h without TP1)", True, fill, k, realised + size * s.r_multiple(px)
            if self.t[k] + Timeframe.M5.delta >= session_end:
                px = c[k] + (sp if d == -1 else 0)
                return "session end close", True, fill, k, realised + size * s.r_multiple(px)

    # ----------------------------------------------------------------- run
    def run(self) -> pd.DataFrame:
        """Simulate every signal in order and return one row per signal (incl. grade B)."""
        b = self.cfg.backtest
        cand = self.model.candidates()
        busy_until = -1
        trades: list[Trade] = []
        day_r: dict = {}
        week_r: dict = {}
        losses_row: dict = {}
        per_session: dict = {}
        for j in cand:
            if j <= busy_until:
                continue
            s = self.model.evaluate(j)
            if s is None:
                continue
            self.signals.append(s)
            snap_regime = regime_at(self.h4, self.regimes, s.time)
            h4_state = self.f.snapshot(s.time).state[Timeframe.H4]
            bias = "BULL" if s.direction == 1 else "BEAR"
            base = dict(signal_time=s.time, side=s.side, grade=s.grade, score=s.score, entry=s.entry,
                        stop_loss=s.stop_loss, tp1=s.tp1, tp2=s.tp2, tp3=s.tp3, kill_zone=s.kill_zone,
                        trend_regime=snap_regime, h4_structure=h4_state, bias=bias)
            if s.grade == "B":
                trades.append(Trade(**base, outcome="grade B: logged, not sent", filled=False,
                                    fill_time=None, exit_time=None, r=0.0))
                continue
            day = self.day[j]
            week = (day - pd.Timedelta(days=day.weekday())).normalize()
            sess_key = (day, s.kill_zone)
            locked = None
            if day_r.get(day, 0.0) * b.risk_pct <= -b.daily_loss_lock_pct:
                locked = "daily loss lock"
            elif week_r.get(week, 0.0) * b.risk_pct <= -b.weekly_loss_lock_pct:
                locked = "weekly loss lock"
            elif losses_row.get(day, 0) >= b.max_losses_in_row:
                locked = "2 losses in a row today"
            elif per_session.get(sess_key, 0) >= b.max_signals_per_session:
                locked = "2 signals already this session"
            if locked:
                trades.append(Trade(**base, outcome=f"skipped: {locked}", filled=False,
                                    fill_time=None, exit_time=None, r=0.0))
                continue
            per_session[sess_key] = per_session.get(sess_key, 0) + 1
            outcome, filled, fill, exit_idx, r = self._simulate(s)
            busy_until = exit_idx
            trades.append(Trade(**base, outcome=outcome, filled=filled,
                                fill_time=self.t[fill] if fill is not None else None,
                                exit_time=self.t[exit_idx] + Timeframe.M5.delta, r=round(float(r), 4)))
            if filled:
                day_r[day] = day_r.get(day, 0.0) + r
                week_r[week] = week_r.get(week, 0.0) + r
                losses_row[day] = losses_row.get(day, 0) + 1 if r < 0 else 0
        return pd.DataFrame([asdict(t) for t in trades])
