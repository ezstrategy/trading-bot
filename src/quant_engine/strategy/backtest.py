"""Event-driven backtest of the entry model on closed M5 candles.

Order and position handling is in :mod:`.lifecycle` (shared with live signals).

* One signal at a time. Grade B setups are logged but not traded.
* Risk locks (in R, 1R = ``risk_pct``): daily loss lock, weekly loss lock,
  two losses in a row stops for the day, max two signals per kill zone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import pandas as pd

from ..config import StrategyConfig
from ..features.context import MarketFeatures
from ..features.news import NewsCalendar
from ..features.sessions import trading_day
from ..timeframes import Timeframe
from .entry import EntryModel, Signal
from .lifecycle import kill_zone_end, simulate
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


_kz_end = kill_zone_end   # kept for compatibility


class Backtester:
    """Runs the entry model over all stored history."""

    def __init__(self, features: MarketFeatures, cfg: StrategyConfig, news: NewsCalendar | None = None) -> None:
        self.f, self.cfg = features, cfg
        self.model = EntryModel(features, cfg, news)
        m5 = features.bars[Timeframe.M5]
        self.t = features.times[Timeframe.M5]
        self.h, self.l, self.c = (m5[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))
        h4 = features.bars[Timeframe.H4]
        b = cfg.backtest
        self.h4 = h4
        self.regimes = trend_regime_series(h4, b.regime_ema_period, b.regime_slope_bars, b.regime_min_slope_atr)
        self.day = trading_day(self.t, cfg.sessions)
        self.signals: list[Signal] = []
        self.results: dict = {}          # signal time -> TradeResult (traded signals)
        self.decisions: list[tuple[Signal, str]] = []   # every signal and what happened to it

    # ---------------------------------------------------------- simulation
    def _simulate(self, s: Signal) -> tuple[str, bool, int | None, int, float]:
        """Returns (outcome, filled, fill_idx, exit_idx, R); see :mod:`.lifecycle`."""
        res = simulate(s, self.t, self.h, self.l, self.c, self.cfg)
        self.results[s.time] = res
        if res.outcome == "pending":     # signal on the last candle of the data
            return "pending at end of data", False, None, res.exit_idx, 0.0
        return res.outcome, res.filled, res.fill_idx, res.exit_idx, res.r

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
                self.decisions.append((s, trades[-1].outcome))
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
                self.decisions.append((s, trades[-1].outcome))
                continue
            per_session[sess_key] = per_session.get(sess_key, 0) + 1
            outcome, filled, fill, exit_idx, r = self._simulate(s)
            busy_until = exit_idx
            trades.append(Trade(**base, outcome=outcome, filled=filled,
                                fill_time=self.t[fill] if fill is not None else None,
                                exit_time=self.t[exit_idx] + Timeframe.M5.delta, r=round(float(r), 4)))
            self.decisions.append((s, outcome))
            if filled:
                day_r[day] = day_r.get(day, 0.0) + r
                week_r[week] = week_r.get(week, 0.0) + r
                losses_row[day] = losses_row.get(day, 0) + 1 if r < 0 else 0
        if not trades:
            return pd.DataFrame(columns=[f.name for f in fields(Trade)])
        return pd.DataFrame([asdict(t) for t in trades])
