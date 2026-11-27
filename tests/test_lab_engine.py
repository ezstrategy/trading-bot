"""Quant Lab Phase 3: event-driven engine, multi-timeframe alignment, indicators, look-ahead harness.

Candles are hand-built so each expected number can be worked out on paper. Prices are BID;
spread 0.08 (8 points), point 0.01, 1 lot = 100 oz, so a 1.00 move on 1 lot = $100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest.engine import ExecutionConfig, run_backtest
from quant_engine.backtest.lookahead import check_no_lookahead
from quant_engine.backtest.market import MarketData
from quant_engine.backtest.mtf import align, closed_index
from quant_engine.indicators import ema, ema_slope, opening_range, session_of, trend_state
from quant_engine.instruments import InstrumentSpec
from quant_engine.strategies.base import OrderIntent, Strategy, StrategyMeta
from quant_engine.timeframes import Timeframe

SPEC = InstrumentSpec(symbol="XAUUSD", contract_size=100, min_lot=0.01, max_lot=100, lot_step=0.01, tick_size=0.01,
                      tick_value=1.0, digits=2, commission_per_lot=0.0, typical_spread=0.08)
NOSLIP = ExecutionConfig(slippage_points=0)


def bars(rows, start="2026-01-05 10:00", freq="5min", spread=8.0):
    t = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    df["tick_volume"], df["spread"], df["real_volume"] = 1.0, spread, 0.0
    return df


FLAT = (2000.0, 2000.5, 1999.5, 2000.0)


class Scripted(Strategy):
    """Emits fixed intents at fixed candle indices (test double)."""
    meta = StrategyMeta(id="scripted", version="0", name="scripted", direction="both")

    def __init__(self, plan: dict[int, list[OrderIntent]], sid: str = "scripted", regimes=None):
        super().__init__()
        self.plan, self._regimes = plan, regimes
        self.meta = StrategyMeta(id=sid, version="0", name=sid, direction="both")

    def prepare(self, data):
        pass

    def on_bar(self, view):
        return [OrderIntent(**{**x.__dict__}) for x in self.plan.get(view.i, [])]

    def allowed_regimes(self):
        return self._regimes


def run(rows, plan, cfg=NOSLIP, m1=None, spec=SPEC, **kw):
    data = {Timeframe.M5: bars(rows, **kw)}
    if m1 is not None:
        data[Timeframe.M1] = m1
    return run_backtest(MarketData(data, 0.01), [Scripted(plan)], spec, cfg)


# ------------------------------------------------------------------ entry timing / fills
def test_signal_fills_at_next_open_with_spread_and_slippage():
    rows = [FLAT, (2001.0, 2001.2, 2000.8, 2001.0)] + [FLAT] * 3
    res = run(rows, {0: [OrderIntent(side=1, stop=1990.0, target=2100.0, time_exit_bars=2)]},
              cfg=ExecutionConfig(slippage_points=2))
    tr = res.trades.iloc[0]
    assert tr["signal_time"] == pd.Timestamp("2026-01-05 10:05", tz="UTC")      # close of candle 0
    assert tr["entry_time"] == pd.Timestamp("2026-01-05 10:05", tz="UTC")       # open of candle 1
    assert tr["entry"] == pytest.approx(2001.0 + 0.08 + 0.02)                   # ask + 2 points slippage
    assert tr["exit_reason"] == "time" and tr["bars_held"] == 2
    assert tr["exit_time"] == pd.Timestamp("2026-01-05 10:15", tz="UTC")        # close of the 2nd candle held
    assert tr["exit"] == pytest.approx(2000.0 - 0.02)                           # bid close - slippage


def test_short_pays_spread_at_exit_and_is_stopped_on_the_ask():
    # short at 2000 (bid), stop 2001: candle high 2000.95 + 0.08 spread = 2001.03 >= stop
    rows = [FLAT, FLAT, (2000.0, 2000.95, 1999.9, 2000.5)]
    res = run(rows, {0: [OrderIntent(side=-1, stop=2001.0, target=1990.0)]})
    tr = res.trades.iloc[0]
    assert tr["entry"] == pytest.approx(2000.0) and tr["exit_reason"] == "stop"
    assert tr["exit"] == pytest.approx(2001.0) and tr["r"] == pytest.approx(-1.0)
    assert tr["spread_cost"] == pytest.approx(0.08 * 100 * tr["lots"])


def test_stop_first_when_stop_and_target_are_in_one_m5_candle():
    rows = [FLAT, FLAT, (2000.2, 2003.0, 1998.0, 2001.0)]
    res = run(rows, {0: [OrderIntent(side=1, stop=1999.0, target=2002.0)]})
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop" and tr["intrabar"] == "stop_first"


def _m1(start, rows):
    return bars(rows, start=start, freq="1min")


def test_m1_resolves_the_order_inside_the_m5_candle():
    rows = [FLAT, FLAT, (2000.2, 2003.0, 1998.0, 2001.0)]
    plan = {0: [OrderIntent(side=1, stop=1999.0, target=2002.0)]}
    target_first = _m1("2026-01-05 10:10", [(2000.2, 2002.5, 2000.1, 2002.4), (2002.4, 2003.0, 2001.0, 2001.5),
                                            (2001.5, 2001.6, 1998.0, 1998.5), (1998.5, 2000, 1998.4, 2000), (2000, 2001, 1999.9, 2001)])
    tr = run(rows, plan, m1=target_first).trades.iloc[0]
    assert tr["exit_reason"] == "target" and tr["intrabar"] == "m1"
    assert tr["exit_time"] == pd.Timestamp("2026-01-05 10:11", tz="UTC")
    both_in_one_minute = _m1("2026-01-05 10:10", [(2000.2, 2003.0, 1998.0, 2001.0)] + [(2001, 2001.2, 2000.9, 2001)] * 4)
    tr = run(rows, plan, m1=both_in_one_minute).trades.iloc[0]
    assert tr["exit_reason"] == "stop" and tr["intrabar"] == "m1_stop_first"
    tr = run(rows, plan, m1=target_first, cfg=ExecutionConfig(slippage_points=0, use_m1=False)).trades.iloc[0]
    assert tr["intrabar"] == "stop_first"


def test_gap_through_the_stop_fills_at_the_open():
    rows = [FLAT, FLAT, (1995.0, 1996.0, 1994.0, 1995.5)]
    tr = run(rows, {0: [OrderIntent(side=1, stop=1999.0)]}).trades.iloc[0]
    assert tr["exit"] == pytest.approx(1995.0) and tr["r"] < -1.0


def test_target_from_r_uses_the_filled_risk():
    rows = [FLAT, FLAT, (2000.0, 2004.0, 1999.9, 2003.0)]
    tr = run(rows, {0: [OrderIntent(side=1, stop=1999.08, target_r=2.0)]}).trades.iloc[0]
    # entry 2000.08, risk 1.00, target 2002.08
    assert tr["target"] == pytest.approx(2002.08) and tr["exit"] == pytest.approx(2002.08) and tr["r"] == pytest.approx(2.0)


def test_risk_bounds_and_wrong_side_stop_are_skipped():
    res = run([FLAT] * 3, {0: [OrderIntent(side=1, stop=1999.0, risk_bounds=(0.1, 0.5))],
                           1: [OrderIntent(side=1, stop=2001.0)]})
    assert res.trades.empty
    assert res.skips["scripted"] == {"stop size outside limits": 1, "stop on the wrong side of the fill": 1}


def test_data_gap_cancels_the_fill():
    df = bars([FLAT] * 4)
    df.loc[1:, "time"] = df.loc[1:, "time"] + pd.Timedelta(minutes=30)
    res = run_backtest(MarketData({Timeframe.M5: df}, 0.01), [Scripted({0: [OrderIntent(side=1, stop=1990)]})], SPEC, NOSLIP)
    assert res.trades.empty and res.skips["scripted"] == {"data gap before fill": 1}


# ------------------------------------------------------------------ costs, sizing, equity
def test_position_size_commission_gross_net_and_equity():
    spec = SPEC.model_copy(update={"commission_per_lot": 7.0})
    rows = [FLAT, FLAT, (2000.0, 2003.0, 1999.9, 2002.5)]
    res = run(rows, {0: [OrderIntent(side=1, stop=1997.58, target=2002.08)]}, spec=spec,
              cfg=ExecutionConfig(slippage_points=0, initial_equity=10_000))
    tr = res.trades.iloc[0]
    # risk 2.50 -> $50 / ($100 x 2.50) = 0.20 lots
    assert tr["lots"] == pytest.approx(0.20) and tr["risk_money"] == pytest.approx(50.0)
    assert tr["commission"] == pytest.approx(1.40)
    assert tr["net_pnl"] == pytest.approx(2.0 * 100 * 0.20 - 1.40)              # +2.00 move
    assert tr["spread_cost"] == pytest.approx(0.08 * 100 * 0.20)
    assert tr["gross_pnl"] == pytest.approx(tr["net_pnl"] + 1.40 + 1.60)
    assert tr["r"] == pytest.approx((40 - 1.4) / 50) and tr["r_gross"] == pytest.approx(41.6 / 50)   # 2.08 x 100 x 0.20
    assert tr["equity_after"] == pytest.approx(10_000 + 38.6)
    assert res.equity["equity"].tolist() == pytest.approx([10_000, 10_038.6])


def test_lots_follow_current_equity():
    rows = [FLAT, FLAT, (2000.0, 2000.1, 1996.0, 1997.0), FLAT, FLAT, FLAT]
    plan = {0: [OrderIntent(side=1, stop=1997.58)], 3: [OrderIntent(side=1, stop=1997.58)]}
    res = run(rows, plan, cfg=ExecutionConfig(slippage_points=0, initial_equity=10_000, max_concurrent=1))
    a, b = res.trades.iloc[0], res.trades.iloc[1]
    assert a["net_pnl"] == pytest.approx(-50.0)
    assert b["equity_before"] == pytest.approx(9_950) and b["lots"] == pytest.approx(0.19)   # 49.75/250 = 0.199 -> 0.19


def test_spread_floor_assumed_when_missing_and_stress_multiplier():
    rows = [FLAT, FLAT, FLAT]
    plan = {0: [OrderIntent(side=1, stop=1990.0, time_exit_bars=1)]}
    tr = run(rows, plan, spread=np.nan).trades.iloc[0]
    assert tr["entry"] == pytest.approx(2000.08) and bool(tr["spread_assumed"])
    tr = run(rows, plan, spread=3.0).trades.iloc[0]
    assert tr["entry"] == pytest.approx(2000.08) and not bool(tr["spread_assumed"])     # floored at 0.08
    tr = run(rows, plan, spread=20.0, cfg=ExecutionConfig(slippage_points=0, spread_multiplier=2.0)).trades.iloc[0]
    assert tr["entry"] == pytest.approx(2000.40)


def test_min_lot_too_risky_is_skipped_not_rounded_up():
    res = run([FLAT] * 3, {0: [OrderIntent(side=1, stop=1950.0)]}, cfg=ExecutionConfig(slippage_points=0, initial_equity=100))
    assert res.trades.empty and "lot below broker minimum for this risk" in res.skips["scripted"]


# ------------------------------------------------------------------ duplicates / concurrency / regime / windows
def test_duplicates_and_busy_strategy_are_rejected():
    it = OrderIntent(side=1, stop=1990.0, time_exit_bars=3)
    res = run([FLAT] * 8, {0: [it, it], 1: [it]})
    assert len(res.trades) == 1
    out = res.signals["outcome"].tolist()
    assert out == ["queued", "duplicate", "skipped: strategy already in a trade"]


def test_regime_filter_blocks_and_h4_regime_is_causal():
    data = MarketData({Timeframe.M5: bars([FLAT] * 3)}, 0.01)
    s = Scripted({0: [OrderIntent(side=1, stop=1990.0)]}, regimes={"bullish"})
    res = run_backtest(data, [s], SPEC, NOSLIP)
    assert res.trades.empty and res.signals["outcome"].iat[0] == "blocked: regime"
    assert res.signals["regime"].iat[0] == "unknown"                   # no H4 loaded


def test_trade_window_limits_signals_not_exits():
    rows = [FLAT] * 6
    plan = {i: [OrderIntent(side=1, stop=1990.0, time_exit_bars=3)] for i in range(6)}
    res = run_backtest(MarketData({Timeframe.M5: bars(rows)}, 0.01), [Scripted(plan)], SPEC, NOSLIP,
                       trade_from=pd.Timestamp("2026-01-05 10:10", tz="UTC"),
                       trade_until=pd.Timestamp("2026-01-05 10:15", tz="UTC"))
    assert len(res.signals) == 1 and len(res.trades) == 1
    assert res.trades["exit_time"].iat[0] == pd.Timestamp("2026-01-05 10:25", tz="UTC")


def test_open_position_at_end_of_data_is_closed_and_labelled():
    res = run([FLAT] * 3, {0: [OrderIntent(side=1, stop=1990.0)]})
    assert res.trades["exit_reason"].iat[0] == "end_of_data"


# ------------------------------------------------------------------ multi-timeframe and indicators
def test_htf_candle_usable_only_after_it_closes():
    h1 = pd.date_range("2026-01-05 09:00", periods=3, freq="h", tz="UTC")
    m5_close = pd.DatetimeIndex(["2026-01-05 09:55", "2026-01-05 10:00", "2026-01-05 10:05"], tz="UTC")
    idx = closed_index(h1, Timeframe.H1, m5_close)
    assert idx.tolist() == [-1, 0, 0]                                    # 09:00 H1 closes at 10:00
    assert align(np.array([1.0, 2.0, 3.0]), idx).tolist()[1:] == [1.0, 1.0]
    assert np.isnan(align(np.array([1.0, 2.0, 3.0]), idx)[0])


def test_market_data_truncate_drops_unclosed_candles():
    m5 = bars([FLAT] * 24, start="2026-01-05 09:00")
    h1 = bars([FLAT] * 2, start="2026-01-05 09:00", freq="h")
    d = MarketData({Timeframe.M5: m5, Timeframe.H1: h1}, 0.01).truncate(pd.Timestamp("2026-01-05 10:30", tz="UTC"))
    assert len(d) == 18 and len(d.frame(Timeframe.H1)) == 1


def test_indicators():
    x = np.arange(1.0, 61.0)
    e = ema(x, 20)
    assert np.isnan(e[18]) and e[19] == pytest.approx(pd.Series(x).ewm(span=20, adjust=False).mean().iat[19])
    assert ema_slope(np.array([1.0, 2.0, 4.0, 7.0]), 2).tolist()[2:] == [3.0, 5.0]
    fast, slow = ema(x, 5), ema(x, 10)
    assert trend_state(x, fast, slow, 3)[-1] == "bullish" and trend_state(x, fast, slow, 3)[0] == "sideways"
    # London 08:00 is 07:00 UTC in summer and 08:00 UTC in winter
    t = pd.DatetimeIndex(list(pd.date_range("2026-07-06 06:30", periods=24, freq="5min", tz="UTC"))
                         + list(pd.date_range("2026-01-05 07:30", periods=24, freq="5min", tz="UTC")))
    hi = np.arange(48.0)
    orr = opening_range(t, hi, hi - 1, "Europe/London", 8 * 60, 30)
    assert orr["first_idx"].tolist() == [30, 6] and orr["complete"].all()       # sorted by date: Jan, Jul
    assert orr["high"].tolist() == [35.0, 11.0]
    assert list(session_of(pd.DatetimeIndex(["2026-01-05 13:00", "2026-01-05 03:00"], tz="UTC"))) == [
        "london_ny_overlap", "asia"]


# ------------------------------------------------------------------ look-ahead harness
class Cheater(Strategy):
    """Buys when the NEXT candle closes higher: look-ahead on purpose."""
    meta = StrategyMeta(id="cheater", version="0", name="cheater", direction="long")

    def prepare(self, data):
        c = data.base.c
        self.up = np.r_[c[1:] > c[:-1], False]

    def on_bar(self, view):
        return [OrderIntent(side=1, stop=view.data.base.c[view.i] - 5)] if self.up[view.i] else []


def _walk(n=400, seed=1):
    rng = np.random.default_rng(seed)
    c = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    o = np.r_[2000, c[:-1]]
    return bars(list(zip(o, np.maximum(o, c) + 0.2, np.minimum(o, c) - 0.2, c)), start="2026-01-05 00:00")


def test_lookahead_harness_catches_a_cheating_strategy():
    data = MarketData({Timeframe.M5: _walk()}, 0.01)
    rep = check_no_lookahead(Cheater, data, cuts=6)
    assert not rep["ok"] and rep["differences"]


def test_lookahead_harness_passes_a_causal_strategy():
    class Causal(Cheater):
        def prepare(self, data):
            c = data.base.c
            self.up = np.r_[False, c[1:] > c[:-1]]
    rep = check_no_lookahead(Causal, MarketData({Timeframe.M5: _walk()}, 0.01), cuts=6)
    assert rep["ok"] and rep["intents_full"] > 0 and len(rep["cuts"]) >= 4
