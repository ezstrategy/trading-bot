"""Order fill and exit rules of the backtester, on hand-made M5 candles."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import StrategyConfig
from quant_engine.strategy.backtest import Backtester, _kz_end
from quant_engine.strategy.entry import Signal

SP = 0.30   # spread in price (30 points)
START = pd.Timestamp("2026-11-23 12:00", tz="UTC")   # 07:00 New York, NY kill zone


def sim(bars, direction=1, entry=100.0, sl=98.0, tp1=102.0, tp2=104.0, tp3=None, extreme=98.5, cfg=None):
    """Run ``_simulate`` for a signal sent at the close of row 0 of ``bars`` (list of (h, l, c))."""
    bt = object.__new__(Backtester)
    bt.cfg = cfg or StrategyConfig()
    bt.results = {}
    arr = np.array(bars, dtype="float64")
    bt.h, bt.l, bt.c = arr[:, 0], arr[:, 1], arr[:, 2]
    bt.t = pd.DatetimeIndex(pd.date_range(START, periods=len(bars), freq="5min"))
    s = Signal(bt.t[0] + pd.Timedelta(minutes=5), 0, direction, entry, sl, tp1, tp2, tp3, 70, "A",
               0, extreme, "newyork_kz", SP)
    return bt._simulate(s)


FAR = (100.8, 100.5, 100.6)     # buy setup: candle that does not reach the 100.0 entry (ask = low + 0.3)


def test_buy_limit_fills_on_ask_then_stop_loss():
    # low 99.8 -> ask 100.1 does not fill; low 99.7 -> ask 100.0 fills
    out = sim([FAR, (100.5, 99.8, 100.2), (100.4, 99.7, 99.9), (99.9, 97.9, 98.6)])
    assert out == ("stop loss", True, 2, 3, -1.0)


def test_fill_candle_checks_only_the_stop():
    # the fill candle also trades above TP1: ignored, only its stop is checked
    out = sim([FAR, (102.5, 99.0, 101.0), (101.5, 100.5, 101.0)] + [(101.2, 100.6, 101.0)] * 30)
    assert out[0] == "time stop (2h without TP1)" and out[2] == 1
    assert sim([FAR, (102.5, 97.5, 101.0)]) == ("stop loss", True, 1, 1, -1.0)


def test_stop_checked_before_target_on_the_same_candle():
    out = sim([FAR, (100.2, 99.6, 100.1), (102.5, 97.9, 100.0)])
    assert out[:2] == ("stop loss", True) and out[4] == -1.0


def test_tp1_then_breakeven_plus_spread():
    out = sim([FAR, (100.2, 99.6, 100.1), (102.1, 100.5, 101.8), (101.0, 100.2, 100.3)])
    assert out[0] == "breakeven after TP1"
    assert out[4] == pytest.approx(0.5 * 1.0 + 0.5 * SP / 2.0)   # half at +1R, half at +spread


def test_tp1_and_tp2_without_tp3():
    out = sim([FAR, (100.2, 99.6, 100.1), (102.1, 100.5, 101.8), (104.2, 101.5, 104.0)])
    assert out[0] == "TP2 hit" and out[4] == pytest.approx(0.5 * 1 + 0.5 * 2)


def test_tp1_tp2_tp3_partials():
    bars = [FAR, (100.2, 99.6, 100.1), (106.5, 100.5, 106.0)]   # all three targets in one candle
    out = sim(bars, tp3=106.0)
    assert out[0] == "TP3 hit" and out[4] == pytest.approx(0.5 * 1 + 0.25 * 2 + 0.25 * 3)


def test_order_expires_after_six_candles():
    out = sim([FAR] * 10)
    assert out == ("expired: not filled in 30 min", False, None, 6, 0.0)


def test_order_cancelled_when_tp1_trades_first():
    assert sim([FAR, (102.0, 100.6, 101.5)])[:2] == ("cancelled: TP1 before fill", False)


def test_order_cancelled_on_close_beyond_sweep():
    # gap down through the entry is a fill; a close below the sweep without touching is not possible
    # for a buy, so use a sell: price closes above the sweep extreme before the sell limit fills
    bars = [(99.5, 99.0, 99.2), (99.9, 99.2, 101.6)]
    out = sim(bars, direction=-1, entry=100.0, sl=102.0, tp1=98.0, tp2=96.0, extreme=101.5)
    assert out[:2] == ("cancelled: closed beyond sweep", False)


def test_sell_mirrors_with_spread_on_the_ask():
    # sell limit fills when the bid high reaches 100; TP1 (98) needs ask = low + 0.3 <= 98
    bars = [(99.5, 99.0, 99.2), (100.0, 99.4, 99.8), (99.9, 97.8, 98.2), (99.9, 97.69, 98.0),
            (96.0, 95.6, 95.8)]
    out = sim(bars, direction=-1, entry=100.0, sl=102.0, tp1=98.0, tp2=96.0, extreme=101.5)
    assert out[0] == "TP2 hit" and out[2] == 1 and out[3] == 4
    # stop is hit when the ask (high + spread) reaches it
    out = sim([(99.5, 99.0, 99.2), (100.0, 99.4, 99.8), (101.7, 99.5, 101.0)],
              direction=-1, entry=100.0, sl=102.0, tp1=98.0, tp2=96.0, extreme=101.5)
    assert out[0] == "stop loss" and out[4] == -1.0


def test_time_stop_after_24_candles_without_tp1():
    cfg = StrategyConfig()
    cfg.entry.session_close_hours = 24.0          # keep the session close out of the way
    bars = [FAR, (100.2, 99.6, 100.1)] + [(100.8, 99.9, 100.5)] * 40
    out = sim(bars, cfg=cfg)
    assert out[0] == "time stop (2h without TP1)" and out[3] == 1 + 24
    assert out[4] == pytest.approx(0.5 / 2.0)    # closed at 100.5 bid, entry 100, risk 2


def test_session_end_close():
    # NY kill zone ends 10:00 NY = 15:00 UTC; +2h -> 17:00 UTC. Signal at 12:05 UTC.
    cfg = StrategyConfig()
    cfg.entry.time_stop_candles = 1000
    bars = [FAR, (100.2, 99.6, 100.1)] + [(100.8, 99.9, 100.5)] * 80
    out = sim(bars, cfg=cfg)
    assert out[0] == "session end close"
    assert START + pd.Timedelta(minutes=5) * (out[3] + 1) == pd.Timestamp("2026-11-23 17:00", tz="UTC")


def test_kill_zone_end_follows_new_york_dst():
    cfg = StrategyConfig()
    assert _kz_end(pd.Timestamp("2026-11-23 12:30", tz="UTC"), "newyork_kz", cfg) == \
        pd.Timestamp("2026-11-23 15:00", tz="UTC")
    assert _kz_end(pd.Timestamp("2026-07-01 07:30", tz="UTC"), "london_kz", cfg) == \
        pd.Timestamp("2026-07-01 09:00", tz="UTC")


def test_events_follow_the_trade_and_pending_at_end_of_data():
    from quant_engine.strategy.lifecycle import BREAKEVEN, FILLED, TP1, simulate
    bars = [FAR, (100.2, 99.6, 100.1), (102.1, 100.5, 101.8), (101.0, 100.2, 100.3)]
    arr = np.array(bars)
    t = pd.DatetimeIndex(pd.date_range(START, periods=len(bars), freq="5min"))
    s = Signal(t[0] + pd.Timedelta(minutes=5), 0, 1, 100.0, 98.0, 102.0, 104.0, None, 70, "A", 0, 98.5,
               "newyork_kz", SP)
    res = simulate(s, t, arr[:, 0], arr[:, 1], arr[:, 2], StrategyConfig())
    assert [e.kind for e in res.events] == [FILLED, TP1, BREAKEVEN] and not res.is_open
    assert res.events[1].time == t[2] + pd.Timedelta(minutes=5) and res.events[1].r == 0.5
    # cut after the fill: the trade is still open, only the fill is known
    part = simulate(s, t[:2], arr[:2, 0], arr[:2, 1], arr[:2, 2], StrategyConfig())
    assert part.is_open and [e.kind for e in part.events] == [FILLED]
    # signal on the last candle: pending, nothing happened yet
    none = simulate(s, t[:1], arr[:1, 0], arr[:1, 1], arr[:1, 2], StrategyConfig())
    assert none.is_open and none.outcome == "pending" and none.events == []
