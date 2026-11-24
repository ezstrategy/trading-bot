"""ATR, sessions, swings, structure, bias, displacement, FVG and order blocks."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import SessionsConfig, StrategyConfig
from quant_engine.features.atr import atr, true_range
from quant_engine.features.context import BEAR, BULL, NONE, DealingRange, htf_bias, m15_allows
from quant_engine.features.imbalance import displacement, find_fvgs, order_block
from quant_engine.features.sessions import (in_kill_zone, in_kill_zone_or_grace, new_signals_blocked,
                                            session_labels, trading_day)
from quant_engine.features.structure import BEARISH, BULLISH, RANGING, analyze_structure
from quant_engine.features.swings import find_swings, swings_known_at
from quant_engine.timeframes import Timeframe

from .helpers import candles, from_closes, highs_lows

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
S = SessionsConfig()


# ------------------------------------------------------------------ ATR
def test_atr_wilder():
    b = candles([(10, 11, 9, 10)] * 20)                 # constant TR = 2
    a = atr(b, 14)
    assert a.iloc[:13].isna().all() and np.allclose(a.iloc[13:], 2.0)
    b2 = candles([(10, 11, 9, 10)] * 14 + [(10, 15, 9, 14)])  # TR 6 on the last candle
    assert atr(b2, 14).iat[-1] == pytest.approx((2 * 13 + 6) / 14)


def test_true_range_uses_previous_close():
    b = candles([(10, 11, 9, 10), (14, 15, 13.5, 14)])   # gap up: TR = 15 - 10
    assert true_range(b).iat[1] == 5


def test_atr_is_causal():
    b = from_closes(list(np.cumsum(np.random.default_rng(1).normal(0, 1, 200)) + 100))
    full, part = atr(b, 14), atr(b.iloc[:120], 14)
    assert np.allclose(full.iloc[:120], part, equal_nan=True)


# ------------------------------------------------------------- sessions
@pytest.mark.parametrize("utc,label", [
    ("2026-11-24 07:00", "london_kz"), ("2026-11-24 09:55", "london_kz"), ("2026-11-24 10:00", "off"),
    ("2026-11-24 12:00", "newyork_kz"), ("2026-11-24 14:55", "newyork_kz"), ("2026-11-24 15:00", "london_close"),
    ("2026-11-24 01:00", "asia"), ("2026-11-24 04:55", "asia"), ("2026-11-24 22:00", "rollover"),
    ("2026-11-24 05:30", "off"),
    # summer (New York on EDT): windows shift one hour earlier in UTC
    ("2026-07-14 06:00", "london_kz"), ("2026-07-14 11:00", "newyork_kz"), ("2026-07-14 14:00", "london_close"),
])
def test_session_labels_follow_new_york_time(utc, label):
    assert session_labels(pd.DatetimeIndex([T(utc)]), S)[0] == label


def test_kill_zones_in_ist_match_spec():
    """Spec: winter kill zones are 12:30-15:30 and 17:30-20:30 IST."""
    ist = pd.date_range("2026-11-24 00:00", "2026-11-24 23:55", freq="5min", tz="Asia/Kolkata")
    kz = ist[in_kill_zone(ist.tz_convert("UTC"), S)]
    first_lon, last_ny = kz[0], kz[-1]
    assert (first_lon.hour, first_lon.minute) == (12, 30) and (last_ny.hour, last_ny.minute) == (20, 25)
    assert len(kz) == 2 * 36


def test_grace_period():
    t = pd.DatetimeIndex([T("2026-11-24 10:25"), T("2026-11-24 10:30")])
    assert list(in_kill_zone(t, S)) == [False, False]
    assert list(in_kill_zone_or_grace(t, S)) == [True, False]


def test_trading_day_starts_at_17_new_york():
    t = pd.DatetimeIndex([T("2026-11-23 21:55"), T("2026-11-23 22:00"), T("2026-07-14 20:59"), T("2026-07-14 21:00")])
    d = trading_day(t, S)
    assert [x.strftime("%m-%d") for x in d] == ["11-23", "11-24", "07-14", "07-15"]


def test_signal_blocks():
    t = pd.DatetimeIndex([T("2026-11-20 14:55"), T("2026-11-20 15:00"), T("2026-11-22 23:30"), T("2026-11-23 13:00")])
    assert list(new_signals_blocked(t, S)) == [False, True, True, False]  # Fri 09:55 / 10:00 NY, Sunday, Monday


# --------------------------------------------------------------- swings
def test_fractal_swings_and_confirmation_time():
    b = highs_lows([1, 2, 5, 3, 2, 4, 6, 4, 3], [0, 1, 3, 1.5, 0.5, 2, 4, 2, 1])
    sw = find_swings(b, Timeframe.M5, 2)
    hi = sw[sw["kind"] == "high"]
    assert list(hi["idx"]) == [2, 6]
    lo = sw[sw["kind"] == "low"]
    assert list(lo["idx"]) == [4]
    first = hi.iloc[0]
    assert first["confirmed_idx"] == 4 and first["known_at"] == b["time"].iat[4] + pd.Timedelta(minutes=5)
    assert len(swings_known_at(sw, b["time"].iat[4])) == 0  # candle 4 still open
    assert len(swings_known_at(sw, b["time"].iat[4] + pd.Timedelta(minutes=5))) == 1


def test_equal_highs_are_not_fractals():
    b = highs_lows([1, 2, 5, 5, 2, 1, 1])
    assert find_swings(b, Timeframe.M5, 2).query("kind == 'high'").empty


def test_swings_never_on_unconfirmed_edge():
    b = highs_lows([1, 2, 3, 4, 9, 5])  # peak at index 4 has only one candle after it
    assert find_swings(b, Timeframe.M5, 2).query("kind == 'high'").empty


# ------------------------------------------------------------ structure
def zigzag(points: list[float], steps: int = 3) -> list[float]:
    out = []
    for a, b in zip(points[:-1], points[1:]):
        out += list(np.linspace(a, b, steps + 1)[:-1])
    return out + [points[-1]]


def test_bos_choch_and_states():
    # Up-trend: higher highs and lows, then a sharp drop through the last higher low.
    closes = zigzag([100, 104, 102, 106, 104, 108, 103, 101, 99], 3)
    b = from_closes(closes, wick=0.1)
    r = analyze_structure(b, Timeframe.M5, 2, ranging_lookback=10)
    kinds = list(zip(r.events["kind"], r.events["direction"]))
    assert ("BOS", 1) in kinds and kinds[-1][1] == -1 and kinds[-1][0] == "CHoCH"
    first_up = r.events[r.events["direction"] == 1].iloc[0]
    assert b["close"].iat[first_up["idx"]] > first_up["level"]           # broken on a close
    assert b["close"].iat[first_up["idx"] - 1] <= first_up["level"]
    ups = r.events[r.events["direction"] == 1]
    assert (r.state.iloc[ups["idx"].iat[-1]:r.events["idx"].iat[-1]] == BULLISH).all()
    assert r.state.iat[0] == RANGING


def test_wick_through_is_not_a_break():
    # swing high 105 at idx 2, later a candle wicks to 106 but closes 104
    rows = [(100, 101, 99, 100), (100, 103, 99.5, 102), (102, 105, 101, 103), (103, 103.5, 101, 102),
            (102, 102.5, 100, 101), (101, 106, 100.5, 104), (104, 104.5, 102, 103)]
    r = analyze_structure(candles(rows), Timeframe.M5, 2)
    assert r.events.empty


def test_opposite_breaks_inside_lookback_mean_ranging():
    closes = zigzag([100, 104, 102, 106, 103, 99, 101, 107], 2)
    b = from_closes(closes, wick=0.05)
    r = analyze_structure(b, Timeframe.M5, 1, ranging_lookback=10)
    assert len(r.events) >= 2 and r.events["direction"].iat[-1] != r.events["direction"].iat[-2]
    assert r.state.iat[r.events["idx"].iat[-1]] == RANGING
    r_short = analyze_structure(b, Timeframe.M5, 1, ranging_lookback=1)
    assert r_short.state.iat[r.events["idx"].iat[-1]] in (BULLISH, BEARISH)


def test_close_below_leg_low_flips_to_bearish():
    closes = zigzag([100, 104, 102, 106], 3) + [105, 104.5, 104.8, 101.5]
    r = analyze_structure(from_closes(closes, wick=0.05), Timeframe.M5, 1, ranging_lookback=1)
    assert list(r.events["kind"]) == ["BOS", "CHoCH"]
    assert r.events["leg_origin"].iat[0] == pytest.approx(101.95)
    assert r.state.iat[12] == BULLISH and r.state.iat[13] == BEARISH
    assert (r.state.iloc[:8] == RANGING).all()


# ----------------------------------------------------------------- bias
@pytest.mark.parametrize("h4,h1,bias,pts", [
    (BULLISH, BULLISH, BULL, 20), (RANGING, BULLISH, BULL, 10), (BEARISH, BEARISH, BEAR, 20),
    (RANGING, BEARISH, BEAR, 10), (BULLISH, BEARISH, NONE, 0), (BEARISH, BULLISH, NONE, 0),
    (BULLISH, RANGING, NONE, 0), (RANGING, RANGING, NONE, 0)])
def test_htf_bias_table(h4, h1, bias, pts):
    assert htf_bias(h4, h1) == (bias, pts)


def test_m15_gate():
    assert m15_allows(BULLISH, 1) and m15_allows(RANGING, 1) and not m15_allows(BEARISH, 1)
    assert m15_allows(BEARISH, -1) and not m15_allows(BULLISH, -1)


def test_dealing_range():
    dr = DealingRange(100, 110)
    assert dr.equilibrium == 105 and dr.zone(103) == "discount" and dr.zone(107) == "premium"


# ------------------------------------------------- displacement / FVG / OB
def test_displacement_rules():
    b = candles([(10, 11, 9, 10)] * 14 + [(10, 13, 9.9, 12.9), (10, 13, 7, 12.5), (12.5, 12.6, 8.4, 8.5)])
    d = displacement(b, atr(b, 14), 1.2, 0.6)
    assert list(d.iloc[-3:]) == [1, 0, -1]   # big body / big body but long wick / bearish


def test_fvg_detection_size_and_invalidation():
    rows = [(10, 11, 9, 10)] * 14 + [
        (10, 10.5, 9.8, 10.4),     # c1 high 10.5
        (10.4, 12.8, 10.3, 12.7),  # displacement
        (12.7, 13.0, 11.5, 12.9),  # c3 low 11.5 > 10.5 -> bullish FVG [10.5, 11.5]
        (12.9, 13.0, 11.0, 11.2),  # trades into gap, closes inside
        (11.2, 11.3, 10.0, 10.2),  # closes below 10.5 -> invalidated
    ]
    b = candles(rows)
    f = find_fvgs(b, Timeframe.M5, atr(b, 14), point=0.01, min_size_atr=0.25, min_size_spreads=3)
    bull = f[(f["direction"] == 1) & (f["c3_idx"] == 16)].iloc[0]
    assert (bull["bottom"], bull["top"], bull["ce"]) == (10.5, 11.5, 11.0)
    assert bull["valid_size"] and bull["known_at"] == b["time"].iat[16] + pd.Timedelta(minutes=5)
    assert bull["invalidated_at"] == b["time"].iat[18] + pd.Timedelta(minutes=5)
    wide_spread = find_fvgs(candles(rows, spread=50), Timeframe.M5, atr(b, 14), 0.01, 0.25, 3)
    assert not wide_spread[(wide_spread["direction"] == 1) & (wide_spread["c3_idx"] == 16)]["valid_size"].iat[0]


def test_bearish_fvg():
    b = candles([(10, 10.2, 9.5, 9.6)] + [(9.6, 9.7, 8, 8.1), (8.1, 9.0, 7.8, 7.9)])  # high(c3)=9.0 < low(c1)=9.5
    f = find_fvgs(b, Timeframe.M5, pd.Series([1.0] * 3), 0.01, 0, 0)
    r = f[f["direction"] == -1].iloc[0]
    assert (r["bottom"], r["top"]) == (9.0, 9.5)


def test_order_block():
    b = candles([(10, 10.5, 9.5, 10.2), (10.2, 10.3, 9.6, 9.8), (9.8, 9.9, 9.4, 9.7), (9.7, 12, 9.6, 11.9)])
    assert order_block(b, 3, 1) == (9.4, 9.8, 2)       # last down-close candle: [low, open]
    assert order_block(b, 3, -1) == (10.0, 10.5, 0)    # last up-close candle: [open, high]
    assert order_block(candles([(10, 11, 9, 10.5)]), 0, 1) is None


def test_strategy_config_loaded(cfg):
    assert isinstance(cfg.strategy, StrategyConfig)
    assert cfg.strategy.swing_length(Timeframe.M5) == 2 and cfg.strategy.swing_length(Timeframe.H4) == 3
    assert cfg.strategy.sessions.newyork_kz.start == 7 * 60
