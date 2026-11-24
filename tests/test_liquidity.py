"""Liquidity pools and sweep rules."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import LiquidityConfig, SessionsConfig
from quant_engine.features import liquidity as liq
from quant_engine.features.atr import atr
from quant_engine.timeframes import Timeframe

from .helpers import candles

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
S, L = SessionsConfig(), LiquidityConfig()


def flat_day(start: str, end: str, price: float = 100.0) -> pd.DataFrame:
    t = pd.date_range(T(start), T(end), freq="5min", inclusive="left")
    return candles([(price, price + 0.5, price - 0.5, price)] * len(t), start=start)


def test_session_pools_from_new_york_windows():
    # Trading day 2026-11-24 runs 23 Nov 22:00 UTC -> 24 Nov 22:00 UTC (winter).
    m5 = flat_day("2026-11-23 22:00", "2026-11-24 22:00")
    t = pd.DatetimeIndex(m5["time"])
    asia = (t >= T("2026-11-24 01:00")) & (t < T("2026-11-24 05:00"))
    london = (t >= T("2026-11-24 07:00")) & (t < T("2026-11-24 10:00"))
    m5.loc[asia & (t == T("2026-11-24 02:00")), "high"] = 103.0
    m5.loc[asia & (t == T("2026-11-24 03:00")), "low"] = 98.0
    m5.loc[london & (t == T("2026-11-24 08:00")), "high"] = 104.0
    m5.loc[t == T("2026-11-24 13:00"), "low"] = 95.0
    rows = {(k, side): (price, known) for k, _, side, price, known, _ in liq.session_pools(m5, S)}
    assert rows[("asia", "high")] == (103.0, T("2026-11-24 05:00"))
    assert rows[("asia", "low")] == (98.0, T("2026-11-24 05:00"))
    assert rows[("london", "high")] == (104.0, T("2026-11-24 10:00"))
    assert rows[("previous_day", "low")] == (95.0, T("2026-11-24 22:00"))


def test_session_pool_not_created_before_window_ends():
    m5 = flat_day("2026-11-23 22:00", "2026-11-24 04:00")  # Asia still running
    kinds = {r[0] for r in liq.session_pools(m5, S)}
    assert "asia" not in kinds and "previous_day" not in kinds


def test_equal_levels_and_tolerance():
    sw = pd.DataFrame({"idx": [1, 9, 20], "kind": ["high"] * 3, "price": [100.0, 100.05, 101.0],
                       "time": [T("2026-11-24 08:00"), T("2026-11-24 08:40"), T("2026-11-24 09:40")],
                       "known_at": [T("2026-11-24 08:15"), T("2026-11-24 08:55"), T("2026-11-24 09:55")]})
    rows = liq.equal_level_pools(sw, 0.1, pd.Timedelta(days=1))
    assert len(rows) == 1 and rows[0][3] == 100.05 and rows[0][4] == T("2026-11-24 08:55")
    assert liq.equal_level_pools(sw, 0.01, pd.Timedelta(days=1)) == []


def test_taken_and_live_pools():
    m5 = flat_day("2026-11-24 07:00", "2026-11-24 09:00")
    m5.loc[10, "high"] = 102.0
    pools = liq.build_pools(m5, [("m5_swing", liq.MINOR, "high", 101.0, T("2026-11-24 07:10"), T("2026-11-24 07:00"))])
    assert pools["taken_at"].iat[0] == m5["time"].iat[10] and pools["taken_idx"].iat[0] == 10
    assert len(liq.live_pools(pools, m5["time"].iat[10])) == 1                        # candle 10 not closed yet
    assert len(liq.live_pools(pools, m5["time"].iat[10] + pd.Timedelta(minutes=5))) == 0
    assert len(liq.live_pools(pools, T("2026-11-24 07:05"))) == 0                     # not known yet


# ------------------------------------------------------------------- sweeps
def sweep_setup(candles_after: list[tuple], start="2026-11-24 12:00", level=99.0):
    """14 quiet candles (ATR ~1) then the test candles; a sell-side pool at ``level``."""
    base = [(100, 100.5, 99.5, 100)] * 14
    m5 = candles(base + candles_after, start=pd.Timestamp(T(start) - pd.Timedelta(minutes=70)).strftime("%Y-%m-%d %H:%M"))
    pools = liq.build_pools(m5, [("previous_day", liq.HTF, "low", level, m5["time"].iat[0], m5["time"].iat[0])])
    return m5, pools, liq.detect_sweeps(m5, pools, atr(m5, 14), 0.01, L, S)


def test_same_candle_sweep():
    m5, _, sw = sweep_setup([(100, 100.2, 98.6, 99.8)])
    assert len(sw) == 1
    s = sw[0]
    assert s.direction == 1 and s.same_candle and s.extreme == 98.6 and s.depth == pytest.approx(0.4)
    assert s.known_at == m5["time"].iat[14] + pd.Timedelta(minutes=5) and s.tier == liq.HTF


def test_reclaim_on_next_candle():
    m5, _, sw = sweep_setup([(100, 100.2, 98.6, 98.8), (98.8, 99.6, 98.7, 99.5)])
    assert len(sw) == 1 and not sw[0].same_candle and sw[0].reclaim_idx == 15
    assert sw[0].known_at == m5["time"].iat[15] + pd.Timedelta(minutes=5)


def test_two_closes_below_is_a_breakout():
    *_, sw = sweep_setup([(100, 100.2, 98.6, 98.8), (98.8, 98.9, 98.5, 98.7), (98.7, 99.6, 98.6, 99.5)])
    assert sw == []


def test_too_deep_is_not_a_sweep():
    *_, sw = sweep_setup([(100, 100.2, 97.0, 99.5)])  # 2.0 below the level, ATR ~1.2
    assert sw == []


def test_must_trade_through_by_a_tick():
    *_, sw = sweep_setup([(100, 100.2, 99.0, 99.8)])  # touches, never trades below
    assert sw == []


def test_outside_kill_zone_is_ignored():
    *_, sw = sweep_setup([(100, 100.2, 98.6, 99.8)], start="2026-11-24 05:30")
    assert sw == []


def test_buy_side_sweep():
    base = [(100, 100.5, 99.5, 100)] * 14
    m5 = candles(base + [(100, 101.4, 99.8, 100.6)], start="2026-11-24 10:50")
    pools = liq.build_pools(m5, [("asia", liq.SESSION, "high", 101.0, m5["time"].iat[0], m5["time"].iat[0])])
    sw = liq.detect_sweeps(m5, pools, atr(m5, 14), 0.01, L, S)
    assert len(sw) == 1 and sw[0].direction == -1 and sw[0].extreme == 101.4


def test_duplicate_pools_give_one_sweep_with_best_tier():
    m5, _, _ = sweep_setup([(100, 100.2, 98.6, 99.8)])
    rows = [("m5_swing", liq.MINOR, "low", 99.0, m5["time"].iat[0], m5["time"].iat[0]),
            ("previous_day", liq.HTF, "low", 99.0, m5["time"].iat[0], m5["time"].iat[0])]
    sw = liq.detect_sweeps(m5, liq.build_pools(m5, rows), atr(m5, 14), 0.01, L, S)
    assert len(sw) == 1 and sw[0].pool_kind == "previous_day"
