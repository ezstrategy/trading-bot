"""Multi-timeframe snapshot, volatility filters and the no-look-ahead guarantee."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import VolatilityConfig
from quant_engine.features.atr import atr
from quant_engine.features.context import MarketFeatures, STRUCTURE_TFS
from quant_engine.features.volatility import (CHAOS, DEAD, NORMAL, UNKNOWN, atr_time_of_day_ratio, spread_ok,
                                              spread_ratio, volatility_regime)
from quant_engine.timeframes import Timeframe

from .fake_mt5 import NOW_UTC
from .helpers import candles

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
START = T("2026-10-12")


@pytest.fixture(scope="module")
def bars():
    from quant_engine.config import ConnectionConfig
    from quant_engine.data.collector import MarketDataCollector
    from quant_engine.mt5.client import MT5Client
    from quant_engine.mt5.server_time import ServerClock

    from .fake_mt5 import FakeMT5
    c = MT5Client(ConnectionConfig(), mt5_module=FakeMT5())
    c.connect()
    col = MarketDataCollector(c, "XAUUSD.r", ServerClock(), now=lambda: NOW_UTC)
    return {tf: col.bars(tf, START, NOW_UTC) for tf in STRUCTURE_TFS}


@pytest.fixture(scope="module")
def features(bars):
    from quant_engine.config import StrategyConfig
    return MarketFeatures(bars, StrategyConfig(), point=0.01)


# --------------------------------------------------------------- volatility
def test_time_of_day_ratio_and_regimes():
    days = pd.date_range("2026-10-01", periods=30, freq="D", tz="UTC")
    t = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (8, 9)])
    m5 = pd.DataFrame({"time": t, "spread": 20.0})
    a = pd.Series(np.where(t.hour == 8, 1.0, 2.0))
    a.iloc[-2] = 0.5    # last day 08:00: half of normal -> dead
    a.iloc[-1] = 6.0    # last day 09:00: 3x normal -> chaos
    r = atr_time_of_day_ratio(m5, a, days=20)
    assert r.iloc[:5].isna().all()
    assert r.iat[-3] == pytest.approx(1.0) and r.iat[-2] == pytest.approx(0.5) and r.iat[-1] == pytest.approx(3.0)
    reg = volatility_regime(r, VolatilityConfig())
    assert list(reg.iloc[-3:]) == [NORMAL, DEAD, CHAOS] and reg.iat[0] == UNKNOWN


def test_spread_filters():
    t = pd.date_range("2026-11-01", periods=200, freq="5min", tz="UTC")
    sp = np.full(200, 20.0)
    sp[-1] = 40.0
    r = spread_ratio(pd.DataFrame({"time": t, "spread": sp}))
    assert r.iat[-1] == pytest.approx(2.0) and r.iat[-2] == pytest.approx(1.0)
    cfg = VolatilityConfig()
    assert not spread_ok(2.0, None, 0.2, cfg) and spread_ok(1.2, 2.0, 0.2, cfg)
    assert not spread_ok(1.0, 0.5, 0.2, cfg)  # stop smaller than 5 spreads


# ----------------------------------------------------------------- snapshot
def test_snapshot_contents(features):
    s = features.snapshot(T("2026-11-23 13:00"))
    assert s.m5_bar["time"] == T("2026-11-23 12:55")
    assert s.session == "newyork_kz" and s.in_kill_zone and not s.new_signals_blocked
    assert set(s.state) == set(STRUCTURE_TFS) and s.bias in ("BULL", "BEAR", "NONE")
    assert all(v > 0 for v in s.atr.values())
    assert s.dealing_range is not None and s.dealing_range.low < s.dealing_range.high
    assert (s.pools["known_at"] <= s.decision_time).all()
    assert s.pools["taken_at"].isna().all() or (s.pools["taken_at"].dropna() >= s.decision_time).all()
    assert all(sw.known_at <= s.decision_time for sw in s.recent_sweeps)
    assert set(s.pools["kind"]) >= {"previous_day", "asia", "london"}
    assert s.pools.duplicated(["side", "price"]).sum() == 0


def test_htf_snapshot_uses_only_closed_candles(features):
    s = features.snapshot(T("2026-11-23 13:05"))
    h1 = features.bars[Timeframe.H1]
    k = int((pd.DatetimeIndex(h1["time"]) + Timeframe.H1.delta).searchsorted(T("2026-11-23 13:05"), "right")) - 1
    assert h1["time"].iat[k] == T("2026-11-23 12:00")  # the 13:00 H1 candle is still forming
    assert s.atr[Timeframe.H1] == features.atr[Timeframe.H1].iat[k]


def test_snapshot_before_any_data_raises(features):
    with pytest.raises(ValueError):
        features.snapshot(START)


def test_missing_timeframe_rejected(bars):
    from quant_engine.config import StrategyConfig
    with pytest.raises(ValueError):
        MarketFeatures({Timeframe.M5: bars[Timeframe.M5]}, StrategyConfig(), 0.01)


# ------------------------------------------------------------ no look-ahead
def _truncate(bars, t):
    return {tf: b[b["time"] + tf.delta <= t].reset_index(drop=True) for tf, b in bars.items()}


def _comparable(s):
    fvgs = s.recent_fvgs.drop(columns=["invalidated_at"])
    pools = s.pools.drop(columns=["taken_at", "taken_idx"]).reset_index(drop=True)
    return {
        "state": s.state, "bias": (s.bias, s.htf_points), "atr": s.atr, "session": s.session,
        "m15": (s.m15_allows_long, s.m15_allows_short), "blocked": s.new_signals_blocked,
        "dr": s.dealing_range, "pools": pools.to_dict("list"), "sweeps": s.recent_sweeps,
        "fvgs": fvgs.to_dict("list"), "events": s.m5_events.to_dict("list"),
        "vol": (round(s.vol_ratio, 12) if not np.isnan(s.vol_ratio) else None, s.vol_regime),
        "spread": round(s.spread_ratio, 12) if not np.isnan(s.spread_ratio) else None,
    }


@pytest.mark.parametrize("when", [
    "2026-11-02 08:35",   # first London session after the US DST change
    "2026-11-17 01:25",   # inside the Asia range (pool not complete yet)
    "2026-11-20 14:55",   # Friday, last New York kill-zone candle
    "2026-11-23 05:00",   # exactly when the Asia range completes
    "2026-11-23 12:40",   # New York kill zone
    "2026-11-24 10:00",   # last closed candle in the data
])
def test_snapshot_identical_with_future_data_removed(bars, features, when):
    """Recomputing everything on data that ends at ``when`` must give the same snapshot.
    Any feature that peeked at later candles would differ here."""
    t = T(when)
    full = features.snapshot(t)
    from quant_engine.config import StrategyConfig
    part = MarketFeatures(_truncate(bars, t), StrategyConfig(), point=0.01).snapshot(t)
    a, b = _comparable(full), _comparable(part)
    for key in a:
        assert a[key] == b[key], key


def test_structure_state_is_prefix_stable(bars):
    """Per-candle structure state on every timeframe never changes when later candles are added."""
    from quant_engine.features.structure import analyze_structure
    for tf in STRUCTURE_TFS:
        b = bars[tf]
        cut = int(len(b) * 0.6)
        full = analyze_structure(b, tf, 2 if tf is Timeframe.M5 else 3).state
        part = analyze_structure(b.iloc[:cut], tf, 2 if tf is Timeframe.M5 else 3).state
        assert (full.iloc[:cut].to_numpy() == part.to_numpy()).all(), tf


def test_fvgs_and_displacement_prefix_stable(bars):
    from quant_engine.features.imbalance import displacement, find_fvgs
    m5 = bars[Timeframe.M5]
    cut = 5000
    a_full, a_part = atr(m5, 14), atr(m5.iloc[:cut], 14)
    assert (displacement(m5, a_full).iloc[:cut].to_numpy() == displacement(m5.iloc[:cut], a_part).to_numpy()).all()
    f_full = find_fvgs(m5, Timeframe.M5, a_full, 0.01)
    f_part = find_fvgs(m5.iloc[:cut], Timeframe.M5, a_part, 0.01)
    cols = ["c3_idx", "direction", "bottom", "top", "valid_size"]
    pd.testing.assert_frame_equal(f_full[f_full["c3_idx"] < cut][cols].reset_index(drop=True),
                                  f_part[cols].reset_index(drop=True))
