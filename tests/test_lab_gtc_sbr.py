"""Quant Lab Phase 4: GTC-SBR bullish / bearish plug-ins."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest.engine import ExecutionConfig, run_backtest
from quant_engine.backtest.lookahead import check_no_lookahead
from quant_engine.backtest.market import MarketData
from quant_engine.config import CostsConfig
from quant_engine.instruments import load_instruments
from quant_engine.research.gtc_sbr import DEFAULTS, find_trades
from quant_engine.strategies import available, create, get
from quant_engine.timeframes import Timeframe

from .conftest import REPO
from .synthetic import synthetic_bars, synthetic_m1
from .test_gtc_sbr import _day, _flat15

SPEC = load_instruments(REPO / "config").get()
NOSLIP = ExecutionConfig(slippage_points=0, use_m1=False)


@pytest.fixture(scope="module")
def synth():
    return synthetic_bars(days=200, seed=3)


def _md(b):
    return MarketData({Timeframe.M5: b[Timeframe.M5], Timeframe.M15: b[Timeframe.M15], Timeframe.H4: b[Timeframe.H4]}, 0.01)


def test_registered_with_ids_versions_and_rules():
    assert {"gtc_sbr_bull", "gtc_sbr_bear"} <= set(available())
    for sid, d in (("gtc_sbr_bull", "long"), ("gtc_sbr_bear", "short")):
        s = create(sid)
        info = s.describe()
        assert info["version"] == "1.0.0" and info["direction"] == d and info["family"] == "gtc_sbr"
        for k in ("entry", "exit", "stop_loss", "take_profit", "sessions", "regime", "risk"):
            assert info[k]
        assert info["params"]["target_r"] == 1.5 and len(info["fingerprint"]) == 16
    assert create("gtc_sbr_bull", {"target_r": 2}).fingerprint() != create("gtc_sbr_bull").fingerprint()
    with pytest.raises(Exception):
        create("gtc_sbr_bull", {"no_such_param": 1})
    with pytest.raises(KeyError):
        get("nope")


def test_hand_built_bull_day():
    m5 = _day()
    m15 = _flat15(m5)
    data = MarketData({Timeframe.M5: m5, Timeframe.M15: m15}, 0.01)
    p = {"trend_filter": False, "news_ny_minutes": []}
    res = run_backtest(data, [create("gtc_sbr_bull", p), create("gtc_sbr_bear", p)], SPEC, NOSLIP)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["strategy_id"] == "gtc_sbr_bull" and t["side"] == "BUY" and t["session"] == "london"
    assert t["entry_time"] == pd.Timestamp("2026-01-05 08:50", tz="UTC")
    assert t["entry"] == pytest.approx(2007.08) and t["exit_reason"] == "target" and t["r"] == pytest.approx(1.5)
    assert res.skips["gtc_sbr_bear"].get("rule: first breakout in the other direction") == 1


def test_same_trades_as_the_existing_gtc_sbr_code(synth):
    old, _ = find_trades(synth[Timeframe.M5], synth[Timeframe.M15], synth[Timeframe.H4], DEFAULTS, 0.01,
                         CostsConfig(min_spread_points=8, slippage_points=0))
    res = run_backtest(_md(synth), [create("gtc_sbr_bull"), create("gtc_sbr_bear")], SPEC, NOSLIP)
    new = res.trades
    assert len(old) > 20 and len(new) == len(old)
    m = old.merge(new, on="entry_time", suffixes=("_old", "_new"))
    assert len(m) == len(old)
    assert (m["side_old"] == m["side_new"]).all() and (m["exit_how"] == m["exit_reason"]).all()
    assert np.allclose(m["entry_old"], m["entry_new"], atol=0.006)
    assert np.allclose(m["r_old"], m["r_new"], atol=1e-6)                   # spread-only R, no slippage / commission
    assert set(new.loc[new["side"] == "BUY", "strategy_id"]) == {"gtc_sbr_bull"}
    assert set(new.loc[new["side"] == "SELL", "strategy_id"]) == {"gtc_sbr_bear"}


def test_no_lookahead_on_both_plugins(synth):
    data = _md(synth)
    for sid in ("gtc_sbr_bull", "gtc_sbr_bear"):
        rep = check_no_lookahead(lambda: create(sid), data, cuts=6, seed=2)
        assert rep["ok"], rep["differences"]
        assert rep["intents_full"] > 5


def test_regime_filter_option(synth):
    data = _md(synth)
    res = run_backtest(data, [create("gtc_sbr_bull", {"regime_filter": ["bullish"]})], SPEC, NOSLIP)
    assert set(res.trades["regime"]) <= {"bullish"}
    assert res.skips["gtc_sbr_bull"].get("blocked: regime", 0) > 0


def test_runs_with_m1_intrabar_and_costs():
    m1 = synthetic_m1(days=60, seed=3)
    from quant_engine.data.dukascopy import resample
    b = {tf: resample(m1, tf) for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H4)}
    data = MarketData({**b, Timeframe.M1: m1}, 0.01)
    res = run_backtest(data, [create("gtc_sbr_bull"), create("gtc_sbr_bear")], SPEC, ExecutionConfig())
    if len(res.trades):
        assert (res.trades["r"] <= res.trades["r_gross"] + 1e-12).all()
        assert res.trades["intrabar"].isin(["single", "m1", "m1_stop_first", "stop_first", "close"]).all()
    assert res.data["m1_available"] and res.execution["slippage_points"] == 2
