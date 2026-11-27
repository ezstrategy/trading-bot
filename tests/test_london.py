"""London breakout signal mode: same trades as the research code, no look-ahead, live messages."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import LondonConfig, load_config
from quant_engine.live.engine import SignalEngine
from quant_engine.live.state import CLOSED, EngineState
from quant_engine.research.classics import lb_day_groups
from quant_engine.research.events import regime_per_bar
from quant_engine.research.lb_phase2 import ORIGINAL, lb_trades
from quant_engine.research.model import regime_kwargs
from quant_engine.strategy import london as lb
from quant_engine.strategy.regime import trend_regime_series
from quant_engine.timeframes import Timeframe

from .live_helpers import SYMBOL, Clock, Recorder, ReplayCollector
from .synthetic import synthetic_bars

SP = 0.08


@pytest.fixture(scope="module")
def strat():
    return load_config("config", load_env_file=False).strategy


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(days=120, seed=7)


def _research(bars, strat, params):
    m5, h4 = bars[Timeframe.M5], bars[Timeframe.H4]
    reg = regime_per_bar(pd.DatetimeIndex(m5["time"]), h4, trend_regime_series(h4, **regime_kwargs(strat)))
    return lb_trades(m5, SP, params, reg, lb_day_groups(m5, strat.sessions))


def test_default_settings_are_the_tested_rules(strat):
    p = strat.london
    assert strat.mode == "london" and p.is_tested_version()
    assert (p.target_r, p.breakeven_r, p.close_ny, p.trend_filter, p.stop_cap_pct) == \
           (ORIGINAL["target_r"], ORIGINAL["be_r"], "16:00", True, None)
    assert p.buffer_pct / 100 == pytest.approx(ORIGINAL["buffer"])
    assert not LondonConfig(target_r=2.0).is_tested_version()


@pytest.mark.parametrize("cap", [None, 0.6])
def test_same_trades_as_the_research_code(bars, strat, cap):
    """The live code replays history to exactly the EXP-020 (natural stop) / EXP-016 (capped) trades."""
    s = strat.model_copy(deep=True)
    s.london.stop_cap_pct = cap
    mine = lb.london_trades(bars[Timeframe.M5], bars[Timeframe.H4], s, SP)
    ref = _research(bars, strat, {**ORIGINAL, "stop_cap": None if cap is None else cap / 100})
    assert len(mine) == len(ref) > 20
    assert (pd.DatetimeIndex(mine["entry_time"]) == pd.DatetimeIndex(ref["entry_time"])).all()
    assert list(mine["side"]) == list(ref["side"])
    np.testing.assert_allclose(mine["r"], ref["r"])
    np.testing.assert_allclose(mine["stop"], ref["stop"])
    reason = {"session_close": "time", "breakeven_stop": "breakeven", "stop": "stop", "target": "target"}
    assert [reason[x] for x in mine["exit_reason"]] == list(ref["exit_reason"])


def test_no_look_ahead_events_only_grow(bars, strat):
    """Replaying a day candle by candle never changes an event already known (no repainting)."""
    m5, h4 = bars[Timeframe.M5], bars[Timeframe.H4]
    m5 = m5[(m5["time"] >= "2026-07-01") & (m5["time"] < "2026-07-11")].reset_index(drop=True)
    final = {d.setup_time: d for d in lb.days_known_at(m5, h4, strat, SP, m5["time"].iat[-1] + Timeframe.M5.delta)
             if d.setup_time}
    checked = 0
    for now in pd.date_range(m5["time"].iat[0] + Timeframe.M5.delta, m5["time"].iat[-1], freq="5min"):
        for d in lb.days_known_at(m5, h4, strat, SP, now):
            if d.setup_time is None or d.setup_time not in final:
                continue
            assert all(e.time <= now for e in d.events)
            f = final[d.setup_time]
            assert [(e.kind, e.time) for e in d.events] == [(e.kind, e.time) for e in f.events[:len(d.events)]]
            checked += 1
    assert checked > 500


def _engine(tmp_path, bars, start, balance=10_000.0):
    cfg = load_config("config", load_env_file=False)
    cfg.engine.live.state_dir = str(tmp_path / "signals")
    cfg.engine.live.lookback_days = {"M5": 4, "M15": 4, "H1": 10, "H4": 60}
    cfg.strategy.news.enabled = False
    clk = Clock(start)
    rec = Recorder()
    eng = SignalEngine(cfg, ReplayCollector(bars, clk), SYMBOL, rec, now=clk, balance=lambda: balance)
    return eng, clk, rec


def _traded_day(bars, strat):
    tr = lb.london_trades(bars[Timeframe.M5], bars[Timeframe.H4], strat, SP)
    tr = tr[(tr["entry_time"] > "2026-08-03") & (tr["exit_reason"] != "session_close")]
    return tr.iloc[0]


def _run_day(eng, clk, until):
    while clk.now <= until:
        eng.cycle()
        clk.now += pd.Timedelta(minutes=5)


def test_live_setup_then_followups_match_the_backtest(tmp_path, bars, strat):
    row = _traded_day(bars, strat)
    setup = pd.Timestamp(row["entry_time"]).tz_convert("America/New_York").normalize() + pd.Timedelta(hours=1)
    setup = setup.tz_convert("UTC")
    eng, clk, rec = _engine(tmp_path, bars, str((setup - pd.Timedelta(minutes=10)).tz_localize(None)))
    _run_day(eng, clk, pd.Timestamp(row["exit_time"]) + pd.Timedelta(minutes=30))
    texts = [t for t, _ in rec.sent]
    assert "LONDON BREAKOUT SETUP" in texts[0] and "no trade was placed" in texts[0]
    assert ("BUY STOP" in texts[0]) or ("SELL STOP" in texts[0])
    assert "Settings differ" not in texts[0]
    assert any(f"{row['side']} triggered" in t for t in texts[1:])
    assert all(reply == 1 for _, reply in rec.sent[1:])               # follow-ups reply to the setup
    st = EngineState(tmp_path / "signals")
    (sent,) = st.signals.values()
    assert sent.status == CLOSED and sent.r == pytest.approx(row["r"], abs=1e-4)
    assert sent.signal.side == row["side"] and sent.lots > 0
    # restart: nothing is sent again
    eng2, clk2, rec2 = _engine(tmp_path, bars, str(clk.now.tz_localize(None)))
    eng2.cycle()
    assert rec2.sent == []


def test_engine_started_after_the_breakout_does_not_send_late(tmp_path, bars, strat):
    row = _traded_day(bars, strat)
    start = pd.Timestamp(row["entry_time"]) + pd.Timedelta(minutes=15)
    eng, clk, rec = _engine(tmp_path, bars, str(start.tz_localize(None)))
    eng.cycle()
    assert rec.sent == []
    (sent,) = EngineState(tmp_path / "signals").signals.values()
    assert sent.outcome.startswith("missed")


def test_setup_skipped_when_min_lot_risks_too_much(tmp_path, bars, strat):
    row = _traded_day(bars, strat)
    setup = pd.Timestamp(row["entry_time"]).tz_convert("America/New_York").normalize() + pd.Timedelta(hours=1)
    eng, clk, rec = _engine(tmp_path, bars, str(setup.tz_convert("UTC").tz_localize(None)), balance=20.0)
    eng.cycle()
    (text, _), = rec.sent
    assert "London breakout skipped" in text and "never tightened" in text
    (sent,) = EngineState(tmp_path / "signals").signals.values()
    assert sent.status == CLOSED and sent.outcome.startswith("skipped")


def test_backtest_mode_london_uses_the_live_code(tmp_path, bars, strat, cfg):
    from quant_engine.data.storage import ParquetStore
    from quant_engine.strategy.runner import run_backtest
    store = ParquetStore(cfg.data.storage.root)
    for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4):
        store.write_bars("XAUUSD", tf, bars[tf])
    s = run_backtest(cfg, store, "XAUUSD", 0.01, tmp_path / "rep", mode="london")
    n = len(lb.london_trades(bars[Timeframe.M5], bars[Timeframe.H4], cfg.strategy,
                             cfg.strategy.research.costs.min_spread_points * 0.01))
    assert s["overall"]["trades"] == n
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "London breakout" in md and "By H4 trend" in md and "slippage" in md
