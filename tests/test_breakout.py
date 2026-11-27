"""Long-only D1 breakout module: D1 building, levels, state machine, trade simulation, risk, scan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.breakout.backtest import run_backtest, trades_of, write_report
from quant_engine.breakout.config import BreakoutConfig, StrategyParams, load_breakout_config
from quant_engine.breakout.daily import bar_sessions, build_daily, day_close_time, trading_day
from quant_engine.breakout.risk import Exposure, RiskCandidate, RiskEngine
from quant_engine.breakout.rules import channel_high, daily_atr, prev_month_high, strategies
from quant_engine.breakout.scan import BLOCKED, GENERATED, IGNORED, prepare, scan
from quant_engine.breakout.state import (ACTIVE, SIGNAL, WAITING, WAITING_FRESH, BreakoutStateMachine)
from quant_engine.breakout.trade import EXPIRED, NO_FILL, SL, TP, M5Arrays, TradeCosts, simulate_long
from quant_engine.config import ConfigError
from quant_engine.instruments import InstrumentSpec

from .breakout_helpers import flat_days, m5_from_daily, session_times, trading_days

UTC = "UTC"


def spec(**kw) -> InstrumentSpec:
    base = dict(symbol="XAUUSD", contract_size=100, min_lot=0.01, max_lot=100, lot_step=0.01, tick_size=0.01,
                tick_value=1.0, digits=2, commission_per_lot=0.0, typical_spread=0.08)
    return InstrumentSpec(**{**base, **kw})


def monthly_series():
    """Jan 2024 quiet 2000-2010 (high 2010); breakout D1 close 2020 on Tue 6 Feb; stays above all Feb."""
    days = list(pd.bdate_range("2023-12-01", "2024-03-29"))
    ohlc = []
    for d in days:
        if d < pd.Timestamp("2024-02-06"):
            hi = 2010.0 if d == pd.Timestamp("2024-01-17") else 2008.0
            ohlc.append((2004.0, hi, 1996.0, 2005.0))
        elif d == pd.Timestamp("2024-02-06"):
            ohlc.append((2005.0, 2022.0, 2003.0, 2020.0))
        elif d.month == 2:
            ohlc.append((2020.0, 2028.0, 2016.0, 2024.0))
        else:
            ohlc.append((2024.0, 2027.0, 2019.0, 2023.0))
    return days, ohlc


# ---------------------------------------------------------------- daily
def test_trading_day_ny_close_boundary_and_dst():
    t = pd.DatetimeIndex(["2024-03-10 21:55", "2024-03-10 22:00", "2024-07-01 20:55", "2024-07-01 21:00",
                          "2024-01-07 23:00"], tz=UTC)
    d = trading_day(t)
    # Sun 10 Mar 2024 is the US DST switch: 17:00 NY = 21:00 UTC; Sunday-evening bars belong to Monday
    assert list(d.strftime("%a %d")) == ["Mon 11", "Mon 11", "Mon 01", "Tue 02", "Mon 08"]
    ct = day_close_time(pd.DatetimeIndex(["2024-01-08", "2024-07-01"]))
    assert list(ct.strftime("%H:%M")) == ["22:00", "21:00"]


def test_build_daily_aggregates_and_excludes_forming_day():
    days = trading_days("2024-01-08", 3)
    m5 = m5_from_daily(days, [(10, 12, 9, 11), (11, 15, 10, 14), (14, 16, 13, 15)])
    d1 = build_daily(m5)
    assert len(d1) == 3 and d1["n_bars"].eq(276).all() and d1["complete"].all()
    assert d1.iloc[1][["open", "high", "low", "close"]].tolist() == [11, 15, 10, 14]
    part = m5[m5["time"] < pd.Timestamp("2024-01-10 12:00", tz=UTC)]
    d1p = build_daily(part)
    assert list(d1p["day"].dt.day) == [8, 9]            # Wednesday 10th is still forming
    thin = m5.drop(m5.index[300:500])
    assert not build_daily(thin)["complete"].iat[1]


def test_bar_sessions_counts_forming_day_after_final_days():
    days = trading_days("2024-01-08", 2)
    m5 = m5_from_daily(days, [(10, 12, 9, 11), (11, 15, 10, 14)])
    d1 = build_daily(m5.iloc[:400])
    s = bar_sessions(pd.DatetimeIndex(m5["time"].iloc[:400]), d1)
    assert s[0] == 0 and s[-1] == 1 and len(d1) == 1


# ---------------------------------------------------------------- levels
def test_prev_month_high_uses_only_completed_previous_month():
    days, ohlc = monthly_series()
    d1 = build_daily(m5_from_daily(days, ohlc))
    lvl, note = prev_month_high(d1)
    dd = pd.DatetimeIndex(d1["day"])
    assert np.isnan(lvl[0]) and "no data" in note[0]                       # December: nothing before
    assert np.all(lvl[(dd.month == 1)] == 2008.0)                          # December's high
    assert np.all(lvl[(dd.month == 2)] == 2010.0)                          # January's high, not February's
    assert np.all(lvl[(dd.month == 3)] == 2028.0)


def test_prev_month_high_partial_first_month_is_not_used():
    days, ohlc = monthly_series()
    keep = [i for i, d in enumerate(days) if d >= pd.Timestamp("2023-12-15")]
    d1 = build_daily(m5_from_daily([days[i] for i in keep], [ohlc[i] for i in keep]))
    lvl, note = prev_month_high(d1)
    jan = pd.DatetimeIndex(d1["day"]).month == 1
    assert np.isnan(lvl[jan]).all() and "partial first month" in note[jan][0]


def test_levels_have_no_lookahead():
    days, ohlc = flat_days("2023-01-02", 160, drift=0.8, rng=14, seed=3)
    d1 = build_daily(m5_from_daily(days, ohlc))
    full_a, _ = prev_month_high(d1)
    full_b, _ = channel_high(d1, 20)
    full_atr = daily_atr(d1, 14)
    for cut in (40, 77, 120):
        a, _ = prev_month_high(d1.iloc[:cut])
        b, _ = channel_high(d1.iloc[:cut], 20)
        np.testing.assert_array_equal(a, full_a[:cut])
        np.testing.assert_array_equal(b, full_b[:cut])
        np.testing.assert_allclose(daily_atr(d1.iloc[:cut], 14), full_atr[:cut])


def test_channel_high_excludes_current_candle():
    d1 = pd.DataFrame({"high": [1.0, 5, 3, 4, 9, 2]})
    lvl, _ = channel_high(d1, 3)
    assert np.isnan(lvl[:3]).all() and list(lvl[3:]) == [5, 5, 9]


# ---------------------------------------------------------- state machine
def _feed(sm, closes, levels, start="2024-02-01"):
    out = []
    for i, (c, l) in enumerate(zip(closes, levels)):
        d = pd.Timestamp(start) + pd.Timedelta(days=i)
        out.append(sm.on_d1_close(d.tz_localize(UTC), d, c, l))
    return out


def test_state_machine_not_armed_at_start_and_no_repeat_while_above():
    sm = BreakoutStateMachine("close_below_level")
    assert _feed(sm, [110, 111, 112], [100] * 3) == [False] * 3      # already above at data start
    assert sm.state == WAITING_FRESH
    assert _feed(sm, [99, 105, 106, 107], [100] * 4, "2024-02-04") == [False, True, False, False]


def test_state_machine_close_below_rearms_after_position():
    sm = BreakoutStateMachine("close_below_level")
    _feed(sm, [99], [100])
    assert sm.state == WAITING and sm.armed
    assert _feed(sm, [101], [100], "2024-02-02") == [True]
    sm.signal_generated(pd.Timestamp("2024-02-02", tz=UTC))
    sm.entry_filled(pd.Timestamp("2024-02-03", tz=UTC))
    assert sm.state == ACTIVE
    assert _feed(sm, [98, 102], [100, 100], "2024-02-03") == [False, False]   # ignored: position active
    assert any("ignored" in h[3] for h in sm.history)
    sm.position_closed(pd.Timestamp("2024-02-06", tz=UTC), "sl_hit")
    assert sm.state == WAITING_FRESH                                         # the 102 breakout disarmed it
    assert _feed(sm, [99, 103], [100, 100], "2024-02-07") == [False, True]


def test_state_machine_new_month_mode_one_breakout_per_month():
    sm = BreakoutStateMachine("new_month")
    res = _feed(sm, [95, 99, 101, 98, 102], [100] * 5, "2024-01-29")   # 29,30,31 Jan, 1,2 Feb
    assert res == [False, False, False, False, True]                    # armed only from February


# ------------------------------------------------------------ trade sim
def _arrays(days, ohlc, spread_points=10, path=None):
    m5 = m5_from_daily(days, ohlc, spread_points, path)
    d1 = build_daily(m5)
    sess = bar_sessions(pd.DatetimeIndex(m5["time"]), d1)
    return d1, M5Arrays.build(m5, sess, pd.DatetimeIndex(d1["day"]), 0.01)


COSTS = TradeCosts(spread_floor=0.08, slippage=0.02)


def test_entry_next_m5_open_with_spread_and_target():
    days = trading_days("2024-01-08", 3)
    d1, m = _arrays(days, [(100, 101, 99, 100), (100, 100, 100, 100), (100, 140, 100, 140)], spread_points=30)
    after = d1["close_time"].iat[0]
    r = simulate_long(m, after, atr=10, stop_atr_mult=1.5, target_r=2, max_sessions=5, costs=COSTS)
    assert r.entry_time == m.t[276] and r.entry_time >= after
    assert r.spread == pytest.approx(0.30) and r.entry == pytest.approx(100 + 0.30 + 0.02)
    assert r.stop == pytest.approx(r.entry - 15) and r.target == pytest.approx(r.entry + 30)
    assert r.status == TP and r.exit == pytest.approx(r.target - 0.02)
    assert r.r == pytest.approx((r.target - 0.02 - r.entry) / 15)
    assert r.r < 2.0 and r.r_gross > r.r


def test_same_candle_stop_and_target_counts_stop_first():
    days = trading_days("2024-01-08", 2)
    d1, m = _arrays(days, [(100, 101, 99, 100), (100, 100, 100, 100)])
    e = 276
    m.h[e + 3], m.l[e + 3] = 200.0, 50.0                     # one candle touches both
    r = simulate_long(m, d1["close_time"].iat[0], 10, 1.5, 2, 5, COSTS)
    assert r.status == SL and r.exit_idx == e + 3 and "stop counted first" in r.note
    assert r.exit == pytest.approx(r.stop - 0.02)


def test_gap_below_stop_fills_at_open():
    days = trading_days("2024-01-08", 3)
    d1, m = _arrays(days, [(100, 101, 99, 100), (100, 100, 100, 100), (80, 80, 70, 75)])
    r = simulate_long(m, d1["close_time"].iat[0], 10, 1.5, 2, 5, COSTS)
    assert r.status == SL and r.exit == pytest.approx(80 - 0.02) and r.r < -1.3


def test_expiry_at_close_of_fifth_session_and_triple_swap():
    days = trading_days("2024-01-08", 8)                     # Mon 8 .. Wed 17
    ohlc = [(100, 101, 99, 100)] + [(100, 102, 98, 101)] * 7
    d1, m = _arrays(days, ohlc)
    costs = TradeCosts(spread_floor=0.08, slippage=0.02, swap_long_points=-50, swap_triple_weekday=2)
    r = simulate_long(m, d1["close_time"].iat[0], 10, 1.5, 2, 5, costs)   # entry session = Tue 9
    assert r.status == EXPIRED and r.sessions_held == 5
    assert r.exit_time == d1["close_time"].iat[5]          # end of Mon 15 (5th session: Tue..Mon)
    assert r.nights == 4 and r.swap_nights_weighted == 6   # Wed 10 rollover counts 3
    assert r.r_swap == pytest.approx(-50 * 0.01 * 6 / 15)
    assert r.r == pytest.approx(r.r_price + r.r_swap)


def test_spread_above_maximum_blocks_entry_and_sell_is_disabled():
    days = trading_days("2024-01-08", 2)
    d1, m = _arrays(days, [(100, 101, 99, 100), (100, 100, 100, 100)], spread_points=500)
    r = simulate_long(m, d1["close_time"].iat[0], 10, 1.5, 2, 5, COSTS, max_spread=1.0)
    assert r.status == NO_FILL and "spread" in r.note
    with pytest.raises(NotImplementedError):
        simulate_long(m, d1["close_time"].iat[0], 10, 1.5, 2, 5, COSTS, side=-1)


# ------------------------------------------------------------------- risk
def test_risk_uses_broker_tick_value_not_points():
    s = BreakoutConfig().risk
    a = RiskEngine(s, spec()).assess(RiskCandidate("prev_month_high", 1, 2000, 1970), [], 100_000)
    # $500 at 0.5%: 30 price units x $100 per unit per lot = $3,000/lot -> 0.16 lots
    assert a.approved and a.lots == 0.16 and a.risk_pct <= 0.5
    b = RiskEngine(s, spec(contract_size=10, tick_value=0.1)).assess(RiskCandidate("prev_month_high", 1, 2000, 1970), [], 100_000)
    assert b.lots == pytest.approx(1.66)


def test_risk_min_lot_skip_and_allowed_levels():
    eng = RiskEngine(BreakoutConfig().risk, spec())
    d = eng.assess(RiskCandidate("prev_month_high", 1, 2000, 1955), [], 500)
    assert not d.approved and d.reason.startswith("skipped: minimum lot") and "9.00%" in d.reason
    assert not eng.assess(RiskCandidate("prev_month_high", 1, 2000, 1990), [], 1e6, risk_pct=2.0).approved


def test_risk_one_position_per_strategy_correlation_and_combined_cap():
    s = BreakoutConfig().risk
    eng = RiskEngine(s, spec())
    c = RiskCandidate("channel_20d", 100, 2000, 1970)
    assert "already has" in eng.assess(c, [Exposure("channel_20d", 50, 0.5)], 1e5).reason
    corr = eng.assess(c, [Exposure("prev_month_high", 100, 0.5)], 1e5)
    assert not corr.approved and "not doubled" in corr.reason
    far = eng.assess(c, [Exposure("prev_month_high", 90, 0.5)], 1e5)
    assert far.approved
    s2 = s.model_copy(update={"max_combined_open_pct": 0.5})
    assert "combined" in RiskEngine(s2, spec()).assess(c, [Exposure("prev_month_high", 90, 0.5)], 1e5).reason


def test_risk_split_policy_shares_one_budget_same_day():
    s = BreakoutConfig().risk.model_copy(update={"correlated_policy": "split"})
    eng = RiskEngine(s, spec())
    ds = eng.batch([RiskCandidate("prev_month_high", 5, 2000, 1970), RiskCandidate("channel_20d", 5, 2000, 1970)], [], 1e5)
    assert all(d.approved for d in ds) and sum(d.risk_pct for d in ds) <= 0.5 + 1e-9


def test_risk_primary_only_batch_keeps_primary():
    eng = RiskEngine(BreakoutConfig().risk, spec())
    ds = eng.batch([RiskCandidate("channel_20d", 5, 2000, 1970), RiskCandidate("prev_month_high", 5, 2000, 1970)], [], 1e5)
    assert not ds[0].approved and ds[1].approved and "prev_month_high" in ds[0].reason


# -------------------------------------------------------------- config
def test_config_defaults_and_review_gate(tmp_path):
    c = load_breakout_config("config")
    assert c.mode == "paper" and c.risk.per_trade_pct == 0.5 and c.risk.max_combined_open_pct == 1.0
    assert c.strategies.prev_month_high.reset_mode == "new_month"
    with pytest.raises(Exception):
        BreakoutConfig(mode="live")
    assert BreakoutConfig(mode="live", live_review_approved_by="owner 2026-12-01").mode == "live"
    with pytest.raises(Exception):
        BreakoutConfig(risk={"per_trade_pct": 2.0})
    (tmp_path / "breakout.yaml").write_text("mode: live\n")
    with pytest.raises(ConfigError):
        load_breakout_config(tmp_path)


# ------------------------------------------------------------------ scan
def test_scan_prev_month_high_signal_once_and_explained():
    days, ohlc = monthly_series()
    m5 = m5_from_daily(days, ohlc)
    cfg = BreakoutConfig()
    d1, m = prepare(m5, cfg)
    a = strategies(cfg)[0]
    res = scan(a, d1, m, cfg, COSTS)
    sig = res.signals
    assert [e.day.strftime("%Y-%m-%d") for e in sig] == ["2024-02-06"]
    e = sig[0]
    assert e.level == 2010.0 and e.d1_close == 2020.0 and e.key == "prev_month_high:1.0.0:2024-02-06"
    assert e.trade.entry_time == pd.Timestamp("2024-02-06 23:00", tz=UTC)     # 18:00 NY reopen
    assert e.trade.stop == pytest.approx(e.trade.entry - 1.5 * e.atr)
    assert "above the highest high of January 2024" in e.explanation
    assert all(c["ok"] for c in e.checks)


def test_scan_signals_do_not_change_when_future_data_is_added():
    days, ohlc = flat_days("2023-01-02", 260, drift=1.2, rng=16, seed=11)
    m5 = m5_from_daily(days, ohlc)
    cfg = BreakoutConfig()
    full = {s.id: scan(s, *prepare(m5, cfg), cfg, COSTS) for s in strategies(cfg)}
    cut = (pd.Timestamp(days[180]) + pd.Timedelta(hours=23)).tz_localize(UTC)
    for s in strategies(cfg):
        d1, m = prepare(m5, cfg, as_of=cut)
        part = scan(s, d1, m, cfg, COSTS, as_of=cut)
        keys_full = [e.key for e in full[s.id].events if e.signal_time <= cut]
        assert [e.key for e in part.events] == keys_full
        for e in part.signals:
            f = next(x for x in full[s.id].events if x.key == e.key)
            assert e.level == f.level and e.atr == pytest.approx(f.atr)
            if e.trade.is_final:
                assert e.trade.r == pytest.approx(f.trade.r)


def test_scan_blocks_incomplete_breakout_day():
    days, ohlc = monthly_series()
    m5 = m5_from_daily(days, ohlc)
    t = pd.DatetimeIndex(m5["time"])
    bo = pd.Timestamp("2024-02-06")
    drop = (t >= pd.Timestamp("2024-02-06 00:00", tz=UTC)) & (t < pd.Timestamp("2024-02-06 18:00", tz=UTC))
    cfg = BreakoutConfig()
    d1, m = prepare(m5[~drop], cfg)
    res = scan(strategies(cfg)[0], d1, m, cfg, COSTS)
    ev = [e for e in res.events if e.day == bo]
    assert ev and ev[0].kind == BLOCKED and "missing data" in ev[0].failed_checks[0]
    assert not res.signals


def test_channel_strategy_ignores_breakouts_during_position():
    days, ohlc = flat_days("2023-01-02", 300, drift=2.0, rng=12, seed=5)
    cfg = BreakoutConfig()
    d1, m = prepare(m5_from_daily(days, ohlc), cfg)
    res = scan(strategies(cfg)[1], d1, m, cfg, COSTS)
    sig = res.signals
    assert sig
    for a, b in zip(sig, sig[1:]):                         # one position at a time
        assert b.signal_time >= a.trade.exit_time
    assert len({e.key for e in res.events}) == len(res.events)


def test_backtest_run_and_report(tmp_path):
    days, ohlc = flat_days("2021-01-04", 700, drift=0.6, rng=18, seed=2)
    m5 = m5_from_daily(days, ohlc)
    cfg = BreakoutConfig()
    run = run_backtest(m5, cfg, spec())
    per = run.results["per_strategy"]
    assert set(per) == {"prev_month_high", "channel_20d"}
    for p in per.values():
        assert p["full_history"]["trades"] >= 1
        assert {"win_rate_pct", "profit_factor", "expectancy_r", "net_r", "max_drawdown_r", "max_losing_streak",
                "avg_hold_hours", "longest_recovery"} <= set(p["full_history"])
        assert all(w["chosen"] == "spec" for w in p["walk_forward"])
        assert "swap_stress" in p
    assert "NOT INCLUDED" in run.swap_source
    s500 = run.results["sized"]["full_history_500"]["summary"]
    assert s500["signals"] == s500["taken"] + s500["skipped"]
    out = write_report(run, tmp_path / "rep")
    md = (out / "report.md").read_text()
    assert "Swap: NOT INCLUDED" in md and "stop counted first" in md
    assert (out / "results.json").exists() and (out / "events_prev_month_high.csv").exists()
    tr = trades_of(run.scans["prev_month_high"])
    assert (pd.to_datetime(tr["entry_time"]) >= pd.to_datetime(tr["signal_time"])).all()
