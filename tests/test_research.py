"""Pattern miner: trade outcomes, events without look-ahead, the multiple-testing guard,
a planted pattern that must be found, pure noise that must NOT produce patterns, the
pattern strategy in the normal backtester, and the CLI. Synthetic data only: these
tests check the method, not how gold behaves."""
import json

import numpy as np
import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.config import StrategyConfig
from quant_engine.data.storage import ParquetStore
from quant_engine.features.context import MarketFeatures
from quant_engine.research.events import build_events
from quant_engine.research.miner import Pattern, benjamini_hochberg, mine, non_overlapping, t_stats
from quant_engine.research.model import PatternModel, load_patterns, regime_kwargs
from quant_engine.research.outcomes import trade_outcomes
from quant_engine.strategy.backtest import Backtester
from quant_engine.strategy.modes import build_model
from quant_engine.timeframes import Timeframe

from .synthetic import TFS, synthetic_bars

CFG = StrategyConfig()


def _events(bars, cfg=CFG):
    return build_events(bars[Timeframe.M5], bars[Timeframe.H4], cfg.research, cfg.sessions, 0.01,
                        cfg.backtest.assumed_spread_points, regime_kwargs(cfg))


@pytest.fixture(scope="module")
def planted():
    """250 days where price rises ~4 USD in the first 30 minutes of 13:00 UTC every day."""
    return synthetic_bars(days=250, seed=3, plant_hour=13, plant_move=4.0)


@pytest.fixture(scope="module")
def planted_result(planted):
    return _events(planted), mine(_events(planted), CFG.research)


# ------------------------------------------------------------------ outcomes
def _bars(rows, start="2026-06-01 10:00"):
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="5min")
    o, h, l, c = (np.array([r[i] for r in rows], dtype=float) for i in range(4))
    return pd.DatetimeIndex(t), o, h, l, c


def test_long_hits_target_and_pays_spread():
    t, o, h, l, c = _bars([(100, 100, 100, 100), (100, 100.5, 99.8, 100.4), (100.4, 102.5, 100.3, 102)] + [(102, 102, 102, 102)] * 3)
    sp, a = np.full(len(c), 0.2), np.full(len(c), 1.0)
    r, ex = trade_outcomes(t, o, h, l, c, sp, a, 1, 1.0, 2.0, 3)
    assert r[0] == 2.0 and ex[0] == 2          # entry 100.2 (ask), target 102.2 hit on candle 2


def test_stop_checked_before_target_on_the_same_candle():
    t, o, h, l, c = _bars([(100, 100, 100, 100), (100, 103, 98, 100)] + [(100, 100, 100, 100)] * 3)
    r, ex = trade_outcomes(t, o, h, l, c, np.zeros(len(c)), np.ones(len(c)), 1, 1.0, 1.0, 3)
    assert r[0] == -1.0 and ex[0] == 1


def test_short_time_exit_pays_spread_and_gap_is_skipped():
    t, o, h, l, c = _bars([(100, 100, 100, 100), (100, 100.1, 99.7, 99.8), (99.8, 99.9, 99.6, 99.7)] + [(99.7,) * 4] * 3)
    sp = np.full(len(c), 0.1)
    r, _ = trade_outcomes(t, o, h, l, c, sp, np.ones(len(c)), -1, 1.0, 5.0, 2)
    assert r[0] == pytest.approx((100 - (99.7 + 0.1)) / 1.0)     # closed at the ask after 2 candles
    gap = t.insert(len(t), t[-1] + pd.Timedelta(days=2))[1:]       # weekend between candles
    r2, ex2 = trade_outcomes(gap, o, h, l, c, sp, np.ones(len(c)), -1, 1.0, 5.0, 2)
    assert np.isnan(r2[len(c) - 3]) and ex2[len(c) - 3] == -1


# ------------------------------------------------------------------ statistics
def test_benjamini_hochberg_matches_hand_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8
    assert benjamini_hochberg(p, 0.2).sum() == 7


def test_t_stats_and_non_overlap():
    st = t_stats(np.array([1.0, -1.0, 1.0, 1.0]))
    assert st["trades"] == 4 and st["avg_r"] == 0.5 and 0 < st["p"] < 0.5
    ex = np.array([3, 4, 5, 6, 9, -1, 7, 8, 9, 10, 11])
    assert non_overlapping(np.array([0, 1, 4, 5, 7]), ex).tolist() == [0, 4]


def test_pattern_description_says_continuation_or_reversal():
    p = Pattern("swept 2h low", "neutral", "london_kz", 1, 1.0, 2.0, 12)
    assert "reversal" in p.describe(-1) and "London open" in p.describe(-1) and "BUY" in p.describe(1)


# ------------------------------------------------------------------ events
def test_events_use_no_future_candles(planted):
    bars = {tf: b for tf, b in planted.items()}
    cut = bars[Timeframe.M5]["time"].iat[len(bars[Timeframe.M5]) // 2]
    prefix = {tf: b[b["time"] + tf.delta <= cut + Timeframe.M5.delta].reset_index(drop=True) for tf, b in bars.items()}
    full, part = _events(bars), _events(prefix)
    n = len(part.c)
    for name, mask in part.events.items():
        assert (full.events[name][:n] == mask).all(), name
    assert (full.regime[:n] == part.regime).all()


def test_every_event_family_fires(planted):
    et = _events(planted)
    for name in ("swept 2h low", "swept 6h high", "broke 2h high", "swept prev-day low", "impulse up",
                 "big move down", "range top touch", "range bottom touch", "hour 13:00 UTC"):
        assert et.events[name].sum() > 0, name
    assert set(np.unique(et.regime)) <= {"bullish", "bearish", "neutral"}


# ------------------------------------------------------------------ mining
def test_planted_pattern_is_found_and_confirmed_on_unseen_data(planted_result):
    _, res = planted_result
    s = res.survivors
    hit = s[(s["event"] == "hour 13:00 UTC") & (s["side"] == "BUY")]
    assert len(hit) >= 1
    assert (hit["verdict"] == "confirmed").any()
    assert (hit["val_avg_r"] > 0).all()
    assert res.n_tests > 1000                       # many candidates were tried
    # no pattern is "found" in a market by SELLING into the planted rise
    assert not ((s["event"] == "hour 13:00 UTC") & (s["side"] == "SELL")).any()


def test_pure_noise_gives_no_confirmed_patterns():
    noise = synthetic_bars(days=250, seed=11, drift=0.0)
    res = mine(_events(noise), CFG.research)
    assert res.n_tests > 1000
    confirmed = res.survivors[res.survivors["verdict"] == "confirmed"] if len(res.survivors) else res.survivors
    assert len(confirmed) == 0
    # without the multiple-testing guard plenty of candidates would have looked "significant"
    assert (res.tested["p"] < 0.05).sum() > 0


def test_discovery_never_uses_unseen_data(planted_result):
    et, res = planted_result
    split = int(len(et.c) * CFG.research.discover_frac)
    assert res.split_time == et.times[split]
    # same discovery numbers when the unseen part is replaced by garbage
    et2 = type(et)(**{**et.__dict__})
    for k in ("o", "h", "l", "c"):
        arr = getattr(et, k).copy()
        arr[split + 1:] = arr[split + 1:] * 0 + arr[split]
        setattr(et2, k, arr)
    res2 = mine(et2, CFG.research)
    a = res.tested.set_index(["event", "regime", "session", "direction", "sl_atr", "tp_atr", "horizon"])
    b = res2.tested.set_index(["event", "regime", "session", "direction", "sl_atr", "tp_atr", "horizon"])
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common, "avg_r"], b.loc[common, "avg_r"])


# ------------------------------------------------------------------ pattern strategy
def test_pattern_model_trades_only_unseen_period_and_wins_on_planted_data(planted, planted_result):
    _, res = planted_result
    pats = json.loads(res.survivors.to_json(orient="records"))
    f = MarketFeatures(planted, CFG, 0.01)
    model = PatternModel(f, CFG, None, patterns=pats, start=res.split_time)
    trades = Backtester(f, CFG, None, model=model).run()
    filled = trades[trades["filled"]]
    assert len(filled) >= 10
    assert (pd.to_datetime(trades["signal_time"]) >= res.split_time).all()
    assert (trades["strategy"] == "patterns").all()
    assert filled["r"].mean() > 0


def test_build_model_patterns_needs_a_file(tmp_path):
    cfg = StrategyConfig(research={"patterns_file": str(tmp_path / "none.json")})
    f = MarketFeatures(synthetic_bars(days=15, seed=2), cfg, 0.01)
    with pytest.raises(ValueError, match="quant_engine patterns"):
        build_model("patterns", f, cfg)
    (tmp_path / "p.json").write_text(json.dumps({"patterns": [
        {"event": "hour 13:00 UTC", "regime": "neutral", "session": "any", "direction": 1, "sl_atr": 1.0,
         "tp_atr": 1.0, "horizon": 12, "verdict": "failed"}]}))
    assert load_patterns(tmp_path / "p.json", ["confirmed"]) == []
    assert len(load_patterns(tmp_path / "p.json")) == 1


# ------------------------------------------------------------------ CLI
def test_patterns_cli_writes_plain_report(config_dir, tmp_path, monkeypatch, capsys, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "rep"
    assert cli.run(["--config", str(config_dir), "patterns", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    assert "In plain words" in md and "hour 13:00 UTC" in md and "18:30 IST" in md
    assert "Whole method on unseen data" in md
    for name in ("all_candidates.csv", "patterns.csv", "patterns.json", "holdout_backtest.md", "holdout_trades.csv"):
        assert (rep / name).exists(), name
    saved = json.loads((tmp_path / "reports" / "patterns_latest.json").read_text())
    assert any(p["verdict"] == "confirmed" for p in saved["patterns"])
    assert "Report:" in capsys.readouterr().out


def test_patterns_cli_without_data_explains_what_to_run(config_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.run(["--config", str(config_dir), "patterns"]) == 1
    assert "quant_engine download" in capsys.readouterr().err


# ------------------------------------------------------------------ levels, price action, news
def _m5(rows, start="2026-06-02 12:00"):
    """M5 bars from (open, high, low, close) rows, preceded by 30 quiet candles for the ATR."""
    quiet = [(4010.0, 4010.5, 4009.5, 4010.0)] * 30
    rows = quiet + rows
    t = pd.date_range(pd.Timestamp(start, tz="UTC") - pd.Timedelta(minutes=5 * 30), periods=len(rows), freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    df["tick_volume"], df["spread"], df["real_volume"] = 50.0, 30.0, 0.0
    return df


def _h4_for(m5):
    from quant_engine.data.dukascopy import resample
    m1 = m5.copy()
    return resample(m1, Timeframe.H4)


def _ev(m5, **rc):
    cfg = StrategyConfig(research=rc)
    return build_events(m5, _h4_for(m5), cfg.research, cfg.sessions, 0.01, 30, regime_kwargs(cfg))


def test_round_number_rejection_break_and_pin():
    m5 = _m5([(4010, 4011, 3998, 4009),     # wicks below 4000, closes back above: support + bullish pin
              (4009, 4040, 4008, 4038),     # nothing at a level
              (4038, 4052, 4037, 4049),     # wick above 4050, closes below: resistance
              (4049, 4056, 4048, 4054)])    # close crosses 4050: break up
    et = _ev(m5)
    k = 30
    assert et.events["round 50 support"][k] and et.events["round 100 support"][k]
    assert et.events["bullish pin at round"][k]
    assert et.events["round 50 resistance"][k + 2] and not et.events["round 100 resistance"][k + 2]
    assert et.events["round 50 break up"][k + 3] and not et.events["round 100 break up"][k + 3]
    assert not et.events["round 50 support"][k + 1]
    assert et.natural["round 50 support"] == 1 and et.natural["bearish pin at round"] == -1


def test_news_spike_only_at_release_time_and_fixed_spread():
    # 2026-06-05 is the first Friday of June: 08:30 New York = 12:30 UTC (summer)
    m5 = _m5([(4010, 4030, 4009, 4028), (4028, 4029, 4027, 4028)], start="2026-06-05 12:30")
    et = _ev(m5, fixed_spread_points=20)
    k = 30
    assert et.events["news spike up"][k] and et.events["NFP spike up"][k]
    assert not et.events["news spike up"][k + 1]
    assert np.allclose(et.spread, 0.20)
    m5b = _m5([(4010, 4030, 4009, 4028)], start="2026-06-05 12:35")     # same candle 5 minutes later
    assert not _ev(m5b).events["news spike up"][30]


def test_report_has_family_luck_table(planted_result):
    from quant_engine.research.runner import family_summary
    _, res = planted_result
    fam = family_summary(res.tested)
    assert {"time of day", "round numbers", "major support / resistance"} <= set(fam["family"])
    assert fam["candidates"].sum() == len(res.tested)
    assert np.allclose(fam["expected_by_luck"], (fam["candidates"] * 0.05).round(1))
    assert fam.set_index("family").loc["time of day", "p_below_0.05"] >= 1     # the planted hour


# ------------------------------------------------------------------ H1 / H4 swing versions
def _repo_cfg():
    import yaml
    from pathlib import Path
    return StrategyConfig(**yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "strategy.yaml").read_text()))


def test_timeframe_overrides_apply():
    rc = _repo_cfg().research
    h1 = rc.for_timeframe("H1")
    assert h1.horizons == [8, 24] and 3.0 in h1.tp_atr and h1.fixed_spread_points == rc.fixed_spread_points
    assert rc.for_timeframe("M5") is rc


def test_h1_events_use_no_future_candles(planted):
    cfg = _repo_cfg()
    rc = cfg.research.for_timeframe("H1")
    h1, h4 = planted[Timeframe.H1], planted[Timeframe.H4]
    cut = h1["time"].iat[len(h1) // 2] + Timeframe.H1.delta
    full = build_events(h1, h4, rc, cfg.sessions, 0.01, 30, regime_kwargs(cfg), Timeframe.H1)
    part = build_events(h1[h1["time"] + Timeframe.H1.delta <= cut], h4[h4["time"] + Timeframe.H4.delta <= cut],
                        rc, cfg.sessions, 0.01, 30, regime_kwargs(cfg), Timeframe.H1)
    n = len(part.c)
    for name, mask in part.events.items():
        assert (full.events[name][:n] == mask).all(), name
    assert "swept 24h low" in full.events and full.tf == Timeframe.H1


def test_h1_patterns_found_and_traded_on_m5_at_h1_closes(planted):
    cfg = _repo_cfg()
    rc = cfg.research.for_timeframe("H1")
    et = build_events(planted[Timeframe.H1], planted[Timeframe.H4], rc, cfg.sessions, 0.01, 30,
                      regime_kwargs(cfg), Timeframe.H1)
    res = mine(et, rc)
    assert (res.survivors["verdict"] == "confirmed").any()
    assert (res.tested["timeframe"] == "H1").all()
    pats = json.loads(res.survivors.to_json(orient="records"))
    f = MarketFeatures(planted, cfg, 0.01)
    model = PatternModel(f, cfg, None, patterns=pats, start=res.split_time, timeframe="H1")
    trades = Backtester(f, cfg, None, model=model).run()
    filled = trades[trades["filled"]]
    assert len(filled) >= 10 and filled["r"].mean() > 0
    st = pd.to_datetime(trades["signal_time"])
    assert (st.dt.minute == 0).all() and (st >= res.split_time).all()   # sent when the H1 candle closes


def test_patterns_cli_h4(config_dir, tmp_path, monkeypatch, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "rep4"
    assert cli.run(["--config", str(config_dir), "patterns", "--timeframe", "H4", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    assert "pattern timeframe**: H4 (swing trades" in md and "H4 candles" in md
    assert json.loads((rep / "patterns.json").read_text())["timeframe"] == "H4"


# ------------------------------------------------------------------ price action
def test_engulfing_and_inside_bar_break():
    m5 = _m5([(4012, 4013, 4009, 4010),     # bearish candle
              (4009.8, 4014, 4009.5, 4013),  # bullish engulfing
              (4013, 4020, 4005, 4015),      # mother bar
              (4015, 4018, 4008, 4012),      # inside bar
              (4012, 4022, 4011, 4021)])     # closes above the mother bar
    et = _ev(m5)
    k = 30
    assert et.events["bullish engulfing"][k + 1] and not et.events["bearish engulfing"][k + 1]
    assert et.events["inside bar break up"][k + 4] and not et.events["inside bar break up"][k + 3]
    assert et.natural["inside bar break down at resistance"] == -1


def test_only_price_action_families_are_tested(planted):
    from quant_engine.research.events import PRICE_ACTION_FAMILIES, event_family
    rc = CFG.research.model_copy(update={"only_families": list(PRICE_ACTION_FAMILIES), "min_trades_discover": 10})
    res = mine(_events(planted), rc)
    assert res.n_tests > 0 and set(res.tested["family"]) <= set(PRICE_ACTION_FAMILIES)
    assert event_family("bullish engulfing") == "price action"
    assert event_family("bearish pin at resistance") == "price action at levels"


def test_patterns_cli_price_action_since_date(config_dir, tmp_path, monkeypatch, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "rep_pa"
    assert cli.run(["--config", str(config_dir), "patterns", "--timeframe", "H1", "--price-action",
                    "--start", "2026-08-01", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    assert "idea families tested**: price action, price action at levels" in md
    assert "period (UTC)**: 2026-08-0" in md      # 2026-08-01 is a Saturday


def test_smc_fvg_and_order_block_retests():
    m5 = _m5([(4010, 4010.5, 4009.5, 4009.8),   # small bearish candle = order block
              (4009.8, 4016, 4009.7, 4015.8),   # impulse up (creates the FVG with the next candle)
              (4015.8, 4018, 4012, 4017),       # low 4012 > high 4010.5 two candles back: bullish FVG
              (4017, 4017.5, 4011, 4013),       # taps into the gap, closes above its bottom: FVG retest
              (4013, 4013.5, 4009.9, 4012)])    # back into the order block and holds: OB retest
    et = _ev(m5)
    k = 30
    assert et.events["bullish FVG retest"][k + 3] and not et.events["bullish FVG retest"][k + 4]
    assert et.events["bullish OB retest"][k + 4]


def test_smc_asia_sweep_after_asia_only(planted):
    from quant_engine.research.events import event_family
    et = _events(planted)
    ny = et.times.tz_convert("America/New_York")
    hit = et.events["swept Asia low"]
    assert hit.sum() > 0
    m = np.asarray(ny.hour * 60 + ny.minute)[hit]
    assert ((m < 17 * 60)).all()                                    # after Asia (00:00-17:00 New York)
    day = pd.Series(hit).groupby(np.asarray(et.times.tz_convert("America/New_York").date)).sum()
    assert day.max() <= 2                                           # first sweep per trading day
    assert event_family("swept Asia low") == event_family("bullish FVG retest") == "smc"


# ------------------------------------------------------------------ big fast moves + simple setups
def test_sweep_rejection_and_break_retest_setups():
    m5 = _m5([(4003, 4004, 3998, 4003.5),    # sweeps below 4000 and closes at the top: rejection at support
              (4003.5, 4012, 4003, 4011),
              (4011, 4052, 4010, 4051),       # closes above 4050: breaks a round number
              (4051, 4053, 4049.9, 4052)])    # back to 4050 and closes above: break + retest
    et = _ev(m5)
    k = 30
    assert et.events["sweep + rejection at support"][k]
    assert et.events["break + retest of resistance"][k + 3]
    assert not et.events["break + retest of resistance"][k + 1]


def test_forward_moves_measure_from_next_open():
    from quant_engine.research.bigmoves import forward_moves
    m5 = _m5([(4010, 4011, 4009, 4010), (4010, 4014, 4009, 4013), (4013, 4013.5, 4006, 4007)] + [(4007,) * 4] * 8)
    et = _ev(m5)
    up, dn = forward_moves(et, 6)
    k = 30
    assert up[k] == pytest.approx(4014 - 4010) and dn[k] == pytest.approx(4010 - 4006)


def test_bigmoves_cli_reports_win_rate_and_breakeven(config_dir, tmp_path, monkeypatch, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "bm"
    assert cli.run(["--config", str(config_dir), "bigmoves", "--start", "2026-06-01", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    assert "1 pip = $0.1" in md and "breakeven_win_pct" in md and "What came before a 30-pip move" in md
    assert "| 2026 |" in md
    cand = pd.read_csv(rep / "strategy_candidates.csv")
    assert set(cand["exit_units"]) == {"usd"} and set(cand["horizon"]) == {6}
    assert set(cand["family"]) == {"S/R + price action + liquidity"}


def test_bigmoves_small_targets_and_exit_summary(config_dir, tmp_path, monkeypatch, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "bm10"
    assert cli.run(["--config", str(config_dir), "bigmoves", "--start", "2026-06-01", "--targets", "10,20",
                    "--stops", "5,10", "--windows", "6,12", "--move-pips", "10,20", "--spread-points", "8",
                    "--out", str(rep)]) == 0
    cand = pd.read_csv(rep / "strategy_candidates.csv")
    assert set(cand["tp_atr"].round(2)) == {1.0, 2.0} and set(cand["sl_atr"].round(2)) == {0.5, 1.0}
    assert set(cand["horizon"]) == {6, 12}
    md = (rep / "report.md").read_text()
    assert "Every target / stop / time limit" in md and "| 20 | 10 | 60 |" in md
    assert "spread charged**: $0.08" in md
    assert "30 pips" not in md.split("## 1.")[0] or "10 pips = $1" in md


# ------------------------------------------------------------------ the owner's entry
def _entry_bars(tail):
    """``_m5`` adds 30 quiet candles first, so ``tail[0]`` is candle 30."""
    return _m5(tail + [(4012, 4012.5, 4011.5, 4012)] * 10)


ENTRY_P = {**__import__("quant_engine.research.entries", fromlist=["DEFAULTS"]).DEFAULTS, "min_risk_atr": 0.0}


def test_entry_buy_fires_on_sweep_then_green_close_above_sweep_high():
    from quant_engine.research.entries import find_signals
    # 30: sweep below 4009 (low 4005, high 4009.5); 31: red, no trigger; 32: green close 4010 > 4009.5
    m5 = _entry_bars([(4009, 4009.5, 4005, 4006), (4006, 4007, 4005.5, 4006.5), (4006.5, 4010.5, 4006, 4010)])
    s = find_signals(m5, ENTRY_P, 0.08)
    assert list(np.nonzero(s.signal[1])[0]) == [32]
    assert s.sweep_idx[1][32] == 30
    assert s.stop[1][32] < 4005 and s.risk[1][32] == pytest.approx(4010.08 - s.stop[1][32])
    assert not s.signal[-1][:33].any()


def test_entry_no_signal_if_low_breaks_or_too_late():
    from quant_engine.research.entries import find_signals
    broke = _entry_bars([(4009, 4009.5, 4005, 4006), (4006, 4006.5, 4004, 4004.5), (4004.5, 4010.5, 4004.4, 4010)])
    s = find_signals(broke, ENTRY_P, 0.08)
    assert s.sweep_idx[1][32] != 30          # the break at 31 is a new sweep, not a confirmation of 30
    late = _entry_bars([(4009, 4009.5, 4005, 4006)] + [(4006, 4007, 4005.5, 4006.5)] * 3
                       + [(4006.5, 4010.5, 4006, 4010)])
    s2 = find_signals(late, ENTRY_P, 0.08)
    assert not s2.signal[1][34] or s2.sweep_idx[1][34] != 30


def test_entry_signal_does_not_use_future_candles():
    from quant_engine.research.entries import find_signals
    base = _entry_bars([(4009, 4009.5, 4005, 4006), (4006, 4007, 4005.5, 4006.5), (4006.5, 4010.5, 4006, 4010)])
    full = find_signals(base, ENTRY_P, 0.08)
    cut = find_signals(base.iloc[:33].reset_index(drop=True), ENTRY_P, 0.08)
    assert cut.signal[1][32] == full.signal[1][32]
    assert cut.stop[1][32] == pytest.approx(full.stop[1][32])


def test_entries_cli_report_and_examples(config_dir, tmp_path, monkeypatch, planted):
    monkeypatch.chdir(tmp_path)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, planted[tf])
    rep = tmp_path / "en"
    assert cli.run(["--config", str(config_dir), "entries", "--start", "2026-06-01", "--out", str(rep),
                    "--example", "2026-07-01 13:00,BUY"]) == 0
    md = (rep / "report.md").read_text()
    assert "Does the rule catch your trades?" in md and "spread charged**: $0.08" in md
    assert "breakeven_win_pct" in md
    sigs = pd.read_csv(rep / "signals.csv")
    assert len(sigs) > 0 and set(sigs["side"]) <= {"BUY", "SELL"}
    cand = pd.read_csv(rep / "candidates.csv")
    assert set(cand["target_r"]) <= {1.0, 2.0, 3.0} and set(cand["filter"]) <= {"any", "flush", "with trend"}
