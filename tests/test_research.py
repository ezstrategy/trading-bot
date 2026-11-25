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
