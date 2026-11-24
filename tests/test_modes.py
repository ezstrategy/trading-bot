"""The three regime strategies and the switch. Synthetic prices: mechanics only, no performance."""
import pandas as pd
import pytest

from quant_engine.config import StrategyConfig
from quant_engine.features.context import MarketFeatures
from quant_engine.strategy.backtest import Backtester
from quant_engine.strategy.modes import (
    BearishRallyModel,
    BullishTrendModel,
    RangeFadeModel,
    RegimeModel,
    build_model,
)
from quant_engine.timeframes import Timeframe

from .synthetic import synthetic_bars

SEEDS = {"bullish": 9, "sideways": 9, "bearish": 7}   # synthetic histories where each mode has setups


@pytest.fixture(scope="module")
def worlds():
    cfg = StrategyConfig()
    out = {}
    for seed in set(SEEDS.values()):
        bars = synthetic_bars(days=75, seed=seed)
        out[seed] = (bars, MarketFeatures(bars, cfg, point=0.01))
    return cfg, out


@pytest.fixture(scope="module")
def world(worlds):
    """(bars, cfg, features, signals) where bars/features are the bullish/sideways history;
    bearish signals come from their own history (``bearish_world``)."""
    cfg, w = worlds
    sigs = {}
    for mode, seed in SEEDS.items():
        m = build_model(mode, w[seed][1], cfg)
        sigs[mode] = [s for s in (m.evaluate(j) for j in m.candidates()) if s]
    bars, f = w[SEEDS["bullish"]]
    return bars, cfg, f, sigs


def test_each_mode_finds_setups_with_its_own_shape(world):
    _, cfg, f, sigs = world
    assert all(sigs[m] for m in sigs)
    for s in sigs["bullish"]:                       # buy the dip: longs, BOS trigger, runner, long time stop
        assert s.direction == 1 and s.strategy == "bullish" and s.time_stop_candles == 36
        assert s.stop_loss < s.sweep_extreme < s.entry and s.tp2 - s.entry >= 2 * s.risk - 1e-9
        ev = f.structure[Timeframe.M5].events
        assert ((ev["idx"] == s.m5_idx) & (ev["kind"] == "BOS") & (ev["direction"] == 1)).any()
    for s in sigs["bearish"]:                       # sell the rally: shorts, CHoCH, no runner, fast
        assert s.direction == -1 and s.tp3 is None and s.time_stop_candles == 18
        assert s.stop_loss > s.sweep_extreme > s.entry and s.entry - s.tp2 >= 2 * s.risk - 1e-9
        assert "raided buy stops" in s.reasons[1]
    for s in sigs["sideways"]:                      # fade: entry at the swept edge, middle then far edge
        snap = f.snapshot(s.time)               # same history as bullish (seed 9)
        dr = snap.dealing_range
        assert s.tp1 == pytest.approx(dr.equilibrium) and s.tp3 is None
        assert dr.low <= s.entry <= dr.high and (s.tp2 - s.entry) * s.direction >= 1.5 * s.risk - 1e-9
        assert "False breakout" in s.reasons[1]


def test_factor_points_add_up_to_at_most_100(world):
    _, _, _, sigs = world
    for mode, lst in sigs.items():
        for s in lst:
            assert s.score == sum(s.factors.values()) <= 100


def test_modes_use_different_triggers(world):
    _, cfg, f, _ = world
    bull, bear, side = (BullishTrendModel(f, cfg), BearishRallyModel(f, cfg), RangeFadeModel(f, cfg))
    assert set(bull.candidates()).isdisjoint(bear.candidates())       # BOS up vs CHoCH down
    assert set(side.candidates()) != set(bull.candidates())


def test_regime_switch_only_trades_the_matching_mode(world):
    _, cfg, f, _ = world
    model = RegimeModel(f, cfg)
    bt = Backtester(f, cfg, model=model)
    trades = bt.run()
    assert len(trades)
    want = {"bullish": "bullish", "bearish": "bearish", "neutral": "sideways"}
    assert all(want[r] == s for r, s in zip(trades["trend_regime"], trades["strategy"]))
    assert any(k.startswith("0 ") for k in model.rejections)


def test_require_regime_blocks_other_markets(world):
    _, cfg, f, sigs = world
    strict = BullishTrendModel(f, cfg, require_regime=True)
    got = [strict.evaluate(s.m5_idx) for s in sigs["bullish"]]
    regimes = [strict.regime(s.time) for s in sigs["bullish"]]
    assert all((g is not None) == (r == "bullish") for g, r in zip(got, regimes))


@pytest.mark.parametrize("mode", ["bullish", "bearish", "sideways"])
def test_no_look_ahead_each_mode(worlds, world, mode):
    """Rebuild everything from candles closed by the signal time: the same signal comes out."""
    cfg, w = worlds
    bars = w[SEEDS[mode]][0]
    s = world[3][mode][0]
    past = {tf: b[b["time"] + tf.delta <= s.time].reset_index(drop=True) for tf, b in bars.items()}
    f2 = MarketFeatures(past, cfg, point=0.01)
    again = build_model(mode, f2, cfg).evaluate(s.m5_idx)
    assert again == s


def test_compare_report_and_mode_flag(config_dir, tmp_path, monkeypatch):
    import json

    from quant_engine import cli
    from quant_engine.config import load_config
    from quant_engine.data.storage import ParquetStore
    monkeypatch.chdir(tmp_path)
    cfg = load_config(str(config_dir), load_env_file=False)
    store = ParquetStore(cfg.data.storage.root)
    for tf, b in synthetic_bars(days=45, seed=9).items():
        store.write_bars("XAUUSD", tf, b)
    out = tmp_path / "cmp"
    assert cli.run(["--config", str(config_dir), "backtest", "--mode", "compare", "--out", str(out)]) == 0
    table = pd.DataFrame(json.loads((out / "summary.json").read_text())["comparison"])
    assert set(table["mode"]) == {"v1", "regime", "bullish", "bearish", "sideways"}
    assert set(table["market"]) == {"all", "bullish", "bearish", "neutral"}
    assert "Strategy comparison" in (out / "comparison.md").read_text()
    assert cli.run(["--config", str(config_dir), "backtest", "--mode", "sideways", "--out", str(tmp_path / "s")]) == 0
    assert "sideways" in (tmp_path / "s" / "report.md").read_text()
