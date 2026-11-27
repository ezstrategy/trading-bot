import numpy as np
import pandas as pd
import pytest

from quant_engine.research import lb_phase2 as P


def _store(tmp_path, days=160, seed=3):
    from quant_engine.data.storage import ParquetStore
    from .synthetic import TFS, synthetic_bars
    b = synthetic_bars(days=days, seed=seed)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, b[tf])
    return store, b


def test_original_is_the_exp015_trend_version():
    from quant_engine.research.london_oos import VERSIONS
    assert P.ORIGINAL == VERSIONS["trend version (EXP-012)"]
    assert P.EXITS["A 3R, BE 1R (original)"] == {"target_r": 3.0, "be_r": 1.0}
    assert P.EXPECTED_EXP015 == {"trades": 914, "net_r": 81.4}


def test_stop_entry_matches_original_code(config_dir, tmp_path):
    from quant_engine.config import load_config
    from quant_engine.research.classics import lb_day_groups, london_breakout
    from quant_engine.timeframes import Timeframe
    cfg = load_config(config_dir)
    _, b = _store(tmp_path)
    m5 = b[Timeframe.M5]
    g = lb_day_groups(m5, cfg.strategy.sessions)
    for params in (P.ORIGINAL, {**P.ORIGINAL, "trend": "none"}):
        orig = pd.DataFrame(london_breakout(m5, 0.08, cfg.strategy.sessions, params, None, g))
        mine = P.lb_trades(m5, 0.08, params, None, g)
        assert len(orig) == len(mine) > 10
        assert np.allclose(orig["r"], mine["r"])
        assert (pd.DatetimeIndex(orig["entry_time"]) == pd.DatetimeIndex(mine["entry_time"])).all()


def test_walk_exit_trailing_and_reasons():
    o = np.array([100, 100, 101, 102.5, 102, 101.2])
    h = np.array([100.5, 101, 102.1, 102.6, 102.2, 101.3])
    l = np.array([99.5, 99.8, 100.9, 102.0, 101.0, 101.0])
    c = o.copy()
    # buy at 100, stop 99 (risk 1), target 103; +2R reached at candle 2 (high 102.1) -> trail 1R behind best
    j, px, why = P.walk_exit(1, 5, 1, 100.0, 99.0, 103.0, o, h, l, c, 0.0, None, 2.0, 1.0)
    assert why == "trail" and j == 4 and px == pytest.approx(101.6)
    j, px, why = P.walk_exit(1, 5, 1, 100.0, 99.0, 103.0, o, h, l, c, 0.0, 1.0)
    assert why == "time" and j == 5
    j, px, why = P.walk_exit(1, 5, 1, 100.0, 99.0, 102.0, o, h, l, c, 0.0, 1.0)
    assert why == "target" and px == 102.0


def test_close_and_retest_entries_never_enter_before_the_signal(config_dir, tmp_path):
    from quant_engine.config import load_config
    from quant_engine.research.classics import lb_day_groups
    from quant_engine.timeframes import Timeframe
    cfg = load_config(config_dir)
    _, b = _store(tmp_path)
    m5 = b[Timeframe.M5]
    g = lb_day_groups(m5, cfg.strategy.sessions)
    st = P.lb_trades(m5, 0.08, P.ORIGINAL, None, g).set_index("day")
    cl = P.lb_trades(m5, 0.08, P.ORIGINAL, None, g, entry="close").set_index("day")
    rt = P.lb_trades(m5, 0.08, P.ORIGINAL, None, g, entry="retest").set_index("day")
    assert len(cl) and len(rt) <= len(cl) + len(st)
    common = cl.index.intersection(rt.index)
    assert (rt.loc[common, "entry_i"] > cl.loc[common, "entry_i"] - 1).all()
    t = pd.DatetimeIndex(m5["time"])
    c = m5["close"].to_numpy()
    for _, row in cl.iterrows():       # the candle before the entry closed beyond the breakout level
        i = int(row["entry_i"]) - 1
        assert (c[i] > row["range_hi"]) if row["side"] == "BUY" else (c[i] < row["range_lo"])
    assert not cl["trigger_after_fill"].any() and st["trigger_after_fill"].all()


def test_day_features_use_only_the_past(config_dir, tmp_path):
    from quant_engine.config import load_config
    from quant_engine.research.classics import lb_day_groups
    from quant_engine.timeframes import Timeframe
    cfg = load_config(config_dir)
    _, b = _store(tmp_path)
    m5 = b[Timeframe.M5]
    g = lb_day_groups(m5, cfg.strategy.sessions)
    full = P.day_features(m5, g)
    cut = len(g) // 2
    part = P.day_features(m5.iloc[: g[cut][-1] + 1], g[: cut + 1])
    cols = ["datr", "range_atr", "range_pct", "vol_pct", "pdh", "pdl", "pdc"]
    pd.testing.assert_frame_equal(full.loc[:cut, cols].reset_index(drop=True), part[cols].reset_index(drop=True))
    assert full["range_pct"].iloc[:60].isna().all()
    assert full["us_news"].isna().all()


def test_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1, 400)
    s = P.boot_stats(x)
    assert float(s["exp_ci"].split(" to ")[0]) > 0 and s["p_boot"] < 0.01
    z = rng.normal(0, 1, 400)
    assert P.boot_stats(z)["p_boot"] > 0.01
    assert P.holm({"a": 0.001, "b": 0.02, "c": 0.06}) == {"a": True, "b": True, "c": False}
    assert P.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert P.newey_west_t(x)[0] > 3


def test_news_flags_from_calendar(config_dir, tmp_path):
    from quant_engine.config import load_config
    from quant_engine.research.classics import lb_day_groups
    from quant_engine.timeframes import Timeframe
    cfg = load_config(config_dir)
    _, b = _store(tmp_path, days=40)
    m5 = b[Timeframe.M5]
    g = lb_day_groups(m5, cfg.strategy.sessions)
    day10 = pd.DatetimeIndex(m5["time"])[g[10][0]]
    nfp = (day10.tz_convert("America/New_York").normalize() + pd.Timedelta(days=1, hours=8, minutes=30)).tz_convert("UTC")
    cal = pd.DataFrame({"time": [nfp, nfp], "currency": ["USD", "GBP"], "impact": ["High", "Low"], "title": ["NFP", "x"]})
    cal = pd.concat([cal, pd.DataFrame({"time": [pd.DatetimeIndex(m5["time"])[0], pd.DatetimeIndex(m5["time"])[-1]],
                                        "currency": ["EUR", "EUR"], "impact": ["Low", "Low"], "title": ["a", "b"]})])
    df = P.day_features(m5, g, cal)
    hit = df.index[df["us_news"] == 1].tolist()
    assert len(hit) == 1 and df["uk_news"].fillna(0).sum() == 0


def test_cli_runs_everything_and_stops_if_not_reproduced(config_dir, tmp_path, monkeypatch):
    from quant_engine import cli
    from quant_engine.config import load_config
    store, b = _store(tmp_path, days=200, seed=8)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(P, "SPLITS", (("development", "2026-06-01", "2026-09-01"),
                                       ("validation", "2026-09-01", "2026-11-01"), ("test", "2026-11-01", "2100-01-01")))
    end = pd.DatetimeIndex(b[list(b)[0]]["time"])[-1] + pd.Timedelta(days=1)
    cfg = load_config(config_dir)
    rep = tmp_path / "p2"
    s = P.run_lb_phase2(cfg, store, "XAUUSD", 0.01, rep, pd.Timestamp("2026-06-01", tz="UTC"), None, None,
                        expected=None, repro_end=end)
    assert s["reproduced"]
    md = (rep / "report.md").read_text()
    for k in ("Part 1", "REPRODUCED", "Newey-West", "Contract and sizing", "Part 4", "H1 range width",
              "H2 body_frac", "H3 room", "H4 volatility", "H6 retest entry", "F 3R, trail after 2R",
              "not tested: no historical economic calendar"):
        assert k in md, k
    cand = pd.read_csv(rep / "candidates.csv")
    assert cand["class"].str[0].isin(["1", "2", "3"]).all()
    assert (rep / "detail" / "original.md").exists()
    assert len(list((rep / "detail").glob("*.md"))) == len(cand) + 1
    # wrong expected numbers: stops after Part 1
    rep2 = tmp_path / "p2b"
    s2 = P.run_lb_phase2(cfg, store, "XAUUSD", 0.01, rep2, pd.Timestamp("2026-06-01", tz="UTC"), None, None,
                         expected={"trades": 1, "net_r": 0.0}, repro_end=end)
    assert not s2["reproduced"] and "NOT REPRODUCED" in (rep2 / "report.md").read_text()
    assert not (rep2 / "candidates.csv").exists()
    assert cli.run(["--config", str(config_dir), "lb-phase2", "--start", "2026-06-01", "--out", str(tmp_path / "c")]) in (0, 1)
