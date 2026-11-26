"""Rulebook infrastructure: four-state regime, costs, and the 14 required metrics."""
import numpy as np
import pandas as pd
import pytest

from quant_engine.config import CostsConfig, Regime4Config
from quant_engine.research.metrics import apply_costs, core_metrics, full_report, report_markdown, session_of
from quant_engine.research.regime4 import regime4_series


def _h4(closes, spread=1.0):
    t = pd.date_range("2026-01-05", periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, float)
    return pd.DataFrame({"time": t, "open": c, "high": c + spread, "low": c - spread, "close": c})


def test_regime4_trend_flat_and_uncertain():
    up = _h4(np.linspace(2000, 2300, 150))
    assert regime4_series(up).iat[-1] == "bullish"
    dn = _h4(np.linspace(2300, 2000, 150))
    assert regime4_series(dn).iat[-1] == "bearish"
    rng = np.random.default_rng(1)
    flat = _h4(2000 + rng.normal(0, 0.3, 150), spread=1.0)
    assert regime4_series(flat).iat[-1] == "sideways"
    assert regime4_series(up).iloc[:40].eq("uncertain").all()       # EMA not formed yet


def test_regime4_needs_confirmation_and_ignores_future():
    closes = np.r_[np.full(120, 2000.0) + np.random.default_rng(2).normal(0, 0.3, 120), np.linspace(2000, 2100, 30)]
    s = regime4_series(_h4(closes))
    first = int(np.argmax(s.to_numpy() == "bullish"))
    one_bar = (s.iloc[:first] != "bullish").all()
    assert one_bar and first > 120
    cut = regime4_series(_h4(closes[:first + 1]))
    assert cut.iat[-1] == s.iat[first]                               # same label without future candles
    assert regime4_series(_h4(closes), Regime4Config(confirm_bars=1)).eq("bullish").idxmax() <= first


def test_apply_costs_converts_slippage_and_commission_to_r():
    t = pd.DataFrame({"entry_time": pd.date_range("2026-01-05", periods=2, freq="D", tz="UTC"),
                      "side": ["BUY", "SELL"], "r": [1.0, -1.0], "risk": [10.0, 5.0]})
    out = apply_costs(t, CostsConfig(slippage_points=2, commission_per_lot=7), point=0.01)
    extra = 2 * 2 * 0.01 + 7 / 100                                   # $0.04 slippage + $0.07 commission
    assert out["r"].tolist() == pytest.approx([1 - extra / 10, -1 - extra / 5])
    assert out["r_gross"].tolist() == [1.0, -1.0]


def test_core_metrics_values():
    r = np.array([2.0, -1.0, -1.0, -1.0, 2.0, -1.0])
    m = core_metrics(r, 0.5)
    assert m["trades"] == 6 and m["win_rate_pct"] == pytest.approx(33.3)
    assert m["profit_factor"] == 1.0 and m["expectancy_r"] == 0.0 and m["net_r"] == 0.0
    assert m["max_consecutive_losses"] == 3 and m["avg_win_r"] == 2.0 and m["avg_loss_r"] == -1.0
    assert m["avg_reward_risk"] == 2.0 and m["max_drawdown_r"] == 3.0
    assert m["max_drawdown_pct"] == pytest.approx(100 * (1 - 0.995 ** 3), abs=0.1)


def test_sessions_follow_new_york_clock_through_dst():
    # 08:30 UTC: London (03:30 NY) in summer, 03:30 NY... winter 03:30 NY = 08:30 UTC too; 13:00 UTC: overlap in summer
    summer = pd.DatetimeIndex(["2026-07-01 13:00", "2026-07-01 18:00", "2026-07-01 02:00"], tz="UTC")
    assert list(session_of(summer)) == ["london_ny_overlap", "new_york", "asia"]
    winter = pd.DatetimeIndex(["2026-01-05 13:00", "2026-01-05 12:30"], tz="UTC")
    assert list(session_of(winter)) == ["london_ny_overlap", "london"]


def test_full_report_has_all_breakdowns():
    t = pd.DataFrame({"entry_time": pd.date_range("2026-01-05 13:00", periods=40, freq="17h", tz="UTC"),
                      "side": ["BUY", "SELL"] * 20, "r": np.tile([1.5, -1.0, -1.0, 2.0], 10),
                      "regime": ["bullish", "bearish", "sideways", "uncertain"] * 10})
    rep = full_report(t)
    assert rep["overall"]["trades"] == 40
    assert set(rep["side"]["group"]) == {"BUY", "SELL"} and len(rep["month"]) >= 1
    assert set(rep["regime"]["group"]) == {"bullish", "bearish", "sideways", "uncertain"}
    md = report_markdown("test", rep)
    for word in ("Long / short", "By session", "By market regime", "Monthly", "max_consecutive_losses"):
        assert word in md


def test_first_touch_order_and_pessimism():
    from quant_engine.research.study import first_touch
    h = np.array([10, 11, 13, 10.0]); l = np.array([10, 9.5, 9.0, 10.0]); c = np.array([10, 10, 10, 10.0])
    idx = np.array([0, 0])
    r = first_touch(h, l, c, idx, np.array([10.8, 12.5]), np.array([9.0, 8.0]))
    assert r.tolist() == [1, 1]
    assert first_touch(h, l, c, np.array([1]), np.array([12.5]), np.array([9.0])).tolist() == [-1]   # both in one candle: down


def test_study_cli_uses_development_part_only(config_dir, tmp_path, monkeypatch):
    from quant_engine import cli
    from quant_engine.data.storage import ParquetStore
    from .synthetic import TFS, synthetic_bars
    monkeypatch.chdir(tmp_path)
    b = synthetic_bars(days=150, seed=4)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, b[tf])
    rep = tmp_path / "st"
    assert cli.run(["--config", str(config_dir), "study", "--start", "2026-01-01", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    last = b[TFS[0]]["time"].iat[-1]
    assert "studied part**: development" in md and f"{last:%Y-%m-%d}" not in md.split("studied part")[1].split("\n")[0]
    for k in ("1. Time in each regime", "2. Drift", "3. Momentum", "4. Breakouts", "5. Sweeps"):
        assert k in md
