"""Dashboard service layer, chart and a headless render of the Streamlit page."""
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.dashboard import services
from quant_engine.dashboard.chart import build_chart
from quant_engine.data.storage import ParquetStore
from quant_engine.timeframes import Timeframe

from .fake_mt5 import NOW_UTC, FakeMT5

APP = Path(__file__).resolve().parents[1] / "src" / "quant_engine" / "dashboard" / "app.py"
NOW = lambda: NOW_UTC  # noqa: E731


@pytest.fixture
def filled(cfg):
    store = ParquetStore(cfg.data.storage.root)
    services.download(cfg, store, list(Timeframe), 30, FakeMT5(), now=NOW)
    return store


def test_check_connection(cfg):
    info = services.check_connection(cfg, FakeMT5(), now=NOW)
    assert info["connected"] and info["broker_symbol"] == "XAUUSD.r" and info["point"] == 0.01
    assert info["server_offset_hours"] == 2 and info["last_tick_utc"].startswith("2026-11-24 10:02:29")


def test_download_and_status(cfg, filled):
    st = services.data_status(cfg, filled, now=NOW_UTC)
    assert list(st["timeframe"]) == ["M1", "M5", "M15", "H1", "H4"]
    assert (st["quality"] == "OK").all() and (st["candles"] > 0).all()
    assert st.loc[st["timeframe"] == "M5", "last_close_age"].iat[0] == "2m"


def test_status_without_data(cfg):
    st = services.data_status(cfg, ParquetStore(cfg.data.storage.root), now=NOW_UTC)
    assert (st["quality"] == "no data").all()
    with pytest.raises(ValueError, match="download first"):
        services.build_features(services.load_bars(ParquetStore(cfg.data.storage.root), "XAUUSD"), cfg, 0.01)


def test_snapshot_summary_and_cards(cfg, filled):
    bars = services.load_bars(filled, "XAUUSD")
    feats = services.build_features(bars, cfg, 0.01)
    t = services.default_decision_time(bars)
    assert t == pd.Timestamp("2026-11-24 10:00", tz="UTC")
    snap = feats.snapshot(t)
    summary = services.snapshot_summary(snap)
    json.dumps(summary, default=str)
    assert set(summary["structure"]) == {"M5", "M15", "H1", "H4"}
    assert summary["h1_dealing_range"]["price_zone"] in ("discount", "premium", "equilibrium")
    cards = services.market_status(snap)
    assert len(cards) == 8 and all(tone in ("good", "bad", "neutral") for *_, tone in cards)
    assert dict((c[0], c[1]) for c in cards)["Session (NY time)"] == "London KZ"


def test_chart_draws_only_known_data(cfg, filled):
    bars = services.load_bars(filled, "XAUUSD")
    feats = services.build_features(bars, cfg, 0.01)
    snap = feats.snapshot(pd.Timestamp("2026-11-23 13:00", tz="UTC"))
    fig = build_chart(feats, snap, Timeframe.M5, lookback=120, time_zone="Asia/Kolkata")
    candles = fig.data[0]
    assert len(candles.x) == 120
    assert pd.Timestamp(candles.x[-1]).tz_convert("UTC") == pd.Timestamp("2026-11-23 12:55", tz="UTC")
    assert len(fig.data) > 1 and len(fig.layout.shapes) > 0
    bare = build_chart(feats, snap, Timeframe.H1, lookback=50, show=set())
    assert len(bare.data) == 1 and len(bare.layout.shapes) == 0


def _app(cfg_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("QE_CONFIG_DIR", str(cfg_dir))
    return AppTest.from_file(str(APP), default_timeout=120)


def test_app_renders_with_data(config_dir, cfg, filled, monkeypatch):
    at = _app(config_dir, monkeypatch).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Bias (H4/H1)" in labels and "Volatility" in labels
    assert any("cannot place trades" in c.value for c in at.caption)
    assert not at.error


def test_app_without_data_asks_for_download(config_dir, monkeypatch):
    at = _app(config_dir, monkeypatch).run()
    assert not at.exception
    assert any("No market data stored yet" in i.value for i in at.info)


def test_app_reports_mt5_errors_without_crashing(config_dir, monkeypatch):
    at = _app(config_dir, monkeypatch).run()
    at.sidebar.button[0].click().run()   # "Check MT5 connection" without MetaTrader5 installed
    assert not at.exception
    assert any("Connecting to MT5 failed" in e.value for e in at.error)


def test_dashboard_command_launches_streamlit_on_localhost(monkeypatch, config_dir):
    calls = {}

    def fake_call(cmd, env):
        calls["cmd"], calls["env"] = cmd, env
        return 0
    monkeypatch.setattr("subprocess.call", fake_call)
    assert cli.run(["--config", str(config_dir), "dashboard", "--port", "8600"]) == 0
    cmd = calls["cmd"]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("app.py")
    assert cmd[cmd.index("--server.address") + 1] == "localhost" and "8600" in cmd
    assert calls["env"]["QE_CONFIG_DIR"] == str(config_dir)


def test_app_signals_backtest_and_news_tabs(config_dir, cfg, filled, monkeypatch, tmp_path):
    from quant_engine.live.state import EngineState, SentSignal
    from quant_engine.strategy.entry import Signal
    monkeypatch.chdir(tmp_path)
    st = EngineState("data/signals")
    s = Signal(pd.Timestamp("2026-11-23 13:00", tz="UTC"), 1, 1, 2650.0, 2647.0, 2653.0, 2656.0, None, 70,
               "A", 0, 2647.5, "newyork_kz", 0.3)
    st.signals[s.time.isoformat()] = SentSignal(s, 1, 0.16, status="closed", outcome="stop loss", r=-1.0)
    st.save()
    at = _app(config_dir, monkeypatch).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Signals sent" in [m.label for m in at.metric]
    assert any("No backtest yet" in i.value for i in at.info)
    next(r for r in at.radio if r.label == "Data").set_value("mt5").run()
    run_bt = next(b for b in at.button if b.label == "Run backtest")
    run_bt.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Saved to reports" in x.value for x in at.success)
    assert "Win rate %" in [m.label for m in at.metric]


def test_research_tab_shows_gold_results_and_reports(config_dir, monkeypatch, tmp_path):
    import json
    import shutil
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "research_data").mkdir()
    shutil.copy(Path(__file__).parents[1] / "research_data" / "exp016_trades_original.csv", tmp_path / "research_data")
    nat = tmp_path / "reports" / "lb_natural_mt5_20261127T000000"
    nat.mkdir(parents=True)
    (nat / "report.md").write_text("# EXP-020 test report")
    (nat / "results.json").write_text(json.dumps({"sizing": [{"version": "natural", "window": "all", "start_balance": 500,
                                                               "signals": 10, "taken": 7, "skipped": 3}]}))
    assert services.research_reports()["lb_natural"] == Path("reports") / nat.name
    s = services.trade_summary(services.gold_trades())
    assert s["trades"] == 2154
    at = _app(config_dir, monkeypatch).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "R/trade" in [m.label for m in at.metric]
    assert any("EXP-020 test report" in m.value for m in at.markdown)
    assert any("Not run on this PC yet: python -m quant_engine montecarlo" in i.value for i in at.info)
