"""Web app (FastAPI + static front end): every page's API works without MetaTrader 5."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quant_engine import cli
from quant_engine.dashboard import services
from quant_engine.data.storage import ParquetStore
from quant_engine.timeframes import Timeframe
from quant_engine.webapp.api import STATIC, create_app

from .fake_mt5 import NOW_UTC, FakeMT5

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app(cfg))


@pytest.fixture
def filled(cfg):
    store = ParquetStore(cfg.data.storage.root)
    services.download(cfg, store, list(Timeframe), 30, FakeMT5(), now=lambda: NOW_UTC)
    return store


def test_index_static_and_plotly_are_served_locally(client):
    r = client.get("/")
    assert r.status_code == 200 and "/static/js/app.js" in r.text and "/vendor/plotly.min.js" in r.text
    assert client.get("/static/css/glass.css").status_code == 200
    assert client.get("/static/js/pages/market.js").status_code == 200
    js = client.get("/vendor/plotly.min.js")
    assert js.status_code == 200 and "plotly" in js.text[:2000].lower()


def test_every_page_module_is_registered():
    app_js = (STATIC / "js" / "app.js").read_text()
    for page in (STATIC / "js" / "pages").glob("*.js"):
        assert f'./pages/{page.name}"' in app_js, page.name


def test_status_without_data(client):
    s = client.get("/api/status").json()
    assert s["instrument"] == "XAUUSD" and s["execution_mode"] == "signals_only" and s["has_data"] is False
    m = client.get("/api/market").json()
    assert m["ready"] is False and "download" in m["message"].lower()
    q = client.get("/api/data/quality").json()
    assert {r["quality"] for r in q} == {"no data"}


def test_market_snapshot_and_chart(client, filled):
    r = client.get("/api/market", params={"tf": "M15", "lookback": 60, "tz": "Asia/Kolkata", "overlays": "pools,range"})
    d = r.json()
    assert r.status_code == 200 and d["ready"]
    assert d["decision_time"].startswith("2026-11-24 10:00")
    assert len(d["cards"]) == 8 and {c["tone"] for c in d["cards"]} <= {"good", "bad", "neutral"}
    assert set(d["summary"]["structure"]) == {"M5", "M15", "H1", "H4"}
    assert d["figure"]["data"][0]["type"] == "candlestick"


def test_market_review_time_uses_only_known_data(client, filled):
    d = client.get("/api/market", params={"at": "2026-11-23T13:00", "lookback": 120, "overlays": ""}).json()
    assert d["decision_time"].startswith("2026-11-23 13:00")
    x = d["figure"]["data"][0]["x"]
    assert len(x) == 120 and pd.Timestamp(x[-1]).tz_convert("UTC") == pd.Timestamp("2026-11-23 12:55", tz="UTC")
    assert client.get("/api/market", params={"tf": "W9"}).status_code == 400


def test_mt5_errors_are_reported_not_crashing(client):
    r = client.post("/api/mt5/check")   # MetaTrader5 is not installed here
    assert r.status_code == 502 and "Connecting to MT5 failed" in r.json()["detail"]
    assert client.post("/api/mt5/refresh").status_code == 502


def test_signals_backtest_and_news(client, cfg, filled):
    from quant_engine.live.state import EngineState, SentSignal
    from quant_engine.strategy.entry import Signal
    st = EngineState(cfg.engine.live.state_dir)
    s = Signal(pd.Timestamp("2026-11-23 13:00", tz="UTC"), 1, 1, 2650.0, 2647.0, 2653.0, 2656.0, None, 70,
               "A", 0, 2647.5, "newyork_kz", 0.3)
    st.signals[s.time.isoformat()] = SentSignal(s, 1, 0.16, status="closed", outcome="stop loss", r=-1.0)
    st.save()
    d = client.get("/api/signals").json()
    assert d["stats"] == {"sent": 1, "closed": 1, "total_r": -1.0} and d["signals"][0]["entry"] == 2650.0

    assert client.get("/api/backtests").json()["reports"] == []
    assert client.post("/api/backtests/run", json={"source": "mt5", "mode": "nope"}).status_code == 400
    r = client.post("/api/backtests/run", json={"source": "mt5", "mode": "regime"})
    assert r.status_code == 200, r.text
    name = r.json()["report"]
    assert client.get("/api/backtests").json()["reports"] == [name]
    rep = client.get(f"/api/backtests/{name}").json()
    assert "trades" in rep["overall"] and "<h" in rep["report_html"]
    assert client.get("/api/backtests/..%2F..%2Fetc").status_code == 404

    assert client.get("/api/news").status_code in (200, 502)


def test_research_endpoints(client, tmp_path):
    (tmp_path / "research_data").mkdir()
    shutil.copy(REPO / "research_data" / "exp016_trades_original.csv", tmp_path / "research_data")
    nat = tmp_path / "reports" / "lb_natural_mt5_20261127T000000"
    nat.mkdir(parents=True)
    (nat / "report.md").write_text("# EXP-020 test report\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    (nat / "results.json").write_text(json.dumps({"sizing": [{"version": "natural", "window": "all", "signals": 10,
                                                               "taken": 7, "skipped": 3}]}))
    d = client.get("/api/research").json()
    assert d["gold"]["full"]["summary"]["trades"] == 2154
    assert d["gold"]["since_2023"]["summary"]["trades"] < 2154
    assert d["natural"]["sizing"][0]["skipped"] == 3 and d["montecarlo"] is None
    assert [r["key"] for r in d["reports"]] == ["lb_natural"]
    rep = client.get("/api/research/report/lb_natural").json()
    assert "<table>" in rep["html"] and "EXP-020 test report" in rep["html"]
    assert client.get("/api/research/report/nope").status_code == 404


def test_api_has_no_trading_routes(cfg):
    app = create_app(cfg)
    paths = " ".join(getattr(r, "path", "") for r in app.routes).lower()
    for word in ("order", "trade/", "execute", "close_position"):
        assert word not in paths


def test_dashboard_command_starts_webapp_on_localhost(monkeypatch, config_dir):
    calls = {}

    def fake_run(app, host, port, log_level):
        calls.update(app=app, host=host, port=port)
    monkeypatch.setattr("uvicorn.run", fake_run)
    assert cli.run(["--config", str(config_dir), "dashboard", "--port", "8600", "--no-browser"]) == 0
    assert calls["host"] == "127.0.0.1" and calls["port"] == 8600
    assert calls["app"].title == "XAUUSD Quant Engine"
