"""Breakout signal store, paper service, CLI and web API (no MetaTrader 5 needed)."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quant_engine import cli
from quant_engine.breakout.config import BreakoutConfig
from quant_engine.breakout.service import BreakoutService, market_expected_open
from quant_engine.breakout.store import AWAITING, BUY, CANCELLED, SL, TP, SignalStore, StoreError
from quant_engine.data.storage import ParquetStore
from quant_engine.timeframes import Timeframe
from quant_engine.webapp.api import create_app

from .breakout_helpers import m5_from_daily
from .test_breakout import monthly_series, spec

UTC = "UTC"


class FakeCollector:
    """Closed M5 candles up to ``now`` from a prepared frame."""

    def __init__(self, m5: pd.DataFrame, now):
        self.m5, self.now = m5, now
        self.calls = 0

    def bars(self, tf, start, end, closed_only=True):
        self.calls += 1
        t = pd.DatetimeIndex(self.m5["time"])
        keep = (t >= start) & (t + pd.Timedelta(minutes=5) <= self.now())
        return self.m5[keep].reset_index(drop=True)


@pytest.fixture
def m5():
    days, ohlc = monthly_series()
    return m5_from_daily(days, ohlc)


def _row(key="prev_month_high:1.0.0:2024-02-06", **kw):
    base = {"key": key, "source": "paper", "strategy_id": "prev_month_high", "strategy_name": "A", "version": "1.0.0",
            "side": "BUY", "day": "2024-02-06", "signal_time": "2024-02-06T22:00:00+00:00", "breakout_level": 2010.0,
            "d1_close": 2020.0, "atr": 12.0, "entry_est": 2020.1, "stop_est": 2002.1, "target_est": 2056.1,
            "generated": 1, "risk_reason": "ok", "suggested_lots": 0.01, "risk_pct": 0.36}
    return {**base, **kw}


# ------------------------------------------------------------------ store
def test_store_is_permanent_and_prevents_duplicates(tmp_path):
    st = SignalStore(tmp_path / "s.sqlite")
    assert st.insert(_row()) and not st.insert(_row())
    assert st.get("prev_month_high:1.0.0:2024-02-06")["status"] == BUY
    with sqlite3.connect(tmp_path / "s.sqlite") as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM signals")
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE events SET kind = 'x'")


def test_store_manual_workflow_keeps_paper_and_actual_apart(tmp_path):
    st = SignalStore(tmp_path / "s.sqlite")
    k = "prev_month_high:1.0.0:2024-02-06"
    st.insert(_row())
    with pytest.raises(StoreError):
        st.mark_executed(k, 2020.5, 0.01)                      # approve first
    st.update_paper(k, {"entry_time": "2024-02-06T23:00:00+00:00", "entry": 2020.2, "stop": 2002.2,
                        "target": 2056.2, "spread": 0.1, "status": "open"})
    assert st.get(k)["status"] == AWAITING
    st.approve(k, "looks fine")
    st.mark_executed(k, 2020.5, 0.02, note="filled on phone")
    st.update_paper(k, {"status": "sl_hit", "exit_time": "2024-02-08T10:00:00+00:00", "exit": 2002.18, "r": -1.0})
    r = st.record_exit(k, 2050.0, TP, money_per_price_unit_lot=100.0)
    assert r["status"] == TP and r["paper_status"] == "sl_hit" and r["paper_r"] == -1.0
    assert r["exec_r"] == pytest.approx((2050 - 2020.5) / (2020.5 - 2002.2)) and r["exec_pnl"] == pytest.approx(59.0)
    perf = st.performance()
    assert perf["paper"]["net_r"] == -1.0 and perf["manual"]["trades"] == 1 and perf["manual"]["net_r"] > 0
    kinds = [e["kind"] for e in st.events(k)][::-1]
    assert kinds == ["generated", "entry_known", "approved", "executed", "paper_exit", "exit_recorded"]


def test_store_reject_and_risk_skip_are_cancelled(tmp_path):
    st = SignalStore(tmp_path / "s.sqlite")
    st.insert(_row())
    assert st.reject("prev_month_high:1.0.0:2024-02-06", "no")["status"] == CANCELLED
    st.insert(_row("channel_20d:1.0.0:2024-02-06", strategy_id="channel_20d", risk_reason="correlated with prev_month_high"))
    assert st.get("channel_20d:1.0.0:2024-02-06")["status"] == CANCELLED
    with pytest.raises(StoreError):
        st.approve("channel_20d:1.0.0:2024-02-06")


# ---------------------------------------------------------------- service
def test_market_hours():
    assert not market_expected_open(pd.Timestamp("2024-02-10 12:00", tz=UTC))       # Saturday
    assert not market_expected_open(pd.Timestamp("2024-02-06 22:30", tz=UTC))       # 17:30 NY break
    assert market_expected_open(pd.Timestamp("2024-02-07 10:00", tz=UTC))


def _service(tmp_path, m5, clock, **kw):
    cfg = BreakoutConfig(store={"path": str(tmp_path / "b" / "signals.sqlite")})
    st = SignalStore(cfg.store.path)
    svc = BreakoutService(cfg, FakeCollector(m5, lambda: clock[0]), spec(), st, now=lambda: clock[0],
                          equity=lambda: 100_000.0, spread_now=lambda: 0.12, **kw)
    return svc, st


def test_service_stores_signal_once_and_tracks_paper(tmp_path, m5):
    clock = [pd.Timestamp("2024-02-06 22:30", tz=UTC)]              # breakout D1 closed 22:00 UTC
    svc, st = _service(tmp_path, m5, clock)
    out = svc.cycle()
    assert out["data_ok"] and "prev_month_high:1.0.0:2024-02-06" in out["new"]
    r = st.get("prev_month_high:1.0.0:2024-02-06")
    assert r["status"] == BUY and r["entry"] is None and r["entry_est"] == pytest.approx(2020.0 + 0.08 + 0.02)
    assert r["suggested_lots"] > 0 and r["risk_pct"] <= 0.5 and r["spread_at_signal"] == 0.12
    assert "January 2024" in r["explanation"] and r["snapshot"]["d1"][-1]["close"] == 2020.0
    ch = st.get("channel_20d:1.0.0:2024-02-06")
    assert ch["status"] == CANCELLED and "not doubled" in ch["risk_reason"]       # correlated, same breakout
    clock[0] = pd.Timestamp("2024-02-07 10:00", tz=UTC)
    out2 = svc.cycle()
    assert out2["new"] == [] and out2["updated"] >= 1
    r = st.get("prev_month_high:1.0.0:2024-02-06")
    assert r["status"] == AWAITING and r["entry_time"].startswith("2024-02-06T23:00")
    clock[0] = pd.Timestamp("2024-02-20 10:00", tz=UTC)
    svc.cycle()
    r = st.get("prev_month_high:1.0.0:2024-02-06")
    assert r["paper_status"] in ("tp_hit", "sl_hit", "expired") and r["paper_r"] is not None
    assert len(st.all()) == 2
    assert (tmp_path / "b" / "state.json").exists()


def test_service_never_issues_late_or_stale_signals(tmp_path, m5):
    clock = [pd.Timestamp("2024-02-09 12:00", tz=UTC)]              # 3 days after the breakout
    svc, st = _service(tmp_path, m5, clock)
    assert svc.cycle()["new"] == [] and st.all() == []
    clock[0] = pd.Timestamp("2024-04-03 12:00", tz=UTC)              # data ends 29 Mar: stale
    out = svc.cycle()
    assert not out["data_ok"] and "not current" in out["note"]


def test_paper_mode_never_sends_and_off_mode_refuses(tmp_path, m5):
    class Spy:
        sent = []
        def send(self, text, reply_to=None):
            self.sent.append(text)
    clock = [pd.Timestamp("2024-02-06 22:30", tz=UTC)]
    spy = Spy()
    svc, _ = _service(tmp_path, m5, clock, notifier=spy)
    svc.cycle()
    assert spy.sent == []
    with pytest.raises(ValueError):
        BreakoutService(BreakoutConfig(mode="off"), None, spec(), SignalStore(tmp_path / "x.sqlite"))


# ------------------------------------------------------------- CLI + web
def test_cli_backtest_on_stored_bars(cfg, config_dir, tmp_path, m5, capsys):
    ParquetStore(cfg.data.storage.root).write_bars("XAUUSD", Timeframe.M5, m5)
    out = tmp_path / "rep"
    rc = cli.run(["--config", str(config_dir), "breakout-backtest", "--quick", "--out", str(out)])
    assert rc == 0 and (out / "report.md").exists()
    text = capsys.readouterr().out
    assert "prev_month_high" in text and "Swap: NOT INCLUDED" in text


def test_web_api_breakout_page(cfg, config_dir, tmp_path, monkeypatch, m5):
    monkeypatch.chdir(tmp_path)
    c = TestClient(create_app(cfg, config_dir=str(config_dir)))
    d = c.get("/api/breakout").json()
    assert d["mode"] == "paper" and d["state"] is None and d["signals"] == [] and d["backtest"] is None
    assert c.get("/api/breakout/backtest").json()["report"] is None
    ParquetStore(cfg.data.storage.root).write_bars("XAUUSD", Timeframe.M5, m5)
    d = c.get("/api/breakout").json()
    assert [s["strategy_id"] for s in d["state"]["strategies"]] == ["prev_month_high", "channel_20d"]
    rep = c.post("/api/breakout/backtest").json()
    assert rep["report"]["results"]["per_strategy"]["prev_month_high"]["full_history"]["trades"] == 1
    st = SignalStore(tmp_path / "data" / "breakout" / "signals.sqlite")
    st.insert(_row())
    k = "prev_month_high:1.0.0:2024-02-06"
    assert c.post(f"/api/breakout/signals/{k}/execute", json={"entry": 2020, "lots": 0.01}).status_code == 409
    assert c.post(f"/api/breakout/signals/{k}/approve", json={"note": "ok"}).json()["approval"] == "approved"
    assert c.post(f"/api/breakout/signals/{k}/execute", json={"entry": 2020, "lots": 0.01}).json()["executed"] == 1
    r = c.post(f"/api/breakout/signals/{k}/exit", json={"exit": 2002, "result": SL}).json()
    assert r["status"] == SL and r["exec_r"] < 0
    assert len(c.get(f"/api/breakout/signals/{k}").json()["events"]) == 4
    assert c.get("/api/breakout/signals/nope").status_code == 404
    assert c.get("/static/js/pages/breakout.js").status_code == 200


def test_breakout_run_once_with_fake_mt5(config_dir, tmp_path, monkeypatch, capsys):
    """Integration: CLI -> MT5 session (fake) -> broker spec -> service cycle -> store."""
    from .fake_mt5 import NOW_UTC, FakeMT5
    monkeypatch.chdir(tmp_path)
    rc = cli.run(["--config", str(config_dir), "breakout-run", "--once"], mt5_module=FakeMT5(), now=lambda: NOW_UTC)
    out = capsys.readouterr().out
    assert rc == 0 and '"mode": "paper"' in out
    assert (tmp_path / "data" / "breakout" / "signals.sqlite").exists()


def test_instrument_spec_reads_swap_in_points_mode():
    from types import SimpleNamespace

    from quant_engine.instruments import InstrumentSpec
    info = SimpleNamespace(name="XAUUSD", trade_contract_size=100.0, volume_min=0.01, volume_max=50.0, volume_step=0.01,
                           trade_tick_size=0.01, trade_tick_value=1.0, digits=2, currency_profit="USD",
                           swap_mode=1, swap_long=-62.5, swap_short=21.0)
    s = InstrumentSpec.from_symbol_info(info, "USD", base=spec())
    assert s.swap_long_points == -62.5 and "swap_long_points" in s.confirmed
    s2 = InstrumentSpec.from_symbol_info(SimpleNamespace(**{**info.__dict__, "swap_mode": 5}), "USD", base=spec())
    assert s2.swap_long_points is None
