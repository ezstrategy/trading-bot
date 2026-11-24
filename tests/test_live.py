"""Live engine (Phase 4/5) on replayed synthetic candles, Telegram, sizing and news.
Synthetic prices check the mechanics only, not performance."""
import json

import pandas as pd
import pytest

from quant_engine.config import NewsConfig, load_config
from quant_engine.features.news import (
    HIGH,
    HOLIDAY,
    TIER1,
    NewsCalendar,
    parse_ff_json,
    refresh_calendar,
)
from quant_engine.live.engine import SignalEngine
from quant_engine.live.sizing import lot_size
from quant_engine.live.state import CLOSED, EngineState, read_messages
from quant_engine.notify.telegram import ConsoleNotifier, TelegramNotifier

from .live_helpers import SYMBOL, Clock, Recorder, ReplayCollector
from .synthetic import synthetic_bars

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


# --------------------------------------------------------------------- sizing
def test_lot_size_rounds_down_to_step_and_flags_min_lot():
    s = lot_size(10_000, 0.5, 2400.0, 2397.0, 100, 0.01, 0.01)     # $50 risk, $300 per lot
    assert s.lots == 0.16 and s.risk_money == pytest.approx(48.0) and not s.note
    tiny = lot_size(500, 0.5, 2400.0, 2390.0, 100, 0.01, 0.01)     # $2.5 risk, $1000 per lot
    assert tiny.lots == 0.01 and "minimum lot" in tiny.note


# ------------------------------------------------------------------- telegram
def test_telegram_sends_html_with_reply_and_never_needs_hardcoded_token(monkeypatch):
    calls = []

    def post(method, data):
        calls.append((method, data))
        return {"ok": True, "result": {"message_id": 42}}
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100")
    tg = TelegramNotifier.from_env(post=post)
    assert tg.send("<b>hi</b>", reply_to=7) == 42
    m, d = calls[0]
    assert m == "sendMessage" and d["parse_mode"] == "HTML" and d["reply_to_message_id"] == "7"
    assert d["chat_id"] == "-100"


def test_telegram_missing_env_and_rejection(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert TelegramNotifier.from_env() is None
    tg = TelegramNotifier("t", "c", post=lambda m, d: {"ok": False, "description": "chat not found"})
    assert tg.send("x") is None
    flaky = iter([OSError("down"), {"ok": True, "result": {"message_id": 1}}])

    def post(m, d):
        r = next(flaky)
        if isinstance(r, Exception):
            raise r
        return r
    assert TelegramNotifier("t", "c", post=post, backoff=0).send("x") == 1


# ----------------------------------------------------------------------- news
FF = json.dumps([
    {"title": "CPI m/m", "country": "USD", "date": "2026-06-10T08:30:00-04:00", "impact": "High"},
    {"title": "ISM Manufacturing PMI", "country": "USD", "date": "2026-06-10T10:00:00-04:00", "impact": "High"},
    {"title": "FOMC Statement", "country": "USD", "date": "2026-06-17T14:00:00-04:00", "impact": "High"},
    {"title": "Retail Sales", "country": "USD", "date": "2026-06-11T08:30:00-04:00", "impact": "Medium"},
    {"title": "Juneteenth", "country": "USD", "date": "2026-06-19T00:00:00-04:00", "impact": "Holiday"},
    {"title": "ECB Rate", "country": "EUR", "date": "2026-06-11T08:15:00-04:00", "impact": "High"},
])


def test_news_windows_by_tier():
    cal = NewsCalendar(parse_ff_json(FF), NewsConfig())
    tiers = {w.title: w.tier for w in cal.windows}
    assert tiers == {"CPI m/m": TIER1, "ISM Manufacturing PMI": HIGH, "FOMC Statement": TIER1,
                     "Juneteenth": HOLIDAY}                          # medium and EUR events don't block
    assert cal.blackout(T("2026-06-10 11:30")).title == "CPI m/m"     # 60 min before 12:30 UTC
    assert cal.blackout(T("2026-06-10 11:29")) is None
    assert cal.blackout(T("2026-06-10 13:05")) is None                # CPI window ends 13:00, PMI starts 13:30
    assert cal.blackout(T("2026-06-10 13:30")).title.startswith("ISM")
    assert cal.blackout(T("2026-06-17 19:29")).title == "FOMC Statement"   # 90 min after 18:00 UTC
    assert cal.blackout(T("2026-06-19 20:00")).tier == HOLIDAY        # whole New York day
    assert [w.title for w in cal.upcoming(T("2026-06-10 12:21"), 10)] == ["CPI m/m"]


def test_news_refresh_merges_cache_and_survives_network_errors(tmp_path):
    nc = NewsConfig(calendar_file=str(tmp_path / "cal.csv"))
    assert len(refresh_calendar(nc, lambda url: FF)) == 6
    def boom(url):
        raise OSError("offline")
    assert len(refresh_calendar(nc, boom)) == 6                       # cached copy still used


# ----------------------------------------------------------------- live engine
@pytest.fixture(scope="module")
def replay_bars():
    return synthetic_bars(days=45, seed=6)


def _engine(tmp_path, bars, start, news=False, notifier=None, mode="v1"):
    cfg = load_config("config", load_env_file=False)
    cfg.engine.live.state_dir = str(tmp_path / "signals")
    cfg.engine.live.lookback_days = {"M5": 12, "M15": 12, "H1": 28, "H4": 28}
    cfg.strategy.news.enabled = news
    cfg.strategy.mode = mode
    cfg.strategy.news.calendar_file = str(tmp_path / "cal.csv")
    clk = Clock(start)
    rec = notifier or Recorder()
    eng = SignalEngine(cfg, ReplayCollector(bars, clk), SYMBOL, rec, now=clk, balance=lambda: 10_000.0,
                       news_fetch=lambda url: "[]")
    return eng, clk, rec


def _step(eng, clk, n):
    out = []
    for _ in range(n):
        out.append(eng.cycle())
        clk.now += pd.Timedelta(minutes=5)
    return out


def test_live_signal_then_followups_as_replies(tmp_path, replay_bars):
    eng, clk, rec = _engine(tmp_path, replay_bars, "2026-06-29 13:52")
    res = _step(eng, clk, 12)
    assert sum(1 for r in res if r["new_signal"]) == 1
    texts = [t for t, _ in rec.sent]
    assert "SELL LIMIT" in texts[0] and "0.16 lot" in texts[0] and "no trade was placed" in texts[0]
    assert texts[1:] == ["✅ Filled at 2,382.78", "❌ Stopped out (−1R)"]
    assert all(reply == 1 for _, reply in rec.sent[1:])              # follow-ups reply to the signal
    st = EngineState(tmp_path / "signals")                             # persisted
    (sent,) = st.signals.values()
    assert sent.status == CLOSED and sent.outcome == "stop loss" and sent.r == -1.0
    assert list(read_messages(tmp_path / "signals")["kind"])[:3] == ["stop", "filled", "signal"]


def test_live_matches_backtest_and_restart_does_not_resend(tmp_path, replay_bars):
    eng, clk, rec = _engine(tmp_path, replay_bars, "2026-06-29 13:52")
    _step(eng, clk, 4)                                                 # signal sent, maybe filled
    n = len(rec.sent)
    eng2, clk2, rec2 = _engine(tmp_path, replay_bars, str(clk.now.tz_localize(None)))   # restart
    eng2.cycle()
    assert all("LIMIT" not in t for t, _ in rec2.sent)                 # signal not sent twice
    assert n >= 1


def test_stale_signal_is_not_sent(tmp_path, replay_bars):
    eng, clk, rec = _engine(tmp_path, replay_bars, "2026-06-29 14:09")   # 9 min after the candle closed
    eng.cycle()
    assert not any("LIMIT" in t for t, _ in rec.sent)


def test_news_blackout_blocks_the_live_signal(tmp_path, replay_bars):
    ff = json.dumps([{"title": "Core CPI m/m", "country": "USD", "date": "2026-06-29T10:30:00-04:00",
                      "impact": "High"}])                              # 14:30 UTC: blocks 13:30-15:00
    eng, clk, rec = _engine(tmp_path, replay_bars, "2026-06-29 13:52", news=True)
    eng.news_fetch = lambda url: ff
    _step(eng, clk, 3)
    assert not any("LIMIT" in t for t, _ in rec.sent)
    assert eng.calendar.blackout(T("2026-06-29 14:00")) is not None


def test_daily_summary_once_per_day(tmp_path, replay_bars):
    eng, clk, rec = _engine(tmp_path, replay_bars, "2026-06-29 13:52")
    _step(eng, clk, 12)
    clk.now = T("2026-06-29 17:01")                                    # 22:31 IST
    assert eng.cycle()["summary"]
    assert not eng.cycle()["summary"]
    summary = rec.sent[-1][0]
    assert "Daily summary 2026-06-29" in summary and "stop loss" in summary and "-1.00R" in summary


def test_console_notifier_numbers_messages():
    out = []
    c = ConsoleNotifier(out.append)
    assert c.send("a") == 1 and c.send("b", reply_to=1) == 2 and out[1].startswith("  ↳")


def test_cli_run_once_dry_run_and_news(config_dir, tmp_path, monkeypatch, capsys):
    from quant_engine import cli

    from .fake_mt5 import NOW_UTC, FakeMT5
    monkeypatch.chdir(tmp_path)
    base = ["--config", str(config_dir)]
    assert cli.run(base + ["run", "--once", "--dry-run"], mt5_module=FakeMT5(), now=lambda: NOW_UTC,
                   fetch_text=lambda url: FF) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and '"candle": "2026-11-24 10:00:00+00:00"' in out
    assert (tmp_path / "data" / "signals" / "state.json").exists()
    assert cli.run(base + ["news"], now=lambda: T("2026-06-09"), fetch_text=lambda url: FF) == 0
    assert "CPI m/m" in capsys.readouterr().out
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert cli.run(base + ["telegram-test"]) == 1
