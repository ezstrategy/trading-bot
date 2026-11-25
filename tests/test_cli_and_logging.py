import json
import logging

import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.logging_setup import JsonFormatter, get_logger, setup_logging
from quant_engine.session import open_session

from .fake_mt5 import NOW_UTC, FakeMT5


@pytest.fixture
def run(config_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeMT5()

    def _run(*args):
        return cli.run(["--config", str(config_dir), *args], mt5_module=fake, now=lambda: NOW_UTC)
    return _run


def test_check(run, capsys):
    assert run("check") == 0
    text = capsys.readouterr().out
    assert '"broker_symbol": "XAUUSD.r"' in text and '"server_clock_mode": "ny_close"' in text
    assert '"execution_mode": "signals_only"' in text


def test_symbols(run, capsys):
    assert run("symbols") == 0
    assert "using XAUUSD.r for XAUUSD" in capsys.readouterr().out


def test_download_then_validate(run, capsys, tmp_path):
    assert run("download", "--timeframes", "M5,H1", "--start", "2026-11-16", "--end", "2026-11-24") == 0
    out = capsys.readouterr().out
    assert "XAUUSD M5" in out and "quality=OK" in out
    assert (tmp_path / "data" / "raw" / "XAUUSD" / "M5.parquet").exists()
    assert list((tmp_path / "reports").glob("download_*.json"))
    assert run("validate", "--timeframes", "M5,H1") == 0


def test_validate_flags_bad_data(run, tmp_path, capsys):
    run("download", "--timeframes", "M5", "--start", "2026-11-16", "--end", "2026-11-24")
    p = tmp_path / "data" / "raw" / "XAUUSD" / "M5.parquet"
    df = pd.read_parquet(p)
    df.loc[10, "high"] = df.loc[10, "low"] - 5
    df.to_parquet(p, index=False)
    assert run("validate", "--timeframes", "M5") == 1


def test_bad_symbol_returns_error(config_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.run(["--config", str(config_dir), "check"], mt5_module=FakeMT5(symbols=["EURUSD"])) == 1


def test_bad_config_returns_2(tmp_path):
    assert cli.run(["--config", str(tmp_path / "nope"), "check"]) == 2


def test_session_calibrates_clock(cfg):
    with open_session(cfg, mt5_module=FakeMT5(), now=lambda: NOW_UTC, sleep=lambda s: None) as s:
        assert s.symbol.name == "XAUUSD.r" and s.clock.mode == "ny_close"
        assert s.clock.detected_offset_hours == 2


def test_json_log_lines_and_secret_masking():
    rec = logging.LogRecord("quant_engine.x", logging.INFO, __file__, 1, "bars_chunk", None, None)
    rec.timeframe, rec.rows, rec.password = "M5", 12, "hunter2"
    line = json.loads(JsonFormatter().format(rec))
    assert line["event"] == "bars_chunk" and line["timeframe"] == "M5" and line["rows"] == 12
    assert line["password"] == "***" and line["ts"].endswith("+00:00")


def test_setup_logging_writes_file(tmp_path):
    f = tmp_path / "logs" / "e.log"
    setup_logging("INFO", json_format=False, file=str(f))
    get_logger("t").info("hello", extra={"k": 1})
    for h in logging.getLogger("quant_engine").handlers:
        h.flush()
    assert json.loads(f.read_text().splitlines()[-1])["event"] == "hello"


def test_context_command(run, capsys):
    assert run("context") == 1  # nothing stored yet
    assert run("download", "--timeframes", "M5,M15,H1,H4", "--start", "2026-10-20", "--end", "2026-11-24 10:00") == 0
    capsys.readouterr()
    assert run("context", "--at", "2026-11-23 13:00") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["session"] == "newyork_kz" and set(out["structure"]) == {"M5", "M15", "H1", "H4"}
    assert out["bias"] in ("BULL", "BEAR", "NONE")
