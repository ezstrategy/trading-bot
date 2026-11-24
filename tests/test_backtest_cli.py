"""`dukascopy` and `backtest` commands with a fake download (no network, no MT5)."""
import json
import lzma

import numpy as np
import pandas as pd

from quant_engine import cli
from quant_engine.data import dukascopy as dk

from .synthetic import synthetic_m1


def _fake_feed(m1: pd.DataFrame):
    """fetch(url) serving Dukascopy-format files cut from synthetic M1 bars."""
    by_day = {d: g for d, g in m1.groupby(m1["time"].dt.date)}
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        y, m, d = url.split("/")[-4:-1]
        day = pd.Timestamp(int(y), int(m) + 1, int(d)).date()      # month is zero-based in the URL
        g = by_day.get(day)
        if g is None:
            return b""
        secs = ((g["time"] - pd.Timestamp(day, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
        px = {k: np.round(g[k].to_numpy() * 1000).astype(int) for k in ("open", "close", "low", "high")}
        raw = b"".join(dk.RECORD.pack(int(s), int(o), int(c), int(l), int(h), 5.0)
                       for s, o, c, l, h in zip(secs, px["open"], px["close"], px["low"], px["high"]))
        return lzma.compress(raw, format=lzma.FORMAT_ALONE)
    fetch.calls = calls
    return fetch


def test_dukascopy_download_resume_and_backtest(config_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fetch = _fake_feed(synthetic_m1(start="2026-06-01", days=45, seed=6))
    args = ["--config", str(config_dir), "dukascopy", "--start", "2026-06-01", "--end", "2026-07-16"]
    assert cli.run(args, fetch=fetch) == 0
    out = capsys.readouterr().out
    assert "XAUUSD_DUKASCOPY M5" in out and "'failed': 0" in out
    assert not any("/2026/05/06/" in u for u in fetch.calls)      # Saturday 2026-06-06 skipped
    n = len(fetch.calls)
    assert cli.run(args, fetch=fetch) == 0 and len(fetch.calls) == n   # resume: nothing fetched again

    rep = tmp_path / "rep"
    assert cli.run(["--config", str(config_dir), "backtest", "--source", "dukascopy", "--out", str(rep)]) == 0
    summary = json.loads((rep / "summary.json").read_text())
    assert "Dukascopy" in summary["meta"]["data"] and summary["overall"]["trades"] >= 1
    md = (rep / "report.md").read_text()
    assert "## By Market regime" in md and "## Signal funnel" in md
    trades = pd.read_csv(rep / "trades.csv")
    assert {"trend_regime", "h4_structure", "outcome", "r"} <= set(trades.columns)


def test_backtest_without_data_explains_what_to_run(config_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.run(["--config", str(config_dir), "backtest", "--source", "dukascopy"]) == 1
    assert "quant_engine dukascopy" in capsys.readouterr().err
