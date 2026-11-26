"""GTC-SBR rules on hand-built candles, no look-ahead, and the CLI report."""
import numpy as np
import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.config import CostsConfig, StrategyConfig
from quant_engine.data.storage import ParquetStore
from quant_engine.research.gtc_sbr import DEFAULTS, find_trades, m15_trend
from quant_engine.timeframes import Timeframe

from .synthetic import TFS, synthetic_bars

COSTS = CostsConfig(min_spread_points=8, slippage_points=0)
P = {**DEFAULTS, "trend_filter": False, "news_ny": []}


def _day():
    """5 Jan 2026 (winter): London opens 08:00 UTC. Quiet until then, range 2000-2006, breakout,
    retest on the 3rd candle after the breakout, then a rally to the target."""
    rows = [(2003.0, 2003.5, 2002.5, 2003.0)] * 48                      # 04:00-07:55
    rows += [(2003, 2006, 2002, 2005), (2005, 2006, 2001, 2002), (2002, 2004, 2000, 2003),
             (2003, 2005, 2001, 2004), (2004, 2006, 2002, 2005), (2005, 2006, 2003, 2005.5)]   # 08:00-08:25 range
    rows += [(2005.5, 2008, 2005.4, 2007.5),                            # 08:30 breakout close 2007.5
             (2007.5, 2008.5, 2007, 2008), (2008, 2008.4, 2007.2, 2007.6),
             (2007.6, 2007.8, 2006.1, 2007.0),                          # 08:45 retest (3rd), closes above 2006
             (2007.0, 2008.0, 2006.8, 2007.8)]                          # 08:50 entry candle
    rows += [(2007.8 + k, 2009.0 + k, 2007.6 + k, 2008.8 + k) for k in range(12)]
    rows += [(2020.0, 2020.5, 2019.5, 2020.0)] * 20
    t = pd.date_range("2026-01-05 04:00", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    df["tick_volume"], df["spread"], df["real_volume"] = 50.0, 8.0, 0.0
    return df


def _flat15(m5):
    g = m5.set_index("time").resample("15min", label="left", closed="left")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last()}).dropna().reset_index()


def test_bull_breakout_retest_entry_stop_target():
    m5 = _day()
    tr, sk = find_trades(m5, _flat15(m5), _flat15(m5).iloc[::16].reset_index(drop=True), P, 0.01, COSTS)
    assert len(tr) == 1, sk
    x = tr.iloc[0]
    assert x["side"] == "BUY" and x["session"] == "london"
    assert x["entry_time"] == pd.Timestamp("2026-01-05 08:50", tz="UTC")
    assert x["entry"] == pytest.approx(2007.0 + 0.08)                    # next open + spread
    assert x["stop"] < 2006.1 and x["range_high"] == 2006 and x["range_low"] == 2000
    assert x["exit_how"] == "target" and x["r"] == pytest.approx(1.5)


def test_no_trade_when_retest_comes_too_early_or_stop_breaks():
    m5 = _day()
    rows = m5.copy()                    # candles: 48-53 range, 54 breakout, 57 = the real retest
    for k in range(57, 63):             # no touch on the 3rd-8th candle after the breakout
        rows.loc[k, "low"] = max(rows.loc[k, "low"], 2007.0)
    rows.loc[55, "low"] = 2006.0        # a touch on the 1st candle is too early
    tr, sk = find_trades(rows, _flat15(rows), _flat15(rows).iloc[::16].reset_index(drop=True), P, 0.01, COSTS)
    assert len(tr) == 0 and sk["no retest"] == 1


def test_trend_filter_needs_completed_m15_bull_trend():
    up = pd.DataFrame({"close": np.linspace(2000, 2100, 80)})
    assert m15_trend(up, DEFAULTS)[-1] == "bullish"
    assert m15_trend(up.iloc[::-1].reset_index(drop=True), DEFAULTS)[-1] == "bearish"
    m5 = _day()
    tr, sk = find_trades(m5, _flat15(m5), _flat15(m5).iloc[::16].reset_index(drop=True),
                         {**P, "trend_filter": True}, 0.01, COSTS)
    assert len(tr) == 0 and sk["against or no trend"] == 1             # flat history: no trend


def test_trades_do_not_change_when_future_data_is_cut():
    b = synthetic_bars(days=200, seed=3)
    m5, m15, h4 = b[Timeframe.M5], b[Timeframe.M15], b[Timeframe.H4]
    full, _ = find_trades(m5, m15, h4, DEFAULTS, 0.01, COSTS)
    assert len(full) > 5
    cut_t = full["exit_time"].iat[len(full) // 2]
    cut, _ = find_trades(m5[m5["time"] < cut_t], m15[m15["time"] < cut_t], h4[h4["time"] < cut_t],
                         DEFAULTS, 0.01, COSTS)
    done = lambda d: d[d["entry_time"] + pd.Timedelta(minutes=5 * 13) <= cut_t].reset_index(drop=True)  # full time-exit window inside
    cols = ["entry_time", "side", "entry", "stop", "r"]
    pd.testing.assert_frame_equal(done(full)[cols], done(cut)[cols])


def test_gtc_cli_report(config_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = synthetic_bars(days=200, seed=3)
    store = ParquetStore(tmp_path / "data")
    for tf in TFS:
        store.write_bars("XAUUSD", tf, b[tf])
    rep = tmp_path / "gtc"
    assert cli.run(["--config", str(config_dir), "gtc", "--start", "2026-01-01", "--out", str(rep)]) == 0
    md = (rep / "report.md").read_text()
    for s in ("Acceptance checks", "Experiment D: London vs New York", "Experiment E", "Walk-forward",
              "Higher costs", "Parameter stability", "Full metrics", "GUESS", "Audit checklist", "decision"):
        assert s in md
    a = pd.read_csv(rep / "trades_A.csv")
    assert {"lots", "equity_before", "r_gross", "cost_r"} <= set(a.columns)
    assert any(d in md for d in ("REJECTED", "RESEARCH REQUIRED", "VALIDATED FOR PAPER TRADING"))
