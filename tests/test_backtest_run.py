"""Backtest loop: risk locks, grade B logging, regimes, report, and an end-to-end
run on synthetic data with a no-look-ahead check. Synthetic data only checks
the mechanics; it says nothing about real performance."""
import lzma
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_engine.config import StrategyConfig
from quant_engine.data import dukascopy as dk
from quant_engine.features.context import MarketFeatures
from quant_engine.features.sessions import trading_day
from quant_engine.strategy import report
from quant_engine.strategy.backtest import Backtester
from quant_engine.strategy.entry import Signal
from quant_engine.strategy.regime import BEARISH, BULLISH, NEUTRAL, regime_at, trend_regime_series
from quant_engine.timeframes import Timeframe

from .synthetic import synthetic_bars, synthetic_m1


# ------------------------------------------------------------------ dukascopy
def _bi5(rows):
    raw = b"".join(dk.RECORD.pack(*r) for r in rows)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_day_url_uses_zero_based_month():
    assert dk.day_url("xauusd", date(2026, 1, 5)).endswith("/XAUUSD/2026/00/05/BID_candles_min_1.bi5")
    assert "/2026/11/31/" in dk.day_url("XAUUSD", date(2026, 12, 31))


def test_parse_day_decodes_prices_and_utc_times_and_drops_flat_candles():
    raw = _bi5([(0, 2650_120, 2650_450, 2649_900, 2650_600, 12.5),
                (60, 2650_450, 2650_450, 2650_450, 2650_450, 0.0),    # flat, no volume -> closed minute
                (120, 2650_450, 2651_000, 2650_400, 2651_100, 3.0)])
    df = dk.parse_day(raw, date(2026, 11, 23))
    assert list(df["time"]) == [pd.Timestamp("2026-11-23 00:00", tz="UTC"),
                                pd.Timestamp("2026-11-23 00:02", tz="UTC")]
    r = df.iloc[0]
    assert (r.open, r.high, r.low, r.close) == (2650.12, 2650.6, 2649.9, 2650.45)
    assert df["spread"].eq(30.0).all()
    assert dk.parse_day(b"", date(2026, 11, 23)).empty


def test_load_folder_reads_dated_files(tmp_path):
    (tmp_path / "2026-11-24.bi5").write_bytes(_bi5([(0, 1000, 1001, 999, 1002, 1.0)]))
    (tmp_path / "2026-11-23.bi5").write_bytes(_bi5([(60, 1000, 1001, 999, 1002, 1.0)]))
    df = dk.load_folder(tmp_path)
    assert list(df["time"].dt.strftime("%m-%d %H:%M")) == ["11-23 00:01", "11-24 00:00"]


def test_resample_m5_and_h4_on_new_york_close_grid():
    m1 = synthetic_m1(start="2026-11-02", days=4)
    m5 = dk.resample(m1, Timeframe.M5)
    first = m1.iloc[:5]
    assert m5.iloc[0][["open", "high", "low", "close"]].tolist() == [
        first.open.iat[0], first.high.max(), first.low.min(), first.close.iat[-1]]
    h4 = dk.resample(m1, Timeframe.H4)
    assert set(h4["time"].dt.hour) == {2, 6, 10, 14, 18, 22}          # winter: server midnight = 22:00 UTC
    summer = dk.resample(synthetic_m1(start="2026-07-06", days=2), Timeframe.H4)
    assert set(summer["time"].dt.hour) == {1, 5, 9, 13, 17, 21}       # summer: 21:00 UTC


# --------------------------------------------------------------------- regime
def _h4(closes):
    t = pd.date_range("2026-01-05", periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame({"time": t, "high": c + 2, "low": c - 2, "close": c})


def test_trend_regime_labels():
    up = _h4(np.linspace(2000, 2300, 120))
    down = _h4(np.linspace(2300, 2000, 120))
    flat = _h4(2000 + np.sin(np.arange(120)) * 5)
    assert trend_regime_series(up).iat[-1] == BULLISH
    assert trend_regime_series(down).iat[-1] == BEARISH
    assert trend_regime_series(flat).iat[-1] == NEUTRAL
    assert trend_regime_series(up).iat[10] == NEUTRAL                  # not enough history yet


def test_regime_at_uses_only_closed_h4_candles():
    h4 = _h4([1.0] * 3)
    reg = pd.Series([BULLISH, BEARISH, NEUTRAL])
    t0 = h4["time"].iat[0]
    assert regime_at(h4, reg, t0 + pd.Timedelta(hours=3, minutes=59)) == NEUTRAL   # first H4 still open
    assert regime_at(h4, reg, t0 + pd.Timedelta(hours=4)) == BULLISH
    assert regime_at(h4, reg, t0 + pd.Timedelta(hours=8, minutes=5)) == BEARISH


# ----------------------------------------------------------------- risk locks
class _StubModel:
    def __init__(self, signals):
        self.signals = {s.m5_idx: s for s in signals}

    def candidates(self):
        return sorted(self.signals)

    def evaluate(self, j):
        return self.signals.get(j)


def _stub_backtester(signals, results):
    """Backtester whose model and simulation are replaced: ``results[j]`` is the R of signal j."""
    cfg = StrategyConfig()
    bt = object.__new__(Backtester)
    bt.cfg = cfg
    bt.t = pd.DatetimeIndex(pd.date_range("2026-11-23 12:00", periods=500, freq="5min", tz="UTC"))
    bt.day = trading_day(bt.t, cfg.sessions)
    bt.h4 = pd.DataFrame({"time": pd.date_range("2026-11-20", periods=40, freq="4h", tz="UTC")})
    bt.regimes = pd.Series([NEUTRAL] * 40)
    bt.f = SimpleNamespace(snapshot=lambda t: SimpleNamespace(state={Timeframe.H4: "ranging"}))
    bt.model = _StubModel(signals)
    bt.signals = []
    bt._simulate = lambda s: ("sim", True, s.m5_idx + 1, s.m5_idx + 2, results[s.m5_idx])
    return bt


def _sig(j, grade="A", kz="newyork_kz"):
    t = pd.Timestamp("2026-11-23 12:00", tz="UTC") + pd.Timedelta(minutes=5 * (j + 1))
    return Signal(t, j, 1, 100.0, 98.0, 102.0, 104.0, None, 70, grade, j, 98.5, kz, 0.3)


def test_two_losses_in_a_row_stop_the_day():
    bt = _stub_backtester([_sig(0), _sig(10, kz="london_kz"), _sig(20, kz="other")],
                          {0: -1.0, 10: -1.0, 20: 2.0})
    out = bt.run()
    assert list(out["outcome"]) == ["sim", "sim", "skipped: 2 losses in a row today"]
    assert out["r"].sum() == -2.0


def test_max_two_signals_per_session_and_grade_b_not_traded():
    bt = _stub_backtester([_sig(0), _sig(5, grade="B"), _sig(10), _sig(20)], {0: 1.0, 10: -1.0, 20: 1.0})
    out = bt.run()
    assert list(out["outcome"]) == ["sim", "grade B: logged, not sent", "sim",
                                    "skipped: 2 signals already this session"]
    assert not out.loc[1, "filled"]


def test_daily_loss_lock_at_minus_two_percent():
    bt = _stub_backtester([_sig(0), _sig(10, kz="a"), _sig(20, kz="b")], {0: -4.0, 10: 1.0, 20: 1.0})
    assert bt.run()["outcome"].iat[1] == "skipped: daily loss lock"     # -4R x 0.5% = -2%


def test_report_stats_and_markdown():
    tr = pd.DataFrame({"filled": [True, True, True, False], "r": [2.0, -1.0, 0.5, 0.0],
                       "signal_time": pd.date_range("2026-11-02", periods=4, freq="7D", tz="UTC"),
                       "trend_regime": [BULLISH, BULLISH, BEARISH, NEUTRAL], "h4_structure": "bullish",
                       "grade": "A", "side": "BUY", "kill_zone": "newyork_kz",
                       "outcome": ["TP2 hit", "stop loss", "breakeven after TP1", "expired"]})
    s = report.stats(tr)
    assert s["trades"] == 3 and s["win_rate_pct"] == pytest.approx(66.7) and s["total_r"] == 1.5
    assert s["profit_factor"] == 2.5 and s["max_drawdown_r"] == 1.0 and s["return_pct"] == 0.75
    by = report.breakdown(tr, "trend_regime")
    assert list(by["trend_regime"]) == [BEARISH, BULLISH] and list(by["trades"]) == [1, 2]
    md = report.to_markdown("T", {"data": "synthetic"}, tr, {"1 outside kill zone": 3}, 0.5)
    assert "## By Market regime" in md and "| bullish |" in md and "1 outside kill zone" in md


# ----------------------------------------------------- end to end, synthetic
@pytest.fixture(scope="module")
def synth():
    bars = synthetic_bars(days=45, seed=6)
    cfg = StrategyConfig()
    bt = Backtester(MarketFeatures(bars, cfg, point=0.01), cfg)
    return bars, cfg, bt, bt.run()


def test_end_to_end_run_produces_a_traded_signal(synth):
    _, _, bt, trades = synth
    assert len(bt.signals) >= 1 and trades["filled"].any()
    s = bt.signals[0]
    assert s.grade in ("A+", "A", "B") and 50 <= s.score <= 100
    assert (s.entry - s.stop_loss) * s.direction > 0 and (s.tp2 - s.entry) * s.direction >= 2 * s.risk
    assert sum(bt.model.rejections.values()) > 0
    assert set(trades["trend_regime"]) <= {BULLISH, BEARISH, NEUTRAL}


def test_signal_is_identical_with_only_past_data(synth):
    """No look-ahead: rebuild every feature from candles closed by the signal time only."""
    bars, cfg, bt, _ = synth
    s = bt.signals[0]
    past = {tf: b[b["time"] + tf.delta <= s.time].reset_index(drop=True) for tf, b in bars.items()}
    again = Backtester(MarketFeatures(past, cfg, point=0.01), cfg).model.evaluate(s.m5_idx)
    assert again == s
