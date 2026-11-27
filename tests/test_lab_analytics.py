"""Quant Lab Phase 5: performance analytics on hand-made trades with known answers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_engine.analytics import drawdown, layer_metrics, monthly_returns, performance, report_markdown
from quant_engine.analytics.performance import daily_returns, sharpe_sortino


def trades():
    # net P/L: +100, -50, -50, +150 (equity 10,000 -> 10,100 -> 10,050 -> 10,000 -> 10,150)
    net = [100.0, -50.0, -50.0, 150.0]
    exits = pd.to_datetime(["2026-01-05 10:00", "2026-01-06 10:00", "2026-01-30 10:00", "2026-02-02 10:00"], utc=True)
    df = pd.DataFrame({"exit_time": exits, "entry_time": exits - pd.Timedelta(hours=1), "net_pnl": net,
                       "spread_cost": 4.0, "slippage_cost": 1.0, "commission": 2.0, "risk_money": 50.0,
                       "side": ["BUY", "SELL", "BUY", "SELL"], "session": "london", "regime": "bullish",
                       "strategy_id": "s", "exit_reason": ["target", "stop", "stop", "target"], "intrabar": "single"})
    df["gross_pnl"] = df["net_pnl"] + 7.0
    df["r"] = df["net_pnl"] / 50
    df["r_gross"] = df["gross_pnl"] / 50
    df["r_after_spread"] = (df["gross_pnl"] - 4.0) / 50
    return df


def test_net_metrics():
    m = layer_metrics(trades(), "net", 10_000, 0.5)
    assert m["trades"] == 4 and m["net_profit"] == 150
    assert m["gross_profit"] == 250 and m["gross_loss"] == -100 and m["profit_factor"] == 2.5
    assert m["win_rate_pct"] == 50 and m["expectancy"] == 37.5 and m["expectancy_r"] == 0.75
    assert m["max_drawdown"] == 100 and m["max_drawdown_pct"] == pytest.approx(100 / 10_100 * 100, abs=0.01)
    assert m["max_consecutive_losses"] == 2 and m["return_pct"] == 1.5


def test_gross_layers_are_separate_and_larger():
    p = performance(trades(), 10_000, 0.5)
    lay = p["layers"]
    assert lay["before_costs"]["net_profit"] == 150 + 4 * 7
    assert lay["after_spread"]["net_profit"] == 150 + 4 * 3
    assert lay["net"]["net_profit"] == 150
    assert lay["before_costs"]["expectancy_r"] > lay["after_spread"]["expectancy_r"] > lay["net"]["expectancy_r"]


def test_breakdowns_and_monthly():
    p = performance(trades(), 10_000, 0.5)
    side = p["side"].set_index("group")
    assert side.loc["BUY", "net_profit"] == 50 and side.loc["SELL", "net_profit"] == 100
    mo = monthly_returns(trades(), 10_000)
    assert mo["month"].tolist() == ["2026-01", "2026-02"] and mo["net_pnl"].tolist() == [0.0, 150.0]
    assert mo["return_pct"].tolist() == [0.0, 1.5]
    assert set(p["exit_reason"]["group"]) == {"stop", "target"}
    md = report_markdown(p, "x")
    assert "NET (after all costs)" in md and "Monthly returns" in md and "Long / short" in md


def test_daily_returns_sharpe_sortino():
    r = daily_returns(trades(), 10_000, trades()["net_pnl"].to_numpy())
    assert len(r) == 21                                   # weekdays 5 Jan - 2 Feb 2026
    assert r.iloc[0] == pytest.approx(0.01) and r.iloc[1] == pytest.approx(-50 / 10_100)
    sh, so = sharpe_sortino(r)
    assert sh == pytest.approx(round(r.mean() / r.std(ddof=1) * math.sqrt(252), 2))
    assert so == pytest.approx(round(r.mean() / math.sqrt((np.minimum(r, 0) ** 2).mean()) * math.sqrt(252), 2))
    assert sharpe_sortino(pd.Series([0.01])) == (None, None)


def test_drawdown_series_and_empty():
    dd = drawdown(pd.Series([100.0, 120.0, 90.0, 130.0]))
    assert dd["drawdown"].tolist() == [0, 0, -30, 0] and dd["drawdown_pct"].iat[2] == pytest.approx(-25)
    assert performance(trades().iloc[0:0])["layers"]["net"] == {"trades": 0}
