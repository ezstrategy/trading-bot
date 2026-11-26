"""Performance analytics for engine trades (Quant Lab module 8).

Three clearly separated layers over the **same trades**:

* ``before_costs``: spread, slippage and commission added back (what the rules "made");
* ``after_spread``: only slippage and commission added back;
* ``net``: what the account actually got. This is the result that counts.

Money values come from the engine (lots x broker tick value). Sharpe and Sortino use daily
returns (realised P/L by exit day / equity at the start of that day) over every weekday of the
period, annualised with sqrt(252). GUESS: 252 trading days; gold trades ~260 weekdays a year.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..research.metrics import core_metrics
from ..research.runner import _table

LAYERS = {"before_costs": "gross_pnl", "after_spread": None, "net": "net_pnl"}


def _layer_pnl(trades: pd.DataFrame, layer: str) -> np.ndarray:
    if layer == "before_costs":
        return trades["gross_pnl"].to_numpy(float)
    if layer == "after_spread":
        return (trades["gross_pnl"] - trades["spread_cost"]).to_numpy(float)
    return trades["net_pnl"].to_numpy(float)


def _layer_r(trades: pd.DataFrame, layer: str) -> np.ndarray:
    return {"before_costs": trades["r_gross"], "after_spread": trades["r_after_spread"],
            "net": trades["r"]}[layer].to_numpy(float)


def drawdown(equity: pd.Series) -> pd.DataFrame:
    """Peak, drawdown in money and in % of the peak for an equity series."""
    eq = equity.astype(float).to_numpy()
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    return pd.DataFrame({"equity": eq, "peak": peak, "drawdown": eq - peak,
                         "drawdown_pct": np.where(peak > 0, (eq - peak) / peak * 100, 0.0)}, index=equity.index)


def daily_returns(trades: pd.DataFrame, initial_equity: float, pnl: np.ndarray,
                  start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.Series:
    """Daily returns on every weekday from ``start`` to ``end`` (0 on days without exits)."""
    if trades.empty:
        return pd.Series(dtype=float)
    day = pd.DatetimeIndex(trades["exit_time"]).tz_convert("UTC").normalize()
    by_day = pd.Series(pnl, index=day).groupby(level=0).sum()
    s = pd.Timestamp(start).normalize() if start is not None else by_day.index[0]
    e = pd.Timestamp(end).normalize() if end is not None else by_day.index[-1]
    days = pd.bdate_range(s.tz_localize(None) if s.tzinfo else s, e.tz_localize(None) if e.tzinfo else e).tz_localize("UTC")
    pnl_d = by_day.reindex(days.union(by_day.index), fill_value=0.0)
    eq_start = initial_equity + pnl_d.cumsum().shift(1, fill_value=0.0)
    return (pnl_d / eq_start).rename("return")


def sharpe_sortino(ret: pd.Series) -> tuple[float | None, float | None]:
    """Annualised Sharpe and Sortino (risk-free 0)."""
    if len(ret) < 2:
        return None, None
    m, sd = ret.mean(), ret.std(ddof=1)
    down = math.sqrt(float((np.minimum(ret, 0) ** 2).mean()))
    sh = None if sd == 0 else round(float(m / sd * math.sqrt(252)), 2)
    so = None if down == 0 else round(float(m / down * math.sqrt(252)), 2)
    return sh, so


def layer_metrics(trades: pd.DataFrame, layer: str, initial_equity: float, risk_pct: float,
                  start=None, end=None) -> dict:
    """Every module-8 metric for one cost layer."""
    if trades.empty:
        return {"trades": 0}
    t = trades.sort_values("exit_time")
    pnl = _layer_pnl(t, layer)
    r = _layer_r(t, layer)
    eq = initial_equity + np.cumsum(pnl)
    dd = drawdown(pd.Series(np.r_[initial_equity, eq]))
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    sh, so = sharpe_sortino(daily_returns(t, initial_equity, pnl, start, end))
    streak = best = 0
    for x in pnl:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    rm = core_metrics(r, risk_pct)
    return {"trades": len(t), "net_profit": round(float(pnl.sum()), 2),
            "gross_profit": round(float(wins.sum()), 2), "gross_loss": round(float(losses.sum()), 2),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if losses.sum() < 0 else None,
            "win_rate_pct": round(100 * len(wins) / len(t), 1),
            "expectancy": round(float(pnl.mean()), 2), "expectancy_r": round(float(r.mean()), 4),
            "avg_r": round(float(r.mean()), 4), "avg_win_r": rm.get("avg_win_r"), "avg_loss_r": rm.get("avg_loss_r"),
            "return_pct": round(float(eq[-1] / initial_equity - 1) * 100, 2),
            "max_drawdown": round(float(-dd["drawdown"].min()), 2),
            "max_drawdown_pct": round(float(-dd["drawdown_pct"].min()), 2),
            "max_consecutive_losses": int(best), "sharpe": sh, "sortino": so, "p_value": rm.get("p_value")}


def _group(trades: pd.DataFrame, key: pd.Series, initial_equity: float, risk_pct: float) -> pd.DataFrame:
    rows = []
    for g, part in trades.groupby(key.to_numpy(), sort=True):
        m = layer_metrics(part, "net", initial_equity, risk_pct)
        rows.append({"group": g, **{k: m.get(k) for k in ("trades", "win_rate_pct", "net_profit", "profit_factor",
                                                          "expectancy_r", "max_drawdown_pct")}})
    return pd.DataFrame(rows)


def monthly_returns(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    """Net P/L and return % per calendar month of exit (equity at the start of the month)."""
    if trades.empty:
        return pd.DataFrame(columns=["month", "trades", "net_pnl", "return_pct", "net_r"])
    t = trades.sort_values("exit_time")
    month = pd.DatetimeIndex(t["exit_time"]).tz_convert("UTC").strftime("%Y-%m")
    g = t.groupby(month)
    pnl = g["net_pnl"].sum()
    start_eq = initial_equity + pnl.cumsum().shift(1, fill_value=0.0)
    return pd.DataFrame({"month": pnl.index, "trades": g.size().to_numpy(), "net_pnl": pnl.round(2).to_numpy(),
                         "return_pct": (pnl / start_eq * 100).round(2).to_numpy(),
                         "net_r": g["r"].sum().round(2).to_numpy()})


def performance(trades: pd.DataFrame, initial_equity: float = 10_000.0, risk_pct: float = 0.5,
                start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    """All analytics for a set of engine trades."""
    out = {"layers": {k: layer_metrics(trades, k, initial_equity, risk_pct, start, end) for k in LAYERS}}
    if trades.empty:
        return out | {k: pd.DataFrame() for k in ("side", "session", "regime", "strategy", "exit_reason",
                                                  "intrabar", "year", "monthly")}
    t = trades.sort_values("exit_time").reset_index(drop=True)
    yr = pd.Series(pd.DatetimeIndex(t["entry_time"]).year.astype(str))
    for name, key in (("side", t["side"]), ("session", t["session"]), ("regime", t["regime"]),
                      ("strategy", t["strategy_id"]), ("exit_reason", t["exit_reason"]),
                      ("intrabar", t["intrabar"]), ("year", yr)):
        out[name] = _group(t, key, initial_equity, risk_pct)
    out["monthly"] = monthly_returns(t, initial_equity)
    return out


def report_markdown(perf: dict, title: str = "Performance") -> str:
    """Markdown with the three layers side by side and every breakdown."""
    lay = perf["layers"]
    keys = [k for k in lay["net"].keys()] if lay["net"].get("trades") else ["trades"]
    rows = [{"metric": k, "before costs": lay["before_costs"].get(k), "after spread": lay["after_spread"].get(k),
             "NET (after all costs)": lay["net"].get(k)} for k in keys]
    md = [f"### {title}\n", "Gross = the same trades with spread, slippage and commission added back. "
          "Only the NET column is what an account would have received.\n",
          _table(pd.DataFrame(rows), ["metric", "before costs", "after spread", "NET (after all costs)"])]
    for k, name in (("side", "Long / short (net)"), ("session", "By session (net)"), ("regime", "By H4 regime (net)"),
                    ("strategy", "By strategy (net)"), ("exit_reason", "By exit (net)"),
                    ("intrabar", "By intrabar resolution (net)"), ("year", "By year (net)"),
                    ("monthly", "Monthly returns (net)")):
        df = perf.get(k)
        if df is not None and len(df):
            md += [f"\n{name}\n", _table(df, list(df.columns))]
    return "\n".join(md) + "\n"
