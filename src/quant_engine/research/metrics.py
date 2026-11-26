"""Costs and the 14 required performance metrics (rulebook rules 6 and 8).

Trades are a DataFrame with at least ``entry_time``, ``side`` (BUY / SELL), ``r`` (result in
multiples of the initial risk, spread already paid by the bid/ask simulation) and ``risk``
(initial stop distance in price). :func:`apply_costs` subtracts slippage and commission;
:func:`full_report` computes every metric, with breakdowns by side, session, month and regime.
Account results use 0.5% (``risk_pct``) of current equity per trade, compounded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CostsConfig
from ..features.sessions import ny_minutes
from .miner import t_stats
from .runner import _table

# Rulebook rule 11 sessions, New York clock (so daylight saving is handled):
# Asia 19:00-03:00, London 03:00-08:00, London-NY overlap 08:00-12:00, New York 12:00-17:00.
SESSION_ORDER = ("asia", "london", "london_ny_overlap", "new_york", "off")


def session_of(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Rule-11 session of each UTC timestamp."""
    m = ny_minutes(times)
    return np.select([(m >= 19 * 60) | (m < 3 * 60), m < 8 * 60, m < 12 * 60, m < 17 * 60],
                     ["asia", "london", "london_ny_overlap", "new_york"], "off")


def extra_cost_price(costs: CostsConfig, point: float, fills: int = 2) -> float:
    """Slippage (``fills`` market / stop fills) plus commission, in price units per unit traded."""
    return fills * costs.slippage_points * point + costs.commission_per_lot / costs.contract_size


def apply_costs(trades: pd.DataFrame, costs: CostsConfig, point: float, fills: int = 2) -> pd.DataFrame:
    """Copy of ``trades`` with ``r_gross`` kept and ``r`` reduced by slippage and commission."""
    t = trades.copy()
    extra = extra_cost_price(costs, point, fills)
    t["r_gross"] = t["r"]
    t["cost_r"] = extra / t["risk"]
    t["r"] = t["r"] - t["cost_r"]
    return t


def _equity(r: np.ndarray, risk_pct: float) -> np.ndarray:
    return np.cumprod(1 + risk_pct / 100 * r) if len(r) else np.ones(1)


def core_metrics(r: np.ndarray, risk_pct: float = 0.5) -> dict:
    """Metrics 1-10 for one set of trade results (R)."""
    n = len(r)
    if n == 0:
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    eq = _equity(r, risk_pct)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    dd_pct = float(((peak[1:] - eq) / peak[1:]).max() * 100)
    cum = np.cumsum(r)
    dd_r = float((np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max())
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    st = t_stats(r)
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    return {"trades": n, "win_rate_pct": round(100 * len(wins) / n, 1),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else np.nan,
            "net_profit_pct": round(float(eq[-1] - 1) * 100, 1), "net_r": round(float(r.sum()), 1),
            "max_drawdown_pct": round(dd_pct, 1), "max_drawdown_r": round(dd_r, 1),
            "avg_reward_risk": round(avg_w / -avg_l, 2) if avg_l < 0 else np.nan,
            "expectancy_r": round(float(r.mean()), 3),
            "avg_win_r": round(avg_w, 2), "avg_loss_r": round(avg_l, 2),
            "max_consecutive_losses": int(best), "p_value": round(st["p"], 4)}


def _by(trades: pd.DataFrame, key: pd.Series, risk_pct: float, order=None) -> pd.DataFrame:
    rows = []
    groups = trades.groupby(key.to_numpy(), sort=True)
    names = [g for g in (order or []) if g in groups.groups] + [g for g in groups.groups if g not in (order or [])]
    for g in names:
        m = core_metrics(groups.get_group(g)["r"].to_numpy(), risk_pct)
        rows.append({"group": g, **{k: m.get(k) for k in ("trades", "win_rate_pct", "profit_factor",
                                                          "expectancy_r", "net_r", "net_profit_pct",
                                                          "max_drawdown_pct")}})
    return pd.DataFrame(rows)


def full_report(trades: pd.DataFrame, risk_pct: float = 0.5) -> dict:
    """All 14 metrics: ``overall`` (1-10) plus tables ``side``, ``session``, ``month``, ``regime``."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    out = {"overall": core_metrics(t["r"].to_numpy(), risk_pct)}
    if t.empty:
        return out | {k: pd.DataFrame() for k in ("side", "session", "month", "regime")}
    sess = t["session"] if "session" in t else pd.Series(session_of(t["entry_time"]))
    month = pd.Series(pd.DatetimeIndex(t["entry_time"]).strftime("%Y-%m"))
    out["side"] = _by(t, t["side"], risk_pct, ["BUY", "SELL"])
    out["session"] = _by(t, sess, risk_pct, list(SESSION_ORDER))
    out["month"] = _by(t, month, risk_pct)
    out["regime"] = _by(t, t["regime"], risk_pct) if "regime" in t else pd.DataFrame()
    return out


def report_markdown(title: str, rep: dict) -> str:
    """Markdown block with every metric and breakdown."""
    o = rep["overall"]
    lines = [f"### {title}\n", _table(pd.DataFrame([o]), list(o.keys()))]
    for k, name in (("side", "Long / short"), ("session", "By session (New York clock)"),
                    ("regime", "By market regime"), ("month", "Monthly")):
        df = rep.get(k)
        if df is not None and len(df):
            lines += [f"\n{name}\n", _table(df, list(df.columns))]
    return "\n".join(lines) + "\n"
