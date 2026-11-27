"""Performance metrics of breakout trades, in R (risk multiples) with costs included."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _streak(neg: np.ndarray) -> int:
    best = cur = 0
    for x in neg:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def equity_r(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net R by exit time (the equity curve in R)."""
    if trades.empty:
        return pd.DataFrame(columns=["time", "cum_r", "drawdown_r"])
    t = trades.sort_values("exit_time")
    cum = t["r"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return pd.DataFrame({"time": t["exit_time"].to_numpy(), "cum_r": cum, "drawdown_r": cum - peak})


def longest_recovery(trades: pd.DataFrame) -> dict:
    """Longest time from an equity peak (in R) to the first new high; an unfinished one counts to the
    last trade and is flagged."""
    eq = equity_r(trades)
    if eq.empty:
        return {"days": 0, "from": None, "to": None, "recovered": True}
    times = pd.DatetimeIndex(eq["time"])
    peak, peak_t, under = 0.0, pd.Timestamp(trades["entry_time"].min()), False
    best = {"days": 0.0, "from": None, "to": None, "recovered": True}
    for v, t in zip(eq["cum_r"], times):
        if v >= peak - 1e-12:
            if under and (t - peak_t).total_seconds() / 86400 > best["days"]:
                best = {"days": (t - peak_t).total_seconds() / 86400, "from": peak_t, "to": t, "recovered": True}
            peak, peak_t, under = max(peak, v), t, False
        else:
            under = True
    if under and (times[-1] - peak_t).total_seconds() / 86400 > best["days"]:
        best = {"days": (times[-1] - peak_t).total_seconds() / 86400, "from": peak_t, "to": times[-1],
                "recovered": False}
    best["days"] = round(best["days"], 1)
    for k in ("from", "to"):
        best[k] = None if best[k] is None else pd.Timestamp(best[k]).strftime("%Y-%m-%d")
    return best


def summary(trades: pd.DataFrame) -> dict:
    """The spec's headline metrics for closed trades (``r`` = net R)."""
    t = trades[trades["status"].isin(["tp_hit", "sl_hit", "expired"])] if len(trades) else trades
    n = len(t)
    if n == 0:
        return {"trades": 0}
    r = t.sort_values("exit_time")["r"].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else math.inf
    eq = equity_r(t)
    hold = (pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).dt.total_seconds() / 3600
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    tstat = float(r.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    return {
        "trades": n, "win_rate_pct": round(100 * len(wins) / n, 1), "profit_factor": round(pf, 2),
        "expectancy_r": round(float(r.mean()), 3), "net_r": round(float(r.sum()), 2),
        "max_drawdown_r": round(float(-eq["drawdown_r"].min()), 2), "max_losing_streak": _streak(r <= 0),
        "avg_hold_hours": round(float(hold.mean()), 1), "avg_sessions": round(float(t["sessions_held"].mean()), 2),
        "tp_hit": int((t["status"] == "tp_hit").sum()), "sl_hit": int((t["status"] == "sl_hit").sum()),
        "expired": int((t["status"] == "expired").sum()), "t_stat": None if math.isnan(tstat) else round(tstat, 2),
        "gross_r_per_trade": round(float(t["r_gross"].mean()), 3),
        "cost_r_per_trade": round(float((t["r_gross"] - t["r"]).mean()), 3),
        "same_candle_stop_first": int(t.get("note", pd.Series(dtype=str)).astype(str).str.contains("same M5").sum())
        if "note" in t else 0,
        "longest_recovery": longest_recovery(t),
    }


def by_period(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Net R per month (``freq='M'``) or year (``'Y'``) by entry time."""
    if trades.empty:
        return pd.DataFrame(columns=["period", "trades", "net_r", "win_rate_pct"])
    t = trades[trades["status"].isin(["tp_hit", "sl_hit", "expired"])].copy()
    if t.empty:
        return pd.DataFrame(columns=["period", "trades", "net_r", "win_rate_pct"])
    et = pd.DatetimeIndex(t["entry_time"]).tz_convert("UTC").tz_localize(None)
    t["period"] = et.to_period(freq).astype(str)
    g = t.groupby("period")["r"]
    return pd.DataFrame({"trades": g.size(), "net_r": g.sum().round(2),
                         "win_rate_pct": (100 * g.apply(lambda x: (x > 0).mean())).round(1)}).reset_index()
