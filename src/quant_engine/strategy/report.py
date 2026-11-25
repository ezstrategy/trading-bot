"""Backtest statistics and breakdowns (by regime, grade, side, session)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stats(trades: pd.DataFrame, risk_pct: float = 0.5) -> dict:
    """Headline numbers for filled trades. R = multiples of risk; % assumes ``risk_pct`` per R."""
    f = trades[trades["filled"]] if len(trades) else trades
    if f.empty:
        return {"trades": 0}
    r = f["r"].astype(float)
    wins, losses = r[r > 0], r[r < 0]
    eq = r.cumsum()
    dd = float((eq.cummax().clip(lower=0) - eq).max())
    return {
        "trades": int(len(f)),
        "win_rate_pct": round(100 * float((r > 0).mean()), 1),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() else None,
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
        "max_drawdown_r": round(dd, 2),
        "return_pct": round(float(r.sum()) * risk_pct, 2),
        "max_drawdown_pct": round(dd * risk_pct, 2),
    }


def breakdown(trades: pd.DataFrame, by: str, risk_pct: float = 0.5) -> pd.DataFrame:
    """``stats`` per value of column ``by`` (filled trades only)."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    f = trades[trades["filled"]]
    for key, g in f.groupby(by, sort=True):
        rows.append({by: key, **stats(g, risk_pct)})
    return pd.DataFrame(rows)


def funnel(trades: pd.DataFrame, rejections: dict[str, int]) -> pd.DataFrame:
    """Why candidate candles did or did not become trades."""
    rows = [{"stage": k, "count": v} for k, v in sorted(rejections.items())]
    if len(trades):
        for k, v in trades["outcome"].value_counts().items():
            rows.append({"stage": f"signal -> {k}", "count": int(v)})
    return pd.DataFrame(rows)


def monthly(trades: pd.DataFrame) -> pd.DataFrame:
    """Filled trades and R per calendar month."""
    f = trades[trades["filled"]] if len(trades) else trades
    if f.empty:
        return pd.DataFrame(columns=["month", "trades", "total_r"])
    m = pd.to_datetime(f["signal_time"], utc=True).dt.strftime("%Y-%m")
    g = f.groupby(m)["r"]
    return pd.DataFrame({"month": g.sum().index, "trades": g.count().to_numpy(),
                         "total_r": g.sum().round(2).to_numpy()})


def to_markdown(title: str, meta: dict, trades: pd.DataFrame, rejections: dict, risk_pct: float) -> str:
    """A readable report."""
    def tbl(df: pd.DataFrame) -> str:
        if df.empty:
            return "_none_\n"
        cols = list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v)
                                            for v in r.tolist()) + " |")
        return "\n".join(lines) + "\n"

    s = stats(trades, risk_pct)
    out = [f"# {title}\n"]
    out += [f"- **{k}**: {v}" for k, v in meta.items()]
    out += ["\n## Overall (filled trades)\n", tbl(pd.DataFrame([s]))]
    if s.get("trades", 0) < 30:
        out += ["> Fewer than 30 filled trades: too few to judge the strategy. "
                "Treat these numbers as a mechanics check only.\n"]
    for by, name in (("trend_regime", "Market regime (H4 EMA50 trend)"),
                     ("h4_structure", "H4 structure state"), ("grade", "Grade"),
                     ("side", "Direction"), ("kill_zone", "Kill zone")):
        out += [f"\n## By {name}\n", tbl(breakdown(trades, by, risk_pct))]
    out += ["\n## By month\n", tbl(monthly(trades))]
    out += ["\n## Signal funnel\n", tbl(funnel(trades, rejections))]
    return "\n".join(out)
