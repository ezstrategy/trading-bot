"""Polish the London breakout (Asian range) without fooling ourselves.

``python -m quant_engine london`` (polish_london.bat). Every combination of the changes in
:data:`GRID` is backtested. The best one is chosen on the **older data only** and then judged on
the newer, unseen data. The report also shows each change on its own, a year-by-year table, and
how well "good on the older data" carried over to the unseen data across all versions.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..features.sessions import trading_day
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .classics import LB_DEFAULT, lb_day_groups, london_breakout
from .events import regime_per_bar
from .miner import benjamini_hochberg, t_stats
from .model import regime_kwargs
from .runner import _table

log = get_logger(__name__)

GRID = {"target_r": [1.5, 2.0, 2.5, 3.0, 4.0], "be_r": [None, 1.0, 1.5], "close_min": [11 * 60, 14 * 60, 16 * 60],
        "range_filter": [True, False], "stop": ["range", "mid"], "trend": ["none", "with"]}
# Versions aimed at a 50%+ win rate: smaller targets, half closed at TP1, earlier break-even.
GRID_HIGH_WIN = {"target_r": [0.5, 0.75, 1.0, 1.5, 2.0], "tp1_r": [None, 0.5, 1.0], "be_r": [None, 0.5, 1.0],
                 "close_min": [11 * 60, 14 * 60, 16 * 60], "range_filter": [True, False], "stop": ["range", "mid"],
                 "trend": ["none", "with"]}
GRIDS = {"standard": GRID, "high-win": GRID_HIGH_WIN}
NAMES = {"target_r": "target (R)", "tp1_r": "TP1: half closed at (R)", "be_r": "break-even at (R)", "close_min": "close by (New York)",
         "range_filter": "skip tiny / huge ranges", "stop": "stop at", "trend": "trend filter"}


def _label(k: str, v) -> str:
    if k == "close_min":
        return f"{v // 60:02d}:{v % 60:02d} NY"
    if k in ("be_r", "tp1_r"):
        return ("never" if k == "be_r" else "none") if v is None else f"{v:g}R"
    if k == "stop":
        return "other side of range" if v == "range" else "middle of range"
    if k == "trend":
        return "off" if v == "none" else "only with H4 trend"
    return str(v)


def r_stats(r: np.ndarray, risk_pct: float) -> dict:
    """Trades, win rate, average / total R, profit factor, max drawdown (R), p, return at ``risk_pct``."""
    st = t_stats(r)
    eq = np.cumsum(r) if len(r) else np.zeros(1)
    dd = float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max())
    w, lo = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": st["trades"], "win_pct": st["win_rate_pct"], "avg_r": st["avg_r"], "total_r": st["total_r"],
            "pf": round(float(w / lo), 2) if lo > 0 else np.nan, "max_dd_r": round(dd, 1), "p": round(st["p"], 4),
            "return_pct": round(risk_pct * float(r.sum()), 1)}


def run_london(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
               start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, source_note: str = "",
               spread_points: float | None = 8.0, discover_frac: float = 0.6, min_trades: int = 300,
               grid: str = "standard", min_win: float = 0.0) -> dict:
    """Backtest every version in ``GRIDS[grid]`` and write ``report.md``, ``versions.csv`` and trade files.
    The chosen version is the best average R on the older data among versions with at least
    ``min_trades`` older trades and an older win rate of at least ``min_win`` %."""
    g_ = GRIDS[grid]
    sc = cfg.strategy
    risk_pct = sc.backtest.risk_pct
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, Timeframe.H4, start, end)
    if m5.empty or h4.empty:
        raise ValueError(f"No stored {dataset} M5/H4 data")
    if spread_points is None:      # the broker's own spread: median bar spread over the last year
        recent = m5[m5["time"] >= m5["time"].iat[-1] - pd.Timedelta(days=365)]
        spread_points = float(recent["spread"].median()) if "spread" in recent and recent["spread"].notna().any() else 0.0
    sp = spread_points * point
    t = pd.DatetimeIndex(m5["time"])
    split = t[int(len(t) * discover_frac)]
    regime = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
    groups = lb_day_groups(m5, sc.sessions)
    keys = list(g_)
    names = {k: NAMES[k] for k in keys}
    rows, trades = [], {}
    combos = list(itertools.product(*g_.values()))
    base_key = tuple(LB_DEFAULT[k] for k in keys)
    if base_key not in combos:
        combos.append(base_key)                       # always include the published rules
    for combo in combos:
        p = dict(zip(keys, combo))
        if p.get("tp1_r") is not None and p["tp1_r"] >= p["target_r"]:
            continue
        tr = pd.DataFrame(london_breakout(m5, sp, sc.sessions, p, regime, groups))
        if tr.empty:
            continue
        old = tr["entry_time"] < split
        so, su = r_stats(tr.loc[old, "r"].to_numpy(), risk_pct), r_stats(tr.loc[~old, "r"].to_numpy(), risk_pct)
        rows.append({**{names[k]: _label(k, v) for k, v in p.items()}, **{f"older_{k}": v for k, v in so.items()},
                     **{f"unseen_{k}": v for k, v in su.items()}, "_key": combo})
        trades[combo] = tr
    res = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["older_bh_pass"] = benjamini_hochberg(res["older_p"].to_numpy(), sc.research.fdr_q)

    base = res[res["_key"].map(lambda k: k == base_key)].iloc[0]
    eligible = res[(res["older_trades"] >= min_trades) & (res["older_win_pct"] >= min_win)]
    if eligible.empty:
        eligible = res
    best = eligible.sort_values("older_avg_r", ascending=False).iloc[0]
    n_unseen_tests = 1
    verdict = ("confirmed" if best["unseen_avg_r"] > 0 and best["unseen_p"] < 0.05 / n_unseen_tests
               else "held up (not proven)" if best["unseen_avg_r"] > 0 else "failed")

    # one change at a time from the chosen version
    one = []
    bkey = dict(zip(keys, best["_key"]))
    for k in keys:
        for v in g_[k]:
            key = tuple(v if kk == k else bkey[kk] for kk in keys)
            r = res[res["_key"].map(lambda kk_: kk_ == key)]
            if len(r):
                r = r.iloc[0]
                one.append({"change": names[k], "value": _label(k, v) + (" (chosen)" if v == bkey[k] else ""),
                            "older_trades": r["older_trades"], "older_win_pct": r["older_win_pct"],
                            "older_avg_r": r["older_avg_r"],
                            "unseen_trades": r["unseen_trades"], "unseen_avg_r": r["unseen_avg_r"],
                            "unseen_win_pct": r["unseen_win_pct"], "unseen_return_pct": r["unseen_return_pct"]})
    onedf = pd.DataFrame(one)

    top = eligible.sort_values("older_avg_r", ascending=False).head(20)
    carried = int((top["unseen_avg_r"] > 0).sum())
    corr = float(res[["older_avg_r", "unseen_avg_r"]].corr().iloc[0, 1])

    def yearly(tr: pd.DataFrame) -> pd.DataFrame:
        y = tr.assign(year=pd.DatetimeIndex(tr["entry_time"]).year)
        return pd.DataFrame([{"year": int(yr), **r_stats(g["r"].to_numpy(), risk_pct)} for yr, g in y.groupby("year")])

    bt, btr = trades[base_key], trades[best["_key"]]
    bt.to_csv(out / "trades_published.csv", index=False)
    btr.to_csv(out / "trades_best.csv", index=False)
    res.drop(columns="_key").to_csv(out / "versions.csv", index=False)
    crossed = int((trading_day(btr["entry_time"], sc.sessions)
                   != trading_day(pd.DatetimeIndex(btr["exit_time"]) + Timeframe.M5.delta - pd.Timedelta(seconds=1),
                                  sc.sessions)).sum())

    pcols = list(names.values())
    hw = res[res["older_win_pct"] >= 50]
    hw_ok = hw[(hw["unseen_win_pct"] >= 50) & (hw["unseen_avg_r"] > 0) & (hw["older_avg_r"] > 0)]
    hw_top = hw[hw["older_avg_r"] > 0].sort_values("older_avg_r", ascending=False).head(15)
    scols = ["trades", "win_pct", "avg_r", "total_r", "pf", "max_dd_r", "return_pct", "p"]
    both = pd.DataFrame([{"version": "published rules", "data": d, **{c: base[f"{d}_{c}"] for c in scols}}
                         for d in ("older", "unseen")] +
                        [{"version": "best on older data", "data": d, **{c: best[f"{d}_{c}"] for c in scols}}
                         for d in ("older", "unseen")])
    md = [f"# London breakout on {dataset}: what makes it better?\n",
          f"- **versions searched**: {grid}" + (f"; chosen among versions winning at least {min_win:g}% on the older data"
                                               if min_win else ""),
          f"- **data**: {source_note or dataset}",
          f"- **period (UTC)**: {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}; unseen data from {split:%Y-%m-%d}",
          f"- **spread**: {sp:.5g} in price ({spread_points:g} points of {point:g}) on every trade; returns at {risk_pct:g}% risk per trade (simple sum)",
          f"- **versions tested**: {len(res)} (every combination of the changes below)",
          f"- **trades held past the 17:00 New York rollover (swap)**: {crossed}\n",
          "## Published rules vs the best version chosen on the older data\n",
          "Best version: " + ", ".join(f"{k} = {best[k]}" for k in pcols) + f" -> **{verdict}** on the unseen data.\n",
          _table(both, ["version", "data", *scols]),
          f"\n## Versions winning 50%+ (older data: {len(hw)}; also profitable on older data: "
          f"{int((hw['older_avg_r'] > 0).sum())}; still 50%+ and profitable on unseen data: {len(hw_ok)})\n",
          _table(hw_top, [*pcols, "older_trades", "older_win_pct", "older_avg_r", "unseen_trades", "unseen_win_pct",
                          "unseen_avg_r", "unseen_return_pct"]),
          "\n## Each change on its own (everything else as the chosen version)\n",
          _table(onedf, list(onedf.columns)),
          "\n## Did 'best on the older data' carry over?\n",
          f"- Of the 20 best versions on the older data, **{carried}** were also profitable on the unseen data.",
          f"- Correlation between older and unseen average R across all {len(res)} versions: **{corr:.2f}** "
          "(near 0 = picking on the older data is luck; clearly above 0 = the changes really matter).",
          f"- Versions passing the false-discovery filter on the older data: {int(res['older_bh_pass'].sum())}\n",
          _table(top, [*pcols, "older_trades", "older_avg_r", "unseen_trades", "unseen_avg_r", "unseen_return_pct"]),
          "\n## Year by year: published rules\n", _table(yearly(bt), ["year", *scols]),
          "\n## Year by year: best version\n", _table(yearly(btr), ["year", *scols]),
          "\n_Research on past data, not a promise of future profit. Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"instrument": dataset, "spread_points": spread_points, "point": point, "versions": len(res), "best": {k: best[k] for k in pcols}, "verdict": verdict,
               "best_unseen_avg_r": best["unseen_avg_r"], "published_unseen_avg_r": base["unseen_avg_r"],
               "best_unseen_win_pct": best["unseen_win_pct"], "high_win_versions_ok_unseen": int(len(hw_ok)),
               "top20_carried": carried, "corr": round(corr, 2), "report": str(out / "report.md")}
    log.info("london_done", extra={k: v for k, v in summary.items() if k != "best"})
    return summary
