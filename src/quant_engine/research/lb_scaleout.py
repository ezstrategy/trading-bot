"""EXP-018: scale-out exit on the unchanged gold London breakout (owner idea, 2026-11-27).

``python -m quant_engine lb-scaleout`` (lb_scaleout.bat). Owner: "3 lots of 0.01, targets T1 T2 T3, so if
gold reaches T1 and goes down we still get a profit". Fixed before the run (rules 1 and 7):

* Entry, stop, trend filter, close time: the EXP-016 original, unchanged (stop-order entry).
* The position is split into 3 equal parts on the same entry and stop. Part k exits at its target Tk,
  at the stop, or at 16:00 NY. After T1 the remaining parts' stop moves to entry (break-even).
* Versions: **S1 (main) T1 1R / T2 2R / T3 3R, BE after T1**; nearby S2 0.75 / 1.5 / 3R (BE after
  T1 = 0.75R); S3 1 / 2 / 4R; S4 1 / 2 / 3R with no break-even (shows what the BE step does).
* Same costs as EXP-016 (historical spread, at least $0.08, 2 points slippage per fill per unit,
  commission 0). A trade's R is the average of its three parts (equal size, same stop distance).
* Compared with the original day by day (paired block bootstrap). Same parts as EXP-016; the owner's
  main window is 2023 onward. Win rate is reported two ways: trade net > 0, and T1 reached.
* Lot sizing: each part rounded down to 0.01 lot, so the smallest position is 3 x 0.01 lots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .classics import lb_day_groups
from .events import regime_per_bar
from .lb_phase2 import ORIGINAL, _detail, _f, _row, _split_of, lb_trades, paired_boot, walk_exit, with_costs
from .model import regime_kwargs
from .regime4 import regime4_series
from .runner import _table

log = get_logger(__name__)

VERSIONS = {
    "S1 1/2/3R, BE after T1 (main)": {"targets": (1.0, 2.0, 3.0), "be": True},
    "S2 0.75/1.5/3R, BE after T1": {"targets": (0.75, 1.5, 3.0), "be": True},
    "S3 1/2/4R, BE after T1": {"targets": (1.0, 2.0, 4.0), "be": True},
    "S4 1/2/3R, no BE": {"targets": (1.0, 2.0, 3.0), "be": False},
}
WINDOWS = (("since 2023 (main)", "2023-01-01", "2100-01-01"), ("last 2 years", None, "2100-01-01"),
           ("2015-2019", "2015-01-01", "2020-01-01"), ("2020-2022", "2020-01-01", "2023-01-01"),
           ("all", "2000-01-01", "2100-01-01"))


def scale_out(m5: pd.DataFrame, base: pd.DataFrame, sp: float, targets: tuple, be: bool,
              close_end: np.ndarray, be_at: float | None = None, be_first: bool = False) -> pd.DataFrame:
    """Re-run the exits of the original trades in ``base`` as three equal parts. Parts 2 and 3 move
    their stop to entry once price reaches ``be_at`` (default T1); ``be_first`` applies it to part 1 too."""
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = pd.DatetimeIndex(m5["time"])
    rows = []
    for _, tr in base.iterrows():
        i0, side = int(tr["entry_i"]), 1 if tr["side"] == "BUY" else -1
        entry, stop, risk = tr["entry"], tr["stop"], tr["risk"]
        end_i = int(close_end[int(tr["day"])])
        parts, exits = [], []
        for k, tg in enumerate(targets):
            be_r = (be_at if be_at is not None else targets[0]) if (be and (k > 0 or be_first)) else None
            j, px, why = walk_exit(i0, end_i, side, entry, stop, entry + side * tg * risk, o, h, l, c, sp, be_r)
            parts.append(side * (px - entry) / risk)
            exits.append((j, why))
        rows.append({**{k: tr[k] for k in ("day", "entry_i", "entry_time", "side", "entry", "stop", "risk")},
                     "exit_time": t[max(j for j, _ in exits)], "r": float(np.mean(parts)),
                     "part1_r": parts[0], "part2_r": parts[1], "part3_r": parts[2],
                     "t1_hit": exits[0][1] == "target", "exit_reason": "/".join(w for _, w in exits)})
    return pd.DataFrame(rows)


def _min_balance(stop: np.ndarray, contract: float, risk_pct: float, lots: float = 0.03) -> np.ndarray:
    return lots * contract * stop / (risk_pct / 100)


def run_lb_scaleout(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                    start: pd.Timestamp, end: pd.Timestamp | None = None, source_note: str = "") -> dict:
    """Run every version once, write ``report.md``, ``detail/*.md`` and trade CSVs."""
    sc = cfg.strategy
    costs, risk_pct = sc.research.costs, sc.backtest.risk_pct
    out = Path(out_dir)
    (out / "detail").mkdir(parents=True, exist_ok=True)
    base_sp = costs.min_spread_points * point
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, Timeframe.H4, start - pd.Timedelta(days=60), end)
    if m5.empty or h4.empty:
        raise ValueError(f"No stored {dataset} M5/H4 data")
    t = pd.DatetimeIndex(m5["time"])
    reg2 = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
    reg4 = regime_per_bar(t, h4, regime4_series(h4, sc.research.regime4))
    groups = lb_day_groups(m5, sc.sessions)
    bar_sp = m5["spread"].to_numpy(float) * point if "spread" in m5 else np.full(len(m5), base_sp)
    from ..features.sessions import ny_minutes
    mins = ny_minutes(t)
    close_end = np.array([g[(mins[g] >= 60) & (mins[g] < ORIGINAL["close_min"])][-1]
                          if ((mins[g] >= 60) & (mins[g] < ORIGINAL["close_min"])).any() else g[-1] for g in groups])
    raw_orig = lb_trades(m5, base_sp, ORIGINAL, reg2, groups)
    last = t[-1]
    wins = [(n, pd.Timestamp(a, tz="UTC") if a else last - pd.DateOffset(years=2), pd.Timestamp(b, tz="UTC"))
            for n, a, b in WINDOWS]

    def prep(tr, k=1.0):
        x = with_costs(tr, bar_sp, base_sp, costs, point, k)
        x["split"] = _split_of(x["entry_time"])
        x["regime"] = reg4[x["entry_i"].to_numpy(int)]
        return x

    strategies = {"Original (3R, BE 1R, one position)": raw_orig}
    for name, v in VERSIONS.items():
        strategies[name] = scale_out(m5, raw_orig, base_sp, v["targets"], v["be"], close_end)
    orig = prep(raw_orig)
    rows, summary = [], {}
    for name, raw in strategies.items():
        tr = prep(raw)
        st = {f"x{k:g}": prep(raw, k) for k in (1.0, 1.5, 2.0)}
        for wn, a, b in wins:
            sel = (pd.DatetimeIndex(tr["entry_time"]) >= a) & (pd.DatetimeIndex(tr["entry_time"]) < b)
            x = tr[sel]
            m = _row(x["r"].to_numpy(), risk_pct)
            row = {"exit": name, "window": wn, **m,
                   "T1_reached%": round(100 * float(x["t1_hit"].mean()), 1) if "t1_hit" in x and len(x) else np.nan,
                   "exp_x2": round(float(st["x2"][sel]["r"].mean()), 3) if sel.any() else np.nan}
            if name.startswith("Original"):
                row |= {"diff_vs_original": 0.0, "p": np.nan}
            else:
                so = orig[(pd.DatetimeIndex(orig["entry_time"]) >= a) & (pd.DatetimeIndex(orig["entry_time"]) < b)]
                d = (x.set_index("day")["r"] - so.set_index("day")["r"]).dropna().to_numpy()
                row["diff_vs_original"], row["p"] = paired_boot(d)
            rows.append(row)
        (out / "detail" / f"{_slug(name)}.md").write_text(_detail(name, tr, risk_pct, st), encoding="utf-8")
        tr.to_csv(out / f"trades_{_slug(name)}.csv", index=False)
        main = next(r_ for r_ in rows if r_["exit"] == name and r_["window"] == "since 2023 (main)")
        summary[name] = {k: main.get(k) for k in ("trades", "win%", "T1_reached%", "PF", "exp_R", "DD%",
                                                   "max_losses", "diff_vs_original", "p")}
    tab = pd.DataFrame(rows)
    cols = ["exit", "window", "trades", "win%", "T1_reached%", "PF", "exp_R", "exp_CI95", "DD%", "avg_win",
            "avg_loss", "max_losses", "exp_x2", "diff_vs_original", "p"]
    # class for S1 (main) by the EXP-016 rules: screen 2015-2019, confirm 2020-2022, then 2023+
    def g(name, wn, k):
        return next((r_[k] for r_ in rows if r_["exit"] == name and r_["window"] == wn), np.nan)
    verdicts = {}
    for name in VERSIONS:
        dev_d, dev_p = _f(g(name, "2015-2019", "diff_vs_original")), _f(g(name, "2015-2019", "p"), 1.0)
        val_d, val_p = _f(g(name, "2020-2022", "diff_vs_original")), _f(g(name, "2020-2022", "p"), 1.0)
        tst_d = _f(g(name, "since 2023 (main)", "diff_vs_original"))
        verdicts[name] = ("better than the original (development, validation and 2023+)"
                          if dev_d > 0 and dev_p < 0.05 and val_d > 0 and val_p < 0.05 / len(VERSIONS) and tst_d > 0
                          else "not better than the original" if dev_d <= 0 or val_d <= 0 or tst_d <= 0
                          else "not proven better (not significant)")
    stops = orig["risk"].to_numpy()
    mb = _min_balance(stops, costs.contract_size, risk_pct)
    md = ["# EXP-018: scale-out exit (3 equal parts) on the unchanged gold London breakout\n",
          f"- **data**: {source_note or dataset}, {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}",
          "- **entries**: exactly the original's trades (same days, entry, stop); only the exit changes",
          f"- **costs**: historical spread (at least ${base_sp:.2f}), slippage {costs.slippage_points:g} points per fill, "
          f"commission {costs.commission_per_lot:g}; exp_x2 = double costs",
          f"- **risk**: {risk_pct:g}% per trade, compounded; trade R = average of the three parts",
          "- **rules fixed before the run**: module docstring of `research/lb_scaleout.py`\n",
          "## Results by window (2023 onward is the main window)\n",
          "win% = trade closed with net R > 0; T1_reached% = the first part hit its target. diff_vs_original = "
          "R per trade minus the original on the same trades; p = one-sided block-bootstrap p that it is above 0.\n",
          _table(tab, cols),
          "\n## Verdict per version (same rule as EXP-016: better on 2015-2019 with p < 0.05, confirmed on "
          f"2020-2022 with p < {0.05 / len(VERSIONS):.4f}, and better since 2023)\n",
          _table(pd.DataFrame([{"exit": k, "verdict": v} for k, v in verdicts.items()]), ["exit", "verdict"]),
          "\n## Lot sizing with 3 parts of at least 0.01 lot\n",
          f"Smallest position 3 x 0.01 = 0.03 lots. Account needed so that 0.03 lots risks at most {risk_pct:g}%: "
          f"median stop ${np.median(stops):.2f} -> ${np.median(mb):,.0f}; 90% of trades -> ${np.percentile(mb, 90):,.0f}; "
          f"95% -> ${np.percentile(mb, 95):,.0f}. Below that, 3 x 0.01 lots risks more than {risk_pct:g}% on wide-range days.\n",
          "\nEvery version's monthly, yearly, long / short, session, regime and cost tables: `detail/<name>.md`.",
          "\n_Research on past data, signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    tab.to_csv(out / "results.csv", index=False)
    log.info("lb_scaleout_done", extra={"versions": len(VERSIONS)})
    return {"since_2023": summary, "verdicts": verdicts, "report": str(out / "report.md")}


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")
