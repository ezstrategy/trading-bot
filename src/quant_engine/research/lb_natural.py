"""EXP-020: natural stop first, position sizing second (owner rule, 2026-11-27).

``python -m quant_engine lb-natural`` (lb_natural.bat). Owner: "The stop-loss must be determined by
market structure and strategy invalidation, not account affordability. I would rather miss a trade
than enter with an artificially tight stop-loss." Fixed before the run (rules 1 and 7):

* Entry and exits: the EXP-016 original, unchanged (Asian range 20:00-01:00 NY, stop orders, H4 trend
  filter, 3R, break-even at 1R, close 16:00 NY).
* Stop: the opposite boundary of the Asian range, **never tightened**. The original capped the stop at
  0.6% of price (it moved the stop closer on wide-range days); that cap breaks the owner's rule, so the
  main version here drops it ("natural stop"). The capped original is shown next to it, unchanged.
* Sizing: lots for at most 2% of current equity at the natural stop, costs included (spread, slippage on
  both fills, commission), rounded down to 0.01; if that is below 0.01 lot, i.e. 0.01 lot would risk more
  than 2%, the signal is skipped. Margin at 1:500 is checked too (skip if it exceeds equity). Equity
  compounds trade by trade from the starting balance.
* Balances $500, $1,000, $2,500, $5,000; windows since 2023 (main) and all history. Same costs as
  EXP-016. One run; the skip counts are the answer the owner asked for, not a tuned result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..live.sizing import lot_size
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .classics import lb_day_groups
from .events import regime_per_bar
from .lb_phase2 import ORIGINAL, _detail, _row, _split_of, boot_stats, lb_trades, with_costs
from .model import regime_kwargs
from .regime4 import regime4_series
from .runner import _table

log = get_logger(__name__)

NATURAL = {**ORIGINAL, "stop_cap": None}
VERSIONS = {"natural stop (owner rule, main)": NATURAL, "original, stop capped at 0.6%": ORIGINAL}
BALANCES = (500.0, 1000.0, 2500.0, 5000.0)
WINDOWS = (("since 2023 (main)", "2023-01-01"), ("all history", None))


def sized_run(tr: pd.DataFrame, balance: float, risk_pct: float = 2.0, contract: float = 100.0,
              volume_min: float = 0.01, volume_step: float = 0.01, leverage: float = 500.0) -> tuple[dict, pd.DataFrame]:
    """Walk ``tr`` (costed trades: entry, stop, risk, r, cost_r) in order with compounding equity.

    Each signal is sized by :func:`live.sizing.lot_size` with ``skip_if_over``: the stop is the trade's
    own stop, the extra cost per ounce is ``risk * cost_r`` (what costs add to a full-stop loss), and a
    signal whose 0.01 lot would lose more than ``risk_pct`` is skipped. Returns a summary and the
    per-signal ledger.
    """
    e, peak, max_dd = balance, balance, 0.0
    rows = []
    for x in tr.itertuples(index=False):
        extra = max(float(x.risk) * float(x.cost_r), 0.0)
        s = lot_size(e, risk_pct, float(x.entry), float(x.stop), contract, volume_min, volume_step,
                     cost_per_unit=extra, skip_if_over=True)
        reason = "0.01 lot over limit" if s.skipped else ""
        if not s.skipped and s.lots * contract * float(x.entry) / leverage > e:
            reason = "margin"
        if reason:
            rows.append({"entry_time": x.entry_time, "taken": False, "reason": reason, "lots": 0.0,
                         "min_lot_risk_pct": 100 * volume_min * contract * (float(x.risk) + extra) / e,
                         "risk_pct": 0.0, "pnl": 0.0, "equity": e, "r": float(x.r)})
            continue
        pnl = s.lots * contract * float(x.risk) * float(x.r)
        rows.append({"entry_time": x.entry_time, "taken": True, "reason": "", "lots": s.lots,
                     "min_lot_risk_pct": 100 * volume_min * contract * (float(x.risk) + extra) / e,
                     "risk_pct": 100 * s.risk_money / e, "pnl": pnl, "equity": e + pnl, "r": float(x.r)})
        e += pnl
        peak = max(peak, e)
        max_dd = max(max_dd, 1 - e / peak)
    led = pd.DataFrame(rows)
    if led.empty:
        return {"signals": 0}, led
    tk = led[led["taken"]]
    sk = led[~led["taken"]]
    gains, losses = tk.loc[tk["pnl"] > 0, "pnl"].sum(), -tk.loc[tk["pnl"] < 0, "pnl"].sum()
    summ = {"start_balance": balance, "signals": len(led), "taken": len(tk), "skipped": len(sk),
            "skipped%": round(100 * len(sk) / len(led), 1),
            "skipped_min_lot": int((sk["reason"] == "0.01 lot over limit").sum()),
            "skipped_margin": int((sk["reason"] == "margin").sum()),
            "end_balance": round(e, 2), "return%": round(100 * (e / balance - 1), 1),
            "max_dd%": round(100 * max_dd, 1), "PF_$": round(gains / losses, 2) if losses > 0 else np.nan,
            "taken_exp_R": round(float(tk["r"].mean()), 3) if len(tk) else np.nan,
            "skipped_exp_R": round(float(sk["r"].mean()), 3) if len(sk) else np.nan,
            "median_risk%": round(float(tk["risk_pct"].median()), 2) if len(tk) else np.nan,
            "max_risk%": round(float(tk["risk_pct"].max()), 2) if len(tk) else np.nan,
            "worst_loss%": round(float((-tk["pnl"] / (tk["equity"] - tk["pnl"])).max() * 100), 2) if len(tk) else np.nan,
            "first_signal": str(led["entry_time"].iloc[0])[:10], "last_signal": str(led["entry_time"].iloc[-1])[:10]}
    return summ, led


def sizing_table(trades: dict[str, pd.DataFrame], balances=BALANCES, windows=WINDOWS, risk_pct: float = 2.0,
                 leverage: float = 500.0) -> pd.DataFrame:
    """:func:`sized_run` for every version x window x balance."""
    rows = []
    for name, tr in trades.items():
        for wn, since in windows:
            x = tr if since is None else tr[pd.to_datetime(tr["entry_time"], utc=True) >= pd.Timestamp(since, tz="UTC")]
            for b in balances:
                s, _ = sized_run(x, b, risk_pct, leverage=leverage)
                rows.append({"version": name, "window": wn, **s})
    return pd.DataFrame(rows)


def run_lb_natural(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                   start: pd.Timestamp, end: pd.Timestamp | None = None, source_note: str = "",
                   balances=BALANCES, risk_pct: float = 2.0, leverage: float = 500.0) -> dict:
    """Build both versions' trades once, then the 2% sizing ledger; write ``report.md`` and CSVs."""
    sc = cfg.strategy
    costs, rep_risk = sc.research.costs, sc.backtest.risk_pct
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
    trades, st_all = {}, {}
    for name, p in VERSIONS.items():
        raw = lb_trades(m5, base_sp, p, reg2, groups)
        st = {}
        for k in (1.0, 1.5, 2.0):
            x = with_costs(raw, bar_sp, base_sp, costs, point, k)
            x["split"] = _split_of(x["entry_time"])
            x["regime"] = reg4[x["entry_i"].to_numpy(int)]
            st[f"x{k:g}"] = x
        trades[name], st_all[name] = st["x1"], st
    perf = []
    for name, tr in trades.items():
        for wn, since in WINDOWS:
            x = tr if since is None else tr[pd.DatetimeIndex(tr["entry_time"]) >= pd.Timestamp(since, tz="UTC")]
            r = x["r"].to_numpy()
            perf.append({"version": name, "window": wn, **_row(r, rep_risk), "p_boot": boot_stats(r)["p_boot"],
                         "stop_widened_vs_cap%": round(100 * float(x["stop_capped"].mean()), 1)
                         if name.startswith("original") else np.nan,
                         "median_stop_$": round(float(x["risk"].median()), 2),
                         "p95_stop_$": round(float(x["risk"].quantile(.95)), 2)})
    nat = trades["natural stop (owner rule, main)"]
    capped_days = int(trades["original, stop capped at 0.6%"]["stop_capped"].sum())
    sz = sizing_table(trades, balances, WINDOWS, risk_pct, leverage)
    pcols = ["version", "window", "trades", "win%", "PF", "exp_R", "exp_CI95", "DD%", "max_losses", "p_boot",
             "median_stop_$", "p95_stop_$", "stop_widened_vs_cap%"]
    scols = ["version", "window", "start_balance", "signals", "taken", "skipped", "skipped%", "skipped_margin",
             "end_balance", "return%", "max_dd%", "PF_$", "taken_exp_R", "skipped_exp_R", "median_risk%",
             "max_risk%", "worst_loss%"]
    md = ["# EXP-020: natural stop first, position sizing second\n",
          f"- **data**: {source_note}, {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}",
          "- **rules**: EXP-016 original entry and exits unchanged; stop = opposite Asian range boundary. "
          f"The original's 0.6% stop cap tightened the stop on {capped_days} of "
          f"{len(trades['original, stop capped at 0.6%'])} trades; the natural version never tightens it.",
          f"- **costs**: bar spread (at least ${base_sp:.2f}), {costs.slippage_points:g} points slippage per fill, "
          f"commission {costs.commission_per_lot:g} per lot; all included in the risk used for sizing",
          f"- **sizing**: at most {risk_pct:g}% of current equity at the stop, rounded down to 0.01 lot; skip if "
          f"0.01 lot would risk more; margin checked at 1:{leverage:g}; equity compounds\n",
          "## Skipped signals under the 2% rule\n", _table(sz, scols),
          "\nskipped = signals not taken because 0.01 lot would risk more than the limit (plus margin skips, "
          "shown separately). taken_exp_R / skipped_exp_R = average R of the signals taken / skipped. "
          "worst_loss% = largest single loss as % of equity before the trade (costs included, so it can sit "
          "slightly above the stop-only figure but never above the limit at the full stop).\n",
          "## Strategy result per trade (R, costs included, before sizing)\n", _table(pd.DataFrame(perf), pcols),
          f"\nDD% here uses the research convention of {rep_risk:g}% per trade; the sized ledger above uses "
          "real lots.\n",
          "\n_One run, rules fixed before the result (rule 7). Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    for name, tr in trades.items():
        slug = "natural" if name.startswith("natural") else "original_capped"
        tr.to_csv(out / f"trades_{slug}.csv", index=False)
        (out / "detail" / f"{slug}.md").write_text(_detail(name, tr, rep_risk, st_all[name]), encoding="utf-8")
    for b in balances:
        _, led = sized_run(nat, b, risk_pct, leverage=leverage)
        led.to_csv(out / f"ledger_natural_{int(b)}.csv", index=False)
    res = {"sizing": sz.to_dict("records"), "performance": perf, "capped_trades": capped_days,
           "report": str(out / "report.md")}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log.info("lb_natural_done", extra={"capped_trades": capped_days})
    return res
