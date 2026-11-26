"""GTC-SBR: Gold Trend Continuation - Session Breakout and Retest (owner's spec, 2026-11-26).

``python -m quant_engine gtc`` (test_gtc_sbr.bat). Rules as specified by the owner; every place
where the spec leaves a choice open is marked GUESS and listed in the report.

* **Trend (M15, completed candles only)**: bullish = EMA20 > EMA50, close > EMA20, both EMAs
  rising over ``slope_bars`` M15 candles; bearish = mirror; otherwise sideways (no trade).
* **Opening range**: first 30 minutes after 08:00 London time (London) and 08:20 New York time
  (New York), with daylight saving from the real time zones.
* **Breakout**: a completed M5 candle closes at least 0.1 ATR(14, M5) beyond the range.
* **Retest**: within completed candles 3-8 after the breakout candle, a candle comes back to
  within 0.15 ATR of the broken boundary; the confirmation candle closes back beyond it.
* **Entry**: next M5 open (buy at the ask). **Stop**: beyond the retest structure + 0.2 ATR.
  **Target**: fixed R. **Time exit**: after 12 completed M5 candles. Stop first if both touch.
* **Filters**: one trade per session, no overlap; no entries 15 min before to 30 min after
  high-impact US releases; no entry when the spread is above 1.5 x its past median for that
  half-hour (past data only).
* **Costs**: historical bar spread (floored at the owner's quoted spread), commission, slippage
  on both fills (:mod:`.metrics`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, CostsConfig
from ..data.storage import ParquetStore
from ..features.atr import atr as atr_fn
from ..logging_setup import get_logger
from ..timeframes import Timeframe
from .events import regime_per_bar
from .metrics import apply_costs, core_metrics, full_report, report_markdown
from .regime4 import regime4_series
from .runner import _table

log = get_logger(__name__)

DEFAULTS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "slope_bars": 3,              # GUESS: "slopes positive" = EMA higher than 3 M15 candles ago
    "trend_filter": True,
    "sessions": {"london": ("Europe/London", 8 * 60), "new_york": ("America/New_York", 8 * 60 + 20)},
    "range_minutes": 30,
    "breakout_atr": 0.1,
    "breakout_window_bars": 24,   # GUESS: the breakout must come within 2 hours of the range end
    "retest_min": 3, "retest_max": 8,   # GUESS: retest candle is the 3rd to 8th candle after the breakout
    "retest_tol_atr": 0.15,
    "sl_buffer_atr": 0.2,
    "min_risk_atr": 0.3, "max_risk_atr": 3.0,   # GUESS: "unreasonable stop" = outside 0.3-3 ATR
    "target_r": 1.5,
    "time_exit_bars": 12,
    "news_ny": [8 * 60 + 30, 10 * 60],   # GUESS: no historical calendar; block these slots every weekday
    "news_before": 15, "news_after": 30,
    "spread_mult": 1.5, "spread_bucket_min": 30, "spread_window": 360, "spread_min_obs": 20,
    "contract_size": 100.0,
}

GUESSES = [
    "Trend slope: each EMA is higher (lower) than 3 completed M15 candles ago.",
    "Breakout must happen within 2 hours (24 M5 candles) after the opening range ends; only the first "
    "breakout of the session counts (if it is against the trend, the session is skipped).",
    "Retest window: the 3rd to 8th completed candle after the breakout candle; first touch counts.",
    "Confirmation candle: the retest candle itself if it closes beyond the boundary, otherwise the next one.",
    "Retest structure for the stop: candles from the retest candle to the confirmation candle.",
    "Unreasonable stop: below 0.3 ATR or above 3 ATR (M5 ATR at the confirmation candle).",
    "News: no historical calendar is stored, so entries are blocked 08:15-09:00 and 09:45-10:30 New York "
    "time on every weekday (US high-impact release slots). A run without it is shown as a sensitivity check.",
    "Spread filter: median of the previous 360 candles in the same New York half-hour (about 60 days); "
    "needs 20 observations, otherwise the candle is allowed.",
    "Slippage 2 points per fill on entry and exit (market / stop orders); commission 0 (owner).",
]


def m15_trend(m15: pd.DataFrame, p: dict) -> np.ndarray:
    """Trend label per M15 candle (known at its close)."""
    c = m15["close"].astype(float)
    f = c.ewm(span=p["ema_fast"], adjust=False, min_periods=p["ema_fast"]).mean()
    s = c.ewm(span=p["ema_slow"], adjust=False, min_periods=p["ema_slow"]).mean()
    k = p["slope_bars"]
    bull = (f > s) & (c > f) & (f > f.shift(k)) & (s > s.shift(k))
    bear = (f < s) & (c < f) & (f < f.shift(k)) & (s < s.shift(k))
    return np.where(bull, "bullish", np.where(bear, "bearish", "sideways"))


def past_spread_median(m5: pd.DataFrame, p: dict) -> np.ndarray:
    """Median bar spread of earlier candles in the same New York half-hour (no future data)."""
    t = pd.DatetimeIndex(m5["time"]).tz_convert("America/New_York")
    bucket = (t.hour * 60 + t.minute) // p["spread_bucket_min"]
    sp = pd.Series(m5["spread"].to_numpy(float) if "spread" in m5 else np.full(len(m5), np.nan))
    med = sp.groupby(np.asarray(bucket)).transform(
        lambda s: s.shift(1).rolling(p["spread_window"], min_periods=p["spread_min_obs"]).median())
    return med.to_numpy()


def find_trades(m5: pd.DataFrame, m15: pd.DataFrame, h4: pd.DataFrame, p: dict, point: float,
                costs: CostsConfig, spread_factor: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """All GTC-SBR trades (gross R after spread) plus counts of why setups were skipped."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    a = atr_fn(m5, p["atr_period"]).to_numpy(float)
    raw_sp = m5["spread"].to_numpy(float) if "spread" in m5 else np.full(n, np.nan)
    sp = np.maximum(np.nan_to_num(raw_sp, nan=costs.min_spread_points), costs.min_spread_points) * point * spread_factor
    med = past_spread_median(m5, p)
    trend = m15_trend(m15, p)
    closes15 = pd.DatetimeIndex(m15["time"]) + Timeframe.M15.delta
    reg4 = regime_per_bar(t, h4, regime4_series(h4))
    ny = t.tz_convert("America/New_York")
    ny_min = np.asarray(ny.hour * 60 + ny.minute)
    weekday = np.asarray(ny.dayofweek) < 5
    step = Timeframe.M5.delta
    skips = {k: 0 for k in ("no range", "no breakout", "against or no trend", "no retest", "no confirmation",
                            "news", "spread", "stop size", "data gap", "overlap")}
    rows, busy_until = [], pd.Timestamp.min.tz_localize("UTC")
    events = []
    for name, (tz, open_min) in p["sessions"].items():
        loc = t.tz_convert(tz)
        lmin = np.asarray(loc.hour * 60 + loc.minute)
        ldate = np.asarray(loc.normalize().tz_localize(None))
        in_rng = (lmin >= open_min) & (lmin < open_min + p["range_minutes"])
        for d in np.unique(ldate[in_rng]):
            ridx = np.nonzero(in_rng & (ldate == d))[0]
            events.append((t[ridx[0]], name, ridx))
    events.sort(key=lambda e: e[0])
    nbars_rng = p["range_minutes"] // 5
    for _, name, ridx in events:
        if len(ridx) != nbars_rng or t[ridx[-1]] - t[ridx[0]] != (nbars_rng - 1) * step:
            skips["no range"] += 1
            continue
        hi, lo = h[ridx].max(), l[ridx].min()
        start = ridx[-1] + 1
        b, side = None, 0
        for i in range(start, min(start + p["breakout_window_bars"], n)):
            if t[i] - t[ridx[-1]] != (i - ridx[-1]) * step or not a[i] > 0:
                break
            if c[i] >= hi + p["breakout_atr"] * a[i]:
                b, side = i, 1
                break
            if c[i] <= lo - p["breakout_atr"] * a[i]:
                b, side = i, -1
                break
        if b is None:
            skips["no breakout"] += 1
            continue
        k15 = int(closes15.searchsorted(t[b] + step, side="right")) - 1
        tr = trend[k15] if k15 >= 0 else "sideways"
        if p["trend_filter"] and tr != ("bullish" if side == 1 else "bearish"):
            skips["against or no trend"] += 1
            continue
        bound = hi if side == 1 else lo
        tol = p["retest_tol_atr"] * a[b]
        k = None
        for i in range(b + p["retest_min"], min(b + p["retest_max"], n - 1) + 1):
            if (side == 1 and l[i] <= bound + tol) or (side == -1 and h[i] + sp[i] >= bound - tol):
                k = i
                break
        if k is None:
            skips["no retest"] += 1
            continue
        conf = k if (c[k] - bound) * side > 0 else (k + 1 if k + 1 < n and (c[k + 1] - bound) * side > 0 else None)
        if conf is None:
            skips["no confirmation"] += 1
            continue
        j = conf + 1
        end = j + p["time_exit_bars"] - 1
        if end >= n or t[end] - t[b] != (end - b) * step:
            skips["data gap"] += 1
            continue
        if t[j] < busy_until:
            skips["overlap"] += 1
            continue
        if weekday[j] and any(-p["news_before"] <= ny_min[j] - m <= p["news_after"] for m in p["news_ny"]):
            skips["news"] += 1
            continue
        if np.isfinite(med[conf]) and np.isfinite(raw_sp[conf]) and raw_sp[conf] > p["spread_mult"] * med[conf]:
            skips["spread"] += 1
            continue
        if side == 1:
            stop = l[k:conf + 1].min() - p["sl_buffer_atr"] * a[conf]
            entry = o[j] + sp[j]
        else:
            stop = h[k:conf + 1].max() + sp[conf] + p["sl_buffer_atr"] * a[conf]
            entry = o[j]
        risk = (entry - stop) * side
        if not (p["min_risk_atr"] * a[conf] <= risk <= p["max_risk_atr"] * a[conf]):
            skips["stop size"] += 1
            continue
        tp = entry + side * p["target_r"] * risk
        exit_px, ex, how = None, end, "time"
        for i in range(j, end + 1):
            if side == 1:
                if l[i] <= stop:
                    exit_px, ex, how = min(o[i], stop), i, "stop"
                    break
                if h[i] >= tp:
                    exit_px, ex, how = max(o[i], tp), i, "target"
                    break
            else:
                if h[i] + sp[i] >= stop:
                    exit_px, ex, how = max(o[i] + sp[i], stop), i, "stop"
                    break
                if l[i] + sp[i] <= tp:
                    exit_px, ex, how = min(o[i] + sp[i], tp), i, "target"
                    break
        if exit_px is None:
            exit_px = c[end] + (sp[end] if side == -1 else 0.0)
        busy_until = t[ex] + step
        rows.append({"entry_time": t[j], "exit_time": t[ex] + step, "session": name,
                     "side": "BUY" if side == 1 else "SELL", "trend_m15": tr, "regime": reg4[j],
                     "range_high": round(hi, 2), "range_low": round(lo, 2), "entry": round(entry, 2),
                     "stop": round(stop, 2), "target": round(tp, 2), "exit": round(exit_px, 2), "exit_how": how,
                     "risk": risk, "atr": a[conf], "spread_paid": sp[j] if side == 1 else sp[ex],
                     "r": side * (exit_px - entry) / risk})
    cols = ["entry_time", "exit_time", "session", "side", "trend_m15", "regime", "range_high", "range_low", "entry",
            "stop", "target", "exit", "exit_how", "risk", "atr", "spread_paid", "r"]
    return pd.DataFrame(rows, columns=cols), skips


def with_sizes(trades: pd.DataFrame, risk_pct: float, contract: float, equity0: float = 10_000.0) -> pd.DataFrame:
    """Lots per trade from 0.5% of current equity, the stop distance and the contract size."""
    t = trades.copy()
    eq, lots, eqs = equity0, [], []
    for r, risk in zip(t["r"], t["risk"]):
        lot = eq * risk_pct / 100 / (risk * contract)
        lots.append(round(lot, 2))
        eqs.append(round(eq, 2))
        eq *= 1 + risk_pct / 100 * r
    t["equity_before"], t["lots"] = eqs, lots
    return t


def run_gtc(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
            start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, source_note: str = "") -> dict:
    """Experiments A-F, splits, walk-forward, cost stress and parameter stability; writes ``report.md``."""
    sc = cfg.strategy
    costs = sc.research.costs
    risk_pct = sc.backtest.risk_pct
    bars = {tf: store.read_bars(dataset, tf, start, end) for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H4)}
    if any(b.empty for b in bars.values()):
        raise ValueError(f"No stored {dataset} M5/M15/H4 data")
    m5, m15, h4 = bars[Timeframe.M5], bars[Timeframe.M15], bars[Timeframe.H4]
    t0, t1 = m5["time"].iat[0], m5["time"].iat[-1]
    span = t1 - t0
    cut1, cut2 = t0 + span * 0.6, t0 + span * 0.8
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run(p_over: dict | None = None, spread_factor: float = 1.0, slip: float | None = None) -> tuple[pd.DataFrame, dict]:
        p = {**DEFAULTS, **(p_over or {})}
        cst = costs if slip is None else costs.model_copy(update={"slippage_points": slip})
        tr, sk = find_trades(m5, m15, h4, p, point, cst, spread_factor)
        return apply_costs(tr, cst, point), sk

    def part(tr: pd.DataFrame, a_, b_) -> pd.DataFrame:
        return tr[(tr["entry_time"] >= a_) & (tr["entry_time"] < b_)] if len(tr) else tr

    def m(tr: pd.DataFrame) -> dict:
        return core_metrics(tr["r"].to_numpy(), risk_pct) if len(tr) else {"trades": 0}

    def gross(tr: pd.DataFrame) -> dict:
        if not len(tr):
            return {}
        r = tr["r"].to_numpy()
        return {"gross_profit_r": round(float(r[r > 0].sum()), 1), "gross_loss_r": round(float(r[r <= 0].sum()), 1)}

    exps = {"A: base, 1.5R": {}, "B: 2R target": {"target_r": 2.0}, "C: 1R target": {"target_r": 1.0},
            "F: no trend filter": {"trend_filter": False}}
    res, trades_by = [], {}
    for name, over in exps.items():
        tr, sk = run(over)
        trades_by[name] = tr
        for pname, (a_, b_) in {"development": (t0, cut1), "validation": (cut1, cut2),
                                "out-of-sample": (cut2, t1 + pd.Timedelta(days=1))}.items():
            x = part(tr, a_, b_)
            res.append({"experiment": name, "data": pname, **m(x), **gross(x)})
        slug = name.split(":")[0]
        tr.to_csv(out / f"trades_{slug}.csv", index=False)
        if name.startswith("A"):
            skips_a = sk
    resdf = pd.DataFrame(res)
    n_tests = len(exps)
    a_tr = with_sizes(trades_by["A: base, 1.5R"], risk_pct, DEFAULTS["contract_size"])
    a_tr.to_csv(out / "trades_A.csv", index=False)
    rep_a = full_report(a_tr, risk_pct)

    def split_sides(tr, col, vals):
        return pd.DataFrame([{"group": v, **{f"{pn}_{k}": m(part(tr[tr[col] == v], a_, b_)).get(k)
                                             for pn, (a_, b_) in (("dev", (t0, cut1)), ("val", (cut1, cut2)),
                                                                  ("oos", (cut2, t1 + pd.Timedelta(days=1))))
                                             for k in ("trades", "expectancy_r", "profit_factor")}}
                             for v in vals])
    d_tab = split_sides(trades_by["A: base, 1.5R"], "session", ["london", "new_york"])
    e_tab = split_sides(trades_by["A: base, 1.5R"], "side", ["BUY", "SELL"])

    # walk-forward: consecutive 6-month windows, fixed rules (nothing is refitted)
    wf = []
    a_all = trades_by["A: base, 1.5R"]
    w0 = pd.Timestamp(t0).normalize()
    while w0 < t1 and len(a_all):
        w1 = w0 + pd.DateOffset(months=6)
        x = part(a_all, w0, w1)
        mm = m(x)
        wf.append({"window": f"{w0:%Y-%m} to {w1:%Y-%m}", "trades": mm.get("trades", 0),
                   "expectancy_r": mm.get("expectancy_r"), "profit_factor": mm.get("profit_factor"),
                   "net_r": mm.get("net_r")})
        w0 = w1
    wfdf = pd.DataFrame(wf)
    wf_pos = int((wfdf["net_r"] > 0).sum()) if len(wfdf) else 0

    stress = []
    for label, kw in (("as tested", {}), ("spread x1.5", {"spread_factor": 1.5}), ("spread x2", {"spread_factor": 2.0}),
                      ("slippage 5 points per fill", {"slip": 5.0}), ("no news filter (sensitivity)", {"p_over": {"news_ny": []}})):
        tr, _ = run(**kw)
        stress.append({"costs": label, **{f"{pn}_{k}": m(part(tr, a_, b_)).get(k)
                                          for pn, (a_, b_) in (("all", (t0, t1 + pd.Timedelta(days=1))),
                                                               ("oos", (cut2, t1 + pd.Timedelta(days=1))))
                                          for k in ("trades", "expectancy_r", "profit_factor")}})
    stdf = pd.DataFrame(stress)

    stab = []
    for k_, vals in (("breakout_atr", [0.05, 0.1, 0.15]), ("retest_tol_atr", [0.10, 0.15, 0.20]),
                     ("sl_buffer_atr", [0.15, 0.2, 0.25]), ("time_exit_bars", [9, 12, 15]), ("slope_bars", [1, 3, 5])):
        for v in vals:
            tr, _ = run({k_: v})
            mm = m(part(tr, t0, cut1))
            stab.append({"setting": k_, "value": v, "base": v == DEFAULTS[k_], "dev_trades": mm.get("trades", 0),
                         "dev_expectancy_r": mm.get("expectancy_r"), "dev_profit_factor": mm.get("profit_factor")})
    stabdf = pd.DataFrame(stab)

    oos_a = resdf[(resdf["experiment"].str.startswith("A")) & (resdf["data"] == "out-of-sample")].iloc[0]
    all_a = rep_a["overall"]
    checks = {
        "positive out-of-sample expectancy after costs": bool(oos_a.get("expectancy_r", -1) > 0),
        f"out-of-sample p below {0.05 / n_tests:.4f} (Bonferroni, {n_tests} rule variants)": bool(oos_a.get("p_value", 1) < 0.05 / n_tests),
        "profit factor above 1.3 (all data)": bool((all_a.get("profit_factor") or 0) > 1.3),
        "max drawdown below 10%": bool(all_a.get("max_drawdown_pct", 100) < 10),
        "at least 200 trades": bool(all_a.get("trades", 0) >= 200),
        "positive in most walk-forward windows": bool(len(wfdf) and wf_pos > len(wfdf) / 2),
        "positive with spread x1.5": bool((stdf.loc[1, "all_expectancy_r"] or -1) > 0),
    }
    passed = sum(checks.values())
    if not checks["positive out-of-sample expectancy after costs"] or (all_a.get("expectancy_r") or 0) <= 0:
        decision = "REJECTED"
    elif passed == len(checks):
        decision = "VALIDATED FOR PAPER TRADING (pending the independent audit)"
    else:
        decision = "RESEARCH REQUIRED"

    pcols = ["experiment", "data", "trades", "win_rate_pct", "gross_profit_r", "gross_loss_r", "net_r",
             "net_profit_pct", "profit_factor", "expectancy_r", "max_drawdown_pct", "avg_win_r", "avg_loss_r",
             "max_consecutive_losses", "p_value"]
    md = ["# GTC-SBR: Gold Trend Continuation, Session Breakout and Retest\n",
          f"- **data**: {source_note or dataset}, M5 / M15 / H4",
          f"- **period (UTC)**: {t0:%Y-%m-%d} to {t1:%Y-%m-%d}",
          f"- **split by time**: development to {cut1:%Y-%m-%d}, validation to {cut2:%Y-%m-%d}, "
          "untouched out-of-sample after that",
          f"- **costs**: historical bar spread (at least {costs.min_spread_points:g} points), commission "
          f"{costs.commission_per_lot:g} per lot, slippage {costs.slippage_points:g} points per fill",
          f"- **risk**: {risk_pct:g}% of equity per trade, lots = risk / (stop distance x {DEFAULTS['contract_size']:g} oz)",
          f"- **decision**: **{decision}**\n",
          "## Acceptance checks (experiment A)\n"] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in checks.items()] + [
          "\n## Experiments A, B, C, F (one change each) by data part\n",
          f"Multiple testing: {n_tests} rule variants, so a single result needs p < {0.05 / n_tests:.4f}.\n",
          _table(resdf, pcols),
          "\n## Experiment D: London vs New York (experiment A trades)\n", _table(d_tab, list(d_tab.columns)),
          "\n## Experiment E: bullish (BUY) vs bearish (SELL) (experiment A trades)\n", _table(e_tab, list(e_tab.columns)),
          "\n## Walk-forward: 6-month windows, rules fixed (experiment A)\n",
          f"Positive windows: {wf_pos} of {len(wfdf)}.\n", _table(wfdf, list(wfdf.columns)),
          "\n## Higher costs (experiment A)\n", _table(stdf, list(stdf.columns)),
          "\n## Parameter stability: nearby values, development data only (nothing chosen from this)\n",
          _table(stabdf, list(stabdf.columns)),
          "\n## Why setups were skipped (experiment A)\n",
          _table(pd.DataFrame([skips_a]), list(skips_a.keys())),
          "\n## Full metrics, experiment A, all data\n", report_markdown("Experiment A", rep_a),
          "\n## Choices the spec left open (GUESS)\n"] + [f"- {g}" for g in GUESSES] + [
          "\n## Audit checklist\n",
          "- Trend uses only M15 candles closed by the breakout candle's close (tested in tests/test_gtc_sbr.py).",
          "- Entry is the open of the candle after the confirmation candle; buys pay the ask.",
          "- Stop is checked before the target on every candle; a gap through the stop fills at the open.",
          "- Spread filter uses only earlier candles; ATR is at the decision candle.",
          "- Signals computed on data cut at the entry are the same as on the full data (test).",
          "\n_Research on past data, not a promise of future profit. Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"decision": decision, "checks_passed": f"{passed}/{len(checks)}",
               "A_all": {k: all_a.get(k) for k in ("trades", "win_rate_pct", "profit_factor", "expectancy_r",
                                                   "net_profit_pct", "max_drawdown_pct")},
               "A_oos": {k: oos_a.get(k) for k in ("trades", "expectancy_r", "profit_factor", "p_value")},
               "walk_forward_positive": f"{wf_pos}/{len(wfdf)}", "report": str(out / "report.md")}
    log.info("gtc_done", extra={"decision": decision})
    return summary
