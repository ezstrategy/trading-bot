"""The owner's entry: dip / sweep + reversal candle, stop under the low, target in R.

``python -m quant_engine entries`` (find_entries.bat). Rules (BUY; SELL is the mirror):

1. **Sweep low**: candle ``k`` makes a low below the lowest low of the previous
   ``lookback`` candles (a new 30-minute low that takes the nearby sell stops).
2. **Reversal**: within ``confirm_bars`` candles after ``k``, a green candle closes above the
   high of candle ``k`` while no candle since ``k`` went below its low. The first such candle
   is the signal candle ``j`` (known at its close).
3. **Entry** at the next candle's open (ask). **Stop** = sweep low minus a buffer
   (``max(buffer_atr x ATR, 2 x spread)``). Risk must be between ``min_risk_atr`` and
   ``max_risk_atr`` ATR, otherwise skipped.
4. **Target** = ``R`` x risk for each R in ``targets_r``; close after ``horizons`` candles
   if neither is hit (stop checked first on every candle, spread paid).

VWAP (daily, resets 17:00 New York, tick-volume weighted, +-1 standard deviation band) filters
are tested too: ``vwap band sweep`` (the sweep went beyond the band on the far side: below
VWAP - 1 sd for buys), ``vwap reclaim`` (the sweep went through VWAP and the signal candle
closed back on the trade's side of it), ``with vwap`` (signal closes on the trade's side of
VWAP and VWAP slopes that way over 30 min), ``discount`` (buys below VWAP, sells above).

Filters tested as separate variants: ``any``; ``flush`` (the fall into the low was at least
``flush_atr`` ATR over the ``lookback`` candles before it, like a stop run); ``with trend``
(H4 market type agrees: bullish for buys, bearish for sells). Each variant x session x side
x target x time limit is one candidate. Candidates are chosen on the older part of the data
and checked on the newer part, with the same false-discovery filter as :mod:`.miner`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..features.atr import atr as atr_fn
from ..features.context import STRUCTURE_TFS
from ..features.sessions import session_labels
from ..features.vwap import daily_vwap
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .events import regime_per_bar
from .miner import benjamini_hochberg, non_overlapping, t_stats
from .model import regime_kwargs
from .outcomes import trade_outcomes
from .runner import _table

log = get_logger(__name__)

DEFAULTS = {"lookback": 6, "confirm_bars": 3, "buffer_atr": 0.1, "min_risk_atr": 0.5, "max_risk_atr": 4.0,
            "flush_atr": 2.0, "targets_r": [1.0, 2.0, 3.0], "horizons": [12, 36], "spread_points": 8.0,
            "discover_frac": 0.6, "min_trades": 30}

# The owner's three example trades (screenshots of 2026-11-25/26), in UTC. IST = UTC + 5:30.
# GUESS: times read off the screenshots to the nearest 5-minute candle.
OWNER_EXAMPLES = [("2026-11-25 15:30", 1), ("2026-11-25 21:20", 1), ("2026-11-26 01:50", 1)]


@dataclass
class EntrySignals:
    """Signal candles and their stop distances, per side (+1 buy, -1 sell)."""

    times: pd.DatetimeIndex
    signal: dict[int, np.ndarray]     # side -> bool per candle (signal at the close)
    risk: dict[int, np.ndarray]       # side -> stop distance in USD from the expected entry
    stop: dict[int, np.ndarray]       # side -> stop price
    flush: dict[int, np.ndarray]
    sweep_idx: dict[int, np.ndarray]


def find_signals(m5: pd.DataFrame, p: dict, spread: float) -> EntrySignals:
    """Apply rules 1-3 on closed M5 candles (no look-ahead: everything is known at candle j's close)."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    a = atr_fn(m5, 14).to_numpy(dtype="float64")
    n, lb, cb = len(c), int(p["lookback"]), int(p["confirm_bars"])
    prior_lo = pd.Series(l).shift(1).rolling(lb, min_periods=lb).min().to_numpy()
    prior_hi = pd.Series(h).shift(1).rolling(lb, min_periods=lb).max().to_numpy()
    sweep = {1: l < prior_lo, -1: h > prior_hi}
    out = {k: {d: (np.zeros(n, bool) if k in ("sig", "flush") else np.full(n, np.nan if k != "sidx" else -1))
               for d in (1, -1)} for k in ("sig", "risk", "stop", "flush", "sidx")}
    for d in (1, -1):
        ks = np.nonzero(sweep[d])[0]
        for k in ks:
            ext = l[k] if d == 1 else h[k]
            trigger = h[k] if d == 1 else l[k]
            for j in range(k + 1, min(k + cb, n - 1) + 1):
                broke = (l[k + 1:j + 1].min() < ext) if d == 1 else (h[k + 1:j + 1].max() > ext)
                if broke:
                    break
                ok = (c[j] > trigger and c[j] > o[j]) if d == 1 else (c[j] < trigger and c[j] < o[j])
                if not ok:
                    continue
                atr_j = a[j]
                if not atr_j > 0:
                    break
                buf = max(p["buffer_atr"] * atr_j, 2 * spread)
                stop = ext - buf if d == 1 else ext + buf
                entry = c[j] + (spread if d == 1 else 0.0)
                risk = (entry - stop) * d
                if p["min_risk_atr"] * atr_j <= risk <= p["max_risk_atr"] * atr_j and not out["sig"][d][j]:
                    out["sig"][d][j] = True
                    out["risk"][d][j] = risk
                    out["stop"][d][j] = stop
                    out["sidx"][d][j] = k
                    start = max(k - lb, 0)
                    fall = (h[start:k].max() - ext) if d == 1 else (ext - l[start:k].min())
                    out["flush"][d][j] = fall >= p["flush_atr"] * a[k]
                break
    return EntrySignals(t, out["sig"], out["risk"], out["stop"], out["flush"],
                        {d: out["sidx"][d].astype(int) for d in (1, -1)})


def run_entries(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                source_note: str = "", examples: list[tuple[str, int]] | None = None,
                overrides: dict | None = None) -> dict:
    """Backtest the entry and write ``report.md``, ``candidates.csv`` and ``signals.csv``.
    ``examples``: (UTC time, side) of the owner's own trades, to check that the rules fire."""
    sc = cfg.strategy
    p = {**DEFAULTS, **getattr(sc.research, "bigmoves", {}).get("entries", {}), **(overrides or {})}
    bars = {tf: store.read_bars(dataset, tf, start, end) for tf in STRUCTURE_TFS}
    missing = [tf.value for tf, b in bars.items() if b.empty]
    if missing:
        raise ValueError(f"No stored {dataset} data for {', '.join(missing)}")
    m5, h4 = bars[Timeframe.M5], bars[Timeframe.H4]
    spread = float(p["spread_points"]) * point
    sig = find_signals(m5, p, spread)
    t = sig.times
    o, h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(c)
    regime = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
    sess = session_labels(t, sc.sessions)
    vw, vsd = daily_vwap(m5, sc.sessions)
    a = atr_fn(m5, 14).to_numpy(dtype="float64")
    slope = vw - np.concatenate([np.full(6, np.nan), vw[:-6]])          # 30-minute VWAP change
    side_z = {d: np.where(vsd > 0, d * (c - vw) / np.where(vsd > 0, vsd, 1.0), np.nan) for d in (1, -1)}
    vfilt = {}
    for d in (1, -1):
        k = sig.sweep_idx[d]
        has = k >= 0
        kk = np.where(has, k, 0)
        ext = l[kk] if d == 1 else h[kk]
        far = vw[kk] - d * vsd[kk]
        vfilt[d] = {"vwap band sweep": has & ((ext - far) * d < 0) & (vsd[kk] > 0),
                    "vwap reclaim": has & ((ext - vw[kk]) * d < 0) & ((c - vw) * d > 0),
                    "with vwap": ((c - vw) * d > 0) & (slope * d > 0),
                    "discount": (c - vw) * d < 0}
    split = int(n * float(p["discover_frac"]))
    sp_arr = np.full(n, spread)
    thirds = np.linspace(0, split, 4).astype(int)
    rows, outcomes = [], {}
    for d in (1, -1):
        risk = np.nan_to_num(sig.risk[d], nan=1.0)  # non-signal candles are never used
        for R in p["targets_r"]:
            for hz in p["horizons"]:
                outcomes[(d, R, hz)] = trade_outcomes(t, o, h, l, c, sp_arr, risk, d, 1.0, float(R), int(hz))
        trend_ok = regime == ("bullish" if d == 1 else "bearish")
        variants = {"any": sig.signal[d], "flush": sig.signal[d] & sig.flush[d],
                    "with trend": sig.signal[d] & trend_ok,
                    **{k: sig.signal[d] & v for k, v in vfilt[d].items()}}
        for vname, vmask in variants.items():
            for s in ("any", "asia", "london_kz", "newyork_kz", "london_close"):
                m = vmask if s == "any" else vmask & (sess == s)
                idx = np.nonzero(m)[0]
                for R in p["targets_r"]:
                    for hz in p["horizons"]:
                        r, ex = outcomes[(d, R, hz)]
                        di = idx[(idx < split) & (ex[idx] >= 0) & (ex[idx] < split)]
                        vi = idx[(idx >= split) & (ex[idx] >= 0)]
                        dt, vt = non_overlapping(di, ex), non_overlapping(vi, ex)
                        if len(dt) < int(p["min_trades"]):
                            continue
                        st, sv = t_stats(r[dt]), t_stats(r[vt])
                        pos = sum(1 for a_, b_ in zip(thirds[:-1], thirds[1:])
                                  if (x := dt[(dt >= a_) & (dt < b_)]).size >= 5 and r[x].mean() > 0)
                        rows.append({"side": "BUY" if d == 1 else "SELL", "filter": vname, "session": s,
                                     "target_r": R, "minutes": hz * 5, **st, "positive_thirds": pos,
                                     **{f"val_{k}": v for k, v in sv.items()},
                                     "breakeven_win_pct": round(100 / (1 + R), 1)})
    cand = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(cand):
        cand["bh_pass"] = benjamini_hochberg(cand["p"].to_numpy(), sc.research.fdr_q)
        cand["survives"] = cand["bh_pass"] & (cand["avg_r"] > 0.05) & (cand["positive_thirds"] >= 2)
        nsurv = max(int(cand["survives"].sum()), 1)
        cand["verdict"] = np.where(~cand["survives"], "",
                                   np.where((cand["val_avg_r"] > 0) & (cand["val_p"] < 0.05 / nsurv), "confirmed",
                                            np.where(cand["val_avg_r"] > 0, "held up (not proven)", "failed")))
        cand = cand.sort_values("t", ascending=False).reset_index(drop=True)
    cand.to_csv(out / "candidates.csv", index=False)
    vf_rows = cand[cand["filter"].isin(["any", *vfilt[1].keys()]) & (cand["session"] == "any")] if len(cand) else cand

    # every signal with its 3R / 3 h result, for inspection
    sig_rows = []
    for d in (1, -1):
        r3, _ = outcomes[(d, max(p["targets_r"]), max(p["horizons"]))]
        for j in np.nonzero(sig.signal[d])[0]:
            sig_rows.append({"signal_close_utc": t[j] + Timeframe.M5.delta, "side": "BUY" if d == 1 else "SELL",
                             "entry_approx": round(c[j] + (spread if d == 1 else 0), 2),
                             "stop": round(sig.stop[d][j], 2), "risk_usd": round(sig.risk[d][j], 2),
                             "flush": bool(sig.flush[d][j]), "regime": regime[j], "session": sess[j],
                             "vwap": round(float(vw[j]), 2), "vs_vwap_sd": round(float(side_z[d][j]), 2),
                             **{k.replace(" ", "_"): bool(v[j]) for k, v in vfilt[d].items()},
                             f"r_at_{max(p['targets_r']):g}R": None if np.isnan(r3[j]) else round(float(r3[j]), 2)})
    sigs = pd.DataFrame(sig_rows).sort_values("signal_close_utc") if sig_rows else pd.DataFrame()
    sigs.to_csv(out / "signals.csv", index=False)

    ex_rows = []
    for when, side in examples or []:
        w = pd.Timestamp(when, tz="UTC")
        near = sigs[(sigs["side"] == ("BUY" if side == 1 else "SELL"))
                    & ((sigs["signal_close_utc"] - w).abs() <= pd.Timedelta(minutes=30))] if len(sigs) else sigs
        ex_rows.append({"your trade (IST)": f"{w.tz_convert('Asia/Kolkata'):%d %b %H:%M}",
                        "side": "BUY" if side == 1 else "SELL",
                        "rule fired": "yes" if len(near) else "no",
                        "signal (IST)": "" if not len(near) else
                        f"{near['signal_close_utc'].iat[0].tz_convert('Asia/Kolkata'):%d %b %H:%M}",
                        "entry": "" if not len(near) else near["entry_approx"].iat[0],
                        "stop": "" if not len(near) else near["stop"].iat[0],
                        "result_3R": "" if not len(near) else near.iloc[0, -1]})
    exdf = pd.DataFrame(ex_rows)

    # VWAP just before each marked entry (the last candle closed at the marked time)
    closes = t + Timeframe.M5.delta
    vw_rows = []
    for when, side in examples or []:
        w = pd.Timestamp(when, tz="UTC")
        j = int(closes.searchsorted(w, side="right")) - 1
        if j < 6:
            continue
        lo6, hi6 = l[j - 5:j + 1].min(), h[j - 5:j + 1].max()
        ext = lo6 if side == 1 else hi6
        vw_rows.append({"your entry (IST)": f"{w.tz_convert('Asia/Kolkata'):%d %b %H:%M}",
                        "side": "BUY" if side == 1 else "SELL", "price": round(c[j], 2), "vwap": round(vw[j], 2),
                        "price - vwap ($)": round(c[j] - vw[j], 2),
                        "in ATR": round((c[j] - vw[j]) / a[j], 2) if a[j] > 0 else np.nan,
                        "in band sd": round((c[j] - vw[j]) / vsd[j], 2) if vsd[j] > 0 else np.nan,
                        "vwap 30-min slope ($)": round(slope[j], 2),
                        "30-min low/high vs vwap (sd)": round((ext - vw[j]) / vsd[j], 2) if vsd[j] > 0 else np.nan,
                        "touched far band": bool((ext - (vw[j] - side * vsd[j])) * side < 0)})
    vwdf = pd.DataFrame(vw_rows)

    # every signal (3R / 3 h): result by where the signal closed versus VWAP, on the trade's side
    buckets = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    labels = ["2+ sd wrong side", "1-2 sd wrong side", "0-1 sd wrong side", "0-1 sd right side",
              "1-2 sd right side", "2+ sd right side"]
    brows = []
    for d in (1, -1):
        r3, ex3 = outcomes[(d, max(p["targets_r"]), max(p["horizons"]))]
        idx = np.nonzero(sig.signal[d])[0]
        idx = idx[ex3[idx] >= 0]
        z = pd.cut(side_z[d][idx], buckets, labels=labels)
        for part, sel in (("older", idx < split), ("unseen", idx >= split)):
            for lab in labels:
                rr = r3[idx[sel & (z == lab)]]
                brows.append({"side": "BUY" if d == 1 else "SELL", "data": part, "signal vs vwap": lab,
                              "signals": len(rr), "win_rate_pct": round(100 * float((rr > 0).mean()), 1) if len(rr) else np.nan,
                              "avg_r": round(float(rr.mean()), 3) if len(rr) else np.nan})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(out / "vwap_buckets.csv", index=False)

    allrow = cand[(cand["filter"] == "any") & (cand["session"] == "any")] if len(cand) else cand
    cols = ["side", "filter", "session", "target_r", "minutes", "trades", "win_rate_pct", "breakeven_win_pct",
            "avg_r", "total_r", "p", "val_trades", "val_win_rate_pct", "val_avg_r", "verdict"]
    surv = cand[cand["survives"]] if len(cand) else cand
    md = ["# Your entry: dip / sweep + reversal candle\n",
          f"- **data**: {source_note or dataset}",
          f"- **period (UTC)**: {m5['time'].iat[0]:%Y-%m-%d} to {m5['time'].iat[-1] + Timeframe.M5.delta:%Y-%m-%d}",
          f"- **spread charged**: ${spread:.2f} per trade",
          f"- **rules**: new {int(p['lookback']) * 5}-minute low (high), then within {int(p['confirm_bars'])} candles "
          "a candle closing beyond the sweep candle's high (low) in the trade's direction; stop beyond the "
          f"sweep extreme + buffer; risk {p['min_risk_atr']:g}-{p['max_risk_atr']:g} ATR",
          f"- **older data / unseen data split**: unseen from {t[min(split, n - 1)]:%Y-%m-%d}",
          f"- **signals found**: {int(sig.signal[1].sum()):,} buys, {int(sig.signal[-1].sum()):,} sells\n",
          "## Does the rule catch your trades?\n",
          _table(exdf, list(exdf.columns)) if len(exdf) else "_no examples given_\n",
          "\n## VWAP just before your marked entries\n",
          "Daily VWAP (resets 17:00 New York = 03:30 IST), tick-volume weighted. 'sd' = the VWAP band width "
          "(1 standard deviation). Wrong side = below VWAP for a buy.\n",
          _table(vwdf, list(vwdf.columns)) if len(vwdf) else "_none_\n",
          f"\n## Every signal by distance from VWAP ({max(p['targets_r']):g}R target, "
          f"{max(p['horizons']) * 5} min limit; overlapping signals included)\n",
          _table(bdf, list(bdf.columns)),
          "\n## The plain rule, every signal, any session (older data, then unseen data)\n",
          "breakeven_win_pct = win rate needed to break even at that target (before spread). R = multiples of the risk.\n",
          _table(allrow, cols),
          "\n## VWAP filters, any session\n",
          _table(vf_rows, cols),
          f"\n## Best versions on the older data ({len(cand)} tested)\n",
          _table(cand.head(15), cols),
          f"\n## Passed the strict filter: {len(surv)}\n",
          _table(surv, cols) if len(surv) else "_none_\n",
          "\n_Research on past data, not a promise of future profit. Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"buys": int(sig.signal[1].sum()), "sells": int(sig.signal[-1].sum()), "tested": int(len(cand)),
               "passed": int(len(surv)),
               "verdicts": surv["verdict"].value_counts().to_dict() if len(surv) else {},
               "examples_caught": int((exdf["rule fired"] == "yes").sum()) if len(exdf) else 0,
               "report": str(out / "report.md")}
    log.info("entries_done", extra={"dataset": dataset, **summary})
    return summary
