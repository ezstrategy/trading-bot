"""``python -m quant_engine patterns``: mine stored history and write a plain-language report.

Output folder:

* ``report.md``: what was found, in plain language, with the held-out results.
* ``all_candidates.csv``: every candidate tested (discovery period only).
* ``patterns.csv`` / ``patterns.json``: patterns that passed discovery, with held-out verdicts.
  ``patterns.json`` is also copied to ``research.patterns_file`` for ``mode: patterns``.
* ``holdout_backtest.md`` / ``holdout_trades.csv``: every pattern that passed discovery,
  run through the normal backtester (risk locks, spread, limit fills) on the held-out
  period only. This is the honest out-of-sample number for the whole method.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..features.context import STRUCTURE_TFS, MarketFeatures
from ..logging_setup import get_logger
from ..strategy import report
from ..strategy.backtest import Backtester
from ..timeframes import Timeframe
from .events import build_events
from .miner import SESSION_NAMES, MiningResult, mine
from .model import PatternModel, regime_kwargs

log = get_logger(__name__)
IST = "Asia/Kolkata"


def _table(df: pd.DataFrame, cols: list[str]) -> str:
    if df is None or df.empty:
        return "_none_\n"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                v = float(v)
                cells.append("" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _ist_hint(event: str) -> str:
    if event.startswith("hour ") and event.endswith(" UTC"):
        hh = int(event[5:7])
        ist = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=hh)).tz_convert(IST)
        return f" ({ist:%H:%M} IST)"
    return ""


def family_summary(tested: pd.DataFrame) -> pd.DataFrame:
    """Per idea family: how many candidates looked significant vs what luck alone gives."""
    if tested is None or tested.empty or "family" not in tested:
        return pd.DataFrame()
    g = tested.groupby("family")
    out = pd.DataFrame({
        "candidates": g.size(),
        "p_below_0.05": g["p"].apply(lambda x: int((x < 0.05).sum())),
        "best_avg_r": g["avg_r"].max().round(3),
        "median_avg_r": g["avg_r"].median().round(3),
    })
    out["expected_by_luck"] = (out["candidates"] * 0.05).round(1)
    out["share_pct"] = (100 * out["p_below_0.05"] / out["candidates"]).round(1)
    return out.reset_index().sort_values("share_pct", ascending=False)


def write_report(res: MiningResult, meta: dict, holdout: dict, out: Path) -> str:
    """Plain-language report (also returned)."""
    s = res.survivors
    counts = s["verdict"].value_counts().to_dict() if len(s) else {}
    md = ["# Gold pattern search\n"]
    md += [f"- **{k}**: {v}" for k, v in meta.items()]
    md += [
        "\n## In plain words\n",
        f"I tested **{res.n_tests:,} candidate patterns** (event x market type x session x BUY/SELL x exit rule) "
        f"on the older data, up to **{res.split_time:%Y-%m-%d}**. With that many tries, some look great by "
        "luck alone, so a pattern only counts if it passes a strict false-discovery filter, beats costs and "
        "works in most of the discovery period. Then each survivor is checked on the newer data it never saw.\n",
        f"- Passed discovery: **{len(s)}**",
        f"- Confirmed on the unseen data: **{counts.get('confirmed', 0)}**",
        f"- Held up but not proven yet: **{counts.get('held up (not proven)', 0)}**",
        f"- Failed on the unseen data: **{counts.get('failed', 0)}**",
        f"- Too few trades on the unseen data to judge: **{counts.get('not enough trades', 0)}**\n",
    ]
    if not len(s):
        md.append("> Nothing passed. That is a real result: on this history no simple, repeatable pattern "
                  "beat costs reliably enough. The 'best leads' table shows the closest ones (not proven).\n")
    md += ["## Patterns that passed discovery\n",
           "R = profit in units of the stop distance after spread. 'val_' columns are the unseen data.\n",
           _table(s.assign(event=s["event"] + s["event"].map(_ist_hint)) if len(s) else s,
                  ["verdict", "regime", "side", "event", "session", "sl_atr", "tp_atr", "horizon",
                   "trades", "win_rate_pct", "avg_r", "p", "val_trades", "val_win_rate_pct", "val_avg_r",
                   "val_total_r", "val_p"])]
    if len(s):
        md += ["\nDescriptions:\n"] + [f"- **{r.verdict}**: {r.description}" for r in s.itertuples()] + [""]
    md += ["\n## Whole method on unseen data (normal backtester)\n",
           "Every pattern that passed discovery, traded together through the normal backtester "
           "(limit fills, spread, daily/weekly loss locks) on the unseen period only.\n",
           _table(pd.DataFrame([holdout]), list(holdout.keys())) if holdout else "_no signals_\n"]
    if holdout and holdout.get("trades", 0) < 30:
        md.append("> Fewer than 30 trades: too few to judge.\n")
    md += ["\n## Each idea family: more than luck?\n",
           "If an idea family had no real edge, about 5% of its candidates would still show p < 0.05 by "
           "luck alone. A family well above 5% hints at something real; near 5% means noise.\n",
           _table(family_summary(res.tested), ["family", "candidates", "p_below_0.05", "expected_by_luck",
                                               "share_pct", "best_avg_r", "median_avg_r"])]
    md += ["\n## Best leads that did not pass (information only, not proven)\n",
           _table(res.leads, ["regime", "side", "event", "session", "tp_atr", "horizon", "trades",
                              "avg_r", "p", "bh_pass", "positive_thirds", "val_trades", "val_avg_r"]),
           "\n## Market types\n",
           "bullish / bearish = H4 close above / below its EMA50 with the EMA sloping that way by at "
           "least 0.5 H4 ATR over 10 candles; neutral = everything else (sideways). Sessions: "
           + ", ".join(f"{k} = {v}" for k, v in SESSION_NAMES.items()) + " (New York clock).\n",
           "\n_These are research results on past data, not a promise of future profit. "
           "Signals only: nothing here places trades._\n"]
    text = "\n".join(md)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def run_patterns(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 source_note: str = "", backtest: bool = True, timeframe: str = "M5",
                 only_families: list[str] | None = None) -> dict:
    """Mine ``dataset`` on ``timeframe`` candles (M5 scalps, or H1 / H4 swings) and write the
    report files into ``out_dir``. Returns a summary dict."""
    sc = cfg.strategy
    tf = Timeframe.parse(timeframe)
    rc = sc.research.for_timeframe(tf.value)
    if only_families:
        rc = rc.model_copy(update={"only_families": list(only_families)})
    bars = {tf: store.read_bars(dataset, tf, start, end) for tf in STRUCTURE_TFS}
    missing = [tf.value for tf, b in bars.items() if b.empty]
    if missing:
        raise ValueError(f"No stored {dataset} data for {', '.join(missing)}")
    m5 = bars[Timeframe.M5]
    entry = bars[tf]
    quality = validate_bars(entry, tf, cfg.data.quality, dataset=f"{dataset} {tf.value}")
    et = build_events(entry, bars[Timeframe.H4], rc, sc.sessions, point,
                      sc.backtest.assumed_spread_points, regime_kwargs(sc), tf)
    res = mine(et, rc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.tested.to_csv(out / "all_candidates.csv", index=False)
    res.survivors.to_csv(out / "patterns.csv", index=False)
    pats = [] if res.survivors.empty else json.loads(res.survivors.to_json(orient="records"))
    payload = {"made": pd.Timestamp.now(tz="UTC").isoformat(), "data": source_note or dataset, "timeframe": tf.value,
               "split_time": res.split_time.isoformat(), "patterns": pats}
    (out / "patterns.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pf = Path(rc.patterns_file)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    holdout: dict = {}
    if backtest and pats:
        features = MarketFeatures(bars, sc, point)
        model = PatternModel(features, sc, None, patterns=pats, start=res.split_time, timeframe=tf)
        trades = Backtester(features, sc, None, model=model).run()
        trades.to_csv(out / "holdout_trades.csv", index=False)
        holdout = report.stats(trades, sc.backtest.risk_pct)
        (out / "holdout_backtest.md").write_text(report.to_markdown(
            "Patterns on unseen data", {"from": f"{res.split_time:%Y-%m-%d %H:%M} UTC"}, trades,
            model.rejections, sc.backtest.risk_pct), encoding="utf-8")

    regime_share = pd.Series(et.regime).value_counts(normalize=True).round(3).to_dict()
    meta = {
        "data": source_note or dataset,
        "period (UTC)": f"{m5['time'].iat[0]:%Y-%m-%d} to {m5['time'].iat[-1] + Timeframe.M5.delta:%Y-%m-%d}",
        "discovery / unseen split": f"older {rc.discover_frac:.0%} / newer {1 - rc.discover_frac:.0%} "
                                    f"(unseen from {res.split_time:%Y-%m-%d %H:%M} UTC)",
        "pattern timeframe": (f"{tf.value} (swing trades: stops/targets in {tf.value} ATR, "
                              f"held up to {max(rc.horizons) * tf.minutes / 60:g} h)" if tf != Timeframe.M5
                              else "M5 (scalps)"),
        f"{tf.value} candles": f"{len(entry):,}",
        "time in each market (H4)": ", ".join(f"{k} {v:.0%}" for k, v in regime_share.items()),
        f"{tf.value} data quality issues": quality.counts() or "none",
        "idea families tested": ", ".join(rc.only_families) if rc.only_families else "all",
        "spread": (f"fixed {rc.fixed_spread_points:g} points in every year" if rc.fixed_spread_points is not None
                   else f"from the bars ({sc.backtest.assumed_spread_points:g} points when missing)"),
    }
    write_report(res, meta, holdout, out)
    summary = {"timeframe": tf.value, "tested": res.n_tests, "passed_discovery": int(len(res.survivors)),
               "verdicts": res.survivors["verdict"].value_counts().to_dict() if len(res.survivors) else {},
               "holdout_backtest": holdout, "report": str(out / "report.md"), "patterns_file": str(pf)}
    log.info("patterns_done", extra={"dataset": dataset, **{k: v for k, v in summary.items() if k != "holdout_backtest"}})
    return summary
