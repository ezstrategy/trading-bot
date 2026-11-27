"""Run a backtest on stored bars and write the report files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..features.context import STRUCTURE_TFS, MarketFeatures
from ..features.news import NewsCalendar, load_calendar
from ..logging_setup import get_logger
from ..timeframes import Timeframe
from . import report
from .backtest import Backtester
from .modes import MODES, build_model

log = get_logger(__name__)


def run_backtest(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 source_note: str = "", mode: str | None = None) -> dict:
    """Backtest ``dataset`` (a stored instrument name) and write trades.csv,
    summary.json and report.md into ``out_dir``. Returns the summary dict.

    ``mode``: v1, regime, bullish, bearish, sideways (default: ``strategy.mode``), or
    ``compare`` to run every mode on every market and write ``comparison.md``."""
    mode = mode or cfg.strategy.mode
    bars = {tf: store.read_bars(dataset, tf, start, end) for tf in STRUCTURE_TFS}
    missing = [tf.value for tf, b in bars.items() if b.empty]
    if missing:
        raise ValueError(f"No stored {dataset} data for {', '.join(missing)}")
    m5 = bars[Timeframe.M5]
    quality = validate_bars(m5, Timeframe.M5, cfg.data.quality, dataset=f"{dataset} M5")
    features = MarketFeatures(bars, cfg.strategy, point)
    first, last = m5["time"].iat[0], m5["time"].iat[-1]
    cal = NewsCalendar(load_calendar(cfg.strategy.news.calendar_file), cfg.strategy.news)
    use_news = cfg.strategy.news.enabled and cal.covers(first, last)
    news = cal if use_news else None
    if mode == "london":
        return _london(cfg, bars, dataset, point, out_dir, source_note or dataset, quality)
    if mode == "compare":
        return _compare(cfg, features, news, out_dir, source_note or dataset, first, last)
    bt = Backtester(features, cfg.strategy, news, model=build_model(mode, features, cfg.strategy, news))
    trades = bt.run()
    risk = cfg.strategy.backtest.risk_pct
    meta = {
        "data": source_note or dataset,
        "strategy mode": mode,
        "period (UTC)": f"{m5['time'].iat[0]:%Y-%m-%d %H:%M} to {m5['time'].iat[-1] + Timeframe.M5.delta:%Y-%m-%d %H:%M}",
        "M5 candles": len(m5),
        "M5 data quality issues": quality.counts() or "none",
        "risk per trade": f"{risk}% (1R)",
        "assumed spread when missing": f"{cfg.strategy.backtest.assumed_spread_points:g} points",
        "news filter": (f"applied ({len(cal.windows)} blocking windows from {cfg.strategy.news.calendar_file})"
                        if use_news else "not applied (the cached calendar does not cover this period)"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    summary = {"meta": {k: str(v) for k, v in meta.items()}, "overall": report.stats(trades, risk),
               "by_trend_regime": report.breakdown(trades, "trend_regime", risk).to_dict("records") if len(trades) else [],
               "rejections": bt.model.rejections}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(report.to_markdown(f"Backtest: {dataset}", meta, trades,
                                                      bt.model.rejections, risk))
    log.info("backtest_done", extra={"dataset": dataset, "signals": len(trades), "out": str(out)})
    return summary


def _london(cfg: AppConfig, bars, dataset: str, point: float, out_dir, note: str, quality) -> dict:
    """London breakout backtest, replayed with the live signal code (strategy/london.py).

    Costs as in the research runs (EXP-016/020): the simulation pays the spread floor on every
    fill; the historical bar spread above it, slippage on both fills and commission are charged
    per trade."""
    from ..research.metrics import apply_costs
    from .london import london_trades
    sc = cfg.strategy
    costs, risk = sc.research.costs, sc.backtest.risk_pct
    base_sp = costs.min_spread_points * point
    m5 = bars[Timeframe.M5]
    tr = london_trades(m5, bars[Timeframe.H4], sc, base_sp)
    if len(tr):
        t = pd.DatetimeIndex(m5["time"])
        bar_sp = m5["spread"].to_numpy(float) * point if "spread" in m5 else np.full(len(m5), base_sp)
        extra = np.maximum(bar_sp[t.get_indexer(pd.DatetimeIndex(tr["entry_time"]))] - base_sp, 0.0)
        tr["r"] = tr["r"] - extra / tr["risk"]
        tr = apply_costs(tr, costs, point)
        tr["filled"] = True
        tr = tr.rename(columns={"trend": "trend_regime"})
    meta = {"data": note, "strategy mode": "london (EXP-016 rules, natural stop)"
            + ("" if sc.london.is_tested_version() else " - SETTINGS CHANGED, untested"),
            "period (UTC)": f"{m5['time'].iat[0]:%Y-%m-%d %H:%M} to {m5['time'].iat[-1] + Timeframe.M5.delta:%Y-%m-%d %H:%M}",
            "M5 candles": len(m5), "M5 data quality issues": quality.counts() or "none",
            "risk per trade": f"{risk}% (1R)",
            "costs": f"spread at least {costs.min_spread_points:g} points (historical bar spread above it charged), "
                     f"slippage {costs.slippage_points:g} points per fill, commission {costs.commission_per_lot:g}/lot",
            "news filter": "not used (not part of the tested rules)"}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "trades.csv", index=False)
    summary = {"meta": {k: str(v) for k, v in meta.items()}, "overall": report.stats(tr, risk),
               "by_trend_regime": report.breakdown(tr, "trend_regime", risk).to_dict("records") if len(tr) else [],
               "rejections": {}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [f"# Backtest: {dataset}, London breakout\n", *[f"- **{k}**: {v}" for k, v in meta.items()], ""]
    o = summary["overall"]
    lines += ["| " + " | ".join(o) + " |", "|" + "---|" * len(o), "| " + " | ".join(str(v) for v in o.values()) + " |", ""]
    if len(tr):
        from ..research.runner import _table
        by = pd.DataFrame(summary["by_trend_regime"])
        ex = tr["exit_reason"].value_counts().rename_axis("exit").reset_index(name="trades")
        lines += ["## By H4 trend\n", _table(by, list(by.columns)), "", "## Exit reasons\n", _table(ex, ["exit", "trades"])]
    lines.append("\n_Research stage (RESEARCH REQUIRED): use for paper signals only. Signals only, no orders._")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("backtest_done", extra={"dataset": dataset, "mode": "london", "signals": len(tr), "out": str(out)})
    return summary


def _compare(cfg: AppConfig, features: MarketFeatures, news, out_dir, note: str, first, last) -> dict:
    """Run every mode on every market. Single modes are NOT limited to their own market here,
    so the table shows whether each one really does better where it is meant to trade."""
    risk = cfg.strategy.backtest.risk_pct
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, all_trades = [], []
    for mode in MODES:
        bt = Backtester(features, cfg.strategy, news, model=build_model(mode, features, cfg.strategy, news))
        tr = bt.run()
        tr.insert(0, "mode", mode)
        all_trades.append(tr)
        rows.append({"mode": mode, "market": "all", **report.stats(tr, risk)})
        for reg in ("bullish", "bearish", "neutral"):
            rows.append({"mode": mode, "market": reg, **report.stats(tr[tr["trend_regime"] == reg] if len(tr) else tr, risk)})
    table = pd.DataFrame(rows)
    pd.concat(all_trades, ignore_index=True).to_csv(out / "trades.csv", index=False)
    cols = ["mode", "market", "trades", "win_rate_pct", "avg_r", "total_r", "profit_factor", "max_drawdown_r"]
    table = table.reindex(columns=cols)
    md = ["# Strategy comparison\n", f"- **data**: {note}",
          f"- **period (UTC)**: {first:%Y-%m-%d} to {last:%Y-%m-%d}",
          "- **how to read**: each single mode runs on every market. The bullish mode should do best in "
          "bullish markets, bearish in bearish, sideways in neutral; `regime` switches between them.",
          "- Fewer than 30 trades in a cell is too few to judge.\n",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        md.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    summary = {"meta": {"data": note, "mode": "compare"}, "comparison": table.to_dict("records"),
               "overall": table[(table["mode"] == "regime") & (table["market"] == "all")].iloc[0].dropna().to_dict(),
               "by_trend_regime": [], "rejections": {}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text("\n".join(md) + "\n")
    return summary
