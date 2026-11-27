"""EXP-015: one-shot test of the London breakout on untouched data (default 2015-2019).

``python -m quant_engine london-oos`` (london_untouched_test.bat). Two versions are fixed in
advance and nothing is changed after seeing the result (rulebook rules 1 and 7):

* **published**: the rules from the source (mql5 blog 777540), as in EXP-011.
* **trend version**: the version chosen on 2020-2024 data in EXP-012 (H4 trend filter, 3R,
  break-even at 1R, close by 16:00 New York, no range-size filter).

No London-breakout test has used 2015-2019. Costs: the simulation pays a $0.08 spread; the
historical bar spread above that is added per trade, plus slippage on both fills (rule 6).
Every rule-8 metric is reported. Bonferroni over the two versions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .classics import LB_DEFAULT, lb_day_groups, london_breakout
from .events import regime_per_bar
from .metrics import apply_costs, full_report, report_markdown
from .model import regime_kwargs
from .regime4 import regime4_series

log = get_logger(__name__)

VERSIONS = {
    "published": dict(LB_DEFAULT),
    "trend version (EXP-012)": {**LB_DEFAULT, "target_r": 3.0, "be_r": 1.0, "close_min": 16 * 60,
                                "range_filter": False, "stop": "range", "trend": "with"},
}


def run_london_oos(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                   start: pd.Timestamp, end: pd.Timestamp, source_note: str = "") -> dict:
    """Run both fixed versions once on [start, end) and write ``report.md`` and trade files."""
    sc = cfg.strategy
    costs = sc.research.costs
    risk_pct = sc.backtest.risk_pct
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, Timeframe.H4, start - pd.Timedelta(days=60), end)
    if m5.empty or h4.empty:
        raise ValueError(f"No stored {dataset} M5/H4 data for {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    quality = validate_bars(m5, Timeframe.M5, cfg.data.quality, dataset=f"{dataset} M5")
    base_sp = costs.min_spread_points * point
    t = pd.DatetimeIndex(m5["time"])
    reg4 = regime_per_bar(t, h4, regime4_series(h4, sc.research.regime4))
    reg2 = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
    bar_sp = (m5["spread"].to_numpy(float) * point) if "spread" in m5 else np.full(len(m5), base_sp)
    groups = lb_day_groups(m5, sc.sessions)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# EXP-015: London breakout on untouched data (one shot)\n",
          f"- **data**: {source_note or dataset}",
          f"- **period (UTC)**: {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d} (no London-breakout test used it before)",
          f"- **costs**: historical bar spread (at least ${base_sp:.2f}), commission {costs.commission_per_lot:g}/lot, "
          f"slippage {costs.slippage_points:g} points per fill",
          f"- **M5 data quality issues**: {quality.counts() or 'none'}",
          f"- **risk**: {risk_pct:g}% of equity per trade (compounded)",
          f"- **versions fixed in advance**: {len(VERSIONS)}; a result counts only with p < {0.05 / len(VERSIONS):.3f}\n"]
    summary = {}
    for name, p in VERSIONS.items():
        tr = pd.DataFrame(london_breakout(m5, base_sp, sc.sessions, p, reg2, groups))
        if tr.empty:
            md.append(f"## {name}\n\n_no trades_\n")
            summary[name] = {"trades": 0}
            continue
        idx = t.get_indexer(pd.DatetimeIndex(tr["entry_time"]))
        tr["risk"] = (tr["entry"] - tr["stop"]).abs()
        extra_sp = np.maximum(bar_sp[idx] - base_sp, 0.0)        # historical spread above the simulated $0.08
        tr["extra_spread"] = extra_sp
        tr["r"] = tr["r"] - extra_sp / tr["risk"]
        tr["regime"] = reg4[idx]
        tr = apply_costs(tr, costs, point)
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        tr.to_csv(out / f"trades_{slug}.csv", index=False)
        rep = full_report(tr, risk_pct)
        o = rep["overall"]
        verdict = ("REJECTED (no edge after costs)" if o["expectancy_r"] <= 0 else
                   "positive and significant" if o["p_value"] < 0.05 / len(VERSIONS) else
                   "positive but not significant")
        md += [f"## {name}: {verdict}\n", "Rules: " + ", ".join(f"{k}={v}" for k, v in p.items()) + "\n",
               report_markdown(name, rep)]
        summary[name] = {k: o.get(k) for k in ("trades", "win_rate_pct", "profit_factor", "expectancy_r",
                                                "net_profit_pct", "max_drawdown_pct", "p_value")} | {"verdict": verdict}
    md.append("\n_One-shot test: these rules are not changed after this result (rule 7). Signals only._\n")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("london_oos_done", extra={"versions": list(summary)})
    return summary | {"report": str(out / "report.md")}
