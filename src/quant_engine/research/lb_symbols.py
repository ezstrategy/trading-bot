"""EXP-017: the unchanged London breakout (EXP-016 original) on other instruments, one shot.

``python -m quant_engine lb-symbols`` (lb_symbols.bat). Owner asked on 2026-11-26 whether other
metals / FX pairs do better. Fixed before the run (rules 1 and 7):

* Rules: :data:`lb_phase2.ORIGINAL`, unchanged (Asian range 20:00-01:00 NY, stop orders, H4 trend
  filter, stop at the other side, 3R, break-even 1R, close 16:00 NY). The regime, buffer and stop cap
  are all relative to price / ATR, so nothing is rescaled per instrument.
* Instruments: XAGUSD, EURUSD, GBPUSD, USDJPY (Bonferroni over 4: p < 0.0125). XAUUSD is shown as
  the reference only.
* Costs: each instrument's own historical MT5 bar spread (floor 1 point), slippage
  ``costs.slippage_points`` points per fill (GUESS, same point count for all), commission as
  configured (0, owner); stress x1.5 and x2.
* Verdict on all available data from ``start``; the same three date parts as EXP-016 are shown.
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
from .classics import lb_day_groups
from .events import regime_per_bar
from .lb_phase2 import ORIGINAL, _detail, _row, _split_of, boot_stats, lb_trades, newey_west_t, with_costs
from .model import regime_kwargs
from .regime4 import regime4_series
from .runner import _table

log = get_logger(__name__)

SYMBOLS = ("XAGUSD", "EURUSD", "GBPUSD", "USDJPY")
REFERENCE = "XAUUSD"


def run_lb_symbols(cfg: AppConfig, store: ParquetStore, points: dict[str, float], out_dir: str | Path,
                   start: pd.Timestamp, end: pd.Timestamp | None = None, symbols=SYMBOLS,
                   reference: str | None = REFERENCE, source_note: str = "MT5 broker bars (UTC)") -> dict:
    """Run :data:`ORIGINAL` once per instrument and write ``report.md`` and ``detail/<symbol>.md``."""
    sc = cfg.strategy
    costs, risk_pct = sc.research.costs, sc.backtest.risk_pct
    out = Path(out_dir)
    (out / "detail").mkdir(parents=True, exist_ok=True)
    alpha = 0.05 / len(symbols)
    md = ["# EXP-017: unchanged London breakout on other instruments (one shot)\n",
          f"- **data**: {source_note}, from {start:%Y-%m-%d}",
          "- **rules**: EXP-016 original, unchanged: " + ", ".join(f"{k}={v}" for k, v in ORIGINAL.items()),
          f"- **costs**: each instrument's historical bar spread (floor 1 point), slippage {costs.slippage_points:g} "
          f"points per fill, commission {costs.commission_per_lot:g}; stress x1.5 / x2",
          f"- **instruments fixed in advance**: {', '.join(symbols)}; a result counts only with p < {alpha:.4f} "
          f"(Bonferroni over {len(symbols)}); {reference or 'no'} reference row\n"]
    rows, summary = [], {}
    for sym in ([reference] if reference else []) + list(symbols):
        point = points.get(sym)
        m5 = store.read_bars(sym, Timeframe.M5, start, end)
        h4 = store.read_bars(sym, Timeframe.H4, start - pd.Timedelta(days=60), end)
        if point is None or m5.empty or h4.empty:
            rows.append({"instrument": sym, "period": "no stored M5/H4 data", "trades": 0})
            summary[sym] = {"trades": 0, "verdict": "no data"}
            continue
        q = validate_bars(m5, Timeframe.M5, cfg.data.quality, dataset=f"{sym} M5")
        t = pd.DatetimeIndex(m5["time"])
        reg2 = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
        reg4 = regime_per_bar(t, h4, regime4_series(h4, sc.research.regime4))
        groups = lb_day_groups(m5, sc.sessions)
        base_sp = point
        bar_sp = m5["spread"].to_numpy(float) * point if "spread" in m5 else np.full(len(m5), base_sp)
        raw = lb_trades(m5, base_sp, ORIGINAL, reg2, groups)
        if raw.empty:
            rows.append({"instrument": sym, "period": f"{t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}", "trades": 0})
            summary[sym] = {"trades": 0, "verdict": "no trades"}
            continue
        st = {}
        for k in (1.0, 1.5, 2.0):
            x = with_costs(raw, bar_sp, base_sp, costs, point, k)
            x["split"] = _split_of(x["entry_time"])
            x["regime"] = reg4[x["entry_i"].to_numpy(int)]
            st[f"x{k:g}"] = x
        tr = st["x1"]
        r = tr["r"].to_numpy()
        b, (nw_t, nw_p) = boot_stats(r), newey_west_t(r)
        m = _row(r, risk_pct)
        is_ref = sym == reference
        verdict = ("reference" if is_ref else
                   "REJECTED (no edge after costs)" if m["exp_R"] <= 0 else
                   "positive and significant" if b["p_boot"] < alpha and nw_p < alpha else
                   "positive but not significant")
        rows.append({"instrument": sym + (" (reference)" if is_ref else ""),
                     "period": f"{t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}", **m, "NW_p": nw_p,
                     "median_spread_pts": round(float(np.median(tr["bar_spread"]) / point), 1),
                     "cost_R": round(float(tr["cost_r"].mean()), 3),
                     "exp_x2": round(float(st["x2"]["r"].mean()), 3),
                     "years_positive": _years_pos(tr), "quality": q.counts() or "none", "verdict": verdict})
        summary[sym] = {k: rows[-1][k] for k in ("trades", "win%", "PF", "exp_R", "DD%", "p_boot", "verdict")}
        (out / "detail" / f"{sym}.md").write_text(_detail(f"{sym}: original London breakout", tr, risk_pct, st),
                                                 encoding="utf-8")
        tr.to_csv(out / f"trades_{sym}.csv", index=False)
    cols = ["instrument", "period", "trades", "win%", "PF", "exp_R", "exp_CI95", "DD%", "avg_win", "avg_loss",
            "max_losses", "p_boot", "NW_p", "median_spread_pts", "cost_R", "exp_x2", "years_positive", "quality",
            "verdict"]
    tab = pd.DataFrame(rows).reindex(columns=cols)
    md += ["## Results (all available data, costs included)\n", _table(tab, cols),
           "\nexp_CI95 = moving-block bootstrap 95% interval of R per trade; p_boot and NW_p = one-sided p that it "
           "is above 0; exp_x2 = R per trade at double costs. A positive result here would still be "
           "'promising, needs forward validation', not a validated strategy. Full metrics per instrument "
           "(by part, costs, years, months, long / short, sessions, regimes): `detail/<instrument>.md`.\n",
           "\n_One-shot test, rules not changed after this result (rule 7). Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("lb_symbols_done", extra={"symbols": list(summary)})
    return summary | {"report": str(out / "report.md")}


def _years_pos(tr: pd.DataFrame) -> str:
    y = tr.groupby(pd.DatetimeIndex(tr["entry_time"]).year)["r"].sum()
    return f"{int((y > 0).sum())}/{len(y)}"
