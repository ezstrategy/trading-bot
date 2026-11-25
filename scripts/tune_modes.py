"""Compare strategy settings on stored MT5 bars, split into two halves of the history.

Builds the market features once, then backtests each variant (a dict of
``strategy`` overrides) in ``regime`` mode and in its single modes. A change is
only worth keeping if it helps in BOTH halves, not just over the whole period.

    python scripts/tune_modes.py [--dataset XAUUSD] [--point 0.01] [--out reports/tuning.md]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from quant_engine.config import load_config
from quant_engine.data.storage import ParquetStore
from quant_engine.features.context import STRUCTURE_TFS, MarketFeatures
from quant_engine.strategy import report
from quant_engine.strategy.backtest import Backtester
from quant_engine.strategy.modes import build_model
from quant_engine.timeframes import Timeframe

VARIANTS: dict[str, dict] = {
    "current": {},
    "sideways TP1 1R": {"modes.sideways.tp1_max_r": 1.0},
    "sideways TP1 1.5R": {"modes.sideways.tp1_max_r": 1.5},
    "sideways stop >= 1 ATR": {"modes.sideways.min_stop_atr": 1.0},
    "sideways TP1 1R + stop >= 1 ATR": {"modes.sideways.tp1_max_r": 1.0, "modes.sideways.min_stop_atr": 1.0},
    "sideways TP1 1.5R + stop >= 1 ATR": {"modes.sideways.tp1_max_r": 1.5, "modes.sideways.min_stop_atr": 1.0},
}


def with_overrides(cfg, overrides: dict):
    """Copy of the strategy config with dotted-path overrides applied."""
    s = copy.deepcopy(cfg)
    for path, value in overrides.items():
        obj = s
        *parts, last = path.split(".")
        for p in parts:
            obj = getattr(obj, p)
        setattr(obj, last, value)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="XAUUSD")
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--modes", default="regime,sideways,bullish,bearish")
    ap.add_argument("--out", default="reports/tuning.md")
    a = ap.parse_args()
    cfg = load_config()
    store = ParquetStore(cfg.data.storage.root)
    bars = {tf: store.read_bars(a.dataset, tf) for tf in STRUCTURE_TFS}
    features = MarketFeatures(bars, cfg.strategy, a.point)
    t = features.times[Timeframe.M5]
    split = t[0] + (t[-1] - t[0]) / 2
    risk = cfg.strategy.backtest.risk_pct
    rows = []
    for name, ov in VARIANTS.items():
        sc = with_overrides(cfg.strategy, ov)
        for mode in a.modes.split(","):
            tr = Backtester(features, sc, None, model=build_model(mode, features, sc, None)).run()
            if len(tr):
                tr["signal_time"] = pd.to_datetime(tr["signal_time"], utc=True)
            parts = {"all": tr,
                     "1st half": tr[tr["signal_time"] < split] if len(tr) else tr,
                     "2nd half": tr[tr["signal_time"] >= split] if len(tr) else tr}
            for part, sub in parts.items():
                st = report.stats(sub, risk)
                rows.append({"variant": name, "mode": mode, "period": part, **st})
            print(name, mode, report.stats(tr, risk), flush=True)
    cols = ["variant", "mode", "period", "trades", "win_rate_pct", "avg_r", "total_r", "profit_factor",
            "max_drawdown_r"]
    df = pd.DataFrame(rows).reindex(columns=cols)
    md = [f"# Tuning on {a.dataset} M5 {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d} (split {split:%Y-%m-%d})\n",
          "Win rate counts TP and partial wins; breakeven after TP1 is a small win or zero.\n",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
