"""Work backwards from big fast moves: ``python -m quant_engine bigmoves``.

1. **How often**: for every M5 candle, the largest move up and down within the next
   ``window_bars`` candles (from the next candle's open). A "big move" is at least
   ``move_pips`` pips (gold: 1 pip = ``pip_usd`` = $0.10, so 30 pips = $3).
2. **What came before**: for each setup (event), how much more often a big move in the
   setup's direction followed it than after an average candle (the "lift"), measured on the
   discovery period only.
3. **Simple strategy test**: the combined setups (S/R + price action + liquidity) traded with
   fixed dollar targets and stops (``target_pips`` x ``stop_pips``), closed after
   ``window_bars`` candles, spread paid, using the same discovery / unseen split and
   false-discovery filter as :mod:`.miner`. Win rate AND profit after costs are reported:
   a high win rate with large losses can still lose money.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..features.context import STRUCTURE_TFS
from ..logging_setup import get_logger
from ..timeframes import Timeframe
from .events import SETUP_FAMILY, EventTable, build_events, event_family
from .miner import mine
from .model import regime_kwargs
from .runner import _table

log = get_logger(__name__)


def forward_moves(et: EventTable, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(max rise, max fall) in USD within ``window`` candles after each candle's close,
    measured from the next candle's open (NaN when the data ends or a gap > 3 h is crossed)."""
    n = len(et.c)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    m = n - window - 1
    if m <= 0:
        return up, dn
    j = np.arange(m)
    entry = et.o[j + 1]
    hi = np.max(np.stack([et.h[j + k] for k in range(1, window + 1)]), axis=0)
    lo = np.min(np.stack([et.l[j + k] for k in range(1, window + 1)]), axis=0)
    tv = et.times.to_numpy()
    ok = (tv[j + window] - tv[j]) <= (window * et.tf.delta + pd.Timedelta(hours=3)).to_timedelta64()
    up[:m] = np.where(ok, hi - entry, np.nan)
    dn[:m] = np.where(ok, entry - lo, np.nan)
    return up, dn


def move_frequency(et: EventTable, up: np.ndarray, dn: np.ndarray, sizes_usd: list[float]) -> pd.DataFrame:
    """Per year: share of candles followed by a move of each size (either direction)."""
    yr = et.times.year
    rows = []
    for y in sorted(set(yr)):
        m = (yr == y) & ~np.isnan(up)
        row = {"year": int(y), "candles": int(m.sum()), "median_price": round(float(np.median(et.c[m])), 0)}
        for s in sizes_usd:
            big = (up[m] >= s) | (dn[m] >= s)
            row[f"move >= ${s:g} (%)"] = round(100 * float(big.mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def lift_table(et: EventTable, up: np.ndarray, dn: np.ndarray, size: float, split: int,
               families: tuple[str, ...]) -> pd.DataFrame:
    """For each event (discovery period only): how often a ``size`` move in the event's own
    direction followed it, versus after any candle."""
    disc = np.arange(len(et.c)) < split
    base_ok = disc & ~np.isnan(up)
    base_up = float((up[base_ok] >= size).mean())
    base_dn = float((dn[base_ok] >= size).mean())
    rows = []
    for name, mask in et.events.items():
        fam = event_family(name)
        if fam not in families:
            continue
        sel = mask & base_ok
        if sel.sum() < 30:
            continue
        d = et.natural[name]
        hit = (up[sel] >= size) if d == 1 else (dn[sel] >= size)
        wrong = (dn[sel] >= size) if d == 1 else (up[sel] >= size)
        base = base_up if d == 1 else base_dn
        rows.append({"setup": name, "family": fam, "direction": "BUY" if d == 1 else "SELL",
                     "times_seen": int(sel.sum()), "big_move_its_way_pct": round(100 * float(hit.mean()), 1),
                     "big_move_against_pct": round(100 * float(wrong.mean()), 1),
                     "after_any_candle_pct": round(100 * base, 1),
                     "lift": round(float(hit.mean()) / base, 2) if base > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame()


def run_bigmoves(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 source_note: str = "") -> dict:
    """Run the three steps on M5 candles and write ``report.md`` plus CSV files."""
    sc = cfg.strategy
    bm = {"pip_usd": 0.10, "move_pips": [30, 50], "window_bars": 6,
          "target_pips": [30, 50], "stop_pips": [20, 30, 50], **sc.research.bigmoves}
    pip = float(bm["pip_usd"])
    window = int(bm["window_bars"])
    bars = {tf: store.read_bars(dataset, tf, start, end) for tf in STRUCTURE_TFS}
    missing = [tf.value for tf, b in bars.items() if b.empty]
    if missing:
        raise ValueError(f"No stored {dataset} data for {', '.join(missing)}")
    rc = sc.research.model_copy(update={
        "exit_units": "usd", "sl_atr": [p * pip for p in bm["stop_pips"]],
        "tp_atr": [p * pip for p in bm["target_pips"]], "horizons": [window],
        "only_families": [SETUP_FAMILY]})
    m5 = bars[Timeframe.M5]
    et = build_events(m5, bars[Timeframe.H4], rc, sc.sessions, point, sc.backtest.assumed_spread_points,
                      regime_kwargs(sc), Timeframe.M5)
    split = int(len(et.c) * rc.discover_frac)
    up, dn = forward_moves(et, window)
    sizes = [p * pip for p in bm["move_pips"]]
    freq = move_frequency(et, up, dn, sizes)
    lift = lift_table(et, up, dn, sizes[0], split,
                      (SETUP_FAMILY, "price action", "price action at levels", "smc", "round numbers",
                       "major support / resistance"))
    res = mine(et, rc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out / "move_frequency.csv", index=False)
    lift.to_csv(out / "what_came_before.csv", index=False)
    res.tested.to_csv(out / "strategy_candidates.csv", index=False)
    res.survivors.to_csv(out / "strategy_survivors.csv", index=False)
    pats = [] if res.survivors.empty else json.loads(res.survivors.to_json(orient="records"))
    (out / "patterns.json").write_text(json.dumps({"timeframe": "M5", "patterns": pats}, indent=2), encoding="utf-8")

    cost = float(rc.fixed_spread_points if rc.fixed_spread_points is not None
                 else sc.backtest.assumed_spread_points) * point
    t = res.tested.copy()
    if len(t):
        t["target_pips"] = (t["tp_atr"] / pip).round().astype(int)
        t["stop_pips"] = (t["sl_atr"] / pip).round().astype(int)
        t["breakeven_win_pct"] = (100 * (t["sl_atr"] + cost) / (t["sl_atr"] + t["tp_atr"])).round(1)
        t["avg_usd_per_trade"] = (t["avg_r"] * t["sl_atr"]).round(2)
    cols = ["regime", "side", "event", "session", "target_pips", "stop_pips", "trades", "win_rate_pct",
            "breakeven_win_pct", "avg_usd_per_trade", "p", "bh_pass"]
    best = t.drop_duplicates(["event", "regime", "session", "direction"]).head(15) if len(t) else t
    hi_win = t[t["win_rate_pct"] >= 70].head(10) if len(t) else t

    s = res.survivors
    md = ["# Big fast moves: what comes before them, and a simple strategy test\n",
          f"- **data**: {source_note or dataset}",
          f"- **period (UTC)**: {m5['time'].iat[0]:%Y-%m-%d} to {m5['time'].iat[-1] + Timeframe.M5.delta:%Y-%m-%d}",
          f"- **pip**: 1 pip = ${pip:g} (so {bm['move_pips'][0]} pips = ${sizes[0]:g}); window = "
          f"{window} M5 candles ({window * 5} minutes)",
          f"- **spread charged**: ${cost:.2f} per trade",
          f"- **discovery / unseen split**: unseen from {res.split_time:%Y-%m-%d}\n",
          "## 1. How often do big moves happen?\n",
          "Share of M5 candles followed by a move of at least this size (either way) within the window. "
          "Gold's price roughly doubled since 2020, so a $3 move is much more common now.\n",
          _table(freq, list(freq.columns)),
          f"\n## 2. What came before a {bm['move_pips'][0]}-pip move? (discovery period)\n",
          "lift = how many times more often a big move in the setup's direction followed it than after "
          "an average candle. lift 1.0 = no information. Note that a big move AGAINST is usually about as likely.\n",
          _table(lift.head(20), ["setup", "direction", "times_seen", "big_move_its_way_pct",
                                 "big_move_against_pct", "after_any_candle_pct", "lift"]),
          "\n## 3. Simple strategy test (fixed targets and stops, spread paid)\n",
          f"Tested {res.n_tests} combinations of setup x market x session x target x stop. "
          f"Passed discovery: **{len(s)}**; confirmed on unseen data: "
          f"**{int((s['verdict'] == 'confirmed').sum()) if len(s) else 0}**.\n",
          "breakeven_win_pct = win rate needed just to break even with that target, stop and spread.\n",
          "### Best combinations on the discovery data\n",
          _table(best, cols),
          "\n### Combinations with a 70%+ win rate (discovery data)\n",
          _table(hi_win, cols) if len(hi_win) else "_none with enough trades_\n"]
    if len(s):
        md += ["\n### Passed discovery, checked on unseen data\n",
               _table(s, ["verdict", "regime", "side", "event", "session", "tp_atr", "sl_atr", "trades",
                          "win_rate_pct", "avg_r", "val_trades", "val_win_rate_pct", "val_avg_r"])]
    md += ["\n_Research on past data, not a promise of future profit. Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"tested": res.n_tests, "passed_discovery": int(len(s)),
               "verdicts": s["verdict"].value_counts().to_dict() if len(s) else {},
               "report": str(out / "report.md")}
    log.info("bigmoves_done", extra={"dataset": dataset, **summary})
    return summary
