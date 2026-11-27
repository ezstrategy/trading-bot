"""Rule 2 study: how gold behaves in each market regime, before any strategy is written.

``python -m quant_engine study`` (study_regimes.bat). Descriptive only: nothing here is a
strategy and nothing is optimised. By default only the **development** part (first 60% by
time) is studied, so later validation and out-of-sample tests stay separate.

Questions answered per four-state regime (bullish / bearish / sideways / uncertain), worked out on
H4 by default or on H1 with ``--regime-tf H1`` (EXP-021: same settings, counted in H1 candles):

1. How much time the market spends in each regime and how long a spell lasts.
2. Drift: the average next-hour move (in M5 ATR) by rule-11 session, in the regime's direction.
3. Momentum or mean reversion: correlation of the last hour's move with the next hour's move.
4. Breakouts: after an M5 close beyond the previous 2-hour high / low, how often price runs
   another 1 ATR before coming back 1 ATR (follow-through), with and against the regime.
5. Sweeps: after a new 2-hour low / high is taken but the candle closes back inside, how often
   price then moves 1 ATR in the reversal direction before 1 ATR the other way, both measured
   from the sweep candle's close (symmetric, so 50% = no edge).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..features.atr import atr as atr_fn
from ..logging_setup import get_logger
from ..timeframes import Timeframe
from .events import regime_per_bar
from .metrics import SESSION_ORDER, session_of
from .regime4 import STATES, regime4_series
from .runner import _table

log = get_logger(__name__)
HOUR = 12        # M5 candles


def first_touch(h: np.ndarray, l: np.ndarray, c: np.ndarray, idx: np.ndarray, up: np.ndarray,
                dn: np.ndarray, horizon: int = 36) -> np.ndarray:
    """+1 if price reaches ``up`` before ``dn`` within ``horizon`` candles after ``idx``, -1 if
    ``dn`` first (or both in one candle), 0 if neither."""
    out = np.zeros(len(idx), dtype=int)
    for n_, (i, u, d) in enumerate(zip(idx, up, dn)):
        for k in range(i + 1, min(i + 1 + horizon, len(c))):
            if l[k] <= d:
                out[n_] = -1
                break
            if h[k] >= u:
                out[n_] = 1
                break
    return out


def _rate(x: np.ndarray) -> dict:
    hit = x != 0
    return {"cases": int(len(x)), "decided": int(hit.sum()),
            "first_way_pct": round(100 * float((x[hit] == 1).mean()), 1) if hit.any() else np.nan}


def run_study(cfg: AppConfig, store: ParquetStore, dataset: str, out_dir: str | Path,
              start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
              part: str = "development", source_note: str = "", regime_tf: Timeframe = Timeframe.H4) -> dict:
    """Write ``report.md`` with the five tables described in the module docs."""
    sc = cfg.strategy
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, regime_tf, start, end)      # regime candles (H4, or H1 for EXP-021)
    if m5.empty or h4.empty:
        raise ValueError(f"No stored {dataset} M5/{regime_tf.value} data")
    t_all = pd.DatetimeIndex(m5["time"])
    cut = t_all[0] + (t_all[-1] - t_all[0]) * 0.6
    if part == "development":
        m5 = m5[m5["time"] < cut].reset_index(drop=True)
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_fn(m5, 14).to_numpy(float)
    h4s = regime4_series(h4, sc.research.regime4)
    reg = regime_per_bar(t, h4, h4s, ctx_tf=regime_tf)
    sess = session_of(t)
    n = len(c)

    # 1. time in regime and spell length (H4 candles)
    h4_in = h4[pd.DatetimeIndex(h4["time"]) < t[-1]]
    lab = h4s.loc[h4_in.index].to_numpy()
    spells = pd.Series((lab[1:] != lab[:-1]).cumsum()).groupby(lab[1:]).value_counts() if len(lab) > 1 else pd.Series()
    t1 = []
    for s in STATES:
        share = float((lab == s).mean() * 100) if len(lab) else 0
        lens = spells.loc[s].to_numpy() if s in spells.index.get_level_values(0) else np.array([])
        t1.append({"regime": s, "time_pct": round(share, 1), "spells": int(len(lens)),
                   f"median_spell_{regime_tf.value.lower()}": float(np.median(lens)) if len(lens) else np.nan,
                   f"mean_spell_{regime_tf.value.lower()}": round(float(lens.mean()), 1) if len(lens) else np.nan})

    # 2. drift: next-hour move in ATR (signed by regime direction for trends)
    fwd = np.full(n, np.nan)
    fwd[:-HOUR] = (c[HOUR:] - c[:-HOUR]) / a[:-HOUR]
    gap_ok = np.zeros(n, bool)
    gap_ok[:-HOUR] = (t[HOUR:] - t[:-HOUR]) == HOUR * Timeframe.M5.delta
    hourly = (np.arange(n) % HOUR == 0) & gap_ok & np.isfinite(fwd) & (a > 0)
    t2 = []
    for s in STATES:
        sign = 1 if s == "bullish" else -1 if s == "bearish" else 1
        for se in SESSION_ORDER[:-1]:
            x = fwd[hourly & (reg == s) & (sess == se)] * sign
            se_ = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 2 else np.nan
            t2.append({"regime": s, "session": se, "hours": int(len(x)),
                       "avg_next_hour_atr": round(float(x.mean()), 3) if len(x) else np.nan,
                       "t_stat": round(float(x.mean() / se_), 2) if len(x) > 2 and se_ > 0 else np.nan})

    # 3. momentum vs mean reversion
    back = np.full(n, np.nan)
    back[HOUR:] = (c[HOUR:] - c[:-HOUR]) / a[HOUR:]
    t3 = []
    for s in STATES:
        m = hourly & (reg == s) & np.isfinite(back)
        if m.sum() > 30:
            t3.append({"regime": s, "hours": int(m.sum()),
                       "corr_last_vs_next_hour": round(float(np.corrcoef(back[m], fwd[m])[0, 1]), 3)})

    # 4. breakouts of the previous 2-hour range, 5. sweeps that close back inside
    lb = 24
    prior_hi = pd.Series(h).shift(1).rolling(lb, min_periods=lb).max().to_numpy()
    prior_lo = pd.Series(l).shift(1).rolling(lb, min_periods=lb).min().to_numpy()
    ok = (a > 0) & np.isfinite(prior_hi)
    brk_up, brk_dn = ok & (c > prior_hi), ok & (c < prior_lo)
    sw_lo = ok & (l < prior_lo) & (c > prior_lo)
    sw_hi = ok & (h > prior_hi) & (c < prior_hi)
    # one case per hour at most, so cases are closer to independent
    hour_key = np.asarray(t.floor("h").asi8 // 10**9)
    def once(mask):
        idx = np.nonzero(mask)[0]
        _, first = np.unique(hour_key[idx], return_index=True)
        return idx[first]
    t4, t5 = [], []
    for s in STATES:
        for name, mask, d in (("break up", brk_up, 1), ("break down", brk_dn, -1)):
            idx = once(mask & (reg == s))
            r = first_touch(h, l, c, idx, c[idx] + a[idx], c[idx] - a[idx]) * d
            with_ = ("with" if (s == "bullish" and d == 1) or (s == "bearish" and d == -1) else
                     "against" if s in ("bullish", "bearish") else "-")
            t4.append({"regime": s, "breakout": name, "vs regime": with_, **_rate(r)})
        for name, mask, d in (("swept low, closed back above", sw_lo, 1), ("swept high, closed back below", sw_hi, -1)):
            idx = once(mask & (reg == s))
            # symmetric: both levels 1 ATR from the sweep candle's close (a fair 50/50 baseline)
            r = first_touch(h, l, c, idx, c[idx] + a[idx], c[idx] - a[idx]) * d
            t5.append({"regime": s, "sweep": name, **_rate(r)})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {"1_time_in_regime": pd.DataFrame(t1), "2_drift": pd.DataFrame(t2), "3_momentum": pd.DataFrame(t3),
            "4_breakouts": pd.DataFrame(t4), "5_sweeps": pd.DataFrame(t5)}
    for k, df in tabs.items():
        df.to_csv(out / f"{k}.csv", index=False)
    md = ["# Gold behaviour by market regime (descriptive study, rule 2)\n",
          f"- **data**: {source_note or dataset}",
          f"- **studied part**: {part} ({t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d})",
          f"- **regime**: four-state module on **{regime_tf.value}** (bullish / bearish / sideways / uncertain), "
          f"known at each {regime_tf.value} close; settings counted in {regime_tf.value} candles",
          "- Nothing here is a strategy or a profit claim; costs are not included. Moves are in M5 ATR(14).\n",
          "## 1. Time in each regime\n", _table(tabs["1_time_in_regime"], list(tabs["1_time_in_regime"].columns)),
          "\n## 2. Drift: average next-hour move by session (trend regimes signed in their direction)\n",
          "Positive in a bullish row = price kept rising; |t_stat| above about 2 is unlikely to be luck.\n",
          _table(tabs["2_drift"], list(tabs["2_drift"].columns)),
          "\n## 3. Momentum or mean reversion (last hour vs next hour)\n",
          "Positive = moves tend to continue; negative = they tend to reverse.\n",
          _table(tabs["3_momentum"], list(tabs["3_momentum"].columns)),
          "\n## 4. Breakouts of the previous 2-hour range: +1 ATR more before -1 ATR back?\n",
          "50% = a coin flip (before costs).\n", _table(tabs["4_breakouts"], list(tabs["4_breakouts"].columns)),
          "\n## 5. Sweeps that close back inside: +1 ATR reversal before -1 ATR, both from the close?\n",
          "50% = a coin flip (before costs).\n",
          _table(tabs["5_sweeps"], list(tabs["5_sweeps"].columns)),
          "\n_Descriptive research on past data. Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("study_done", extra={"part": part, "bars": n})
    return {"part": part, "regime_tf": regime_tf.value, "from": f"{t[0]:%Y-%m-%d}", "to": f"{t[-1]:%Y-%m-%d}", "report": str(out / "report.md")}
