"""EXP-016: Phase 2 research on the London Asian-range breakout (owner brief, 2026-11-27).

``python -m quant_engine lb-phase2`` (lb_phase2.bat). The original strategy is the EXP-012 /
EXP-015 "trend version" and is never changed: :data:`ORIGINAL` is a copy of it, and
:func:`lb_trades` in ``entry="stop"`` mode must give exactly the trades of
:func:`classics.london_breakout` (checked at run time; the run stops if not).

Plan, written before any result was seen (rules 1 and 7):

* **Part 1** reproduces EXP-015 (914 trades, +81.4R on 2015-2019) and stops if that fails; reports
  the historical spread, contract size / lot rounding, the test history, a Newey-West t-test and
  a moving-block bootstrap (blocks of 10 trades; at most one trade a day, so day-level).
* **Parts 2-3** are one change at a time against the original: H1 range width, H2 breakout candle
  (as a filter only with close-confirmed entries, because with stop entries the candle is known
  only after the fill), H3 previous-day levels, H4 volatility, H5 news (only if a historical
  calendar file exists), H6 entry method, exits A-F.
* **Data split** (chronological): development 2015-2019, validation 2020-2022, test 2023+.
  None of it is untouched (2020-2026 chose the version, 2015-2019 tested it in EXP-015), so the
  best possible label here is "promising"; only future data can validate.
* **Screen**: development improvement > 0 with bootstrap p < 0.05 and at least 100 kept trades.
  **Confirm**: validation improvement > 0 and significant after Holm over all screened
  candidates. Then nearby settings (same sign), costs x1.5 / x2, yearly results, test period.
* **Part 4** explains 2019 with the same features and checks each explanation on other years.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..features.atr import atr as atr_fn
from ..features.sessions import ny_minutes
from ..live.sizing import lot_size
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .classics import LB_RANGE, lb_day_groups, london_breakout
from .events import regime_per_bar
from .london_oos import VERSIONS
from .metrics import _by, core_metrics, full_report, report_markdown
from .model import regime_kwargs
from .regime4 import regime4_series
from .runner import _table

log = get_logger(__name__)

ORIGINAL = dict(VERSIONS["trend version (EXP-012)"])
EXPECTED_EXP015 = {"trades": 914, "net_r": 81.4}           # EXP-015 result, 2015-01-01 to 2020-01-01
SPLITS = (("development", "2015-01-01", "2020-01-01"), ("validation", "2020-01-01", "2023-01-01"),
          ("test", "2023-01-01", "2100-01-01"))
CUTS = (1 / 3, 2 / 3)
CUTS_NEAR = ((0.25, 0.75), (0.40, 0.60))
BUCKETS3 = ("low", "middle", "high")
EXITS = {   # Part 3, fixed in the brief. F: from +2R the stop trails 1R behind the best price.
    "A 3R, BE 1R (original)": {"target_r": 3.0, "be_r": 1.0},
    "B 3R, no BE": {"target_r": 3.0, "be_r": None},
    "C 2R, BE 1R": {"target_r": 2.0, "be_r": 1.0},
    "D 2R, no BE": {"target_r": 2.0, "be_r": None},
    "E 2.5R, BE 1R": {"target_r": 2.5, "be_r": 1.0},
    "F 3R, trail after 2R": {"target_r": 3.0, "be_r": None, "trail_after_r": 2.0, "trail_r": 1.0},
}
EXITS_NEAR = {
    "B 3R, no BE": [{"target_r": 2.5, "be_r": None}, {"target_r": 3.5, "be_r": None}],
    "C 2R, BE 1R": [{"target_r": 1.5, "be_r": 1.0}, {"target_r": 2.0, "be_r": 0.75}],
    "D 2R, no BE": [{"target_r": 1.5, "be_r": None}, {"target_r": 2.5, "be_r": None}],
    "E 2.5R, BE 1R": [{"target_r": 2.5, "be_r": 0.75}, {"target_r": 2.5, "be_r": 1.25}],
    "F 3R, trail after 2R": [{"target_r": 3.0, "be_r": None, "trail_after_r": 1.5, "trail_r": 1.0},
                             {"target_r": 3.0, "be_r": None, "trail_after_r": 2.5, "trail_r": 1.0}],
}
RETEST_BARS = 12            # GUESS: a retest must come within 1 hour of the confirming close
FALSE_BREAK_BARS = 6        # descriptive: closed back inside the range within 30 min of entry


# --------------------------------------------------------------------------- simulation

def walk_exit(i0: int, end_i: int, side: int, entry: float, stop: float, target: float, o, h, l, c, sp: float,
              be_r: float | None = None, trail_after_r: float | None = None, trail_r: float = 1.0
              ) -> tuple[int, float, str]:
    """Like :func:`classics._walk` (fill_bar_target=False) plus an optional trailing stop and the
    exit reason: stop / breakeven / trail / target / time. Stop is checked before target."""
    risk = abs(entry - stop)
    state, best = "stop", entry
    for i in range(i0, end_i + 1):
        if side == 1:
            if l[i] <= stop:
                return i, (min(o[i], stop) if i > i0 else stop), state
            if i > i0 and h[i] >= target:
                return i, max(o[i], target), "target"
            if be_r is not None and h[i] - entry >= be_r * risk and stop < entry:
                stop, state = entry, "breakeven"
            best = max(best, h[i])
            if trail_after_r is not None and best - entry >= trail_after_r * risk and best - trail_r * risk > stop:
                stop, state = best - trail_r * risk, "trail"
        else:
            if h[i] + sp >= stop:
                return i, (max(o[i] + sp, stop) if i > i0 else stop), state
            if i > i0 and l[i] + sp <= target:
                return i, min(o[i] + sp, target), "target"
            if be_r is not None and entry - (l[i] + sp) >= be_r * risk and stop > entry:
                stop, state = entry, "breakeven"
            best = min(best, l[i] + sp)
            if trail_after_r is not None and entry - best >= trail_after_r * risk and best + trail_r * risk < stop:
                stop, state = best + trail_r * risk, "trail"
    return end_i, c[end_i] + (sp if side == -1 else 0.0), "time"


def lb_trades(m5: pd.DataFrame, sp: float, p: dict, regime: np.ndarray | None, groups: list[np.ndarray],
              entry: str = "stop", exits: dict | None = None, retest_bars: int = RETEST_BARS) -> pd.DataFrame:
    """London breakout with a choice of entry (``stop`` = the original stop order, ``close`` = next
    open after an M5 close beyond the level, ``retest`` = limit back at the level within
    ``retest_bars`` after that close) and exits (``target_r``, ``be_r``, ``trail_after_r``,
    ``trail_r``). Range, buffer, stop, trend filter and close time come from ``p``. One trade a day
    at most; the first breakout (touch or close) decides the day, as in the original."""
    ex = {"target_r": p["target_r"], "be_r": p["be_r"], "trail_after_r": None, "trail_r": 1.0, **(exits or {})}
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a5 = atr_fn(m5, 14).to_numpy(float)
    mins = ny_minutes(t)
    in_range = (mins >= LB_RANGE[0]) | (mins < LB_RANGE[1])
    active = (mins >= LB_RANGE[1]) & (mins < p["close_min"])
    rows = []
    for d, idx in enumerate(groups):
        r_idx, a_idx = idx[in_range[idx]], idx[active[idx]]
        if len(r_idx) < 50 or not len(a_idx):
            continue
        hi, lo = h[r_idx].max(), l[r_idx].min()
        px = c[r_idx[-1]]
        if p["range_filter"] and not 0.003 <= (hi - lo) / px <= 0.016:
            continue
        allowed = {1, -1}
        if p["trend"] == "with" and regime is not None:
            rg = regime[r_idx[-1]]
            allowed = {1} if rg == "bullish" else {-1} if rg == "bearish" else {1, -1}
        buy_lvl, sell_lvl = hi + p["buffer"] * px, lo - p["buffer"] * px
        mid = (hi + lo) / 2
        end_i = a_idx[-1]
        trig = side = None
        for i in a_idx:
            if entry == "stop":
                up, dn = h[i] + sp >= buy_lvl, l[i] <= sell_lvl
            else:
                up, dn = c[i] > buy_lvl, c[i] < sell_lvl
            if not (up or dn):
                continue
            if up and dn:
                up = abs(buy_lvl - o[i] - sp) <= abs(o[i] - sell_lvl)
            s = 1 if up else -1
            if s in allowed:
                trig, side = i, s
            break
        if trig is None:
            continue
        if entry == "stop":
            i0 = trig
            px_in = max(buy_lvl, o[i0] + sp) if side == 1 else min(sell_lvl, o[i0])
        elif entry == "close":
            i0 = trig + 1
            if i0 > end_i:
                continue
            px_in = o[i0] + (sp if side == 1 else 0.0)
        else:
            i0 = None
            for k in range(trig + 1, min(trig + retest_bars, end_i) + 1):
                if side == 1 and l[k] + sp <= buy_lvl:
                    i0, px_in = k, min(o[k] + sp, buy_lvl)
                    break
                if side == -1 and h[k] >= sell_lvl:
                    i0, px_in = k, max(o[k], sell_lvl)
                    break
            if i0 is None:
                continue
        cap = p.get("stop_cap", 0.006)       # None = natural stop, never tightened (EXP-020)
        if side == 1:
            nat = lo if p["stop"] == "range" else mid
            stop = nat if cap is None else max(nat, px_in * (1 - cap))
        else:
            nat = (hi if p["stop"] == "range" else mid) + sp
            stop = nat if cap is None else min(nat, px_in * (1 + cap))
        risk = abs(px_in - stop)
        if risk <= 0 or (px_in - stop) * side <= 0:
            continue
        tgt = px_in + side * ex["target_r"] * risk
        j, px_out, why = walk_exit(i0, end_i, side, px_in, stop, tgt, o, h, l, c, sp, ex["be_r"],
                                   ex["trail_after_r"], ex["trail_r"])
        rng_ = h[trig] - l[trig]
        lvl = buy_lvl if side == 1 else sell_lvl
        back = [k for k in range(i0, min(i0 + FALSE_BREAK_BARS, len(c))) if (c[k] < hi if side == 1 else c[k] > lo)]
        rows.append({"day": d, "entry_i": i0, "entry_time": t[i0], "exit_time": t[j],
                     "stop_capped": bool(stop != nat), "natural_stop": nat,
                     "side": "BUY" if side == 1 else "SELL", "entry": px_in, "exit": px_out, "stop": stop,
                     "risk": risk, "r": side * (px_out - px_in) / risk, "exit_reason": why,
                     "range_hi": hi, "range_lo": lo, "trigger_after_fill": entry == "stop",
                     "body_frac": abs(c[trig] - o[trig]) / rng_ if rng_ > 0 else np.nan,
                     "close_pos": ((c[trig] - l[trig]) / rng_ if side == 1 else (h[trig] - c[trig]) / rng_)
                     if rng_ > 0 else np.nan,
                     "break_dist_atr": side * (c[trig] - lvl) / a5[trig] if a5[trig] > 0 else np.nan,
                     "rejection_wick": ((h[trig] - max(o[trig], c[trig])) if side == 1 else
                                        (min(o[trig], c[trig]) - l[trig])) / rng_ if rng_ > 0 else np.nan,
                     "false_break": bool(back),
                     "entry_hour_ny": int(mins[i0] // 60)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- day features

def _past_pct(x: np.ndarray, window: int = 250, min_n: int = 60) -> np.ndarray:
    """Share of the previous ``window`` values below each value (past only); NaN early on."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        prev = x[max(0, i - window):i]
        prev = prev[np.isfinite(prev)]
        if len(prev) >= min_n and np.isfinite(x[i]):
            out[i] = float((prev < x[i]).mean())
    return out


def day_features(m5: pd.DataFrame, groups: list[np.ndarray], calendar: pd.DataFrame | None = None,
                 close_min: int = 16 * 60) -> pd.DataFrame:
    """One row per trading day (17:00-17:00 NY), everything known by 01:00 NY that day:
    Asian range / daily ATR(14) and its past percentile (H1), daily ATR as % of price and its past
    percentile (H4), previous day high / low / close (H3), news flags (H5)."""
    t = pd.DatetimeIndex(m5["time"])
    h, l, c = (m5[k].to_numpy(float) for k in ("high", "low", "close"))
    mins = ny_minutes(t)
    in_range = (mins >= LB_RANGE[0]) | (mins < LB_RANGE[1])
    rows = []
    for d, idx in enumerate(groups):
        r_idx = idx[in_range[idx]]
        rows.append({"day": d, "start": t[idx[0]], "d_high": h[idx].max(), "d_low": l[idx].min(),
                     "d_close": c[idx[-1]],
                     "range_w": (h[r_idx].max() - l[r_idx].min()) if len(r_idx) else np.nan,
                     "px": c[r_idx[-1]] if len(r_idx) else np.nan})
    df = pd.DataFrame(rows)
    prev_c = df["d_close"].shift(1)
    tr = np.maximum(df["d_high"] - df["d_low"], np.maximum((df["d_high"] - prev_c).abs(), (df["d_low"] - prev_c).abs()))
    df["datr"] = tr.rolling(14, min_periods=14).mean().shift(1)          # completed days only
    df["pdh"], df["pdl"], df["pdc"] = df["d_high"].shift(1), df["d_low"].shift(1), prev_c
    df["range_atr"] = df["range_w"] / df["datr"]
    df["range_pct"] = _past_pct(df["range_atr"].to_numpy(float))
    df["vol_pct_price"] = df["datr"] / df["px"]
    df["vol_pct"] = _past_pct(df["vol_pct_price"].to_numpy(float))
    df["us_news"] = df["uk_news"] = np.nan
    if calendar is not None and len(calendar):
        ev = calendar[calendar["impact"].astype(str).str.lower() == "high"]
        ev_t = pd.DatetimeIndex(ev["time"])
        ev_m = ny_minutes(ev_t)
        in_win = (ev_m >= LB_RANGE[1]) & (ev_m < close_min)
        starts = pd.DatetimeIndex(df["start"])
        k = starts.searchsorted(ev_t, side="right") - 1
        all_t = pd.DatetimeIndex(calendar["time"])          # coverage from every row, any impact
        k0, k1 = (starts.searchsorted(x, side="right") - 1 for x in (all_t.min(), all_t.max()))
        covered = (np.arange(len(df)) >= max(k0, 0)) & (np.arange(len(df)) <= k1)
        us, uk = np.zeros(len(df), bool), np.zeros(len(df), bool)
        cur = ev["currency"].astype(str).str.upper().to_numpy()
        for kk, w, cu in zip(k, in_win, cur):
            if kk >= 0 and w:
                if cu == "USD":
                    us[kk] = True
                elif cu == "GBP":
                    uk[kk] = True
        df["us_news"] = np.where(covered, us.astype(float), np.nan)
        df["uk_news"] = np.where(covered, uk.astype(float), np.nan)
    return df


def _bucket(x: np.ndarray, cuts=CUTS, labels=BUCKETS3) -> np.ndarray:
    out = np.full(len(x), "unknown", dtype=object)
    f = np.isfinite(x)
    out[f] = np.where(x[f] < cuts[0], labels[0], np.where(x[f] < cuts[1], labels[1], labels[2]))
    return out


# --------------------------------------------------------------------------- statistics

def block_boot(x: np.ndarray, reps: int = 2000, block: int = 10, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap resamples of ``x`` (rows = resamples)."""
    n = len(x)
    rng = np.random.default_rng(seed)
    b = min(block, n)
    starts = rng.integers(0, n - b + 1, size=(reps, math.ceil(n / b)))
    idx = (starts[..., None] + np.arange(b)).reshape(reps, -1)[:, :n]
    return x[idx]


def boot_stats(r: np.ndarray, reps: int = 2000, seed: int = 0) -> dict:
    """95% block-bootstrap intervals for expectancy and profit factor, centred one-sided p (mean > 0)."""
    if len(r) < 20:
        return {"exp_ci": "", "pf_ci": "", "p_boot": np.nan}
    s = block_boot(r, reps, seed=seed)
    m = s.mean(1)
    pos, neg = np.where(s > 0, s, 0).sum(1), -np.where(s < 0, s, 0).sum(1)
    pf = np.where(neg > 0, pos / np.where(neg > 0, neg, 1), np.nan)
    mu = r.mean()
    e_lo, e_hi = np.percentile(m, [2.5, 97.5])
    p_lo, p_hi = np.nanpercentile(pf, [2.5, 97.5])
    return {"exp_ci": f"{e_lo:+.3f} to {e_hi:+.3f}", "pf_ci": f"{p_lo:.2f} to {p_hi:.2f}",
            "p_boot": round(float(((m - mu) >= mu).mean()), 4)}


def newey_west_t(r: np.ndarray) -> tuple[float, float]:
    """HAC t-statistic of the mean (Bartlett kernel) and its one-sided normal p."""
    n = len(r)
    if n < 10:
        return np.nan, np.nan
    e = r - r.mean()
    lag = int(4 * (n / 100) ** (2 / 9))
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    t = r.mean() / math.sqrt(v / n) if v > 0 else np.nan
    return round(t, 2), round(0.5 * math.erfc(t / math.sqrt(2)), 4) if np.isfinite(t) else np.nan


def diff_boot(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 1) -> tuple[float, float]:
    """Mean(a) - mean(b), and the centred one-sided bootstrap p that it is > 0 (independent groups)."""
    if len(a) < 20 or len(b) < 20:
        return (round(float(a.mean() - b.mean()), 3) if len(a) and len(b) else np.nan), np.nan
    d = a.mean() - b.mean()
    bd = block_boot(a, reps, seed=seed).mean(1) - block_boot(b, reps, seed=seed + 7).mean(1)
    return round(float(d), 3), round(float(((bd - d) >= d).mean()), 4)


def paired_boot(x: np.ndarray, reps: int = 2000, seed: int = 2) -> tuple[float, float]:
    """Mean of paired day-level differences and its centred one-sided bootstrap p (> 0)."""
    if len(x) < 20:
        return (round(float(x.mean()), 3) if len(x) else np.nan), np.nan
    m = block_boot(x, reps, seed=seed).mean(1)
    mu = x.mean()
    return round(float(mu), 3), round(float(((m - mu) >= mu).mean()), 4)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise ``alpha``."""
    items = sorted(((p, k) for k, p in pvals.items() if np.isfinite(p)))
    out, m = {k: False for k in pvals}, len(items)
    for i, (p, k) in enumerate(items):
        if p > alpha / (m - i):
            break
        out[k] = True
    return out


# --------------------------------------------------------------------------- costs

def with_costs(tr: pd.DataFrame, bar_sp: np.ndarray, base_sp: float, costs, point: float, k: float = 1.0
               ) -> pd.DataFrame:
    """Charge ``k`` x (historical bar spread, at least ``base_sp``) + ``k`` x slippage on two fills +
    commission. The simulation already paid ``base_sp`` once."""
    t = tr.copy()
    if t.empty:
        return t
    spread = np.maximum(bar_sp[t["entry_i"].to_numpy(int)], base_sp)
    extra = k * spread - base_sp + k * (2 * costs.slippage_points * point + costs.commission_per_lot / costs.contract_size)
    t["r_gross"] = t["r"]
    t["bar_spread"] = spread
    t["cost_r"] = extra / t["risk"]
    t["r"] = t["r"] - t["cost_r"]
    return t


# --------------------------------------------------------------------------- report helpers

def _split_of(times: pd.Series) -> np.ndarray:
    tt = pd.DatetimeIndex(times)
    out = np.full(len(tt), "", dtype=object)
    for name, a, b in _splits():
        out[(tt >= pd.Timestamp(a, tz="UTC")) & (tt < pd.Timestamp(b, tz="UTC"))] = name
    return out


def _splits() -> tuple:
    return SPLITS


def _row(r: np.ndarray, risk_pct: float) -> dict:
    m = core_metrics(r, risk_pct)
    if not m.get("trades"):
        return {"trades": 0}
    b = boot_stats(r)
    return {"trades": m["trades"], "win%": m["win_rate_pct"], "PF": m["profit_factor"], "exp_R": m["expectancy_r"],
            "exp_CI95": b["exp_ci"], "DD%": m["max_drawdown_pct"], "avg_win": m["avg_win_r"],
            "avg_loss": m["avg_loss_r"], "max_losses": m["max_consecutive_losses"], "p_boot": b["p_boot"]}


def _yearly(tr: pd.DataFrame, risk_pct: float) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    return _by(tr, pd.Series(pd.DatetimeIndex(tr["entry_time"]).year.astype(str)), risk_pct)


def _detail(name: str, tr: pd.DataFrame, risk_pct: float, stress: dict[str, pd.DataFrame]) -> str:
    """Every Part 6 item for one experiment (whole period, with split, cost and yearly tables)."""
    lines = [f"# {name}\n"]
    rows = []
    for sp_name, _, _ in _splits():
        x = tr[tr["split"] == sp_name]["r"].to_numpy()
        rows.append({"part": sp_name, **_row(x, risk_pct)})
    rows.append({"part": "all", **_row(tr["r"].to_numpy(), risk_pct)})
    lines += ["## By data part (with bootstrap 95% interval)\n", _table(pd.DataFrame(rows), list(rows[0].keys()) if rows[0].get("trades") else ["part", "trades"])]
    cr = [{"costs": k, **{s: round(float(v[v["split"] == s]["r"].mean()), 3) if (v["split"] == s).any() else np.nan
                          for s, _, _ in _splits()}} for k, v in stress.items()]
    lines += ["\n## Expectancy (R) by cost level\n", _table(pd.DataFrame(cr), ["costs"] + [s for s, _, _ in _splits()])]
    if "exit_reason" in tr:
        er = tr.groupby("exit_reason")["r"].agg(["count", "mean"]).round(3).reset_index()
        lines += ["\n## Exit reasons (time = closed at the session cut-off)\n", _table(er, list(er.columns))]
    lines += ["\n## Yearly\n", _table(_yearly(tr, risk_pct), ["group", "trades", "win_rate_pct", "profit_factor",
                                                             "expectancy_r", "net_r", "net_profit_pct", "max_drawdown_pct"])]
    lines.append(report_markdown("Whole period: long / short, session, regime, monthly", full_report(tr, risk_pct)))
    return "\n".join(lines)


HISTORY = """| Step | Data used | What was chosen | Variants looked at |
|---|---|---|---|
| EXP-002 to 005 | 2015-2026 | pattern mining (not this strategy) | thousands, 0 passed |
| EXP-011 | 2020-2026 | London breakout picked from 9 public strategies | 9 |
| EXP-012 | 2020-2026 | trend version picked from a grid on 2020-2024 | 360 |
| EXP-015 | 2015-2019 | one-shot test, versions fixed in advance | 2 (Bonferroni) |
| EXP-016 (this) | 2015-2026 | one change at a time, screened on 2015-2019 | listed below |

Code check: `find_classics.bat`, `polish_london.bat` and `london_50pct.bat` all pass `--start 2020-01-01`;
no London-breakout run touched 2015-2019 before EXP-015. After EXP-015 and this study, **no historical
period is untouched**; data from 2026-11-27 onward is reserved for forward (paper) validation."""


# --------------------------------------------------------------------------- main

def run_lb_phase2(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                  start: pd.Timestamp, end: pd.Timestamp | None = None, calendar_file: str | Path | None = None,
                  source_note: str = "", expected: dict | None = EXPECTED_EXP015,
                  repro_end: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC")) -> dict:
    """Run EXP-016 once and write ``report.md`` plus ``detail/*.md`` and trade CSVs to ``out_dir``."""
    sc = cfg.strategy
    costs, risk_pct = sc.research.costs, sc.backtest.risk_pct
    out = Path(out_dir)
    (out / "detail").mkdir(parents=True, exist_ok=True)
    base_sp = costs.min_spread_points * point
    md = ["# EXP-016: London Asian-range breakout, Phase 2 research\n",
          f"- **data**: {source_note or dataset}",
          f"- **costs**: historical M5 bar spread (at least ${base_sp:.2f}), slippage {costs.slippage_points:g} "
          f"points per fill, commission {costs.commission_per_lot:g}/lot; stress x1.5 and x2",
          f"- **risk**: {risk_pct:g}% of equity per trade, compounded",
          "- **split**: development 2015-2019, validation 2020-2022, test 2023 onward (none untouched, see Part 1)",
          "- **rules fixed before the run**: see the module docstring of `research/lb_phase2.py`\n"]

    # ---- Part 1a: reproduce EXP-015 exactly (same window, same code path)
    rep_end = repro_end
    m5r = store.read_bars(dataset, Timeframe.M5, start, min(rep_end, end) if end is not None else rep_end)
    h4r = store.read_bars(dataset, Timeframe.H4, start - pd.Timedelta(days=60), rep_end)
    if m5r.empty or h4r.empty:
        raise ValueError(f"No stored {dataset} M5/H4 data from {start:%Y-%m-%d}")
    tr_ = pd.DatetimeIndex(m5r["time"])
    reg_r = regime_per_bar(tr_, h4r, trend_regime_series(h4r, **regime_kwargs(sc)))
    g_r = lb_day_groups(m5r, sc.sessions)
    orig = pd.DataFrame(london_breakout(m5r, base_sp, sc.sessions, ORIGINAL, reg_r, g_r))
    mine = lb_trades(m5r, base_sp, ORIGINAL, reg_r, g_r)
    bar_sp_r = m5r["spread"].to_numpy(float) * point if "spread" in m5r else np.full(len(m5r), base_sp)
    same_code = len(orig) == len(mine) and len(mine) > 0 and np.allclose(orig["r"].to_numpy(), mine["r"].to_numpy())
    rep_tr = with_costs(mine, bar_sp_r, base_sp, costs, point)
    got = {"trades": int(len(rep_tr)), "net_r": round(float(rep_tr["r"].sum()), 1) if len(rep_tr) else 0.0}
    reproduced = same_code and (expected is None or (got["trades"] == expected["trades"]
                                                     and abs(got["net_r"] - expected["net_r"]) <= 0.15))
    md += ["## Part 1. Verify the existing result\n",
           f"**Reproduction of EXP-015 (2015-2019)**: {got['trades']} trades, net {got['net_r']}R "
           f"(EXP-015: {expected['trades'] if expected else '-'} trades, {expected['net_r'] if expected else '-'}R). "
           f"New simulator matches the original code trade for trade: {'yes' if same_code else 'NO'}. "
           f"**{'REPRODUCED' if reproduced else 'NOT REPRODUCED: stopped here, nothing else was run'}**\n"]
    if not reproduced:
        (out / "report.md").write_text("\n".join(md), encoding="utf-8")
        return {"reproduced": False, "got": got, "report": str(out / "report.md")}

    # ---- load the whole period
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, Timeframe.H4, start - pd.Timedelta(days=60), end)
    quality = validate_bars(m5, Timeframe.M5, cfg.data.quality, dataset=f"{dataset} M5")
    t = pd.DatetimeIndex(m5["time"])
    reg2 = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kwargs(sc)))
    reg4 = regime_per_bar(t, h4, regime4_series(h4, sc.research.regime4))
    groups = lb_day_groups(m5, sc.sessions)
    bar_sp = m5["spread"].to_numpy(float) * point if "spread" in m5 else np.full(len(m5), base_sp)
    cal = None
    if calendar_file and Path(calendar_file).exists():
        cal = pd.read_csv(calendar_file)
        cal["time"] = pd.to_datetime(cal["time"], utc=True)
    days = day_features(m5, groups, cal, ORIGINAL["close_min"])
    md.append(f"- **period (UTC)**: {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}; M5 quality issues: {quality.counts() or 'none'}\n")

    def prep(tr: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
        if tr.empty:
            return tr.assign(split=pd.Series(dtype=object))
        x = with_costs(tr, bar_sp, base_sp, costs, point, k)
        x["split"] = _split_of(x["entry_time"])
        x["regime"] = reg4[x["entry_i"].to_numpy(int)]
        x["trend"] = reg2[x["entry_i"].to_numpy(int)]
        return x.merge(days, on="day", how="left")

    raw = {"original": lb_trades(m5, base_sp, ORIGINAL, reg2, groups)}
    base = prep(raw["original"])
    stress = lambda tr: {f"x{k:g}": prep(tr, k) for k in (1.0, 1.5, 2.0)}  # noqa: E731
    base_stress = stress(raw["original"])

    # ---- Part 1b: costs, sizing, significance
    sp_year = base.assign(year=pd.DatetimeIndex(base["entry_time"]).year).groupby("year").agg(
        trades=("r", "size"), median_spread=("bar_spread", "median"),
        p95_spread=("bar_spread", lambda s: float(np.percentile(s, 95))),
        above_008_pct=("bar_spread", lambda s: round(100 * float((s > base_sp + 1e-9).mean()), 1)),
        cost_r=("cost_r", "mean")).round(3).reset_index()
    sizes = [lot_size(10_000, risk_pct, e, s, costs.contract_size, 0.01, 0.01)
             for e, s in zip(base["entry"], base["stop"])]
    real_pct = np.array([100 * z.risk_money / 10_000 for z in sizes])
    md += ["**Spread actually charged (entry candle's MT5 spread; MT5 keeps one value per candle, which can be "
           "below the spread at the fill, hence the x1.5 / x2 stress)**\n", _table(sp_year, list(sp_year.columns)),
           f"\n**Contract and sizing**: contract size {costs.contract_size:g} oz/lot, point {point:g}. "
           f"Stop distance median ${base['risk'].median():.2f} (5-95%: ${base['risk'].quantile(.05):.2f}-"
           f"${base['risk'].quantile(.95):.2f}). On a $10,000 account at {risk_pct:g}% with 0.01-lot steps the "
           f"real risk is {real_pct.mean():.3f}% on average (min {real_pct.min():.3f}%, max {real_pct.max():.3f}%); "
           f"results in R assume exact {risk_pct:g}%.\n"]
    sig = []
    for sp_name, _, _ in _splits() + (("all", "", ""),):
        x = (base if sp_name == "all" else base[base["split"] == sp_name])["r"].to_numpy()
        if len(x) < 20:
            continue
        nw_t, nw_p = newey_west_t(x)
        b = boot_stats(x)
        sig.append({"part": sp_name, "trades": len(x), "exp_R": round(float(x.mean()), 3), "exp_CI95": b["exp_ci"],
                    "PF_CI95": b["pf_ci"], "p_boot": b["p_boot"], "NW_t": nw_t, "NW_p": nw_p,
                    "plain_p": core_metrics(x, risk_pct)["p_value"]})
    md += ["**Significance, dependence-aware** (moving-block bootstrap, 10-trade blocks, 2,000 resamples; "
           "Newey-West t). Only development (2015-2019) was out of sample for this version; validation and "
           "test are where it was chosen.\n", _table(pd.DataFrame(sig), list(sig[0].keys()) if sig else ["part"]),
           "\n**Test history and multiple-testing risk**\n", HISTORY, "\n"]

    # ---- candidates (one change each)
    cands: dict[str, dict] = {}

    def add_filter(name, hyp, mask_fn, near_fns, base_tr=base, base_name="original", raw_tr=None, note=""):
        cands[name] = {"hyp": hyp, "kind": "filter", "mask": mask_fn, "near": near_fns, "base": base_tr,
                       "base_name": base_name, "raw": raw_tr if raw_tr is not None else raw["original"], "note": note}

    def pct_masks(col, label):
        f = lambda cuts: (lambda d: _bucket(d[col].to_numpy(float), cuts) == label)  # noqa: E731
        return f(CUTS), [f(c) for c in CUTS_NEAR]

    for lab in BUCKETS3:
        m, n = pct_masks("range_pct", lab)
        add_filter(f"H1 range width {lab} third", "H1", m, n)
    for lab in BUCKETS3:
        m, n = pct_masks("vol_pct", lab)
        add_filter(f"H4 volatility {lab} third", "H4", m, n)

    # H3: room to the opposing previous-day level, in R
    def room(d):
        buy = d["side"].to_numpy() == "BUY"
        lvl = np.where(buy, d["pdh"], d["pdl"])
        rr = np.where(buy, lvl - d["entry"], d["entry"] - lvl) / d["risk"]
        return np.where(np.isfinite(rr) & (rr <= 0), np.inf, rr)       # level already behind the entry = clear
    def room_mask(a, b_):
        def f(d):
            x = room(d)
            ok = ~np.isnan(x)
            if a is not None:
                ok &= x >= a
            if b_ is not None:
                ok &= x < b_
            return ok
        return f
    for lab, ab, near in (("under 1R", (None, 1.0), [(None, 0.75), (None, 1.25)]),
                          ("1-3R", (1.0, 3.0), [(0.75, 2.5), (1.25, 3.5)]),
                          ("3R+ or clear", (3.0, None), [(2.5, None), (3.5, None)])):
        add_filter(f"H3 room to prev-day level {lab}", "H3", room_mask(*ab), [room_mask(*z) for z in near])
    for lab, sgn in (("with prev close", 1), ("against prev close", -1)):
        add_filter(f"H3 entry {lab}", "H3",
                   (lambda s: lambda d: (np.where(d["side"] == "BUY", 1, -1) * (d["entry"] - d["pdc"]) * s) > 0)(sgn), [])
    if cal is not None and days["us_news"].notna().any():
        add_filter("H5 US high-impact news day", "H5", lambda d: d["us_news"] == 1, [])
        add_filter("H5 UK high-impact news day", "H5", lambda d: d["uk_news"] == 1, [])
        add_filter("H5 no US / UK high-impact news", "H5", lambda d: (d["us_news"] == 0) & (d["uk_news"] == 0), [])
    news_note = ("tested with " + str(calendar_file)) if cal is not None else \
        "**not tested: no historical economic calendar file was given** (`--calendar`)"

    # H6 entries and exits: whole strategies, compared day by day
    raw["H6 close-confirmed entry"] = lb_trades(m5, base_sp, ORIGINAL, reg2, groups, entry="close")
    raw["H6 retest entry"] = lb_trades(m5, base_sp, ORIGINAL, reg2, groups, entry="retest")
    for name in ("H6 close-confirmed entry", "H6 retest entry"):
        near = ([lambda: lb_trades(m5, base_sp, ORIGINAL, reg2, groups, entry="retest", retest_bars=6),
                 lambda: lb_trades(m5, base_sp, ORIGINAL, reg2, groups, entry="retest", retest_bars=24)]
                if "retest" in name else [])
        cands[name] = {"hyp": "H6", "kind": "strategy", "raw": raw[name], "near": near}
    for name, ex in EXITS.items():
        if name.startswith("A"):
            continue
        raw[name] = lb_trades(m5, base_sp, ORIGINAL, reg2, groups, exits=ex)
        near = [(lambda e: lambda: lb_trades(m5, base_sp, ORIGINAL, reg2, groups, exits=e))(e)
                for e in EXITS_NEAR.get(name, [])]
        cands[name] = {"hyp": "exit", "kind": "strategy", "raw": raw[name], "near": near}

    # H2 candle quality: filters on close-confirmed entries (dev-period terciles of each feature)
    close_tr = prep(raw["H6 close-confirmed entry"])
    dev_close = close_tr[close_tr["split"] == "development"]
    for feat in ("body_frac", "close_pos", "break_dist_atr", "rejection_wick"):
        qs = {c_: tuple(np.nanquantile(dev_close[feat], c_)) if len(dev_close) else (np.nan, np.nan)
              for c_ in (CUTS,) + CUTS_NEAR}
        for lab in BUCKETS3:
            f = (lambda ft, lb_: lambda q: lambda d: _bucket(d[ft].to_numpy(float), q) == lb_)(feat, lab)
            add_filter(f"H2 {feat} {lab} third", "H2", f(qs[CUTS]), [f(qs[c_]) for c_ in CUTS_NEAR],
                       base_tr=close_tr, base_name="close-confirmed entry", raw_tr=raw["H6 close-confirmed entry"])

    # ---- evaluate
    def day_r(tr: pd.DataFrame, split: str | None) -> pd.Series:
        x = tr if split is None else tr[tr["split"] == split]
        return x.groupby("day")["r"].sum()

    def paired(tr_b: pd.DataFrame, tr_a: pd.DataFrame, split: str | None) -> tuple[float, float]:
        a, b = day_r(tr_a, split), day_r(tr_b, split)
        idx = a.index.union(b.index)
        return paired_boot((b.reindex(idx, fill_value=0) - a.reindex(idx, fill_value=0)).to_numpy())

    results, kept_tr = {}, {}
    for name, cd in cands.items():
        res = {"hypothesis": cd["hyp"], "change": name}
        if cd["kind"] == "filter":
            bt = cd["base"]
            mk = np.asarray(cd["mask"](bt), bool)
            kept, rest = bt[mk], bt[~mk]
            kept_tr[name] = kept
            for s, _, _ in _splits():
                k_, r_ = kept[kept["split"] == s]["r"].to_numpy(), rest[rest["split"] == s]["r"].to_numpy()
                d, p = diff_boot(k_, r_)
                res |= {f"{s}_n": len(k_), f"{s}_exp": round(float(k_.mean()), 3) if len(k_) else np.nan,
                        f"{s}_diff": d, f"{s}_p": p}
            res["compared_with"] = f"the other {cd['base_name']} trades"
            near_same = []
            for nf in cd["near"]:
                mm = np.asarray(nf(bt), bool)
                dv = bt[bt["split"].isin(["development", "validation"])]
                mmv = mm[bt["split"].isin(["development", "validation"]).to_numpy()]
                a_, b_ = dv[mmv]["r"], dv[~mmv]["r"]
                near_same.append(bool(len(a_) and len(b_) and a_.mean() > b_.mean()))
            res["near_same_sign"] = (all(near_same) if near_same else "n/a")
            sx = {k: v[np.asarray(cd["mask"](v), bool)] for k, v in
                  (stress(cd["raw"]) if cd["base_name"] != "original" else base_stress).items()}
        else:
            tr = prep(cd["raw"])
            kept_tr[name] = tr
            for s, _, _ in _splits():
                x = tr[tr["split"] == s]["r"].to_numpy()
                d, p = paired(tr, base, s)
                res |= {f"{s}_n": len(x), f"{s}_exp": round(float(x.mean()), 3) if len(x) else np.nan,
                        f"{s}_diff": d, f"{s}_p": p}
            res["compared_with"] = "the original, R per day"
            near_same = []
            for nf in cd["near"]:
                nt = prep(nf())
                dv = lambda z: z[z["split"].isin(["development", "validation"])]  # noqa: E731
                near_same.append(paired(dv(nt), dv(base), None)[0] > 0)
            res["near_same_sign"] = (all(near_same) if near_same else "n/a")
            sx = stress(cd["raw"])
        res["val_exp_x1.5"] = round(float(sx["x1.5"][sx["x1.5"]["split"] == "validation"]["r"].mean()), 3) \
            if (sx["x1.5"]["split"] == "validation").any() else np.nan
        res["val_exp_x2"] = round(float(sx["x2"][sx["x2"]["split"] == "validation"]["r"].mean()), 3) \
            if (sx["x2"]["split"] == "validation").any() else np.nan
        yv = kept_tr[name]
        yrs = sorted(set(pd.DatetimeIndex(yv["entry_time"]).year) & set(range(2020, 2100)))
        better = 0
        for y in yrs:
            in_y = lambda z: z[pd.DatetimeIndex(z["entry_time"]).year == y]  # noqa: E731
            if cd["kind"] == "filter":
                bt = cd["base"]
                mk = np.asarray(cd["mask"](bt), bool)
                a_, b_ = in_y(bt[mk])["r"], in_y(bt[~mk])["r"]
                better += bool(len(a_) and len(b_) and a_.mean() > b_.mean())
            else:
                better += day_r(in_y(yv), None).sum() > day_r(in_y(base), None).sum()
        res["years_better_2020+"] = f"{better}/{len(yrs)}"
        results[name] = res
        (out / "detail" / f"{_slug(name)}.md").write_text(_detail(name, yv, risk_pct, sx), encoding="utf-8")

    # screen on development, confirm on validation (Holm), classify
    screened = {k: v for k, v in results.items() if _f(v["development_diff"]) > 0
                and _f(v["development_p"], 1.0) < 0.05 and v["development_n"] >= 100}
    confirmed = holm({k: (_f(v["validation_p"], 1.0) if _f(v["validation_diff"]) > 0 else 1.0)
                      for k, v in screened.items()})
    for k, v in results.items():
        if k in screened:
            ok = (confirmed.get(k) and v["near_same_sign"] in (True, "n/a") and _f(v["val_exp_x2"]) > 0
                  and _f(v["test_diff"]) > 0)
            v["class"] = ("3 promising, needs forward validation" if ok else
                          "2 interesting, insufficient evidence" if _f(v["validation_diff"]) > 0 else "1 rejected")
        elif _f(v["development_diff"]) > 0 and _f(v["development_p"], 1.0) < 0.20:
            v["class"] = "2 interesting, insufficient evidence"
        else:
            v["class"] = "1 rejected"
        v["screen"] = "pass" if k in screened else "fail"
        v["validation_holm"] = bool(confirmed.get(k, False))

    # ---- report: Parts 2-3, 5, 6, 7
    tab = pd.DataFrame(results.values())
    cols = ["hypothesis", "change", "compared_with", "development_n", "development_exp", "development_diff",
            "development_p", "screen", "validation_n", "validation_exp", "validation_diff", "validation_p",
            "validation_holm", "test_n", "test_exp", "test_diff", "near_same_sign", "val_exp_x1.5", "val_exp_x2",
            "years_better_2020+", "class"]
    orig_rows = [{"part": s, **_row(base[base["split"] == s]["r"].to_numpy(), risk_pct)} for s, _, _ in _splits()]
    (out / "detail" / "original.md").write_text(_detail("Original (unchanged)", base, risk_pct, base_stress),
                                               encoding="utf-8")
    md += ["## The original strategy, unchanged (reference for every comparison)\n",
           _table(pd.DataFrame(orig_rows), list(orig_rows[0].keys())),
           "\n## Parts 2, 3, 5 and 7. One change at a time\n",
           "diff = expectancy (R per trade) of the kept trades minus the other trades for filters; for H6 and exits, "
           "R per trading day minus the original's. p = one-sided block-bootstrap p that diff > 0. "
           f"Screen: development diff > 0, p < 0.05, 100+ trades ({len(screened)} of {len(results)} passed). "
           "Confirm: validation diff > 0 and Holm-significant over the screened set. Promising also needs nearby "
           "settings with the same sign, positive validation expectancy at x2 costs and a positive test diff. "
           "Nothing can be 'validated for paper trading' here because no historical data is untouched.\n",
           f"- H5 news: {news_note}",
           "- H2 is measured on close-confirmed entries (base = close-confirmed entry). With the original stop "
           "orders the breakout candle closes after the fill, so H2 cannot filter the original without look-ahead.",
           f"- Retest entry: a limit back at the breakout level within {RETEST_BARS} candles (GUESS: 1 hour).\n",
           _table(tab, cols), "\nEvery experiment's full metrics (monthly, yearly, long / short, sessions, regimes, "
           "costs, exit reasons): `detail/<name>.md`.\n"]

    # ---- Part 4: 2019
    yr = pd.DatetimeIndex(base["entry_time"]).year
    b4 = base.assign(year=yr, period=np.where(yr == 2019, "2019", np.where(yr < 2019, "2015-2018", "2020+")))
    b4["range_third"] = _bucket(b4["range_pct"].to_numpy(float))
    b4["vol_third"] = _bucket(b4["vol_pct"].to_numpy(float))
    b4["trend_state"] = np.where(b4["trend"].isin(["bullish", "bearish"]), "with H4 trend", "neutral (both ways)")
    b4["body_third"] = _bucket(b4["body_frac"].to_numpy(float), tuple(np.nanquantile(
        b4.loc[b4["year"] < 2019, "body_frac"], CUTS)) if (b4["year"] < 2019).any() else CUTS)
    b4["entry_window"] = np.where(b4["entry_hour_ny"] < 3, "01-03 NY (pre-London)",
                                  np.where(b4["entry_hour_ny"] < 8, "03-08 NY (London)", "08+ NY"))
    b4["false_breakout"] = np.where(b4["false_break"], "closed back inside in 30 min", "held")
    flips = pd.Series(reg2).ne(pd.Series(reg2).shift()).to_numpy()
    day_start = np.array([g[0] for g in groups])
    flips_day = np.add.reduceat(flips.astype(int), day_start) if len(day_start) else np.array([])
    fd = pd.Series(flips_day).rolling(20, min_periods=5).sum().shift(1).to_numpy()
    b4["regime_changes_20d"] = _bucket(fd[b4["day"].to_numpy(int)], (2, 5), ("0-1", "2-4", "5+"))
    factors = ["range_third", "vol_third", "trend_state", "side", "body_third", "entry_window", "false_breakout",
               "regime_changes_20d"]
    if b4["us_news"].notna().any():
        b4["us_news_day"] = np.where(b4["us_news"] == 1, "US news day", "no US news")
        factors.append("us_news_day")
    p4 = []
    for f in factors:
        g = b4.groupby([f, "period"])["r"].agg(["size", "mean", "sum"]).unstack("period")
        for lev in g.index:
            row = {"factor": f, "bucket": lev}
            for per in ("2019", "2015-2018", "2020+"):
                row[f"{per} trades"] = int(g.loc[lev, ("size", per)]) if ("size", per) in g.columns and pd.notna(g.loc[lev, ("size", per)]) else 0
                row[f"{per} exp_R"] = round(float(g.loc[lev, ("mean", per)]), 3) if ("mean", per) in g.columns and pd.notna(g.loc[lev, ("mean", per)]) else np.nan
            row["2019 net_R"] = round(float(g.loc[lev, ("sum", "2019")]), 1) if ("sum", "2019") in g.columns and pd.notna(g.loc[lev, ("sum", "2019")]) else 0.0
            p4.append(row)
    p4 = pd.DataFrame(p4)
    md += ["## Part 4. Why 2019 lost\n",
           f"2019: {int((b4['period'] == '2019').sum())} trades, net {b4.loc[b4['period'] == '2019', 'r'].sum():.1f}R "
           "(costs included). A bucket explains 2019 only if it is also weak in 2015-2018 and 2020+; a bucket that "
           "is bad only in 2019 is not a usable filter (brief: no filter just to remove 2019).\n",
           _table(p4, list(p4.columns))]
    md.append("\n_Research on past data, signals only. The original strategy is unchanged._\n")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    tab.to_csv(out / "candidates.csv", index=False)
    p4.to_csv(out / "part4_2019.csv", index=False)
    base.drop(columns=["start"], errors="ignore").to_csv(out / "trades_original.csv", index=False)
    log.info("lb_phase2_done", extra={"candidates": len(results), "screened": len(screened)})
    summary = {"reproduced": True, "got": got, "screened": sorted(screened),
               "classes": {k: v["class"] for k, v in results.items() if not v["class"].startswith("1")},
               "report": str(out / "report.md")}
    return summary


def _f(x, default: float = -np.inf) -> float:
    """Float value, or ``default`` for None / NaN (so comparisons fail safely)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")
