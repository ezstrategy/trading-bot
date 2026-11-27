"""EXP-019: Monte Carlo account-survival study of the frozen gold London breakout (owner, 2026-11-27).

``python -m quant_engine montecarlo --trades research_data/exp016_trades_original.csv``.
Nothing about the strategy changes: the input is the EXP-016 original trade list (net R after the
historical spread and slippage). Fixed before the run:

* Paths: circular block bootstrap of the real trade sequence (blocks of 10 trades keep losing and
  winning streaks together), 10,000 paths, 500 trades each. The **same** resampled paths are used for
  every risk level (common random numbers), so differences come from sizing only.
* Sizing: lots = equity x risk% / (stop distance x 100 oz), rounded **down** to 0.01. When that is
  below 0.01 lot the main policy still trades 0.01 lot (so the real risk is above target; reported),
  the alternative policy skips the trade.
* Profit per trade = lots x 100 x stop distance x net R. Costs are inside net R (historical spread at
  least $0.08, 2 points slippage per fill); commission 0 (owner); swap 0 because every trade closes
  by 16:00 New York, before the 17:00 rollover (checked on the data).
* Margin at 1:leverage = lots x 100 x entry price / leverage; margin level at the stop =
  (equity - loss at stop) / margin; stop-out assumed at 50% margin level (GUESS: typical retail).
* Edge uncertainty: the same paths are re-run with the edge shifted to the lower end of its 95%
  interval and to zero, because the measured edge itself is uncertain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.sessions import ny_minutes
from ..logging_setup import get_logger
from .lb_phase2 import boot_stats

log = get_logger(__name__)

CONTRACT = 100.0
MIN_LOT = 0.01
STOP_OUT = 0.50


def load_trades(path: str | Path, since: str | None = None) -> pd.DataFrame:
    """Trade list with entry_time, entry, risk ($ stop distance), r (net R), exit_time."""
    t = pd.read_csv(path)
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    if "exit_time" in t:
        t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values("entry_time").reset_index(drop=True)
    if since:
        t = t[t["entry_time"] >= pd.Timestamp(since, tz="UTC")].reset_index(drop=True)
    return t


def block_paths(n_trades: int, n_paths: int, length: int, block: int = 10, seed: int = 0) -> np.ndarray:
    """Indices of circular block-bootstrap paths, shape (n_paths, length)."""
    rng = np.random.default_rng(seed)
    nb = -(-length // block)
    starts = rng.integers(0, n_trades, size=(n_paths, nb))
    return ((starts[..., None] + np.arange(block)) % n_trades).reshape(n_paths, -1)[:, :length]


def simulate(r: np.ndarray, risk: np.ndarray, price: np.ndarray, idx: np.ndarray, balance: float,
             risk_pct: float, leverage: float, policy: str = "min_lot") -> dict:
    """Equity paths for one risk level. Returns equity (n_paths, length+1) and per-trade sizing stats."""
    n_paths, length = idx.shape
    eq = np.empty((n_paths, length + 1))
    eq[:, 0] = balance
    over = skipped = 0
    real_risk, margin_used, min_level = [], [], np.full(n_paths, np.inf)
    alive = np.ones(n_paths, bool)
    for k in range(length):
        j = idx[:, k]
        e = eq[:, k]
        per_lot_risk = risk[j] * CONTRACT
        raw = e * risk_pct / 100 / per_lot_risk
        lots = np.floor(raw / MIN_LOT + 1e-9) * MIN_LOT
        small = lots < MIN_LOT
        if policy == "min_lot":
            lots = np.where(small, MIN_LOT, lots)
            over += int((small & alive).sum())
        else:
            lots = np.where(small, 0.0, lots)
            skipped += int((small & alive).sum())
        margin = lots * CONTRACT * price[j] / leverage
        can = alive & (margin <= e) & (lots > 0)         # not enough free margin: no trade
        lots = np.where(can, lots, 0.0)
        loss_at_stop = lots * per_lot_risk
        with np.errstate(divide="ignore", invalid="ignore"):
            lvl = np.where(lots > 0, (e - loss_at_stop) / np.where(margin > 0, margin, 1), np.inf)
        min_level = np.minimum(min_level, lvl)
        pnl = lots * per_lot_risk * r[j]
        pnl = np.maximum(pnl, -(e - STOP_OUT * margin))      # broker stop-out closes a trade too big for the account
        eq[:, k + 1] = np.maximum(e + pnl, 0.0)
        if k % 5 == 0:
            m = lots > 0
            real_risk.append(100 * loss_at_stop[m] / e[m])
            margin_used.append(100 * margin[m] / e[m])
        alive &= eq[:, k + 1] >= MIN_LOT * CONTRACT * float(np.median(risk))   # can still afford 0.01 lot
    rr = np.concatenate(real_risk) if real_risk else np.array([np.nan])
    mu = np.concatenate(margin_used) if margin_used else np.array([np.nan])
    total = n_paths * length
    return {"equity": eq, "over_pct": 100 * over / total, "skipped_pct": 100 * skipped / total,
            "real_risk_median": float(np.median(rr)), "real_risk_p95": float(np.percentile(rr, 95)),
            "real_risk_max": float(rr.max()), "margin_pct_median": float(np.median(mu)),
            "margin_pct_max": float(mu.max()), "stop_out_paths_pct": float(100 * (min_level < STOP_OUT).mean()),
            "alive_end_pct": float(100 * alive.mean())}


def path_stats(eq: np.ndarray, balance: float, horizons=(100, 250, 500)) -> dict:
    """Survival, drawdown, target and recovery statistics of equity paths."""
    peak = np.maximum.accumulate(eq, axis=1)
    dd = 1 - eq / np.where(peak > 0, peak, 1)
    low = eq.min(axis=1)
    out = {f"p_lose_{x}pct": float(100 * (low <= balance * (1 - x / 100)).mean()) for x in (20, 30, 50, 70)}
    out |= {f"p_dd_over_{x}pct": float(100 * (dd.max(axis=1) > x / 100).mean()) for x in (10, 20, 30, 50)}
    out["p_reach_1000"] = float(100 * (eq.max(axis=1) >= 2 * balance).mean())
    out["median_max_dd_pct"] = float(100 * np.median(dd.max(axis=1)))
    for h in horizons:
        if h < eq.shape[1]:
            e = eq[:, h]
            out[f"end_{h}"] = {"p5": float(np.percentile(e, 5)), "median": float(np.median(e)),
                               "p95": float(np.percentile(e, 95)), "p_below_start": float(100 * (e < balance).mean())}
    for x in (20, 30):
        hit_rec, dur = 0, []
        hit = 0
        for row_eq, row_dd, row_pk in zip(eq, dd, peak):
            w = np.nonzero(row_dd >= x / 100)[0]
            if not len(w):
                continue
            hit += 1
            k = w[0]
            target = row_pk[k]
            back = np.nonzero(row_eq[k:] >= target)[0]
            if len(back):
                hit_rec += 1
                dur.append(int(back[0]))
        out[f"dd{x}_hit_pct"] = float(100 * hit / len(eq))
        out[f"dd{x}_recovered_pct"] = float(100 * hit_rec / hit) if hit else np.nan
        out[f"dd{x}_recovery_trades_median"] = float(np.median(dur)) if dur else np.nan
        out[f"dd{x}_recovery_trades_p90"] = float(np.percentile(dur, 90)) if dur else np.nan
    return out


def streaks(r: np.ndarray, full_stop: np.ndarray | None = None) -> dict:
    """Longest run of non-winning trades (r <= 0) and of full stop-outs."""
    def longest(mask):
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best
    out = {"longest_non_winning": longest(r <= 0)}
    if full_stop is not None:
        out["longest_full_stops"] = longest(full_stop)
    return out


def run_montecarlo(trades_csv: str | Path, out_dir: str | Path, balance: float = 500.0, leverage: float = 500.0,
                   risks=(1, 2, 3, 4, 5), n_paths: int = 10_000, length: int = 500, block: int = 10,
                   since: str | None = "2023-01-01", seed: int = 7, chart_paths: int = 25,
                   scale_to_last_price: bool = False) -> dict:
    """Run every risk level on the same paths and write ``results.json`` and ``report.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_tr = load_trades(trades_csv)
    tr = load_trades(trades_csv, since)
    r, risk, price = (tr[k].to_numpy(float) for k in ("r", "risk", "entry"))
    if scale_to_last_price:          # stop distances in today's dollars (gold was ~$1,200 in 2015)
        f = all_tr["entry"].iloc[-1] / price
        risk, price = risk * f, price * f
    full_stop = (tr["exit_reason"] == "stop").to_numpy() if "exit_reason" in tr else None
    # swap check: every exit before the 17:00 NY rollover of its entry day
    cross = 0
    if "exit_time" in tr:
        mi, mo = ny_minutes(pd.DatetimeIndex(tr["entry_time"])), ny_minutes(pd.DatetimeIndex(tr["exit_time"]))
        same = (tr["exit_time"] - tr["entry_time"]) < pd.Timedelta(hours=16)
        cross = int((~(same & (mo < 17 * 60) & (mo >= mi))).sum())
    b = boot_stats(r)
    lo_edge = float(b["exp_ci"].split(" to ")[0])
    idx = block_paths(len(r), n_paths, length, block, seed)
    scen = {"as measured": r, f"edge at lower 95% bound ({lo_edge:+.3f}R)": r - r.mean() + lo_edge,
            "zero edge": r - r.mean(), "costs x2": r - tr["cost_r"].to_numpy(float) if "cost_r" in tr else r}
    res = {"inputs": {"trades_used": int(len(tr)), "since": since, "all_trades": int(len(all_tr)),
                      "balance": balance, "leverage": leverage, "paths": n_paths, "length": length, "block": block,
                      "mean_r": float(r.mean()), "edge_ci95": b["exp_ci"], "win_rate": float(100 * (r > 0).mean()),
                      "stop_median": float(np.median(risk)), "stop_p95": float(np.percentile(risk, 95)),
                      "stop_min": float(risk.min()), "stop_max": float(risk.max()),
                      "price_median": float(np.median(price)), "price_last": float(price[-1]),
                      "trades_crossing_rollover": cross, **streaks(r, full_stop)},
           "levels": {}, "skip_policy": {}, "scenarios": {}, "block_check": {}, "charts": {}, "streak_mc": {}}
    for rp in risks:
        for pol, key in (("min_lot", "levels"), ("skip", "skip_policy")):
            s = simulate(r, risk, price, idx, balance, rp, leverage, pol)
            res[key][rp] = {k: v for k, v in s.items() if k != "equity"} | path_stats(s["equity"], balance)
            eq = s["equity"]
            peak = np.maximum.accumulate(eq, axis=1)
            dd = 100 * (1 - eq / np.where(peak > 0, peak, 1))
            steps = list(range(0, length + 1, 5))
            res["charts"].setdefault(pol, {})[rp] = {
                "steps": steps,
                **{f"eq_p{q}": np.percentile(eq[:, steps], q, axis=0).round(2).tolist() for q in (5, 25, 50, 75, 95)},
                "samples": eq[:chart_paths, steps].round(2).tolist(),
                "dd_p50": np.percentile(dd[:, steps], 50, axis=0).round(1).tolist(),
                "dd_p95": np.percentile(dd[:, steps], 95, axis=0).round(1).tolist(),
                "dd_hist": np.histogram(dd.max(axis=1), bins=np.arange(0, 101, 5))[0].tolist()}
            for name, rr in scen.items():
                if name == "as measured":
                    continue
                sc = simulate(rr, risk, price, idx, balance, rp, leverage, pol)
                p = path_stats(sc["equity"], balance)
                res["scenarios"].setdefault(pol, {}).setdefault(name, {})[rp] = {
                    k: p[k] for k in ("p_lose_50pct", "p_lose_70pct", "p_dd_over_30pct", "p_reach_1000")} | \
                    {"end_500_median": p["end_500"]["median"] if "end_500" in p else np.nan}
        for bl in (1, 30):
            ib = block_paths(len(r), min(n_paths, 4000), length, bl, seed + bl)
            for pol in ("min_lot", "skip"):
                p = path_stats(simulate(r, risk, price, ib, balance, rp, leverage, pol)["equity"], balance)
                res["block_check"].setdefault(pol, {}).setdefault(f"block {bl}", {})[rp] = {
                    k: p[k] for k in ("p_lose_50pct", "p_dd_over_30pct")}
    mx = np.array([streaks(r[row] )["longest_non_winning"] for row in idx[:4000]])
    res["streak_mc"] = {"median_longest": float(np.median(mx)), **{f"p_ge_{n}": float(100 * (mx >= n).mean())
                                                                    for n in (20, 27, 35, 40)}}
    worst_r = float(np.percentile(r[r < 0], 5)) if (r < 0).any() else -1.0
    res["streak_impact"] = {rp: {n: round(100 * (1 - (1 - rp / 100 * abs(worst_r)) ** n), 1) for n in (10, 19, 27, 35, 40)}
                            for rp in risks}
    res["inputs"]["loss_per_stop_r"] = worst_r
    res["inputs"]["account_for_risk_pct"] = {rp: {"median_stop": round(MIN_LOT * CONTRACT * float(np.median(risk)) / (rp / 100)),
                                                  "p95_stop": round(MIN_LOT * CONTRACT * float(np.percentile(risk, 95)) / (rp / 100))}
                                             for rp in risks}
    (out / "results.json").write_text(json.dumps(res, default=float, indent=1), encoding="utf-8")
    log.info("montecarlo_done", extra={"risks": list(risks), "paths": n_paths})
    return res
