"""Find repeatable patterns per market type and check them on data they never saw.

How it works (designed to avoid fooling ourselves):

1. **Many candidates.** Every event (:mod:`.events`) x H4 market (bullish / bearish /
   neutral) x session x direction (BUY / SELL) x exit rule (stop, target, time limit)
   is one candidate pattern. Thousands are tested, so some look good by pure luck.
2. **Discovery period only.** Candidates are scored on the older part of the history
   (``discover_frac``). Trades of one pattern never overlap (the next trade starts after
   the previous one ended), so the trades are close to independent.
3. **Multiple-testing guard.** Each candidate gets a one-sided p-value (average R > 0).
   The Benjamini-Hochberg procedure keeps the false discovery rate at ``fdr_q`` across
   ALL candidates tested. A survivor must also beat costs by ``min_avg_r`` per trade and
   be positive in at least ``min_stable_thirds`` of the three thirds of the discovery period.
4. **Held-out check (walk forward).** Survivors are then measured on the newer part of the
   history. ``confirmed`` needs a positive average with p < ``validate_p`` / (number of
   survivors) (Bonferroni); ``held up`` is positive but not proven; ``failed`` is not positive.

GUESS: a normal approximation of the t-test is used for p-values (at least 30 trades).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import ResearchConfig
from ..strategy.regime import BEARISH, BULLISH, NEUTRAL
from .events import EventTable, event_family
from .outcomes import trade_outcomes

REGIMES = (BULLISH, BEARISH, NEUTRAL)
SESSIONS = ("any", "asia", "london_kz", "newyork_kz", "london_close")
SESSION_NAMES = {"any": "any time", "asia": "Asia session", "london_kz": "London open",
                 "newyork_kz": "New York open", "london_close": "London close"}


@dataclass(frozen=True)
class Pattern:
    """One candidate: when ``event`` happens in ``regime`` (and ``session``), trade ``direction``."""

    event: str
    regime: str
    session: str
    direction: int
    sl_atr: float
    tp_atr: float
    horizon: int

    @property
    def side(self) -> str:
        return "BUY" if self.direction == 1 else "SELL"

    def describe(self, natural: int, tf_minutes: int = 5, usd: bool = False) -> str:
        """Plain-language description."""
        style = "continuation" if self.direction == natural else "reversal"
        when = "" if self.session == "any" else f" during {SESSION_NAMES.get(self.session, self.session)}"
        exits = (f"stop ${self.sl_atr:g}, target ${self.tp_atr:g}" if usd else
                 f"stop {self.sl_atr:g} ATR, target {self.tp_atr:g} ATR")
        hold = self.horizon * tf_minutes
        hold_txt = f"{hold:g} min" if hold < 120 else f"{hold / 60:g} h"
        return f"{self.regime} market: {self.side} after '{self.event}'{when} ({style}); {exits}, max {hold_txt}"


def t_stats(r: np.ndarray) -> dict:
    """Trades, win rate, average/total R, profit factor, t and one-sided p (mean > 0)."""
    n = int(len(r))
    if n == 0:
        return {"trades": 0, "win_rate_pct": np.nan, "avg_r": np.nan, "total_r": 0.0,
                "profit_factor": np.nan, "t": np.nan, "p": 1.0}
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else (np.inf if mean > 0 else -np.inf if mean < 0 else 0.0)
    p = 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else (0.0 if t > 0 else 1.0)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": n, "win_rate_pct": round(100 * float((r > 0).mean()), 1), "avg_r": round(mean, 4),
            "total_r": round(float(r.sum()), 2), "profit_factor": round(float(wins / losses), 2) if losses > 0 else np.nan,
            "t": round(float(t), 3) if np.isfinite(t) else float(t), "p": float(p)}


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """True for the p-values kept at false discovery rate ``q``."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.nonzero(below)[0].max())
        keep[order[:k + 1]] = True
    return keep


def non_overlapping(idx: np.ndarray, exit_idx: np.ndarray) -> np.ndarray:
    """Event rows taken in time order, skipping any that start before the last trade ended."""
    out, busy = [], -1
    for j in idx:
        if j > busy and exit_idx[j] >= 0:
            out.append(j)
            busy = exit_idx[j]
    return np.asarray(out, dtype=np.int64)


@dataclass
class MiningResult:
    tested: pd.DataFrame          # every candidate with discovery stats
    survivors: pd.DataFrame       # passed discovery, with held-out stats and verdict
    leads: pd.DataFrame           # best candidates that did NOT pass, with held-out stats (information only)
    split_time: pd.Timestamp      # first candle of the held-out period
    n_tests: int


def _conditions(et: EventTable, only: list[str] | None = None):
    for name, mask in et.events.items():
        if only and event_family(name) not in only:
            continue
        for reg in REGIMES:
            base = mask & (et.regime == reg) & ~et.blocked
            sessions = ("any",) if name in et.hour_events else SESSIONS
            for sess in sessions:
                m = base if sess == "any" else base & (et.session == sess)
                yield name, reg, sess, m


def mine(et: EventTable, rc: ResearchConfig) -> MiningResult:
    """Run discovery on the older ``discover_frac`` of ``et`` and check survivors on the rest."""
    n = len(et.c)
    split = int(n * rc.discover_frac)
    thirds = np.linspace(0, split, 4).astype(int)
    exits = [(d, sl, tp, hz) for d in (1, -1) for sl in rc.sl_atr for tp in rc.tp_atr for hz in rc.horizons]
    usd = rc.exit_units == "usd"
    scale = np.ones(len(et.c)) if usd else et.atr      # dollar exits: distances are in USD
    outcomes = {k: trade_outcomes(et.times, et.o, et.h, et.l, et.c, et.spread, scale, *k, tf=et.tf) for k in exits}
    rows, trade_sets = [], {}
    for name, reg, sess, m in _conditions(et, rc.only_families):
        idx = np.nonzero(m)[0]
        disc_idx = idx[idx < split]
        if len(disc_idx) < rc.min_trades_discover:
            continue
        for key in exits:
            r, ex = outcomes[key]
            d_idx = disc_idx[(ex[disc_idx] >= 0) & (ex[disc_idx] < split)]
            taken = non_overlapping(d_idx, ex)
            if len(taken) < rc.min_trades_discover:
                continue
            pat = Pattern(name, reg, sess, *key)
            st = t_stats(r[taken])
            pos = sum(1 for a, b in zip(thirds[:-1], thirds[1:])
                      if ((sel := taken[(taken >= a) & (taken < b)]).size >= 5 and r[sel].mean() > 0))
            rows.append({**asdict(pat), "side": pat.side, "natural": et.natural[name], **st,
                         "positive_thirds": pos})
            trade_sets[pat] = idx
    tested = pd.DataFrame(rows)
    split_time = et.times[split] if split < n else et.times[-1]
    if tested.empty:
        empty = pd.DataFrame()
        return MiningResult(tested, empty, empty, split_time, 0)
    tested["bh_pass"] = benjamini_hochberg(tested["p"].to_numpy(), rc.fdr_q)
    tested["survives"] = (tested["bh_pass"] & (tested["avg_r"] >= rc.min_avg_r)
                          & (tested["positive_thirds"] >= rc.min_stable_thirds))
    tested = tested.sort_values("t", ascending=False).reset_index(drop=True)
    tested["family"] = tested["event"].map(event_family)
    tested["timeframe"] = et.tf.value
    tested["exit_units"] = rc.exit_units
    tested["description"] = [Pattern(*r[:7]).describe(r[7], et.tf.minutes, usd) for r in tested[
        ["event", "regime", "session", "direction", "sl_atr", "tp_atr", "horizon", "natural"]].itertuples(index=False)]

    surv = tested[tested["survives"]].drop_duplicates(["event", "regime", "session", "direction"])
    surv = surv.head(rc.max_patterns)
    keys = ["event", "regime", "session", "direction"]
    found = set(map(tuple, surv[keys].to_numpy().tolist()))
    leads = tested[~tested["survives"]].drop_duplicates(keys)
    leads = leads[[tuple(k) not in found for k in leads[keys].to_numpy().tolist()]].head(10)

    def held_out(df: pd.DataFrame, n_family: int) -> pd.DataFrame:
        out = []
        for row in df.itertuples(index=False):
            pat = Pattern(row.event, row.regime, row.session, row.direction, row.sl_atr, row.tp_atr, row.horizon)
            r, ex = outcomes[(pat.direction, pat.sl_atr, pat.tp_atr, pat.horizon)]
            idx = trade_sets[pat]
            v_idx = idx[(idx >= split) & (ex[idx] >= 0)]
            st = t_stats(r[non_overlapping(v_idx, ex)])
            if st["trades"] < rc.min_trades_validate:
                verdict = "not enough trades"
            elif st["avg_r"] > 0 and st["p"] < rc.validate_p / max(n_family, 1):
                verdict = "confirmed"
            elif st["avg_r"] > 0:
                verdict = "held up (not proven)"
            else:
                verdict = "failed"
            out.append({f"val_{k}": v for k, v in st.items()} | {"verdict": verdict})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1) if out else df

    survivors = held_out(surv, len(surv))
    leads = held_out(leads, len(leads))
    return MiningResult(tested, survivors, leads, split_time, len(tested))
