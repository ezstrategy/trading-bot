"""Data-quality checks for bars and ticks.

Bars are checked for: unsorted times, duplicates, timestamps off the
candle grid, NaN values, non-positive prices, impossible OHLC (high below
open/close, low above open/close), price spikes, large open-to-previous-close
gaps, prices outside the instrument's plausible bounds, and missing candles.
Missing candles inside expected market closures (weekend, daily break)
are not reported as errors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import QualityConfig
from ..timeframes import Timeframe

ERROR, WARNING = "error", "warning"


@dataclass
class Issue:
    """One data-quality finding."""

    kind: str
    severity: str
    time: str | None
    detail: str


@dataclass
class QualityReport:
    """Result of a validation run."""

    dataset: str
    rows: int
    first: str | None = None
    last: str | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no error-level issues."""
        return not any(i.severity == ERROR for i in self.issues)

    def counts(self) -> dict[str, int]:
        """Number of issues per kind."""
        out: dict[str, int] = {}
        for i in self.issues:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out

    def add(self, kind: str, severity: str, time: object, detail: str) -> None:
        self.issues.append(Issue(kind, severity, None if time is None else str(time), detail))

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {"dataset": self.dataset, "rows": self.rows, "first": self.first, "last": self.last,
                "ok": self.ok, "counts": self.counts(), "issues": [asdict(i) for i in self.issues]}


# ------------------------------------------------------------- market hours
def is_market_closed(ts: pd.Timestamp, q: QualityConfig) -> bool:
    """Whether XAUUSD is normally closed at UTC instant ``ts`` (weekend or daily break)."""
    wd, h = ts.weekday(), ts.hour
    c, o = q.weekly_close, q.weekly_open
    week_pos = wd * 24 + h
    close_pos, open_pos = c.weekday * 24 + c.hour, o.weekday * 24 + o.hour
    if close_pos <= open_pos:
        if close_pos <= week_pos < open_pos:
            return True
    elif week_pos >= close_pos or week_pos < open_pos:
        return True
    b0, b1 = q.daily_break_utc
    return b0 <= h < b1 if b0 <= b1 else (h >= b0 or h < b1)


def _slot_closed(start: pd.Timestamp, tf: Timeframe, q: QualityConfig) -> bool:
    """A missing candle slot is 'expected' if the market is closed for all of it."""
    probes = [start, start + tf.delta / 2, start + tf.delta - pd.Timedelta(minutes=1)]
    return all(is_market_closed(p, q) for p in probes)


# ---------------------------------------------------------------------- bars
def validate_bars(bars: pd.DataFrame, tf: Timeframe, q: QualityConfig,
                  dataset: str = "", max_listed: int = 200) -> QualityReport:
    """Run every bar check and return a :class:`QualityReport`.

    Args:
        bars: frame in the engine's bar schema (UTC ``time`` = candle open).
        tf: timeframe of the candles.
        q: market-hours and threshold settings.
        max_listed: cap on individual issues listed per kind (counts stay exact
            in the summary issue).
    """
    rep = QualityReport(dataset or tf.value, len(bars))
    if bars.empty:
        rep.add("empty", WARNING, None, "no bars")
        return rep
    t = pd.DatetimeIndex(bars["time"])
    rep.first, rep.last = str(t.min()), str(t.max())
    listed: dict[str, int] = {}

    def add(kind: str, sev: str, when: object, detail: str) -> None:
        listed[kind] = listed.get(kind, 0) + 1
        if listed[kind] <= max_listed:
            rep.add(kind, sev, when, detail)

    if t.tz is None or str(t.tz) != "UTC":
        add("not_utc", ERROR, None, f"time column timezone is {t.tz}, expected UTC")
    if not t.is_monotonic_increasing:
        add("unsorted", ERROR, None, "times are not in ascending order")

    dup = t.duplicated(keep="first")
    for when in t[dup]:
        add("duplicate", ERROR, when, "duplicate candle time")

    # Grid alignment: compare each bar's phase with the dominant phase of its
    # trading week. H4 candles follow the broker's server midnight, which moves
    # by an hour in UTC when DST changes, but only ever over a weekend. Shifting
    # by two days puts a whole Sunday-open to Friday-close week in one ISO week.
    step = tf.minutes * 60
    secs = np.asarray((t - pd.Timestamp(0, tz=t.tz)) // pd.Timedelta(seconds=1))
    phase = pd.Series(secs % step, index=t)
    week = pd.Series((t + pd.Timedelta(days=2)).strftime("%G-%V"), index=t)
    dominant = phase.groupby(week.to_numpy()).transform(lambda s: s.mode().iat[0])
    for when in t[(phase != dominant).to_numpy()]:
        add("misaligned", ERROR, when, f"open time not on the {tf.value} grid")

    o, h, l, c = (bars[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    nan = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
    for when in t[nan]:
        add("nan", ERROR, when, "missing OHLC value")
    nonpos = ~nan & ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0))
    for when in t[nonpos]:
        add("non_positive", ERROR, when, "price <= 0")
    bad = ~nan & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))
    for i in np.flatnonzero(bad):
        add("invalid_ohlc", ERROR, t[i], f"O={o[i]} H={h[i]} L={l[i]} C={c[i]}")

    rng = pd.Series(h - l)
    med = rng.rolling(100, min_periods=20).median().shift(1)
    spike = (rng > q.max_spike_atr * med).to_numpy() & ~nan
    for i in np.flatnonzero(spike):
        add("spike", WARNING, t[i], f"range {rng.iat[i]:.2f} is >{q.max_spike_atr}x the recent median")

    if q.max_gap_atr > 0 and len(o) > 1:
        gap = np.abs(o[1:] - c[:-1])
        gmask = np.r_[False, gap > q.max_gap_atr * med.to_numpy()[1:]] & ~nan
        for i in np.flatnonzero(gmask):
            add("gap", WARNING, t[i], f"open {o[i]} is {abs(o[i] - c[i - 1]):.2f} from the previous close "
                                      f"(>{q.max_gap_atr}x the recent median range)")
    if q.price_bounds is not None:
        lo_b, hi_b = q.price_bounds
        oob = ~nan & ((l < lo_b) | (h > hi_b))
        for i in np.flatnonzero(oob):
            add("out_of_bounds", ERROR, t[i], f"L={l[i]} H={h[i]} outside [{lo_b}, {hi_b}]")

    # Missing candles between consecutive (unique, sorted) bars.
    u = t[~dup].sort_values()
    useconds = np.asarray((u - pd.Timestamp(0, tz=u.tz)) // pd.Timedelta(seconds=1))
    gaps = np.flatnonzero(np.diff(useconds) > step)
    for g in gaps:
        a, b = u[g], u[g + 1]
        # A slot only counts if the whole candle fits before the next real one
        # (handles the 1h H4 grid shift across DST weekends).
        slots = pd.date_range(a + tf.delta, b - tf.delta, freq=tf.pandas_freq)
        missing = [s for s in slots if not _slot_closed(s, tf, q)]
        if missing:
            add("missing", ERROR if tf != Timeframe.M1 else WARNING, missing[0],
                f"{len(missing)} {tf.value} candle(s) missing between {a} and {b}")

    for kind, n in listed.items():
        if n > max_listed:
            rep.add(kind + "_truncated", WARNING, None, f"{n} {kind} issues in total, {max_listed} listed")
    return rep


# --------------------------------------------------------------------- ticks
def validate_ticks(ticks: pd.DataFrame, q: QualityConfig, dataset: str = "ticks",
                   max_listed: int = 200) -> QualityReport:
    """Check bid/ask ticks for bad prices, crossed quotes, wide spreads, order and duplicates."""
    rep = QualityReport(dataset, len(ticks))
    if ticks.empty:
        rep.add("empty", WARNING, None, "no ticks")
        return rep
    t = pd.DatetimeIndex(ticks["time"])
    rep.first, rep.last = str(t.min()), str(t.max())
    bid, ask = ticks["bid"].to_numpy(dtype="float64"), ticks["ask"].to_numpy(dtype="float64")
    checks = [
        ("non_positive", ERROR, (bid <= 0) | (ask <= 0), "bid or ask <= 0"),
        ("crossed", ERROR, ask < bid, "ask below bid"),
        ("wide_spread", WARNING, (ask - bid) > q.max_tick_spread, f"spread > {q.max_tick_spread}"),
    ]
    for kind, sev, mask, detail in checks:
        idx = np.flatnonzero(mask)
        for i in idx[:max_listed]:
            rep.add(kind, sev, t[i], f"{detail} (bid={bid[i]}, ask={ask[i]})")
        if len(idx) > max_listed:
            rep.add(kind + "_truncated", WARNING, None, f"{len(idx)} {kind} issues in total")
    if not t.is_monotonic_increasing:
        rep.add("unsorted", ERROR, None, "tick times are not in ascending order")
    dups = ticks.duplicated(keep="first").to_numpy()
    if dups.any():
        rep.add("duplicate", WARNING, t[dups][0], f"{int(dups.sum())} exact duplicate ticks")
    return rep
