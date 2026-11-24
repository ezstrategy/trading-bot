"""Economic calendar and news blackout windows (Strategy Spec v1).

* Tier 1 (NFP, CPI, FOMC, rate decisions): no new signals from 60 min before
  to 30 min after (FOMC: 90 min after, to cover the press conference).
* Other high-impact events: 30 min before to 15 min after.
* Bank holidays: the whole New York calendar day.
* Medium/low impact: no block.

The calendar is a table ``time`` (UTC), ``currency``, ``impact``
(High/Medium/Low/Holiday), ``title``. It is cached as CSV; the free weekly
ForexFactory export (JSON with local-offset timestamps) can be fetched and
merged into it. Only events of ``cfg.currencies`` are used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import NewsConfig
from ..logging_setup import get_logger
from .sessions import NY

log = get_logger(__name__)
COLUMNS = ["time", "currency", "impact", "title"]
TIER1, HIGH, HOLIDAY, OTHER = "tier1", "high", "holiday", "other"


def empty_calendar() -> pd.DataFrame:
    return pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns, UTC]"), "currency": [],
                         "impact": [], "title": []})


def parse_ff_json(text: str) -> pd.DataFrame:
    """Parse the ForexFactory weekly export: [{title, country, date, impact}, ...]."""
    rows = json.loads(text)
    df = pd.DataFrame([{"time": pd.Timestamp(r["date"]).tz_convert("UTC"), "currency": r.get("country", ""),
                        "impact": r.get("impact", ""), "title": r.get("title", "")} for r in rows])
    return df[COLUMNS] if len(df) else empty_calendar()


def load_calendar(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return empty_calendar()
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[COLUMNS]


def save_calendar(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Merge ``df`` into the CSV at ``path`` (newer rows win) and return the merged table."""
    merged = pd.concat([load_calendar(path), df], ignore_index=True)
    merged = merged.drop_duplicates(["time", "currency", "title"], keep="last").sort_values("time")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)
    return merged.reset_index(drop=True)


def refresh_calendar(cfg: NewsConfig, fetch: Callable[[str], str] | None = None) -> pd.DataFrame:
    """Download the weekly feed, merge it into the cache and return the cache.
    On a network error the cached calendar is returned and the error is logged."""
    import urllib.request

    def _fetch(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "quant-engine"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8")

    try:
        new = parse_ff_json((fetch or _fetch)(cfg.feed_url))
        log.info("news_refreshed", extra={"events": len(new)})
        return save_calendar(new, cfg.calendar_file)
    except Exception as exc:  # keep running on the cached calendar
        log.warning("news_refresh_failed", extra={"error": str(exc)})
        return load_calendar(cfg.calendar_file)


@dataclass
class NewsWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    event_time: pd.Timestamp
    title: str
    tier: str

    def label(self) -> str:
        return f"{self.title} ({self.tier}) at {self.event_time:%H:%M} UTC"


class NewsCalendar:
    """Blocking windows built from a calendar table."""

    def __init__(self, events: pd.DataFrame, cfg: NewsConfig) -> None:
        self.cfg = cfg
        ev = events[events["currency"].isin(cfg.currencies)] if len(events) else events
        self.windows: list[NewsWindow] = [w for w in (self._window(r) for r in ev.itertuples()) if w]
        self.windows.sort(key=lambda w: w.start)
        self.first = ev["time"].min() if len(ev) else None
        self.last = ev["time"].max() if len(ev) else None

    def classify(self, title: str, impact: str) -> str:
        imp = str(impact).lower()
        if imp == "holiday":
            return HOLIDAY
        if imp != "high":
            return OTHER
        t = str(title).lower()
        return TIER1 if any(k.lower() in t for k in self.cfg.tier1_keywords) else HIGH

    def _window(self, r) -> NewsWindow | None:
        c, tier, t = self.cfg, self.classify(r.title, r.impact), r.time
        if tier == HOLIDAY:
            if not c.block_holidays:
                return None
            day = t.tz_convert(NY).normalize()
            return NewsWindow(day.tz_convert("UTC"), (day + pd.Timedelta(days=1)).tz_convert("UTC"), t, r.title, tier)
        if tier == TIER1:
            after = c.fomc_after_min if "fomc" in str(r.title).lower() or "federal funds" in str(r.title).lower() \
                else c.tier1_after_min
            return NewsWindow(t - pd.Timedelta(minutes=c.tier1_before_min), t + pd.Timedelta(minutes=after),
                              t, r.title, tier)
        if tier == HIGH:
            return NewsWindow(t - pd.Timedelta(minutes=c.high_before_min), t + pd.Timedelta(minutes=c.high_after_min),
                              t, r.title, tier)
        return None

    def blackout(self, t: pd.Timestamp) -> NewsWindow | None:
        """The window blocking new signals at ``t``, if any."""
        if not self.cfg.enabled:
            return None
        return next((w for w in self.windows if w.start <= t < w.end), None)

    def upcoming(self, t: pd.Timestamp, minutes: int) -> list[NewsWindow]:
        """Tier 1 / high events (not holidays) starting within ``minutes`` after ``t``."""
        lim = t + pd.Timedelta(minutes=minutes)
        return [w for w in self.windows if w.tier in (TIER1, HIGH) and t < w.event_time <= lim]

    def covers(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """True when the calendar spans the whole period (so a backtest can use it)."""
        return self.first is not None and self.first <= start and self.last >= end - pd.Timedelta(days=7)
