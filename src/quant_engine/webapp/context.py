"""Shared state of the web app: config, store, MT5 lock and the feature cache."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import AppConfig
from ..dashboard import services
from ..data.storage import ParquetStore
from ..features.context import STRUCTURE_TFS


def md_html(text: str) -> str:
    """Render a local markdown report to HTML (tables and fenced code supported)."""
    import markdown
    return markdown.markdown(text or "", extensions=["tables", "fenced_code", "sane_lists"])


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN -> null, timestamps -> ISO 8601 UTC)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


@dataclass
class AppContext:
    """Everything a route needs. One instance per running app."""

    cfg: AppConfig
    store: ParquetStore
    point: float = 0.01
    config_dir: str | None = None
    mt5_info: dict | None = None
    mt5_lock: threading.Lock = field(default_factory=threading.Lock)   # MT5 is not thread safe
    _cache_lock: threading.Lock = field(default_factory=threading.Lock)
    _cache: dict[tuple, Any] = field(default_factory=dict)

    @property
    def canonical(self) -> str:
        return self.cfg.engine.instrument

    def fingerprint(self) -> tuple:
        """Changes whenever a stored file changes, so cached features refresh."""
        out = []
        for tf in STRUCTURE_TFS:
            p = self.store.bars_path(self.canonical, tf)
            out.append((tf.value, p.stat().st_mtime if p.exists() else 0))
        return tuple(out)

    def has_data(self) -> bool:
        return all(m for _, m in self.fingerprint())

    def features(self, point: float):
        """(bars, MarketFeatures) for the stored history, cached per data version and point."""
        key = (self.fingerprint(), point)
        with self._cache_lock:
            if key not in self._cache:
                bars = services.load_bars(self.store, self.canonical)
                self._cache.clear()   # keep only the newest version
                self._cache[key] = (bars, services.build_features(bars, self.cfg, point))
            return self._cache[key]
