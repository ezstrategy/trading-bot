"""Versioned, never-modified research datasets (Quant Lab module 1).

Every import (a CSV file, or a snapshot of the MT5 bars already downloaded into
:class:`~quant_engine.data.storage.ParquetStore`) becomes a **dataset version**::

    data/catalog/XAUUSD/M5/<version_id>/
        original.csv | original.parquet   the source exactly as received (byte copy)
        bars.parquet                      normalised bars: engine schema, UTC open times
        quality.json                      full data-quality report
        manifest.json                     source, settings, checksums, row counts, quality summary

Rules:

* A version folder is written once (to a temp folder, then renamed) and its files are made
  read-only. Nothing in this module modifies or deletes a version.
* Re-importing the same source with the same settings returns the existing version instead of
  creating a copy; any difference (another file, timezone, layout) creates a new version.
* Nothing is repaired. Missing candles stay missing, spreads that are not in the source stay
  empty. Rows are put in time order (a change of order only, recorded in the manifest);
  duplicates and other problems are kept and reported.
* :meth:`DatasetCatalog.load` refuses a version whose quality report has a *blocking* issue
  (``data.yaml: quality.blocking_issues``) unless ``allow_issues=True`` is passed, and
  :meth:`DatasetCatalog.verify` re-checks the checksums.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import QualityConfig
from ..logging_setup import get_logger
from ..mt5.server_time import ServerClock
from ..timeframes import Timeframe
from .csv_import import ParseResult, check_timeframe, parse_csv
from .quality import ERROR, WARNING, QualityReport, validate_bars
from .schema import BAR_COLUMNS

log = get_logger(__name__)
SCHEMA_VERSION = 1


class DataQualityError(ValueError):
    """A dataset version has blocking quality issues."""


class DatasetNotFound(LookupError):
    """No dataset version with that id."""


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bars_fingerprint(bars: pd.DataFrame) -> str:
    """Content hash of a bar frame that does not depend on the Parquet writer version."""
    h = hashlib.sha256()
    h.update(np.asarray(pd.DatetimeIndex(bars["time"]).asi8, dtype="int64").tobytes())
    for k in BAR_COLUMNS[1:]:
        h.update(np.ascontiguousarray(bars[k].to_numpy(dtype="float64")).tobytes())
    return h.hexdigest()


def _settings_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _read_only(folder: Path) -> None:
    for p in folder.iterdir():
        os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


@dataclass
class ImportResult:
    """What an import did."""

    version_id: str
    created: bool                 # False when the same source + settings already existed
    manifest: dict

    @property
    def quality_ok(self) -> bool:
        """True when the version has no blocking quality issue."""
        return not self.manifest["quality"]["blocking"]


class DatasetCatalog:
    """The dataset versions under ``root`` (normally ``data/catalog``)."""

    def __init__(self, root: str | Path, quality: QualityConfig | None = None) -> None:
        self.root = Path(root)
        self.quality = quality or QualityConfig()

    # ------------------------------------------------------------ listing
    def _manifest_paths(self) -> list[Path]:
        return sorted(self.root.glob("*/*/*/manifest.json")) if self.root.exists() else []

    def manifests(self, symbol: str | None = None, tf: Timeframe | str | None = None) -> list[dict]:
        """All manifests, oldest first, optionally for one symbol / timeframe."""
        out = []
        for p in self._manifest_paths():
            m = json.loads(p.read_text(encoding="utf-8"))
            if symbol and m["symbol"] != symbol.upper():
                continue
            if tf and m["timeframe"] != Timeframe.parse(tf).value:
                continue
            out.append(m)
        return sorted(out, key=lambda m: (m["created_utc"], m["version_id"]))

    def list(self, symbol: str | None = None, tf: Timeframe | str | None = None) -> pd.DataFrame:
        """One row per version: id, symbol, timeframe, source, range, rows, quality."""
        rows = [{"version_id": m["version_id"], "symbol": m["symbol"], "timeframe": m["timeframe"],
                 "source": m["source"]["kind"], "source_name": m["source"].get("name", ""),
                 "first_utc": m["first_utc"], "last_utc": m["last_utc"], "rows": m["rows"],
                 "spread": "from source" if m["spread_in_source"] else "none (assumed in backtests)",
                 "blocking_issues": sum(m["quality"]["blocking"].values()),
                 "warnings": sum(v for k, v in m["quality"]["counts"].items() if k not in m["quality"]["blocking"]),
                 "created_utc": m["created_utc"]} for m in self.manifests(symbol, tf)]
        return pd.DataFrame(rows)

    def folder(self, version_id: str) -> Path:
        """Folder of a version (raises :class:`DatasetNotFound`)."""
        hits = list(self.root.glob(f"*/*/{version_id}/manifest.json")) if self.root.exists() else []
        if not hits:
            raise DatasetNotFound(version_id)
        return hits[0].parent

    def manifest(self, version_id: str) -> dict:
        """The manifest of one version."""
        return json.loads((self.folder(version_id) / "manifest.json").read_text(encoding="utf-8"))

    def quality_report(self, version_id: str) -> dict:
        """The full stored quality report of one version."""
        return json.loads((self.folder(version_id) / "quality.json").read_text(encoding="utf-8"))

    def latest(self, symbol: str, tf: Timeframe | str, usable_only: bool = True) -> str | None:
        """Newest version id for a symbol / timeframe (by default only versions without blocking issues)."""
        ms = [m for m in self.manifests(symbol, tf) if not (usable_only and m["quality"]["blocking"])]
        return ms[-1]["version_id"] if ms else None

    # ------------------------------------------------------------ loading
    def load(self, version_id: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
             allow_issues: bool = False) -> pd.DataFrame:
        """Normalised bars of a version, optionally limited to UTC ``[start, end)``.

        Raises :class:`DataQualityError` when the version has blocking issues, unless
        ``allow_issues`` (then the caller is responsible for reporting that choice).
        """
        m = self.manifest(version_id)
        if m["quality"]["blocking"] and not allow_issues:
            raise DataQualityError(f"{version_id} has blocking data issues {m['quality']['blocking']}; "
                                   "see `data-show` and decide before using it (allow_issues=True)")
        df = pd.read_parquet(self.folder(version_id) / "bars.parquet")
        if start is not None:
            df = df[df["time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["time"] < pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def verify(self, version_id: str) -> dict:
        """Recompute checksums of the original and the normalised bars against the manifest."""
        m, f = self.manifest(version_id), self.folder(version_id)
        orig = f / m["source"]["stored_as"]
        bars = pd.read_parquet(f / "bars.parquet")
        res = {"original_ok": orig.exists() and sha256_file(orig) == m["source"]["sha256"],
               "bars_ok": bars_fingerprint(bars) == m["bars_fingerprint"] and len(bars) == m["rows"]}
        res["ok"] = res["original_ok"] and res["bars_ok"]
        return res

    # ------------------------------------------------------------ writing
    def _existing(self, symbol: str, tf: Timeframe, source_key: str, settings_hash: str) -> dict | None:
        for m in self.manifests(symbol, tf):
            if m["source"]["key"] == source_key and m["settings_hash"] == settings_hash:
                return m
        return None

    def _create(self, *, symbol: str, tf: Timeframe, bars: pd.DataFrame, original: Path, original_suffix: str,
                source: dict, settings: dict, extra_issues: list[tuple[str, str, object, str]],
                note: str, now: pd.Timestamp | None, source_key: str | None = None) -> ImportResult:
        symbol = symbol.upper()
        source_sha = sha256_file(original)
        key = source_key or source_sha
        sh = _settings_hash({"symbol": symbol, "tf": tf.value, **settings})
        if (old := self._existing(symbol, tf, key, sh)) is not None:
            log.info("dataset_exists", extra={"version_id": old["version_id"]})
            return ImportResult(old["version_id"], False, old)

        # Put rows in time order (stable, so duplicates keep file order); values are untouched.
        t_in = pd.DatetimeIndex(bars["time"])
        out_of_order = int((np.diff(t_in.asi8) < 0).sum()) if len(bars) > 1 else 0
        bars = bars.sort_values("time", kind="stable").reset_index(drop=True)
        rep: QualityReport = validate_bars(bars, tf, self.quality, dataset=f"{symbol} {tf.value}")
        if out_of_order:
            rep.add("reordered", WARNING, None, f"{out_of_order} rows were out of time order in the source; "
                                                "stored sorted by time (values unchanged)")
        for kind, sev, when, detail in extra_issues:
            rep.add(kind, sev, when, detail)
        if not bars["spread"].notna().any():
            rep.add("no_spread", WARNING, None, "the source has no spread; backtests must use an assumed "
                                                "spread and say so (no spread is invented here)")
        counts = rep.counts()
        blocking = {k: v for k, v in counts.items() if k in self.quality.blocking_issues}
        created = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
        fp = bars_fingerprint(bars)
        vid = f"{symbol}_{tf.value}_{created:%Y%m%dT%H%M%SZ}_{fp[:8]}"
        manifest = {
            "schema_version": SCHEMA_VERSION, "version_id": vid, "symbol": symbol, "timeframe": tf.value,
            "created_utc": created.isoformat(), "note": note,
            "source": {**source, "sha256": source_sha, "key": key, "stored_as": f"original{original_suffix}",
                       "bytes": original.stat().st_size},
            "settings": settings, "settings_hash": sh,
            "rows": len(bars), "first_utc": str(bars["time"].iat[0]) if len(bars) else None,
            "last_utc": str(bars["time"].iat[-1]) if len(bars) else None,
            "time_convention": "UTC, candle open time", "price_side": settings.get("price_side", "bid"),
            "spread_in_source": bool(bars["spread"].notna().any()),
            "bars_fingerprint": fp,
            "quality": {"ok": rep.ok and not blocking, "counts": counts, "blocking": blocking,
                        "errors": sum(1 for i in rep.issues if i.severity == ERROR)},
        }
        dest = self.root / symbol / tf.value / vid
        tmp = dest.with_name(dest.name + ".tmp")
        if dest.exists():
            raise FileExistsError(f"{dest} already exists; versions are never overwritten")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        shutil.copyfile(original, tmp / f"original{original_suffix}")
        bars.to_parquet(tmp / "bars.parquet", index=False)
        (tmp / "quality.json").write_text(json.dumps(rep.to_dict(), indent=1, default=str), encoding="utf-8")
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, dest)
        _read_only(dest)
        log.info("dataset_created", extra={"version_id": vid, "rows": len(bars), "blocking": blocking})
        return ImportResult(vid, True, manifest)

    def import_csv(self, path: str | Path, symbol: str, tf: Timeframe | str, *, source_tz: str = "UTC",
                   clock: ServerClock | None = None, source_name: str = "", price_side: str = "bid",
                   layout: str | None = None, note: str = "", now: pd.Timestamp | None = None) -> ImportResult:
        """Import a CSV file as a new dataset version (see :func:`~.csv_import.parse_csv`).

        Args:
            path: the CSV file; it is copied byte for byte into the version.
            symbol: canonical instrument, e.g. ``XAUUSD``.
            tf: timeframe of the candles (M1, M5, M15, ...); checked against the file.
            source_tz: timezone of the file's times (``UTC``, ``+02:00``, an IANA zone, ``mt5_server``).
            clock: broker server clock for ``mt5_server`` (record its settings in ``source_name``/``note``).
            source_name: free text, e.g. "MT5 export, broker X, account demo".
            price_side: ``bid`` (MT5 charts), ``mid`` or ``ask``; recorded, not converted.
        """
        tf = Timeframe.parse(tf)
        path = Path(path)
        res: ParseResult = parse_csv(path, source_tz=source_tz, clock=clock, layout=layout)
        check_timeframe(res, tf)
        settings = {**res.settings(), "price_side": price_side,
                    "server_clock": None if clock is None or source_tz.lower() != "mt5_server" else
                    {"mode": clock.mode, "fixed_offset_hours": clock.fixed_offset_hours, "timezone": clock.timezone}}
        issues = [("unparseable_row", ERROR, None, f"line {ln}: {why}") for ln, why in res.unparseable[:200]]
        if len(res.unparseable) > 200:
            issues.append(("unparseable_row_truncated", WARNING, None, f"{len(res.unparseable)} unparseable rows in total"))
        src = {"kind": "csv", "name": source_name or path.name, "original_filename": path.name}
        return self._create(symbol=symbol, tf=tf, bars=res.bars, original=path, original_suffix=path.suffix or ".csv",
                            source=src, settings=settings, extra_issues=issues, note=note, now=now)

    def import_frame(self, bars: pd.DataFrame, symbol: str, tf: Timeframe | str, *, source_kind: str,
                     source_name: str, settings: dict | None = None, note: str = "",
                     now: pd.Timestamp | None = None) -> ImportResult:
        """Import bars that are already in the engine schema (UTC), e.g. MT5 bars from the store.

        The frame itself is saved as ``original.parquet`` (the exact rows received).
        """
        tf = Timeframe.parse(tf)
        missing = [c for c in BAR_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"bars are missing columns {missing}")
        t = pd.DatetimeIndex(bars["time"])
        if t.tz is None or str(t.tz) != "UTC":
            raise ValueError("bars must have tz-aware UTC times (convert at the source boundary)")
        frame = bars[BAR_COLUMNS].reset_index(drop=True)
        self.root.mkdir(parents=True, exist_ok=True)
        tmpfile = self.root / f".incoming_{os.getpid()}.parquet"
        frame.to_parquet(tmpfile, index=False)
        try:
            return self._create(symbol=symbol, tf=tf, bars=frame.copy(), original=tmpfile, original_suffix=".parquet",
                                source={"kind": source_kind, "name": source_name},
                                settings={"source_tz": "UTC (converted at source)", "price_side": "bid",
                                          **(settings or {})},
                                extra_issues=[], note=note, now=now, source_key="bars:" + bars_fingerprint(frame))
        finally:
            tmpfile.unlink(missing_ok=True)
