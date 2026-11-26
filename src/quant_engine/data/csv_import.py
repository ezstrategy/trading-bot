"""Parse historical OHLC CSV files into the engine's bar schema (UTC open times).

Supported layouts (detected from the header, or forced with ``layout``):

* **MT5 export** (MetaTrader 5 "Bars" export): tab separated, header
  ``<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>``, dates like ``2024.01.02``.
* **MT5 no header**: ``date,time,open,high,low,close,tickvol[,vol,spread]`` with no header row.
* **Generic**: any delimiter, a header with a time column (``time``, ``datetime``, ``timestamp``,
  ``date`` + ``time``, ...) and ``open, high, low, close``; optional ``tick_volume`` / ``volume``,
  ``spread``, ``real_volume``. A numeric time column is read as epoch seconds (or ms).

Times are converted to UTC from ``source_tz``:

* ``UTC``; a fixed offset like ``+02:00`` / ``-05:00``; any IANA zone (``Europe/London``);
* ``mt5_server``: the broker's server clock, converted with the same
  :class:`~quant_engine.mt5.server_time.ServerClock` rules the MT5 collector uses.

Nothing is invented: a file without a spread column gets an empty (NaN) spread, and missing
candles stay missing. Rows that cannot be parsed are listed (with their line number) in the
result, never silently dropped; they are not in the normalised bars because they have no value
that could be stored. The original file is kept untouched by :mod:`.catalog`.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..mt5.server_time import ServerClock
from ..timeframes import Timeframe
from .schema import BAR_COLUMNS

ALIASES = {
    "time": ["time", "datetime", "date_time", "timestamp", "gmt time", "local time", "open time", "opentime"],
    "date": ["date", "day"],
    "clock": ["time", "hour"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "tick_volume": ["tick_volume", "tickvol", "tick volume", "ticks", "volume", "vol"],
    "real_volume": ["real_volume", "vol", "real volume"],
    "spread": ["spread"],
}
MT5_NOHEADER = ["date", "clock", "open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
_OFFSET = re.compile(r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$", re.I)


@dataclass
class ParseResult:
    """Bars parsed from a CSV file plus everything needed to reproduce the parse."""

    bars: pd.DataFrame
    layout: str
    columns: dict[str, str]                    # engine column -> file column
    source_tz: str
    rows_in_file: int
    unparseable: list[tuple[int, str]] = field(default_factory=list)   # (line number, reason)
    spread_in_file: bool = False
    inferred_minutes: float | None = None

    def settings(self) -> dict:
        """JSON-serialisable import settings (stored in the dataset manifest)."""
        return {"layout": self.layout, "columns": self.columns, "source_tz": self.source_tz,
                "rows_in_file": self.rows_in_file, "unparseable_rows": len(self.unparseable),
                "spread_in_file": self.spread_in_file, "inferred_minutes": self.inferred_minutes}


def _norm(name: str) -> str:
    return name.strip().strip("<>").strip().lower().replace("-", "_")


def _sniff(text: str) -> tuple[str, bool]:
    """Delimiter and whether the first line is a header."""
    first = text.splitlines()[0] if text else ""
    try:
        delim = csv.Sniffer().sniff(first, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = "\t" if "\t" in first else ","
    cells = [c.strip() for c in first.split(delim)]
    has_header = any(re.search(r"[A-Za-z<]", c) and not re.fullmatch(r"[\d.:/\- T]+", c) for c in cells)
    return delim, has_header


def _match(cols: list[str], want: str, taken: set[str]) -> str | None:
    normed = {_norm(c): c for c in cols if c not in taken}
    for a in ALIASES[want]:
        if a in normed:
            return normed[a]
    return None


def _map_columns(cols: list[str]) -> tuple[str, dict[str, str]]:
    normed = [_norm(c) for c in cols]
    taken: set[str] = set()
    m: dict[str, str] = {}
    layout = "mt5_export" if any(c.strip().startswith("<") for c in cols) else "generic"
    if "date" in normed and ("time" in normed or "hour" in normed):
        m["date"] = _match(cols, "date", taken)
        taken.add(m["date"])
        m["clock"] = _match(cols, "clock", taken)
        taken.add(m["clock"])
    else:
        t = _match(cols, "time", taken) or _match(cols, "date", taken)
        if t is None:
            raise ValueError(f"no time column found in {cols}")
        m["time"] = t
        taken.add(t)
    for k in ("open", "high", "low", "close"):
        c = _match(cols, k, taken)
        if c is None:
            raise ValueError(f"no {k!r} column found in {cols}")
        m[k] = c
        taken.add(c)
    if layout == "mt5_export":   # MT5: <TICKVOL> is tick volume, <VOL> is real volume
        for k, names in (("tick_volume", ["tickvol"]), ("real_volume", ["vol"])):
            c = next((x for x in cols if _norm(x) in names and x not in taken), None)
            if c:
                m[k] = c
                taken.add(c)
    for k in ("tick_volume", "real_volume", "spread"):
        if k not in m and (c := _match(cols, k, taken)) is not None:
            m[k] = c
            taken.add(c)
    return layout, m


def _parse_times(raw: pd.Series, numeric_ok: bool = True) -> pd.Series:
    """Naive wall-clock datetimes (NaT where unparseable)."""
    s = raw.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    if numeric_ok and num.notna().all() and len(s):
        unit = "ms" if num.abs().max() > 1e11 else "s"
        return pd.to_datetime(num, unit=unit)
    s = s.str.replace(".", "-", regex=False) if s.str.match(r"^\d{4}\.\d{2}\.\d{2}").any() else s
    out = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:   # explicit offsets in the file win
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def to_utc(wall: pd.Series, source_tz: str, clock: ServerClock | None = None) -> pd.Series:
    """Convert naive wall-clock times in ``source_tz`` to tz-aware UTC."""
    tz = source_tz.strip()
    valid = wall.notna()
    out = pd.Series(pd.NaT, index=wall.index, dtype="datetime64[ns, UTC]")
    if not valid.any():
        return out
    w = pd.DatetimeIndex(wall[valid])
    if tz.upper() in ("UTC", "GMT", "Z", "ETC/UTC"):
        conv = w.tz_localize("UTC")
    elif (m := _OFFSET.match(tz)):
        sign = 1 if m.group(1) == "+" else -1
        off = pd.Timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        conv = (w - sign * off).tz_localize("UTC")
    elif tz.lower() == "mt5_server":
        if clock is None:
            raise ValueError("source_tz 'mt5_server' needs the broker ServerClock (mt5.yaml server_time)")
        epoch = (w - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
        conv = clock.to_utc(np.asarray(epoch, dtype="int64"))
    else:
        conv = w.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    out.loc[valid] = conv
    return out


def parse_csv(path: str | Path | None = None, *, text: str | None = None, source_tz: str = "UTC",
              clock: ServerClock | None = None, layout: str | None = None) -> ParseResult:
    """Read a CSV file (or ``text``) and return UTC bars in the engine schema.

    Args:
        path: file to read (UTF-8, UTF-16 MT5 exports are detected).
        text: CSV content instead of a file (tests).
        source_tz: timezone of the times in the file (see module docstring).
        clock: server clock for ``source_tz='mt5_server'``.
        layout: force ``"mt5_noheader"``; otherwise detected.
    """
    if text is None:
        data = Path(path).read_bytes()
        enc = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        text = data.decode(enc)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    delim, has_header = _sniff(text)
    if layout == "mt5_noheader" or not has_header:
        df = pd.read_csv(io.StringIO(text), sep=delim, header=None, dtype=str, skip_blank_lines=True)
        names = MT5_NOHEADER[: df.shape[1]]
        if df.shape[1] < 6:
            raise ValueError(f"a CSV without a header needs at least 6 columns (date, time, OHLC); got {df.shape[1]}")
        df.columns = names
        layout_name, cmap = "mt5_noheader", {n: n for n in names}
        line0 = 1
    else:
        df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, skip_blank_lines=True)
        layout_name, cmap = _map_columns(list(df.columns))
        line0 = 2
    df = df.reset_index(drop=True)
    if "date" in cmap:
        wall = _parse_times(df[cmap["date"]].astype(str).str.strip() + " " + df[cmap["clock"]].astype(str).str.strip(),
                            numeric_ok=False)
    else:
        wall = _parse_times(df[cmap["time"]])
    times = to_utc(wall, source_tz, clock)
    out = pd.DataFrame({"time": times})
    for k in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
        out[k] = pd.to_numeric(df[cmap[k]], errors="coerce") if k in cmap else np.nan
    if "tick_volume" not in cmap:
        out["tick_volume"] = np.nan
    bad: list[tuple[int, str]] = []
    reason = np.where(out["time"].isna(), "unparseable or non-existent time",
                      np.where(out[["open", "high", "low", "close"]].isna().any(axis=1), "unparseable price", ""))
    for i in np.flatnonzero(reason != ""):
        bad.append((int(i) + line0, str(reason[i])))
    keep = reason == ""
    bars = out.loc[keep, BAR_COLUMNS].reset_index(drop=True)
    bars["time"] = bars["time"].astype("datetime64[ns, UTC]")
    for k in BAR_COLUMNS[1:]:
        bars[k] = bars[k].astype("float64")
    t = pd.DatetimeIndex(bars["time"]).sort_values().unique()
    inferred = float(pd.Series(np.diff(t.asi8) / 60e9).median()) if len(t) > 2 else None
    return ParseResult(bars=bars, layout=layout_name, columns={k: str(v) for k, v in cmap.items()},
                       source_tz=source_tz, rows_in_file=len(df), unparseable=bad,
                       spread_in_file="spread" in cmap, inferred_minutes=inferred)


def check_timeframe(result: ParseResult, tf: Timeframe) -> None:
    """Raise if the file's typical candle spacing does not match the declared timeframe."""
    m = result.inferred_minutes
    if m is not None and abs(m - tf.minutes) > 1e-9:
        raise ValueError(f"declared timeframe {tf.value} ({tf.minutes} min) but the file's median candle "
                         f"spacing is {m:g} min")
