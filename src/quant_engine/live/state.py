"""Persistent state of the live engine (JSON) and its message log (JSON lines)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..strategy.entry import Signal

PENDING, OPEN, CLOSED = "pending", "open", "closed"


def signal_to_dict(s: Signal) -> dict:
    d = asdict(s)
    d["time"] = s.time.isoformat()
    return d


def signal_from_dict(d: dict) -> Signal:
    d = dict(d)
    d["time"] = pd.Timestamp(d["time"]).tz_convert("UTC")
    return Signal(**d)


@dataclass
class SentSignal:
    signal: Signal
    message_id: int | None
    lots: float
    size_note: str = ""
    status: str = PENDING
    events_sent: int = 0
    outcome: str = "pending"
    r: float | None = None
    alerts: list[str] = field(default_factory=list)   # news event times already alerted
    extra: dict = field(default_factory=dict)          # mode-specific data (london: lots per side)

    @property
    def key(self) -> str:
        return self.signal.time.isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = signal_to_dict(self.signal)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SentSignal":
        d = dict(d)
        d["signal"] = signal_from_dict(d["signal"])
        return cls(**d)


class EngineState:
    """``state.json`` (signals, summaries) plus ``messages.jsonl`` (everything sent or logged)."""

    def __init__(self, folder: str | Path) -> None:
        self.dir = Path(folder)
        self.path, self.log_path = self.dir / "state.json", self.dir / "messages.jsonl"
        self.signals: dict[str, SentSignal] = {}
        self.summaries: list[str] = []
        self.last_candle: str | None = None
        self.news_refreshed: str | None = None
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.signals = {k: SentSignal.from_dict(v) for k, v in raw.get("signals", {}).items()}
            self.summaries = raw.get("summaries", [])
            self.last_candle = raw.get("last_candle")
            self.news_refreshed = raw.get("news_refreshed")

    def active(self) -> list[SentSignal]:
        return [s for s in self.signals.values() if s.status != CLOSED]

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signals": {k: v.to_dict() for k, v in self.signals.items()},
                                   "summaries": self.summaries[-60:], "last_candle": self.last_candle,
                                   "news_refreshed": self.news_refreshed}, indent=1, default=str))
        tmp.replace(self.path)

    def log(self, kind: str, text: str, when: pd.Timestamp, key: str | None = None, sent: bool = True) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": when.isoformat(), "kind": kind, "signal": key, "sent": sent,
                                "text": text}) + "\n")


def read_messages(folder: str | Path, limit: int = 200) -> pd.DataFrame:
    """Last ``limit`` logged messages (newest first) for the dashboard."""
    p = Path(folder) / "messages.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["time", "kind", "signal", "sent", "text"])
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-limit:] if x.strip()]
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
