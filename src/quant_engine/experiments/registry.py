"""Append-only experiment registry (Quant Lab module 6).

SQLite file plus one artefact folder per experiment::

    data/experiments/registry.db
    data/experiments/LAB-0001/  experiment.json, report.md, trades.csv, equity.csv, signals.csv, ...

Experiments, data splits and the event log can only be **inserted**: SQLite triggers abort any
UPDATE or DELETE, and an artefact folder that already exists is never written to. Status changes
and research notes are new rows in ``events``; the current status is the latest one.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

STATUSES = ("RECORDED", "REJECTED", "RESEARCH REQUIRED", "VALIDATED FOR PAPER TRADING", "APPROVED FOR LIVE EVALUATION")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, seq INTEGER UNIQUE NOT NULL, created_utc TEXT NOT NULL, kind TEXT NOT NULL,
    title TEXT, hypothesis TEXT, programme TEXT, data_part TEXT, date_from TEXT, date_to TEXT,
    strategies TEXT NOT NULL, families TEXT NOT NULL, dataset_ids TEXT NOT NULL, execution TEXT NOT NULL,
    results TEXT NOT NULL, git_commit TEXT, test_set_used INTEGER NOT NULL DEFAULT 0, parent_id TEXT,
    artefacts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    time_utc TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS splits (
    programme TEXT PRIMARY KEY, created_utc TEXT NOT NULL, data_start TEXT NOT NULL, dev_end TEXT NOT NULL,
    val_end TEXT NOT NULL, data_end TEXT NOT NULL, dataset_ids TEXT NOT NULL, note TEXT);
"""
_TRIGGERS = [f"CREATE TRIGGER IF NOT EXISTS no_{op.lower()}_{tbl} BEFORE {op} ON {tbl} "
             f"BEGIN SELECT RAISE(ABORT, '{tbl} are append-only'); END;"
             for tbl in ("experiments", "events", "splits") for op in ("UPDATE", "DELETE")]


class RegistryError(RuntimeError):
    """Refused registry operation."""


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit (with '+dirty' when there are uncommitted changes), or None."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (sha + ("+dirty" if dirty else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


@dataclass
class Split:
    """Chronological development / validation / untouched test split of one research programme."""

    programme: str
    data_start: pd.Timestamp
    dev_end: pd.Timestamp
    val_end: pd.Timestamp
    data_end: pd.Timestamp

    def part(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """UTC ``[from, until)`` of ``development``, ``validation``, ``test``, ``dev+val`` or ``all``."""
        return {"development": (self.data_start, self.dev_end), "validation": (self.dev_end, self.val_end),
                "test": (self.val_end, self.data_end), "dev+val": (self.data_start, self.val_end),
                "all": (self.data_start, self.data_end)}[name]


class ExperimentRegistry:
    """The registry under ``root`` (normally ``data/experiments``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "registry.db"
        with self._conn() as c:
            c.executescript(_SCHEMA)
            for trig in _TRIGGERS:
                c.execute(trig)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Connection that commits on success, rolls back on error and is always closed."""
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    # ----------------------------------------------------------------- splits
    def register_split(self, split: Split, dataset_ids: dict | None = None, note: str = "") -> Split:
        """Store a programme's split. The first split of a programme is permanent; asking for a
        different one raises (so the test period can never quietly move)."""
        old = self.split(split.programme)
        if old is not None:
            if (old.data_start, old.dev_end, old.val_end, old.data_end) != (split.data_start, split.dev_end,
                                                                            split.val_end, split.data_end):
                raise RegistryError(f"programme {split.programme!r} already has a different split "
                                    f"(test from {old.val_end}); start a new programme name instead")
            return old
        with self._conn() as c:
            c.execute("INSERT INTO splits VALUES (?,?,?,?,?,?,?,?)",
                      (split.programme, _now(), str(split.data_start), str(split.dev_end), str(split.val_end),
                       str(split.data_end), json.dumps(dataset_ids or {}), note))
        return split

    def split(self, programme: str) -> Split | None:
        """The stored split of a programme, or None."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM splits WHERE programme=?", (programme,)).fetchone()
        if r is None:
            return None
        return Split(programme, *(pd.Timestamp(r[k]) for k in ("data_start", "dev_end", "val_end", "data_end")))

    def splits(self) -> pd.DataFrame:
        """All programmes and their split dates."""
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM splits ORDER BY created_utc", c)

    # ------------------------------------------------------------ experiments
    def _next_id(self, c: sqlite3.Connection) -> tuple[int, str]:
        n = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM experiments").fetchone()[0]
        return n, f"LAB-{n:04d}"

    def create(self, *, kind: str, strategies: list[dict], dataset_ids: dict, execution: dict, results: dict,
               artefacts: dict[str, str | bytes | pd.DataFrame], title: str = "", hypothesis: str = "",
               programme: str | None = None, data_part: str | None = None, date_from=None, date_to=None,
               test_set_used: bool = False, parent_id: str | None = None, status: str = "RECORDED",
               note: str = "", commit: str | None = None) -> str:
        """Insert an experiment and write its artefacts; returns the new id."""
        if status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        families = sorted({s.get("family") or s["id"] for s in strategies})
        with self._conn() as c:
            seq, eid = self._next_id(c)
            folder = self.root / eid
            if folder.exists():
                raise RegistryError(f"{folder} already exists; experiments are never overwritten")
            tmp = folder.with_name(eid + ".tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            for name, content in artefacts.items():
                p = tmp / name
                if isinstance(content, pd.DataFrame):
                    content.to_csv(p, index=False)
                elif isinstance(content, bytes):
                    p.write_bytes(content)
                else:
                    p.write_text(content, encoding="utf-8")
            record = {"id": eid, "kind": kind, "title": title, "hypothesis": hypothesis, "programme": programme,
                      "data_part": data_part, "date_from": None if date_from is None else str(date_from),
                      "date_to": None if date_to is None else str(date_to), "strategies": strategies,
                      "dataset_ids": dataset_ids, "execution": execution, "results": results,
                      "git_commit": commit, "test_set_used": bool(test_set_used), "parent_id": parent_id}
            (tmp / "experiment.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
            tmp.rename(folder)
            c.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (eid, seq, _now(), kind, title, hypothesis, programme, data_part, record["date_from"],
                       record["date_to"], json.dumps(strategies, default=str), json.dumps(families),
                       json.dumps(dataset_ids, default=str), json.dumps(execution, default=str),
                       json.dumps(results, default=str), commit, int(bool(test_set_used)), parent_id,
                       json.dumps(sorted(artefacts) + ["experiment.json"])))
            c.execute("INSERT INTO events (experiment_id, time_utc, kind, value) VALUES (?,?,?,?)",
                      (eid, _now(), "status", status))
            if note:
                c.execute("INSERT INTO events (experiment_id, time_utc, kind, value) VALUES (?,?,?,?)",
                          (eid, _now(), "note", note))
        return eid

    def add_note(self, eid: str, text: str) -> None:
        """Append a research note."""
        self._event(eid, "note", text)

    def set_status(self, eid: str, status: str) -> None:
        """Append a status change (the old status stays in the history)."""
        if status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        self._event(eid, "status", status)

    def _event(self, eid: str, kind: str, value: str) -> None:
        self.get(eid)
        with self._conn() as c:
            c.execute("INSERT INTO events (experiment_id, time_utc, kind, value) VALUES (?,?,?,?)",
                      (eid, _now(), kind, value))

    def get(self, eid: str) -> dict:
        """One experiment with its current status, notes and full history."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
            if r is None:
                raise RegistryError(f"no experiment {eid}")
            ev = [dict(x) for x in c.execute("SELECT time_utc, kind, value FROM events WHERE experiment_id=? ORDER BY seq",
                                             (eid,))]
        d = dict(r)
        for k in ("strategies", "families", "dataset_ids", "execution", "results", "artefacts"):
            d[k] = json.loads(d[k])
        d["test_set_used"] = bool(d["test_set_used"])
        d["status"] = [e["value"] for e in ev if e["kind"] == "status"][-1]
        d["notes"] = [e for e in ev if e["kind"] == "note"]
        d["history"] = ev
        d["folder"] = str(self.root / eid)
        return d

    def list(self) -> pd.DataFrame:
        """One row per experiment with headline net results."""
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM experiments ORDER BY seq")]
            st = {r["experiment_id"]: r["value"] for r in c.execute(
                "SELECT experiment_id, value FROM events WHERE kind='status' ORDER BY seq")}
        out = []
        for r in rows:
            res = json.loads(r["results"])
            net = res.get("net", {})
            out.append({"id": r["id"], "created_utc": r["created_utc"][:19], "kind": r["kind"], "title": r["title"],
                        "strategies": ",".join(f"{s['id']}@{s['version']}" for s in json.loads(r["strategies"])),
                        "programme": r["programme"], "part": r["data_part"], "test_set": bool(r["test_set_used"]),
                        "trades": net.get("trades"), "net_profit": net.get("net_profit"),
                        "profit_factor": net.get("profit_factor"), "expectancy_r": net.get("expectancy_r"),
                        "max_dd_pct": net.get("max_drawdown_pct"), "status": st.get(r["id"])})
        return pd.DataFrame(out)

    def compare(self, ids: list[str]) -> pd.DataFrame:
        """Net (and before-cost) metrics of several experiments side by side (one column each)."""
        cols = {}
        for eid in ids:
            e = self.get(eid)
            res = e["results"]
            col = {"strategies": ",".join(f"{s['id']}@{s['version']}" for s in e["strategies"]),
                   "part": e["data_part"], "period": f"{e['date_from']} to {e['date_to']}", "status": e["status"]}
            col.update({f"net {k}": v for k, v in res.get("net", {}).items()})
            col.update({f"before costs {k}": res.get("before_costs", {}).get(k)
                        for k in ("net_profit", "profit_factor", "expectancy_r")})
            cols[eid] = col
        return pd.DataFrame(cols)

    def test_uses(self, family: str, programme: str | None = None) -> int:
        """How many experiments already read the test period for a strategy family."""
        with self._conn() as c:
            rows = c.execute("SELECT families, programme FROM experiments WHERE test_set_used=1").fetchall()
        return sum(1 for r in rows if family in json.loads(r["families"]) and (programme is None or r["programme"] == programme))
