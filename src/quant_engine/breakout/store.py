"""Permanent record of every breakout signal (SQLite, append-only history).

Two tables:

* ``signals``: one row per signal key (``strategy:version:breakout-day``; UNIQUE, so a breakout
  can never be stored twice). It holds the signal as generated (levels, ATR, spread, costs,
  suggested size, checks, explanation, market-data snapshot), the theoretical **paper** outcome,
  the owner's **manual** decision and execution, and the current display status.
* ``events``: every change to a signal (generated, entry known, paper exit, approved, rejected,
  executed, exit recorded), with UTC time and details. Rows are never updated or deleted.

There is no delete function. Theoretical (paper) results and actually executed trades are kept in
separate columns and reported separately.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

# display statuses (owner spec section 5)
WAITING, BUY, AWAITING, ACTIVE = "Waiting", "BUY signal", "Awaiting manual approval", "Active"
TP, SL, EXPIRED, CANCELLED = "TP hit", "SL hit", "Expired", "Cancelled"
FINAL_LABEL = {"tp_hit": TP, "sl_hit": SL, "expired": EXPIRED}
STATUSES = [WAITING, BUY, AWAITING, ACTIVE, TP, SL, EXPIRED, CANCELLED]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  key TEXT PRIMARY KEY,
  source TEXT NOT NULL,              -- paper | live
  strategy_id TEXT NOT NULL, strategy_name TEXT, version TEXT NOT NULL, rule_hash TEXT,
  side TEXT NOT NULL, day TEXT NOT NULL, signal_time TEXT NOT NULL, created_at TEXT NOT NULL,
  breakout_level REAL, d1_close REAL, atr REAL,
  entry_est REAL, stop_est REAL, target_est REAL,
  entry_time TEXT, entry REAL, stop REAL, target REAL, rr REAL,
  spread_at_signal REAL, spread_entry REAL, est_costs TEXT,
  suggested_lots REAL, risk_pct REAL, risk_money REAL, equity REAL, risk_reason TEXT,
  generated INTEGER NOT NULL,        -- 1 = all checks passed; 0 = breakout recorded but blocked
  block_reason TEXT,
  paper_status TEXT, paper_exit_time TEXT, paper_exit REAL, paper_r REAL,
  approval TEXT NOT NULL DEFAULT 'pending', approval_time TEXT, approval_note TEXT,
  executed INTEGER NOT NULL DEFAULT 0, exec_time TEXT, exec_entry REAL, exec_lots REAL,
  exec_exit_time TEXT, exec_exit REAL, exec_result TEXT, exec_r REAL, exec_pnl REAL, exec_note TEXT,
  status TEXT NOT NULL, explanation TEXT, checks TEXT, snapshot TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, time TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS signals_no_delete BEFORE DELETE ON signals BEGIN SELECT RAISE(ABORT, 'signals are permanent'); END;
"""


class StoreError(ValueError):
    """An action that is not allowed for the signal's current state."""


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def derive_status(r: dict) -> str:
    """Display status from the stored fields."""
    if not r.get("generated") or (r.get("approval") == "rejected" and not r.get("executed")) \
            or (r.get("risk_reason") not in (None, "", "ok") and not r.get("executed")):
        return CANCELLED
    if r.get("executed"):
        return r.get("exec_result") or ACTIVE
    ps = r.get("paper_status")
    if ps in FINAL_LABEL:
        return FINAL_LABEL[ps]
    if not r.get("entry_time"):
        return BUY
    if r.get("approval") == "approved":
        return ACTIVE
    return AWAITING


class SignalStore:
    """SQLite file holding signals and their history (thread-safe for the web app)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------- reads
    def get(self, key: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE key = ?", (key,)).fetchone()
        return None if r is None else self._decode(dict(r))

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def all(self, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM signals ORDER BY signal_time DESC, key LIMIT ?", (limit,)).fetchall()
        return [self._decode(dict(r)) for r in rows]

    def events(self, key: str | None = None, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM events" + (" WHERE key = ?" if key else "") + " ORDER BY id DESC LIMIT ?"
            rows = c.execute(q, ((key, limit) if key else (limit,))).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"] or "{}")} for r in rows]

    def open_signals(self) -> list[dict]:
        """Signals whose paper trade or manual trade is not finished."""
        return [r for r in self.all(5000) if r["status"] in (BUY, AWAITING, ACTIVE)]

    @staticmethod
    def _decode(r: dict) -> dict:
        for k in ("checks", "snapshot", "est_costs"):
            if r.get(k):
                r[k] = json.loads(r[k])
        return r

    # ------------------------------------------------------------ writes
    def _event(self, c: sqlite3.Connection, key: str, kind: str, detail: dict, when: str | None = None) -> None:
        c.execute("INSERT INTO events (key, time, kind, detail) VALUES (?, ?, ?, ?)",
                  (key, when or _now(), kind, json.dumps(detail, default=str)))

    def insert(self, row: dict) -> bool:
        """Store a new signal; returns False (and changes nothing) when the key already exists."""
        row = dict(row)
        row.setdefault("created_at", _now())
        row["status"] = derive_status(row)
        row["updated_at"] = row["created_at"]
        for k in ("checks", "snapshot", "est_costs"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], default=str)
        cols = list(row)
        with self._lock, self._conn() as c:
            try:
                c.execute(f"INSERT INTO signals ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                          [row[k] for k in cols])
            except sqlite3.IntegrityError:
                return False
            self._event(c, row["key"], "generated" if row.get("generated") else "blocked",
                        {"status": row["status"], "reason": row.get("block_reason") or row.get("risk_reason")},
                        row["created_at"])
        return True

    def _update(self, key: str, changes: dict, kind: str, detail: dict) -> dict:
        with self._lock, self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE key = ?", (key,)).fetchone()
            if r is None:
                raise KeyError(f"no signal {key!r}")
            cur = {**dict(r), **changes}
            cur["status"] = derive_status(cur)
            cur["updated_at"] = _now()
            sets = {**changes, "status": cur["status"], "updated_at": cur["updated_at"]}
            c.execute(f"UPDATE signals SET {', '.join(f'{k} = ?' for k in sets)} WHERE key = ?",
                      [*sets.values(), key])
            self._event(c, key, kind, {**detail, "status": cur["status"]})
        return self.get(key)  # type: ignore[return-value]

    def update_paper(self, key: str, trade: dict) -> dict | None:
        """Apply the newest theoretical trade state (entry known, exit) from a scan."""
        r = self.get(key)
        if r is None:
            return None
        ch = {}
        if trade.get("entry_time") and not r.get("entry_time"):
            ch.update({k: trade[k] for k in ("entry_time", "entry", "stop", "target")})
            ch["spread_entry"] = trade.get("spread")
        if trade.get("status") != r.get("paper_status"):
            ch["paper_status"] = trade.get("status")
            if trade.get("status") in FINAL_LABEL:
                ch.update({"paper_exit_time": trade.get("exit_time"), "paper_exit": trade.get("exit"),
                           "paper_r": trade.get("r")})
        if not ch:
            return r
        kind = "paper_exit" if ch.get("paper_status") in FINAL_LABEL else ("entry_known" if "entry_time" in ch else "paper_update")
        return self._update(key, ch, kind, {k: v for k, v in ch.items()})

    def approve(self, key: str, note: str = "") -> dict:
        r = self._require(key)
        if r["approval"] != "pending" or r["status"] == CANCELLED:
            raise StoreError(f"signal is {r['status']} / {r['approval']}: cannot approve")
        return self._update(key, {"approval": "approved", "approval_time": _now(), "approval_note": note},
                            "approved", {"note": note})

    def reject(self, key: str, note: str = "") -> dict:
        r = self._require(key)
        if r["executed"] or r["approval"] == "rejected":
            raise StoreError("signal already executed or rejected")
        return self._update(key, {"approval": "rejected", "approval_time": _now(), "approval_note": note},
                            "rejected", {"note": note})

    def mark_executed(self, key: str, entry: float, lots: float, when: str | None = None, note: str = "") -> dict:
        r = self._require(key)
        if r["approval"] != "approved":
            raise StoreError("approve the signal before marking it executed")
        if r["executed"]:
            raise StoreError("already marked as executed")
        if not (entry > 0 and lots > 0):
            raise StoreError("entry price and lot size must be positive")
        return self._update(key, {"executed": 1, "exec_entry": float(entry), "exec_lots": float(lots),
                                  "exec_time": when or _now(), "exec_note": note},
                            "executed", {"entry": entry, "lots": lots, "time": when, "note": note})

    def record_exit(self, key: str, exit_price: float, result: str, when: str | None = None,
                    money_per_price_unit_lot: float = 100.0, note: str = "") -> dict:
        """Record the actual exit of a manually executed trade. R uses the signal's stop distance
        measured from the actual entry."""
        r = self._require(key)
        if not r["executed"]:
            raise StoreError("mark the signal as executed first")
        if r["exec_exit"] is not None:
            raise StoreError("exit already recorded")
        if result not in (TP, SL, EXPIRED, CANCELLED, "Closed manually"):
            raise StoreError(f"result must be one of TP hit, SL hit, Expired, Closed manually, Cancelled")
        stop = r["stop"] if r["stop"] is not None else r["stop_est"]
        risk = (r["exec_entry"] - stop) if stop is not None else None
        rr = (exit_price - r["exec_entry"]) / risk if risk and risk > 0 else None
        pnl = (exit_price - r["exec_entry"]) * r["exec_lots"] * money_per_price_unit_lot
        return self._update(key, {"exec_exit": float(exit_price), "exec_exit_time": when or _now(),
                                  "exec_result": result, "exec_r": rr, "exec_pnl": round(pnl, 2),
                                  "exec_note": (r.get("exec_note") or "") + (f" | {note}" if note else "")},
                            "exit_recorded", {"exit": exit_price, "result": result, "r": rr, "pnl": pnl})

    def _require(self, key: str) -> dict:
        r = self.get(key)
        if r is None:
            raise KeyError(f"no signal {key!r}")
        return r

    # ----------------------------------------------------------- reports
    def performance(self) -> dict:
        """Separate summaries: paper (theoretical, every generated signal) and manually executed."""
        rows = self.all(100000)
        gen = [r for r in rows if r["generated"]]
        paper = [r["paper_r"] for r in gen if r["paper_status"] in FINAL_LABEL and r["paper_r"] is not None]
        manual = [r["exec_r"] for r in rows if r["executed"] and r["exec_r"] is not None]

        def s(x: list[float]) -> dict:
            if not x:
                return {"trades": 0}
            w = [v for v in x if v > 0]
            neg = -sum(v for v in x if v <= 0)
            return {"trades": len(x), "win_rate_pct": round(100 * len(w) / len(x), 1),
                    "net_r": round(sum(x), 2), "expectancy_r": round(sum(x) / len(x), 3),
                    "profit_factor": round(sum(w) / neg, 2) if neg > 0 else None}
        by_src = {}
        for src in sorted({r["source"] for r in gen}):
            by_src[src] = s([r["paper_r"] for r in gen if r["source"] == src and r["paper_status"] in FINAL_LABEL
                             and r["paper_r"] is not None])
        return {"paper": s(paper), "paper_by_source": by_src, "manual": s(manual),
                "signals": len(gen), "blocked": len(rows) - len(gen),
                "open": sum(r["status"] in (BUY, AWAITING, ACTIVE) for r in rows)}
