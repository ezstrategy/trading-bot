"""Breakout page: long-only D1 breakout strategies (state, signals, manual workflow, backtest).

Signals only. The manual actions only change the local signal record; nothing is sent to MT5.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...breakout.backtest import run_backtest, write_report
from ...breakout.commands import spec_for
from ...breakout.config import BreakoutConfig, load_breakout_config
from ...breakout.daily import NY
from ...breakout.rules import strategies
from ...breakout.scan import prepare, scan
from ...breakout.service import state_path
from ...breakout.backtest import costs_for
from ...breakout.store import STATUSES, SignalStore, StoreError
from ...timeframes import Timeframe
from ..context import AppContext, md_html
from .deps import ctx

router = APIRouter(prefix="/api/breakout", tags=["breakout"])
_bt_lock = threading.Lock()


def _cfg(c: AppContext) -> tuple[BreakoutConfig, str | None]:
    d = c.config_dir or os.getenv("QE_CONFIG_DIR", "config")
    return load_breakout_config(d), d


def _store(b: BreakoutConfig) -> SignalStore:
    return SignalStore(b.store.path)


def _broker_offset(c: AppContext) -> dict:
    """Broker (server) time offset for display: from the MT5 check when known, else NY+7 (GUESS)."""
    info = c.mt5_info or {}
    if info.get("server_offset_hours") is not None:
        return {"hours": info["server_offset_hours"], "source": "MT5 server clock", "ny_plus_7": False}
    ny = pd.Timestamp.now(tz=NY)
    return {"hours": ny.utcoffset().total_seconds() / 3600 + 7, "source": "GUESS: New York + 7h (NY-close broker)",
            "ny_plus_7": True}


def _state_from_store(c: AppContext, b: BreakoutConfig, cdir: str | None) -> dict | None:
    """Strategy states computed from the stored M5 bars when the paper service has not run."""
    last = c.store.last_bar_time(c.canonical, Timeframe.M5)
    if last is None:
        return None
    m5 = c.store.read_bars(c.canonical, Timeframe.M5, last - pd.Timedelta(days=150))
    if m5.empty:
        return None
    spec = spec_for(cdir, b)
    d1, m = prepare(m5, b, c.point)
    costs, swap = costs_for(b, spec)
    return {"time": str(m5["time"].iat[-1] + Timeframe.M5.delta), "mode": "computed from stored bars",
            "swap": swap, "strategies": [scan(s, d1, m, b, costs).current() for s in strategies(b)]}


@router.get("")
def overview(c: AppContext = Depends(ctx)) -> dict:
    """Config, strategy states, stored signals and the separate performance summaries."""
    b, cdir = _cfg(c)
    p = state_path(b)
    state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else _state_from_store(c, b, cdir)
    st = _store(b)
    return {"mode": b.mode, "review_approved_by": b.live_review_approved_by, "statuses": STATUSES,
            "risk": b.risk.model_dump(), "strategies": [s.describe() for s in strategies(b, only_enabled=False)],
            "state": state, "signals": st.all(300), "performance": st.performance(),
            "broker_offset": _broker_offset(c), "backtest": _latest_summary(b)}


@router.get("/signals/{key}")
def signal(key: str, c: AppContext = Depends(ctx)) -> dict:
    b, _ = _cfg(c)
    st = _store(b)
    r = st.get(key)
    if r is None:
        raise HTTPException(404, f"no signal {key}")
    return {"signal": r, "events": st.events(key)}


class Note(BaseModel):
    note: str = ""


class Execution(BaseModel):
    entry: float = Field(gt=0)
    lots: float = Field(gt=0)
    time: str | None = None
    note: str = ""


class Exit(BaseModel):
    exit: float = Field(gt=0)
    result: str
    time: str | None = None
    note: str = ""


def _act(fn, *a, **kw) -> dict:
    try:
        return fn(*a, **kw)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except StoreError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/signals/{key}/approve")
def approve(key: str, body: Note, c: AppContext = Depends(ctx)) -> dict:
    b, _ = _cfg(c)
    return _act(_store(b).approve, key, body.note)


@router.post("/signals/{key}/reject")
def reject(key: str, body: Note, c: AppContext = Depends(ctx)) -> dict:
    b, _ = _cfg(c)
    return _act(_store(b).reject, key, body.note)


@router.post("/signals/{key}/execute")
def execute(key: str, body: Execution, c: AppContext = Depends(ctx)) -> dict:
    b, _ = _cfg(c)
    return _act(_store(b).mark_executed, key, body.entry, body.lots, body.time, body.note)


@router.post("/signals/{key}/exit")
def record_exit(key: str, body: Exit, c: AppContext = Depends(ctx)) -> dict:
    b, cdir = _cfg(c)
    mpu = spec_for(cdir, b).money_per_price_unit(1.0)
    return _act(_store(b).record_exit, key, body.exit, body.result, body.time, mpu, body.note)


# ------------------------------------------------------------------ backtest
def _reports(b: BreakoutConfig) -> list[Path]:
    root = Path(b.reports_dir)
    return sorted([p for p in root.glob("*") if (p / "results.json").exists()], reverse=True) if root.exists() else []


def _latest_summary(b: BreakoutConfig) -> dict | None:
    reps = _reports(b)
    if not reps:
        return None
    res = json.loads((reps[0] / "results.json").read_text(encoding="utf-8"))
    return {"folder": str(reps[0]), "data": res["data"], "swap": res["swap_source"],
            "per_strategy": {k: {"main_window": v["main_window"], "full_history": v["full_history"]}
                             for k, v in res["results"]["per_strategy"].items()}}


@router.get("/backtest")
def backtest(folder: str | None = None, c: AppContext = Depends(ctx)) -> dict:
    """A saved backtest report (newest by default): tables, report HTML and equity curves in R."""
    b, _ = _cfg(c)
    reps = _reports(b)
    if not reps:
        return {"report": None, "reports": []}
    pick = next((p for p in reps if p.name == folder), reps[0])
    res = json.loads((pick / "results.json").read_text(encoding="utf-8"))
    eq = {}
    for f in pick.glob("equity_*.csv"):
        df = pd.read_csv(f)
        eq[f.stem.removeprefix("equity_")] = {"t": df["time"].astype(str).tolist(), "r": df["cum_r"].round(3).tolist()}
    md = (pick / "report.md").read_text(encoding="utf-8") if (pick / "report.md").exists() else ""
    return {"report": {"folder": pick.name, "results": res["results"], "data": res["data"], "swap": res["swap_source"],
                       "strategies": res["strategies"], "html": md_html(md), "equity": eq},
            "reports": [p.name for p in reps]}


@router.post("/backtest")
def run_bt(c: AppContext = Depends(ctx)) -> dict:
    """Run the backtest on the stored M5 bars (all of them) and save a new report."""
    b, cdir = _cfg(c)
    m5 = c.store.read_bars(c.canonical, Timeframe.M5)
    if m5.empty:
        raise HTTPException(400, "No stored M5 candles. Download MT5 history on the Data page first.")
    if not _bt_lock.acquire(blocking=False):
        raise HTTPException(409, "A backtest is already running.")
    try:
        from ...data.quality import validate_bars
        q = validate_bars(m5, Timeframe.M5, c.cfg.data.quality, dataset="M5")
        spec = spec_for(cdir, b)
        info = {"dataset": f"stored MT5 bars {c.store.bars_path(c.canonical, Timeframe.M5)}",
                "quality": ("OK" if q.ok else "ISSUES REPORTED (bars used as they are)") + f" {q.counts()}",
                "instrument_profile": spec.name, "instrument_guesses": spec.guesses()}
        run = run_backtest(m5, b, spec, c.point, info)
        folder = Path(b.reports_dir) / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
        write_report(run, folder)
    finally:
        _bt_lock.release()
    return backtest(folder.name, c)
