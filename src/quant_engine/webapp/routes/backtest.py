"""Backtest page: run a backtest on stored bars and browse saved reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...dashboard import services
from ..context import AppContext, md_html, records
from .deps import ctx

router = APIRouter(prefix="/api/backtests", tags=["backtest"])
MODES = {"london": "London breakout + H4 trend (live mode)", "regime": "3-mode switch (by market)", "compare": "Compare all modes x markets",
         "bullish": "Bullish: buy the dip", "bearish": "Bearish: sell the rally",
         "sideways": "Sideways: fade the range", "v1": "v1: sweep reversal"}
SOURCES = {"dukascopy": "Dukascopy (free history)", "mt5": "MT5 (downloaded)"}


class RunRequest(BaseModel):
    source: str = "mt5"
    mode: str = "london"


def _report(name: str) -> Path:
    for p in services.backtest_reports():
        if p.name == name:
            return p
    raise HTTPException(404, "report not found")


@router.get("")
def list_reports() -> dict:
    """Report folder names (newest first) plus the choices for a new run."""
    return {"reports": [p.name for p in services.backtest_reports()], "modes": MODES, "sources": SOURCES}


@router.post("/run")
def run(req: RunRequest, c: AppContext = Depends(ctx)) -> dict:
    """Run a backtest now and return the new report's name."""
    if req.mode not in MODES or req.source not in SOURCES:
        raise HTTPException(400, "unknown mode or data source")
    try:
        out = services.run_backtest_now(c.cfg, c.store, req.source, c.point, req.mode)
    except Exception as exc:
        raise HTTPException(500, f"Running backtest failed: {exc}") from exc
    return {"report": out.name}


@router.get("/{name}")
def report(name: str) -> dict:
    """One report: summary, regime table, equity curve, trades and markdown."""
    p = _report(name)
    summary = json.loads((p / "summary.json").read_text())
    out: dict = {"name": name, "comparison": summary.get("comparison"), "overall": summary.get("overall"),
                 "by_regime": summary.get("by_trend_regime", []), "equity": [], "trades": []}
    tcsv = p / "trades.csv"
    if tcsv.exists():
        tr = pd.read_csv(tcsv)
        f = tr[tr["filled"]] if "filled" in tr else tr
        out["equity"] = [round(float(v), 3) for v in f["r"].cumsum()] if "r" in f else []
        out["trades"] = records(tr)
    md = p / "report.md"
    out["report_html"] = md_html(md.read_text(encoding="utf-8")) if md.exists() else ""
    return out
