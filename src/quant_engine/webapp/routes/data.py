"""Data page: MT5 connection check, history download and data quality."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...dashboard import services
from ...features.context import STRUCTURE_TFS
from ...timeframes import Timeframe
from ..context import AppContext, records
from .deps import ctx

router = APIRouter(prefix="/api", tags=["data"])


class DownloadRequest(BaseModel):
    days: int = Field(30, ge=1, le=730)
    timeframes: list[str] | None = None


@router.get("/status")
def status(c: AppContext = Depends(ctx)) -> dict:
    """App-wide facts the front end shows in its header."""
    e = c.cfg.engine
    return {"instrument": c.canonical, "execution_mode": e.execution_mode, "has_data": c.has_data(),
            "point": c.point, "mt5": c.mt5_info,
            "timeframes": [t.value for t in e.timeframes.collect], "default_days": c.cfg.data.download.default_days,
            "risk_pct": e.risk.per_trade_pct}


@router.get("/data/quality")
def quality(c: AppContext = Depends(ctx)) -> list[dict]:
    """Rows, first/last candle, age and validation result per timeframe."""
    df = services.data_status(c.cfg, c.store)
    for col in ("first", "last"):
        df[col] = df[col].astype(str).replace({"None": None, "NaT": None})
    return records(df)


@router.post("/mt5/check")
def check(c: AppContext = Depends(ctx)) -> dict:
    """Connect to MT5, read symbol info and server clock, disconnect."""
    try:
        with c.mt5_lock:
            info = services.check_connection(c.cfg)
    except Exception as exc:  # shown in the UI, never crashes the app
        raise HTTPException(502, f"Connecting to MT5 failed: {exc}") from exc
    c.mt5_info = info
    c.point = float(info.get("point") or c.point)
    return info


@router.post("/mt5/download")
def download(req: DownloadRequest, c: AppContext = Depends(ctx)) -> list[dict]:
    """Download (or top up) history from MT5."""
    try:
        tfs = [Timeframe.parse(t) for t in (req.timeframes or [t.value for t in c.cfg.engine.timeframes.collect])]
        with c.mt5_lock:
            res = services.download(c.cfg, c.store, tfs, req.days)
    except Exception as exc:
        raise HTTPException(502, f"Downloading from MT5 failed: {exc}") from exc
    return [{"timeframe": r.dataset.split()[-1], "rows_stored": r.rows_stored} for r in res]


@router.post("/mt5/refresh")
def refresh(c: AppContext = Depends(ctx)) -> dict:
    """Live mode: fetch the newest candles of the structure timeframes (last 2 days)."""
    try:
        with c.mt5_lock:
            services.download(c.cfg, c.store, list(STRUCTURE_TFS), 2)
    except Exception as exc:
        raise HTTPException(502, f"Refreshing from MT5 failed: {exc}") from exc
    return {"ok": True}
