"""Market page: context cards, annotated chart and liquidity tables."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ...dashboard import services
from ...dashboard.chart import build_chart
from ...timeframes import Timeframe
from ..context import AppContext
from .deps import ctx

router = APIRouter(prefix="/api/market", tags=["market"])
OVERLAYS = ("kill_zones", "pools", "fvgs", "sweeps", "structure", "range")


@router.get("")
def market(tf: str = "M5", lookback: int = 180, overlays: str = ",".join(OVERLAYS), tz: str = "UTC",
           at: str | None = None, point: float | None = None, c: AppContext = Depends(ctx)) -> dict:
    """Snapshot at ``at`` (UTC ISO; default: close of the last stored M5 candle)."""
    if not c.has_data():
        return {"ready": False, "message": "No market data stored yet. Open MT5, then download history on the Data page."}
    try:
        chart_tf = Timeframe.parse(tf)
        when = pd.Timestamp(at) if at else None
        if when is not None:
            when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    lookback = max(20, min(int(lookback), 1500))
    try:
        bars, feats = c.features(point or c.point)
        snap = feats.snapshot(when or services.default_decision_time(bars))
    except ValueError as exc:
        return {"ready": False, "message": str(exc)}
    show = {o for o in overlays.split(",") if o in OVERLAYS}
    fig = build_chart(feats, snap, chart_tf, lookback, show, tz)
    fig.update_layout(title=None)
    cards = [{"label": lab, "value": val, "tone": tone} for lab, val, tone in services.market_status(snap)]
    return {"ready": True, "decision_time": str(snap.decision_time), "cards": cards,
            "summary": json.loads(json.dumps(services.snapshot_summary(snap), default=str)),
            "figure": json.loads(fig.to_json())}
