"""Signals page: every signal the live engine sent and the message log."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...dashboard import services
from ...live.state import read_messages
from ..context import AppContext, records
from .deps import ctx

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("")
def signals(c: AppContext = Depends(ctx)) -> dict:
    """Signals (newest first), counts and the last 100 messages."""
    d = c.cfg.engine.live.state_dir
    sig = services.signals_table(d)
    closed = sig[sig["r"].notna()] if len(sig) else sig
    return {"signals": records(sig), "messages": records(read_messages(d, 100)),
            "stats": {"sent": int(len(sig)), "closed": int(len(closed)),
                      "total_r": round(float(closed["r"].sum()), 2) if len(closed) else 0.0}}
