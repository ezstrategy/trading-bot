"""News page: upcoming signal-blocking windows from the economic calendar."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ...dashboard import services
from ..context import AppContext, records
from .deps import ctx

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("")
def news(refresh: bool = False, c: AppContext = Depends(ctx)) -> list[dict]:
    """News blocks that have not ended yet (refresh=true downloads the calendar first)."""
    try:
        df = services.news_table(c.cfg, pd.Timestamp.now(tz="UTC"), refresh=refresh)
    except Exception as exc:
        raise HTTPException(502, f"Calendar problem: {exc}") from exc
    return records(df)
