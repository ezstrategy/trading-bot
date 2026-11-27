"""FastAPI dependency giving routes the shared :class:`AppContext`."""
from __future__ import annotations

from fastapi import Request

from ..context import AppContext


def ctx(request: Request) -> AppContext:
    """The app's shared context."""
    return request.app.state.ctx
