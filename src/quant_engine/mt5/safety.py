"""Hard guard that makes order execution impossible.

The engine is SIGNALS ONLY. Every access to the MetaTrader5 module goes
through :class:`SafeMT5`, which refuses any trading function. This is a
second line of defence on top of never writing such calls in the code
(a test also scans the source tree for them).
"""
from __future__ import annotations

from typing import Any

#: Functions of the MetaTrader5 package that submit or pre-validate trade requests.
FORBIDDEN = frozenset({"order_send", "order_check"})


class ForbiddenOperationError(RuntimeError):
    """Raised when code tries to use an order/trading function."""


class SafeMT5:
    """Read-only proxy around the ``MetaTrader5`` module."""

    def __init__(self, module: Any) -> None:
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN or name.startswith("order_send"):
            raise ForbiddenOperationError(
                f"mt5.{name} is blocked: this engine is signals-only and never trades")
        return getattr(object.__getattribute__(self, "_module"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise ForbiddenOperationError("SafeMT5 is read-only")
