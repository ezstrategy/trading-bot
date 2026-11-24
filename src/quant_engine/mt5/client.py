"""MetaTrader 5 terminal connection with retry and graceful reconnection."""
from __future__ import annotations

import time
from types import ModuleType
from typing import Any, Callable

from ..config import ConnectionConfig, MT5Credentials
from ..logging_setup import get_logger
from .safety import SafeMT5

log = get_logger(__name__)

#: MetaTrader5 last_error() codes that mean the terminal link was lost.
CONNECTION_ERROR_CODES = frozenset({-10000, -10001, -10002, -10003, -10004, -10005})
AUTH_FAILED = -6


class MT5Error(RuntimeError):
    """An MT5 call failed. ``code`` is the terminal's last_error code."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message if code is None else f"{message} (mt5 error {code})")
        self.code = code


class MT5NotAvailableError(MT5Error):
    """The MetaTrader5 Python package or terminal is not available."""


def _import_mt5() -> ModuleType:
    try:
        import MetaTrader5  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise MT5NotAvailableError(
            "The MetaTrader5 package is not installed. It only runs on Windows with the "
            "MT5 terminal installed: pip install MetaTrader5") from exc
    return MetaTrader5


class MT5Client:
    """Owns the terminal session and wraps every MT5 call.

    * Credentials come from :class:`MT5Credentials` (environment only).
    * All calls go through :class:`SafeMT5`, so trading functions are blocked.
    * If a call fails because the terminal link dropped, the client reconnects
      with exponential backoff and retries the call.

    Args:
        conn: retry/timeout settings.
        credentials: account login. If incomplete, the terminal's currently
            logged-in account is used.
        mt5_module: injectable module for tests; defaults to ``MetaTrader5``.
        sleep: injectable sleep function for tests.
    """

    def __init__(self, conn: ConnectionConfig, credentials: MT5Credentials | None = None,
                 mt5_module: Any | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.conn = conn
        self.credentials = credentials or MT5Credentials()
        self._raw = mt5_module
        self._mt5: SafeMT5 | None = SafeMT5(mt5_module) if mt5_module is not None else None
        self._sleep = sleep
        self.connected = False

    # ----------------------------------------------------------- lifecycle
    @property
    def mt5(self) -> SafeMT5:
        """The guarded MetaTrader5 module."""
        if self._mt5 is None:
            self._raw = _import_mt5()
            self._mt5 = SafeMT5(self._raw)
        return self._mt5

    def _initialize_once(self) -> None:
        kwargs: dict[str, Any] = {"timeout": self.conn.timeout_ms}
        c = self.credentials
        if c.path:
            kwargs["path"] = c.path
        if c.complete:
            kwargs.update(login=c.login, password=c.password, server=c.server)
        if not self.mt5.initialize(**kwargs):
            code, msg = self._last_error()
            self.mt5.shutdown()
            raise MT5Error(f"initialize failed: {msg}", code)
        info = self.mt5.terminal_info()
        if info is None or not getattr(info, "connected", False):
            self.mt5.shutdown()
            raise MT5Error("terminal started but is not connected to the broker server", -10004)

    def connect(self) -> None:
        """Start (or attach to) the MT5 terminal, retrying with backoff.

        Raises:
            MT5Error: when all attempts fail, or immediately on bad credentials.
        """
        delay = self.conn.retry_backoff_seconds
        for attempt in range(1, self.conn.max_retries + 1):
            try:
                self._initialize_once()
                self.connected = True
                acct = self.mt5.account_info()
                log.info("mt5_connected", extra={
                    "attempt": attempt,
                    "server": getattr(acct, "server", None),
                    "login": getattr(acct, "login", None),
                    "trade_mode": getattr(acct, "trade_mode", None)})
                return
            except MT5Error as exc:
                self.connected = False
                if exc.code == AUTH_FAILED:
                    log.error("mt5_auth_failed", extra={"server": self.credentials.server})
                    raise
                log.warning("mt5_connect_failed", extra={"attempt": attempt, "error": str(exc),
                                                         "retry_in_s": delay})
                if attempt == self.conn.max_retries:
                    raise MT5Error(f"could not connect after {attempt} attempts: {exc}", exc.code) from exc
                self._sleep(delay)
                delay = min(delay * 2, self.conn.retry_backoff_max_seconds)

    def disconnect(self) -> None:
        """Shut down the terminal session (safe to call twice)."""
        if self._mt5 is not None:
            try:
                self.mt5.shutdown()
            finally:
                self.connected = False
                log.info("mt5_disconnected")

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # --------------------------------------------------------------- calls
    def _last_error(self) -> tuple[int | None, str]:
        try:
            err = self.mt5.last_error()
            return int(err[0]), str(err[1])
        except Exception:  # pragma: no cover - defensive
            return None, "unknown error"

    def is_alive(self) -> bool:
        """True if the terminal is reachable and connected to the broker."""
        try:
            info = self.mt5.terminal_info()
        except Exception:
            return False
        return info is not None and bool(getattr(info, "connected", False))

    def call(self, fn: str, *args: Any, allow_none: bool = False, **kwargs: Any) -> Any:
        """Call ``mt5.<fn>`` with automatic reconnection.

        Args:
            fn: name of the MetaTrader5 function (trading functions are blocked).
            allow_none: return None instead of raising when MT5 returns None for
                a reason other than a lost connection (e.g. no data in range).

        Raises:
            MT5Error: if the call keeps failing.
        """
        if not self.connected:
            self.connect()
        func = getattr(self.mt5, fn)
        for attempt in range(1, self.conn.max_retries + 1):
            result = func(*args, **kwargs)
            if result is not None:
                return result
            code, msg = self._last_error()
            lost = code in CONNECTION_ERROR_CODES or not self.is_alive()
            if not lost:
                if allow_none:
                    log.debug("mt5_call_empty", extra={"fn": fn, "code": code, "error": msg})
                    return None
                raise MT5Error(f"mt5.{fn} failed: {msg}", code)
            log.warning("mt5_connection_lost", extra={"fn": fn, "attempt": attempt, "code": code})
            self.connected = False
            if attempt == self.conn.max_retries:
                break
            self.connect()
        raise MT5Error(f"mt5.{fn} failed after reconnect attempts", -10004)
