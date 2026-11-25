"""Telegram delivery through the Bot API (token and chat id from the environment only)."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Callable, Protocol

from ..logging_setup import get_logger

log = get_logger(__name__)


class Notifier(Protocol):
    def send(self, text: str, reply_to: int | None = None) -> int | None:
        """Send ``text`` (HTML); returns the message id, or None if it was not delivered."""


class TelegramNotifier:
    """Sends HTML messages to one chat. Follow-ups reply to the signal message."""

    API = "https://api.telegram.org"

    def __init__(self, token: str, chat_id: str, post: Callable[[str, dict], dict] | None = None,
                 retries: int = 3, backoff: float = 2.0) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram token and chat id are required")
        self._token, self.chat_id = token, chat_id
        self._post = post or self._http_post
        self.retries, self.backoff = retries, backoff

    @classmethod
    def from_env(cls, **kw) -> "TelegramNotifier | None":
        """Build from TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, or None when they are not set."""
        token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(token, chat, **kw) if token and chat else None

    def _http_post(self, method: str, data: dict) -> dict:
        url = f"{self.API}/bot{self._token}/{method}"
        body = urllib.parse.urlencode(data).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:      # Telegram explains errors in the body
            return json.loads(exc.read().decode() or "{}") or {"ok": False, "description": str(exc)}

    def send(self, text: str, reply_to: int | None = None) -> int | None:
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}
        if reply_to:
            data["reply_to_message_id"] = str(reply_to)
            data["allow_sending_without_reply"] = "true"
        for attempt in range(self.retries):
            try:
                res = self._post("sendMessage", data)
            except Exception as exc:   # network error: retry (the token is never logged)
                log.warning("telegram_retry", extra={"attempt": attempt + 1, "error": type(exc).__name__})
                time.sleep(self.backoff * (attempt + 1))
                continue
            if res.get("ok"):
                return int(res["result"]["message_id"])
            log.error("telegram_rejected", extra={"description": str(res.get("description", ""))[:200]})
            return None
        return None


class ConsoleNotifier:
    """Dry run: hands each message to ``out`` (the CLI prints it) instead of Telegram."""

    def __init__(self, out: Callable[[str], None]) -> None:
        self.out, self._n = out, 0

    def send(self, text: str, reply_to: int | None = None) -> int | None:
        self._n += 1
        self.out(("  ↳ " if reply_to else "") + text)
        return self._n
