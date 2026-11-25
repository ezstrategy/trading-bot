"""The engine must never be able to place a trade."""
import re
from pathlib import Path

import pytest

from quant_engine.mt5.safety import ForbiddenOperationError, SafeMT5

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("fn", ["order_send", "order_check"])
def test_safe_proxy_blocks_trading(fake, fn):
    with pytest.raises(ForbiddenOperationError):
        getattr(SafeMT5(fake), fn)
    assert not fake.order_send_called


def test_client_cannot_send_orders(client, fake):
    with pytest.raises(ForbiddenOperationError):
        client.mt5.order_send({"action": 1})
    with pytest.raises(ForbiddenOperationError):
        client.call("order_send", {"action": 1})
    assert not fake.order_send_called


def test_proxy_is_read_only(fake):
    with pytest.raises(ForbiddenOperationError):
        SafeMT5(fake).order_send = lambda r: None


def test_read_functions_pass_through(fake):
    assert SafeMT5(fake).TIMEFRAME_M5 == 5


def test_source_never_calls_order_send():
    pattern = re.compile(r"(\.order_send\b|order_send\s*\(|order_check\s*\()")
    offenders = [f"{p}:{n}" for p in SRC.rglob("*.py")
                 for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if pattern.search(line)]
    assert offenders == []
