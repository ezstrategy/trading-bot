import pytest

from quant_engine.config import ConnectionConfig, MT5Credentials
from quant_engine.mt5.client import MT5Client, MT5Error


def make(fake, **kw):
    sleeps = []
    conn = ConnectionConfig(max_retries=kw.pop("retries", 4), retry_backoff_seconds=1, retry_backoff_max_seconds=3)
    c = MT5Client(conn, kw.pop("creds", None), mt5_module=fake, sleep=sleeps.append)
    return c, sleeps


def test_connect_ok(fake):
    c, sleeps = make(fake)
    c.connect()
    assert c.connected and c.is_alive() and sleeps == []


def test_credentials_passed_to_initialize(fake):
    creds = MT5Credentials(login=42, password="pw", server="Demo", path=r"C:\mt5\terminal64.exe")
    c, _ = make(fake, creds=creds)
    c.connect()
    kw = fake.init_kwargs[-1]
    assert kw["login"] == 42 and kw["server"] == "Demo" and kw["path"].endswith("terminal64.exe")


def test_without_credentials_uses_logged_in_terminal(fake):
    c, _ = make(fake)
    c.connect()
    assert "login" not in fake.init_kwargs[-1]


def test_retries_with_exponential_backoff(fake):
    fake.fail_init = 3
    c, sleeps = make(fake)
    c.connect()
    assert c.connected and sleeps == [1, 2, 3]  # capped at retry_backoff_max_seconds


def test_gives_up_after_max_retries(fake):
    fake.fail_init = 10
    c, sleeps = make(fake, retries=3)
    with pytest.raises(MT5Error, match="after 3 attempts"):
        c.connect()
    assert not c.connected and len(sleeps) == 2


def test_auth_failure_is_not_retried(fake):
    fake.auth_fail = True
    c, sleeps = make(fake)
    with pytest.raises(MT5Error) as e:
        c.connect()
    assert e.value.code == -6 and sleeps == [] and len(fake.init_kwargs) == 1


def test_reconnects_when_link_drops(fake):
    c, _ = make(fake)
    c.connect()
    fake.drop_next_calls = 2
    rates = c.call("copy_rates_from_pos", "XAUUSD.r", fake.TIMEFRAME_M5, 0, 10)
    assert len(rates) == 10
    assert len(fake.init_kwargs) == 3  # first connect + two reconnects


def test_persistent_disconnect_raises(fake):
    c, _ = make(fake, retries=2)
    c.connect()
    fake.drop_next_calls = 99
    with pytest.raises(MT5Error):
        c.call("copy_rates_from_pos", "XAUUSD.r", fake.TIMEFRAME_M5, 0, 10)


def test_call_connects_lazily(fake):
    c, _ = make(fake)
    assert c.call("symbols_get")
    assert c.connected


def test_non_connection_error(fake):
    c, _ = make(fake)
    c.connect()
    from datetime import datetime, timezone
    args = ("NOPE", fake.TIMEFRAME_M5, datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert c.call("copy_rates_range", *args, allow_none=True) is None
    with pytest.raises(MT5Error) as e:
        c.call("copy_rates_range", *args)
    assert e.value.code == -4


def test_context_manager_disconnects(fake):
    c, _ = make(fake)
    with c:
        assert fake.initialized
    assert not fake.initialized and not c.connected


def test_missing_package_is_not_retried(monkeypatch):
    import quant_engine.mt5.client as mod
    from quant_engine.mt5.client import MT5NotAvailableError

    def boom():
        raise MT5NotAvailableError("not installed")
    monkeypatch.setattr(mod, "_import_mt5", boom)
    sleeps = []
    c = MT5Client(ConnectionConfig(), sleep=sleeps.append)
    with pytest.raises(MT5NotAvailableError):
        c.connect()
    assert sleeps == []
