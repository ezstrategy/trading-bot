import numpy as np
import pandas as pd
import pytest

from quant_engine.config import ServerTimeConfig
from quant_engine.mt5.server_time import ServerClock


def server_epoch(wall: str) -> int:
    return int(pd.Timestamp(wall).value // 1_000_000_000)


@pytest.mark.parametrize("wall,utc", [
    ("2026-01-15 10:00", "2026-01-15 08:00"),   # winter: GMT+2
    ("2026-07-15 10:00", "2026-07-15 07:00"),   # summer: GMT+3
    ("2026-03-10 10:00", "2026-03-10 07:00"),   # US already on DST, EU not yet
    ("2026-11-02 00:00", "2026-11-01 22:00"),   # day after US DST ends
])
def test_ny_close_conversion(wall, utc):
    assert ServerClock(mode="ny_close").to_utc([server_epoch(wall)])[0] == pd.Timestamp(utc, tz="UTC")


def test_fixed_and_timezone_modes():
    e = [server_epoch("2026-07-15 10:00")]
    assert ServerClock(mode="fixed", fixed_offset_hours=2).to_utc(e)[0] == pd.Timestamp("2026-07-15 08:00", tz="UTC")
    assert ServerClock(mode="timezone", timezone="Europe/London").to_utc(e)[0] == pd.Timestamp("2026-07-15 09:00", tz="UTC")


def test_milliseconds_unit():
    ms = server_epoch("2026-01-15 10:00") * 1000 + 250
    assert ServerClock().to_utc([ms], unit="ms")[0] == pd.Timestamp("2026-01-15 08:00:00.250", tz="UTC")


def test_result_is_utc():
    idx = ServerClock().to_utc(np.array([server_epoch("2026-01-15 10:00")]))
    assert str(idx.tz) == "UTC"


@pytest.mark.parametrize("mode", ["ny_close", "fixed", "timezone"])
def test_to_server_round_trip(mode):
    clock = ServerClock(mode=mode, fixed_offset_hours=3, timezone="Europe/Athens")
    for utc in pd.date_range("2026-01-01", "2026-12-31", freq="17D", tz="UTC"):
        assert clock.to_utc([clock.to_server(utc).timestamp()])[0] == utc


def test_calibrate_detects_ny_close():
    cfg = ServerTimeConfig(mode="auto")
    clock = ServerClock.from_config(cfg)
    now = pd.Timestamp("2026-07-15 12:00", tz="UTC").value / 1e9
    assert clock.calibrate(now + 3 * 3600 - 2, cfg, now_utc=now)
    assert clock.mode == "ny_close" and clock.detected_offset_hours == 3


def test_calibrate_switches_to_fixed_for_other_brokers():
    cfg = ServerTimeConfig(mode="auto")
    clock = ServerClock.from_config(cfg)
    now = pd.Timestamp("2026-07-15 12:00", tz="UTC").value / 1e9
    assert clock.calibrate(now + 0 - 5, cfg, now_utc=now)  # broker on UTC
    assert clock.mode == "fixed" and clock.fixed_offset_hours == 0


def test_calibrate_ignores_stale_tick():
    cfg = ServerTimeConfig(mode="auto", auto_max_tick_age_seconds=60)
    clock = ServerClock.from_config(cfg)
    now = pd.Timestamp("2026-07-18 12:00", tz="UTC").value / 1e9  # Saturday
    assert not clock.calibrate(now + 3 * 3600 - 13 * 60, cfg, now_utc=now)  # 13 min old
    assert clock.mode == "ny_close"
