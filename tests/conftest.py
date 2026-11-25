"""Shared fixtures: repository config pointed at a temp data folder and a fake MT5."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from quant_engine.config import AppConfig, load_config
from quant_engine.data.collector import MarketDataCollector
from quant_engine.mt5.client import MT5Client
from quant_engine.mt5.server_time import ServerClock

from .fake_mt5 import NOW_UTC, FakeMT5

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH", "QE_CONFIG_DIR", "QE_LOG_LEVEL"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of the repo's config/ with storage and logs redirected to tmp_path."""
    d = tmp_path / "config"
    shutil.copytree(REPO / "config", d)
    data = yaml.safe_load((d / "data.yaml").read_text())
    data["storage"]["root"] = str(tmp_path / "data")
    (d / "data.yaml").write_text(yaml.safe_dump(data))
    eng = yaml.safe_load((d / "engine.yaml").read_text())
    eng["logging"]["file"] = str(tmp_path / "logs" / "engine.log")
    (d / "engine.yaml").write_text(yaml.safe_dump(eng))
    return d


@pytest.fixture
def cfg(config_dir: Path) -> AppConfig:
    return load_config(config_dir, load_env_file=False)


@pytest.fixture
def fake() -> FakeMT5:
    return FakeMT5()


@pytest.fixture
def client(cfg: AppConfig, fake: FakeMT5) -> MT5Client:
    c = MT5Client(cfg.mt5.connection, cfg.credentials, mt5_module=fake, sleep=lambda s: None)
    c.connect()
    return c


@pytest.fixture
def collector(client: MT5Client) -> MarketDataCollector:
    return MarketDataCollector(client, "XAUUSD.r", ServerClock(mode="ny_close"), now=lambda: NOW_UTC)
