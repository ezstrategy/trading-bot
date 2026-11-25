import pytest
import yaml

from quant_engine.config import ConfigError, MT5Credentials, load_config
from quant_engine.timeframes import Timeframe


def test_repo_config_loads(cfg):
    assert cfg.engine.instrument == "XAUUSD"
    assert cfg.engine.execution_mode == "signals_only"
    assert cfg.engine.timeframes.entry is Timeframe.M5
    assert cfg.engine.timeframes.confirmation is Timeframe.M15
    assert cfg.engine.timeframes.context == [Timeframe.H1, Timeframe.H4]
    assert set(cfg.engine.timeframes.collect) == set(Timeframe)
    assert cfg.engine.risk.per_trade_pct == 0.5
    assert cfg.mapping("xauusd").candidates[0] == "XAUUSD"


def test_credentials_come_from_env_only(monkeypatch, config_dir):
    monkeypatch.setenv("MT5_LOGIN", "5550123")
    monkeypatch.setenv("MT5_PASSWORD", "s3cret")
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
    cfg = load_config(config_dir, load_env_file=False)
    assert cfg.credentials.login == 5550123 and cfg.credentials.complete
    assert "s3cret" not in repr(cfg.credentials) and "s3cret" not in repr(cfg)


def test_no_credentials_in_yaml(config_dir):
    text = " ".join(p.read_text().lower() for p in config_dir.glob("*.yaml"))
    assert "password:" not in text and "login:" not in text


def test_bad_login_rejected(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "abc")
    with pytest.raises(ConfigError):
        MT5Credentials.from_env()


def test_missing_file(config_dir):
    (config_dir / "mt5.yaml").unlink()
    with pytest.raises(ConfigError, match="Missing config file"):
        load_config(config_dir, load_env_file=False)


@pytest.mark.parametrize("patch", [
    {"execution_mode": "live"},
    {"timeframes": {"entry": "M5", "confirmation": "M15", "context": ["H1", "H4"], "collect": ["M5"]}},
    {"timeframes": {"entry": "M7"}},
    {"risk": {"per_trade_pct": 50}},
])
def test_invalid_engine_values(config_dir, patch):
    data = yaml.safe_load((config_dir / "engine.yaml").read_text())
    data.update(patch)
    (config_dir / "engine.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_config(config_dir, load_env_file=False)


def test_invalid_regex(config_dir):
    (config_dir / "symbols.yaml").write_text("XAUUSD:\n  pattern: '(['\n")
    with pytest.raises(ConfigError):
        load_config(config_dir, load_env_file=False)


def test_timeframe_parse():
    assert Timeframe.parse("m15") is Timeframe.M15
    assert Timeframe.H4.delta.total_seconds() == 4 * 3600
    assert Timeframe.H1.pandas_freq == "1h" and Timeframe.M5.pandas_freq == "5min"
    with pytest.raises(ValueError):
        Timeframe.parse("D1")
