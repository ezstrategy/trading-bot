"""Configuration loading and validation.

Non-secret settings live in YAML files under ``config/``. Secrets (MT5 login,
password, server, Telegram token) are read from environment variables or a
local ``.env`` file only, never from YAML and never hardcoded.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .timeframes import Timeframe


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


# ------------------------------------------------------------------ engine.yaml
class TimeframesConfig(BaseModel):
    entry: Timeframe = Timeframe.M5
    confirmation: Timeframe = Timeframe.M15
    context: list[Timeframe] = Field(default_factory=lambda: [Timeframe.H1, Timeframe.H4])
    collect: list[Timeframe] = Field(default_factory=lambda: list(Timeframe))

    @field_validator("entry", "confirmation", mode="before")
    @classmethod
    def _parse_one(cls, v: object) -> Timeframe:
        return Timeframe.parse(v)  # type: ignore[arg-type]

    @field_validator("context", "collect", mode="before")
    @classmethod
    def _parse_many(cls, v: object) -> list[Timeframe]:
        return [Timeframe.parse(x) for x in v]  # type: ignore[union-attr]

    @model_validator(mode="after")
    def _collect_covers_used(self) -> "TimeframesConfig":
        used = {self.entry, self.confirmation, *self.context}
        missing = used - set(self.collect)
        if missing:
            raise ValueError(f"timeframes.collect must include {sorted(t.value for t in missing)}")
        return self


class RiskConfig(BaseModel):
    per_trade_pct: float = Field(0.5, gt=0, le=5)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = Field(True, alias="json")
    file: str | None = "logs/engine.log"

    model_config = {"populate_by_name": True}


class EngineConfig(BaseModel):
    instrument: str = "XAUUSD"
    style: str = "intraday_scalping"
    execution_mode: Literal["signals_only"] = "signals_only"
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# --------------------------------------------------------------------- mt5.yaml
class ConnectionConfig(BaseModel):
    timeout_ms: int = Field(60000, gt=0)
    max_retries: int = Field(5, ge=1)
    retry_backoff_seconds: float = Field(2.0, ge=0)
    retry_backoff_max_seconds: float = Field(60.0, ge=0)


ServerTimeMode = Literal["auto", "ny_close", "fixed", "timezone"]


class ServerTimeConfig(BaseModel):
    mode: ServerTimeMode = "auto"
    fallback_mode: Literal["ny_close", "fixed", "timezone"] = "ny_close"
    fixed_offset_hours: float = Field(2, ge=-12, le=14)
    timezone: str = "Etc/UTC"
    auto_round_minutes: int = Field(30, gt=0)
    auto_max_tick_age_seconds: int = Field(120, gt=0)


class MT5Config(BaseModel):
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    server_time: ServerTimeConfig = Field(default_factory=ServerTimeConfig)


class MT5Credentials(BaseModel):
    """MT5 account credentials, populated from the environment only."""

    login: int | None = None
    password: str | None = Field(default=None, repr=False)
    server: str | None = None
    path: str | None = None

    @property
    def complete(self) -> bool:
        """True when login, password and server are all set."""
        return bool(self.login and self.password and self.server)

    @classmethod
    def from_env(cls) -> "MT5Credentials":
        """Read MT5_LOGIN, MT5_PASSWORD, MT5_SERVER and MT5_PATH."""
        raw_login = os.getenv("MT5_LOGIN", "").strip()
        if raw_login and not raw_login.isdigit():
            raise ConfigError("MT5_LOGIN must be a number (your MT5 account number)")
        return cls(login=int(raw_login) if raw_login else None,
                   password=os.getenv("MT5_PASSWORD") or None,
                   server=os.getenv("MT5_SERVER") or None,
                   path=os.getenv("MT5_PATH") or None)


# ----------------------------------------------------------------- symbols.yaml
class SymbolMapping(BaseModel):
    broker_symbol: str = ""
    candidates: list[str] = Field(default_factory=list)
    pattern: str = ""

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        if v:
            re.compile(v)
        return v


# -------------------------------------------------------------------- data.yaml
class StorageConfig(BaseModel):
    root: str = "data"
    format: Literal["parquet"] = "parquet"


class DownloadConfig(BaseModel):
    default_days: int = Field(90, gt=0)
    chunk_days: dict[Timeframe, int] = Field(default_factory=lambda: {
        Timeframe.M1: 7, Timeframe.M5: 30, Timeframe.M15: 60, Timeframe.H1: 180, Timeframe.H4: 365})
    tick_chunk_hours: int = Field(6, gt=0)

    @field_validator("chunk_days", mode="before")
    @classmethod
    def _parse_keys(cls, v: dict) -> dict:
        return {Timeframe.parse(k): int(n) for k, n in v.items()}


class WeekPoint(BaseModel):
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)


class QualityConfig(BaseModel):
    weekly_close: WeekPoint = WeekPoint(weekday=4, hour=20)
    weekly_open: WeekPoint = WeekPoint(weekday=6, hour=23)
    daily_break_utc: tuple[int, int] = (20, 23)
    max_spike_atr: float = Field(15.0, gt=0)
    max_tick_spread: float = Field(5.0, gt=0)


class DataConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


# ----------------------------------------------------------------------- loader
class AppConfig(BaseModel):
    engine: EngineConfig
    mt5: MT5Config
    symbols: dict[str, SymbolMapping]
    data: DataConfig
    credentials: MT5Credentials = Field(default_factory=MT5Credentials, repr=False)

    def mapping(self, canonical: str | None = None) -> SymbolMapping:
        """Symbol mapping for a canonical instrument (defaults to engine.instrument)."""
        name = (canonical or self.engine.instrument).upper()
        if name not in self.symbols:
            raise ConfigError(f"No symbol mapping for {name} in symbols.yaml")
        return self.symbols[name]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_config(config_dir: str | Path | None = None, *, load_env_file: bool = True) -> AppConfig:
    """Load and validate all YAML config files plus environment credentials.

    Args:
        config_dir: folder holding engine.yaml, mt5.yaml, symbols.yaml and data.yaml.
            Defaults to $QE_CONFIG_DIR or ./config.
        load_env_file: read a local .env file into the environment first.

    Raises:
        ConfigError: if a file is missing or a value is invalid.
    """
    if load_env_file:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:  # pragma: no cover - optional
            pass
    root = Path(config_dir or os.getenv("QE_CONFIG_DIR", "config"))
    try:
        cfg = AppConfig(
            engine=EngineConfig(**_read_yaml(root / "engine.yaml")),
            mt5=MT5Config(**_read_yaml(root / "mt5.yaml")),
            symbols={k.upper(): SymbolMapping(**(v or {}))
                     for k, v in _read_yaml(root / "symbols.yaml").items()},
            data=DataConfig(**_read_yaml(root / "data.yaml")),
            credentials=MT5Credentials.from_env(),
        )
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"Invalid configuration in {root}: {exc}") from exc
    level = os.getenv("QE_LOG_LEVEL")
    if level:
        cfg.engine.logging.level = level
    return cfg
