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
    # owner rule 2026-11-27: skip a signal when the minimum lot would risk more than per_trade_pct
    skip_if_min_lot_over: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = Field(True, alias="json")
    file: str | None = "logs/engine.log"

    model_config = {"populate_by_name": True}


class LiveConfig(BaseModel):
    """Live signal loop (Phase 4/5). Signals only: nothing here can trade."""

    poll_seconds: float = Field(10.0, gt=0)
    lookback_days: dict[str, int] = Field(default_factory=lambda: {"M5": 30, "M15": 30, "H1": 60, "H4": 200})
    max_signal_age_minutes: float = Field(4.0, gt=0)    # never send a signal older than this
    state_dir: str = "data/signals"
    daily_summary_time: str = "22:30"
    summary_timezone: str = "Asia/Kolkata"
    balance_fallback: float = Field(10000.0, gt=0)      # used only if MT5 gives no account balance


class EngineConfig(BaseModel):
    instrument: str = "XAUUSD"
    style: str = "intraday_scalping"
    execution_mode: Literal["signals_only"] = "signals_only"
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)


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


# ---------------------------------------------------------------- strategy.yaml
_HHMM = r"^([01]\d|2[0-4]):[0-5]\d$"


def hhmm_to_minutes(value: str) -> int:
    """'07:30' -> 450 minutes after midnight ('24:00' -> 1440)."""
    h, m = value.split(":")
    return int(h) * 60 + int(m)


class Window(BaseModel):
    """A [start, end) window in New York local time, stored as minutes after midnight."""

    start: int
    end: int

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, v: object) -> object:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            for x in v:
                if not re.match(_HHMM, str(x)):
                    raise ValueError(f"time {x!r} must be HH:MM")
            return {"start": hhmm_to_minutes(v[0]), "end": hhmm_to_minutes(v[1])}
        return v


class StructureConfig(BaseModel):
    ranging_lookback: int = Field(10, ge=1)


class DisplacementConfig(BaseModel):
    min_body_atr: float = Field(1.2, gt=0)
    min_body_ratio: float = Field(0.6, gt=0, le=1)


class FVGConfig(BaseModel):
    min_size_atr: float = Field(0.25, ge=0)
    min_size_spreads: float = Field(3.0, ge=0)


class LiquidityConfig(BaseModel):
    sweep_max_depth_atr: float = Field(1.5, gt=0)
    sweep_reclaim_candles: int = Field(3, ge=1)
    sweep_valid_candles: int = Field(12, ge=1)
    equal_level_atr_m15: float = Field(0.1, ge=0)
    swing_pool_lookback_days: int = Field(5, ge=1)


class SessionsConfig(BaseModel):
    asia: Window = Window(start=20 * 60, end=24 * 60)
    london_kz: Window = Window(start=2 * 60, end=5 * 60)
    newyork_kz: Window = Window(start=7 * 60, end=10 * 60)
    london_close: Window = Window(start=10 * 60, end=12 * 60)
    rollover: Window = Window(start=16 * 60 + 55, end=18 * 60 + 5)
    kill_zone_grace_minutes: int = Field(30, ge=0)
    friday_cutoff: int = 10 * 60
    trading_day_start: int = 17 * 60

    @field_validator("friday_cutoff", "trading_day_start", mode="before")
    @classmethod
    def _hhmm(cls, v: object) -> int:
        return hhmm_to_minutes(v) if isinstance(v, str) else int(v)  # type: ignore[arg-type]


class VolatilityConfig(BaseModel):
    median_days: int = Field(20, ge=1)
    dead_ratio: float = Field(0.6, gt=0)
    chaos_ratio: float = Field(2.5, gt=0)
    max_spread_ratio: float = Field(1.5, gt=0)
    min_stop_spreads: float = Field(5.0, ge=0)


class NewsConfig(BaseModel):
    """Economic-calendar blackout (Strategy Spec v1, "News and volatility filters")."""

    enabled: bool = True
    currencies: list[str] = Field(default_factory=lambda: ["USD"])
    calendar_file: str = "data/news/calendar.csv"
    feed_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    refresh_minutes: int = Field(60, ge=5)
    tier1_keywords: list[str] = Field(default_factory=lambda: [
        "Non-Farm", "NFP", "CPI", "FOMC", "Federal Funds Rate", "Fed Chair", "Interest Rate Decision"])
    tier1_before_min: int = Field(60, ge=0)
    tier1_after_min: int = Field(30, ge=0)
    fomc_after_min: int = Field(90, ge=0)
    high_before_min: int = Field(30, ge=0)
    high_after_min: int = Field(15, ge=0)
    block_holidays: bool = True
    alert_before_min: int = Field(10, ge=0)


class BullishModeConfig(BaseModel):
    pullback_window: int = Field(36, ge=3)       # M5 candles to look back for the pullback low
    min_pullback_atr: float = Field(1.0, ge=0)   # pullback depth from the prior high, in M5 ATR
    ema_touch_atr: float = Field(0.25, ge=0)     # pullback low within this x M15 ATR of the fast EMA
    tp2_r: float = Field(2.0, gt=0)              # measured-move TP2 when no liquidity is far enough
    tp3_r: float = Field(3.0, ge=0)              # runner (0 = none)
    time_stop_candles: int = Field(36, ge=1)     # trends need time: 3 hours


class BearishModeConfig(BaseModel):
    rally_window: int = Field(24, ge=3)          # M5 candles back to look for the buy-stop raid
    tp2_min_r: float = Field(2.0, gt=0)
    time_stop_candles: int = Field(18, ge=1)     # sell-offs work fast or not at all: 90 minutes


class SidewaysModeConfig(BaseModel):
    min_range_atr_h1: float = Field(2.0, gt=0)   # H1 range width limits, in H1 ATR
    max_range_atr_h1: float = Field(8.0, gt=0)
    max_vol_ratio: float = Field(1.3, gt=0)      # M5 ATR vs its time-of-day median; above = expanding
    edge_zone: float = Field(0.15, gt=0, lt=0.5) # sweep level within this share of the range from an edge
    min_wick: float = Field(0.4, ge=0, le=1)     # rejection wick share of the sweep candle
    opposite_edge_buffer: float = Field(0.1, ge=0, lt=0.5)
    tp2_min_r: float = Field(1.5, gt=0)          # range targets are capped by the opposite edge
    time_stop_candles: int = Field(18, ge=1)


class ModesConfig(BaseModel):
    """The three regime strategies (strategy/modes.py)."""

    ema_fast: int = Field(20, ge=2)
    ema_slow: int = Field(50, ge=3)
    bullish: BullishModeConfig = Field(default_factory=BullishModeConfig)
    bearish: BearishModeConfig = Field(default_factory=BearishModeConfig)
    sideways: SidewaysModeConfig = Field(default_factory=SidewaysModeConfig)


class EntryConfig(BaseModel):
    min_score: int = Field(65, ge=0, le=100)
    a_plus_score: int = Field(80, ge=0, le=100)
    b_grade_score: int = Field(50, ge=0, le=100)
    sl_buffer_atr: float = Field(0.1, ge=0)
    sl_buffer_spreads: float = Field(2.0, ge=0)
    min_stop_atr: float = Field(0.5, gt=0)
    max_stop_atr: float = Field(2.5, gt=0)
    tp1_min_r: float = Field(1.0, gt=0)
    tp2_min_r: float = Field(2.0, gt=0)
    tp3_min_r: float = Field(3.0, gt=0)
    expiry_candles: int = Field(6, ge=1)
    time_stop_candles: int = Field(24, ge=1)
    session_close_hours: float = Field(2.0, ge=0)
    preferred_kill_zone: Literal["london_kz", "newyork_kz"] = "newyork_kz"


class BacktestConfig(BaseModel):
    assumed_spread_points: float = Field(30.0, ge=0)
    risk_pct: float = Field(0.5, gt=0, le=5)
    daily_loss_lock_pct: float = Field(2.0, gt=0)
    weekly_loss_lock_pct: float = Field(5.0, gt=0)
    max_losses_in_row: int = Field(2, ge=1)
    max_signals_per_session: int = Field(2, ge=1)
    regime_ema_period: int = Field(50, ge=2)
    regime_slope_bars: int = Field(10, ge=1)
    regime_min_slope_atr: float = Field(0.5, ge=0)


class CostsConfig(BaseModel):
    """Trading costs charged in research backtests (rulebook rule 6)."""
    min_spread_points: float = Field(8.0, ge=0)        # floor under the historical bar spread (owner: $0.08)
    commission_per_lot: float = Field(0.0, ge=0)       # USD per lot, round trip (owner: 0, 2026-11-26)
    contract_size: float = Field(100.0, gt=0)          # XAUUSD: 1 lot = 100 oz
    slippage_points: float = Field(2.0, ge=0)          # per side on market / stop fills. GUESS


class Regime4Config(BaseModel):
    """Four-state H4 regime (rulebook rule 4). All values in H4 candles / H4 ATR(14)."""
    ema_period: int = Field(50, ge=5)
    slope_bars: int = Field(10, ge=2)
    trend_slope_atr: float = Field(0.5, gt=0)          # EMA rise over slope_bars for a trend. GUESS
    confirm_bars: int = Field(2, ge=1)                 # trend must hold this many H4 closes in a row
    flat_slope_atr: float = Field(0.2, ge=0)           # |EMA change| below this = flat. GUESS
    range_bars: int = Field(30, ge=5)                  # sideways: the last range_bars H4 candles ...
    range_max_atr: float = Field(8.0, gt=0)            # ... span at most this many ATR. GUESS


class ResearchConfig(BaseModel):
    """Pattern mining (``python -m quant_engine patterns``). Distances are in M5 ATR(14).

    Discovery uses the older ``discover_frac`` of the data; the newer part is held out
    and only used to check the patterns found (never to find them)."""
    discover_frac: float = Field(0.6, gt=0.2, lt=0.9)
    sl_atr: list[float] = Field(default_factory=lambda: [1.0])
    tp_atr: list[float] = Field(default_factory=lambda: [1.0, 2.0])
    horizons: list[int] = Field(default_factory=lambda: [12, 36])     # M5 candles (1h, 3h)
    sweep_lookbacks: list[int] = Field(default_factory=lambda: [24, 72])   # M5 candles (2h, 6h)
    impulse_body_atr: float = Field(1.5, gt=0)
    impulse_body_ratio: float = Field(0.6, gt=0, le=1)
    move_bars: int = Field(12, ge=2)
    move_atr: float = Field(3.0, gt=0)
    range_bars: int = Field(48, ge=6)
    range_min_atr: float = Field(3.0, gt=0)
    range_max_atr: float = Field(10.0, gt=0)
    range_edge: float = Field(0.1, gt=0, lt=0.5)
    min_trades_discover: int = Field(30, ge=5)
    min_trades_validate: int = Field(15, ge=3)
    fdr_q: float = Field(0.10, gt=0, lt=1)
    min_avg_r: float = Field(0.05, ge=0)
    min_stable_thirds: int = Field(2, ge=0, le=3)
    validate_p: float = Field(0.05, gt=0, lt=1)
    max_patterns: int = Field(20, ge=1)
    use_verdicts: list[str] = Field(default_factory=lambda: ["confirmed"])
    patterns_file: str = "reports/patterns_latest.json"
    fixed_spread_points: float | None = Field(None, ge=0)   # one spread for every year (None = bar spreads)
    round_steps: list[float] = Field(default_factory=lambda: [50.0, 100.0])  # round-number levels in USD
    level_days: list[int] = Field(default_factory=lambda: [5, 20])          # multi-day highs/lows (S/R)
    pin_wick_ratio: float = Field(0.6, gt=0, lt=1)
    pin_min_atr: float = Field(1.0, gt=0)
    level_touch_atr: float = Field(0.25, ge=0)
    news_times_ny: list[str] = Field(default_factory=lambda: ["08:30", "10:00", "14:00"])
    news_spike_atr: float = Field(2.5, gt=0)
    timeframes: dict[str, dict] = Field(default_factory=dict)   # overrides per entry timeframe (H1, H4)
    only_families: list[str] | None = None   # test only these idea families (None = all)
    exit_units: Literal["atr", "usd"] = "atr"   # usd: sl_atr / tp_atr are dollar distances
    costs: CostsConfig = Field(default_factory=CostsConfig)
    regime4: Regime4Config = Field(default_factory=Regime4Config)
    bigmoves: dict = Field(default_factory=lambda: {
        "pip_usd": 0.10, "move_pips": [30, 50], "window_bars": 6,
        "target_pips": [30, 50], "stop_pips": [20, 30, 50]})

    def for_timeframe(self, tf: str) -> "ResearchConfig":
        """These settings with the overrides for entry timeframe ``tf`` (e.g. H1) applied."""
        extra = self.timeframes.get(str(tf).upper(), {})
        return ResearchConfig(**{**self.model_dump(), **extra}) if extra else self

    @field_validator("sl_atr", "tp_atr", "horizons", "sweep_lookbacks", "round_steps", "level_days")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v or any(x <= 0 for x in v):
            raise ValueError("must be a non-empty list of positive numbers")
        return v


class StrategyConfig(BaseModel):
    atr_period: int = Field(14, ge=1)
    swings: dict[Timeframe, int] = Field(default_factory=lambda: {
        Timeframe.M5: 2, Timeframe.M15: 3, Timeframe.H1: 3, Timeframe.H4: 3})
    structure: StructureConfig = Field(default_factory=StructureConfig)
    displacement: DisplacementConfig = Field(default_factory=DisplacementConfig)
    fvg: FVGConfig = Field(default_factory=FVGConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    mode: Literal["regime", "v1", "bullish", "bearish", "sideways", "patterns"] = "regime"
    news: NewsConfig = Field(default_factory=NewsConfig)
    modes: ModesConfig = Field(default_factory=ModesConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

    @field_validator("swings", mode="before")
    @classmethod
    def _parse_swings(cls, v: dict) -> dict:
        out = {Timeframe.parse(k): int(n) for k, n in v.items()}
        if any(n < 1 for n in out.values()):
            raise ValueError("swing length must be >= 1")
        return out

    def swing_length(self, tf: Timeframe) -> int:
        """Fractal length L for a timeframe (defaults to 3)."""
        return self.swings.get(tf, 3)


# ----------------------------------------------------------------------- loader
class AppConfig(BaseModel):
    engine: EngineConfig
    mt5: MT5Config
    symbols: dict[str, SymbolMapping]
    data: DataConfig
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
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
            strategy=StrategyConfig(**_read_yaml(root / "strategy.yaml"))
            if (root / "strategy.yaml").exists() else StrategyConfig(),
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
