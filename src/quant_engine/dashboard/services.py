"""Everything the dashboard does, as plain functions (no Streamlit here, so it is testable).

The dashboard only reads data and runs the same read-only operations as the
CLI. There is no code path to orders: MT5 access goes through the Phase 1
client, which blocks order functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.downloader import DownloadResult, HistoricalDownloader
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..features.context import STRUCTURE_TFS, MarketFeatures, MarketSnapshot
from ..session import open_session
from ..timeframes import Timeframe


# ------------------------------------------------------------------ data
def load_bars(store: ParquetStore, canonical: str,
              tfs: tuple[Timeframe, ...] = STRUCTURE_TFS) -> dict[Timeframe, pd.DataFrame]:
    """Stored bars for each timeframe (empty frames where nothing is stored)."""
    return {tf: store.read_bars(canonical, tf) for tf in tfs}


def data_status(cfg: AppConfig, store: ParquetStore, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per collected timeframe: rows, first/last candle, age and quality."""
    now = now or pd.Timestamp.now(tz="UTC")
    rows = []
    for tf in cfg.engine.timeframes.collect:
        b = store.read_bars(cfg.engine.instrument, tf)
        if b.empty:
            rows.append({"timeframe": tf.value, "candles": 0, "first": None, "last": None,
                         "last_close_age": None, "quality": "no data", "issues": ""})
            continue
        rep = validate_bars(b, tf, cfg.data.quality, dataset=tf.value)
        last_close = b["time"].iat[-1] + tf.delta
        counts = rep.counts()
        rows.append({"timeframe": tf.value, "candles": len(b), "first": b["time"].iat[0],
                     "last": b["time"].iat[-1], "last_close_age": _age(now - last_close),
                     "quality": "OK" if rep.ok else "ISSUES",
                     "issues": ", ".join(f"{k}: {v}" for k, v in counts.items())})
    return pd.DataFrame(rows)


def _age(d: pd.Timedelta) -> str:
    s = int(d.total_seconds())
    if s < 0:
        return "0m"
    days, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    return f"{days}d {h}h" if days else f"{h}h {m}m" if h else f"{m}m"


# ------------------------------------------------------------------- MT5
def check_connection(cfg: AppConfig, mt5_module: Any | None = None, now=None) -> dict:
    """Connect, resolve the broker symbol, calibrate server time, then disconnect."""
    with open_session(cfg, mt5_module=mt5_module, now=now) as s:
        tick = s.collector.last_tick_server_epoch()
        return {
            "connected": s.client.is_alive(),
            "broker_symbol": s.symbol.name,
            "digits": s.symbol.digits,
            "point": s.symbol.point,
            "clock_mode": s.clock.mode,
            "server_offset_hours": s.clock.offset_at(s.collector.now()).total_seconds() / 3600,
            "last_tick_utc": str(s.clock.to_utc([tick])[0]) if tick else None,
        }


def download(cfg: AppConfig, store: ParquetStore, tfs: list[Timeframe], days: int,
             mt5_module: Any | None = None, now=None) -> list[DownloadResult]:
    """Download (or top up) the last ``days`` of candles for ``tfs``."""
    with open_session(cfg, mt5_module=mt5_module, now=now) as s:
        end = s.collector.now().floor("min")
        start = end - pd.Timedelta(days=days)
        dl = HistoricalDownloader(s.collector, store, cfg.engine.instrument, cfg.data)
        return [dl.download_bars(tf, start, end, resume=True) for tf in tfs]


# -------------------------------------------------------------- features
def build_features(bars: dict[Timeframe, pd.DataFrame], cfg: AppConfig, point: float) -> MarketFeatures:
    """Phase 2 features over the stored history."""
    missing = [tf.value for tf in STRUCTURE_TFS if bars.get(tf) is None or bars[tf].empty]
    if missing:
        raise ValueError(f"no stored data for {', '.join(missing)}; download first")
    return MarketFeatures(bars, cfg.strategy, point)


def default_decision_time(bars: dict[Timeframe, pd.DataFrame]) -> pd.Timestamp:
    """Close time of the last stored M5 candle."""
    m5 = bars[Timeframe.M5]
    return m5["time"].iat[-1] + Timeframe.M5.delta


def snapshot_summary(s: MarketSnapshot) -> dict:
    """JSON-friendly view of a snapshot (used by the dashboard and the CLI)."""
    dr = s.dealing_range
    price = float(s.m5_bar["close"])
    return {
        "decision_time_utc": str(s.decision_time),
        "last_m5_close": price,
        "structure": {tf.value: st for tf, st in s.state.items()},
        "bias": s.bias, "htf_points": s.htf_points,
        "m15_allows": {"long": s.m15_allows_long, "short": s.m15_allows_short},
        "session": s.session, "in_kill_zone": s.in_kill_zone, "new_signals_blocked": s.new_signals_blocked,
        "atr": {tf.value: None if np.isnan(v) else round(v, 3) for tf, v in s.atr.items()},
        "h1_dealing_range": None if dr is None else {
            "low": dr.low, "high": dr.high, "eq": dr.equilibrium, "price_zone": dr.zone(price)},
        "volatility": {"ratio": None if np.isnan(s.vol_ratio) else round(s.vol_ratio, 2),
                       "regime": s.vol_regime},
        "pools": [{"kind": p.kind, "side": p.side, "price": round(p.price, 2)} for p in s.pools.itertuples()],
        "recent_sweeps": [{"pool": w.pool_kind, "side": w.side, "level": round(w.level, 2),
                           "time": str(w.sweep_time)} for w in s.recent_sweeps],
        "recent_m5_breaks": [{"kind": e.kind, "dir": int(e.direction), "level": round(e.level, 2),
                              "time": str(e.time)} for e in s.m5_events.itertuples()],
    }


def market_status(s: MarketSnapshot) -> list[tuple[str, str, str]]:
    """(label, value, tone) cards for the top of the dashboard. tone: good/bad/neutral."""
    tone = {"bullish": "good", "bearish": "bad", "ranging": "neutral",
            "BULL": "good", "BEAR": "bad", "NONE": "neutral"}
    session = {"london_kz": "London KZ", "newyork_kz": "New York KZ", "london_close": "London close",
               "asia": "Asia", "rollover": "Rollover", "off": "Off"}.get(s.session, s.session)
    cards = [("Bias (H4/H1)", f"{s.bias} (+{s.htf_points})" if s.bias != "NONE" else "NONE", tone[s.bias])]
    for tf in (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5):
        cards.append((f"{tf.value} structure", s.state[tf], tone[s.state[tf]]))
    cards += [
        ("Session (NY time)", session, "good" if s.in_kill_zone else "neutral"),
        ("New signals", "blocked" if s.new_signals_blocked else "allowed",
         "bad" if s.new_signals_blocked else "good"),
        ("Volatility", s.vol_regime, "good" if s.vol_regime == "normal" else "bad"),
    ]
    return cards


# ------------------------------------------------------------ phases 3-6
def signals_table(state_dir: str) -> pd.DataFrame:
    """Every signal the live engine sent, newest first."""
    from ..live.state import EngineState
    st = EngineState(state_dir)
    rows = [{"time": s.signal.time, "side": s.signal.side, "grade": s.signal.grade, "score": s.signal.score,
             "entry": s.signal.entry, "stop": s.signal.stop_loss, "tp1": s.signal.tp1, "tp2": s.signal.tp2,
             "lots": s.lots, "status": s.status, "outcome": s.outcome, "r": s.r}
            for s in st.signals.values()]
    df = pd.DataFrame(rows)
    return df.sort_values("time", ascending=False).reset_index(drop=True) if len(df) else df


def backtest_reports(root: str = "reports") -> list[Path]:
    """Backtest report folders, newest first."""
    p = Path(root)
    return sorted((d for d in p.glob("backtest_*") if (d / "summary.json").exists()), reverse=True) if p.exists() else []


def run_backtest_now(cfg: AppConfig, store: ParquetStore, source: str, point: float,
                     mode: str | None = None) -> Path:
    """Run a backtest on stored bars (``mt5`` or ``dukascopy``) and return its report folder."""
    from ..cli import dukascopy_dataset
    from ..strategy.runner import run_backtest
    canonical = cfg.engine.instrument
    ds = dukascopy_dataset(canonical) if source == "dukascopy" else canonical
    mode = mode or cfg.strategy.mode
    out = Path("reports") / f"backtest_{source}_{mode}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
    note = "Dukascopy XAU/USD spot BID M1 (UTC), spread assumed" if source == "dukascopy" else "MT5 broker bars (UTC)"
    run_backtest(cfg, store, ds, point, out, source_note=note, mode=mode)
    return out


def news_table(cfg: AppConfig, now: pd.Timestamp, refresh: bool = False) -> pd.DataFrame:
    """Upcoming news blocks (start, end, tier, title)."""
    from ..features.news import NewsCalendar, load_calendar, refresh_calendar
    nc = cfg.strategy.news
    events = refresh_calendar(nc) if refresh else load_calendar(nc.calendar_file)
    cal = NewsCalendar(events, nc)
    return pd.DataFrame([{"start": w.start, "end": w.end, "event": w.event_time, "tier": w.tier, "title": w.title}
                         for w in cal.windows if w.end > now])
