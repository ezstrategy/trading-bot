"""Command line interface.

    python -m quant_engine check
    python -m quant_engine symbols [--all]
    python -m quant_engine download [--timeframes M1,M5,M15,H1,H4] [--start 2026-08-01] [--end ...] [--days 90]
    python -m quant_engine ticks --start 2026-11-20 [--end 2026-11-21]
    python -m quant_engine validate [--timeframes M5]
    python -m quant_engine latest --timeframe M5 [--count 10]
    python -m quant_engine context [--at "2026-11-24 13:05"] [--point 0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence

import pandas as pd

from .config import ConfigError, load_config
from .data.downloader import HistoricalDownloader, write_report
from .data.quality import validate_bars
from .data.storage import ParquetStore
from .logging_setup import get_logger, setup_logging
from .mt5.client import MT5Error
from .mt5.symbols import SymbolResolutionError, discover_symbols
from .session import open_session
from .timeframes import Timeframe

log = get_logger(__name__)


def _tfs(value: str | None, default: list[Timeframe]) -> list[Timeframe]:
    return [Timeframe.parse(x) for x in value.split(",")] if value else list(default)


def _range(args: argparse.Namespace, default_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("min")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(days=args.days or default_days)
    if start >= end:
        raise SystemExit("--start must be before --end")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant_engine", description="XAUUSD Quant Engine (signals only)")
    p.add_argument("--config", help="config folder (default ./config)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="connect, resolve symbol and show server time offset")
    s = sub.add_parser("symbols", help="list broker symbols")
    s.add_argument("--all", action="store_true", help="list every symbol, not only gold")
    d = sub.add_parser("download", help="download historical bars")
    d.add_argument("--timeframes")
    t = sub.add_parser("ticks", help="download bid/ask ticks")
    for x in (d, t):
        x.add_argument("--start")
        x.add_argument("--end")
        x.add_argument("--days", type=int)
        x.add_argument("--no-resume", action="store_true")
    v = sub.add_parser("validate", help="run quality checks on stored bars")
    v.add_argument("--timeframes")
    lt = sub.add_parser("latest", help="print the latest closed candles")
    lt.add_argument("--timeframe", default="M5")
    lt.add_argument("--count", type=int, default=10)
    cx = sub.add_parser("context", help="market-structure snapshot from stored data (no MT5 needed)")
    cx.add_argument("--at", help="UTC decision time (default: last stored M5 close)")
    cx.add_argument("--point", type=float, default=0.01, help="tick size of your broker symbol")
    return p


def _context(cfg, store: ParquetStore, canonical: str, at: str | None, point: float) -> int:
    from .features.context import STRUCTURE_TFS, MarketFeatures
    bars = {tf: store.read_bars(canonical, tf) for tf in STRUCTURE_TFS}
    empty = [tf.value for tf, b in bars.items() if b.empty]
    if empty:
        print(f"No stored data for {', '.join(empty)}. Run: python -m quant_engine download", file=sys.stderr)
        return 1
    m5 = bars[Timeframe.M5]
    t = pd.Timestamp(at, tz="UTC") if at else m5["time"].iat[-1] + Timeframe.M5.delta
    s = MarketFeatures(bars, cfg.strategy, point).snapshot(t)
    dr = s.dealing_range
    print(json.dumps({
        "decision_time_utc": str(s.decision_time),
        "last_m5_close": float(s.m5_bar["close"]),
        "structure": {tf.value: st for tf, st in s.state.items()},
        "bias": s.bias, "htf_points": s.htf_points,
        "m15_allows": {"long": s.m15_allows_long, "short": s.m15_allows_short},
        "session": s.session, "in_kill_zone": s.in_kill_zone, "new_signals_blocked": s.new_signals_blocked,
        "atr": {tf.value: round(v, 3) for tf, v in s.atr.items()},
        "h1_dealing_range": None if dr is None else {"low": dr.low, "high": dr.high, "eq": dr.equilibrium},
        "volatility": {"ratio": round(s.vol_ratio, 2), "regime": s.vol_regime},
        "pools": [{"kind": p.kind, "side": p.side, "price": round(p.price, 2)} for p in s.pools.itertuples()],
        "recent_sweeps": [{"pool": w.pool_kind, "side": w.side, "level": round(w.level, 2),
                           "time": str(w.sweep_time)} for w in s.recent_sweeps],
        "recent_m5_breaks": [{"kind": e.kind, "dir": int(e.direction), "level": round(e.level, 2),
                              "time": str(e.time)} for e in s.m5_events.itertuples()],
    }, indent=2, default=str))
    return 0


def run(argv: Sequence[str] | None = None, mt5_module: Any | None = None,
        now: Callable[[], pd.Timestamp] | None = None) -> int:
    """Entry point used by ``main`` and by tests (with a fake MT5 module)."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    lc = cfg.engine.logging
    setup_logging(lc.level, lc.json_format, lc.file)
    store = ParquetStore(cfg.data.storage.root)
    canonical = cfg.engine.instrument

    if args.cmd == "validate":
        reports = [validate_bars(store.read_bars(canonical, tf), tf, cfg.data.quality,
                                 dataset=f"{canonical} {tf.value}")
                   for tf in _tfs(args.timeframes, cfg.engine.timeframes.collect)]
        for r in reports:
            print(json.dumps({k: v for k, v in r.to_dict().items() if k != "issues"}))
        return 0 if all(r.ok for r in reports) else 1
    if args.cmd == "context":
        return _context(cfg, store, canonical, args.at, args.point)

    try:
        with open_session(cfg, mt5_module=mt5_module, now=now) as s:
            if args.cmd == "check":
                tick = s.collector.last_tick_server_epoch()
                print(json.dumps({
                    "connected": s.client.is_alive(),
                    "instrument": canonical, "broker_symbol": s.symbol.name,
                    "digits": s.symbol.digits, "contract_size": s.symbol.contract_size,
                    "server_clock_mode": s.clock.mode,
                    "server_offset_now_hours": s.clock.offset_at(s.collector.now()).total_seconds() / 3600,
                    "last_tick_utc": str(s.clock.to_utc([tick])[0]) if tick else None,
                    "execution_mode": cfg.engine.execution_mode}, indent=2))
            elif args.cmd == "symbols":
                syms = discover_symbols(s.client)
                rows = [x for x in syms if args.all or "XAU" in x.name.upper() or "GOLD" in x.name.upper()]
                for x in rows:
                    print(f"{x.name:20s} digits={x.digits} contract={x.contract_size:g} {x.description}")
                print(f"{len(rows)} of {len(syms)} symbols; using {s.symbol.name} for {canonical}")
            elif args.cmd == "download":
                start, end = _range(args, cfg.data.download.default_days)
                dl = HistoricalDownloader(s.collector, store, canonical, cfg.data)
                results = [dl.download_bars(tf, start, end, resume=not args.no_resume)
                           for tf in _tfs(args.timeframes, cfg.engine.timeframes.collect)]
                path = write_report(results, f"reports/download_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}.json")
                for r in results:
                    print(f"{r.dataset:12s} stored={r.rows_stored:7d} quality={'OK' if r.quality.ok else 'ISSUES'} {r.quality.counts()}")
                print(f"Report: {path}")
            elif args.cmd == "ticks":
                start, end = _range(args, 1)
                dl = HistoricalDownloader(s.collector, store, canonical, cfg.data)
                r = dl.download_ticks(start, end)
                print(f"{r.dataset}: {r.rows_stored} ticks, quality={'OK' if r.quality.ok else 'ISSUES'} {r.quality.counts()}")
            elif args.cmd == "latest":
                tf = Timeframe.parse(args.timeframe)
                print(s.collector.latest_bars(tf, args.count).to_string(index=False))
    except (MT5Error, SymbolResolutionError) as exc:
        log.error("command_failed", extra={"cmd": args.cmd, "error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> None:  # pragma: no cover - thin wrapper
    sys.exit(run())
