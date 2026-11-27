"""Command-line actions of the breakout module (wired into ``quant_engine.cli``)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import AppConfig
from ..data.quality import validate_bars
from ..data.storage import ParquetStore
from ..instruments import InstrumentSpec, load_instruments
from ..timeframes import Timeframe
from .backtest import run_backtest, write_report
from .config import BreakoutConfig, load_breakout_config
from .store import SignalStore


def add_parsers(sub: Any) -> None:
    """Add the breakout-* sub-commands."""
    b = sub.add_parser("breakout-backtest", help="backtest the long-only D1 breakout strategies on M5 data")
    b.add_argument("--dataset", help="dataset version id from the catalog (default: stored MT5 M5 bars)")
    b.add_argument("--start", help="first M5 candle to load (UTC date)")
    b.add_argument("--end", help="load candles before this UTC date")
    b.add_argument("--profile", help="instrument profile (default: breakout.yaml costs.instrument_profile)")
    b.add_argument("--out", help="report folder (default reports/breakout/<time>)")
    b.add_argument("--quick", action="store_true", help="skip walk-forward, neighbourhood and swap stress")
    r = sub.add_parser("breakout-run", help="paper signal service for the breakout strategies (never trades)")
    r.add_argument("--once", action="store_true", help="run one cycle and exit")
    r.add_argument("--poll", type=float, default=60.0, help="seconds between cycles")
    sub.add_parser("breakout-status", help="strategy states and stored breakout signals")


def spec_for(config_dir: str | None, bcfg: BreakoutConfig, profile: str | None = None) -> InstrumentSpec:
    """Instrument profile used for sizing and costs."""
    return load_instruments(config_dir).get(profile or bcfg.costs.instrument_profile or None)


def load_m5(cfg: AppConfig, dataset: str | None, start: str | None, end: str | None) -> tuple[pd.DataFrame, dict]:
    """M5 candles from a catalog version or the stored MT5 bars, plus a description."""
    s = pd.Timestamp(start, tz="UTC") if start else None
    e = pd.Timestamp(end, tz="UTC") if end else None
    if dataset:
        from ..data.catalog import DatasetCatalog
        cat = DatasetCatalog(cfg.data.catalog_root(), cfg.data.quality)
        return cat.load(dataset, s, e), {"dataset": f"catalog version {dataset}"}
    store = ParquetStore(cfg.data.storage.root)
    return store.read_bars(cfg.engine.instrument, Timeframe.M5, s, e), {
        "dataset": f"stored MT5 bars {store.bars_path(cfg.engine.instrument, Timeframe.M5)}"}


def backtest_cmd(cfg: AppConfig, config_dir: str | None, args: Any, out: Callable[[str], None] = print) -> int:
    bcfg = load_breakout_config(config_dir)
    spec = spec_for(config_dir, bcfg, args.profile)
    m5, info = load_m5(cfg, args.dataset, args.start, args.end)
    if m5.empty:
        out("No M5 candles found. Download MT5 history first (python -m quant_engine download --timeframes M5).")
        return 1
    q = validate_bars(m5, Timeframe.M5, cfg.data.quality, dataset="M5")
    info["quality"] = ("OK" if q.ok else "ISSUES REPORTED (bars used as they are, nothing patched)") + f" {q.counts()}"
    info["instrument_profile"] = spec.name
    info["instrument_guesses"] = spec.guesses()
    run = run_backtest(m5, bcfg, spec, data_info=info, extras=not args.quick)
    folder = Path(args.out or Path(bcfg.reports_dir) / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S"))
    write_report(run, folder)
    for sid, per in run.results["per_strategy"].items():
        mw, fh = per["main_window"], per["full_history"]
        out(f"{sid}: 2023+ {mw.get('trades', 0)} trades, PF {mw.get('profit_factor')}, {mw.get('expectancy_r')} R/trade | "
            f"full {fh.get('trades', 0)} trades, PF {fh.get('profit_factor')}, {fh.get('expectancy_r')} R/trade")
    out(f"Swap: {run.swap_source}")
    out(f"Report: {folder / 'report.md'}")
    return 0


def status_cmd(cfg: AppConfig, config_dir: str | None, out: Callable[[str], None] = print) -> int:
    from .service import state_path
    bcfg = load_breakout_config(config_dir)
    p = state_path(bcfg)
    if p.exists():
        st = json.loads(p.read_text(encoding="utf-8"))
        for s in st["strategies"]:
            out(f"{s['name']}: {s['state_label']} (armed {s['armed']}), next level {s['next_level']}, "
                f"last close {s['last_d1_close']}")
    else:
        out("No state yet: run python -m quant_engine breakout-run --once on the PC.")
    store = SignalStore(bcfg.store.path)
    for r in store.all(20):
        out(f"{r['day']} {r['strategy_id']:16s} {r['status']:26s} entry {r['entry'] or r['entry_est']:.2f} "
            f"lots {r['suggested_lots']} {r['risk_reason'] or r['block_reason'] or ''}")
    out(json.dumps(store.performance()))
    return 0


def run_cmd(cfg: AppConfig, config_dir: str | None, session: Any, args: Any, now: Callable[[], pd.Timestamp] | None,
            out: Callable[[str], None] = print) -> int:
    """Start the paper service on an open MT5 session (live spec read from the broker)."""
    from .service import BreakoutService
    bcfg = load_breakout_config(config_dir)
    if bcfg.mode == "off":
        out("breakout.yaml mode is off: nothing to do.")
        return 0
    base = spec_for(config_dir, bcfg)
    info = session.client.call("symbol_info", session.symbol.name)
    acct = session.client.call("account_info", allow_none=True)
    spec = InstrumentSpec.from_symbol_info(info, getattr(acct, "currency", "") or "", name="mt5_live", base=base)
    for w in spec.consistency_warnings():
        out(f"Warning: {w}")

    def equity() -> float | None:
        a = session.client.call("account_info", allow_none=True)
        return float(getattr(a, "equity", 0) or getattr(a, "balance", 0) or 0) or None

    def spread_now() -> float | None:
        t = session.client.call("symbol_info_tick", session.symbol.name, allow_none=True)
        return None if t is None else round(float(t.ask) - float(t.bid), 5)

    notifier = None
    if bcfg.mode == "live":
        from ..notify.telegram import TelegramNotifier
        notifier = TelegramNotifier.from_env()
    svc = BreakoutService(bcfg, session.collector, spec, SignalStore(bcfg.store.path), equity=equity,
                          spread_now=spread_now, notifier=notifier, now=now, point=session.symbol.point or None)
    if args.once:
        out(json.dumps(svc.cycle(), default=str))
        return 0
    out(f"Breakout {bcfg.mode} mode on {session.symbol.name} (signals only, never trades). Ctrl+C to stop.")
    try:
        svc.run_forever(args.poll)
    except KeyboardInterrupt:
        out("Stopped.")
    return 0
