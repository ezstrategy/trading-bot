"""Command line interface.

    python -m quant_engine check
    python -m quant_engine symbols [--all]
    python -m quant_engine download [--timeframes M1,M5,M15,H1,H4] [--start 2026-08-01] [--end ...] [--days 90]
    python -m quant_engine ticks --start 2026-11-20 [--end 2026-11-21]
    python -m quant_engine validate [--timeframes M5]
    python -m quant_engine latest --timeframe M5 [--count 10]
    python -m quant_engine context [--at "2026-11-24 13:05"] [--point 0.01]
    python -m quant_engine dashboard [--port 8501]
    python -m quant_engine dukascopy --start 2026-05-01 [--end 2026-11-24]   (free history, no MT5)
    python -m quant_engine backtest [--source mt5|dukascopy] [--mode regime|v1|bullish|bearish|sideways|compare]
    python -m quant_engine patterns [--source mt5|dukascopy] [--timeframe M5|H1|H4]   (find repeatable patterns)
    python -m quant_engine bigmoves [--start 2020-01-01]   (30-50 pip moves in 30 min: causes + strategy test)
    python -m quant_engine entries [--start 2020-01-01]    (your dip/sweep + reversal candle entry, backtested)
    python -m quant_engine classics [--start 2020-01-01]   (well-known public strategies, backtested)
    python -m quant_engine run [--dry-run] [--once]      (live signals to Telegram; never trades)
    python -m quant_engine news                           (refresh the economic calendar, show blocks)
    python -m quant_engine telegram-test                  (send a test message)
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
    dk = sub.add_parser("dukascopy", help="download free Dukascopy M1 history and build M5-H4 bars")
    dk.add_argument("--start")
    dk.add_argument("--end")
    dk.add_argument("--days", type=int)
    bt = sub.add_parser("backtest", help="backtest the entry model on stored bars (no MT5 needed)")
    bt.add_argument("--source", choices=["mt5", "dukascopy"], default="mt5",
                    help="mt5 = bars from 'download'; dukascopy = bars from 'dukascopy'")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--point", type=float, default=0.01, help="tick size (XAUUSD: 0.01)")
    bt.add_argument("--mode", choices=["regime", "v1", "bullish", "bearish", "sideways", "patterns", "compare"],
                    help="strategy to test (default: strategy.mode in config); compare = all modes x markets")
    bt.add_argument("--out", help="report folder (default reports/backtest_<source>_<time>)")
    pt = sub.add_parser("patterns", help="find repeatable patterns per market type and test them on unseen data")
    pt.add_argument("--source", choices=["mt5", "dukascopy"], default="mt5")
    pt.add_argument("--start")
    pt.add_argument("--end")
    pt.add_argument("--point", type=float, default=0.01, help="tick size (XAUUSD: 0.01)")
    pt.add_argument("--timeframe", choices=["M5", "H1", "H4"], default="M5",
                    help="M5 = scalps; H1 / H4 = swing trades with bigger stops and targets")
    pt.add_argument("--price-action", action="store_true",
                    help="test only price-action setups (pin bars, engulfing, inside bars, at levels or not)")
    pt.add_argument("--out", help="report folder (default reports/patterns_<source>_<timeframe>_<time>)")
    bg = sub.add_parser("bigmoves", help="big fast moves: what comes before them + a simple strategy test")
    bg.add_argument("--source", choices=["mt5", "dukascopy"], default="mt5")
    bg.add_argument("--start", default="2020-01-01")
    bg.add_argument("--end")
    bg.add_argument("--point", type=float, default=0.01)
    bg.add_argument("--targets", help="target pips, comma separated (e.g. 10,20)")
    bg.add_argument("--stops", help="stop pips, comma separated (e.g. 5,10)")
    bg.add_argument("--windows", help="time limits in M5 candles, comma separated (e.g. 6,12)")
    bg.add_argument("--move-pips", help="move sizes for the 'how often' steps (e.g. 10,20)")
    bg.add_argument("--spread-points", type=float, help="spread charged per trade in points (gold: 8 = $0.08)")
    bg.add_argument("--out")
    en = sub.add_parser("entries", help="the owner's entry: sweep of a 30-min low/high + reversal candle")
    en.add_argument("--source", choices=["mt5", "dukascopy"], default="mt5")
    en.add_argument("--start", default="2020-01-01")
    en.add_argument("--end")
    en.add_argument("--point", type=float, default=0.01)
    en.add_argument("--spread-points", type=float, default=8.0, help="spread per trade in points (8 = $0.08)")
    en.add_argument("--example", action="append", default=None,
                    help='your trade to check, "YYYY-MM-DD HH:MM,BUY" in UTC (repeatable)')
    en.add_argument("--out")
    cl = sub.add_parser("classics", help="backtest well-known public strategies (London breakout, ORB, RSI2, ...)")
    cl.add_argument("--source", choices=["mt5", "dukascopy"], default="mt5")
    cl.add_argument("--start", default="2020-01-01")
    cl.add_argument("--end")
    cl.add_argument("--point", type=float, default=0.01)
    cl.add_argument("--spread-points", type=float, default=8.0, help="spread per trade in points (8 = $0.08)")
    cl.add_argument("--out")
    rn = sub.add_parser("run", help="live signal engine: watch MT5, send signals to Telegram (never trades)")
    rn.add_argument("--dry-run", action="store_true", help="print messages instead of sending to Telegram")
    rn.add_argument("--once", action="store_true", help="run one cycle and exit")
    sub.add_parser("news", help="refresh the economic calendar and list upcoming blocks")
    sub.add_parser("telegram-test", help="send a test message to the Telegram chat in .env")
    db = sub.add_parser("dashboard", help="open the local web dashboard")
    db.add_argument("--port", type=int, default=8501)
    return p


def _dashboard(config_dir: str | None, port: int) -> int:
    """Launch Streamlit on localhost only."""
    import os
    import subprocess
    from pathlib import Path
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    env = dict(os.environ)
    if config_dir:
        env["QE_CONFIG_DIR"] = config_dir
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "localhost",
           "--server.port", str(port), "--browser.gatherUsageStats", "false",
           "--client.toolbarMode", "minimal"]
    print(f"Dashboard: http://localhost:{port}  (Ctrl+C to stop)")
    return subprocess.call(cmd, env=env)


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
    from .dashboard.services import snapshot_summary
    print(json.dumps(snapshot_summary(s), indent=2, default=str))
    return 0


def dukascopy_dataset(canonical: str) -> str:
    """Stored instrument name for Dukascopy bars (kept apart from broker data)."""
    return f"{canonical}_DUKASCOPY"


def _dukascopy(cfg, store: ParquetStore, canonical: str, args: argparse.Namespace,
               fetch: Callable[[str], bytes] | None = None,
        fetch_text: Callable[[str], str] | None = None) -> int:
    from pathlib import Path

    from .data import dukascopy as dk
    from .features.context import STRUCTURE_TFS
    start, end = _range(args, 180)
    folder = Path(cfg.data.storage.root) / "dukascopy" / canonical
    counts = dk.download_days(canonical, start.date(), end.date(), folder, fetch=fetch)
    print(f"Dukascopy {canonical}: {counts} -> {folder}")
    m1 = dk.load_folder(folder, assumed_spread_points=cfg.strategy.backtest.assumed_spread_points)
    if m1.empty:
        print("No candles downloaded.", file=sys.stderr)
        return 1
    ds = dukascopy_dataset(canonical)
    for tf in STRUCTURE_TFS:
        n = store.write_bars(ds, tf, dk.resample(m1, tf))
        print(f"{ds} {tf.value}: {n} bars stored")
    return 0 if counts["failed"] == 0 else 1


def _backtest(cfg, store: ParquetStore, canonical: str, args: argparse.Namespace) -> int:
    from .strategy.runner import run_backtest
    ds = dukascopy_dataset(canonical) if args.source == "dukascopy" else canonical
    note = ("Dukascopy XAU/USD spot, BID M1 candles (UTC), resampled to M5/M15/H1/H4; spread assumed"
            if args.source == "dukascopy" else f"MT5 broker bars for {canonical} (UTC)")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    mode = args.mode or cfg.strategy.mode
    out = args.out or f"reports/backtest_{args.source}_{mode}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
    try:
        summary = run_backtest(cfg, store, ds, args.point, out, start, end, note, mode)
    except ValueError as exc:
        hint = "python -m quant_engine dukascopy --days 180" if args.source == "dukascopy" \
            else "python -m quant_engine download"
        print(f"{exc}. Run: {hint}", file=sys.stderr)
        return 1
    print(json.dumps(summary["overall"], indent=2))
    print(f"Report: {out}/report.md  Trades: {out}/trades.csv")
    return 0


def _patterns(cfg, store: ParquetStore, canonical: str, args: argparse.Namespace) -> int:
    from .research.runner import run_patterns
    ds = dukascopy_dataset(canonical) if args.source == "dukascopy" else canonical
    note = ("Dukascopy XAU/USD spot, BID M1 candles (UTC), resampled; spread assumed"
            if args.source == "dukascopy" else f"MT5 broker bars for {canonical} (UTC)")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    out = args.out or f"reports/patterns_{args.source}_{args.timeframe}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
    try:
        from .research.events import PRICE_ACTION_FAMILIES
        summary = run_patterns(cfg, store, ds, args.point, out, start, end, note, timeframe=args.timeframe,
                               only_families=list(PRICE_ACTION_FAMILIES) if args.price_action else None)
    except ValueError as exc:
        hint = "python -m quant_engine dukascopy --days 365" if args.source == "dukascopy" \
            else "python -m quant_engine download --days 365"
        print(f"{exc}. Run: {hint}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    print(f"Report: {out}/report.md")
    return 0


def _news(cfg, fetch_text, now) -> int:
    from .features.news import NewsCalendar, refresh_calendar
    nc = cfg.strategy.news
    cal = NewsCalendar(refresh_calendar(nc, fetch_text), nc)
    t = (now or (lambda: pd.Timestamp.now(tz="UTC")))()
    nxt = [w for w in cal.windows if w.end > t][:15]
    print(f"{len(cal.windows)} blocking windows for {', '.join(nc.currencies)} in {nc.calendar_file}")
    for w in nxt:
        ist = lambda x: x.tz_convert("Asia/Kolkata").strftime("%a %d %b %H:%M")  # noqa: E731
        print(f"{ist(w.start)} - {ist(w.end)} IST  {w.tier:7s} {w.title}")
    return 0


def _run_engine(cfg, s, args, now, fetch_text) -> int:
    from .live.engine import SignalEngine
    from .notify.telegram import ConsoleNotifier, TelegramNotifier
    notifier = None if args.dry_run else TelegramNotifier.from_env()
    if notifier is None:
        print("Dry run: messages are printed here" + ("" if args.dry_run else
              " (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to send them to Telegram)"))
        notifier = ConsoleNotifier(print)

    def balance() -> float | None:
        info = s.client.call("account_info", allow_none=True)
        return float(getattr(info, "balance", 0) or 0) or None

    eng = SignalEngine(cfg, s.collector, s.symbol, notifier, balance=balance, now=now, news_fetch=fetch_text)
    if args.once:
        print(json.dumps(eng.cycle(), default=str))
        return 0
    print(f"Watching {s.symbol.name} on M5 (signals only, never trades). Ctrl+C to stop.")
    try:
        eng.run_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


def run(argv: Sequence[str] | None = None, mt5_module: Any | None = None,
        now: Callable[[], pd.Timestamp] | None = None,
        fetch: Callable[[str], bytes] | None = None,
        fetch_text: Callable[[str], str] | None = None) -> int:
    """Entry point used by ``main`` and by tests (with a fake MT5 module)."""
    args = build_parser().parse_args(argv)
    if args.cmd == "dashboard":
        return _dashboard(args.config, args.port)
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
    if args.cmd == "dukascopy":
        return _dukascopy(cfg, store, canonical, args, fetch)
    if args.cmd == "backtest":
        return _backtest(cfg, store, canonical, args)
    if args.cmd == "bigmoves":
        from .research.bigmoves import run_bigmoves
        ds = dukascopy_dataset(canonical) if args.source == "dukascopy" else canonical
        out = args.out or f"reports/bigmoves_{args.source}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
        try:
            summary = run_bigmoves(cfg, store, ds, args.point, out,
                                   pd.Timestamp(args.start, tz="UTC") if args.start else None,
                                   pd.Timestamp(args.end, tz="UTC") if args.end else None,
                                   f"{'MT5 broker' if args.source == 'mt5' else 'Dukascopy'} bars for {canonical} (UTC)",
                                   overrides={k: [float(x) if k != "window_bars" else int(x) for x in v.split(",")]
                                              for k, v in (("target_pips", args.targets), ("stop_pips", args.stops),
                                                           ("window_bars", args.windows), ("move_pips", args.move_pips))
                                              if v} | ({"spread_points": args.spread_points}
                                                       if args.spread_points is not None else {}))
        except ValueError as exc:
            print(f"{exc}. Run: python -m quant_engine download", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, default=str))
        print(f"Report: {out}/report.md")
        return 0
    if args.cmd == "classics":
        from .research.classics import run_classics
        ds = dukascopy_dataset(canonical) if args.source == "dukascopy" else canonical
        out = args.out or f"reports/classics_{args.source}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
        try:
            summary = run_classics(cfg, store, ds, args.point, out,
                                   pd.Timestamp(args.start, tz="UTC") if args.start else None,
                                   pd.Timestamp(args.end, tz="UTC") if args.end else None,
                                   f"{'MT5 broker' if args.source == 'mt5' else 'Dukascopy'} bars for {canonical} (UTC)",
                                   spread_points=args.spread_points)
        except ValueError as exc:
            print(f"{exc}. Run: python -m quant_engine download", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, default=str))
        print(f"Report: {out}/report.md")
        return 0
    if args.cmd == "entries":
        from .research.entries import OWNER_EXAMPLES, run_entries
        ds = dukascopy_dataset(canonical) if args.source == "dukascopy" else canonical
        out = args.out or f"reports/entries_{args.source}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}"
        examples = OWNER_EXAMPLES if args.example is None else [
            (e.split(",")[0].strip(), 1 if e.split(",")[1].strip().upper() == "BUY" else -1) for e in args.example]
        try:
            summary = run_entries(cfg, store, ds, args.point, out,
                                  pd.Timestamp(args.start, tz="UTC") if args.start else None,
                                  pd.Timestamp(args.end, tz="UTC") if args.end else None,
                                  f"{'MT5 broker' if args.source == 'mt5' else 'Dukascopy'} bars for {canonical} (UTC)",
                                  examples=examples, overrides={"spread_points": args.spread_points})
        except ValueError as exc:
            print(f"{exc}. Run: python -m quant_engine download", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, default=str))
        print(f"Report: {out}/report.md")
        return 0
    if args.cmd == "patterns":
        return _patterns(cfg, store, canonical, args)
    if args.cmd == "news":
        return _news(cfg, fetch_text, now)
    if args.cmd == "telegram-test":
        from .notify.telegram import TelegramNotifier
        tg = TelegramNotifier.from_env()
        if tg is None:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.", file=sys.stderr)
            return 1
        ok = tg.send("✅ XAUUSD Quant Engine connected. Signals only, no trades are placed.")
        print("Sent." if ok else "Telegram refused the message (check the token, chat id, and that you "
              "pressed Start in the bot chat).")
        return 0 if ok else 1

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
            elif args.cmd == "run":
                return _run_engine(cfg, s, args, now, fetch_text)
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
