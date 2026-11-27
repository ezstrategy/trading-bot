"""Paper / live signal service for the breakout strategies (signals only, never trades).

Each :meth:`BreakoutService.cycle`:

1. reads the closed M5 candles of the last ``lookback_days`` from MT5 (the forming candle is
   never used) and checks the data is current;
2. rebuilds the D1 candles that are final now and replays both strategies with the same code as
   the backtest (:func:`.scan.scan`), so live and historical signals follow identical rules;
3. stores each breakout of the last ``max_signal_age_hours`` that is not stored yet: signals get
   a suggested size from the shared :class:`~.risk.RiskEngine` (broker specification from MT5,
   equity from the account), blocked breakouts are stored with their reason;
4. updates every unfinished stored signal from the replay (entry known, paper TP/SL/expiry);
5. writes the per-strategy state for the dashboard; in ``live`` mode only, new signals are also
   sent to Telegram (``live`` is refused by the config until the owner's review).

MT5 errors are caught by the loop and retried on the next cycle; nothing is stored from a failed
or stale read.
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..instruments import InstrumentSpec
from ..logging_setup import get_logger
from ..mt5.client import MT5Error
from ..timeframes import Timeframe
from .backtest import costs_for, scan_slip
from .config import BreakoutConfig
from .daily import NY, session_number
from .risk import Exposure, RiskCandidate, RiskEngine
from .rules import strategies
from .scan import BLOCKED, GENERATED, BreakoutEvent, prepare, scan
from .store import ACTIVE, AWAITING, BUY, SignalStore

log = get_logger(__name__)
LOOKBACK_DAYS = 120   # > 1 month + 20 sessions + ATR warm-up + a 5-session trade


def market_expected_open(now: pd.Timestamp) -> bool:
    """True when gold normally trades (Sun 18:00 - Fri 17:00 New York, minus the daily 17-18 break)."""
    ny = pd.Timestamp(now).tz_convert(NY)
    wd, hm = ny.weekday(), ny.hour + ny.minute / 60
    if wd == 5 or (wd == 4 and hm >= 17) or (wd == 6 and hm < 18):
        return False
    return not (17 <= hm < 18)


def state_path(cfg: BreakoutConfig) -> Path:
    return Path(cfg.store.path).with_name("state.json")


class BreakoutService:
    """Generates and tracks breakout signals from live MT5 data."""

    def __init__(self, cfg: BreakoutConfig, collector: Any, spec: InstrumentSpec, store: SignalStore, *,
                 equity: Callable[[], float | None] | None = None, spread_now: Callable[[], float | None] | None = None,
                 notifier: Any | None = None, now: Callable[[], pd.Timestamp] | None = None,
                 point: float | None = None, lookback_days: int = LOOKBACK_DAYS) -> None:
        if cfg.mode == "off":
            raise ValueError("breakout mode is off in breakout.yaml")
        self.cfg, self.collector, self.spec, self.store = cfg, collector, spec, store
        self.equity = equity or (lambda: None)
        self.spread_now = spread_now or (lambda: None)
        self.notifier = notifier if cfg.mode == "live" else None      # paper mode never sends
        self.now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self.point = point or spec.point
        self.lookback = lookback_days
        self.risk = RiskEngine(cfg.risk, spec)

    def cycle(self) -> dict:
        """One pass (see module docstring). Returns what happened."""
        now = self.now()
        out: dict = {"time": now.isoformat(), "mode": self.cfg.mode, "new": [], "blocked": [], "updated": 0,
                     "data_ok": True, "note": ""}
        m5 = self.collector.bars(Timeframe.M5, now - pd.Timedelta(days=self.lookback), now)
        if m5.empty:
            out.update(data_ok=False, note="MT5 returned no M5 candles")
            log.warning("breakout_no_data", extra=out)
            return out
        last_close = pd.Timestamp(m5["time"].iat[-1]) + Timeframe.M5.delta
        age_min = (now - last_close).total_seconds() / 60
        if market_expected_open(now) and age_min > self.cfg.validation.max_data_age_minutes:
            out.update(data_ok=False, note=f"newest M5 candle closed {age_min:.0f} min ago: data not current")
            log.warning("breakout_stale_data", extra={"age_minutes": age_min})
            return out
        d1, m = prepare(m5, self.cfg, self.point, as_of=now)
        costs, swap_src = costs_for(self.cfg, self.spec)
        scans = {s.id: scan(s, d1, m, self.cfg, costs, as_of=now) for s in strategies(self.cfg)}
        by_key: dict[str, BreakoutEvent] = {e.key: e for r in scans.values() for e in r.events}
        # 1. new breakouts
        recent = [e for e in by_key.values() if e.kind in (GENERATED, BLOCKED)
                  and now - e.signal_time <= pd.Timedelta(hours=self.cfg.validation.max_signal_age_hours)
                  and not self.store.has(e.key)]
        recent.sort(key=lambda e: e.signal_time)
        spread_now = self.spread_now()
        for t0 in sorted({e.signal_time for e in recent}):
            batch = [e for e in recent if e.signal_time == t0]
            self._store_batch(batch, now, spread_now, swap_src, out)
        # 2. follow the unfinished ones
        for r in self.store.open_signals():
            e = by_key.get(r["key"])
            if e is None or e.trade is None:
                continue
            before = r["status"]
            new = self.store.update_paper(r["key"], e.trade.to_dict())
            if new and new["status"] != before:
                out["updated"] += 1
        state = {"time": now.isoformat(), "mode": self.cfg.mode, "data_last_close": last_close.isoformat(),
                 "swap": swap_src, "strategies": [r.current() for r in scans.values()]}
        p = state_path(self.cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
        return out

    def _store_batch(self, batch: list[BreakoutEvent], now: pd.Timestamp, spread_now: float | None,
                     swap_src: str, out: dict) -> None:
        gen = [e for e in batch if e.kind == GENERATED]
        exposures = [Exposure(r["strategy_id"], session_number(pd.Timestamp(r["day"])),
                              float(r["risk_pct"] or 0), r["key"])
                     for r in self.store.open_signals() if r["generated"] and r["risk_reason"] == "ok"]
        equity = self.equity() or self.cfg.risk.account_equity_fallback
        cands = []
        for e in gen:
            entry, stop = self._levels(e)
            cands.append(RiskCandidate(e.strategy_id, session_number(e.day), entry, stop, scan_slip(e) if e.trade and e.trade.entry_time
                                       else self.cfg.costs.slippage_points * self.spec.tick_size, e.key))
        decisions = dict(zip([e.key for e in gen], self.risk.batch(cands, exposures, equity)))
        for e in batch:
            entry, stop = self._levels(e)
            p = next(s for s in strategies(self.cfg) if s.id == e.strategy_id).params
            target = entry + p.target_r * (entry - stop) if stop == stop else None
            d = decisions.get(e.key)
            t = e.trade
            row = {"key": e.key, "source": self.cfg.mode, "strategy_id": e.strategy_id, "strategy_name": e.strategy_name,
                   "version": e.version, "rule_hash": e.rule_hash, "side": e.side, "day": f"{e.day:%Y-%m-%d}",
                   "signal_time": e.signal_time.isoformat(), "created_at": now.isoformat(),
                   "breakout_level": e.level, "d1_close": e.d1_close, "atr": e.atr,
                   "entry_est": entry, "stop_est": stop, "target_est": target, "rr": p.target_r,
                   "spread_at_signal": spread_now, "equity": equity,
                   "est_costs": {"spread_floor": self.cfg.costs.spread_floor,
                                 "slippage_per_fill": self.cfg.costs.slippage_points * self.spec.tick_size,
                                 "commission_per_lot": self.spec.commission_per_lot, "swap": swap_src},
                   "generated": int(e.kind == GENERATED), "block_reason": "; ".join(e.failed_checks) or None,
                   "explanation": e.explanation or "; ".join(e.failed_checks), "checks": e.checks,
                   "snapshot": e.snapshot}
            if d is not None:
                row.update({"suggested_lots": d.lots, "risk_pct": round(d.risk_pct, 3), "risk_money": round(d.risk_money, 2),
                            "risk_reason": d.reason})
            if t is not None and t.entry_time is not None and e.kind == GENERATED:
                row.update({"entry_time": t.entry_time.isoformat(), "entry": t.entry, "stop": t.stop,
                            "target": t.target, "spread_entry": t.spread, "paper_status": t.status})
                if t.is_final:
                    row.update({"paper_exit_time": t.exit_time.isoformat(), "paper_exit": t.exit, "paper_r": t.r})
            if self.store.insert(row):
                (out["new"] if e.kind == GENERATED else out["blocked"]).append(e.key)
                log.info("breakout_signal_stored", extra={"key": e.key, "kind": e.kind, "risk": row.get("risk_reason"),
                                                          "lots": row.get("suggested_lots")})
                if self.notifier is not None and e.kind == GENERATED and row.get("risk_reason") == "ok":
                    self.notifier.send(signal_text(self.store.get(e.key)))

    def _levels(self, e: BreakoutEvent) -> tuple[float, float]:
        """Entry and stop: actual when the entry candle is known, else estimated from the D1 close."""
        t = e.trade
        if t is not None and t.entry_time is not None and t.stop == t.stop:
            return t.entry, t.stop
        p = next(s for s in strategies(self.cfg) if s.id == e.strategy_id).params
        entry = e.d1_close + self.cfg.costs.spread_floor + self.cfg.costs.slippage_points * self.spec.tick_size
        return entry, entry - p.stop_atr_mult * e.atr

    def run_forever(self, poll_seconds: float = 60.0, stop: Callable[[], bool] = lambda: False) -> None:
        """Cycle until ``stop()``; MT5 errors are logged and retried on the next cycle."""
        log.info("breakout_service_started", extra={"mode": self.cfg.mode})
        while not stop():
            try:
                res = self.cycle()
                if res["new"] or res["blocked"] or res["updated"] or not res["data_ok"]:
                    log.info("breakout_cycle", extra=res)
            except MT5Error as exc:
                log.error("breakout_mt5_error", extra={"error": str(exc)})
            except Exception as exc:  # keep running; the error is logged with context
                log.exception("breakout_cycle_failed", extra={"error": str(exc)})
            _time.sleep(poll_seconds)


def signal_text(r: dict) -> str:
    """Telegram text of a stored signal (live mode only)."""
    ist = pd.Timestamp(r["signal_time"]).tz_convert("Asia/Kolkata")
    entry = r.get("entry") or r.get("entry_est")
    stop = r.get("stop") or r.get("stop_est")
    tgt = r.get("target") or r.get("target_est")
    return (f"🟢 BUY XAUUSD — {r['strategy_name']} ({r['version']})\n"
            f"Breakout D1 {r['day']} closed {r['d1_close']:.2f} > level {r['breakout_level']:.2f}\n"
            f"Entry: next M5 open (~{entry:.2f})\nStop: {stop:.2f}  (1.5 x ATR {r['atr']:.2f})\nTarget: {tgt:.2f} (2R)\n"
            f"Lots: {r.get('suggested_lots')} = {r.get('risk_pct')}% risk\nSignal {ist:%d %b %H:%M} IST · manual execution only")
