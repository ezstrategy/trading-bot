"""Live signal engine: same rules as the backtest, run on each newly closed M5 candle.

Each cycle:
1. load closed M5/M15/H1/H4 candles from MT5 (UTC, the forming candle is never used);
2. when a new M5 candle has closed, run the entry model and risk locks exactly as
   the backtest does on the recent history, and send a signal if one fires on
   that candle (only one open signal at a time; never a stale signal);
3. replay every open signal through :func:`strategy.lifecycle.simulate` and
   send the new follow-ups (filled, TP1/breakeven, TP2/TP3, stop, expiry ...)
   as replies to the signal;
4. warn open signals before high-impact news, and send the daily summary.

SIGNALS ONLY: this module reads market data and sends messages. It never
places, modifies or closes orders.
"""
from __future__ import annotations

import time as _time
from typing import Any, Callable

import pandas as pd

from ..config import AppConfig
from ..features.context import STRUCTURE_TFS, MarketFeatures
from ..features.news import NewsCalendar, load_calendar, refresh_calendar
from ..logging_setup import get_logger
from ..mt5.client import MT5Error
from ..mt5.symbols import BrokerSymbol
from ..notify import messages as msg
from ..notify.telegram import Notifier
from ..strategy.backtest import Backtester
from ..strategy.lifecycle import simulate
from ..timeframes import Timeframe
from .sizing import lot_size
from .state import CLOSED, OPEN, PENDING, EngineState, SentSignal

log = get_logger(__name__)


class SignalEngine:
    """Watches the market and sends signals and follow-ups through ``notifier``."""

    def __init__(self, cfg: AppConfig, collector: Any, symbol: BrokerSymbol, notifier: Notifier, *,
                 balance: Callable[[], float | None] | None = None,
                 now: Callable[[], pd.Timestamp] | None = None,
                 news_fetch: Callable[[str], str] | None = None) -> None:
        self.cfg, self.collector, self.symbol, self.notifier = cfg, collector, symbol, notifier
        self.live = cfg.engine.live
        self.balance = balance or (lambda: None)
        self.now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self.news_fetch = news_fetch
        self.state = EngineState(self.live.state_dir)
        self.calendar = NewsCalendar(load_calendar(cfg.strategy.news.calendar_file), cfg.strategy.news)

    # ------------------------------------------------------------ helpers
    def _digits(self) -> int:
        return self.symbol.digits or 2

    def _send(self, kind: str, text: str, key: str | None = None, reply_to: int | None = None) -> int | None:
        mid = self.notifier.send(text, reply_to=reply_to)
        self.state.log(kind, text, self.now(), key, sent=mid is not None)
        return mid

    def _refresh_news(self, now: pd.Timestamp) -> None:
        nc = self.cfg.strategy.news
        if not nc.enabled:
            return
        last = pd.Timestamp(self.state.news_refreshed) if self.state.news_refreshed else None
        if last is None or now - last >= pd.Timedelta(minutes=nc.refresh_minutes):
            self.calendar = NewsCalendar(refresh_calendar(nc, self.news_fetch), nc)
            self.state.news_refreshed = now.isoformat()

    def load_bars(self, now: pd.Timestamp) -> dict[Timeframe, pd.DataFrame]:
        days = self.live.lookback_days
        return {tf: self.collector.bars(tf, now - pd.Timedelta(days=days.get(tf.value, 30)), now)
                for tf in STRUCTURE_TFS}

    # -------------------------------------------------------------- cycle
    def cycle(self) -> dict:
        """One pass. Returns what happened (for logs and tests)."""
        now = self.now()
        out = {"new_signal": None, "followups": 0, "alerts": 0, "summary": False, "candle": None}
        self._refresh_news(now)
        bars = self.load_bars(now)
        m5 = bars[Timeframe.M5]
        if not m5.empty and not any(b.empty for b in bars.values()):
            last_close = m5["time"].iat[-1] + Timeframe.M5.delta
            out["candle"] = str(last_close)
            if self.state.last_candle != last_close.isoformat():
                self._on_new_candle(bars, last_close, now, out)
                self.state.last_candle = last_close.isoformat()
        out["alerts"] = self._news_alerts(now)
        out["summary"] = self._daily_summary(now)
        self.state.save()
        return out

    def _on_new_candle(self, bars, last_close, now, out) -> None:
        cfg = self.cfg.strategy
        features = MarketFeatures(bars, cfg, self.symbol.point or 0.01)
        bt = Backtester(features, cfg, self.calendar if cfg.news.enabled else None)
        bt.run()
        m5 = bars[Timeframe.M5]
        t = pd.DatetimeIndex(m5["time"])
        h, l, c = (m5[k].to_numpy(dtype="float64") for k in ("high", "low", "close"))

        # follow-ups for signals already sent
        for sent in self.state.active():
            out["followups"] += self._follow(sent, t, h, l, c)

        # a new signal on the candle that just closed
        fresh = now - last_close <= pd.Timedelta(minutes=self.live.max_signal_age_minutes)
        for sig, outcome in bt.decisions:
            if sig.time != last_close or sig.time.isoformat() in self.state.signals:
                continue
            if outcome.startswith("grade B") or outcome.startswith("skipped"):
                self.state.log("logged", f"{sig.side} {sig.grade} {sig.score}: {outcome}", now, sig.time.isoformat(),
                               sent=False)
                continue
            if not fresh:
                log.warning("stale_signal_skipped", extra={"signal_time": str(sig.time), "now": str(now)})
                continue
            if self.state.active():
                self.state.log("logged", f"{sig.side} {sig.grade}: skipped, a signal is still open", now,
                               sig.time.isoformat(), sent=False)
                continue
            key = self._publish(sig, now)
            if key:
                out["new_signal"] = key

    def _publish(self, sig, now: pd.Timestamp | None = None) -> str | None:
        bal = self.balance() or self.live.balance_fallback
        risk = self.cfg.engine.risk
        costs, point = self.cfg.strategy.research.costs, self.symbol.point or 0.01
        # costs not inside entry / stop: spread floor + slippage on both fills (owner rule 2026-11-27)
        cost_per_unit = (costs.min_spread_points + 2 * costs.slippage_points) * point
        size = lot_size(bal, risk.per_trade_pct, sig.entry, sig.stop_loss,
                        self.symbol.contract_size or 100.0, self.symbol.volume_min or 0.01,
                        self.symbol.volume_step or 0.01, cost_per_unit=cost_per_unit,
                        commission_per_lot=costs.commission_per_lot, skip_if_over=risk.skip_if_min_lot_over)
        if size.skipped:
            # natural stop first: never move the stop or raise the risk to make a signal fit
            self.state.log("logged", f"{sig.side} {sig.grade}: {size.note}", now or self.now(),
                           sig.time.isoformat(), sent=False)
            log.info("signal_skipped_risk", extra={"signal_time": str(sig.time), "side": sig.side,
                                                   "balance": bal, "note": size.note})
            return None
        text = msg.signal_text(sig, self.symbol.name, size.lots, self.cfg.engine.risk.per_trade_pct,
                               self._digits(), self.cfg.strategy.entry.expiry_candles * 5, size.note)
        mid = self._send("signal", text, sig.time.isoformat())
        sent = SentSignal(sig, mid, size.lots, size.note)
        self.state.signals[sent.key] = sent
        log.info("signal_sent", extra={"signal_time": str(sig.time), "side": sig.side, "grade": sig.grade,
                                       "score": sig.score, "entry": sig.entry, "lots": size.lots})
        return sent.key

    def _follow(self, sent: SentSignal, t, h, l, c) -> int:
        s = sent.signal
        idx = t.get_indexer([s.time - Timeframe.M5.delta])[0]
        if idx < 0:
            sent.status, sent.outcome = CLOSED, "history no longer available"
            return 0
        res = simulate(s, t, h, l, c, self.cfg.strategy, idx)
        new = res.events[sent.events_sent:]
        for ev in new:
            self._send(ev.kind, msg.event_text(ev, s, self._digits()), sent.key, reply_to=sent.message_id)
        sent.events_sent = len(res.events)
        sent.status = OPEN if res.filled and res.is_open else PENDING if res.is_open else CLOSED
        if not res.is_open:
            sent.outcome, sent.r = res.outcome, round(res.r, 4)
        return len(new)

    def _news_alerts(self, now: pd.Timestamp) -> int:
        nc = self.cfg.strategy.news
        if not nc.enabled:
            return 0
        n = 0
        for sent in self.state.active():
            for w in self.calendar.upcoming(now, nc.alert_before_min):
                k = w.event_time.isoformat()
                if k in sent.alerts:
                    continue
                mins = max(1, int((w.event_time - now).total_seconds() // 60))
                self._send("news_alert", msg.news_alert_text(w.title, mins, sent.status == OPEN), sent.key,
                           reply_to=sent.message_id)
                sent.alerts.append(k)
                n += 1
        return n

    def _daily_summary(self, now: pd.Timestamp) -> bool:
        tz = self.live.summary_timezone
        local = now.tz_convert(tz)
        day = local.strftime("%Y-%m-%d")
        hh, mm = (int(x) for x in self.live.daily_summary_time.split(":"))
        if day in self.state.summaries or (local.hour, local.minute) < (hh, mm):
            return False
        rows, week_r = [], 0.0
        week_start = (local.normalize() - pd.Timedelta(days=local.weekday()))
        for s in self.state.signals.values():
            lt = s.signal.time.tz_convert(tz)
            if lt.strftime("%Y-%m-%d") == day:
                rows.append({"time": f"{lt:%H:%M}", "side": s.signal.side, "grade": s.signal.grade,
                             "outcome": s.outcome if s.status == CLOSED else s.status, "r": s.r})
            if lt >= week_start and s.r is not None:
                week_r += s.r
        self._send("summary", msg.daily_summary_text(day, rows, week_r))
        self.state.summaries.append(day)
        return True

    # --------------------------------------------------------------- loop
    def run_forever(self, stop: Callable[[], bool] = lambda: False) -> None:
        """Cycle every ``poll_seconds`` until ``stop()``; MT5 errors are logged and retried."""
        log.info("engine_started", extra={"symbol": self.symbol.name, "mode": self.cfg.engine.execution_mode})
        while not stop():
            try:
                res = self.cycle()
                if res["new_signal"] or res["followups"] or res["alerts"]:
                    log.info("cycle", extra=res)
            except MT5Error as exc:
                log.error("cycle_mt5_error", extra={"error": str(exc)})
            except Exception as exc:  # keep the loop alive; the error is logged with context
                log.exception("cycle_failed", extra={"error": str(exc)})
            _time.sleep(self.live.poll_seconds)
