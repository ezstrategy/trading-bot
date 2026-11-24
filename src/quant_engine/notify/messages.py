"""Telegram message texts (HTML). Follow-ups are sent as replies to the signal."""
from __future__ import annotations

import html

import pandas as pd

from ..strategy import lifecycle as lc
from ..strategy.entry import Signal

IST = "Asia/Kolkata"
KZ_NAME = {"london_kz": "London", "newyork_kz": "New York"}
MODE_NAME = {"bullish": "Uptrend · buy the dip", "bearish": "Downtrend · sell the rally",
             "sideways": "Range · fade the false breakout", "v1": "Sweep reversal"}


def _px(x: float | None, digits: int) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def signal_text(s: Signal, symbol: str, lots: float, risk_pct: float, digits: int = 2,
                expiry_minutes: int = 30, size_note: str = "") -> str:
    """The signal message."""
    head = "🟢 <b>BUY LIMIT</b>" if s.direction == 1 else "🔴 <b>SELL LIMIT</b>"
    p = lambda x: _px(x, digits)  # noqa: E731
    tps = [f"<b>TP1:</b> {p(s.tp1)} ({s.r_multiple(s.tp1):.1f}R) close 50%, then stop to breakeven",
           f"<b>TP2:</b> {p(s.tp2)} ({s.r_multiple(s.tp2):.1f}R)"]
    if s.tp3 is not None:
        tps.append(f"<b>TP3:</b> {p(s.tp3)} ({s.r_multiple(s.tp3):.1f}R)")
    local = s.time.tz_convert(IST)
    lines = [
        f"{head} {html.escape(symbol)}  ·  Grade {s.grade} ({s.score}/100)",
        f"<i>{MODE_NAME.get(s.strategy, s.strategy)}</i>",
        "",
        f"<b>Entry:</b> {p(s.entry)}",
        f"<b>Stop loss:</b> {p(s.stop_loss)}  ({s.risk:.2f} = 1R)",
        *tps,
        f"<b>Size:</b> {lots:g} lot ({risk_pct:g}% risk)" + (f"  ⚠️ {html.escape(size_note)}" if size_note else ""),
        f"<b>Valid for:</b> {expiry_minutes} min (until {(local + pd.Timedelta(minutes=expiry_minutes)):%H:%M} IST)",
        "",
        "<b>Why:</b>",
        *[f"• {html.escape(r, quote=False)}" for r in s.reasons],
        "",
        f"<i>{local:%d %b %H:%M} IST · {KZ_NAME.get(s.kill_zone, s.kill_zone)} session · signal only, "
        f"no trade was placed</i>",
    ]
    return "\n".join(lines)


def event_text(ev: lc.TradeEvent, s: Signal, digits: int = 2) -> str:
    """Follow-up message for a lifecycle event."""
    p = lambda x: _px(x, digits)  # noqa: E731
    be = s.entry + s.direction * s.spread
    side = "low" if s.direction == 1 else "high"
    return {
        lc.FILLED: f"✅ Filled at {p(s.entry)}",
        lc.TP1: f"🎯 TP1 hit (+{s.r_multiple(s.tp1):.1f}R). Close 50%, move stop to {p(be)}",
        lc.TP2: f"🎯 TP2 hit (+{s.r_multiple(s.tp2):.1f}R)" + (". Close 25%, TP3 left" if s.tp3 is not None else ". Trade closed"),
        lc.TP3: f"🏁 TP3 hit (+{s.r_multiple(s.tp3):.1f}R). Trade closed" if s.tp3 is not None else "TP3 hit",
        lc.STOP: "❌ Stopped out (−1R)",
        lc.BREAKEVEN: f"➖ Closed at breakeven after TP1 (total {ev.r:+.2f}R)",
        lc.TIME_STOP: f"⏱ Close at market ({p(ev.price)}): {(s.time_stop_candles or 24) * 5} min without TP1 "
                      f"({ev.r:+.2f}R)",
        lc.SESSION_END: f"🔔 Session over: close at market ({p(ev.price)}), {ev.r:+.2f}R",
        lc.EXPIRED: "⌛ Cancelled: not filled within 30 min",
        lc.CANCEL_TP1: "🚫 Cancelled: price reached TP1 before the entry filled",
        lc.CANCEL_SWEEP: f"🚫 Cancelled: price closed beyond the sweep {side}",
    }.get(ev.kind, ev.kind)


def news_alert_text(title: str, minutes: int, filled: bool) -> str:
    action = "Take profit or move the stop to breakeven" if filled else "Consider cancelling the pending order"
    return f"📰 {html.escape(title)} in {minutes} min. {action}."


def daily_summary_text(day: str, rows: list[dict], week_r: float) -> str:
    """rows: dicts with time, side, grade, outcome, r."""
    if not rows:
        body = ["No signals today."]
    else:
        body = [f"• {r['time']} {r['side']} {r['grade']}: {html.escape(r['outcome'])}"
                + (f" ({r['r']:+.2f}R)" if r.get("r") is not None else "") for r in rows]
    total = sum(r["r"] for r in rows if r.get("r") is not None)
    return "\n".join([f"📊 <b>Daily summary {day}</b>", *body, "",
                      f"Today: {total:+.2f}R · This week: {week_r:+.2f}R"])
