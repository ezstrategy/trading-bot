"""Telegram message texts (HTML). Follow-ups are sent as replies to the signal."""
from __future__ import annotations

import html

import pandas as pd

from ..strategy import lifecycle as lc
from ..strategy.entry import Signal

IST = "Asia/Kolkata"
KZ_NAME = {"london_kz": "London", "newyork_kz": "New York"}
MODE_NAME = {"bullish": "Uptrend · buy the dip", "bearish": "Downtrend · sell the rally",
             "sideways": "Range · fade the false breakout", "v1": "Sweep reversal",
             "london": "London breakout · H4 trend"}


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


# ------------------------------------------------------------------ London breakout mode
def london_setup_text(d, symbol: str, sides: dict[int, tuple[float, str]], skipped: dict[int, str],
                      risk_pct: float, digits: int = 2, tested: bool = True) -> str:
    """Setup message sent when the Asian range closes (01:00 New York).

    ``d``: strategy.london.LondonDay; ``sides``: side -> (lots, size note) that fit the risk limit;
    ``skipped``: side -> reason for sides whose 0.01 lot would risk too much.
    """
    p = lambda x: _px(x, digits)  # noqa: E731
    local = d.setup_time.tz_convert(IST)
    close_local = d.close_time.tz_convert(IST)
    trend = {"bullish": "bullish (buys only)", "bearish": "bearish (sells only)",
             "neutral": "neutral (both sides, first breakout wins)"}.get(d.trend, d.trend)
    lines = [f"🟡 <b>LONDON BREAKOUT SETUP</b> {html.escape(symbol)}",
             f"<i>{MODE_NAME['london']}</i>", "",
             f"<b>Asian range:</b> {p(d.range_low)} – {p(d.range_high)} ({d.range_high - d.range_low:.2f})",
             f"<b>H4 trend:</b> {trend}", ""]
    for side, lvl in sorted(d.levels.items(), reverse=True):
        head = "🟢 <b>BUY STOP</b>" if side == 1 else "🔴 <b>SELL STOP</b>"
        lines.append(f"{head} {p(lvl.entry)}")
        lines.append(f"   Stop {p(lvl.stop)} ({lvl.risk:.2f} = 1R, other side of the range) · "
                     f"Target {p(lvl.target)} ({abs(lvl.target - lvl.entry) / lvl.risk:g}R)")
        if side in sides:
            lots, note = sides[side]
            lines.append(f"   Size: {lots:g} lot ({risk_pct:g}% max risk, costs incl.)"
                         + (f" ⚠️ {html.escape(note)}" if note else ""))
        else:
            lines.append(f"   ⛔ Skipped: {html.escape(skipped.get(side, ''))}")
    lines.append("")
    if len(d.levels) == 2:
        lines.append("One cancels the other: when one side fills, cancel the other order.")
    else:
        other = "below " + p(d.range_low) if 1 in d.levels else "above " + p(d.range_high)
        lines.append(f"If price breaks {other} first, there is no trade today: cancel the order.")
    lines += ["Stop to breakeven at +1R. Close anything still open at "
              f"{close_local:%H:%M} IST on {close_local:%d %b} (16:00 New York).", ""]
    if not tested:
        lines += ["⚠️ <b>Settings differ from the tested version</b> (strategy.london in config): untested.", ""]
    lines.append(f"<i>{local:%d %b %H:%M} IST · paper signal, research stage (not validated) · "
                 f"no trade was placed</i>")
    return "\n".join(lines)


def london_event_text(ev, d, digits: int = 2) -> str:
    """Follow-up for a :class:`strategy.london.Event`."""
    from ..strategy import london as lb
    p = lambda x: _px(x, digits)  # noqa: E731
    side = "BUY" if d.side == 1 else "SELL"
    r = "" if ev.r is None else f"{ev.r:+.2f}R"
    return {
        lb.TRIGGERED: f"✅ {side} triggered at {p(ev.price)} · stop {p(d.initial_stop)} · target {p(d.target)}. "
                      f"Cancel the other order.",
        lb.BREAKEVEN: f"➖ +1R reached: move the stop to breakeven ({p(ev.price)})",
        lb.TARGET: f"🏁 Target hit at {p(ev.price)} ({r}). Trade closed",
        lb.STOP: f"❌ Stopped out at {p(ev.price)} ({r})",
        lb.BE_STOP: f"➖ Closed at breakeven ({r})",
        lb.SESSION_CLOSE: f"🔔 16:00 New York: close at market ({p(ev.price)}), {r}",
        lb.CANCELLED: f"🚫 Cancelled: {html.escape(ev.note)}",
    }.get(ev.kind, ev.kind)


def london_skipped_text(d, symbol: str, reasons: dict[int, str]) -> str:
    """Owner rule: report a signal skipped because even 0.01 lot would risk too much."""
    local = d.setup_time.tz_convert(IST)
    why = "; ".join(f"{'BUY' if s == 1 else 'SELL'}: {r}" for s, r in sorted(reasons.items(), reverse=True))
    return (f"⛔ <b>London breakout skipped</b> {html.escape(symbol)} · {local:%d %b} IST\n"
            f"Asian range {d.range_low:.2f} – {d.range_high:.2f}. The natural stop is too wide for this account: "
            f"{html.escape(why)}.\n<i>The stop is never tightened to fit (owner rule).</i>")
