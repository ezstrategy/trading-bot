"""Well-known public strategies, coded as published, backtested on stored gold data.

``python -m quant_engine classics`` (find_classics.bat). No tuning: each strategy uses the
rules from its source (listed in :data:`SOURCES`). Every trade pays the spread (buys fill
at the ask = bid + spread, sells exit at the ask). Results are split into the older data
and the newer, unseen data, and compared with simply holding gold over the same period.

Trades are simulated on bid candles. Inside a candle the stop is assumed to be hit before
the target (pessimistic). A stop order that fills inside a candle can be stopped out in the
same candle but cannot reach its target there (pessimistic).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.storage import ParquetStore
from ..features.sessions import ny_minutes, trading_day
from ..logging_setup import get_logger
from ..strategy.regime import trend_regime_series
from ..timeframes import Timeframe
from .events import regime_per_bar
from .miner import t_stats
from .model import regime_kwargs
from .runner import _table

log = get_logger(__name__)

SOURCES = {
    "London breakout (Asian range)": "mql5.com/en/blogs/post/777540: Asian range 03:00-08:00 server time, stop orders "
                                     "beyond it, stop at the other side (max 0.6% of price), target 2.5R, break-even "
                                     "at 1.5R, close 21:00 server, range 0.30-1.60% of price",
    "NY opening-range breakout (09:30)": "Zarattini & Aziz (SSRN 4416622) as replicated at mql5.com/en/blogs/post/776235: "
                                         "direction of the first 5-min candle, enter at the next open, stop at its "
                                         "other side, target 10R or the session close",
    "Gold opening-range breakout (08:20 COMEX)": "same rule as above on gold's own COMEX open (08:20 New York), as "
                                                 "in tradingview.com/script/m0xVQspL (Gold ORB)",
    "ICT Silver Bullet (10-11 NY)": "fxopen.com/blog/en/what-is-the-ict-silver-bullet-strategy-and-how-does-it-work: "
                                    "in 10:00-11:00 New York, first fair value gap in the higher-timeframe direction, "
                                    "limit at its edge, stop beyond the FVG candles, 1:2, close by end of day",
    "Gold overnight (buy 16:00, sell 09:30 NY)": "quantifiedstrategies.substack.com/p/a-quantitative-look-at-the-gold-"
                                                 "overnight: buy at the US close, sell at the next US open",
    "Gold US day session (09:30-16:00 NY)": "the other half of the same study (said to be flat)",
    "Connors RSI(2) (daily)": "chartschool.stockcharts.com RSI(2): above the 200-day SMA buy RSI(2) <= 5, below it "
                              "sell RSI(2) >= 95, exit on a close beyond the 5-day SMA, no stop",
    "Turtle System 1 (daily, 20/10)": "Turtle rules (e.g. ig.com turtle trading rules): buy a 20-day high / sell a "
                                      "20-day low, stop 2N (N = 20-day ATR), exit on a 10-day low / high",
    "50/200-day MA cross (daily)": "classic golden / death cross: long while SMA50 > SMA200, short while below",
}

# GUESS: the London-breakout author's "server time" is the usual GMT+2/+3 broker clock, which stays
# 7 hours ahead of New York all year, so 03:00 / 08:00 / 21:00 server = 20:00 / 01:00 / 14:00 New York.
LB_RANGE = (20 * 60, 1 * 60)
LB_CLOSE = 14 * 60
LB_BUFFER = 0.0002          # GUESS: "small buffer" = 0.02% of price (about $0.80 at $4,000)


def _walk(i0: int, end_i: int, side: int, entry: float, stop: float, target: float,
          o, h, l, c, sp: float, be_r: float | None = None, fill_bar_target: bool = True) -> tuple[int, float]:
    """Follow one trade from candle ``i0`` to ``end_i`` (inclusive). Returns (exit index, exit price).
    Buys exit at the bid, sells at the ask (bid + spread)."""
    risk = abs(entry - stop)
    for i in range(i0, end_i + 1):
        if side == 1:
            if l[i] <= stop:
                return i, min(o[i], stop) if i > i0 else stop
            if (fill_bar_target or i > i0) and h[i] >= target:
                return i, max(o[i], target) if i > i0 else target
            if be_r is not None and h[i] - entry >= be_r * risk:
                stop = max(stop, entry)
        else:
            if h[i] + sp >= stop:
                return i, max(o[i] + sp, stop) if i > i0 else stop
            if (fill_bar_target or i > i0) and l[i] + sp <= target:
                return i, min(o[i] + sp, target) if i > i0 else target
            if be_r is not None and entry - (l[i] + sp) >= be_r * risk:
                stop = min(stop, entry)
    return end_i, c[end_i] + (sp if side == -1 else 0.0)


def _trade(t, i_in, i_out, side, entry, exit_, stop=np.nan) -> dict:
    risk = abs(entry - stop) if np.isfinite(stop) else np.nan
    return {"entry_time": t[i_in], "exit_time": t[i_out], "side": "BUY" if side == 1 else "SELL",
            "entry": round(entry, 2), "exit": round(exit_, 2), "stop": round(stop, 2) if np.isfinite(stop) else np.nan,
            "r": side * (exit_ - entry) / risk if risk and risk > 0 else np.nan,
            "ret_pct": 100 * side * (exit_ - entry) / entry}


def _days(t: pd.DatetimeIndex, cfg) -> np.ndarray:
    return pd.Series(trading_day(t, cfg)).factorize()[0]


LB_DEFAULT = {"target_r": 2.5, "be_r": 1.5, "close_min": LB_CLOSE, "range_filter": True,
              "stop": "range", "trend": "none", "buffer": LB_BUFFER}


def london_breakout(m5: pd.DataFrame, sp: float, cfg, params: dict | None = None,
                    regime: np.ndarray | None = None, day_groups: list[np.ndarray] | None = None) -> list[dict]:
    """Asian-range breakout with stop orders (see SOURCES). ``params`` overrides :data:`LB_DEFAULT`:
    ``target_r``, ``be_r`` (None = no break-even), ``close_min`` (NY minutes), ``range_filter``
    (0.30-1.60% of price), ``stop`` ("range" = other side, "mid" = middle of the range),
    ``trend`` ("none", "with" = only the H4 market direction, neutral days trade both ways)."""
    p = {**LB_DEFAULT, **(params or {})}
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins = ny_minutes(t)
    in_range = (mins >= LB_RANGE[0]) | (mins < LB_RANGE[1])
    active = (mins >= LB_RANGE[1]) & (mins < p["close_min"])
    if day_groups is None:
        day_groups = lb_day_groups(m5, cfg)
    out = []
    for idx in day_groups:
        r_idx, a_idx = idx[in_range[idx]], idx[active[idx]]
        if len(r_idx) < 50 or not len(a_idx):
            continue
        hi, lo = h[r_idx].max(), l[r_idx].min()
        px = c[r_idx[-1]]
        if p["range_filter"] and not 0.003 <= (hi - lo) / px <= 0.016:
            continue
        allowed = {1, -1}
        if p["trend"] == "with" and regime is not None:
            rg = regime[r_idx[-1]]
            allowed = {1} if rg == "bullish" else {-1} if rg == "bearish" else {1, -1}
        buy_lvl, sell_lvl = hi + p["buffer"] * px, lo - p["buffer"] * px
        mid = (hi + lo) / 2
        end_i = a_idx[-1]
        for i in a_idx:
            up, dn = h[i] + sp >= buy_lvl, l[i] <= sell_lvl
            if not (up or dn):
                continue
            if up and dn:       # both in one candle: take the side nearer the open
                up = abs(buy_lvl - o[i] - sp) <= abs(o[i] - sell_lvl)
            side = 1 if up else -1
            if side not in allowed:
                break           # OCO: the first breakout decides the day
            if side == 1:
                entry = max(buy_lvl, o[i] + sp)
                stop = max(lo if p["stop"] == "range" else mid, entry * (1 - 0.006))
            else:
                entry = min(sell_lvl, o[i])
                stop = min((hi if p["stop"] == "range" else mid) + sp, entry * (1 + 0.006))
            risk = abs(entry - stop)
            if risk <= 0 or (entry - stop) * side <= 0:
                break
            tgt = entry + side * p["target_r"] * risk
            j, px_out = _walk(i, end_i, side, entry, stop, tgt, o, h, l, c, sp, be_r=p["be_r"],
                              fill_bar_target=False)
            out.append(_trade(t, i, j, side, entry, px_out, stop))
            break
    return out


def lb_day_groups(m5: pd.DataFrame, cfg) -> list[np.ndarray]:
    """Candle indices of each trading day (17:00-17:00 New York)."""
    day = _days(pd.DatetimeIndex(m5["time"]), cfg)
    bounds = np.r_[0, np.nonzero(np.diff(day))[0] + 1, len(day)]
    return [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def opening_range(m5: pd.DataFrame, sp: float, cfg, start_min: int, close_min: int = 16 * 60) -> list[dict]:
    """Zarattini & Aziz 5-minute opening-range breakout from the candle opening at ``start_min`` (NY)."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins, day = ny_minutes(t), _days(t, cfg)
    out = []
    for i in np.nonzero(mins == start_min)[0]:
        if i + 1 >= len(c) or t[i + 1] - t[i] != Timeframe.M5.delta or c[i] == o[i]:
            continue
        side = 1 if c[i] > o[i] else -1
        entry = o[i + 1] + (sp if side == 1 else 0.0)
        stop = l[i] if side == 1 else h[i] + sp
        risk = side * (entry - stop)
        if risk < 2 * sp:                      # the replication's cost floor
            continue
        same = np.nonzero((day == day[i]) & (mins < close_min) & (np.arange(len(c)) > i))[0]
        if not len(same):
            continue
        j, px = _walk(i + 1, same[-1], side, entry, stop, entry + side * 10 * risk, o, h, l, c, sp)
        out.append(_trade(t, i + 1, j, side, entry, px, stop))
    return out


def silver_bullet(m5: pd.DataFrame, sp: float, cfg, h4: pd.DataFrame, regime_kw: dict) -> list[dict]:
    """First FVG in 10:00-11:00 NY in the H4 trend direction; limit at its edge, 2R, out by 16:00 NY.
    GUESS: the 'liquidity' step and the 'bias' are reduced to the H4 market type (neutral = no trade)."""
    t = pd.DatetimeIndex(m5["time"])
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mins, day = ny_minutes(t), _days(t, cfg)
    reg = regime_per_bar(t, h4, trend_regime_series(h4, **regime_kw))
    out, done = [], set()
    for j in np.nonzero((mins >= 600) & (mins < 660))[0]:
        if day[j] in done or j < 2 or t[j] - t[j - 2] != 2 * Timeframe.M5.delta:
            continue
        side = 1 if reg[j] == "bullish" else -1 if reg[j] == "bearish" else 0
        if side == 1 and l[j] > h[j - 2]:
            lvl, stop = l[j], min(l[j - 2], l[j - 1])
        elif side == -1 and h[j] < l[j - 2]:
            lvl, stop = h[j], max(h[j - 2], h[j - 1]) + sp
        else:
            continue
        done.add(day[j])
        rest = np.nonzero((day == day[j]) & (np.arange(len(c)) > j))[0]
        window = rest[mins[rest] < 660]
        end = rest[mins[rest] < 16 * 60]
        for i in window:          # limit order waits until 11:00
            if (side == 1 and l[i] + sp <= lvl) or (side == -1 and h[i] >= lvl):
                entry = min(lvl, o[i] + sp) if side == 1 else max(lvl, o[i])
                risk = side * (entry - stop)
                if risk <= 0 or not len(end):
                    break
                k, px = _walk(i, end[-1], side, entry, stop, entry + side * 2 * risk, o, h, l, c, sp,
                              fill_bar_target=False)
                out.append(_trade(t, i, k, side, entry, px, stop))
                break
    return out


def session_hold(m5: pd.DataFrame, sp: float, enter_min: int, exit_min: int) -> list[dict]:
    """Buy at the open of the candle at ``enter_min`` (NY) and sell at the open of the next candle
    at ``exit_min`` (within 4 days, so weekends are held)."""
    t = pd.DatetimeIndex(m5["time"])
    o = m5["open"].to_numpy(float)
    mins = ny_minutes(t)
    ins, outs = np.nonzero(mins == enter_min)[0], np.nonzero(mins == exit_min)[0]
    res = []
    for i in ins:
        k = outs.searchsorted(i, side="right")
        if k >= len(outs) or t[outs[k]] - t[i] > pd.Timedelta(days=4):
            continue
        j = outs[k]
        res.append(_trade(t, i, j, 1, o[i] + sp, o[j]))
    return res


def daily_bars(m5: pd.DataFrame, cfg) -> pd.DataFrame:
    """Daily candles from M5 (trading day 17:00-17:00 New York)."""
    d = trading_day(m5["time"], cfg)
    g = m5.groupby(np.asarray(d))
    return pd.DataFrame({"time": g["time"].first(), "open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()}).reset_index(drop=True)


def _rsi(c: pd.Series, n: int) -> np.ndarray:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).to_numpy()


def connors_rsi2(d1: pd.DataFrame, sp: float) -> list[dict]:
    """Signal at the daily close, entry next open, exit at the close that crosses the 5-day SMA."""
    t = pd.DatetimeIndex(d1["time"])
    o, c = d1["open"].to_numpy(float), d1["close"].to_numpy(float)
    rsi = _rsi(d1["close"], 2)
    s200, s5 = d1["close"].rolling(200).mean().to_numpy(), d1["close"].rolling(5).mean().to_numpy()
    out, i = [], 200
    while i < len(c) - 1:
        side = 1 if c[i] > s200[i] and rsi[i] <= 5 else -1 if c[i] < s200[i] and rsi[i] >= 95 else 0
        if not side:
            i += 1
            continue
        entry = o[i + 1] + (sp if side == 1 else 0.0)
        j = i + 1
        while j < len(c) - 1 and not ((side == 1 and c[j] > s5[j]) or (side == -1 and c[j] < s5[j])):
            j += 1
        out.append(_trade(t, i + 1, j, side, entry, c[j] + (sp if side == -1 else 0.0)))
        i = j + 1
    return out


def turtle(d1: pd.DataFrame, sp: float) -> list[dict]:
    """20-day breakout stop orders, 2N stop, 10-day opposite breakout exit; one unit, no skip rule."""
    t = pd.DatetimeIndex(d1["time"])
    o, h, l, c = (d1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hi20, lo20 = (pd.Series(h).shift(1).rolling(20).max().to_numpy(), pd.Series(l).shift(1).rolling(20).min().to_numpy())
    hi10, lo10 = (pd.Series(h).shift(1).rolling(10).max().to_numpy(), pd.Series(l).shift(1).rolling(10).min().to_numpy())
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    n_atr = pd.Series(tr).ewm(alpha=1 / 20, adjust=False).mean().shift(1).to_numpy()
    out, i = [], 21
    while i < len(c):
        up, dn = h[i] + sp >= hi20[i], l[i] <= lo20[i]
        if up == dn:
            i += 1
            continue
        side = 1 if up else -1
        entry = max(hi20[i], o[i]) + sp if side == 1 else min(lo20[i], o[i])
        stop = entry - side * 2 * n_atr[i]
        j, px = i, None
        while j < len(c):
            if side == 1:
                ex = max(stop, lo10[j]) if j > i else stop
                if l[j] <= ex:
                    px = min(o[j], ex) if j > i else ex
                    break
            else:
                ex = min(stop, hi10[j]) if j > i else stop
                if h[j] + sp >= ex:
                    px = max(o[j] + sp, ex) if j > i else ex
                    break
            j += 1
        if px is None:
            j, px = len(c) - 1, c[-1] + (sp if side == -1 else 0.0)
        out.append(_trade(t, i, j, side, entry, px, entry - side * 2 * n_atr[i]))
        i = j + 1
    return out


def ma_cross(d1: pd.DataFrame, sp: float) -> list[dict]:
    """Long while SMA50 > SMA200, short while below; switch at the next open after a cross."""
    t = pd.DatetimeIndex(d1["time"])
    o, c = d1["open"].to_numpy(float), d1["close"].to_numpy(float)
    pos = np.sign(d1["close"].rolling(50).mean() - d1["close"].rolling(200).mean()).fillna(0).to_numpy()
    out, i = [], 200
    while i < len(c) - 1:
        side = int(pos[i])
        if not side:
            i += 1
            continue
        j = i + 1
        while j < len(c) - 1 and pos[j] == side:
            j += 1
        entry = o[i + 1] + (sp if side == 1 else 0.0)
        exit_ = (o[j + 1] if j + 1 < len(c) else c[j]) + (sp if side == -1 else 0.0)
        out.append(_trade(t, i + 1, min(j + 1, len(c) - 1), side, entry, exit_))
        i = j
    return out


def summarize(name: str, tr: pd.DataFrame, split: pd.Timestamp, risk_pct: float) -> dict:
    """Stats for one strategy on the older and the unseen part."""
    row: dict = {"strategy": name}
    has_r = len(tr) and tr["r"].notna().all()
    for part, sel in (("older", tr["entry_time"] < split), ("unseen", tr["entry_time"] >= split)) if len(tr) else ():
        x = tr[sel]
        st = t_stats(x["ret_pct"].to_numpy())
        if has_r:
            rr = x["r"].to_numpy()
            gain = risk_pct * rr
            row[f"{part}_avg_r"] = round(float(rr.mean()), 3) if len(rr) else np.nan
        else:
            gain = x["ret_pct"].to_numpy()
        eq = np.cumsum(gain) if len(gain) else np.zeros(1)
        wins, losses = gain[gain > 0].sum(), -gain[gain < 0].sum()
        row |= {f"{part}_trades": st["trades"], f"{part}_win_pct": st["win_rate_pct"],
                f"{part}_return_pct": round(float(gain.sum()), 1),
                f"{part}_profit_factor": round(float(wins / losses), 2) if losses > 0 else np.nan,
                f"{part}_max_dd_pct": round(float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()), 1),
                f"{part}_p": round(st["p"], 4)}
    row["return_basis"] = f"{risk_pct:g}% risk per trade" if has_r else "1x price (no stop)"
    return row


def run_classics(cfg: AppConfig, store: ParquetStore, dataset: str, point: float, out_dir: str | Path,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, source_note: str = "",
                 spread_points: float = 8.0, discover_frac: float = 0.6) -> dict:
    """Backtest every strategy in :data:`SOURCES` and write ``report.md`` plus one trades CSV each."""
    sc = cfg.strategy
    m5 = store.read_bars(dataset, Timeframe.M5, start, end)
    h4 = store.read_bars(dataset, Timeframe.H4, start, end)
    if m5.empty or h4.empty:
        raise ValueError(f"No stored {dataset} M5/H4 data")
    sp = spread_points * point
    ss = sc.sessions
    d1 = daily_bars(m5, ss)
    split = pd.DatetimeIndex(m5["time"])[int(len(m5) * discover_frac)]
    runs = {
        "London breakout (Asian range)": lambda: london_breakout(m5, sp, ss),
        "NY opening-range breakout (09:30)": lambda: opening_range(m5, sp, ss, 9 * 60 + 30),
        "Gold opening-range breakout (08:20 COMEX)": lambda: opening_range(m5, sp, ss, 8 * 60 + 20, 13 * 60 + 30),
        "ICT Silver Bullet (10-11 NY)": lambda: silver_bullet(m5, sp, ss, h4, regime_kwargs(sc)),
        "Gold overnight (buy 16:00, sell 09:30 NY)": lambda: session_hold(m5, sp, 16 * 60, 9 * 60 + 30),
        "Gold US day session (09:30-16:00 NY)": lambda: session_hold(m5, sp, 9 * 60 + 30, 16 * 60),
        "Connors RSI(2) (daily)": lambda: connors_rsi2(d1, sp),
        "Turtle System 1 (daily, 20/10)": lambda: turtle(d1, sp),
        "50/200-day MA cross (daily)": lambda: ma_cross(d1, sp),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in runs.items():
        tr = pd.DataFrame(fn())
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        tr.to_csv(out / f"trades_{slug}.csv", index=False)
        rows.append(summarize(name, tr, split, sc.backtest.risk_pct))
        log.info("classic_done", extra={"strategy": name, "trades": len(tr)})
    res = pd.DataFrame(rows)
    n = len(res)
    def verdict(r) -> str:
        o_, u_ = r.get("older_return_pct", np.nan), r.get("unseen_return_pct", np.nan)
        if not (o_ > 0 and u_ > 0):
            return "loses" if (o_ <= 0 and u_ <= 0) else "mixed"
        return "profitable (proven)" if max(r["older_p"], r["unseen_p"]) < 0.05 / n else "profitable (not proven)"
    res["verdict"] = res.apply(verdict, axis=1)
    order = {"profitable (proven)": 0, "profitable (not proven)": 1, "mixed": 2, "loses": 3}
    res = res.sort_values(["verdict", "unseen_return_pct"], key=lambda s: s.map(order) if s.name == "verdict" else -s)
    res.to_csv(out / "summary.csv", index=False)

    c = d1["close"].to_numpy(float)
    dt = pd.DatetimeIndex(d1["time"])
    k = int(dt.searchsorted(split))
    bh = {"older": 100 * (c[k - 1] / d1["open"].iat[0] - 1), "unseen": 100 * (c[-1] / c[k - 1] - 1)}
    cols = ["verdict", "strategy", "return_basis", "older_trades", "older_win_pct", "older_avg_r", "older_return_pct",
            "older_profit_factor", "older_max_dd_pct", "unseen_trades", "unseen_win_pct", "unseen_avg_r",
            "unseen_return_pct", "unseen_profit_factor", "unseen_max_dd_pct", "unseen_p"]
    md = ["# Public gold strategies, backtested\n",
          f"- **data**: {source_note or dataset}",
          f"- **period (UTC)**: {m5['time'].iat[0]:%Y-%m-%d} to {m5['time'].iat[-1]:%Y-%m-%d}",
          f"- **spread**: ${sp:.2f} on every trade",
          f"- **older / unseen split**: unseen from {split:%Y-%m-%d}",
          f"- **just holding gold**: {bh['older']:+.1f}% on the older part, {bh['unseen']:+.1f}% on the unseen part\n",
          "Returns: strategies with a stop are shown at "
          f"{sc.backtest.risk_pct:g}% account risk per trade (simple sum, no compounding); strategies without "
          "a stop are shown as % of the gold price at 1x (compare with just holding gold). 'proven' needs "
          f"p < {0.05 / n:.4f} on both parts (Bonferroni over {n} strategies).\n",
          _table(res, cols),
          "\n## Where each rule comes from\n"] + [f"- **{k_}**: {v}" for k_, v in SOURCES.items()] + [
          "\n_Coded as published, no tuning. Research on past data, not a promise of future profit. "
          "Signals only._\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"strategies": n, "verdicts": res["verdict"].value_counts().to_dict(),
               "buy_and_hold_pct": {k_: round(v, 1) for k_, v in bh.items()}, "report": str(out / "report.md")}
    log.info("classics_done", extra=summary)
    return summary
