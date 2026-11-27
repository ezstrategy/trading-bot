"""Historical backtest of the breakout strategies on M5 data (owner spec section 7).

* D1 candles are built from the M5 data (:mod:`.daily`); every decision uses completed candles.
* Entry at the next available M5 open; stop and target in the same M5 candle = stop first.
* Spread (data spread, never below ``spread_floor``), slippage, commission and swap are applied;
  when no swap value is known the report says so in its headline and adds a stress row.
* Each strategy is replayed on its own (independent R results). A sized simulation then applies
  the shared :class:`~.risk.RiskEngine` to both strategies together for each account size and
  reports every skipped signal.
* Periods, a chronological train/test split and a walk-forward are reported. The rules are fixed:
  nothing is fitted on any test period. The walk-forward can only choose between parameter sets
  the owner lists in ``backtest.walk_forward.candidates`` (empty by default), on train data only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..instruments import InstrumentSpec
from ..logging_setup import get_logger
from .config import BreakoutConfig
from .daily import session_number
from .metrics import by_period, equity_r, summary
from .risk import Exposure, RiskCandidate, RiskEngine
from .rules import BreakoutStrategy, strategies, with_params
from .scan import GENERATED, M5Arrays, ScanResult, prepare, scan
from .trade import TradeCosts

log = get_logger(__name__)
FINAL = ("tp_hit", "sl_hit", "expired")


def costs_for(cfg: BreakoutConfig, spec: InstrumentSpec, swap_override: float | None | str = "config") -> tuple[TradeCosts, str]:
    """Cost model and where the swap value came from."""
    c = cfg.costs
    if swap_override != "config":
        swap, src = swap_override, "stress (GUESS)" if swap_override is not None else "not included"
    elif c.swap_long_points is not None:
        swap, src = c.swap_long_points, "breakout.yaml costs.swap_long_points"
    elif spec.swap_long_points is not None:
        swap, src = spec.swap_long_points, f"instrument profile {spec.name or spec.symbol}"
    else:
        swap, src = None, "NOT INCLUDED (no broker swap value known)"
    pt = spec.point
    return TradeCosts(spread_floor=c.spread_floor, slippage=c.slippage_points * spec.tick_size,
                      commission_per_lot=c.commission_per_lot if c.commission_per_lot else spec.commission_per_lot,
                      money_per_price_unit_lot=spec.money_per_price_unit(1.0), swap_long_points=swap, point=pt,
                      swap_triple_weekday=c.swap_triple_weekday), src


def events_frame(res: ScanResult) -> pd.DataFrame:
    """Every breakout event of one scan as a table (signals carry their trade)."""
    rows = [e.to_row() for e in res.events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["note"] = [e.trade.note if e.trade else "" for e in res.events]
    return df


def trades_of(res: ScanResult) -> pd.DataFrame:
    df = events_frame(res)
    if df.empty or "status" not in df:
        return pd.DataFrame(columns=["status", "r", "entry_time", "exit_time", "signal_time", "sessions_held", "r_gross"])
    return df[(df["kind"] == GENERATED) & df["status"].isin(FINAL)].reset_index(drop=True)


def _slice(tr: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if tr.empty:
        return tr
    t = pd.DatetimeIndex(tr["signal_time"])
    m = np.ones(len(tr), dtype=bool)
    if start:
        m &= t >= pd.Timestamp(start, tz="UTC")
    if end:
        m &= t < pd.Timestamp(end, tz="UTC")
    return tr[m]


@dataclass
class BacktestRun:
    """Everything one run produced (written to disk by :func:`write_report`)."""
    config: dict
    costs: dict
    swap_source: str
    data: dict
    strategies: list[dict]
    scans: dict[str, ScanResult]
    trades: dict[str, pd.DataFrame]
    events: dict[str, pd.DataFrame]
    results: dict = field(default_factory=dict)


def run_backtest(m5: pd.DataFrame, cfg: BreakoutConfig, spec: InstrumentSpec, point: float | None = None,
                 data_info: dict | None = None, extras: bool = True) -> BacktestRun:
    """Run both strategies over ``m5`` and compute every report table."""
    point = point or spec.point
    d1, m = prepare(m5, cfg, point)
    costs, swap_src = costs_for(cfg, spec)
    strats = strategies(cfg)
    scans = {s.id: scan(s, d1, m, cfg, costs) for s in strats}
    trades = {sid: trades_of(r) for sid, r in scans.items()}
    events = {sid: events_frame(r) for sid, r in scans.items()}
    info = {"m5_candles": int(len(m5)), "first_utc": str(m5["time"].iloc[0]) if len(m5) else None,
            "last_utc": str(m5["time"].iloc[-1]) if len(m5) else None, "d1_candles": int(len(d1)),
            "d1_incomplete": int((~d1["complete"]).sum()) if len(d1) else 0,
            "spread_from_data_share": float(np.isfinite(m.spread).mean()) if len(m.spread) else 0.0,
            **(data_info or {})}
    run = BacktestRun(cfg.model_dump(mode="json"), costs.to_dict(), swap_src, info,
                      [s.describe() for s in strats], scans, trades, events)
    bt = cfg.backtest
    res: dict = {"per_strategy": {}}
    for s in strats:
        tr = trades[s.id]
        ev = events[s.id]
        per = {"full_history": summary(tr), "main_window": summary(_slice(tr, bt.main_window_start, None)),
               "train": summary(_slice(tr, None, bt.train_test_split)),
               "test": summary(_slice(tr, bt.train_test_split, None)),
               "periods": {p.name: summary(_slice(tr, p.start, p.end)) for p in bt.periods},
               "yearly": by_period(tr, "Y").to_dict(orient="records"),
               "monthly": by_period(tr, "M").to_dict(orient="records"),
               "events": {k: int(v) for k, v in ev["kind"].value_counts().items()} if len(ev) else {},
               "blocked_reasons": ev[ev["kind"] != GENERATED]["reason"].value_counts().head(10).to_dict()
               if len(ev) else {},
               "open_at_end": int(((ev["kind"] == GENERATED) & ~ev["status"].isin(FINAL)).sum())
               if len(ev) and "status" in ev else 0}
        if extras:
            per["walk_forward"] = walk_forward(s, d1, m, cfg, costs)
            per["neighbourhood"] = neighbourhood(s, d1, m, cfg, costs)
            if costs.swap_long_points is None:
                stress, _ = costs_for(cfg, spec, cfg.costs.swap_stress_points)
                st = trades_of(scan(s, d1, m, cfg, stress))
                per["swap_stress"] = {"swap_points": cfg.costs.swap_stress_points,
                                      "full_history": summary(st),
                                      "main_window": summary(_slice(st, bt.main_window_start, None))}
        res["per_strategy"][s.id] = per
    res["overlap"] = overlap(trades)
    res["sized"] = {}
    for acct in bt.equity_accounts:
        for label, start in (("main_window", bt.main_window_start), ("full_history", None)):
            led, summ = sized(scans, cfg, spec, float(acct), start)
            res["sized"][f"{label}_{acct:g}"] = {"summary": summ, "ledger": led}
    run.results = res
    log.info("breakout_backtest_done", extra={"strategies": list(scans), "trades": {k: len(v) for k, v in trades.items()}})
    return run


def overlap(trades: dict[str, pd.DataFrame]) -> dict:
    """How often both strategies signal on the same breakout day (correlated exposure)."""
    if len(trades) < 2:
        return {}
    a, b = list(trades.values())[:2]
    da, db = set(a["day"]) if len(a) else set(), set(b["day"]) if len(b) else set()
    return {"same_day_signals": len(da & db), "only_first": len(da - db), "only_second": len(db - da)}


def walk_forward(s: BreakoutStrategy, d1: pd.DataFrame, m: M5Arrays, cfg: BreakoutConfig, costs: TradeCosts) -> list[dict]:
    """Rolling train/test windows. With no candidates this reports the fixed rules on each test year."""
    wf = cfg.backtest.walk_forward
    variants = [("spec", s)] + [(json.dumps(c, sort_keys=True), with_params(s, **c)) for c in wf.candidates]
    tr = {name: trades_of(scan(v, d1, m, cfg, costs)) for name, v in variants}
    if d1.empty:
        return []
    y0, y1 = pd.Timestamp(d1["day"].iat[0]).year, pd.Timestamp(d1["day"].iat[-1]).year
    out = []
    for ts in range(y0 + wf.train_years, y1 + 1, wf.test_years):
        a, b, c = f"{ts - wf.train_years}-01-01", f"{ts}-01-01", f"{ts + wf.test_years}-01-01"
        scores = {n: summary(_slice(t, a, b)) for n, t in tr.items()}
        pick = max(scores, key=lambda n: (scores[n].get("expectancy_r", -9), n == "spec")) if len(variants) > 1 else "spec"
        test = summary(_slice(tr[pick], b, c))
        out.append({"train": f"{a[:4]}-{int(b[:4]) - 1}", "test": f"{b[:4]}" + (f"-{int(c[:4]) - 1}" if wf.test_years > 1 else ""),
                    "chosen": pick, "train_expectancy_r": scores[pick].get("expectancy_r"),
                    "test_trades": test.get("trades", 0), "test_expectancy_r": test.get("expectancy_r"),
                    "test_net_r": test.get("net_r"), "test_profit_factor": test.get("profit_factor")})
    return out


def neighbourhood(s: BreakoutStrategy, d1: pd.DataFrame, m: M5Arrays, cfg: BreakoutConfig, costs: TradeCosts) -> list[dict]:
    """One-at-a-time parameter changes (robustness information only; never used to pick rules)."""
    nb = cfg.backtest.neighbourhood
    out = []
    for k, vals in (("stop_atr_mult", nb.stop_atr_mult), ("target_r", nb.target_r), ("max_sessions", nb.max_sessions)):
        for v in vals:
            t = trades_of(scan(with_params(s, **{k: v}), d1, m, cfg, costs))
            f, w = summary(t), summary(_slice(t, cfg.backtest.main_window_start, None))
            out.append({"change": f"{k} = {v}", "trades": f.get("trades", 0), "expectancy_r": f.get("expectancy_r"),
                        "profit_factor": f.get("profit_factor"), "main_trades": w.get("trades", 0),
                        "main_expectancy_r": w.get("expectancy_r"), "main_profit_factor": w.get("profit_factor")})
    return out


def sized(scans: dict[str, ScanResult], cfg: BreakoutConfig, spec: InstrumentSpec, equity0: float,
          start: str | None = None) -> tuple[list[dict], dict]:
    """Both strategies through the shared risk engine on one account (compounding, chronological).

    Returns the ledger (one row per signal: taken or skipped and why) and a summary.
    """
    eng = RiskEngine(cfg.risk, spec)
    mpu = spec.money_per_price_unit(1.0)
    sigs = []
    for sid, r in scans.items():
        for e in r.signals:
            t = e.trade
            if t is None or t.status not in FINAL:
                continue
            if start and e.signal_time < pd.Timestamp(start, tz="UTC"):
                continue
            sigs.append(e)
    sigs.sort(key=lambda e: (e.signal_time, cfg.risk.primary_order.index(e.strategy_id)
                             if e.strategy_id in cfg.risk.primary_order else 99))
    equity, peak, max_dd = equity0, equity0, 0.0
    open_pos: list[tuple[Exposure, object, float]] = []
    ledger: list[dict] = []
    i = 0

    def settle(until: pd.Timestamp) -> None:
        nonlocal equity, peak, max_dd, open_pos
        keep = []
        for exp, e, lots in sorted(open_pos, key=lambda x: x[1].trade.exit_time):
            t = e.trade
            if t.exit_time <= until:
                pnl = lots * mpu * (t.exit - t.entry) - lots * spec.commission_per_lot
                if t.swap_included:
                    pnl += lots * mpu * t.r_swap * t.risk
                equity += pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, 100 * (peak - equity) / peak)
                ledger.append({"time": t.exit_time.isoformat(), "event": "exit", "strategy_id": e.strategy_id,
                               "day": f"{e.day:%Y-%m-%d}", "lots": lots, "pnl": round(pnl, 2),
                               "equity": round(equity, 2), "status": t.status})
            else:
                keep.append((exp, e, lots))
        open_pos = keep

    while i < len(sigs):
        t0 = sigs[i].signal_time
        batch = [e for e in sigs[i:] if e.signal_time == t0]
        i += len(batch)
        settle(t0)
        cands = [RiskCandidate(e.strategy_id, session_number(e.day), e.trade.entry, e.trade.stop,
                               extra_cost_per_unit=scan_slip(e), signal_key=e.key) for e in batch]
        decs = eng.batch(cands, [x[0] for x in open_pos], equity)
        for e, d in zip(batch, decs):
            row = {"time": e.signal_time.isoformat(), "event": "signal", "strategy_id": e.strategy_id,
                   "day": f"{e.day:%Y-%m-%d}", "taken": d.approved, "reason": d.reason, "lots": d.lots,
                   "risk_pct": round(d.risk_pct, 3), "equity": round(equity, 2), "r": round(e.trade.r, 3)}
            ledger.append(row)
            if d.approved:
                open_pos.append((Exposure(e.strategy_id, session_number(e.day), d.risk_pct, e.key), e, d.lots))
    settle(pd.Timestamp.max.tz_localize("UTC"))
    sig_rows = [r for r in ledger if r["event"] == "signal"]
    taken = [r for r in sig_rows if r["taken"]]
    reasons: dict[str, int] = {}
    for r in sig_rows:
        if not r["taken"]:
            k = ("minimum lot over the risk limit" if r["reason"].startswith("skipped: minimum lot")
                 else r["reason"].split(" on ")[0])
            reasons[k] = reasons.get(k, 0) + 1
    skipped = [r for r in sig_rows if not r["taken"]]
    return ledger, {"start_equity": equity0, "end_equity": round(equity, 2),
                    "return_pct": round(100 * (equity / equity0 - 1), 1), "max_drawdown_pct": round(max_dd, 1),
                    "signals": len(sig_rows), "taken": len(taken), "skipped": len(skipped),
                    "skipped_pct": round(100 * len(skipped) / len(sig_rows), 1) if sig_rows else 0.0,
                    "skipped_reasons": reasons,
                    "skipped_avg_r": round(float(np.mean([r["r"] for r in skipped])), 3) if skipped else None,
                    "window_start": start or "full history"}


def scan_slip(e) -> float:
    """Exit slippage in price units, not inside entry/stop (charged in the loss per lot)."""
    t = e.trade
    return max(0.0, t.r_slippage * t.risk / 2) if t is not None and math.isfinite(t.r_slippage) else 0.0


# ------------------------------------------------------------------ report
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:g}"
    return str(v)


def _summary_table(rows: dict[str, dict]) -> str:
    keys = ["trades", "win_rate_pct", "profit_factor", "expectancy_r", "net_r", "max_drawdown_r",
            "max_losing_streak", "avg_hold_hours", "tp_hit", "sl_hit", "expired", "cost_r_per_trade", "t_stat"]
    head = "| window | " + " | ".join(k.replace("_", " ") for k in keys) + " | longest recovery |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [head, sep]
    for name, s in rows.items():
        rec = s.get("longest_recovery") or {}
        rtxt = "-" if not rec else f"{rec.get('days')} d ({rec.get('from')} to {rec.get('to')}{'' if rec.get('recovered', True) else ', not recovered'})"
        lines.append(f"| {name} | " + " | ".join(_fmt(s.get(k)) for k in keys) + f" | {rtxt} |")
    return "\n".join(lines)


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_none_"
    keys = list(rows[0])
    return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] +
                     ["| " + " | ".join(_fmt(r.get(k)) for k in keys) + " |" for r in rows])


def report_markdown(run: BacktestRun, title: str = "Breakout backtest") -> str:
    """Human-readable report of a run."""
    r, d = run.results, run.data
    swap_line = (f"**Swap: {run.swap_source}.**" if run.costs.get("swap_long_points") is None
                 else f"Swap: {run.costs['swap_long_points']} points/lot/night from {run.swap_source}.")
    out = [f"# {title}", "",
           "Signals only, BUY only. Theoretical results of fixed rules on historical data: not a forecast, "
           "not evidence of profitability on its own (owner spec section 10).", "",
           f"- Data: {d.get('m5_candles')} M5 candles, {d.get('first_utc')} to {d.get('last_utc')} (UTC); "
           f"{d.get('d1_candles')} D1 candles built from M5 ({d.get('d1_incomplete')} incomplete).",
           f"- Dataset: {d.get('dataset', 'stored MT5 bars')}; data quality: {d.get('quality', 'not checked')}.",
           f"- Costs: spread = data spread (never below {run.costs['spread_floor']}), slippage {run.costs['slippage']} per fill "
           f"(targets too), commission {run.costs['commission_per_lot']}/lot. {swap_line}",
           f"- Same-candle stop and target: stop counted first. Entry: next M5 open after the D1 close.", ""]
    for s in run.strategies:
        p = r["per_strategy"][s["id"]]
        out += [f"## {s['name']} ({s['id']} {s['version']}, rules {s['rule_hash']})", "",
                f"- Level: {s['level']}. Entry: {s['entry']}. Stop: {s['stop_loss']}. Target: {s['take_profit']}.",
                f"- Exit: {s['exit']}. Fresh breakout: {s['reset']}.",
                f"- Breakout events: {p['events']}; open at data end: {p['open_at_end']}; "
                f"not traded because: {p['blocked_reasons'] or 'none'}.", "",
                _summary_table({"2023+ (main)": p["main_window"], "full history": p["full_history"],
                                "train (before split)": p["train"], "test (from split)": p["test"],
                                **{k: v for k, v in p["periods"].items()}}), ""]
        if "swap_stress" in p:
            ss = p["swap_stress"]
            out += [f"Swap stress row (GUESS {ss['swap_points']} points/lot/night):", "",
                    _summary_table({"2023+ (main)": ss["main_window"], "full history": ss["full_history"]}), ""]
        out += ["Yearly (net R):", "", _table(p["yearly"]), ""]
        if p.get("walk_forward"):
            out += ["Walk-forward (fixed rules unless candidates are configured):", "", _table(p["walk_forward"]), ""]
        if p.get("neighbourhood"):
            out += ["Parameter neighbourhood (information only, never used to choose rules):", "",
                    _table(p["neighbourhood"]), ""]
    if r.get("overlap"):
        o = r["overlap"]
        out += ["## Correlation between the strategies", "",
                f"Both strategies signalled on the same breakout day {o['same_day_signals']} times "
                f"(only A: {o['only_first']}, only B: {o['only_second']}). The risk engine treats such signals "
                "as one exposure.", ""]
    out += ["## Sized simulation (shared risk engine, both strategies, compounding)", "",
            _table([{"run": k, **{kk: vv for kk, vv in v["summary"].items() if kk != "skipped_reasons"},
                     "skipped reasons": "; ".join(f"{a}: {b}" for a, b in v["summary"]["skipped_reasons"].items())}
                    for k, v in r["sized"].items()]), ""]
    return "\n".join(out)


def write_report(run: BacktestRun, folder: str | Path, title: str = "Breakout backtest") -> Path:
    """Write report.md, results.json and CSV trade / event / ledger files into ``folder``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report_markdown(run, title), encoding="utf-8")
    res = {k: v for k, v in run.results.items() if k != "sized"}
    res["sized"] = {k: v["summary"] for k, v in run.results["sized"].items()}
    payload = {"config": run.config, "costs": run.costs, "swap_source": run.swap_source, "data": run.data,
               "strategies": run.strategies, "results": res}
    (out / "results.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    for sid, df in run.events.items():
        df.to_csv(out / f"events_{sid}.csv", index=False)
        eq = equity_r(run.trades[sid])
        eq.to_csv(out / f"equity_{sid}.csv", index=False)
    for k, v in run.results["sized"].items():
        pd.DataFrame(v["ledger"]).to_csv(out / f"ledger_{k}.csv", index=False)
    return out
