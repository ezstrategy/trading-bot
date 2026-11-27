"""Research page (read only): frozen gold breakout results and research reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ...dashboard import services
from ..context import md_html, records

router = APIRouter(prefix="/api/research", tags=["research"])
MC_KEYS = ["p_lose_30pct", "p_lose_50pct", "p_dd_over_30pct", "p_dd_over_50pct", "p_reach_1000",
           "skipped_pct", "median_max_dd_pct"]
SIZING_COLS = ("version", "window", "start_balance", "signals", "taken", "skipped", "skipped%",
               "end_balance", "return%", "max_dd%", "worst_loss%")


def _gold(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"summary": {"trades": 0}, "equity": {"t": [], "r": []}, "years": []}
    yr = tr.assign(year=tr["entry_time"].dt.year).groupby("year")["r"].agg(
        trades="size", total_R="sum", R_per_trade="mean").round(3).reset_index()
    return {"summary": services.trade_summary(tr),
            "equity": {"t": tr["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                       "r": tr["r"].cumsum().round(3).tolist()},
            "years": records(yr)}


@router.get("")
def research() -> dict:
    """Everything the Research page shows; every number is computed from committed run output."""
    tr = services.gold_trades()
    gold = None
    if not tr.empty:
        gold = {"since_2023": _gold(tr[tr["entry_time"] >= pd.Timestamp("2023-01-01", tz="UTC")]),
                "full": _gold(tr)}
    reps = services.research_reports()
    natural = None
    nat = reps.get("lb_natural")
    if nat and (nat / "results.json").exists():
        res = json.loads((nat / "results.json").read_text())
        sz = pd.DataFrame(res.get("sizing", []))
        natural = {"sizing": records(sz[[c for c in SIZING_COLS if c in sz]]), "ledger": None}
        led = nat / "ledger_natural_500.csv"
        if led.exists():
            lg = pd.read_csv(led, parse_dates=["entry_time"])
            natural["ledger"] = {"t": pd.to_datetime(lg["entry_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                                 "equity": lg["equity"].round(2).tolist()}
    montecarlo = None
    mc = reps.get("montecarlo")
    if mc and (mc / "results.json").exists():
        res = json.loads((mc / "results.json").read_text())
        montecarlo = [{"policy": pol, "risk %": k, **{c: round(v.get(c, float("nan")), 1) for c in MC_KEYS}}
                      for pol, lv in (("skip if 0.01 lot too big", res.get("skip_policy", {})),
                                      ("always at least 0.01", res.get("levels", {}))) for k, v in lv.items()]
        montecarlo = json.loads(pd.DataFrame(montecarlo).to_json(orient="records"))
    log = Path("docs/EXPERIMENT_LOG.md")
    return {"gold": gold, "gold_file": services.GOLD_TRADES, "natural": natural, "montecarlo": montecarlo,
            "reports": [{"key": k, "title": services.RESEARCH[k], "path": str(p)} for k, p in reps.items()],
            "has_log": log.exists()}


@router.get("/report/{key}")
def report(key: str) -> dict:
    """Markdown of the newest report of one experiment."""
    p = services.research_reports().get(key)
    if p is None:
        raise HTTPException(404, "report not found")
    f = p / "report.md"
    return {"key": key, "title": services.RESEARCH[key], "path": str(p),
            "html": md_html(f.read_text(encoding="utf-8") if f.exists() else "_No report.md in this folder._")}


@router.get("/log")
def experiment_log() -> dict:
    """docs/EXPERIMENT_LOG.md rendered to HTML."""
    f = Path("docs/EXPERIMENT_LOG.md")
    return {"html": md_html(f.read_text(encoding="utf-8") if f.exists() else "")}
