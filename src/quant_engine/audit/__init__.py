"""Independent strategy audit (Quant Lab Phase 9, rulebook rules 12 and 14).

:func:`audit_experiment` takes a registered backtest experiment and checks it **without trusting
it**: the data versions still match their checksums, the run reproduces exactly from what was
recorded, every strategy passes the look-ahead harness on the real data, fills happen at the next
candle's open, costs and equity add up trade by trade, there are no duplicate or overlapping
trades, and the test period's reuse is counted. It then classifies the strategy under rule 14:

* ``REJECTED`` when net expectancy is not positive (or a correctness check fails);
* ``VALIDATED FOR PAPER TRADING`` only when every acceptance target passes **and** the audited run
  is the final test on untouched data **and** a validation experiment (walk-forward, cost stress)
  exists for the same rules;
* ``RESEARCH REQUIRED`` otherwise.

``APPROVED FOR LIVE EVALUATION`` is never given automatically: it needs a paper-trading record.
The audit is registered as its own experiment and adds a status event to the audited one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analytics import performance, report_markdown
from ..backtest.engine import ExecutionConfig, run_backtest
from ..backtest.lookahead import check_no_lookahead
from ..data.catalog import DatasetCatalog
from ..experiments.lab import LabSettings, load_market
from ..experiments.registry import ExperimentRegistry, git_commit
from ..instruments import InstrumentSpec
from ..research.runner import _table
from ..strategies import create

EXEC_KEYS = ("risk_pct", "initial_equity", "spread_floor", "spread_multiplier", "slippage_points",
             "slippage_on_targets", "commission_per_lot", "use_m1", "max_concurrent", "regime_config")


def _check(rows: list, name: str, ok: bool, detail: str = "") -> bool:
    rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    return ok


def audit_experiment(registry: ExperimentRegistry, catalog: DatasetCatalog, spec: InstrumentSpec,
                     settings: LabSettings, eid: str, cuts: int = 8) -> dict:
    """Audit one ``backtest`` experiment; returns a summary and registers an ``audit`` experiment."""
    e = registry.get(eid)
    if e["kind"] != "backtest":
        raise ValueError(f"{eid} is a {e['kind']} experiment; audit a backtest experiment")
    rows: list[dict] = []
    folder = Path(e["folder"])
    trades = pd.read_csv(folder / "trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    ids = e["dataset_ids"]

    # 1. data integrity
    real_ids = {tf: v for tf, v in ids.items() if not str(v).startswith("derived")}
    ver = {tf: catalog.verify(v) for tf, v in real_ids.items()}
    _check(rows, "dataset versions unchanged since import (checksums)", all(v["ok"] for v in ver.values()), json.dumps(ver))
    blocking = {tf: catalog.manifest(v)["quality"]["blocking"] for tf, v in real_ids.items()}
    _check(rows, "no blocking data-quality issues in the data used", not any(blocking.values()), json.dumps(blocking))

    # 2. reproducibility
    data = load_market(catalog, settings.symbol, {tf: real_ids.get(tf, "none") for tf in ("M5", "M1", "M15")},
                       allow_issues=True, point=spec.point)
    ex = e["execution"]
    cfg = ExecutionConfig(**{k: ex[k] for k in EXEC_KEYS if k in ex})
    a, b = pd.Timestamp(e["date_from"]), pd.Timestamp(e["date_to"])
    strategies = [create(s["id"], s["params"]) for s in e["strategies"]]
    for s, rec in zip(strategies, e["strategies"]):
        _check(rows, f"{s.meta.id}: same rule set as recorded (id, version, parameters)",
               s.fingerprint() == rec["fingerprint"] and s.meta.version == rec["version"],
               f"now {s.meta.version}/{s.fingerprint()}, recorded {rec['version']}/{rec['fingerprint']}")
    warm = data.between(a - pd.Timedelta(days=settings.warmup_days), b + pd.Timedelta(days=3))
    rerun = run_backtest(warm, strategies, spec, cfg, trade_from=a, trade_until=b).trades
    same = (len(rerun) == len(trades) and np.allclose(rerun["net_pnl"].to_numpy(), trades["net_pnl"].to_numpy())
            and (pd.DatetimeIndex(rerun["entry_time"]) == pd.DatetimeIndex(trades["entry_time"])).all()) if len(trades) else len(rerun) == 0
    _check(rows, "re-run from the record gives identical trades", bool(same), f"{len(rerun)} vs {len(trades)} trades")

    # 3. look-ahead on the real data
    for rec in e["strategies"]:
        rep = check_no_lookahead(lambda r=rec: create(r["id"], r["params"]), warm, cuts=cuts)
        _check(rows, f"{rec['id']}: no look-ahead / repainting ({len(rep['cuts'])} cut points)", rep["ok"],
               json.dumps(rep["differences"][:2], default=str))

    # 4. execution timing and trade sanity
    if len(trades):
        _check(rows, "every entry is at the open of the candle after the signal",
               bool((trades["entry_time"] == trades["signal_time"]).all()))
        _check(rows, "every exit is after its entry", bool((trades["exit_time"] > trades["entry_time"]).all()))
        buys = trades["side"] == "BUY"
        stop_ok = np.where(buys, trades["stop"] < trades["entry"], trades["stop"] > trades["entry"])
        _check(rows, "stops are on the losing side of every entry", bool(stop_ok.all()))
        recon = trades["gross_pnl"] - trades["spread_cost"] - trades["slippage_cost"] - trades["commission"] - trades["net_pnl"]
        _check(rows, "gross - spread - slippage - commission = net on every trade", bool(np.allclose(recon, 0, atol=1e-6)))
        chain = np.allclose(trades["equity_before"].to_numpy()[1:], trades["equity_after"].to_numpy()[:-1])
        final = np.isclose(trades["equity_after"].iat[-1], cfg.initial_equity + trades["net_pnl"].sum())
        _check(rows, "equity recomputes from the trades", bool(chain and final))
        overlap = False
        for _, g in trades.groupby("strategy_id"):
            g = g.sort_values("entry_time")
            overlap |= bool((g["entry_time"].to_numpy()[1:] < g["exit_time"].to_numpy()[:-1]).any())
        _check(rows, "no overlapping trades of one strategy", not overlap)
        _check(rows, "no duplicate trades", not trades.duplicated(["strategy_id", "entry_time"]).any())
        sizes = trades["risk_money"] <= trades["equity_before"] * cfg.risk_pct / 100 + 1e-6
        _check(rows, f"no trade risks more than {cfg.risk_pct}% of equity", bool(sizes.all()))
        stop_first = int(trades["intrabar"].isin(["stop_first", "m1_stop_first"]).sum())
        rows.append({"check": "candles where stop and target were both reachable (stop assumed first)",
                     "result": "INFO", "detail": f"{stop_first} of {len(trades)} trades"})

    # 5. rule 14 classification
    acc = settings.acceptance
    net = performance(trades, cfg.initial_equity, cfg.risk_pct, a, b)["layers"]["net"] if len(trades) else {"trades": 0}
    families = sorted({s.get("family") or s["id"] for s in e["strategies"]})
    others = registry.list()
    fp = sorted(s["fingerprint"] for s in e["strategies"])
    val = None
    for _, r in others[others["kind"] == "validation"].iloc[::-1].iterrows() if len(others) else []:
        v = registry.get(r["id"])
        if v["programme"] == e["programme"] and sorted(s["fingerprint"] for s in v["strategies"]) == fp:
            val = v
            break
    n_variants = max(1, len(others[(others["programme"] == e["programme"]) & (others["kind"] == "backtest")])) if len(others) else 1
    targets = {
        "net expectancy above 0": (net.get("expectancy_r") or -1) > 0,
        f"at least {acc.min_trades} trades": net.get("trades", 0) >= acc.min_trades,
        f"profit factor above {acc.min_profit_factor}": (net.get("profit_factor") or 0) > acc.min_profit_factor,
        f"max drawdown below {acc.max_drawdown_pct}%": (net.get("max_drawdown_pct") or 100) < acc.max_drawdown_pct,
        f"p-value below {acc.alpha / n_variants:.4f} ({n_variants} backtests in this programme)":
            (net.get("p_value") if net.get("p_value") is not None else 1) < acc.alpha / n_variants,
    }
    if val is not None:
        wf = val["results"]["walk_forward"]
        share = wf["positive_windows"] / wf["n_windows"] if wf["n_windows"] else 0
        targets[f"walk-forward: more than {acc.min_walk_forward_positive_share:.0%} of windows positive ({val['id']})"] = \
            share > acc.min_walk_forward_positive_share
        st = {x["scenario"]: x for x in val["results"]["cost_stress"]}
        x15 = st.get("spread x1.5")
        targets["still positive with spread x1.5"] = bool(x15 and (x15.get("expectancy_r") or -1) > 0)
    else:
        targets["a validation experiment (walk-forward, cost stress) exists for these exact rules"] = False
    targets["audited run is the final test on untouched data"] = e["data_part"] == "test"
    correctness_failed = [r["check"] for r in rows if r["result"] == "FAIL"]
    if correctness_failed or not targets["net expectancy above 0"]:
        decision = "REJECTED"
    elif all(targets.values()):
        decision = "VALIDATED FOR PAPER TRADING"
    else:
        decision = "RESEARCH REQUIRED"
    test_uses = {f: registry.test_uses(f) for f in families}

    chk = pd.DataFrame(rows)
    tgt = pd.DataFrame([{"target": k, "result": "PASS" if v else "FAIL"} for k, v in targets.items()])
    md = [f"# Independent audit of {eid}\n",
          f"- audited: {e['title'] or e['kind']} / programme `{e['programme']}` / part **{e['data_part']}** "
          f"({e['date_from']} to {e['date_to']})",
          f"- strategies: {', '.join(s['id'] + '@' + s['version'] for s in e['strategies'])}",
          f"- data versions: {json.dumps(ids)}",
          f"- test-period uses so far for {families}: {test_uses}",
          f"- **classification (rule 14): {decision}**",
          "  (APPROVED FOR LIVE EVALUATION is never automatic: it needs a paper-trading record.)\n",
          "## Correctness checks\n", _table(chk, ["check", "result", "detail"]),
          "\n## Acceptance targets (rulebook rule 9)\n", _table(tgt, ["target", "result"]),
          "\n## Net results of the audited run\n", "```json", json.dumps(net, indent=1, default=str), "```",
          "\n## Manual checklist for the reviewer\n",
          "- Read the strategy rules in the audited report and confirm they match the written hypothesis.",
          "- Check the GUESS items (instrument profile, news windows, spread floor) against the broker.",
          "- Look at 10 random trades in the dashboard's trade replay and confirm entry, stop and exit by eye.",
          "- Confirm the hypothesis was written before the first run on this programme.",
          "\n_Research on past data only. Signals only._\n"]
    aid = registry.create(kind="audit", title=f"audit of {eid}", programme=e["programme"], data_part=e["data_part"],
                          date_from=e["date_from"], date_to=e["date_to"], strategies=e["strategies"],
                          dataset_ids=ids, execution=ex, parent_id=eid, commit=git_commit(),
                          results={"net": net, "checks": rows, "targets": targets, "classification": decision},
                          artefacts={"report.md": "\n".join(md), "checks.csv": chk, "targets.csv": tgt},
                          status=decision if decision != "RECORDED" else "RECORDED")
    registry.set_status(eid, decision)
    registry.add_note(eid, f"audit {aid}: {decision}")
    return {"id": aid, "audited": eid, "passed": int((chk["result"] == "PASS").sum()),
            "failed": correctness_failed, "classification": decision, "targets": targets,
            "folder": str(registry.root / aid)}
