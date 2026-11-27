"""Quant Lab orchestration: load dataset versions, run, analyse and register experiments.

Typical use (see ``docs/QUANT_LAB.md``)::

    python -m quant_engine lab-split --programme gtc2026
    python -m quant_engine lab-run --strategies gtc_sbr_bull,gtc_sbr_bear --programme gtc2026 --part development
    python -m quant_engine lab-validate --strategies gtc_sbr_bull,gtc_sbr_bear --programme gtc2026
    python -m quant_engine lab-run ... --part test --final-test        (once, at the very end)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..analytics import drawdown, performance, report_markdown
from ..backtest.engine import ExecutionConfig, run_backtest
from ..backtest.market import MarketData
from ..config import ConfigError
from ..data.catalog import DatasetCatalog
from ..data.dukascopy import resample
from ..instruments import InstrumentSpec
from ..research.runner import _table
from ..strategies import create
from ..timeframes import Timeframe
from ..validation import TestSetLocked, cost_stress, make_split, sensitivity, walk_forward
from .registry import ExperimentRegistry, RegistryError, Split, git_commit

PARTS = ("development", "validation", "test", "dev+val", "all")


class WalkForwardSettings(BaseModel):
    train_months: int = Field(24, ge=1)
    test_months: int = Field(6, ge=1)
    max_grid: int = Field(12, ge=1, le=50)
    min_trades: int = Field(20, ge=0)
    select_by: str = "expectancy_r"


class AcceptanceSettings(BaseModel):
    min_trades: int = 200
    min_profit_factor: float = 1.3
    max_drawdown_pct: float = 10.0
    min_walk_forward_positive_share: float = 0.5
    alpha: float = 0.05


class LabSettings(BaseModel):
    """``config/lab.yaml``."""
    symbol: str = "XAUUSD"
    instrument_profile: str = ""
    initial_equity: float = Field(10_000.0, gt=0)
    risk_pct: float = Field(0.5, gt=0, le=5)
    split: dict[str, float] = Field(default_factory=lambda: {"development": 0.6, "validation": 0.2})
    warmup_days: int = Field(60, ge=0)
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    cost_stress: list[dict] = Field(default_factory=lambda: [{"name": "as tested"}])
    acceptance: AcceptanceSettings = Field(default_factory=AcceptanceSettings)


def load_lab_settings(config_dir: str | Path | None = None) -> LabSettings:
    """Read ``lab.yaml`` (defaults when it is missing)."""
    p = Path(config_dir or os.getenv("QE_CONFIG_DIR", "config")) / "lab.yaml"
    if not p.exists():
        return LabSettings()
    try:
        return LabSettings(**(yaml.safe_load(p.read_text(encoding="utf-8")) or {}))
    except Exception as exc:
        raise ConfigError(f"Invalid lab.yaml: {exc}") from exc


# --------------------------------------------------------------------- data
def load_market(catalog: DatasetCatalog, symbol: str, versions: dict[str, str] | None = None,
                allow_issues: bool = False, point: float = 0.01) -> MarketData:
    """M5 (required), M1 and M15 (optional) dataset versions as :class:`MarketData`.

    ``versions`` maps ``M1`` / ``M5`` / ``M15`` to a version id or ``latest``; M5 defaults to
    ``latest``, M1 / M15 are used when present. Missing M15 is built from M5, and H4 (for the
    regime) is always built from M5 on the broker's server-midnight grid; both are recorded as
    ``derived from`` the M5 version.
    """
    versions = {k.upper(): v for k, v in (versions or {}).items()}
    ids: dict[str, str] = {}
    bars: dict[Timeframe, pd.DataFrame] = {}
    for tf in (Timeframe.M5, Timeframe.M1, Timeframe.M15):
        want = versions.get(tf.value, "latest" if tf == Timeframe.M5 else "auto")
        if want in ("none", ""):
            continue
        vid = catalog.latest(symbol, tf, usable_only=not allow_issues) if want in ("latest", "auto") else want
        if vid is None:
            if tf == Timeframe.M5:
                raise ValueError(f"no usable {symbol} M5 dataset version; run data-import or data-snapshot first")
            continue
        bars[tf] = catalog.load(vid, allow_issues=allow_issues)
        ids[tf.value] = vid
    m5 = bars[Timeframe.M5]
    if Timeframe.M15 not in bars:
        bars[Timeframe.M15] = resample(m5, Timeframe.M15)
        ids["M15"] = f"derived from {ids['M5']}"
    bars[Timeframe.H4] = resample(m5, Timeframe.H4)
    ids["H4"] = f"derived from {ids['M5']}"
    return MarketData(bars, point, ids)


def default_split(registry: ExperimentRegistry, settings: LabSettings, programme: str, data: MarketData,
                  start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> Split:
    """The programme's stored split, or a new one over the data (registered permanently)."""
    old = registry.split(programme)
    if old is not None:
        return old
    a = pd.Timestamp(start) if start is not None else data.base.t[0].normalize()
    b = pd.Timestamp(end) if end is not None else data.base.t[-1].normalize() + pd.Timedelta(days=1)
    sp = make_split(programme, a, b, settings.split.get("development", 0.6), settings.split.get("validation", 0.2))
    return registry.register_split(sp, data.dataset_ids)


def _strategies(specs: list[tuple[str, dict]]):
    return lambda override=None: [create(sid, {**p, **(override or {})}) for sid, p in specs]


def _exec(settings: LabSettings, overrides: dict | None = None) -> ExecutionConfig:
    return ExecutionConfig(risk_pct=settings.risk_pct, initial_equity=settings.initial_equity, **(overrides or {}))


# --------------------------------------------------------------------- runs
def run_experiment(registry: ExperimentRegistry, data: MarketData, spec: InstrumentSpec, settings: LabSettings,
                   strategies: list[tuple[str, dict]], programme: str, part: str, final_test: bool = False,
                   exec_overrides: dict | None = None, title: str = "", hypothesis: str = "",
                   note: str = "") -> dict:
    """Backtest one part of a programme and register it. Returns the summary."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    split = default_split(registry, settings, programme, data)
    touches_test = part in ("test", "all")
    if touches_test and not final_test:
        raise TestSetLocked(f"the {part!r} part includes the untouched test period "
                            f"({split.val_end:%Y-%m-%d} onwards); pass final_test only for the final check")
    a, b = split.part(part)
    cfg = _exec(settings, exec_overrides)
    strats = _strategies(strategies)()
    warm = data.between(a - pd.Timedelta(days=settings.warmup_days), b + pd.Timedelta(days=3))
    res = run_backtest(warm, strats, spec, cfg, trade_from=a, trade_until=b)
    perf = performance(res.trades, cfg.initial_equity, cfg.risk_pct, a, b)
    families = sorted({s["family"] or s["id"] for s in res.strategies})
    prior_test = {f: registry.test_uses(f) for f in families} if touches_test else {}
    eq = res.equity.copy()
    if len(eq):
        dd = drawdown(eq["equity"])
        eq["drawdown"], eq["drawdown_pct"] = dd["drawdown"].to_numpy(), dd["drawdown_pct"].to_numpy()
    md = [f"# {title or 'Backtest'}: {', '.join(s for s, _ in strategies)} ({part})\n",
          f"- programme `{programme}`: development {split.data_start:%Y-%m-%d} to {split.dev_end:%Y-%m-%d}, "
          f"validation to {split.val_end:%Y-%m-%d}, test to {split.data_end:%Y-%m-%d}",
          f"- traded period (signals): {a:%Y-%m-%d} to {b:%Y-%m-%d} UTC",
          f"- data versions: {json.dumps(data.dataset_ids)}",
          f"- execution: {json.dumps(res.execution, default=str)}",
          f"- instrument profile `{spec.name}`; not confirmed (GUESS): {', '.join(spec.guesses()) or 'none'}",
          f"- hypothesis: {hypothesis or '(not given)'}"]
    if touches_test:
        md.append(f"- **TEST PERIOD USED.** Earlier test uses for {families}: {prior_test}. "
                  "Any further change to these rules can no longer be judged on this period.")
    if res.data.get("spread_assumed_share"):
        md.append(f"- spread missing on {res.data['spread_assumed_share']:.0%} of candles: assumed {res.execution['spread_floor']}")
    md += ["", report_markdown(perf, "Results"), "\n## Why setups did not trade\n",
           _table(pd.DataFrame([{"strategy": k, "reason": r, "count": v} for k, d in res.skips.items()
                                for r, v in sorted(d.items())]), ["strategy", "reason", "count"]),
           "\n## Strategy rules\n", "```json", json.dumps(res.strategies, indent=1), "```",
           "\n_Research on past data only, not a promise of future results. Signals only._\n"]
    results = {k: v for k, v in perf["layers"].items()}
    eid = registry.create(
        kind="backtest", title=title, hypothesis=hypothesis, programme=programme, data_part=part, date_from=a,
        date_to=b, strategies=[{k: s[k] for k in ("id", "version", "family", "fingerprint", "params")} for s in res.strategies],
        dataset_ids=data.dataset_ids, execution=res.execution | {"instrument": spec.model_dump()},
        results=results | {"skips": res.skips, "prior_test_uses": prior_test}, test_set_used=touches_test,
        commit=git_commit(), note=note,
        artefacts={"report.md": "\n".join(md), "trades.csv": res.trades.drop(columns=["reason"]).assign(
            reason=res.trades["reason"].map(json.dumps)), "equity.csv": eq, "signals.csv": res.signals,
            "monthly.csv": perf["monthly"]})
    return {"id": eid, "part": part, "net": results["net"], "before_costs": results["before_costs"],
            "test_set_used": touches_test, "prior_test_uses": prior_test, "folder": str(registry.root / eid)}


def run_validation(registry: ExperimentRegistry, data: MarketData, spec: InstrumentSpec, settings: LabSettings,
                   strategies: list[tuple[str, dict]], programme: str, grid: list[dict] | None = None,
                   neighbours: dict[str, list] | None = None, title: str = "", hypothesis: str = "") -> dict:
    """Walk-forward (dev + validation), parameter sensitivity (development) and cost stress
    (dev + validation), registered as one ``validation`` experiment. Never touches the test period."""
    split = default_split(registry, settings, programme, data)
    cfg = _exec(settings)
    fac = _strategies(strategies)
    wf_s = settings.walk_forward
    a, b = split.part("dev+val")
    wf = walk_forward(data, fac, spec, cfg, a, b, wf_s.train_months, wf_s.test_months, grid, wf_s.max_grid,
                      wf_s.min_trades, wf_s.select_by, settings.warmup_days)
    d0, d1 = split.part("development")
    base = fac()[0].param_dict() if strategies else {}
    sens = sensitivity(data, fac, base, neighbours or {}, spec, cfg, d0, d1, settings.warmup_days) if neighbours else pd.DataFrame()
    stress = cost_stress(data, fac, None, spec, cfg, a, b, settings.cost_stress, settings.warmup_days)
    # monthly / session / regime / side consistency over dev + validation, rules fixed
    res = run_backtest(data.between(a - pd.Timedelta(days=settings.warmup_days), b + pd.Timedelta(days=3)),
                       fac(), spec, cfg, trade_from=a, trade_until=b)
    perf = performance(res.trades, cfg.initial_equity, cfg.risk_pct, a, b)
    mo = perf["monthly"]
    pos_months = int((mo["net_pnl"] > 0).sum()) if len(mo) else 0
    wfw = wf["windows"]
    md = [f"# Validation: {', '.join(s for s, _ in strategies)}\n",
          f"- programme `{programme}`; validation uses development + validation data only "
          f"({a:%Y-%m-%d} to {b:%Y-%m-%d}); the test period from {split.val_end:%Y-%m-%d} is not touched",
          f"- data versions: {json.dumps(data.dataset_ids)}",
          f"- hypothesis: {hypothesis or '(not given)'}",
          f"\n## Walk-forward ({wf['mode']}, {wf_s.train_months}-month train / {wf_s.test_months}-month test)\n",
          f"Positive test windows: {wf['positive_windows']} of {wf['n_windows']}.\n",
          _table(wfw.assign(params=wfw["params"].astype(str)) if len(wfw) else wfw, list(wfw.columns)),
          "\nCombined out-of-window result (net): " + json.dumps(wf["oos"]),
          "\n## Parameter sensitivity (development data, one parameter at a time; nothing is chosen from this)\n",
          _table(sens, list(sens.columns)) if len(sens) else "_not requested_\n",
          "\n## Transaction cost stress (development + validation)\n", _table(stress, list(stress.columns)),
          f"\n## Consistency (development + validation, fixed rules)\n\nPositive months: {pos_months} of {len(mo)}.\n",
          report_markdown(perf, "Fixed rules, development + validation"),
          "\n_Research on past data only. Signals only._\n"]
    results = {"net": perf["layers"]["net"], "before_costs": perf["layers"]["before_costs"],
               "walk_forward": {"positive_windows": wf["positive_windows"], "n_windows": wf["n_windows"], "oos": wf["oos"],
                                "mode": wf["mode"]},
               "cost_stress": stress.to_dict(orient="records"), "positive_months": pos_months, "months": len(mo)}
    eid = registry.create(
        kind="validation", title=title or "validation", hypothesis=hypothesis, programme=programme, data_part="dev+val",
        date_from=a, date_to=b, strategies=[{k: s[k] for k in ("id", "version", "family", "fingerprint", "params")}
                                            for s in res.strategies],
        dataset_ids=data.dataset_ids, execution=res.execution | {"grid": grid, "neighbours": neighbours},
        results=results, commit=git_commit(),
        artefacts={"report.md": "\n".join(md), "walk_forward.csv": wfw.astype(str) if len(wfw) else wfw,
                   "walk_forward_trades.csv": wf["oos_trades"].drop(columns=["reason"], errors="ignore"),
                   "sensitivity.csv": sens, "cost_stress.csv": stress, "monthly.csv": mo,
                   "trades.csv": res.trades.drop(columns=["reason"])})
    return {"id": eid, **results, "folder": str(registry.root / eid)}


__all__ = ["LabSettings", "load_lab_settings", "load_market", "default_split", "run_experiment", "run_validation",
           "RegistryError", "TestSetLocked"]
