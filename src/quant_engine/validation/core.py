"""Validation tools (Quant Lab module 7): splits, walk-forward, parameter sensitivity, cost stress.

None of these ever trades the test period: callers pass the development / validation range, and
:func:`quant_engine.experiments.lab.run_experiment` refuses the test part unless ``final_test``.
Walk-forward re-selection is limited to a small grid declared before the run (``max_grid``), as
the rulebook forbids mass optimisation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pandas as pd

from ..analytics import layer_metrics
from ..backtest.engine import ExecutionConfig, run_backtest
from ..backtest.market import MarketData
from ..experiments.registry import Split
from ..instruments import InstrumentSpec


class TestSetLocked(PermissionError):
    """The untouched test period was requested without ``final_test``."""
    __test__ = False          # not a pytest test class


def make_split(programme: str, start: pd.Timestamp, end: pd.Timestamp, dev: float = 0.6, val: float = 0.2) -> Split:
    """Chronological split of ``[start, end)`` by elapsed time, cut at whole UTC days."""
    if not (0 < dev < 1 and 0 < val < 1 and dev + val < 1):
        raise ValueError("development and validation fractions must be in (0, 1) and sum below 1")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = end - start
    return Split(programme, start, (start + span * dev).floor("D"), (start + span * (dev + val)).floor("D"), end)


StrategyFactory = Callable[[dict | None], list]


def _window(data: MarketData, a: pd.Timestamp, b: pd.Timestamp, warmup_days: int) -> MarketData:
    return data.between(a - pd.Timedelta(days=warmup_days), b + pd.Timedelta(days=3))


def evaluate(data: MarketData, factory: StrategyFactory, params: dict | None, spec: InstrumentSpec,
             cfg: ExecutionConfig, a: pd.Timestamp, b: pd.Timestamp, warmup_days: int = 60) -> tuple[pd.DataFrame, dict]:
    """Trades whose signals fall in ``[a, b)`` and their net metrics."""
    res = run_backtest(_window(data, a, b, warmup_days), factory(params), spec, cfg, trade_from=a, trade_until=b)
    return res.trades, layer_metrics(res.trades, "net", cfg.initial_equity, cfg.risk_pct, a, b)


def walk_forward(data: MarketData, factory: StrategyFactory, spec: InstrumentSpec, cfg: ExecutionConfig,
                 start: pd.Timestamp, end: pd.Timestamp, train_months: int = 24, test_months: int = 6,
                 grid: list[dict] | None = None, max_grid: int = 12, min_trades: int = 20,
                 select_by: str = "expectancy_r", warmup_days: int = 60) -> dict:
    """Rolling walk-forward over ``[start, end)``.

    Without ``grid`` the rules stay fixed and each test window is simply evaluated. With ``grid``
    (at most ``max_grid`` parameter sets) the best set on each training window (by ``select_by``,
    needing ``min_trades``) is chosen and then traded on the following test window only.
    """
    if grid is not None and len(grid) > max_grid:
        raise ValueError(f"grid has {len(grid)} parameter sets; the limit is {max_grid} (rulebook rule 7)")
    windows, oos = [], []
    w = pd.Timestamp(start)
    while True:
        tr_a, tr_b = w, w + pd.DateOffset(months=train_months)
        te_a, te_b = tr_b, min(tr_b + pd.DateOffset(months=test_months), pd.Timestamp(end))
        if te_a >= pd.Timestamp(end):
            break
        chosen, train_m = None, {}
        if grid:
            best = None
            for g in grid:
                _, m = evaluate(data, factory, g, spec, cfg, tr_a, tr_b, warmup_days)
                if m.get("trades", 0) >= min_trades and m.get(select_by) is not None and (
                        best is None or m[select_by] > best[1][select_by]):
                    best = (g, m)
            chosen, train_m = (best if best else (grid[0], {}))
        tr, m = evaluate(data, factory, chosen, spec, cfg, te_a, te_b, warmup_days)
        oos.append(tr.assign(window=f"{te_a:%Y-%m-%d}"))
        windows.append({"train": f"{tr_a:%Y-%m-%d} to {tr_b:%Y-%m-%d}", "test": f"{te_a:%Y-%m-%d} to {te_b:%Y-%m-%d}",
                        "params": chosen or "fixed", "train_trades": train_m.get("trades"),
                        "train_expectancy_r": train_m.get("expectancy_r"), "test_trades": m.get("trades", 0),
                        "test_net_profit": m.get("net_profit"), "test_expectancy_r": m.get("expectancy_r"),
                        "test_profit_factor": m.get("profit_factor")})
        w = w + pd.DateOffset(months=test_months)
    wdf = pd.DataFrame(windows)
    all_oos = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    pos = int((wdf["test_net_profit"].fillna(0) > 0).sum()) if len(wdf) else 0
    return {"windows": wdf, "oos_trades": all_oos, "positive_windows": pos, "n_windows": len(wdf),
            "oos": layer_metrics(all_oos, "net", cfg.initial_equity, cfg.risk_pct) if len(all_oos) else {"trades": 0},
            "mode": "re-selected from grid" if grid else "fixed rules"}


def sensitivity(data: MarketData, factory: StrategyFactory, base: dict, neighbours: dict[str, list],
                spec: InstrumentSpec, cfg: ExecutionConfig, a: pd.Timestamp, b: pd.Timestamp,
                warmup_days: int = 60) -> pd.DataFrame:
    """Each parameter moved to nearby values one at a time (the others stay as configured);
    ``base`` holds the configured values, to mark the base row. Nothing is chosen from this."""
    rows = []
    for k, vals in neighbours.items():
        for v in vals:
            _, m = evaluate(data, factory, {k: v}, spec, cfg, a, b, warmup_days)
            rows.append({"param": k, "value": v, "is_base": base.get(k) == v, "trades": m.get("trades", 0),
                         "net_profit": m.get("net_profit"), "expectancy_r": m.get("expectancy_r"),
                         "profit_factor": m.get("profit_factor"), "max_drawdown_pct": m.get("max_drawdown_pct")})
    return pd.DataFrame(rows)


def cost_stress(data: MarketData, factory: StrategyFactory, params: dict | None, spec: InstrumentSpec,
                cfg: ExecutionConfig, a: pd.Timestamp, b: pd.Timestamp, scenarios: list[dict],
                warmup_days: int = 60) -> pd.DataFrame:
    """The same rules under higher costs (spread multiplier, slippage, commission)."""
    rows = []
    for sc in scenarios:
        over = {k: v for k, v in sc.items() if k != "name"}
        _, m = evaluate(data, factory, params, spec, replace(cfg, **over), a, b, warmup_days)
        rows.append({"scenario": sc.get("name", str(over)), "trades": m.get("trades", 0),
                     "net_profit": m.get("net_profit"), "expectancy_r": m.get("expectancy_r"),
                     "profit_factor": m.get("profit_factor"), "max_drawdown_pct": m.get("max_drawdown_pct")})
    return pd.DataFrame(rows)
