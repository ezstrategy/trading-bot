"""Service layer of the Quant Lab dashboard pages (no Streamlit here, so it is testable)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from ..analytics import drawdown
from ..config import AppConfig
from ..data.catalog import DatasetCatalog
from ..experiments.registry import STATUSES, ExperimentRegistry
from ..strategies import available, create

PAGES = ["Overview", "Historical data", "Strategies", "Experiments", "Backtest configuration", "Equity curve",
         "Drawdown", "Trade history", "Trade replay", "Strategy comparison", "Walk-forward", "Reports"]
UP, DOWN = "#26a69a", "#ef5350"


def read_csv(path: Path) -> pd.DataFrame:
    """CSV artefact, or an empty frame when the file is missing or has no columns."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def catalog(cfg: AppConfig) -> DatasetCatalog:
    """Dataset catalog of this configuration."""
    return DatasetCatalog(cfg.data.catalog_root(), cfg.data.quality)


def registry(cfg: AppConfig) -> ExperimentRegistry:
    """Experiment registry of this configuration."""
    return ExperimentRegistry(Path(cfg.data.storage.root) / "experiments")


def overview(cfg: AppConfig) -> dict:
    """Counts for the overview page."""
    cat, reg = catalog(cfg), registry(cfg)
    ds, ex = cat.list(), reg.list()
    status = ex["status"].value_counts().to_dict() if len(ex) else {}
    return {"dataset_versions": len(ds), "usable_versions": int((ds["blocking_issues"] == 0).sum()) if len(ds) else 0,
            "strategies": len(available()), "experiments": len(ex), "status_counts": status,
            "latest_experiments": ex.tail(5).iloc[::-1] if len(ex) else ex, "programmes": reg.splits()}


def strategies_table() -> pd.DataFrame:
    """One row per registered strategy plug-in."""
    rows = []
    for sid in available():
        d = create(sid).describe()
        rows.append({k: d[k] for k in ("id", "version", "name", "direction", "family", "entry", "exit", "stop_loss",
                                       "take_profit", "sessions", "regime", "risk", "fingerprint")})
    return pd.DataFrame(rows)


def experiment_files(reg: ExperimentRegistry, eid: str) -> dict[str, Path]:
    """Artefact files of an experiment by name."""
    f = Path(reg.get(eid)["folder"])
    return {p.name: p for p in sorted(f.iterdir())} if f.exists() else {}


def load_trades(reg: ExperimentRegistry, eid: str) -> pd.DataFrame:
    """Trade history of an experiment (empty when it has none)."""
    df = read_csv(Path(reg.get(eid)["folder"]) / "trades.csv")
    if df.empty:
        return df
    for c in ("signal_time", "entry_time", "exit_time"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def load_equity(reg: ExperimentRegistry, eid: str) -> pd.DataFrame:
    """Equity curve (time, equity, drawdown, drawdown_pct) of an experiment."""
    eq = read_csv(Path(reg.get(eid)["folder"]) / "equity.csv")
    if len(eq):
        eq["time"] = pd.to_datetime(eq["time"], utc=True)
    else:
        tr = load_trades(reg, eid)
        if tr.empty:
            return pd.DataFrame(columns=["time", "equity", "drawdown", "drawdown_pct"])
        eq = pd.DataFrame({"time": tr["exit_time"], "equity": tr["equity_after"]})
    if "drawdown" not in eq:
        dd = drawdown(eq["equity"])
        eq["drawdown"], eq["drawdown_pct"] = dd["drawdown"].to_numpy(), dd["drawdown_pct"].to_numpy()
    return eq


def equity_figure(curves: dict[str, pd.DataFrame], what: str = "equity", tz: str = "UTC") -> go.Figure:
    """Line chart of equity (or ``drawdown_pct``) for one or more experiments."""
    fig = go.Figure()
    for name, eq in curves.items():
        if len(eq):
            fig.add_trace(go.Scatter(x=eq["time"].dt.tz_convert(tz), y=eq[what], mode="lines", name=name,
                                     fill="tozeroy" if what == "drawdown_pct" and len(curves) == 1 else None))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="equity" if what == "equity" else "drawdown %", hovermode="x unified")
    return fig


def replay_figure(cat: DatasetCatalog, dataset_id: str, trade: pd.Series, before: int = 36, after: int = 24,
                  tz: str = "UTC") -> go.Figure:
    """M5 candles around one trade with entry, stop, target and exit drawn (trade replay)."""
    bars = cat.load(dataset_id, allow_issues=True)
    t0, t1 = pd.Timestamp(trade["entry_time"]), pd.Timestamp(trade["exit_time"])
    i0 = max(0, int(bars["time"].searchsorted(t0)) - before)
    i1 = min(len(bars), int(bars["time"].searchsorted(t1)) + after)
    b = bars.iloc[i0:i1]
    x = b["time"].dt.tz_convert(tz)
    fig = go.Figure(go.Candlestick(x=x, open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                                   increasing_line_color=UP, decreasing_line_color=DOWN, name="M5 (bid)"))
    xs = [t0.tz_convert(tz), t1.tz_convert(tz)]
    for label, y, color, dash in (("entry", trade["entry"], "#1e88e5", "solid"), ("stop", trade["stop"], DOWN, "dash"),
                                  ("target", trade.get("target"), UP, "dash")):
        if y is not None and pd.notna(y):
            fig.add_trace(go.Scatter(x=xs, y=[y, y], mode="lines", name=f"{label} {float(y):.2f}",
                                     line=dict(color=color, dash=dash)))
    fig.add_trace(go.Scatter(x=[t0.tz_convert(tz)], y=[trade["entry"]], mode="markers", name="fill",
                             marker=dict(symbol="triangle-up" if trade["side"] == "BUY" else "triangle-down", size=14,
                                         color="#1e88e5")))
    fig.add_trace(go.Scatter(x=[t1.tz_convert(tz)], y=[trade["exit"]], mode="markers",
                             name=f"exit ({trade['exit_reason']}, {float(trade['r']):+.2f}R net)",
                             marker=dict(symbol="x", size=12, color="#212121")))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def compare_curves(reg: ExperimentRegistry, ids: list[str]) -> dict[str, pd.DataFrame]:
    """Equity curves of several experiments, rebased to 100 at their start."""
    out = {}
    for eid in ids:
        eq = load_equity(reg, eid)
        if len(eq):
            eq = eq.assign(equity=eq["equity"] / eq["equity"].iat[0] * 100)
        out[eid] = eq
    return out


def walk_forward_experiments(reg: ExperimentRegistry) -> pd.DataFrame:
    """Validation experiments (they hold the walk-forward results)."""
    ex = reg.list()
    return ex[ex["kind"] == "validation"] if len(ex) else ex


def csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV download content."""
    return df.to_csv(index=False).encode("utf-8")


def experiment_summary(reg: ExperimentRegistry, eid: str) -> dict:
    """Headline facts of one experiment for the page header."""
    e = reg.get(eid)
    return {"id": eid, "kind": e["kind"], "status": e["status"], "programme": e["programme"], "part": e["data_part"],
            "period": f"{str(e['date_from'])[:10]} to {str(e['date_to'])[:10]}",
            "strategies": ", ".join(f"{s['id']}@{s['version']}" for s in e["strategies"]),
            "test_set_used": e["test_set_used"], "data": json.dumps(e["dataset_ids"]), "net": e["results"].get("net", {}),
            "before_costs": e["results"].get("before_costs", {}), "notes": e["notes"]}
