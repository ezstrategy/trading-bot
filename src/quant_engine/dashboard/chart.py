"""Plotly chart of XAUUSD with the Phase 2 market context drawn on it."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ..features.context import MarketFeatures, MarketSnapshot
from ..features.sessions import KILL_ZONES, NY
from ..timeframes import Timeframe

UP, DOWN = "#26a69a", "#ef5350"
POOL_COLORS = {"previous_day": "#ffb300", "asia": "#7e57c2", "london": "#42a5f5",
               "h1_swing": "#8d6e63", "equal": "#ec407a", "m5_swing": "#9e9e9e"}
KZ_FILL = {"london_kz": "rgba(66,165,245,0.10)", "newyork_kz": "rgba(255,179,0,0.10)"}


def _kill_zone_spans(times: pd.DatetimeIndex, labels) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans, start, cur = [], None, None
    for t, lab in zip(times, labels):
        if lab != cur:
            if cur in KILL_ZONES:
                spans.append((start, t, cur))
            start, cur = t, lab
    if cur in KILL_ZONES and len(times):
        spans.append((start, times[-1] + Timeframe.M5.delta, cur))
    return spans


def build_chart(features: MarketFeatures, snap: MarketSnapshot, tf: Timeframe = Timeframe.M5,
                lookback: int = 180, show: set[str] | None = None, time_zone: str = "UTC") -> go.Figure:
    """Candles up to the snapshot time with optional overlays.

    ``show`` may contain: 'kill_zones', 'pools', 'fvgs', 'sweeps', 'structure', 'range'.
    Only data known at the snapshot time is drawn.
    """
    show = show if show is not None else {"kill_zones", "pools", "fvgs", "sweeps", "structure", "range"}
    t_end = snap.decision_time
    b = features.bars[tf]
    b = b[b["time"] + tf.delta <= t_end].tail(lookback)
    tz = time_zone
    x = pd.DatetimeIndex(b["time"]).tz_convert(tz)
    fig = go.Figure(go.Candlestick(x=x, open=b["open"], high=b["high"], low=b["low"], close=b["close"],
                                   increasing_line_color=UP, decreasing_line_color=DOWN, name=f"XAUUSD {tf.value}"))
    if b.empty:
        return fig
    x0, x1 = x[0], x[-1] + tf.delta

    if "kill_zones" in show:
        m5 = features.bars[Timeframe.M5]
        mask = (m5["time"] >= b["time"].iat[0]) & (m5["time"] < t_end)
        t5 = pd.DatetimeIndex(m5.loc[mask, "time"])
        for a, z, lab in _kill_zone_spans(t5, features.session[mask.to_numpy()]):
            fig.add_vrect(x0=a.tz_convert(tz), x1=z.tz_convert(tz), fillcolor=KZ_FILL[lab], line_width=0,
                          layer="below", annotation_text="London KZ" if lab == "london_kz" else "NY KZ",
                          annotation_position="top left", annotation_font_size=10)

    if "pools" in show:
        lo, hi = b["low"].min(), b["high"].max()
        pad = (hi - lo) * 0.5
        for p in snap.pools.itertuples():
            if not (lo - pad <= p.price <= hi + pad):
                continue
            start = max(pd.Timestamp(p.known_at).tz_convert(tz), x0)
            fig.add_trace(go.Scatter(x=[start, x1], y=[p.price, p.price], mode="lines",
                                     line=dict(color=POOL_COLORS.get(p.kind, "#9e9e9e"), width=1, dash="dot"),
                                     name=f"{p.kind} {p.side}", hovertemplate=f"{p.kind} {p.side} %{{y:.2f}}<extra></extra>",
                                     showlegend=False))

    if "fvgs" in show:
        for f in snap.recent_fvgs[snap.recent_fvgs["valid_size"]].itertuples():
            fig.add_shape(type="rect", x0=pd.Timestamp(f.time).tz_convert(tz) - 2 * Timeframe.M5.delta, x1=x1,
                          y0=f.bottom, y1=f.top, line_width=0, layer="below",
                          fillcolor="rgba(38,166,154,0.18)" if f.direction == 1 else "rgba(239,83,80,0.18)")

    if "sweeps" in show and snap.recent_sweeps:
        sw = snap.recent_sweeps
        fig.add_trace(go.Scatter(
            x=[s.sweep_time.tz_convert(tz) for s in sw], y=[s.extreme for s in sw], mode="markers",
            marker=dict(symbol=["triangle-up" if s.direction == 1 else "triangle-down" for s in sw],
                        size=12, color=[UP if s.direction == 1 else DOWN for s in sw]),
            text=[f"Sweep of {s.pool_kind} {s.side} {s.level:.2f}" for s in sw],
            hovertemplate="%{text}<extra></extra>", name="sweeps", showlegend=False))

    if "structure" in show and tf == Timeframe.M5:
        ev = features.structure[Timeframe.M5].events
        ev = ev[(ev["known_at"] <= t_end) & (ev["time"] >= b["time"].iat[0])]
        for e in ev.itertuples():
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(features.bars[Timeframe.M5]["time"].iat[int(e.swing_idx)]).tz_convert(tz),
                   pd.Timestamp(e.time).tz_convert(tz)], y=[e.level, e.level], mode="lines+text",
                line=dict(color=UP if e.direction == 1 else DOWN, width=1),
                text=["", e.kind], textposition="top left", showlegend=False, hoverinfo="skip"))

    if "range" in show and snap.dealing_range is not None:
        dr = snap.dealing_range
        fig.add_hline(y=dr.equilibrium, line=dict(color="#607d8b", width=1, dash="dash"),
                      annotation_text="H1 equilibrium", annotation_position="bottom right")

    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      hovermode="x unified", showlegend=False,
                      title=dict(text=f"XAUUSD {tf.value} · times in {tz}", font=dict(size=14)))
    return fig
