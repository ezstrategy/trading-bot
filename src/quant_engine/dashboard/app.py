"""XAUUSD Quant Engine dashboard.

Start it with ``python -m quant_engine dashboard`` (or ``start_dashboard.bat``
on Windows). It runs only on your computer at http://localhost:8501.

SIGNALS ONLY: this page reads market data and shows analysis. It has no
way to place, change or close trades.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from quant_engine.config import ConfigError, load_config
from quant_engine.dashboard import services
from quant_engine.dashboard.chart import build_chart
from quant_engine.data.storage import ParquetStore
from quant_engine.features.context import STRUCTURE_TFS
from quant_engine.logging_setup import setup_logging
from quant_engine.timeframes import Timeframe

TIME_ZONES = {"India (IST)": "Asia/Kolkata", "UTC": "UTC", "New York": "America/New_York"}
OVERLAYS = {"Kill zones": "kill_zones", "Liquidity levels": "pools", "Fair value gaps": "fvgs",
            "Sweeps": "sweeps", "BOS / CHoCH": "structure", "H1 equilibrium": "range"}
TONE = {"good": "🟢", "bad": "🔴", "neutral": "⚪"}

st.set_page_config(page_title="XAUUSD Quant Engine", page_icon="📈", layout="wide")


@st.cache_resource
def get_config(config_dir: str):
    cfg = load_config(config_dir)
    setup_logging(cfg.engine.logging.level, cfg.engine.logging.json_format, cfg.engine.logging.file)
    return cfg


def data_fingerprint(store: ParquetStore, canonical: str) -> tuple:
    """Changes whenever a stored file changes, so cached features refresh."""
    out = []
    for tf in STRUCTURE_TFS:
        p = store.bars_path(canonical, tf)
        out.append((tf.value, p.stat().st_mtime if p.exists() else 0))
    return tuple(out)


@st.cache_resource(max_entries=2)
def get_features(_cfg, _store: ParquetStore, canonical: str, point: float, fingerprint: tuple):
    bars = services.load_bars(_store, canonical)
    return bars, services.build_features(bars, _cfg, point)


def mt5_action(label: str, fn, *args):
    try:
        with st.spinner(label):
            return fn(*args)
    except Exception as exc:  # shown to the user, never crashes the page
        st.session_state["last_error"] = f"{label} failed: {exc}"
        return None


# ------------------------------------------------------------------- setup
config_dir = os.getenv("QE_CONFIG_DIR", "config")
try:
    cfg = get_config(config_dir)
except ConfigError as exc:
    st.error(f"Configuration problem: {exc}")
    st.stop()
store = ParquetStore(cfg.data.storage.root)
canonical = cfg.engine.instrument
ss = st.session_state
ss.setdefault("point", 0.01)

st.title("XAUUSD Quant Engine")
st.caption("Signals only · this dashboard reads data and shows analysis; it cannot place trades.")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Operate")
    if st.button("Check MT5 connection", width="stretch"):
        info = mt5_action("Connecting to MT5", services.check_connection, cfg)
        if info:
            ss["mt5_info"] = info
            ss["point"] = float(info.get("point") or ss["point"])
    if "mt5_info" in ss:
        i = ss["mt5_info"]
        st.success(f"Connected · {i['broker_symbol']} · server UTC{i['server_offset_hours']:+g}h ({i['clock_mode']})")
        st.caption(f"Last tick (UTC): {i['last_tick_utc']}")

    with st.expander("Download history", expanded=not any(fp for _, fp in data_fingerprint(store, canonical))):
        days = st.number_input("Days", min_value=1, max_value=730, value=cfg.data.download.default_days)
        tfs = st.multiselect("Timeframes", [t.value for t in cfg.engine.timeframes.collect],
                             default=[t.value for t in cfg.engine.timeframes.collect])
        if st.button("Download", width="stretch", disabled=not tfs):
            res = mt5_action("Downloading from MT5", services.download, cfg, store,
                             [Timeframe.parse(t) for t in tfs], int(days))
            if res:
                st.success("; ".join(f"{r.dataset.split()[-1]}: {r.rows_stored} candles" for r in res))

    st.header("Live mode")
    live = st.toggle("Auto refresh from MT5", value=False,
                     help="Every interval: download the newest candles (M5, M15, H1, H4) and redraw.")
    interval = st.select_slider("Refresh every", options=[30, 60, 120, 300], value=60,
                                format_func=lambda s: f"{s}s")

    st.header("Display")
    chart_tf = Timeframe.parse(st.radio("Chart timeframe", ["M5", "M15", "H1", "H4"], horizontal=True))
    lookback = st.slider("Candles", 60, 600, 180, step=20)
    tz_label = st.selectbox("Times shown in", list(TIME_ZONES))
    overlays = st.multiselect("Overlays", list(OVERLAYS), default=list(OVERLAYS))
    review = st.checkbox("Review a past moment")
    when = None
    if review:
        d = st.date_input("Date (UTC)")
        tm = st.time_input("Time (UTC)", step=300)
        when = pd.Timestamp.combine(d, tm).tz_localize("UTC")
    ss["point"] = st.number_input("Tick size (point)", value=float(ss["point"]), format="%.5f",
                                  help="Filled in automatically by 'Check MT5 connection'.")

if ss.get("last_error"):
    st.error(ss.pop("last_error"))


# -------------------------------------------------------------------- main
def _local(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Show the 'time' column in the chosen time zone."""
    if len(df) and "time" in df:
        df = df.assign(time=pd.to_datetime(df["time"], utc=True).dt.tz_convert(tz).dt.strftime("%d %b %H:%M"))
    return df


def render() -> None:
    if live:
        mt5_action("Refreshing from MT5", services.download, cfg, store, list(STRUCTURE_TFS), 2)
        if ss.get("last_error"):
            st.error(ss.pop("last_error"))
        st.caption(f"Live: refreshed at {pd.Timestamp.now(tz=TIME_ZONES[tz_label]):%H:%M:%S}")
    fp = data_fingerprint(store, canonical)
    if not all(m for _, m in fp):
        st.info("No market data stored yet. Open MT5, then use **Download history** in the sidebar.")
        return
    try:
        bars, feats = get_features(cfg, store, canonical, ss["point"], fp)
        t = when or services.default_decision_time(bars)
        snap = feats.snapshot(t)
    except ValueError as exc:
        st.warning(str(exc))
        return

    tz = TIME_ZONES[tz_label]
    st.subheader(f"Market context at {snap.decision_time.tz_convert(tz):%a %d %b %Y %H:%M} ({tz_label})")
    cards = services.market_status(snap)
    for row in (cards[:4], cards[4:]):
        for col, (label, value, tone) in zip(st.columns(4), row):
            col.metric(label, f"{TONE[tone]} {value}")

    fig = build_chart(feats, snap, chart_tf, lookback, {OVERLAYS[o] for o in overlays}, tz)
    st.plotly_chart(fig, width="stretch")

    summary = services.snapshot_summary(snap)
    left, right = st.columns(2)
    with left:
        st.markdown("**Liquidity levels (untaken)**")
        pools = pd.DataFrame(summary["pools"])
        if len(pools):
            price = summary["last_m5_close"]
            pools["distance"] = (pools["price"] - price).round(2)
            st.dataframe(pools.sort_values("price", ascending=False), hide_index=True, width="stretch")
        else:
            st.caption("None")
        dr = summary["h1_dealing_range"]
        if dr:
            st.caption(f"H1 range {dr['low']:.2f} to {dr['high']:.2f}, equilibrium {dr['eq']:.2f}. "
                       f"Price is in **{dr['price_zone']}**.")
    with right:
        st.markdown("**Recent sweeps (last hour)**")
        st.dataframe(_local(pd.DataFrame(summary["recent_sweeps"]), tz), hide_index=True, width="stretch")
        st.markdown("**Recent M5 structure breaks**")
        brk = pd.DataFrame(summary["recent_m5_breaks"])
        if len(brk):
            brk["dir"] = brk["dir"].map({1: "up", -1: "down"})
        st.dataframe(_local(brk, tz), hide_index=True, width="stretch")
        st.caption("ATR(14): " + " · ".join(f"{k} {v}" for k, v in summary["atr"].items()))

    tab_q, tab_raw = st.tabs(["Data quality", "Raw snapshot"])
    with tab_q:
        st.dataframe(services.data_status(cfg, store), hide_index=True, width="stretch")
    with tab_raw:
        st.code(json.dumps(summary, indent=2, default=str), language="json")


if live:
    st.fragment(render, run_every=interval)()
else:
    render()
