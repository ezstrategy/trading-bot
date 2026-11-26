"""Quant Lab section of the dashboard (12 pages). Rendered inside the "Quant Lab" tab of app.py.

Research and signals only: nothing here can place, change or close a trade.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from . import lab_services as ls


def _pick_experiment(reg, key: str, kinds: tuple[str, ...] = ("backtest", "validation", "audit")) -> str | None:
    ex = reg.list()
    ex = ex[ex["kind"].isin(kinds)] if len(ex) else ex
    if ex.empty:
        st.info("No experiments yet. Use **Backtest configuration** (or `python -m quant_engine lab-run`).")
        return None
    ids = list(ex["id"])[::-1]
    labels = {r["id"]: f"{r['id']} · {r['kind']} · {r['strategies']} · {r['part']} · {r['status']}" for _, r in ex.iterrows()}
    return st.selectbox("Experiment", ids, format_func=lambda i: labels[i], key=key)


def _metrics_row(net: dict) -> None:
    cols = st.columns(6)
    for col, (label, k, fmt) in zip(cols, (("Trades", "trades", "{}"), ("Net profit", "net_profit", "{:,.2f}"),
                                           ("Profit factor", "profit_factor", "{}"), ("Expectancy (R)", "expectancy_r", "{}"),
                                           ("Max drawdown %", "max_drawdown_pct", "{}"), ("Sharpe", "sharpe", "{}"))):
        v = net.get(k)
        col.metric(label, "n/a" if v is None else fmt.format(v))


def render_lab(cfg, tz: str = "UTC") -> None:
    """Draw the Quant Lab section."""
    page = st.radio("Quant Lab page", ls.PAGES, horizontal=True, key="lab_page")
    cat, reg = ls.catalog(cfg), ls.registry(cfg)
    st.caption("Results are NET of spread, slippage and commission unless a column says otherwise. "
               "Research on past data only; no trades are placed.")

    if page == "Overview":
        o = ls.overview(cfg)
        c = st.columns(4)
        c[0].metric("Dataset versions", o["dataset_versions"])
        c[1].metric("Usable versions", o["usable_versions"])
        c[2].metric("Strategies", o["strategies"])
        c[3].metric("Experiments", o["experiments"])
        if o["status_counts"]:
            st.markdown("**Experiments by status:** " + " · ".join(f"{k}: {v}" for k, v in o["status_counts"].items()))
        st.markdown("**Research programmes (fixed splits)**")
        st.dataframe(o["programmes"], hide_index=True, width="stretch")
        st.markdown("**Latest experiments**")
        st.dataframe(o["latest_experiments"], hide_index=True, width="stretch")

    elif page == "Historical data":
        ds = cat.list()
        if ds.empty:
            st.info("No dataset versions yet. Run snapshot_data.bat / `mt5-import`, or import a CSV below.")
        else:
            st.dataframe(ds, hide_index=True, width="stretch")
            vid = st.selectbox("Version", list(ds["version_id"])[::-1], key="lab_ds")
            q = cat.quality_report(vid)
            st.json(cat.manifest(vid), expanded=False)
            st.dataframe(pd.DataFrame(q["issues"]), hide_index=True, width="stretch")
            if st.button("Verify checksums", key="lab_verify"):
                st.write(cat.verify(vid))
        with st.expander("Import a CSV file"):
            up = st.file_uploader("CSV of candles", type=["csv", "txt"], key="lab_upload")
            tf = st.selectbox("Timeframe", ["M1", "M5", "M15"], index=1, key="lab_up_tf")
            tzs = st.text_input("Timezone of the file's times", "mt5_server", key="lab_up_tz",
                                help="UTC, +02:00, Europe/London or mt5_server (broker server clock)")
            if up is not None and st.button("Import", key="lab_import"):
                from ..mt5.server_time import ServerClock
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / up.name
                    p.write_bytes(up.getvalue())
                    try:
                        res = cat.import_csv(p, cfg.engine.instrument, tf, source_tz=tzs,
                                             clock=ServerClock.from_config(cfg.mt5.server_time), source_name=up.name)
                        (st.success if res.quality_ok else st.warning)(
                            f"{res.version_id}: {res.manifest['rows']} candles, issues {res.manifest['quality']['counts']}")
                    except Exception as exc:  # shown, never crashes the page
                        st.error(f"Import failed: {exc}")

    elif page == "Strategies":
        st.dataframe(ls.strategies_table(), hide_index=True, width="stretch")

    elif page == "Experiments":
        eid = _pick_experiment(reg, "lab_exp")
        st.dataframe(reg.list(), hide_index=True, width="stretch")
        if eid:
            s = ls.experiment_summary(reg, eid)
            st.markdown(f"**{eid}** · {s['kind']} · status **{s['status']}** · {s['strategies']} · {s['part']} "
                        f"({s['period']}){' · TEST PERIOD USED' if s['test_set_used'] else ''}")
            _metrics_row(s["net"])
            for n in s["notes"]:
                st.caption(f"{n['time_utc'][:16]} · {n['value']}")
            note = st.text_input("Add a research note", key="lab_note")
            status = st.selectbox("Status change (optional)", ["(no change)", *ls.STATUSES], key="lab_status")
            if st.button("Save note", key="lab_save_note") and (note or status != "(no change)"):
                if note:
                    reg.add_note(eid, note)
                if status != "(no change)":
                    reg.set_status(eid, status)
                st.success("Saved (previous notes and statuses are kept).")

    elif page == "Backtest configuration":
        from ..experiments.lab import load_lab_settings, load_market, run_experiment, run_validation
        from ..instruments import load_instruments
        import os
        cdir = os.getenv("QE_CONFIG_DIR", "config")
        strats = st.multiselect("Strategies", ls.available(), default=[s for s in ls.available() if s.startswith("gtc_sbr")],
                                key="lab_bt_strats")
        programme = st.text_input("Research programme", "gtc_sbr_2020", key="lab_bt_prog",
                                  help="Its development / validation / test split is fixed the first time it is used.")
        part = st.radio("Data part", ["development", "validation", "dev+val", "test"], horizontal=True, key="lab_bt_part")
        final = st.checkbox("I understand: this is the final check on the untouched test period", key="lab_bt_final") \
            if part == "test" else False
        params = st.text_area("Parameters (JSON, applied to each strategy)", "{}", key="lab_bt_params")
        c1, c2, c3 = st.columns(3)
        spread_mult = c1.number_input("Spread multiplier", 1.0, 5.0, 1.0, 0.5, key="lab_bt_sm")
        slip = c2.number_input("Slippage (points per fill)", 0.0, 50.0, 2.0, 1.0, key="lab_bt_slip")
        use_m1 = c3.checkbox("Use M1 for intrabar order", True, key="lab_bt_m1")
        hyp = st.text_input("Hypothesis (why should this work?)", key="lab_bt_hyp")
        go_bt, go_val = st.columns(2)
        run_bt = go_bt.button("Run backtest", key="lab_bt_run", disabled=not strats)
        run_val = go_val.button("Run validation (walk-forward, stress)", key="lab_bt_val", disabled=not strats)
        if run_bt or run_val:
            try:
                settings = load_lab_settings(cdir)
                spec = load_instruments(cdir).get(settings.instrument_profile or None)
                data = load_market(cat, settings.symbol, point=spec.point)
                sp = [(s, json.loads(params or "{}")) for s in strats]
                with st.spinner("Running..."):
                    if run_bt:
                        out = run_experiment(reg, data, spec, settings, sp, programme, part, final,
                                             {"spread_multiplier": spread_mult, "slippage_points": slip, "use_m1": use_m1},
                                             hypothesis=hyp)
                    else:
                        out = run_validation(reg, data, spec, settings, sp, programme, hypothesis=hyp)
                st.success(f"Registered {out['id']}")
                _metrics_row(out["net"])
            except Exception as exc:  # shown, never crashes the page
                st.error(f"Not run: {exc}")

    elif page in ("Equity curve", "Drawdown"):
        eid = _pick_experiment(reg, "lab_eq_" + page, ("backtest", "validation"))
        if eid:
            eq = ls.load_equity(reg, eid)
            what = "equity" if page == "Equity curve" else "drawdown_pct"
            st.plotly_chart(ls.equity_figure({eid: eq}, what, tz), width="stretch")
            st.download_button("Download equity CSV", ls.csv_bytes(eq), f"{eid}_equity.csv", key="lab_eq_dl_" + page)

    elif page == "Trade history":
        eid = _pick_experiment(reg, "lab_tr", ("backtest", "validation"))
        if eid:
            tr = ls.load_trades(reg, eid)
            if tr.empty:
                st.info("No trades.")
            else:
                side = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"], key="lab_tr_side")
                view = tr[tr["side"].isin(side)]
                st.dataframe(view, hide_index=True, width="stretch")
                st.download_button("Download trade history CSV", ls.csv_bytes(view), f"{eid}_trades.csv", key="lab_tr_dl")

    elif page == "Trade replay":
        eid = _pick_experiment(reg, "lab_rp", ("backtest",))
        if eid:
            tr = ls.load_trades(reg, eid)
            if tr.empty:
                st.info("No trades to replay.")
            else:
                k = st.number_input("Trade number", 1, len(tr), 1, key="lab_rp_n")
                row = tr.iloc[int(k) - 1]
                st.write({c: row[c] for c in ("strategy_id", "side", "session", "regime", "entry_time", "entry", "stop",
                                              "target", "exit_time", "exit", "exit_reason", "intrabar", "r", "r_gross")})
                ds = reg.get(eid)["dataset_ids"].get("M5")
                try:
                    st.plotly_chart(ls.replay_figure(cat, ds, row, tz=tz), width="stretch")
                except Exception as exc:  # dataset missing on this machine
                    st.warning(f"Cannot draw candles: {exc}")

    elif page == "Strategy comparison":
        ex = reg.list()
        ids = st.multiselect("Experiments", list(ex["id"]) if len(ex) else [], key="lab_cmp")
        if ids:
            st.dataframe(reg.compare(ids).astype(str), width="stretch")
            st.plotly_chart(ls.equity_figure(ls.compare_curves(reg, ids), "equity", tz), width="stretch")
            st.download_button("Download comparison CSV", reg.compare(ids).to_csv().encode(), "comparison.csv", key="lab_cmp_dl")

    elif page == "Walk-forward":
        wf = ls.walk_forward_experiments(reg)
        if wf.empty:
            st.info("No validation experiments yet (`lab-validate`).")
        else:
            eid = st.selectbox("Validation experiment", list(wf["id"])[::-1], key="lab_wf")
            files = ls.experiment_files(reg, eid)
            for name in ("walk_forward.csv", "cost_stress.csv", "sensitivity.csv"):
                df = ls.read_csv(files[name]) if name in files else pd.DataFrame()
                if len(df):
                    st.markdown(f"**{name.removesuffix('.csv').replace('_', ' ')}**")
                    st.dataframe(df, hide_index=True, width="stretch")

    elif page == "Reports":
        eid = _pick_experiment(reg, "lab_rep")
        if eid:
            files = ls.experiment_files(reg, eid)
            if "report.md" in files:
                st.markdown(files["report.md"].read_text(encoding="utf-8"))
            for name, p in files.items():
                st.download_button(f"Download {name}", p.read_bytes(), f"{eid}_{name}", key=f"lab_rep_{name}")
