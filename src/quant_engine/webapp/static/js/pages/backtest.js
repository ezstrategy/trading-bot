// Backtest page: run a backtest on stored data and browse reports.
import { get, post } from "../api.js";
import { busy, card, COLORS, empty, esc, icon, kpi, lineChart, markdown, skeleton, table, toast } from "../ui.js";

export default {
  id: "backtest", title: "Backtest", icon: "backtest", subtitle: "Reproducible runs on stored data. Fewer than 30 trades says little.",
  async render(el) {
    el.innerHTML = skeleton(2);
    const meta = await get("/api/backtests");
    const opts = (o) => Object.entries(o).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    el.innerHTML = `
      ${card({ title: "New run", iconName: "play", body: `
        <div class="toolbar">
          <div class="field" style="flex:1;min-width:180px"><label>Data</label><select id="src" class="input">${opts(meta.sources)}</select></div>
          <div class="field" style="flex:2;min-width:220px"><label>Strategy</label><select id="mode" class="input">${opts(meta.modes)}</select></div>
          <button id="run" class="btn primary">${icon("play")}Run backtest</button>
        </div>
        <div class="small muted" style="margin-top:10px">Get Dukascopy data with <code>python -m quant_engine dukascopy --days 180</code>.</div>` })}
      <div id="pick"></div><div id="report" style="display:flex;flex-direction:column;gap:18px"></div>`;
    el.querySelector("#src").value = "mt5";
    el.querySelector("#run").onclick = (e) => busy(e.currentTarget, async () => {
      const r = await post("/api/backtests/run", { source: el.querySelector("#src").value, mode: el.querySelector("#mode").value });
      toast(`Saved to reports/${r.report}`, "ok");
      meta.reports.unshift(r.report); drawPicker(r.report);
    });
    function drawPicker(sel) {
      if (!meta.reports.length) { el.querySelector("#pick").innerHTML = empty("<b>No backtest yet.</b><br>Run one above (or double-click start_backtest.bat).", "backtest"); return; }
      el.querySelector("#pick").innerHTML = card({ title: "Report", iconName: "table", body:
        `<select id="rep" class="input" style="width:100%">${meta.reports.map((r) => `<option ${r === sel ? "selected" : ""}>${esc(r)}</option>`).join("")}</select>` });
      el.querySelector("#rep").onchange = (e) => show(e.target.value);
      show(sel || meta.reports[0]);
    }
    async function show(name) {
      const box = el.querySelector("#report");
      box.innerHTML = skeleton(2);
      const r = await get(`/api/backtests/${encodeURIComponent(name)}`);
      if (r.comparison) { box.innerHTML = card({ title: "Every mode in every market", iconName: "layers", body: `<div class="small muted" style="margin-bottom:8px">Each single mode should do best in its own market.</div>${table(r.comparison)}` }); return; }
      const o = r.overall || {};
      const v = (k) => (o[k] == null ? "-" : o[k]);
      const tr = Number(o.total_r);
      box.innerHTML = `
        <div class="grid kpis">${kpi("Trades", v("trades"), "gold")}${kpi("Win rate %", v("win_rate_pct"))}
          ${kpi("Total R", v("total_r"), tr > 0 ? "good" : tr < 0 ? "bad" : "neutral")}${kpi("Profit factor", v("profit_factor"))}${kpi("Max drawdown R", v("max_drawdown_r"), "bad")}</div>
        ${r.equity.length ? card({ title: "Equity curve (R)", iconName: "market", body: '<div id="eq" class="chart sm"></div>' }) : ""}
        ${card({ title: "By market regime", iconName: "layers", body: table(r.by_regime) })}
        <details class="glass"><summary>All signals (${r.trades.length})</summary><div class="body">${table(r.trades)}</div></details>
        <details class="glass"><summary>Full report</summary><div class="body">${markdown(r.report_html)}</div></details>`;
      if (r.equity.length) lineChart(box.querySelector("#eq"), [0, ...r.equity.map((_, i) => i + 1)], [0, ...r.equity], { color: tr >= 0 ? COLORS.good : COLORS.bad });
    }
    drawPicker();
  },
};
