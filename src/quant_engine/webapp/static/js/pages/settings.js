// Settings page: display preferences stored in this browser only (no trading settings here).
import { app, setLive } from "../app.js";
import { card, esc, icon, kpi } from "../ui.js";
import { settings, TIME_ZONES } from "../settings.js";

export default {
  id: "settings", title: "Settings", icon: "settings", subtitle: "Display preferences for this browser. Strategy settings stay in config/*.yaml.",
  render(el) {
    const s = app.status || {};
    el.innerHTML = `
      <div class="grid kpis">${kpi("Instrument", s.instrument || "XAUUSD", "gold")}${kpi("Mode", s.execution_mode || "signals_only", "good")}${kpi("Max risk per signal", (s.risk_pct ?? "-") + "%", "neutral")}</div>
      ${card({ title: "Display", iconName: "settings", body: `
        <div class="toolbar">
          <div class="field" style="flex:1;min-width:180px"><label>Times shown in</label><select id="tz" class="input">${Object.entries(TIME_ZONES).map(([l, v]) => `<option value="${v}" ${v === settings.tz ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></div>
          <div class="field" style="flex:1;min-width:160px"><label>Live refresh every</label><select id="live" class="input">${[30, 60, 120, 300].map((v) => `<option value="${v}" ${v === settings.liveSeconds ? "selected" : ""}>${v}s</option>`).join("")}</select></div>
          <div class="field" style="flex:1;min-width:160px"><label>Tick size (point)</label><input id="pt" class="input" type="number" step="0.00001" value="${settings.point ?? 0.01}"></div>
        </div>
        <div class="small muted" style="margin-top:10px">Tick size is filled in automatically by "Check MT5 connection" on the Data page.</div>` })}
      ${card({ title: "Safety", iconName: "shield", body: `<div class="muted">This app reads data and shows analysis. It has no way to place, change or close trades. The API listens on this PC only (127.0.0.1). API reference: <a href="/api/docs" style="color:var(--gold-2)">/api/docs</a></div>` })}`;
    el.querySelector("#tz").onchange = (e) => (settings.tz = e.target.value);
    el.querySelector("#live").onchange = (e) => { settings.liveSeconds = +e.target.value; if (app.live) setLive(true); };
    el.querySelector("#pt").onchange = (e) => (settings.point = +e.target.value || 0.01);
  },
};
