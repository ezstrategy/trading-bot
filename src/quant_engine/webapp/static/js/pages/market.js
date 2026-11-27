// Market page: bias and structure tiles, annotated price chart, liquidity and structure tables.
import { get } from "../api.js";
import { card, empty, esc, fmtTime, glassLayout, icon, kpi, PLOT_CONFIG, segmented, skeleton, table, waitFor } from "../ui.js";
import { settings, tzLabel } from "../settings.js";

const OVERLAYS = { kill_zones: "Kill zones", pools: "Liquidity", fvgs: "FVGs", sweeps: "Sweeps", structure: "BOS / CHoCH", range: "H1 equilibrium" };
const TFS = { M5: "M5", M15: "M15", H1: "H1", H4: "H4" };

export default {
  id: "market", title: "Market", icon: "market", subtitle: "Live context for XAUUSD across M5, M15, H1 and H4",
  async render(el) {
    let at = null;   // review a past moment (UTC ISO) or null for the latest candle
    el.innerHTML = `
      <div id="hero"></div>
      <div id="cards" class="grid kpis">${skeleton(4)}</div>
      ${card({ title: "Price chart", iconName: "market", body: `
        <div class="toolbar" style="margin-bottom:12px">
          <div class="field"><label>Timeframe</label><div id="tf"></div></div>
          <div class="field" style="min-width:200px;flex:1"><label>Candles: <span id="lb-v">${settings.lookback}</span></label>
            <input id="lb" type="range" min="60" max="600" step="20" value="${settings.lookback}"></div>
          <div class="field"><label>Review a past moment (UTC)</label>
            <div style="display:flex;gap:6px"><input id="at" class="input" type="datetime-local" step="300"><button id="at-clear" class="btn">Latest</button></div></div>
        </div>
        <div class="chips" id="ov" style="margin-bottom:8px"></div>
        <div id="chart" class="chart"></div>` })}
      <div class="grid two">
        <div id="pools"></div>
        <div id="breaks"></div>
      </div>
      <details class="glass"><summary>Raw snapshot (JSON)</summary><div class="body"><pre id="raw" class="json"></pre></div></details>`;

    segmented(el.querySelector("#tf"), TFS, settings.chartTf, (v) => { settings.chartTf = v; load(); });
    const ov = el.querySelector("#ov");
    ov.innerHTML = Object.entries(OVERLAYS).map(([k, l]) => `<button class="chip ${settings.overlays.includes(k) ? "on" : ""}" data-k="${k}">${l}</button>`).join("");
    ov.onclick = (e) => {
      const b = e.target.closest("button"); if (!b) return;
      const s = new Set(settings.overlays); s.has(b.dataset.k) ? s.delete(b.dataset.k) : s.add(b.dataset.k);
      settings.overlays = [...s]; b.classList.toggle("on"); load();
    };
    const lb = el.querySelector("#lb");
    lb.oninput = () => (el.querySelector("#lb-v").textContent = lb.value);
    lb.onchange = () => { settings.lookback = +lb.value; load(); };
    const atIn = el.querySelector("#at");
    atIn.onchange = () => { at = atIn.value ? atIn.value + ":00Z" : null; load(); };
    el.querySelector("#at-clear").onclick = () => { atIn.value = ""; at = null; load(); };

    async function load() {
      const q = new URLSearchParams({ tf: settings.chartTf, lookback: settings.lookback, overlays: settings.overlays.join(","), tz: settings.tz });
      if (at) q.set("at", at);
      if (settings.point) q.set("point", settings.point);
      let d;
      try { d = await get(`/api/market?${q}`); }
      catch (e) { el.querySelector("#cards").innerHTML = empty(`<b>Could not build the market view.</b><br>${esc(e.message)}`); return; }
      if (!d.ready) {
        el.querySelector("#hero").innerHTML = "";
        el.querySelector("#cards").outerHTML = `<div id="cards">${empty(`<b>No market data yet.</b><br>${esc(d.message)} <a href="#/data" style="color:var(--gold-2)">Open Data page</a>`, "data")}</div>`;
        el.querySelector("#chart").closest(".card").style.display = "none";
        return;
      }
      const s = d.summary;
      const bias = d.cards[0];
      const dr = s.h1_dealing_range;
      el.querySelector("#hero").innerHTML = `
        <div class="glass hero tone-${bias.tone}">
          <div><div class="sym">XAUUSD · last closed M5</div><div class="price">${s.last_m5_close.toFixed(2)}</div>
            <div class="muted small">${icon("clock").replace("<svg", '<svg style="width:13px;height:13px;vertical-align:-2px"')} ${esc(fmtTime(d.decision_time, true))} (${esc(tzLabel())})${at ? ' · <span class="badge gold">Review mode</span>' : ""}</div></div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <span class="badge ${bias.tone}">Bias ${esc(bias.value)}</span>
            <span class="badge ${s.new_signals_blocked ? "bad" : "good"}">New signals ${s.new_signals_blocked ? "blocked" : "allowed"}</span>
            <span class="badge">${esc(d.cards[5].value)}</span>
          </div>
        </div>`;
      el.querySelector("#cards").innerHTML = d.cards.map((c) => kpi(c.label, c.value, c.tone)).join("");

      const Plotly = await waitFor("Plotly");
      const fig = d.figure;
      const lay = glassLayout({ ...fig.layout, ...glassLayout(), shapes: fig.layout.shapes, annotations: (fig.layout.annotations || []).map((a) => ({ ...a, font: { ...(a.font || {}), color: "#c9cde0" } })) });
      lay.height = undefined;
      Plotly.react(el.querySelector("#chart"), fig.data, lay, PLOT_CONFIG);

      const pools = (s.pools || []).map((p) => ({ ...p, distance: +(p.price - s.last_m5_close).toFixed(2) })).sort((a, b) => b.price - a.price);
      const zonePct = dr ? Math.max(0, Math.min(100, ((s.last_m5_close - dr.low) / (dr.high - dr.low)) * 100)) : null;
      el.querySelector("#pools").innerHTML = card({ title: "Liquidity levels (untaken)", iconName: "drop", body:
        (dr ? `<div class="small muted">H1 dealing range ${dr.low.toFixed(2)} to ${dr.high.toFixed(2)} · equilibrium ${dr.eq.toFixed(2)} · price in <b style="color:var(--text)">${esc(dr.price_zone)}</b></div>
              <div class="zone-bar"><i style="left:${zonePct}%"></i></div><div class="small muted" style="display:flex;justify-content:space-between;margin-bottom:12px"><span>Discount</span><span>Premium</span></div>` : "") +
        table(pools, [{ key: "kind" }, { key: "side" }, { key: "price" }, { key: "distance", signed: true }], { maxHeight: 340 }) });
      const brk = (s.recent_m5_breaks || []).map((b) => ({ ...b, dir: b.dir === 1 ? "up" : "down" }));
      el.querySelector("#breaks").innerHTML = card({ title: "Sweeps and structure", iconName: "bolt", body: `
        <div class="small muted" style="margin-bottom:6px">Recent sweeps (last hour)</div>${table(s.recent_sweeps, null, { maxHeight: 180 })}
        <div class="small muted" style="margin:14px 0 6px">Recent M5 structure breaks</div>${table(brk, null, { maxHeight: 180 })}
        <div class="chips" style="margin-top:12px">${Object.entries(s.atr).map(([k, v]) => `<span class="badge">ATR ${k} <span class="mono">${v ?? "-"}</span></span>`).join("")}</div>` });
      el.querySelector("#raw").textContent = JSON.stringify(s, null, 2);
    }
    const onRefresh = () => load();
    window.addEventListener("qe:refresh", onRefresh);
    await load();
    return () => window.removeEventListener("qe:refresh", onRefresh);
  },
};
