// Data page: MT5 connection, history download and data-quality checks.
import { get, post } from "../api.js";
import { setMt5Pill } from "../app.js";
import { busy, card, esc, icon, skeleton, table, toast } from "../ui.js";
import { settings } from "../settings.js";

export default {
  id: "data", title: "Data", icon: "data", subtitle: "MetaTrader 5 connection, history download and quality checks",
  async render(el) {
    el.innerHTML = skeleton(2);
    const status = await get("/api/status");
    el.innerHTML = `
      <div class="grid two">
        ${card({ title: "MT5 connection", iconName: "plug", body: `<div id="conn" class="muted" style="margin-bottom:14px">Not checked yet. MT5 must be open on this PC.</div>
          <button id="check" class="btn primary">${icon("plug")}Check MT5 connection</button>` })}
        ${card({ title: "Download history", iconName: "download", body: `
          <div class="toolbar">
            <div class="field" style="width:110px"><label>Days</label><input id="days" class="input" type="number" min="1" max="730" value="${status.default_days}"></div>
            <div class="field" style="flex:1"><label>Timeframes</label><div class="chips" id="tfs">${status.timeframes.map((t) => `<button class="chip on" data-t="${t}">${t}</button>`).join("")}</div></div>
          </div>
          <button id="dl" class="btn" style="margin-top:14px">${icon("download")}Download</button><div id="dlres" class="small muted" style="margin-top:10px"></div>` })}
      </div>
      <div id="q">${skeleton(1)}</div>`;
    const showConn = (i) => {
      el.querySelector("#conn").innerHTML = `<div style="display:flex;gap:8px;flex-wrap:wrap"><span class="badge good">Connected</span><span class="badge">${esc(i.broker_symbol)}</span>
        <span class="badge">Server UTC${i.server_offset_hours >= 0 ? "+" : ""}${i.server_offset_hours}h (${esc(i.clock_mode)})</span><span class="badge">Point ${i.point}</span></div>
        <div class="small" style="margin-top:8px">Last tick (UTC): <span class="mono">${esc(i.last_tick_utc)}</span></div>`;
    };
    if (status.mt5) showConn(status.mt5);
    el.querySelector("#check").onclick = (e) => busy(e.currentTarget, async () => {
      const i = await post("/api/mt5/check");
      showConn(i); setMt5Pill(i); settings.point = i.point; toast("MT5 connected", "ok");
    });
    el.querySelector("#tfs").onclick = (e) => { const b = e.target.closest("button"); if (b) b.classList.toggle("on"); };
    el.querySelector("#dl").onclick = (e) => busy(e.currentTarget, async () => {
      const tfs = [...el.querySelectorAll("#tfs .chip.on")].map((b) => b.dataset.t);
      if (!tfs.length) return toast("Pick at least one timeframe", "err");
      const res = await post("/api/mt5/download", { days: +el.querySelector("#days").value, timeframes: tfs });
      el.querySelector("#dlres").textContent = res.map((r) => `${r.timeframe}: ${r.rows_stored} candles`).join(" · ");
      toast("Download finished", "ok"); loadQ();
    });
    async function loadQ() {
      const rows = await get("/api/data/quality");
      const q = (v) => `<span class="badge ${v === "OK" ? "good" : v === "no data" ? "" : "bad"}">${esc(v)}</span>`;
      el.querySelector("#q").innerHTML = card({ title: "Data quality", iconName: "shield", body: table(rows, [{ key: "timeframe" }, { key: "candles" }, { key: "first" }, { key: "last" },
        { key: "last_close_age", label: "last close age" }, { key: "quality", fmt: q }, { key: "issues" }]) });
    }
    loadQ();
  },
};
