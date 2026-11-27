// News page: windows in which new signals are blocked (USD high-impact news, bank holidays).
import { get } from "../api.js";
import { busy, card, empty, esc, fmtTime, icon, skeleton, table } from "../ui.js";

export default {
  id: "news", title: "News", icon: "news", subtitle: "New signals are blocked inside these windows",
  async render(el) {
    el.innerHTML = skeleton(1);
    async function load(refresh = false) {
      let rows, err = null;
      try { rows = await get(`/api/news${refresh ? "?refresh=true" : ""}`); } catch (e) { err = e.message; rows = []; }
      const tier = (v) => `<span class="badge ${/high|1/i.test(String(v)) ? "bad" : "gold"}">${esc(v)}</span>`;
      el.innerHTML = card({ title: "Upcoming news blocks", iconName: "news",
        actions: `<button id="rf" class="btn">${icon("refresh")}Refresh calendar</button>`,
        body: err ? `<div class="neg">Calendar problem: ${esc(err)}</div>`
          : rows.length ? table(rows, [{ key: "event", label: "event time" }, { key: "title" }, { key: "tier", fmt: tier }, { key: "start", label: "block from" }, { key: "end", label: "block until" }])
            : `<div class="muted">No upcoming news blocks in the cached calendar. Press Refresh (needs internet).</div>` });
      el.querySelector("#rf").onclick = (e) => busy(e.currentTarget, () => load(true));
    }
    await load();
  },
};
