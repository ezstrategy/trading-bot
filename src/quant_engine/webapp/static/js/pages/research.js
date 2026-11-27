// Research page (read only): frozen gold London breakout, sizing study, Monte Carlo, reports.
import { get } from "../api.js";
import { card, COLORS, empty, esc, kpi, lineChart, markdown, segmented, skeleton, table } from "../ui.js";

export default {
  id: "research", title: "Research", icon: "research", subtitle: "Read only. Every number comes from a committed, reproducible run (docs/EXPERIMENT_LOG.md).",
  async render(el) {
    el.innerHTML = skeleton(3);
    const d = await get("/api/research");
    el.innerHTML = `
      <div id="gold"></div>
      ${card({ title: "Natural stop + 2% risk: skipped signals", iconName: "shield", body: d.natural ? `${table(d.natural.sizing)}${d.natural.ledger ? '<div class="small muted" style="margin:14px 0 4px">$500 account, natural stop, 2% max risk (all history)</div><div id="led" class="chart sm"></div>' : ""}` : '<div class="muted">Not run on this PC yet: double-click lb_natural.bat.</div>' })}
      ${card({ title: "Monte Carlo survival ($500, 1:500)", iconName: "layers", body: d.montecarlo ? table(d.montecarlo) : '<div class="muted">Not run on this PC yet: python -m quant_engine montecarlo</div>' })}
      ${card({ title: "All research reports", iconName: "table", body: d.reports.length ? `<select id="rp" class="input" style="width:100%">${d.reports.map((r) => `<option value="${r.key}">${esc(r.title)}</option>`).join("")}</select><div id="rpath" class="small muted" style="margin:8px 0"></div><div id="rmd"></div>` : '<div class="muted">No research reports in reports/ yet.</div>' })}
      ${d.has_log ? '<details class="glass" id="logd"><summary>Experiment log</summary><div class="body" id="log">Loading…</div></details>' : ""}`;

    const goldBox = el.querySelector("#gold");
    if (!d.gold) goldBox.innerHTML = empty(`<b>Gold London breakout</b><br>Trade list not found (${esc(d.gold_file)}).`, "research");
    else {
      goldBox.innerHTML = card({ title: "Gold London breakout (original rules, frozen)", iconName: "research", actions: '<div id="win"></div>',
        body: '<div id="gk" class="grid kpis" style="margin-bottom:14px"></div><div id="geq" class="chart sm"></div><div class="small muted" style="margin:14px 0 6px">By year</div><div id="gy"></div>' });
      const draw = (w) => {
        const g = d.gold[w];
        const s = g.summary;
        goldBox.querySelector("#gk").innerHTML = Object.entries(s).map(([k, v]) => kpi(k, v ?? "-", k === "total R" || k === "R/trade" ? (v > 0 ? "good" : v < 0 ? "bad" : "neutral") : k === "max DD (R)" ? "bad" : "neutral")).join("");
        lineChart(goldBox.querySelector("#geq"), g.equity.t, g.equity.r, { color: COLORS.gold, yTitle: "R (costs included)" });
        goldBox.querySelector("#gy").innerHTML = table(g.years, [{ key: "year" }, { key: "trades" }, { key: "total_R", signed: true }, { key: "R_per_trade", signed: true }]);
      };
      segmented(goldBox.querySelector("#win"), { since_2023: "Since 2023 (main)", full: "Full history" }, "since_2023", draw);
      draw("since_2023");
    }
    if (d.natural?.ledger) lineChart(el.querySelector("#led"), d.natural.ledger.t, d.natural.ledger.equity, { color: COLORS.info, yTitle: "$" });
    const rp = el.querySelector("#rp");
    if (rp) {
      const show = async () => { const r = await get(`/api/research/report/${rp.value}`); el.querySelector("#rpath").textContent = r.path; el.querySelector("#rmd").innerHTML = markdown(r.html); };
      rp.onchange = show; show();
    }
    const logd = el.querySelector("#logd");
    if (logd) logd.addEventListener("toggle", async () => { if (logd.open) el.querySelector("#log").innerHTML = markdown((await get("/api/research/log")).html); }, { once: true });
  },
};
