// Signals page: everything the live engine sent to Telegram (it never trades).
import { get } from "../api.js";
import { card, empty, esc, kpi, skeleton, table } from "../ui.js";

export default {
  id: "signals", title: "Signals", icon: "signals", subtitle: "Signals sent by the live engine (start_signals.bat). It never places trades.",
  async render(el) {
    el.innerHTML = skeleton(2);
    const d = await get("/api/signals");
    const st = d.stats;
    const side = (v) => `<span class="badge ${v === 1 || v === "BUY" || v === "buy" ? "good" : "bad"}">${v === 1 ? "BUY" : v === -1 ? "SELL" : esc(v)}</span>`;
    el.innerHTML = `
      <div class="grid kpis">
        ${kpi("Signals sent", st.sent, "gold")}${kpi("Closed", st.closed, "neutral")}
        ${kpi("Total R (closed)", (st.total_r > 0 ? "+" : "") + st.total_r, st.total_r > 0 ? "good" : st.total_r < 0 ? "bad" : "neutral")}
      </div>
      ${d.signals.length ? card({ title: "Signals", iconName: "signals", body: table(d.signals, [
        { key: "time" }, { key: "side", fmt: side }, { key: "grade" }, { key: "score" }, { key: "entry" }, { key: "stop" },
        { key: "tp1" }, { key: "tp2" }, { key: "lots" }, { key: "status" }, { key: "outcome" }, { key: "r", label: "R", signed: true }]) })
        : empty("<b>No signals sent yet.</b><br>Start the live engine with <code>python -m quant_engine run</code> or double-click start_signals.bat.", "signals")}
      ${card({ title: "Message log", iconName: "table", body: table(d.messages, [{ key: "time" }, { key: "kind" }, { key: "sent" },
        { key: "text", fmt: (v) => `<span style="white-space:pre-wrap;display:block;max-width:640px">${esc(v)}</span>` }]) })}`;
  },
};
