// Breakout page: long-only D1 breakout strategies (A: previous month high, B: 20-day channel).
// Signals only: the buttons here change the local signal record, nothing is ever sent to MT5.
import { get, post } from "../api.js";
import { busy, card, COLORS, empty, esc, fmtTime, kpi, lineChart, markdown, segmented, skeleton, table, toast } from "../ui.js";

const TONE = { "Waiting": "neutral", "BUY signal": "good", "Awaiting manual approval": "gold", "Active": "good",
  "TP hit": "good", "SL hit": "bad", "Expired": "neutral", "Cancelled": "bad" };
const badge = (s) => `<span class="badge ${TONE[s] === "neutral" ? "" : TONE[s] || ""}">${esc(s)}</span>`;
const num = (v, d = 2) => (v == null || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
const signed = (v, d = 2) => (v == null ? "-" : `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${Number(v).toFixed(d)}</span>`);

function brokerTime(iso, off) {
  if (!iso) return "";
  if (off.ny_plus_7) {   // NY-close broker: server clock = New York wall clock + 7h (DST-correct)
    const p = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
      .formatToParts(new Date(iso)).map((x) => [x.type, x.value]));
    const d = new Date(Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour + 7, +p.minute));
    return d.toISOString().slice(5, 16).replace("T", " ");
  }
  const d = new Date(new Date(iso).getTime() + off.hours * 3600e3);
  return d.toISOString().slice(5, 16).replace("T", " ");
}

function stateCard(s) {
  const hist = (s.history || []).slice(-4).reverse().map((h) => `<div class="small muted">${esc(fmtTime(h.time))} · ${esc(h.why)}</div>`).join("");
  return `<div class="glass card"><div class="card-head"><div class="card-title">${esc(s.name)}</div><span class="badge ${s.state === "waiting_for_breakout" ? "gold" : s.state === "position_active" || s.state === "signal_generated" ? "good" : ""}">${esc(s.state_label)}</span></div>
    <div class="grid kpis" style="margin-bottom:10px">
      ${kpi("Next level to close above", num(s.next_level), "gold", false)}${kpi("Last D1 close", num(s.last_d1_close), "neutral", false)}
      ${kpi("Distance", num(s.distance_to_next_level), s.distance_to_next_level > 0 ? "neutral" : "good", false)}${kpi("D1 ATR(14)", num(s.atr), "neutral", false)}
    </div>
    <div class="small muted">Last final D1: ${esc(s.last_d1_day || "-")} · armed for a fresh breakout: <b>${s.armed ? "yes" : "no"}</b></div>${hist}</div>`;
}

function perfKpis(p, bt) {
  const k = (label, s) => kpi(label, s && s.trades ? `${s.trades} tr · ${s.net_r > 0 ? "+" : ""}${s.net_r}R` : "no trades", s && s.net_r > 0 ? "good" : s && s.net_r < 0 ? "bad" : "neutral");
  const btTxt = bt ? Object.entries(bt.per_strategy).map(([sid, v]) => k(`Backtest 2023+ · ${sid}`, v.main_window)).join("") : kpi("Backtest", "not run yet", "neutral");
  return `<div class="grid kpis">${btTxt}${k("Paper (theoretical)", p.paper)}${k("Manually executed", p.manual)}${kpi("Open signals", p.open, "gold")}</div>
    <div class="small muted" style="margin-top:6px">Backtest, paper and manually executed results are kept apart and never mixed.</div>`;
}

function signalsTable(rows, off) {
  return table(rows, [
    { key: "strategy_name", label: "Strategy" }, { key: "side", label: "Dir", fmt: () => '<span class="badge good">BUY</span>' },
    { key: "signal_time", label: "Signal (IST / broker)", fmt: (v) => `${esc(fmtTime(v, true))}<div class="small muted">broker ${esc(brokerTime(v, off))}</div>` },
    { key: "breakout_level", label: "Breakout", fmt: (v) => num(v) },
    { key: "entry", label: "Entry", fmt: (v, r) => v ? num(v) : `~${num(r.entry_est)}` },
    { key: "stop", label: "SL", fmt: (v, r) => num(v ?? r.stop_est) }, { key: "target", label: "TP", fmt: (v, r) => num(v ?? r.target_est) },
    { key: "rr", label: "R:R", fmt: (v) => `1:${num(v, 1)}` }, { key: "atr", label: "ATR", fmt: (v) => num(v) },
    { key: "spread_entry", label: "Spread", fmt: (v, r) => num(v ?? r.spread_at_signal) },
    { key: "suggested_lots", label: "Lots", fmt: (v) => (v ? num(v) : "-") }, { key: "risk_pct", label: "Risk %", fmt: (v) => (v ? num(v) : "-") },
    { key: "status", label: "Status", fmt: (v) => badge(v) },
    { key: "paper_r", label: "Paper R", fmt: (v) => signed(v) },
    { key: "key", label: "", fmt: (v) => `<button class="btn" data-open="${esc(v)}">Details</button>` },
  ]);
}

function detail(d, off) {
  const s = d.signal;
  const checks = (s.checks || []).map((c) => `<div class="small">${c.ok === true ? "✅" : c.ok === false ? "❌" : "⏳"} <b>${esc(c.check)}</b>: ${esc(c.detail)}</div>`).join("");
  const can = { approve: s.approval === "pending" && s.status !== "Cancelled", reject: !s.executed && s.approval !== "rejected",
    exec: s.approval === "approved" && !s.executed, exit: s.executed && s.exec_exit == null };
  return `<div class="glass card" id="detail">
    <div class="card-head"><div class="card-title">${esc(s.strategy_name)} · ${esc(s.day)}</div>${badge(s.status)}</div>
    <p>${esc(s.explanation || "")}</p>
    <div class="grid kpis">
      ${kpi("Signal time", `${fmtTime(s.signal_time, true)} (broker ${brokerTime(s.signal_time, off)})`, "neutral", false)}
      ${kpi("Suggested size", s.suggested_lots ? `${s.suggested_lots} lots · ${num(s.risk_pct)}%` : (s.risk_reason || s.block_reason || "-"), s.suggested_lots ? "good" : "bad", false)}
      ${kpi("Validation", s.generated ? "all checks passed" : "blocked", s.generated ? "good" : "bad", false)}
      ${kpi("Owner decision", `${s.approval}${s.executed ? " · executed" : ""}`, "neutral", false)}
      ${kpi("Paper outcome", s.paper_status ? `${s.paper_status}${s.paper_r != null ? ` · ${num(s.paper_r)}R` : ""}` : "-", s.paper_r > 0 ? "good" : s.paper_r < 0 ? "bad" : "neutral", false)}
      ${s.executed ? kpi("Actual result", s.exec_exit != null ? `${s.exec_result} · ${num(s.exec_r)}R · $${num(s.exec_pnl)}` : `in trade @ ${num(s.exec_entry)} x ${s.exec_lots}`, s.exec_r > 0 ? "good" : s.exec_r < 0 ? "bad" : "neutral", false) : ""}
    </div>
    <h4>Checks</h4>${checks}
    <h4>Manual workflow (records only, never trades)</h4>
    <div class="toolbar" style="flex-wrap:wrap;gap:8px">
      <input class="input" id="note" placeholder="note (optional)">
      <button class="btn primary" data-act="approve" ${can.approve ? "" : "disabled"}>Approve</button>
      <button class="btn" data-act="reject" ${can.reject ? "" : "disabled"}>Reject</button>
    </div>
    <div class="toolbar" style="flex-wrap:wrap;gap:8px;margin-top:8px">
      <input class="input" id="xentry" type="number" step="0.01" placeholder="actual entry price">
      <input class="input" id="xlots" type="number" step="0.01" placeholder="lot size">
      <button class="btn" data-act="execute" ${can.exec ? "" : "disabled"}>Mark as manually executed</button>
    </div>
    <div class="toolbar" style="flex-wrap:wrap;gap:8px;margin-top:8px">
      <input class="input" id="xexit" type="number" step="0.01" placeholder="actual exit price">
      <select class="input" id="xres"><option>TP hit</option><option>SL hit</option><option>Expired</option><option>Closed manually</option></select>
      <button class="btn" data-act="exit" ${can.exit ? "" : "disabled"}>Record exit + result</button>
    </div>
    <h4>History</h4>${table(d.events.map((e) => ({ time: e.time, kind: e.kind, detail: JSON.stringify(e.detail) })))}
    <details><summary class="small muted">Market data snapshot</summary>${table((s.snapshot && s.snapshot.d1) || [])}</details>
  </div>`;
}

async function backtestBox(box) {
  const d = await get("/api/breakout/backtest");
  if (!d.report) {
    box.innerHTML = card({ title: "Backtest", iconName: "backtest", actions: '<button class="btn primary" id="runbt">Run on stored MT5 data</button>',
      body: '<div class="muted">No backtest report yet. On the PC run <code>python -m quant_engine breakout-backtest</code> or press the button (uses the stored M5 history; takes a while on 10+ years).</div>' });
  } else {
    const r = d.report;
    const ids = Object.keys(r.results.per_strategy);
    box.innerHTML = card({ title: `Backtest · ${r.folder}`, iconName: "backtest",
      actions: `<select class="input" id="rsel">${d.reports.map((x) => `<option ${x === r.folder ? "selected" : ""}>${esc(x)}</option>`).join("")}</select><button class="btn primary" id="runbt">Run again</button>`,
      body: `<div class="small muted">${esc(r.data.dataset || "")} · ${esc(r.data.first_utc || "")} to ${esc(r.data.last_utc || "")} · quality ${esc(r.data.quality || "-")} · swap ${esc(r.swap)}</div>
        <div id="bsel" style="margin:12px 0"></div><div id="bwin" style="margin-bottom:10px"></div><div id="bk" class="grid kpis"></div>
        <div id="beq" class="chart sm"></div><div id="btab"></div>
        <details style="margin-top:12px"><summary>Full report</summary>${markdown(r.html)}</details>` });
    let sid = ids[0], win = "main_window";
    const draw = () => {
      const p = r.results.per_strategy[sid], s = p[win] || {};
      const keys = [["trades", "Trades"], ["win_rate_pct", "Win rate %"], ["profit_factor", "Profit factor"], ["expectancy_r", "Expectancy R"], ["net_r", "Net R"],
        ["max_drawdown_r", "Max DD (R)"], ["max_losing_streak", "Max losing streak"], ["avg_hold_hours", "Avg hold (h)"]];
      box.querySelector("#bk").innerHTML = keys.map(([k, l]) => kpi(l, s[k] ?? "-", k === "net_r" || k === "expectancy_r" ? (s[k] > 0 ? "good" : s[k] < 0 ? "bad" : "neutral") : "neutral")).join("") +
        kpi("Longest recovery", s.longest_recovery ? `${s.longest_recovery.days} d${s.longest_recovery.recovered ? "" : " (open)"}` : "-", "neutral");
      const eq = r.equity[sid];
      if (eq && eq.t.length) lineChart(box.querySelector("#beq"), eq.t, eq.r, { color: COLORS.gold, yTitle: "net R (costs included)" });
      box.querySelector("#btab").innerHTML = `<div class="small muted" style="margin:12px 0 6px">Yearly (BUY only)</div>${table(p.yearly)}
        <div class="small muted" style="margin:12px 0 6px">Walk-forward</div>${table(p.walk_forward || [])}
        <div class="small muted" style="margin:12px 0 6px">Sized accounts (both strategies, shared risk engine)</div>${table(Object.entries(r.results.sized).map(([k, v]) => ({ run: k, ...v, skipped_reasons: JSON.stringify(v.skipped_reasons) })))}
        <details><summary class="small muted">Monthly</summary>${table(p.monthly)}</details>`;
    };
    segmented(box.querySelector("#bsel"), Object.fromEntries(ids.map((i) => [i, i === "prev_month_high" ? "A · Previous month high" : "B · 20-day channel"])), sid, (v) => { sid = v; draw(); });
    segmented(box.querySelector("#bwin"), { main_window: "2023+ (main)", full_history: "Full history", train: "Train", test: "Test" }, win, (v) => { win = v; draw(); });
    draw();
    box.querySelector("#rsel").onchange = async (e) => { const x = await get(`/api/breakout/backtest?folder=${encodeURIComponent(e.target.value)}`); d.report = x.report; backtestBox(box); };
  }
  box.querySelector("#runbt").onclick = (e) => busy(e.currentTarget, async () => { await post("/api/breakout/backtest"); toast("Backtest finished"); backtestBox(box); });
}

export default {
  id: "breakout", title: "Breakout", icon: "bolt", subtitle: "Long-only XAUUSD D1 breakouts · paper mode · manual execution only",
  async render(el) {
    el.innerHTML = skeleton(3);
    const d = await get("/api/breakout");
    const off = d.broker_offset;
    const st = d.state;
    el.innerHTML = `
      <div class="glass card"><div class="card-head"><div class="card-title">Mode: ${esc(d.mode.toUpperCase())}</div>
        <span class="badge ${d.mode === "live" ? "good" : "gold"}">${d.mode === "live" ? `live · approved by ${esc(d.review_approved_by)}` : "Live signals disabled until the owner's review"}</span></div>
        <div class="small muted">BUY only (SELL disabled). Risk ${d.risk.per_trade_pct}% per trade (allowed ${d.risk.allowed_risk_pct.join(" / ")}%), max ${d.risk.max_combined_open_pct}% combined, correlated breakouts share one budget (${esc(d.risk.correlated_policy)}). Times in your display zone; broker time ${esc(off.source)}.</div></div>
      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">${st ? st.strategies.map(stateCard).join("") : empty("<b>No market data yet.</b><br>Download MT5 history on the Data page, or run <code>python -m quant_engine breakout-run</code> on the PC.", "bolt")}</div>
      ${st ? `<div class="small muted">State as of ${esc(fmtTime(st.time, true))} (${esc(st.mode)})</div>` : ""}
      ${card({ title: "Performance (kept separate)", iconName: "layers", body: perfKpis(d.performance, d.backtest) })}
      ${card({ title: "Signals", iconName: "signals", body: d.signals.length ? signalsTable(d.signals, off) : '<div class="muted">No breakout signals stored yet. Paper mode stores every breakout from <code>breakout-run</code>.</div>' })}
      <div id="dbox"></div>
      <div id="btbox">${skeleton(1)}</div>`;
    el.addEventListener("click", async (e) => {
      const open = e.target.closest("[data-open]");
      if (open) {
        const key = open.dataset.open;
        const show = async () => {
          const x = await get(`/api/breakout/signals/${encodeURIComponent(key)}`);
          const box = el.querySelector("#dbox");
          box.innerHTML = detail(x, off);
          box.scrollIntoView({ behavior: "smooth" });
          box.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => busy(b, async () => {
            const note = box.querySelector("#note").value;
            const act = b.dataset.act;
            const body = act === "execute" ? { entry: +box.querySelector("#xentry").value, lots: +box.querySelector("#xlots").value, note }
              : act === "exit" ? { exit: +box.querySelector("#xexit").value, result: box.querySelector("#xres").value, note } : { note };
            await post(`/api/breakout/signals/${encodeURIComponent(key)}/${act}`, body);
            toast("Saved (record only, nothing was sent to MT5)");
            show();
          })));
        };
        show();
      }
    });
    backtestBox(el.querySelector("#btbox")).catch((e) => { el.querySelector("#btbox").innerHTML = empty(esc(e.message)); });
  },
};
