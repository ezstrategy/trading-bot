// Small UI kit: icons, cards, KPI tiles, tables, segmented controls, charts, toasts, markdown.
import { settings } from "./settings.js";

const P = {
  market: '<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>',
  signals: '<path d="M4 12a8 8 0 0 1 16 0"/><path d="M8 12a4 4 0 0 1 8 0"/><circle cx="12" cy="12" r="1.5"/><path d="M12 14v7"/>',
  backtest: '<path d="M12 8v4l3 2"/><path d="M3.05 11a9 9 0 1 1 .5 4"/><path d="M3 4v7h7"/>',
  news: '<rect x="3" y="4" width="18" height="17" rx="3"/><path d="M3 9h18M8 2v4M16 2v4"/>',
  research: '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3"/><path d="M7.5 15h9"/>',
  data: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
  play: '<path d="M6 4l14 8-14 8z"/>',
  plug: '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0z"/><path d="M12 18v4"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 21h16"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  layers: '<path d="M12 2l10 6-10 6L2 8z"/><path d="M2 14l10 6 10-6"/>',
  drop: '<path d="M12 3s7 7.5 7 12a7 7 0 0 1-14 0c0-4.5 7-12 7-12z"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
};
export function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${P[name] || P.info}</svg>`;
}

export function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

export function card({ title, iconName = "layers", actions = "", body = "", cls = "" }) {
  return `<div class="glass card ${cls}"><div class="card-head"><div class="card-title">${icon(iconName)}${esc(title)}</div><div class="toolbar">${actions}</div></div>${body}</div>`;
}

export function kpi(label, value, tone = "neutral", dot = true) {
  return `<div class="glass kpi tone-${tone}"><div class="label">${esc(label)}</div><div class="value">${dot ? `<span class="dot ${tone}"></span>` : ""}${esc(value)}</div></div>`;
}

export function empty(text, iconName = "info") {
  return `<div class="glass empty">${icon(iconName)}<div>${text}</div></div>`;
}

export const skeleton = (n = 1) => Array.from({ length: n }, () => '<div class="skeleton"></div>').join("");

/** Format a UTC ISO time in the chosen display time zone. */
export function fmtTime(v, withYear = false) {
  if (v == null || v === "") return "";
  const d = new Date(v);
  if (isNaN(d)) return String(v);
  return d.toLocaleString("en-GB", { timeZone: settings.tz, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false, ...(withYear ? { year: "numeric" } : {}) });
}

const TIME_COLS = new Set(["time", "start", "end", "event", "entry_time", "exit_time", "first", "last", "signal_time", "fill_time"]);

/** Build a table from records. ``cols`` = [{key, label, fmt}] or omitted to use every key. */
export function table(rows, cols, { maxHeight } = {}) {
  if (!rows || !rows.length) return '<div class="muted small" style="padding:8px 2px">Nothing to show yet.</div>';
  cols = cols || Object.keys(rows[0]).map((k) => ({ key: k }));
  const isNum = (k) => rows.some((r) => typeof r[k] === "number") && !rows.some((r) => typeof r[k] === "string");
  const numCols = new Set(cols.filter((c) => isNum(c.key)).map((c) => c.key));
  const cell = (c, r) => {
    const v = r[c.key];
    if (c.fmt) return c.fmt(v, r);
    if (TIME_COLS.has(c.key) && typeof v === "string") return esc(fmtTime(v));
    if (typeof v === "number") {
      const s = Number.isInteger(v) ? (Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v)) : v.toLocaleString("en-US", { maximumFractionDigits: 3 });
      return c.signed ? `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${s}</span>` : s;
    }
    if (typeof v === "boolean") return v ? "yes" : "no";
    return esc(v ?? "");
  };
  return `<div class="table-wrap" ${maxHeight ? `style="max-height:${maxHeight}px"` : ""}><table><thead><tr>${cols
    .map((c) => `<th class="${numCols.has(c.key) ? "num" : ""}">${esc(c.label || c.key.replace(/_/g, " "))}</th>`).join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${cols.map((c) => `<td class="${numCols.has(c.key) ? "num" : ""}">${cell(c, r)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/** Segmented control; calls onChange(value). */
export function segmented(el, options, value, onChange) {
  el.classList.add("seg");
  el.innerHTML = Object.entries(options).map(([v, l]) => `<button data-v="${esc(v)}" class="${v === value ? "on" : ""}">${esc(l)}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  };
}

export function toast(msg, kind = "") {
  const t = h(`<div class="toast ${kind}">${icon(kind === "err" ? "info" : "shield")}<div>${esc(msg)}</div></div>`);
  t.querySelector("svg").style.cssText = "width:18px;height:18px;flex:none;color:" + (kind === "err" ? "var(--bad)" : "var(--good)");
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), kind === "err" ? 8000 : 4000);
}

/** Run an async action with a busy button and error toast. */
export async function busy(btn, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="dot gold pulse"></span>Working…`;
  try { return await fn(); }
  catch (e) { toast(e.message, "err"); return null; }
  finally { btn.disabled = false; btn.innerHTML = old; }
}

/** Wrap report HTML that the server rendered from a local markdown file. */
export function markdown(html) {
  return `<div class="md">${html || ""}</div>`;
}

// ---- Plotly glass theme
export const COLORS = { gold: "#e9b949", good: "#2fd39a", bad: "#ff6b7a", info: "#6aa8ff", muted: "#9aa0b8" };
export function glassLayout(extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#c9cde0", size: 12 },
    margin: { l: 8, r: 56, t: 10, b: 30 },
    xaxis: { gridcolor: "rgba(255,255,255,0.05)", zeroline: false, linecolor: "rgba(255,255,255,0.1)", rangeslider: { visible: false } },
    yaxis: { gridcolor: "rgba(255,255,255,0.06)", zeroline: false, side: "right", linecolor: "rgba(255,255,255,0.1)" },
    hoverlabel: { bgcolor: "rgba(20,22,38,0.95)", bordercolor: "rgba(255,255,255,0.2)", font: { color: "#eef0f7" } },
    showlegend: false, dragmode: "pan",
    ...extra,
  };
}
export const isPhone = () => window.matchMedia("(max-width: 900px)").matches;
export const PLOT_CONFIG = { displaylogo: false, responsive: true, scrollZoom: true, displayModeBar: !isPhone(), modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

export function waitFor(name) {
  return new Promise((res) => { const t = () => (window[name] ? res(window[name]) : setTimeout(t, 50)); t(); });
}

/** Line/area chart for an equity curve. */
export async function lineChart(el, x, y, { color = COLORS.gold, yTitle = "", fill = true } = {}) {
  const Plotly = await waitFor("Plotly");
  Plotly.newPlot(el, [{
    x, y, type: "scatter", mode: "lines", line: { color, width: 2, shape: "linear" },
    fill: fill ? "tozeroy" : "none", fillcolor: color.replace(")", "") + "22",
    hovertemplate: "%{y:.2f}<extra></extra>",
  }], glassLayout({ yaxis: { ...glassLayout().yaxis, title: yTitle } }), PLOT_CONFIG);
}
