// App shell: page registry, hash router, header chips and live mode.
// Add a page: create js/pages/<name>.js exporting {id, title, icon, subtitle?, render(el)} and list it in PAGES.
import { get, post } from "./api.js";
import { icon, toast } from "./ui.js";
import { settings } from "./settings.js";
import market from "./pages/market.js";
import signals from "./pages/signals.js";
import backtest from "./pages/backtest.js";
import news from "./pages/news.js";
import research from "./pages/research.js";
import data from "./pages/data.js";
import settingsPage from "./pages/settings.js";

const PAGES = [market, signals, backtest, news, research, data, settingsPage];
export const app = { status: null, live: false, current: null };

const nav = document.getElementById("nav");
nav.innerHTML = PAGES.map((p) => `<a href="#/${p.id}" data-id="${p.id}">${icon(p.icon)}<span>${p.title}</span></a>`).join("");

let cleanup = null;
async function route() {
  const id = (location.hash.replace(/^#\/?/, "").split("?")[0]) || "market";
  const page = PAGES.find((p) => p.id === id) || PAGES[0];
  nav.querySelectorAll("a").forEach((a) => a.classList.toggle("active", a.dataset.id === page.id));
  document.getElementById("page-title").textContent = page.title;
  document.getElementById("page-sub").textContent = page.subtitle || "";
  document.title = `${page.title} · XAUUSD Quant Engine`;
  if (typeof cleanup === "function") cleanup();
  const view = document.getElementById("view");
  view.replaceWith(view.cloneNode(false));          // drop old listeners, replay the fade-in
  const fresh = document.getElementById("view");
  app.current = page;
  try { cleanup = await page.render(fresh); }
  catch (e) { fresh.innerHTML = `<div class="glass empty">${icon("info")}<div><b>This page failed to load.</b><br>${e.message}</div></div>`; }
}
window.addEventListener("hashchange", route);

export function setMt5Pill(info) {
  const pill = document.getElementById("mt5-pill");
  if (info) pill.innerHTML = `<span class="dot good"></span><span>${info.broker_symbol} · UTC${info.server_offset_hours >= 0 ? "+" : ""}${info.server_offset_hours}h</span>`;
}

// ---- live mode: fetch newest candles from MT5 on a timer and redraw the market page
let liveTimer = null;
const liveBtn = document.getElementById("live-btn");
async function liveTick() {
  try {
    await post("/api/mt5/refresh");
    if (app.current?.id === "market") window.dispatchEvent(new Event("qe:refresh"));
  } catch (e) { toast(e.message, "err"); setLive(false); }
}
export function setLive(on) {
  app.live = on;
  clearInterval(liveTimer);
  liveBtn.classList.toggle("on", on);
  liveBtn.innerHTML = `<span class="dot ${on ? "good pulse" : ""}"></span><span>${on ? `Live · ${settings.liveSeconds}s` : "Live off"}</span>`;
  if (on) { liveTick(); liveTimer = setInterval(liveTick, settings.liveSeconds * 1000); }
}
liveBtn.onclick = () => setLive(!app.live);

(async () => {
  try {
    app.status = await get("/api/status");
    if (settings.point == null) settings.point = app.status.point;
    setMt5Pill(app.status.mt5);
  } catch (e) { toast("Cannot reach the local server: " + e.message, "err"); }
  route();
})();
