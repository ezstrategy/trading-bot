// Per-browser display preferences (time zone, live interval, chart options). Never trading settings.
const KEY = "qe.settings.v1";
const DEFAULTS = {
  tz: "Asia/Kolkata",
  liveSeconds: 60,
  chartTf: "M5",
  lookback: window.matchMedia("(max-width: 900px)").matches ? 90 : 180,   // fewer candles on phones
  overlays: ["kill_zones", "pools", "fvgs", "sweeps", "structure", "range"],
  point: null,
};
export const TIME_ZONES = { "India (IST)": "Asia/Kolkata", "UTC": "UTC", "New York": "America/New_York", "London": "Europe/London" };

let state = { ...DEFAULTS };
try { state = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { /* storage blocked */ }

const listeners = new Set();
export const settings = new Proxy(state, {
  set(t, k, v) {
    t[k] = v;
    try { localStorage.setItem(KEY, JSON.stringify(t)); } catch { /* storage blocked */ }
    listeners.forEach((fn) => fn(k, v));
    return true;
  },
});
export const onSettings = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
export const tzLabel = () => Object.keys(TIME_ZONES).find((k) => TIME_ZONES[k] === settings.tz) || settings.tz;
