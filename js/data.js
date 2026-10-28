// Loads the model's output files. Every file is fetched fresh (no stale browser cache after an update).
import { CONFIG } from "./config.js";

const cache = new Map();

async function getText(name) {
  const url = `${CONFIG.dataBase}${name}${name.includes("?") ? "&" : "?"}t=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Couldn't load ${name} (${res.status})`);
  return res.text();
}

export async function getJSON(name, { fresh = false } = {}) {
  if (!fresh && cache.has(name)) return cache.get(name);
  const data = JSON.parse(await getText(name));
  cache.set(name, data);
  return data;
}

export async function getCSV(name) {
  const text = await getText(name);
  return d3.csvParse(text);
}

// Static files that never change between updates (map shapes, hex layout): normal caching is fine.
const staticCache = new Map();
export async function getStatic(path) {
  if (!staticCache.has(path)) staticCache.set(path, fetch(path).then((r) => {
    if (!r.ok) throw new Error(`Couldn't load ${path}`);
    return r.json();
  }));
  return staticCache.get(path);
}

// Calls onChange(meta) whenever data/meta.json reports a newer generated_at.
export function watchForUpdates(minutes, onChange) {
  let last = null;
  const tick = async () => {
    try {
      const meta = await getJSON("meta.json", { fresh: true });
      if (last && meta.generated_at !== last) {
        cache.clear();
        onChange(meta);
      }
      last = meta.generated_at;
    } catch (e) { /* offline or mid-update: try again next tick */ }
  };
  tick();
  return setInterval(tick, minutes * 60 * 1000);
}
