// Small formatting helpers shared by every tab.
import { CONFIG } from "../config.js";

export const pct = (p, d = 0) => (p == null ? "–" : `${(p * 100).toFixed(d)}%`);
export const pts = (x, d = 1) => `${Math.abs(x).toFixed(d)} pts`;

// Odds people read more easily than 99.7%: cap at ">99%" / "<1%".
export function chance(p) {
  if (p == null) return "–";
  if (p >= 0.995) return ">99%";
  if (p <= 0.005) return "<1%";
  return `${Math.round(p * 100)}%`;
}

export function rating(p) {
  return CONFIG.ratings.find((r) => p >= r.min) || CONFIG.ratings[CONFIG.ratings.length - 1];
}

export const ratingColor = (p) => `var(--${rating(p).key})`;

export function stamp(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " GMT";
}

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(c));
  return n;
}

export const raceLabel = { senate: "Senate", house: "House", governor: "Governor", president: "President" };
