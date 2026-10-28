// The anchor dragon: an idle loop at the desk, and a head peeking over the map that winks out of sync.
import { CONFIG } from "../config.js";

const BASE = "assets/mascot/";
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const loaded = new Map();

function probe(src) {                       // resolves to the src if the image exists, else null
  if (!loaded.has(src)) loaded.set(src, new Promise((ok) => {
    const im = new Image();
    im.onload = () => ok(src);
    im.onerror = () => ok(null);
    im.src = src;
  }));
  return loaded.get(src);
}

export async function anchor(container) {
  const { idleFrames, idleFps } = CONFIG.mascot;
  const frames = (await Promise.all(Array.from({ length: idleFrames }, (_, i) => probe(`${BASE}idle_${i + 1}.png`)))).filter(Boolean);
  if (!frames.length) return;
  const img = document.createElement("img");
  img.alt = "Harvey the dragon, anchoring the election desk";
  img.src = frames[0];
  if (CONFIG.mascot.anchorFacesRight) img.className = "face-right";
  container.append(img);
  if (frames.length < 2 || reduced()) return;
  let k = 0;
  const timer = setInterval(() => {
    if (!img.isConnected) return clearInterval(timer);
    k = (k + 1) % frames.length;
    img.src = frames[k];
  }, 1000 / idleFps);
}

export async function peek(container) {
  const s = CONFIG.mascot.peekStates;
  const [normal, left, right] = await Promise.all([s.normal, s.winkLeft, s.winkRight].map((f) => probe(BASE + f)));
  if (!normal) return;
  const box = document.createElement("div");
  box.className = "peek";
  box.setAttribute("aria-hidden", "true");
  const img = document.createElement("img");
  img.src = normal;
  img.alt = "";
  box.append(img);
  container.append(box);
  if (reduced() || (!left && !right)) return;
  const winks = [left, right].filter(Boolean);
  const [lo, hi] = CONFIG.mascot.winkEverySeconds;
  const show = (src, ms) => new Promise((ok) => { img.src = src; setTimeout(() => { img.src = normal; ok(); }, ms); });
  const loop = async () => {
    if (!img.isConnected) return;
    const roll = Math.random();
    if (roll < 0.2 && winks.length === 2) {           // the uncoordinated double wink
      await show(winks[0], 200); await new Promise((r) => setTimeout(r, 90)); await show(winks[1], 200);
    } else {
      await show(winks[Math.floor(Math.random() * winks.length)], 230);
    }
    setTimeout(loop, (lo + Math.random() * (hi - lo)) * 1000);
  };
  setTimeout(loop, 1500);
}
