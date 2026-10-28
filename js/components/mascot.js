// The anchor dragon: an idle loop at the desk, and a head peeking over the map that winks out of sync.
import { CONFIG } from "../config.js";

const BASE = "assets/mascot/";
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const loaded = new Map();

function probe(src) {                       // resolves to the loaded <img> if the file exists, else null
  if (!loaded.has(src)) loaded.set(src, new Promise((ok) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = () => ok(null);
    im.src = src;
  }));
  return loaded.get(src);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const between = ([lo, hi]) => (lo + Math.random() * (hi - lo)) * 1000;

export async function anchor(container) {
  const { idleFrames, idleFps, idlePauseSeconds, anchorFacesRight } = CONFIG.mascot;
  const frames = (await Promise.all(Array.from({ length: idleFrames }, (_, i) => probe(`${BASE}idle_${i + 1}.png`)))).filter(Boolean);
  if (!frames.length) return;
  // Same scale for every frame: the tallest frame gets the full height, others shrink in proportion,
  // so frames cropped to different sizes don't make the dragon grow and shrink.
  const H = 245, maxH = Math.max(...frames.map((f) => f.naturalHeight));
  const img = document.createElement("img");
  img.alt = "Harvey the dragon, anchoring the election desk";
  if (anchorFacesRight) img.className = "face-right";
  const showFrame = (k) => { img.src = frames[k].src; img.style.height = `${(H * frames[k].naturalHeight) / maxH}px`; };
  showFrame(0);
  container.append(img);
  if (frames.length < 2 || reduced()) return;
  // play the loop once, then rest on frame 1 for a random while, and again
  const loop = async () => {
    while (img.isConnected) {
      await wait(between(idlePauseSeconds));
      for (let k = 1; k <= frames.length && img.isConnected; k++) {
        await wait(1000 / idleFps);
        showFrame(k % frames.length);
      }
    }
  };
  loop();
}

export async function peek(container) {
  const s = CONFIG.mascot.peekStates;
  const [normal, left, right] = (await Promise.all([s.normal, s.winkLeft, s.winkRight].map((f) => probe(BASE + f)))).map((im) => im?.src || null);
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
  const show = (src, ms) => new Promise((ok) => { img.src = src; setTimeout(() => { img.src = normal; ok(); }, ms); });
  const loop = async () => {
    if (!img.isConnected) return;
    const roll = Math.random();
    if (roll < 0.2 && winks.length === 2) {           // the uncoordinated double wink
      await show(winks[0], 200); await new Promise((r) => setTimeout(r, 90)); await show(winks[1], 200);
    } else {
      await show(winks[Math.floor(Math.random() * winks.length)], 230);
    }
    setTimeout(loop, between(CONFIG.mascot.winkEverySeconds));
  };
  setTimeout(loop, 1500);
}
