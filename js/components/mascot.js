// The anchor dragon: an idle loop at the desk, and a head peeking over the map that blinks now and then.
// Frames are drawn on a canvas so they always line up at the same size, however each PNG was cropped.
import { CONFIG } from "../config.js";

const BASE = "assets/mascot/";
const M = {                              // defaults, overridden by CONFIG.mascot
  idleFrames: 8, anchorFacesRight: true,
  loopFps: 6,                            // frames per second while moving
  holdSeconds: [2, 6],                   // random rest on frame 1 and on frame 4
  peekStates: { normal: "peek_normal.png", winkLeft: "peek_wink_left.png", winkRight: "peek_wink_right.png" },
  winkEverySeconds: [3, 8], winkMs: 170,
  ...(CONFIG.mascot || {}),
};
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const between = (range) => { const [lo, hi] = range || [2, 6]; return (lo + Math.random() * (hi - lo)) * 1000; };

function load(src) {                     // the loaded image, or null if the file isn't there
  return new Promise((ok) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = () => ok(null);
    im.src = src;
  });
}

function opaqueBox(im) {                 // bounding box of the non-transparent pixels
  const c = document.createElement("canvas");
  c.width = im.naturalWidth; c.height = im.naturalHeight;
  const g = c.getContext("2d");
  g.drawImage(im, 0, 0);
  let d;
  try { d = g.getImageData(0, 0, c.width, c.height).data; } catch (e) { return { x: 0, y: 0, w: c.width, h: c.height }; }
  let x0 = c.width, y0 = c.height, x1 = -1, y1 = -1;
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
    if (d[(y * c.width + x) * 4 + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  return x1 < 0 ? { x: 0, y: 0, w: c.width, h: c.height } : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// One canvas that can show any of the frames at a constant size.
// Same-size PNGs (exported from one artboard): cropped to their shared box, so they line up exactly.
// Different sizes: each trimmed to the dragon and scaled to the same height, aligned at the bottom and on the tail side.
function sprite(images, { maxPx = 800, alignRight = true } = {}) {
  const sameSize = images.every((im) => im.naturalWidth === images[0].naturalWidth && im.naturalHeight === images[0].naturalHeight);
  const boxes = images.map(opaqueBox);
  let place, W, H;
  if (sameSize) {
    const x0 = Math.min(...boxes.map((b) => b.x)), y0 = Math.min(...boxes.map((b) => b.y));
    const x1 = Math.max(...boxes.map((b) => b.x + b.w)), y1 = Math.max(...boxes.map((b) => b.y + b.h));
    const s = Math.min(1, maxPx / Math.max(x1 - x0, y1 - y0));
    W = Math.round((x1 - x0) * s); H = Math.round((y1 - y0) * s);
    place = () => ({ sx: x0, sy: y0, sw: x1 - x0, sh: y1 - y0, dx: 0, dy: 0, dw: W, dh: H });
  } else {
    H = Math.min(maxPx, Math.max(...boxes.map((b) => b.h)));
    const widths = boxes.map((b) => (b.w * H) / b.h);
    W = Math.round(Math.max(...widths));
    place = (k) => {
      const b = boxes[k], dw = widths[k];
      return { sx: b.x, sy: b.y, sw: b.w, sh: b.h, dx: alignRight ? W - dw : 0, dy: 0, dw, dh: H };
    };
  }
  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  const show = (k) => {
    const p = place(k);
    g.clearRect(0, 0, W, H);
    g.drawImage(images[k], p.sx, p.sy, p.sw, p.sh, p.dx, p.dy, p.dw, p.dh);
  };
  show(0);
  return { canvas, show };
}

export async function anchor(container) {
  const frames = (await Promise.all(Array.from({ length: M.idleFrames }, (_, i) => load(`${BASE}idle_${i + 1}.png`)))).filter(Boolean);
  if (!frames.length) return;
  const { canvas, show } = sprite(frames);
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Harvey the dragon, anchoring the election desk");
  Object.assign(canvas.style, { position: "absolute", left: "14px", bottom: "0", height: "245px", width: "auto",
    maxWidth: "190px", objectFit: "contain", objectPosition: "bottom", transform: M.anchorFacesRight ? "scaleX(-1)" : "" });
  container.append(canvas);
  if (frames.length < 2 || reduced()) return;
  // rest on 1 -> play 1..4 -> rest on 4 -> play 4..8 and back to 1 -> repeat (each rest a random length)
  const n = frames.length, mid = Math.min(3, n - 1);
  const play = async (from, to) => {                 // frame indices, 0-based, wrapping past the last frame
    for (let k = from + 1; k <= to && canvas.isConnected; k++) { await wait(1000 / M.loopFps); show(k % n); }
  };
  while (canvas.isConnected) {
    await wait(between(M.holdSeconds));
    await play(0, mid);
    await wait(between(M.holdSeconds));
    await play(mid, n);
  }
}

export async function peek(container) {
  const s = M.peekStates;
  const [normal, left, right] = await Promise.all([s.normal, s.winkLeft, s.winkRight].map((f) => load(BASE + f)));
  if (!normal) return;
  const imgs = [normal, left, right].filter(Boolean);
  const { canvas, show } = sprite(imgs, { maxPx: 500 });
  const box = document.createElement("div");
  box.className = "peek";
  box.setAttribute("aria-hidden", "true");
  Object.assign(canvas.style, { position: "absolute", left: "0", top: "0", width: "112px", height: "auto" });
  box.append(canvas);
  container.append(box);
  if (reduced() || imgs.length < 2) return;
  const iL = left ? imgs.indexOf(left) : -1, iR = right ? imgs.indexOf(right) : -1;
  await wait(between(M.winkEverySeconds));
  while (canvas.isConnected) {            // every so often: normal -> left blink -> right blink -> normal
    if (iL >= 0) { show(iL); await wait(M.winkMs); }
    if (iR >= 0) { show(iR); await wait(M.winkMs); }
    show(0);
    await wait(between(M.winkEverySeconds));
  }
}
