// The desk band: the anchor dragon + one readout per chamber (P(Dem majority) bar, expected seats, seat histogram).
import { el, chance } from "./format.js";
import { anchor } from "./mascot.js";

function pbar(p) {
  const d = Math.round(p * 100);
  return el("div", {},
    el("div", { class: "pbar", role: "img", "aria-label": `Democrats ${chance(p)}, Republicans ${chance(1 - p)} chance of control` },
      el("span", { class: "d", style: `width:${Math.max(d, 0)}%` }, d >= 12 ? `D ${chance(p)}` : ""),
      el("span", { class: "r" }, 100 - d >= 12 ? `R ${chance(1 - p)}` : "")),
    el("div", { class: "pbar-legend" }, el("span", {}, "Democratic control"), el("span", {}, "Republican control")));
}

function seatHist(ch) {
  const dist = ch.seat_distribution || [];
  if (!dist.length) return null;
  const wrap = el("div", { class: "seathist" });
  const W = 300, H = 56;
  const lo = Math.max(0, Math.min(...dist.map((d) => d.seats)) - 2), hi = Math.max(...dist.map((d) => d.seats)) + 2;
  const x = d3.scaleLinear([lo, Math.max(hi, ch.majority_at + 1)], [0, W]);
  const y = d3.scaleLinear([0, d3.max(dist, (d) => d.p)], [0, H - 26]);
  const bw = Math.max(1, (x(lo + 1) - x(lo)) * 0.85);
  const svg = d3.select(wrap).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("preserveAspectRatio", "none")
    .attr("role", "img").attr("aria-label", `Seat distribution: majority at ${ch.majority_at}`);
  svg.selectAll("rect").data(dist).join("rect")
    .attr("x", (d) => x(d.seats) - bw / 2).attr("width", bw)
    .attr("y", (d) => H - 12 - y(d.p)).attr("height", (d) => y(d.p))
    .attr("fill", (d) => (d.seats >= ch.majority_at ? "var(--d-likely)" : "var(--r-likely)"));
  const g = svg.append("g").attr("class", "maj");
  g.append("line").attr("x1", x(ch.majority_at - 0.5)).attr("x2", x(ch.majority_at - 0.5)).attr("y1", 0).attr("y2", H - 10);
  g.append("text").attr("x", x(ch.majority_at - 0.5) + 4).attr("y", 9).text(`${ch.majority_at} for control`);
  const ax = svg.append("g").attr("class", "axis");
  for (const t of [lo, hi]) ax.append("text").attr("x", x(t)).attr("y", H).attr("text-anchor", t === lo ? "start" : "end").text(t);
  return wrap;
}

// Every readout has the same header row and a two-line seats line, so the bars and charts line up across chambers.
const HEAD = "display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:nowrap;min-height:2.1rem";

function chamberReadout(title, ch, { calledLine, toggle } = {}) {
  return el("div", { class: "readout" },
    el("div", { style: HEAD }, el("h2", {}, title), toggle || null),
    el("div", { class: "big" }, chance(ch.p_dem_majority), el("small", {}, "Democratic chance of control")),
    pbar(ch.p_dem_majority),
    el("p", { class: "sub", style: "min-height:2.7em" },
      `Expected Democratic seats${ch._withInd ? " (incl. independents)" : ""}: ${ch.dem_seats_mean} (80% range ${ch.dem_seats_lo80}–${ch.dem_seats_hi80})`),
    seatHist(ch),
    calledLine ? el("p", { class: "sub" }, calledLine) : null);
}

// Senate counted with independents (no-Democrat races) as Democratic seats, without rerunning anything: the seat
// distribution is combined with each independent's win chance; the exact chance/mean from the pipeline are used if present.
function withIndependents(ch, inds) {
  let dist = new Map((ch.seat_distribution || []).map((d) => [d.seats, d.p]));
  for (const r of inds) {
    const q = r.p_dem_win, next = new Map();
    for (const [k, p] of dist) {
      next.set(k, (next.get(k) || 0) + p * (1 - q));
      next.set(k + 1, (next.get(k + 1) || 0) + p * q);
    }
    dist = next;
  }
  const arr = [...dist].sort((a, b) => a[0] - b[0]).map(([seats, p]) => ({ seats, p }));
  const q = (t) => { let c = 0; for (const d of arr) { c += d.p; if (c >= t) return d.seats; } return arr[arr.length - 1]?.seats; };
  const wi = ch.with_independents || {};
  const mean = arr.reduce((s, d) => s + d.seats * d.p, 0);
  return { ...ch, _withInd: true, seat_distribution: arr, dem_seats_lo80: q(0.1), dem_seats_hi80: q(0.9),
    dem_seats_mean: wi.dem_seats_mean ?? Math.round(mean * 10) / 10,
    p_dem_majority: wi.p_dem_majority ?? arr.filter((d) => d.seats >= ch.majority_at).reduce((s, d) => s + d.p, 0) };
}

function governorReadout(g, races) {
  const toss = races.filter((r) => r.p_dem_win > 0.4 && r.p_dem_win < 0.6).length;
  return el("div", { class: "readout" },
    el("div", { style: HEAD }, el("h2", {}, "Governors")),
    el("div", { class: "big" }, `${g.dem_wins_mean}`, el("small", {}, `Expected Democratic wins, of ${g.races_up} races`)),
    el("p", { class: "sub" }, `${toss} ${toss === 1 ? "race is a toss-up" : "races are toss-ups"} (40–60% either way).`));
}

export function desk(data, { live = false } = {}) {
  const c = data.chambers;
  const called = (ch) => (live && ch.called ? `Called so far: Democrats ${ch.called.D}, Republicans ${ch.called.R}` : null);
  const inds = (data.races.senate || []).filter((r) => r.independent);
  let countInd = false;
  const senate = () => {
    const ch = countInd ? withIndependents(c.senate, inds) : c.senate;
    const btn = (label, on) => el("button", { type: "button", "aria-pressed": String(countInd === on),
      style: `padding:.15rem .55rem;border:0;font:inherit;font-size:.75rem;white-space:nowrap;cursor:pointer;${countInd === on ? "background:var(--spike-gold);color:var(--desk-navy);font-weight:700" : "background:transparent;color:#e9ecfb"}`,
      onclick: () => { if (countInd !== on) { countInd = on; slot.replaceWith(slot = senate()); } } }, label);
    const toggle = inds.length ? el("div", { style: "display:flex;align-items:center;gap:.4rem;font-size:.75rem;color:#c9cdea",
      title: `Independents with no Democrat on the ballot (${inds.map((r) => `${r.independent}, ${r.id}`).join("; ")}) say they won't caucus with either party` },
      el("div", { role: "group", "aria-label": "How to count independents",
        style: "display:inline-flex;border:1.5px solid rgba(255,255,255,.45);border-radius:999px;overflow:hidden" },
        btn("Ind. separate", false), btn("Ind. as Dem", true))) : null;
    return chamberReadout("Senate", ch, { calledLine: called(c.senate), toggle });
  };
  let slot = senate();
  const box = el("section", { class: "desk", "aria-label": "Headline forecast" },
    el("div", { class: "anchor" }),
    el("div", { class: "readouts" },
      slot,
      chamberReadout("House", c.house, { calledLine: called(c.house) }),
      governorReadout(c.governor, data.races.governor)));
  anchor(box.querySelector(".anchor"));
  return box;
}
