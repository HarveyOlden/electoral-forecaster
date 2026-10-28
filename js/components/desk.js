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

function chamberReadout(title, ch, { calledLine } = {}) {
  return el("div", { class: "readout" },
    el("h2", {}, title),
    el("div", { class: "big" }, chance(ch.p_dem_majority), el("small", {}, "Democratic chance of control")),
    pbar(ch.p_dem_majority),
    el("p", { class: "sub" }, `Expected Democratic seats: ${ch.dem_seats_mean} (80% range ${ch.dem_seats_lo80}–${ch.dem_seats_hi80})`),
    seatHist(ch),
    calledLine ? el("p", { class: "sub" }, calledLine) : null);
}

function governorReadout(g, races) {
  const toss = races.filter((r) => r.p_dem_win > 0.4 && r.p_dem_win < 0.6).length;
  return el("div", { class: "readout" },
    el("h2", {}, "Governors"),
    el("div", { class: "big" }, `${g.dem_wins_mean}`, el("small", {}, `Expected Democratic wins, of ${g.races_up} races`)),
    el("p", { class: "sub" }, `${toss} ${toss === 1 ? "race is a toss-up" : "races are toss-ups"} (40–60% either way).`));
}

export function desk(data, { live = false } = {}) {
  const c = data.chambers;
  const called = (ch) => (live && ch.called ? `Called so far: Democrats ${ch.called.D}, Republicans ${ch.called.R}` : null);
  const box = el("section", { class: "desk", "aria-label": "Headline forecast" },
    el("div", { class: "anchor" }),
    el("div", { class: "readouts" },
      chamberReadout("Senate", c.senate, { calledLine: called(c.senate) }),
      chamberReadout("House", c.house, { calledLine: called(c.house) }),
      governorReadout(c.governor, data.races.governor)));
  anchor(box.querySelector(".anchor"));
  return box;
}
