// The side panel for one race: result, confidence, and what moved the forecast most.
import { el, chance, pct, rating, raceLabel } from "./format.js";

let scrim, panel, lastFocus, onCloseCb;

function ensure() {
  if (panel) return;
  scrim = el("div", { class: "panel-scrim", hidden: true, onclick: close });
  panel = el("aside", { class: "panel", hidden: true, role: "dialog", "aria-modal": "true", "aria-labelledby": "panel-title" });
  document.body.append(scrim, panel);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) close(); });
}

export function close() {
  if (!panel || panel.hidden) return;
  panel.hidden = true;
  scrim.hidden = true;
  onCloseCb?.();
  lastFocus?.focus?.();
}

function rangeChart(r) {
  const wrap = el("div", { class: "range" });
  const W = 380, H = 54, x = d3.scaleLinear([0.25, 0.75], [10, W - 10]).clamp(true);
  const svg = d3.select(wrap).append("svg").attr("viewBox", `0 0 ${W} ${H}`)
    .attr("role", "img").attr("aria-label", `Expected Democratic share ${pct(r.dem_share, 1)}, 90% range ${pct(r.lo90, 1)} to ${pct(r.hi90, 1)}`);
  svg.append("line").attr("class", "mid").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 4).attr("y2", 34);
  svg.append("rect").attr("x", x(r.lo90)).attr("width", Math.max(2, x(r.hi90) - x(r.lo90))).attr("y", 12).attr("height", 14)
    .attr("rx", 3).attr("fill", "var(--tossup)");
  svg.append("rect").attr("x", x(r.dem_share) - 2).attr("width", 4).attr("y", 7).attr("height", 24)
    .attr("fill", r.dem_share >= 0.5 ? (r.independent ? "#1f7a3a" : "var(--dem)") : "var(--rep)");
  const ax = svg.append("g").attr("class", "axis");
  [[0.3, "R +40"], [0.4, "R +20"], [0.5, "Even"], [0.6, "D +20"], [0.7, "D +40"]].forEach(([v, t]) =>
    ax.append("text").attr("x", x(v)).attr("y", 50).attr("text-anchor", "middle").text(t));
  return wrap;
}

function driverRow(d) {
  const max = 4;                                       // pts at the end of the little bar
  const w = Math.min(Math.abs(d.shift_pp) / max, 1) * 50;
  const toD = d.shift_pp >= 0;
  return el("li", { class: "driver" },
    el("div", { class: "label" }, d.label),
    d.detail ? el("div", { class: "detail" }, d.detail) : null,
    el("div", { class: "shift" },
      el("div", { class: "shift-bar", "aria-hidden": "true" },
        el("i", { style: `${toD ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:var(--${toD ? "dem" : "rep"})` })),
      el("div", { class: "shift-text", style: `color:var(--${toD ? "dem" : "rep"})` },
        `${toD ? "D" : "R"} +${Math.abs(d.shift_pp).toFixed(1)} pts`)));
}

export function open(r, chamber, { onClose } = {}) {
  ensure();
  onCloseCb = onClose;
  lastFocus = document.activeElement;
  const rt = rating(r.p_dem_win);
  const chipStyle = `background:var(--${rt.key});color:${/safe|likely/.test(rt.key) ? "#fff" : "var(--ink)"}`;
  const live = r.reported_pct != null;
  const items = [
    el("button", { class: "panel-close", "aria-label": "Close", onclick: close }, "×"),
    el("div", { class: "kicker" }, `${raceLabel[chamber]} race, 2026`),
    el("h2", { id: "panel-title" }, r.name),
    el("span", { class: "chip", style: chipStyle }, r.note ? "Not modelled" : r.uncontested ? "Uncontested" : rt.label),
    live ? el("p", { class: "kicker" }, r.status === "called" ? `Called for the ${r.called_for === "D" ? "Democrat" : "Republican"}` :
      `${Math.round(r.reported_pct)}% of the expected vote counted`) : null,
    r.uncontested
      ? el("section", {}, el("p", {}, r.note || `Only one major party is on the ballot here, so the seat goes to the ${r.p_dem_win > 0.5 ? "Democrat" : "Republican"}.`))
      : [
        el("section", {},
          el("h3", {}, "Chance of winning"),
          el("div", { class: "odds" },
            el("div", { class: "side d", style: r.independent ? "color:#1f7a3a" : null }, chance(r.p_dem_win), el("small", {}, dLabel(r))),
            el("div", { class: "side r" }, chance(1 - r.p_dem_win), el("small", {}, "Republican"))),
          el("div", { class: "pbar" }, el("span", { class: "d", style: `width:${r.p_dem_win * 100}%${r.independent ? ";background:#1f7a3a" : ""}` }), el("span", { class: "r" }))),
        r.independent ? el("p", { class: "footnote" }, `No Democrat is on the ballot here; the model treats ${r.independent} as the non-Republican candidate. An independent win is not counted as a Democratic seat.`) : null,
        el("section", {},
          el("h3", {}, "Expected vote"),
          el("p", { class: "range-caption" },
            `${dLabel(r)} ${pct(r.dem_share, 1)}, Republican ${pct(1 - r.dem_share, 1)} of the two-candidate vote. 90% of the time the result should land between ${pct(r.lo90, 1)} and ${pct(r.hi90, 1)} for the ${r.independent ? "independent" : "Democrat"}.`),
          rangeChart(r)),
        r.drivers?.length ? el("section", {},
          el("h3", {}, "What moved this forecast most"),
          el("ol", { class: "drivers" }, r.drivers.slice(0, 5).map(driverRow)),
          el("p", { class: "footnote" },
            "Each figure is how far the forecast moves when the model is run without that piece of information. D +2.0 pts means that information moved the Democrat's expected share up 2 points.")) : null,
      ]];
  panel.replaceChildren(...items.flat().filter(Boolean));
  panel.hidden = false;
  scrim.hidden = false;
  panel.querySelector(".panel-close").focus();
}


function dLabel(r) {
  return r.independent ? `${r.independent} (independent)` : "Democrat";
}
