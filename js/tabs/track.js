// Track record tab: how the model did on past elections it never saw (data/backtest.json).
import { getJSON } from "../data.js";
import { el, pct, raceLabel } from "../components/format.js";

function stats(rows) {
  const n = rows.length;
  const right = rows.filter((r) => (r.pred_dem_share > 0.5) === (r.actual_dem_share > 0.5)).length;
  const err = d3.mean(rows, (r) => Math.abs(r.pred_dem_share - r.actual_dem_share)) * 100;
  const inside = rows.filter((r) => r.actual_dem_share >= r.lo90 && r.actual_dem_share <= r.hi90).length;
  return { n, right, err, inside };
}

function statBlock(big, label) {
  return el("div", { class: "stat" }, el("div", { class: "stat-big" }, big), el("div", { class: "stat-label" }, label));
}

function scatter(rows) {
  const wrap = el("div", { class: "scatter" });
  const W = 520, H = 420, m = { l: 48, r: 14, t: 12, b: 40 };
  const lo = Math.max(0, d3.min(rows, (r) => Math.min(r.pred_dem_share, r.actual_dem_share)) - 0.03);
  const hi = Math.min(1, d3.max(rows, (r) => Math.max(r.pred_dem_share, r.actual_dem_share)) + 0.03);
  const x = d3.scaleLinear([lo, hi], [m.l, W - m.r]), y = d3.scaleLinear([lo, hi], [H - m.b, m.t]);
  const svg = d3.select(wrap).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img")
    .attr("aria-label", "Predicted against actual Democratic share; dots on the diagonal were exactly right");
  svg.append("rect").attr("x", x(0.5)).attr("y", m.t).attr("width", W - m.r - x(0.5)).attr("height", y(0.5) - m.t).attr("fill", "rgba(42,95,211,.06)");
  svg.append("rect").attr("x", m.l).attr("y", y(0.5)).attr("width", x(0.5) - m.l).attr("height", H - m.b - y(0.5)).attr("fill", "rgba(217,71,47,.06)");
  svg.append("line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(hi)).attr("y2", y(hi)).attr("class", "diag");
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")));
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%")));
  svg.append("text").attr("class", "ax-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text("Forecast Democratic share (night before)");
  svg.append("text").attr("class", "ax-title").attr("transform", `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("Actual result");
  const tipNode = el("div", { class: "scatter-tip", hidden: true });
  wrap.append(tipNode);
  svg.append("g").selectAll("circle").data(rows).join("circle")
    .attr("cx", (r) => x(r.pred_dem_share)).attr("cy", (r) => y(r.actual_dem_share)).attr("r", 5.5)
    .attr("class", (r) => ((r.pred_dem_share > 0.5) === (r.actual_dem_share > 0.5) ? "hit" : "miss"))
    .on("mousemove", (ev, r) => {
      tipNode.hidden = false;
      tipNode.textContent = `${r.name}: forecast ${pct(r.pred_dem_share, 1)}, result ${pct(r.actual_dem_share, 1)}`;
      const b = wrap.getBoundingClientRect();
      tipNode.style.left = `${ev.clientX - b.left + 12}px`; tipNode.style.top = `${ev.clientY - b.top + 12}px`;
    })
    .on("mouseleave", () => { tipNode.hidden = true; });
  return wrap;
}

function missTable(rows) {
  const miss = rows.filter((r) => (r.pred_dem_share > 0.5) !== (r.actual_dem_share > 0.5))
    .sort((a, b) => Math.abs(a.pred_dem_share - 0.5) - Math.abs(b.pred_dem_share - 0.5));
  if (!miss.length) return el("p", { class: "empty" }, "Every winner called correctly.");
  return el("div", { class: "table-wrap" }, el("table", { class: "data" },
    el("thead", {}, el("tr", {}, el("th", {}, "Wrong call"), el("th", { class: "num" }, "Forecast (Dem)"), el("th", { class: "num" }, "Result (Dem)"))),
    el("tbody", {}, miss.map((r) => el("tr", {}, el("td", {}, r.name), el("td", { class: "num" }, pct(r.pred_dem_share, 1)),
      el("td", { class: "num" }, pct(r.actual_dem_share, 1)))))));
}

function houseChart(years) {
  const wrap = el("div", { class: "house-seats" });
  const W = 640, H = 230, m = { l: 48, r: 16, t: 14, b: 30 };
  const ys = years.filter((y) => y.house_seats);
  const x = d3.scalePoint(ys.map((y) => y.year), [m.l + 20, W - m.r - 20]);
  const lo = d3.min(ys, (y) => Math.min(y.house_seats.lo90, y.house_seats.actual_dem)) - 8;
  const hi = d3.max(ys, (y) => Math.max(y.house_seats.hi90, y.house_seats.actual_dem)) + 8;
  const yS = d3.scaleLinear([lo, hi], [H - m.b, m.t]);
  const svg = d3.select(wrap).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", "House seats: forecast range against the actual Democratic seat count");
  svg.append("line").attr("class", "maj-line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", yS(218)).attr("y2", yS(218));
  svg.append("text").attr("class", "maj-text").attr("x", W - m.r).attr("y", yS(218) - 5).attr("text-anchor", "end").text("218 for a majority");
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x));
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(yS).ticks(5));
  const g = svg.append("g").selectAll("g").data(ys).join("g");
  g.append("rect").attr("x", (y) => x(y.year) - 9).attr("width", 18).attr("y", (y) => yS(y.house_seats.hi90))
    .attr("height", (y) => yS(y.house_seats.lo90) - yS(y.house_seats.hi90)).attr("rx", 4).attr("class", "range-bar");
  g.append("line").attr("x1", (y) => x(y.year) - 11).attr("x2", (y) => x(y.year) + 11).attr("y1", (y) => yS(y.house_seats.pred_dem))
    .attr("y2", (y) => yS(y.house_seats.pred_dem)).attr("class", "pred-line");
  g.append("circle").attr("cx", (y) => x(y.year)).attr("cy", (y) => yS(y.house_seats.actual_dem)).attr("r", 5.5).attr("class", "actual-dot");
  wrap.append(el("div", { class: "legend" },
    el("span", {}, el("i", { class: "k-range" }), "Forecast 90% range"), el("span", {}, el("i", { class: "k-pred" }), "Forecast"),
    el("span", {}, el("i", { class: "k-actual" }), "Actual Democratic seats")));
  return wrap;
}

export async function render(view) {
  const bt = await getJSON("backtest.json");
  const all = bt.years.flatMap((y) => Object.values(y.races).flat());
  const s = stats(all);
  view.append(
    el("header", { class: "page-head" },
      el("h1", {}, "How it did on past elections"),
      el("p", {}, `Every year below was forecast by a version of the model trained without that year, using only what was known the night before.`)),
    el("div", { class: "stats" },
      statBlock(`${Math.round((s.right / s.n) * 100)}%`, `of ${s.n} statewide winners called correctly`),
      statBlock(`${s.err.toFixed(1)} pts`, "average miss on the Democratic share"),
      statBlock(`${Math.round((s.inside / s.n) * 100)}%`, "of results inside the 90% range (should be about 90%)")));

  const yearBtns = bt.years.map((y) => el("button", { type: "button", "data-y": y.year, "aria-pressed": "false" }, `${y.year}`));
  const kindBox = el("div", { class: "segmented", role: "group", "aria-label": "Race" });
  const body = el("div", { class: "track-body" });
  view.append(el("section", { class: "track" },
    el("div", { class: "map-toolbar" }, el("h2", {}, "Forecast against result"),
      el("div", { class: "segmented years", role: "group", "aria-label": "Year" }, yearBtns), kindBox),
    body));
  view.append(el("section", { class: "track" }, el("div", { class: "map-toolbar" }, el("h2", {}, "House seats")), houseChart(bt.years)));

  let year = bt.years[bt.years.length - 1], kind = null;
  const draw = () => {
    yearBtns.forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.y === year.year)));
    const kinds = Object.keys(year.races);
    if (!kinds.includes(kind)) kind = kinds.includes("senate") ? "senate" : kinds[0];
    kindBox.replaceChildren(...kinds.map((k) => el("button", { type: "button", "aria-pressed": String(k === kind), onclick: () => { kind = k; draw(); } }, raceLabel[k])));
    const rows = year.races[kind];
    const st = stats(rows);
    body.replaceChildren(
      el("div", { class: "track-grid" },
        scatter(rows),
        el("div", {},
          el("p", { class: "track-line" }, `${year.year} ${raceLabel[kind]}: ${st.right} of ${st.n} winners called correctly, average miss ${st.err.toFixed(1)} pts.`),
          missTable(rows))));
  };
  yearBtns.forEach((b) => b.addEventListener("click", () => { year = bt.years.find((y) => y.year === +b.dataset.y); draw(); }));
  draw();
}
