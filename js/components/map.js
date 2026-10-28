// The map: states (Senate, governors) or one hexagon per House district. Colour = chance the Democrat wins.
import { getStatic } from "../data.js";
import { el, chance, ratingColor, rating } from "./format.js";

const tip = (() => {
  let node;
  return {
    show(html, ev) {
      if (!node) { node = el("div", { class: "tooltip", role: "status" }); document.body.append(node); }
      node.innerHTML = html;
      node.hidden = false;
      const x = Math.min(ev.clientX + 14, window.innerWidth - node.offsetWidth - 8);
      node.style.left = `${x}px`;
      node.style.top = `${ev.clientY + 14}px`;
    },
    hide() { if (node) node.hidden = true; },
  };
})();

export async function drawMap(frame, { chamber, races, onSelect }) {
  frame.querySelector("svg")?.remove();
  const byId = new Map(races.map((r) => [r.id, r]));
  let svg, sel;
  const tipHtml = (r) => `<b>${r.name}</b>${r.uncontested ? "Only one major party on the ballot" :
    `Democrat ${chance(r.p_dem_win)} to win${r.reported_pct != null ? `<br>${Math.round(r.reported_pct)}% of the vote counted` : ""}`}`;

  if (chamber === "house") {
    const lay = await getStatic("data/house_hex_layout.json");
    const s = 10, w = Math.sqrt(3) * s;
    const pos = lay.hexes.map((h) => ({ ...h, x: w * (h.q + h.r / 2), y: 1.5 * s * h.r }));
    const [x0, x1] = d3.extent(pos, (p) => p.x), [y0, y1] = d3.extent(pos, (p) => p.y);
    svg = d3.select(frame).append("svg").attr("viewBox", `${x0 - w} ${y0 - s * 1.2} ${x1 - x0 + 2 * w} ${y1 - y0 + 2.4 * s}`)
      .attr("role", "img").attr("aria-label", "House forecast: one hexagon per district, coloured by the Democrat's chance of winning");
    const hex = d3.range(6).map((k) => [s * Math.cos(Math.PI / 6 + (k * Math.PI) / 3), s * Math.sin(Math.PI / 6 + (k * Math.PI) / 3)]);
    sel = svg.append("g").selectAll("path").data(pos).join("path")
      .attr("d", (p) => `M${hex.map(([a, b]) => `${p.x + a * 0.96},${p.y + b * 0.96}`).join("L")}Z`)
      .attr("data-id", (p) => p.id);
    // state labels at each state's centre
    const states = d3.groups(pos, (p) => p.state).map(([st, ps]) => ({ st, x: d3.mean(ps, (p) => p.x), y: d3.mean(ps, (p) => p.y) }));
    svg.append("g").selectAll("text").data(states).join("text").attr("class", "hex-label")
      .attr("x", (d) => d.x).attr("y", (d) => d.y + 3).attr("text-anchor", "middle").text((d) => d.st);
  } else {
    const topo = await getStatic("data/geo/states-albers-10m.json");
    const lay = await getStatic("data/house_hex_layout.json");
    const fips2postal = new Map(lay.states.map((s) => [s.fips, s.postal]));
    const feats = topojson.feature(topo, topo.objects.states).features.map((f) => ({ ...f, id: fips2postal.get(f.id) || f.id }));
    const path = d3.geoPath();
    svg = d3.select(frame).append("svg").attr("viewBox", "0 0 975 610").attr("role", "img")
      .attr("aria-label", `${chamber === "senate" ? "Senate" : "Governor"} forecast by state, coloured by the Democrat's chance of winning`);
    sel = svg.append("g").selectAll("path").data(feats).join("path").attr("d", path).attr("data-id", (f) => f.id);
    svg.append("path").datum(topojson.mesh(topo, topo.objects.states, (a, b) => a !== b))
      .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 0.8).attr("d", path).attr("pointer-events", "none");
  }

  sel.attr("class", (d) => `unit${byId.has(d.id) ? "" : " none"}`)
    .attr("fill", (d) => (byId.has(d.id) ? ratingColor(byId.get(d.id).p_dem_win) : null))
    .attr("tabindex", (d) => (byId.has(d.id) && chamber !== "house" ? 0 : null))
    .attr("aria-label", (d) => (byId.has(d.id) ? `${byId.get(d.id).name}: ${rating(byId.get(d.id).p_dem_win).label}` : null))
    .on("mousemove", (ev, d) => byId.has(d.id) && tip.show(tipHtml(byId.get(d.id)), ev))
    .on("mouseleave", tip.hide)
    .on("click", (ev, d) => { if (byId.has(d.id)) { tip.hide(); onSelect(byId.get(d.id)); } })
    .on("keydown", (ev, d) => { if ((ev.key === "Enter" || ev.key === " ") && byId.has(d.id)) { ev.preventDefault(); onSelect(byId.get(d.id)); } });

  return {
    highlight(id) { sel.classed("selected", (d) => d.id === id); if (id) sel.filter((d) => d.id === id).raise(); },
  };
}

export function legend() {
  const seen = new Set();
  const items = [];
  for (const p of [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]) {
    const r = rating(p);
    if (!seen.has(r.key)) { seen.add(r.key); items.push(el("span", {}, el("i", { style: `background:var(--${r.key})` }), r.label)); }
  }
  items.push(el("span", {}, el("i", { style: "background:var(--muted-fill)" }), "No race this year"));
  return el("div", { class: "legend", "aria-label": "Map colour key" }, items);
}
