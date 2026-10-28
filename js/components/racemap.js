// Map section used by the Forecast and Live tabs: chamber switch, race finder, the map, the key, the peeking dragon.
import { el, raceLabel } from "./format.js";
import { drawMap, legend } from "./map.js";
import * as panel from "./panel.js";
import { peek } from "./mascot.js";

const CHAMBERS = ["senate", "house", "governor"];
let remembered = "senate";                       // keep the chosen chamber when switching tabs

export function raceMap(data, { title = "Every race" } = {}) {
  const frame = el("div", { class: "map-frame" });
  const finder = el("select", { "aria-label": "Find a race" });
  const buttons = CHAMBERS.map((c) => el("button", { type: "button", "aria-pressed": "false", "data-c": c }, raceLabel[c]));
  const section = el("section", { class: "map-section", "aria-label": "Race map" },
    el("div", { class: "map-toolbar" },
      el("h2", {}, title),
      el("div", { class: "segmented", role: "group", "aria-label": "Choose a chamber" }, buttons),
      el("label", { class: "finder" }, finder)),
    frame,
    legend());
  peek(frame);

  let map, chamber;
  const show = async (c) => {
    chamber = remembered = c;
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.c === c)));
    const races = data.races[c] || [];
    const sorted = [...races].sort((a, b) => a.name.localeCompare(b.name, "en", { numeric: true }));
    finder.replaceChildren(el("option", { value: "" }, `Find a ${raceLabel[c].toLowerCase()} race`),
      sorted.map((r) => el("option", { value: r.id }, r.name)));
    map = await drawMap(frame, { chamber: c, races, onSelect: select });
  };
  const select = (r) => {
    map?.highlight(r.id);
    finder.value = r.id;
    panel.open(r, chamber, { onClose: () => { map?.highlight(null); finder.value = ""; } });
  };
  buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.c)));
  finder.addEventListener("change", () => {
    const r = (data.races[chamber] || []).find((x) => x.id === finder.value);
    if (r) select(r);
  });
  show(remembered);
  return section;
}
