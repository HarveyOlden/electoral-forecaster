// Forecast tab: the desk (headline numbers) + the race map.
import { getJSON } from "../data.js";
import { el, stamp } from "../components/format.js";
import { desk } from "../components/desk.js";
import { raceMap } from "../components/racemap.js";

export async function render(view) {
  const f = await getJSON("forecast.json");
  view.append(
    desk(f),
    raceMap(f, { title: "Every race" }),
    el("p", { class: "range-caption", style: "margin-top:1rem" },
      `Forecast as of ${stamp(f.data_cutoff)}. Click a state or district for its forecast and what drove it.`));
}
