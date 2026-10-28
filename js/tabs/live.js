// Live results tab: same desk + map, fed by data/live.json, re-checked every minute while open.
import { CONFIG } from "../config.js";
import { getJSON } from "../data.js";
import { el, stamp } from "../components/format.js";
import { desk } from "../components/desk.js";
import { raceMap } from "../components/racemap.js";

function header(live) {
  return el("div", { class: "live-head" },
    el("div", { class: `live-badge${live?.is_live ? "" : " off"}` }, el("span", { class: "live-dot on", "aria-hidden": "true" }), "LIVE"),
    el("div", {},
      el("h1", {}, live?.is_live ? "Election night: live forecast" : "Live results start on election night"),
      el("p", { class: "live-stamp" }, live?.is_live ? `Last update ${stamp(live.updated_at)}. This page refreshes itself.`
        : "Polls start closing at 6pm Eastern (11pm UK time) on Tuesday 3 November. This page will update as votes are counted.")));
}

const explainer = () => el("aside", { class: "live-explainer" },
  el("h2", {}, "Read these with care"),
  el("p", {}, "On election night the model updates from partial counts, which arrive in a different order in every state (early votes, mail ballots and election-day votes are counted at different times). " +
    "It learned how counts unfold mostly from simulated elections, with only a couple of real election nights to check against, so these numbers can swing more than the pre-election forecast. " +
    "Races are only marked as called once news organisations call them."));

export async function render(view) {
  let live = null;
  try { live = await getJSON("live.json", { fresh: true }); } catch (e) { /* not published yet */ }
  const draw = () => {
    view.replaceChildren(header(live), explainer());
    if (live?.is_live) view.append(desk(live, { live: true }), raceMap(live, { title: "Every race, as counted" }));
  };
  draw();
  const timer = setInterval(async () => {
    try {
      const next = await getJSON("live.json", { fresh: true });
      if (next.updated_at !== live?.updated_at) { live = next; draw(); }
    } catch (e) { /* keep showing the last good update */ }
  }, CONFIG.refreshMinutes.live * 60 * 1000);
  return { destroy: () => clearInterval(timer) };
}
