// App shell: tabs, contact links, ticker, footer, and the router. Each tab lives in js/tabs/<id>.js.
import { CONFIG } from "./config.js";
import { getJSON, watchForUpdates } from "./data.js";
import { el, chance, stamp } from "./components/format.js";
import { close as closePanel } from "./components/panel.js";

const view = document.getElementById("view");
const ICON = {
  mail: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>',
  linkedin: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5ZM3 9.5h4V21H3zM9.5 9.5h3.8v1.6h.06c.53-1 1.83-2.06 3.77-2.06 4.03 0 4.77 2.65 4.77 6.1V21h-4v-5.1c0-1.22-.02-2.78-1.7-2.78-1.7 0-1.96 1.33-1.96 2.7V21h-4z"/></svg>',
};

function buildChrome(meta) {
  const nav = document.getElementById("tabs");
  nav.replaceChildren(...CONFIG.tabs.map((t) => el("a", { href: `#${t.id}`, "data-tab": t.id },
    t.live ? el("span", { class: `live-dot${meta?.live_available ? " on" : ""}`, "aria-hidden": "true" }) : null, t.label)));
  const c = CONFIG.contact;
  document.getElementById("contact").replaceChildren(
    el("a", { href: `mailto:${c.email}`, "aria-label": `Email ${c.email}`, title: c.email, html: ICON.mail }),
    el("a", { href: c.linkedin, target: "_blank", rel: "noopener", "aria-label": "LinkedIn", title: "LinkedIn", html: ICON.linkedin }));
  document.getElementById("demo-banner").hidden = !meta?.demo;
  document.getElementById("footer").replaceChildren(
    el("p", {}, `KERMIT is an independent, experimental forecast. Model output updated ${stamp(meta?.generated_at)}. `,
      el("a", { href: "#sources" }, "Data sources"), "."));
}

async function buildTicker() {
  const t = document.getElementById("ticker");
  try {
    const f = await getJSON("forecast.json");
    const s = f.chambers.senate, h = f.chambers.house, g = f.chambers.governor;
    t.replaceChildren(el("div", { class: "ticker-inner" },
      el("span", { class: "ticker-stamp" }, `Updated ${stamp(f.generated_at)}`),
      el("div", { class: "ticker-items" },
        el("span", {}, `Senate: Democrats ${chance(s.p_dem_majority)} to win control`),
        el("span", {}, `House: Democrats ${chance(h.p_dem_majority)} to win control`),
        el("span", {}, `Governors: ${g.dem_wins_mean} of ${g.races_up} expected Democratic`))));
  } catch (e) { t.replaceChildren(); }
}

let current = null;
async function route() {
  closePanel();
  const id = (location.hash.slice(1) || CONFIG.tabs[0].id).split("/")[0];
  const tab = CONFIG.tabs.find((t) => t.id === id) || CONFIG.tabs[0];
  document.querySelectorAll("#tabs a").forEach((a) => (a.dataset.tab === tab.id ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  current?.destroy?.();
  view.replaceChildren(el("p", { class: "empty" }, "Loading…"));
  try {
    const mod = await import(`./tabs/${tab.id}.js`);
    view.replaceChildren();
    current = (await mod.render(view, { tab })) || null;
    document.title = `${tab.label}: KERMIT election forecast`;
  } catch (e) {
    console.error(e);
    view.replaceChildren(el("p", { class: "empty" }, `This section didn't load (${e.message}). Refresh the page to try again.`));
  }
}

async function start() {
  let meta = null;
  try { meta = await getJSON("meta.json", { fresh: true }); } catch (e) { /* site still works without it */ }
  buildChrome(meta);
  buildTicker();
  window.addEventListener("hashchange", route);
  await route();
  // when the pipeline publishes new output, refresh what's on screen
  watchForUpdates(CONFIG.refreshMinutes.forecast, (m) => { buildChrome(m); buildTicker(); route(); });
}

start();
