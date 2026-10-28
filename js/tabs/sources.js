// Data sources tab: the manifest (data/sources.csv) as a table + a download button.
import { CONFIG } from "../config.js";
import { getCSV } from "../data.js";
import { el } from "../components/format.js";

export async function render(view) {
  const rows = await getCSV("sources.csv");
  view.append(
    el("header", { class: "page-head" },
      el("h1", {}, "Data sources"),
      el("p", {}, "Everything the model was trained on or reads today. Thanks to everyone who publishes election data openly.")),
    el("p", {}, el("a", { class: "btn", href: `${CONFIG.dataBase}sources.csv`, download: "kermit_data_sources.csv" }, "Download the list (CSV)")),
    el("div", { class: "table-wrap", style: "margin-top:1rem" }, el("table", { class: "data" },
      el("thead", {}, el("tr", {}, el("th", {}, "Dataset"), el("th", {}, "From"), el("th", {}, "Used for"))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        el("td", {}, el("div", { style: "font-weight:650" }, r.dataset), el("div", { class: "range-caption" }, r.category)),
        el("td", {}, r.link ? el("a", { href: r.link, target: "_blank", rel: "noopener" }, r.provider) : r.provider),
        el("td", {}, r.used_for)))))));
}
