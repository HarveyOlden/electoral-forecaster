// How it works tab: shows content/how-it-works.html. Write the explainer there (plain HTML), no code changes needed.
import { el } from "../components/format.js";

export async function render(view) {
  const res = await fetch(`content/how-it-works.html?t=${Date.now()}`, { cache: "no-store" });
  const html = res.ok ? await res.text() : "";
  view.append(el("article", { class: "prose", html: html.trim() || "<p>Coming soon.</p>" }));
}
