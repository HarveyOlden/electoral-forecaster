# KERMIT website

A static site: HTML, CSS and JavaScript only, no build step. It reads the model's output from `data/*.json`
(format: `DATA_FORMAT.md`) and never runs the model.

## Preview on your computer
The site uses JavaScript modules, which browsers won't run from a double-clicked file. From this folder:
```
python -m http.server 8000
```
then open http://localhost:8000.

## Where things live
```
index.html                 page shell (header, tabs, ticker, footer)
js/config.js               contact links, data location, refresh timing, tabs list, mascot, map colour bands
js/app.js                  tab router, header, ticker
js/data.js                 loads data/*.json (always fresh)
js/tabs/<tab>.js           one file per tab: forecast, live, track, how, sources
js/components/             shared pieces: desk (headline numbers), map, racemap (map + controls), panel (race detail), mascot, format
css/tokens.css             every colour / font / size
css/layout.css, components.css, tabs/<tab>.css
content/how-it-works.html  write the explainer here (plain HTML)
data/                      model output (replaced by the pipeline) + static map files
assets/mascot/             the dragon
tools/                     demo-data and hex-layout generators
```

**Change a tab:** edit `js/tabs/<tab>.js` (and `css/tabs/<tab>.css`).
**Add a tab:** create `js/tabs/<id>.js` exporting `async function render(view)`, then add `{ id, label }` to `tabs` in `js/config.js`.
**Change colours or fonts:** `css/tokens.css`.

## Mascot files
Put PNGs with transparent backgrounds in `assets/mascot/`, named exactly:

| File | What |
|---|---|
| `idle_1.png` … `idle_8.png` | the 8-frame idle loop at the desk (plays at 1.5 frames per second) |
| `peek_normal.png` | close-up head, eyes open |
| `peek_wink_left.png` | close-up, left eye shut |
| `peek_wink_right.png` | close-up, right eye shut |

Missing frames are skipped, so the site works with any subset. The desk dragon is mirrored so it faces the numbers;
set `anchorFacesRight: false` in `js/config.js` to show it as drawn. Crop the PNGs tight around the dragon,
about 700px tall is plenty.

## Hosting (GitHub Pages, free)
1. Create a **public** repository on GitHub (Pages on a free account needs a public repo; the site is public anyway).
2. Upload everything in this folder to the repository root (**Add file → Upload files**, drag the folder contents in).
3. **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: main, folder: / (root) → Save.**
4. After a minute the site is at `https://<your-username>.github.io/<repo-name>/`.

Updating the forecast = committing new files into `data/`. Pages redeploys on every commit (about a minute), and
open pages pick up the new `meta.json` within `refreshMinutes` (15 min; the Live tab checks every minute).
The scheduled model job will do exactly that commit. It can live in a separate private repo and push only
`data/` into this one.

A custom domain (optional): Settings → Pages → Custom domain.
