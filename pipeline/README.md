# KERMIT forecast pipeline (runs on GitHub Actions)

Every 12 hours until election night, then every 30 minutes during it, GitHub runs `pipeline/run.py`, which
1. pulls fresh inputs and checks them,
2. runs KERMIT (strict mode: each race reads only its own state polls) and the Bayesian model, blends them,
3. works out the top-5 drivers of every race (hide one input group, rerun, measure the shift),
4. writes `data/forecast.json` (or `data/live.json` on the night) and `data/meta.json`, and commits them.
GitHub Pages redeploys on the commit, and open pages pick up the new numbers by themselves.

## What it pulls

| When | Source | Used as |
|---|---|---|
| every run | VoteHub API: generic ballot, Trump approval | national polls, approval |
| every run | Wikipedia 2026 Senate / governor race pages | state polls (nominee match-up only) |
| every pre-election run | NC State Board of Elections per-ballot file | North Carolina early vote |
| election night | New York Times results feeds (AP data) | county counts (House, summed per county), race progress, AP calls |
| frozen in the bundle | everything else: polls up to the export, House district polls, specials, money (12G), incumbency, fundamentals | |

## Checks before anything reaches the model

- **Polls**: shares 5-95, D+R 40-101, dated inside the cycle and not in the future, real pollster (aggregators
  dropped), race actually on that state's ballot, nominee match-up only, duplicates across sources removed
  (likely voters preferred), a poll >25 pts off the median of 4+ polls of the same race dropped. Every drop is
  counted by reason in `data/pipeline_log.json`. A source that fails is skipped; the bundle's polls stay.
- **Approval**: only used if within 5 pts of the export's value.
- **Live counts**: county must exist in the model; votes <= 1.6x plausible midterm turnout; not one-sided past
  1,000 votes; counts never fall by more than 2% (bigger drops: county ignored that run); state totals plausible.
  If under 60% of reporting counties pass, the live update is skipped and the last good `live.json` stays.
- **Output**: every number in range, enough races present.
- **Jump check** (pre-election): if a state moves >10 pts or a chamber's P(majority) moves >0.40 since the last
  published forecast, the update is held. Re-run by hand with *force* if it's real.

A held or failed run publishes nothing new, goes red in the Actions tab (GitHub emails you), and the reason is the
last lines of the run's log and `data/pipeline_log.json`.

## Set-up (once)

1. **Colab**: in the v7 runtime (after the driver cell and the sweep cell), run `colab_export_bundle_cell.py`.
   It refreshes money/approval/jobs/specials, trains the final model (all 7 elections), and writes
   `Electionformer/github_bundle/bundle.pkl.gz` and `model.pt` on your Drive. Download both.
2. **GitHub**, in the website repo:
   - upload `bundle.pkl.gz` and `model.pt` into `pipeline/bundle/` (Add file -> Upload files; both are well under 25 MB)
   - make sure these are in the repo: `.github/workflows/forecast.yml`, everything in `pipeline/`
   - Settings -> Actions -> General -> Workflow permissions -> **Read and write permissions** -> Save
3. **Test**: Actions tab -> *forecast* -> Run workflow -> mode `selftest`. Takes ~5-10 min. The log ends with a JSON
   report; `verdict` should be `OK`. It checks: the model gives Colab's numbers to 3 decimals, Bayes matches,
   the featuriser matches, VoteHub and Wikipedia answer, the NYT parser reads the 2024 feed, a full forecast runs, timing.
4. **First real run**: Run workflow -> mode `pre`. After it goes green the site shows real numbers (demo banner gone).
   From then on the schedule runs it.

## On the night

- `config.py -> LIVE_MODEL`: `"blend"` (default), `"bayes"` (polling model + counted votes only), `"off"`.
  Edit it on GitHub if the live numbers look wrong; the next run uses it.
- Live mode starts 22:30 UTC on Nov 3 and stops 17:00 UTC on Nov 5 (`config.py`).
- Model clock: hours since midnight ET on election day (what the synthetic training used).

## Files

```
pipeline/
  run.py            entry point: modes auto / pre / live / selftest
  config.py         dates, URLs, thresholds, switches
  polls.py          VoteHub + Wikipedia scrapers, merge, checks, poll -> model observations
  early_vote.py     NC early vote from the NCSBE file
  live.py           NYT feeds -> county count batches + race progress/calls, checks
  forecast.py       strict KERMIT + Bayes + blend, seat simulations, drivers, JSON outputs
  kermit_core.py    model / featuriser / Bayes, copied verbatim from the v7 notebook (don't edit)
  bundle/           bundle.pkl.gz + model.pt from Colab
  state/            election-night count history (written by the workflow)
```
