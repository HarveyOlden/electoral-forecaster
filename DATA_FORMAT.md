# Data format (what the model pipeline writes into `data/`)

The site never runs the model. It only reads these files, so the pipeline's job is to overwrite them.
`tools/make_demo_data.py` writes valid examples of every file: copy its structure.

| File | Written | Read by |
|---|---|---|
| `meta.json` | every update (write it **last**) | every page: spots new output, shows the demo banner, lights the Live dot |
| `forecast.json` | every pre-election update | Forecast tab, ticker |
| `live.json` | every election-night update | Live results tab |
| `backtest.json` | once (after the cross-validation run) | Track record tab |
| `sources.csv` | when sources change | Data sources tab |
| `house_hex_layout.json`, `geo/` | never (static) | maps |

## meta.json
```json
{ "demo": false, "generated_at": "2026-11-01T12:00:00Z", "live_available": false }
```
`generated_at` changing is what makes open pages refresh. `live_available: true` makes the Live dot pulse.

## forecast.json
```json
{
  "demo": false,
  "model_version": "KERMIT v7",
  "generated_at": "2026-11-01T12:00:00Z",
  "data_cutoff": "2026-11-01T10:00:00Z",
  "election_date": "2026-11-03",
  "chambers": {
    "senate": {
      "seats_total": 100, "majority_at": 51,
      "dem_seats_mean": 48.3, "dem_seats_lo80": 46, "dem_seats_hi80": 51,
      "p_dem_majority": 0.27,
      "seat_distribution": [{ "seats": 46, "p": 0.08 }, { "seats": 47, "p": 0.15 }],
      "not_up": { "D": 34, "R": 31 },
      "note": "..."
    },
    "house":    { "same fields as senate, majority_at": 218 },
    "governor": { "races_up": 36, "dem_wins_mean": 17.4, "note": "..." }
  },
  "races": {
    "senate":   [ RACE, ... ],
    "house":    [ RACE, ... ],
    "governor": [ RACE, ... ]
  }
}
```
Seat counts include seats not up this year (Senate). `seat_distribution` = share of the model's joint draws
landing on each Democratic seat count.

### RACE
```json
{
  "id": "PA",                    // Senate/governor: state postal code. House: "PA-07", at-large "WY-AL"
  "state": "PA", "name": "Pennsylvania",       // House name e.g. "Pennsylvania 7"
  "district": 7,                 // House only (0 = at-large)
  "p_dem_win": 0.62,             // 0-1
  "dem_share": 0.512,            // expected Dem share of the two-party vote
  "lo90": 0.478, "hi90": 0.546,  // 90% range of dem_share
  "uncontested": false,          // true: one major party only (dem_share/lo90/hi90 may be null, p_dem_win 0 or 1)
  "incumbent_party": "D",        // optional
  "drivers": [                   // top 5 by |shift_pp|: run the model without each input group
    { "label": "Pennsylvania Senate polls", "detail": "14 polls, latest Oct 30", "shift_pp": 2.1 }
  ]
}
```
`shift_pp` = (forecast with everything) − (forecast without that input), in points of Dem share.
Positive = that input helped the Democrat.

## live.json
Same as `forecast.json`, plus:
```json
{ "is_live": true, "updated_at": "2026-11-04T02:30:00Z",
  "chambers": { "senate": { "called": { "D": 12, "R": 15 } }, "house": { "called": { "D": 120, "R": 133 } } } }
```
and on each RACE: `"reported_pct": 64.0`, `"status": "not_started" | "counting" | "called"`, `"called_for": "D" | "R"`.
Before election night: `{"is_live": false}` is enough (the tab shows a "starts on election night" message).

## backtest.json
```json
{ "years": [ { "year": 2024, "type": "presidential",
    "races": { "senate": [ { "id": "PA", "name": "Pennsylvania", "pred_dem_share": 0.52, "actual_dem_share": 0.50,
                              "lo90": 0.48, "hi90": 0.56 } ],
               "governor": [ ... ], "president": [ ... ] },
    "house_seats": { "actual_dem": 215, "pred_dem": 208, "lo90": 186, "hi90": 223 } } ] }
```
Only held-out (leave-one-year-out) night-before forecasts belong here.

## sources.csv
Columns: `category,dataset,provider,link,used_for`.
