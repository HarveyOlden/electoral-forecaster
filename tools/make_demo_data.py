"""Writes DEMO data files in the exact format the site reads (see DATA_FORMAT.md). Every number here is random:
the pipeline replaces these files with real model output. Run from the site folder: python tools/make_demo_data.py"""
import json, math, os, csv, datetime as dt
import numpy as np

rng = np.random.default_rng(7)
LAY = json.load(open("data/house_hex_layout.json"))
NAME = {s["postal"]: s["name"] for s in LAY["states"]}
NOW = dt.datetime(2026, 10, 28, 22, 0, tzinfo=dt.timezone.utc)
iso = lambda t: t.isoformat().replace("+00:00", "Z")
sig = lambda x: 1 / (1 + np.exp(-x))

# rough partisan lean of each state (logit, + = Dem) - demo only
LEAN = dict(AL=-.9, AK=-.4, AZ=-.05, AR=-.9, CA=.6, CO=.25, CT=.45, DE=.4, FL=-.3, GA=-.05, HI=.8, ID=-1.0, IL=.4,
            IN=-.55, IA=-.3, KS=-.5, KY=-.8, LA=-.6, ME=.2, MD=.75, MA=.8, MI=0.02, MN=.12, MS=-.55, MO=-.5, MT=-.6,
            NE=-.6, NV=-.02, NH=.1, NJ=.25, NM=.2, NY=.45, NC=-.06, ND=-1.1, OH=-.3, OK=-1.1, OR=.3, PA=-.02, RI=.5,
            SC=-.55, SD=-.9, TN=-.85, TX=-.3, UT=-.75, VT=.9, VA=.15, WA=.45, WV=-1.3, WI=0.0, WY=-1.5)
ENV = 0.08                                                     # demo national environment (Dem-leaning midterm)
SEN_UP = dict(AL="R", AK="R", AR="R", CO="D", DE="D", GA="D", ID="R", IL="D", IA="R", KS="R", KY="R", LA="R", ME="R",
              MA="D", MI="D", MN="D", MS="R", MT="R", NE="R", NH="D", NJ="D", NM="D", NC="R", OK="R", OR="D", RI="D",
              SC="R", SD="R", TN="R", TX="R", VA="D", WV="R", WY="R", OH="R", FL="R")
GOV_UP = dict(AL="R", AK="R", AZ="D", AR="R", CA="D", CO="D", CT="D", FL="R", GA="R", HI="D", ID="R", IL="D", IA="R",
              KS="D", ME="D", MD="D", MA="D", MI="D", MN="D", NE="R", NV="R", NH="R", NM="D", NY="D", OH="R", OK="R",
              OR="D", PA="D", RI="D", SC="R", SD="R", TN="R", TX="R", VT="R", WI="D", WY="R")
DRIVER_POOL = ["{st} polls for this race", "National generic ballot", "Presidential approval (34%)", "Incumbency",
               "Past results of this race", "Campaign money (pre-general reports)", "House special elections this cycle",
               "Economy: inflation and jobs", "Demographic shift since 2024", "Neighbouring states' polls",
               "Early vote in {st}", "Presidential results trend"]
DETAIL = {"{st} polls for this race": "{n} polls, latest {d}", "National generic ballot": "{n} polls, average D+{m}",
          "Incumbency": "{inc} incumbent running", "Past results of this race": "Last two results on these lines",
          "Campaign money (pre-general reports)": "Dem raised {x}x the Republican", "House special elections this cycle":
          "{n} specials, Dems ran {m} pts ahead on average", "Early vote in {st}": "Returned ballots so far"}


def drivers(st, p):
    picks = rng.choice(len(DRIVER_POOL), 5, replace=False)
    out = []
    for k in picks:
        lab = DRIVER_POOL[k]
        det = DETAIL.get(lab, "")
        out.append(dict(label=lab.format(st=NAME.get(st, st)),
                        detail=det.format(st=NAME.get(st, st), n=int(rng.integers(3, 20)), d="Oct 26",
                                          m=round(float(rng.uniform(1, 6)), 1), inc=rng.choice(["Democratic", "Republican"]),
                                          x=round(float(rng.uniform(1.1, 4)), 1)),
                        shift_pp=round(float(rng.normal(0.6 * np.sign(p - 0.5), 1.4)), 2)))
    return sorted(out, key=lambda d: -abs(d["shift_pp"]))


def race(id_, st, lean, sd, extra=None):
    mu = lean + ENV + rng.normal(0, 0.06)
    p_win = float(1 - 0.5 * (1 + math.erf((0 - mu) / (sd * math.sqrt(2)))))
    r = dict(id=id_, state=st, name=NAME.get(st, st), dem_share=round(float(sig(mu)), 4),
             lo90=round(float(sig(mu - 1.645 * sd)), 4), hi90=round(float(sig(mu + 1.645 * sd)), 4),
             p_dem_win=round(p_win, 4), drivers=drivers(st, p_win))
    r.update(extra or {})
    return r, mu


def chamber_summary(mus, sds, base_d, total, need, tie_goes_to=None, n=4000):
    common = rng.normal(0, 0.05, (n, 1))
    draws = (np.array(mus)[None] + common + rng.normal(0, 1, (n, len(mus))) * np.array(sds)[None]) > 0
    seats = base_d + draws.sum(1)
    hist = np.bincount(seats, minlength=total + 1)
    lo, hi = np.percentile(seats, [10, 90])
    p_maj = float((seats >= need).mean())
    return dict(seats_total=total, majority_at=need, dem_seats_mean=round(float(seats.mean()), 1), dem_seats_lo80=int(lo),
                dem_seats_hi80=int(hi), p_dem_majority=round(p_maj, 4),
                seat_distribution=[dict(seats=int(k), p=round(float(v / n), 5)) for k, v in enumerate(hist) if v > 0])


races = {"senate": [], "house": [], "governor": []}
mus = {k: [] for k in races}
sds = {k: [] for k in races}
for st, inc in SEN_UP.items():
    sd = 0.16
    r, mu = race(st, st, LEAN[st] + 0.13 + (0.05 if inc == "D" else -0.05), sd,
                 dict(incumbent_party=inc, candidates=dict(D=f"Democratic nominee ({st})", R=f"Republican nominee ({st})")))
    races["senate"].append(r); mus["senate"].append(mu); sds["senate"].append(sd)
for st, inc in GOV_UP.items():
    sd = 0.18
    r, mu = race(st, st, LEAN[st] * 0.8 + (0.04 if inc == "D" else -0.04), sd, dict(incumbent_party=inc))
    races["governor"].append(r); mus["governor"].append(mu); sds["governor"].append(sd)
for h in LAY["hexes"]:
    lean = LEAN[h["state"]] + rng.normal(0, 0.7)
    sd = 0.12
    r, mu = race(h["id"], h["state"], lean, sd, dict(district=h["district"],
                 name=f"{NAME[h['state']]} {'at-large' if h['district'] == 0 else h['district']}"))
    if abs(lean) > 1.3 and rng.random() < 0.35:                 # some seats have only one major party on the ballot
        r.update(uncontested=True, p_dem_win=1.0 if lean > 0 else 0.0, dem_share=None, lo90=None, hi90=None, drivers=[])
    races["house"].append(r); mus["house"].append(mu if not r.get("uncontested") else 9 * np.sign(lean)); sds["house"].append(sd)

forecast = dict(
    demo=True, model_version="KERMIT v7 (demo data)", generated_at=iso(NOW), data_cutoff=iso(NOW - dt.timedelta(hours=2)),
    election_date="2026-11-03",
    chambers=dict(
        senate=dict(chamber_summary(mus["senate"], sds["senate"], 34, 100, 51), not_up=dict(D=34, R=31),
                    note="Democrats need 51 seats: a 50-50 Senate is controlled by the Republican vice president."),
        house=dict(chamber_summary(mus["house"], sds["house"], 0, 435, 218), note="218 seats for a majority."),
        governor=dict(races_up=len(races["governor"]), dem_wins_mean=round(float(sum(r["p_dem_win"] for r in races["governor"])), 1),
                      note="Governors' mansions won in 2026 (36 races).")),
    races=races)

# live: same shape + counting progress (demo: mid-count snapshot)
live = json.loads(json.dumps(forecast))
live.update(is_live=True, updated_at=iso(NOW), note="Demo snapshot: not real results.")
for ch, lst in live["races"].items():
    for r in lst:
        rep = float(np.clip(rng.uniform(-0.3, 1.1), 0, 1))
        r["reported_pct"] = round(rep * 100, 1)
        if r.get("uncontested"):
            r["status"] = "called"
        elif rep > 0.85 and abs(r["p_dem_win"] - 0.5) > 0.4:
            r["status"] = "called"
        else:
            r["status"] = "counting" if rep > 0 else "not_started"
        if r["status"] == "called":
            r["called_for"] = "D" if r["p_dem_win"] > 0.5 else "R"
for ch in ("senate", "house"):
    called = [r for r in live["races"][ch] if r.get("status") == "called"]
    live["chambers"][ch]["called"] = dict(D=sum(r["called_for"] == "D" for r in called), R=sum(r["called_for"] == "R" for r in called))

# backtest: held-out years, predicted vs actual
years = []
for y in (2012, 2014, 2016, 2018, 2020, 2022, 2024):
    kinds = {"senate": list(SEN_UP), "governor": list(GOV_UP)[: 12 if y % 4 == 0 else 34]}
    if y % 4 == 0:
        kinds["president"] = [s for s in LEAN]
    yr = dict(year=y, type="presidential" if y % 4 == 0 else "midterm", races={})
    for ch, sts in kinds.items():
        lst = []
        for st in sts[: 34 if ch != "president" else 51]:
            pred = float(sig(LEAN[st] + rng.normal(0, 0.2)))
            act = float(np.clip(pred + rng.normal(0, 0.025), 0.05, 0.95))
            lst.append(dict(id=st, name=NAME[st], pred_dem_share=round(pred, 4), actual_dem_share=round(act, 4),
                            lo90=round(pred - 0.045, 4), hi90=round(pred + 0.045, 4)))
        yr["races"][ch] = lst
    act_h = int(rng.integers(188, 240))
    pred_h = act_h + int(rng.normal(0, 6))
    yr["house_seats"] = dict(actual_dem=act_h, pred_dem=pred_h, lo90=pred_h - 14, hi90=pred_h + 14)
    years.append(yr)
backtest = dict(demo=True, note="Each year was predicted by a model trained without that year (night-before forecast).",
                years=years)

meta = dict(demo=True, generated_at=iso(NOW), live_available=True, files=["forecast.json", "live.json", "backtest.json"])
for name, obj in (("forecast.json", forecast), ("live.json", live), ("backtest.json", backtest), ("meta.json", meta)):
    json.dump(obj, open(f"data/{name}", "w"), separators=(",", ":"))
    print(name, f"{os.path.getsize(f'data/{name}') / 1e3:.0f} KB")
