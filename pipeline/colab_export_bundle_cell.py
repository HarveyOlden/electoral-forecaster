# ============================== KERMIT 2026: refresh inputs, train the final model, export the GitHub bundle ==============================
# Run in the v7 runtime after the driver (c12) and the sweep cell (it uses kermit_loss_clean). If the runtime restarted:
# re-run c01-c12 (everything finished loads from Drive) + the sweep cell, then this.
# Writes Electionformer/github_bundle/{bundle.pkl.gz, model.pt}: download both and upload them to pipeline/bundle/ on GitHub.
#
# 1. refreshes the slow-moving inputs the model already knows how to use: latest jobs/inflation (FRED), Trump approval
#    (VoteHub), 2025-26 House special elections (Wikipedia), pre-general campaign money (FEC 12G reports, due Oct 22)
# 2. fine-tunes ONE final model on all 7 elections (clean district labels, district weight 0.2; steps = median best step
#    of the 7 cross-validation folds, since there is no held-out year left to early-stop on)
# 3. builds the 2026 election (forecast mode) and saves everything GitHub needs to rebuild it with fresh polls/counts,
#    plus a reference forward pass so the GitHub self-test can prove its copy of the code gives the same numbers.
import os, json, gzip, pickle, io, datetime as dt
import numpy as np, pandas as pd, torch, requests

OUT = f"{ROOT}/github_bundle"
os.makedirs(OUT, exist_ok=True)
REFRESH = dict(fred=True, approval=True, specials=True, money=True)
FINAL_W = 0.2
S_ = MODEL["name"]
UA = {"User-Agent": "KERMIT-election-research/1.0"}


def _save_json(name, df):
    df.to_json(f"{DATA}/{name}.json", orient="records", date_format="iso")


# ---------------- 1. refresh ----------------
if REFRESH["fred"]:
    try:
        old = jload("fred_macro_national", cache=False)
        new = []
        for sid in ("UNRATE", "CPIAUCSL"):
            d = pd.read_csv(io.StringIO(requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                                                     headers=UA, timeout=60).text))
            dc_ = next(c for c in d.columns if c.lower() in ("date", "observation_date"))
            new.append(pd.DataFrame(dict(series=sid, date=pd.to_datetime(d[dc_]), value=pd.to_numeric(d[sid], errors="coerce"))))
        new = pd.concat(new).dropna()
        keep = old[~old.series.isin(["UNRATE", "CPIAUCSL"])]
        _save_json("fred_macro_national", pd.concat([keep, new], ignore_index=True))
        print("FRED refreshed: latest", new.groupby("series").date.max().dt.date.to_dict())
    except Exception as e:
        print("FRED refresh failed (kept old):", e)

if REFRESH["approval"]:
    try:
        js = requests.get("https://api.votehub.com/polls", params=dict(poll_type="approval", subject="donald-trump"),
                          headers=UA, timeout=120).json()
        rows = []
        for p in (js if isinstance(js, list) else js.get("polls", [])):
            a = {str(x.get("choice", "")).lower(): x.get("pct") for x in p.get("answers") or []}
            if a.get("approve") is not None:
                rows.append(dict(end_date=pd.to_datetime(p.get("end_date")), approving=float(a["approve"]),
                                 pollster=p.get("pollster"), source="votehub"))
        new = pd.DataFrame(rows).dropna(subset=["end_date"])
        new = new[new.approving.between(20, 70)]
        old = jload("presidential_approval", cache=False)
        old["end_date"] = pd.to_datetime(old.end_date, errors="coerce")
        both = pd.concat([old, new], ignore_index=True).drop_duplicates(["end_date", "approving", "pollster"])
        _save_json("presidential_approval", both)
        rec = new[new.end_date >= "2026-08-01"].approving
        print(f"approval refreshed: {len(new)} VoteHub polls, Aug 1+ mean {rec.mean():.1f}% ({len(rec)} polls)")
    except Exception as e:
        print("approval refresh failed (kept old):", e)

if REFRESH["specials"]:
    try:
        if "pull_specials" in globals():
            p_ = f"{DATA}/house_specials.json"
            bak = p_ + ".bak"
            if os.path.exists(p_):
                os.replace(p_, bak)
            try:
                pull_specials()
                print("specials refreshed")
            except Exception:
                if os.path.exists(bak):
                    os.replace(bak, p_)
                raise
        else:
            print("specials: pull_specials() not in this runtime (run c03 first to refresh) - kept old")
    except Exception as e:
        print("specials refresh failed (kept old):", e)

if REFRESH["money"]:
    try:
        p_ = f"{DATA}/fec_pregeneral.json"
        if os.path.exists(p_):
            m_ = pd.read_json(p_)
            m_[m_.cycle != 2026].to_json(p_, orient="records")         # force a re-pull of 2026 (12G reports now filed)
        MONEY = pull_pregeneral()
        nom = (_wb.sort_values(["won_primary", "ttl_receipts"], ascending=False)
               .drop_duplicates(["cycle", "office", "cand_office_st", "district", "party"]))
        nom = nom.merge(_cn, on=["cand_id", "cycle"], how="left").merge(
            MONEY.rename(columns={"committee_id": "cand_pcc"})[["cand_pcc", "cycle", "receipts", "coh"]], on=["cand_pcc", "cycle"], how="left")
        RM = nom.pivot_table(index=["cycle", "office", "cand_office_st", "district"], columns="party", values=["receipts", "coh"], aggfunc="first")
        RM.columns = [f"{a}_{b}" for a, b in RM.columns]
        RM = RM.reset_index().rename(columns={"cand_office_st": "state"})
        for c in ["receipts_D", "receipts_R", "coh_D", "coh_R"]:
            if c not in RM:
                RM[c] = np.nan
        RM["known"] = RM[["receipts_D", "receipts_R"]].notna().any(axis=1)
        lr_ = lambda a, b: np.clip(np.log1p(a.fillna(0).clip(lower=0)) - np.log1p(b.fillna(0).clip(lower=0)), -9, 9) / 3
        RM["money_ratio"] = np.where(RM.known, lr_(RM.receipts_D, RM.receipts_R), 0.0)
        RM["coh_ratio"] = np.where(RM.known, lr_(RM.coh_D, RM.coh_R), 0.0)
        RM["has_money"] = RM.known.astype(float)
        RACE_MONEY = RM
        print("money refreshed: 2026 races with money known:", RM[(RM.cycle == 2026) & RM.known].groupby("office").size().to_dict())
    except Exception as e:
        print("money refresh failed (kept old):", e)

_JCACHE.clear()
_POLLS = None                      # c06 re-reads polls_tidy

# ---------------- 2. the final model ----------------
steps = []
for y in FOLDS:
    mp = f"{RUN_DIR}/{S_}_ftclean_w{FINAL_W}_{y}/meta.json"
    if os.path.exists(mp):
        steps.append(json.load(open(mp))["best_step"])
N_FINAL = int(max(40, np.median(steps))) if steps else 120
print(f"final fine-tune: {N_FINAL} steps (fold best steps {steps})")
pre = make_model()
pre = train_run(f"{S_}_pretrain", pre, synth_sample, MODEL["max_steps"], MODEL["lr"], WARMUP, rel_of,
                val_fn=synth_val, val_every=VAL_EVERY, patience=MODEL["patience"], min_steps=MODEL["min_steps"],
                accum=ACCUM, wd=MODEL["wd"], seed=SEED)                      # finished -> loads
_orig_loss, _orig_w = kermit_loss, DIST_LOSS_W
try:
    kermit_loss, DIST_LOSS_W = kermit_loss_clean, FINAL_W
    tr_all = [i for i in REAL if not (i["year"] in NIGHT_TEST_ONLY and i["kind"] == "night")]
    final = make_model()
    final.load_state_dict(pre.state_dict())
    final = train_run(f"{S_}_final_w{FINAL_W}", final, real_sampler(tr_all), N_FINAL, FT_LR, FT_WARMUP, rel_of,
                      freeze=FT_FREEZE, val_fn=None, min_steps=0, accum=ACCUM, wd=MODEL["wd"], seed=SEED)
finally:
    kermit_loss, DIST_LOSS_W = _orig_loss, _orig_w
final = final.cpu()
torch.save(final.state_dict(), f"{OUT}/model.pt")
final = final.to(DEV)

# ---------------- 3. the 2026 election ----------------
insts26, w26, info26 = build_real_election(2026, [-24.0], [], forecast=True)
i26 = insts26[0]
print("2026 election:", {k: v for k, v in info26.items() if not str(k).startswith("lab_")})
fund26 = real_fundamentals(2026)
eday = fund26["election_date"]
ev_obs = nc_early_vote_obs(w26, 2026, eday)
spec_obs = specials_obs(w26, 2026, eday)
poll_obs26 = poll_obs(w26, 2026)
up26 = races_up_schedule(w26, 2026)

# North Carolina early vote: what GitHub needs to recompute it daily from the NCSBE file
pres_ = jload("county_president_2000_2024_wide")
nc_ = pres_[pres_.state_po == "NC"]
nc_name2fips = dict(zip(nc_.county_name.astype(str).str.upper().str.strip(), fips5(nc_.county_fips)))
nc_prev = None
cal_ = jload("election_calendar_context").set_index("year")
if have("nc_early_mail_vote_daily"):
    for yp in [2022, 2018, 2024, 2020]:
        if yp in cal_.index:
            nc_prev = nc_daily(pd.Timestamp(cal_.loc[yp].election_date), nc_name2fips)
            if nc_prev is not None:
                nc_prev["lean_prev"] = logit((nc_prev.D + 0.5) / (nc_prev.D + nc_prev.R + 1))
                nc_prev = nc_prev[["fips", "days_out", "lean_prev"]]
                print(f"NC early-vote baseline: {yp}")
                break

# House polls -> county pieces (same as poll_obs)
xw = crosswalk_for(2026)
f2u = {f: k for k, f in enumerate(w26.county_fips)}
x = xw[xw.county_fips.isin(f2u)].copy()
x["u"] = x.county_fips.map(f2u)
nd_ = x.groupby("state_po").district.transform("nunique")
x.loc[nd_ == 1, "district"] = 0
x["pr"] = sig(w26.prior_logit[x.u.values]) * w26.turnout_exp[x.u.values] * x.share_cd.values
x["tv"] = w26.turnout_exp[x.u.values] * x.share_cd.values
dprior = x.groupby(["state_po", "district"]).apply(lambda g: logit(g.pr.sum() / g.tv.sum()))
house_xw = x[["state_po", "district", "u", "share_dc"]].reset_index(drop=True)

# geography for the attention relations
fips = np.asarray(w26.county_fips).astype(str)
g_ = GAZ.reindex(fips)
fset = set(fips)
adj = [(a, b) for a, b in ADJ if a in fset and b in fset]

# what is known about each race (for the site)
so = np.asarray(w26.state_of)
RS = i26["race_static"]
inc_state = {}
for ri, race in enumerate(RACES):
    for s, st in enumerate(w26.states):
        m = (so == s) & (RS[:, ri, 0] > 0.5)
        if m.any() and RS[m, ri, 5].max() > 0.5:
            v = float(np.average(RS[m, ri, 2], weights=np.exp(i26["static"][m, 1])))
            inc_state[(race, st)] = "D" if v > 0.3 else "R" if v < -0.3 else None
P26 = jload("polls_tidy", cache=False)
P26 = P26[P26.year == 2026].copy()
names = {}
for (race, st), g in P26[P26.race.isin(["sen", "gov"])].groupby(["race", "state"]):
    late = g.sort_values("end_date").tail(30)
    if len(late) and late.dem_name.notna().any() and late.rep_name.notna().any():
        names[(race, st)] = dict(D=str(late.dem_name.mode().iloc[0]), R=str(late.rep_name.mode().iloc[0]))

# reference forward pass (GitHub's self-test must reproduce these numbers from the same inputs)
rel26 = REL[2026] if 2026 in REL else build_relations(w26, GAZ, ADJ, w26.dmap)
with torch.no_grad():
    ref = predict(final, to_dev(i26, "nation_nodemo"), rel26)
mb, Lb, db = bayes_predict(i26, w26.state_of, w26.n_states, FUND_MODEL.fits[0], fit_race_coefs([i for i in REAL if i["kind"] == "pre"]))

wd = {k: v for k, v in w26.__dict__.items() if k not in ("lag_swings", "dmaps")}
bundle = dict(
    version=f"KERMIT v7 (clean labels, district weight {FINAL_W}, strict polls, blended with Bayes)",
    exported_at=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    election_date=str(pd.Timestamp(eday).date()),
    model_cfg=dict(MODEL, n_static=len(STATIC_COLS), n_latent=N_LATENT, n_factors=N_FACTORS, final_steps=N_FINAL),
    world=wd,
    inst=dict(static=i26["static"], race_static=i26["race_static"], dist_static=i26["dist_static"], context=i26["context"]),
    obs_static=pd.concat([ev_obs, spec_obs], ignore_index=True),     # replaced by fresh pulls where GitHub can get them
    ev_obs=ev_obs, spec_obs=spec_obs,
    polls_2026=P26, poll_obs_at_export=poll_obs26,
    nc=dict(name2fips=nc_name2fips, prev=nc_prev),
    house_xw=house_xw, dprior={f"{a}|{b}": float(v) for (a, b), v in dprior.items()},
    state_prior=state_prior(w26), national_prior=national_prior(w26),
    geo=dict(lat=g_.lat.values.astype(float), lon=g_.lon.values.astype(float), adj=adj),
    up=up26, races=list(RACES), inc_state={f"{r}|{s}": v for (r, s), v in inc_state.items()}, names={f"{r}|{s}": v for (r, s), v in names.items()},
    fund=dict(approval=float(fund26["approval"]), unemp_change=float(fund26["unemp_change"]), inflation=float(fund26["inflation"])),
    fund_neutral=dict(unemp_change=float(FUND_MODEL.fits[0]["mu"][1]), inflation=float(FUND_MODEL.fits[0]["mu"][2]),
                      approval=float(FUND_MODEL.fits[0]["mu"][3])),
    bayes=dict(fit=FUND_MODEL.fits[0], coefs=fit_race_coefs([i for i in REAL if i["kind"] == "pre"])),
    cal=CAL,
    reference=dict(county=i26["county"].astype(np.float16), state=i26["state"].astype(np.float16),
                   nation=i26["nation_nodemo"].astype(np.float16), mu=ref["mu"], mu_d=ref["mu_d"], bayes_mu=mb),
)
with gzip.open(f"{OUT}/bundle.pkl.gz", "wb") as fh:
    pickle.dump(bundle, fh, protocol=4)
for f in ("bundle.pkl.gz", "model.pt"):
    print(f"{f}: {os.path.getsize(f'{OUT}/{f}') / 1e6:.1f} MB")
print(f"2026 polls in the bundle: {len(P26)} ({P26.groupby('race').size().to_dict()}); NC early-vote reads: {len(ev_obs)}; specials: {len(spec_obs)}")
print(f"done -> download both files from Drive: {OUT}/ and upload them to pipeline/bundle/ in the GitHub repo")
