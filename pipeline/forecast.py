"""The 2026 forecast from a bundle + observations: KERMIT in strict mode (each race reads only its own state polls),
Bayes, the fixed blend, joint simulations for seats, and the top-5 drivers of every race by ablation
(hide one input group, rerun everything, measure how far the race moved)."""
import numpy as np
import pandas as pd
import torch

import config as C
from kermit_core import (World, Bugs, Kermit, featurise, build_relations, to_dev, predict, bayes_predict,
                         combine_fixed, county_to_district, kermit_district_draws, blend_districts, sig,
                         RACES, R_, RACE_STATIC_COLS, DIST_STATIC_COLS, DS_IDX, STATIC_COLS, CONTEXT_COLS, CAL)
import kermit_core

RS_IDX = {c: k for k, c in enumerate(RACE_STATIC_COLS)}
GROUP_COLS = {"incumbency": ["inc", "has_inc"],
              "past": ["last_resid", "has_last", "last_unc", "last2_resid", "has_last2"],
              "money": ["money_ratio", "coh_ratio", "has_money"]}


FEC_WEBALL = "https://www.fec.gov/files/bulk-downloads/2026/weball26.zip"   # FEC "all candidates" summary (no key)
WEBALL_COLS = ["cand_id", "cand_name", "cand_ici", "pty_cd", "party", "ttl_receipts", "trans_from_auth", "ttl_disb",
               "trans_to_auth", "coh_bop", "coh_cop", "cand_contrib", "cand_loans", "other_loans", "cand_loan_repay",
               "other_loan_repay", "debts_owed_by", "ttl_indiv_contrib", "state", "district", "spec_election",
               "prim_election", "run_election", "gen_election", "gen_election_pct", "other_pol_cmte_contrib",
               "pol_pty_contrib", "cvg_end_dt", "indiv_refunds", "cmte_refunds"]
MONEY_TOL = 0.55          # money_ratio units (log ratio / 3): 0.55 = the two sources disagree by more than ~5x
MONEY_MAX_DROP = 0.25     # removing money from more than 25% of the races that have it = the check itself is suspect
MONEY_MIN_CORR = 0.6      # the bundle's ratios and the FEC summary's must broadly agree, or the mapping is broken


def fec_summary_ratios(log=None):
    """Independent second source: FEC's own candidate totals. Nominee per party = primary winner if known, else the top
    fundraiser (same rule as the notebook). Returns {("S", state, 0) | ("H", state, district): money_ratio}."""
    import io, zipfile, requests
    r = requests.get(FEC_WEBALL, headers={"User-Agent": C.USER_AGENT}, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    d = pd.read_csv(z.open(z.namelist()[0]), sep="|", header=None, names=WEBALL_COLS, dtype=str, encoding="latin-1")
    d["office"] = d.cand_id.str[0]
    d = d[d.office.isin(["H", "S"])].copy()
    d["p"] = d.party.str.upper().map({"DEM": "D", "DFL": "D", "REP": "R"})
    d = d.dropna(subset=["p"])
    d["district"] = np.where(d.office == "S", 0, pd.to_numeric(d.district, errors="coerce").fillna(0)).astype(int)
    d["ttl"] = pd.to_numeric(d.ttl_receipts, errors="coerce").fillna(0)
    d["won"] = d.prim_election.fillna("").str.upper().eq("W")
    d = d[d.spec_election.fillna("").str.strip().eq("")]
    d = d.sort_values(["won", "ttl"], ascending=False)
    # only trust the nominee pick when it is unambiguous: FEC flagged a primary winner, or the top fundraiser raised
    # 3x+ the party's runner-up. Otherwise (runoffs, unflagged primaries: e.g. Texas R) FEC may compare the wrong pair.
    g = d.groupby(["office", "state", "district", "p"])
    second = g.ttl.transform(lambda s: s.iloc[1] if len(s) > 1 else 0.0)
    d["sure"] = d.won | (d.ttl >= 3 * second)
    nom = d.drop_duplicates(["office", "state", "district", "p"])
    nom = nom[nom.sure]
    piv = nom.pivot_table(index=["office", "state", "district"], columns="p", values="ttl", aggfunc="first")
    piv = piv.dropna(subset=[c for c in ("D", "R") if c in piv]) if {"D", "R"} <= set(piv.columns) else piv.iloc[:0]
    out = {}
    for (o, st, dn), row in piv.iterrows():
        out[(o, st, int(dn))] = float(np.clip(np.log1p(max(row.D, 0)) - np.log1p(max(row.R, 0)), -9, 9) / 3)
    return out


def clean_money(bundle, fetch=fec_summary_ratios):
    """Money inputs that are clearly broken are set to "unknown" (has_money 0, ratios 0), which the model saw plenty
    of in training. Every race with money is compared with FEC's own candidate totals (an independent second source);
    where they disagree by more than ~5x, the race's money is cleared. This is only applied if the check passes its
    own sanity tests (enough races compared, the two sources broadly agree, and it would not remove money from an
    implausibly large share of races). A ratio at the clip ("raised 8,103x") is NOT cleared on its own: most are real
    (a safe seat whose challenger raised ~nothing, exactly as in training); the FEC check separates real from unmatched.
    Only if FEC is unreachable are capped ratios cleared instead, under the same share guard. Returns a report."""
    rs = np.array(bundle["inst"]["race_static"], copy=True)
    ds = np.array(bundle["inst"]["dist_static"], copy=True)
    w = bundle["world"]
    states, so = list(w["states"]), np.asarray(w["state_of"])
    i_mr, i_cr, i_hm = RS_IDX["money_ratio"], RS_IDX["coh_ratio"], RS_IDX["has_money"]
    j_mr, j_cr, j_hm = DS_IDX["money_ratio"], DS_IDX["coh_ratio"], DS_IDX["has_money"]
    rep = dict(capped=0, checked=0, disagree=[], applied=False, reason="")

    def zero_rs(mask_c, ri):
        rs[mask_c, ri, i_mr] = 0; rs[mask_c, ri, i_cr] = 0; rs[mask_c, ri, i_hm] = 0

    def zero_ds(k):
        ds[k, j_mr] = 0; ds[k, j_cr] = 0; ds[k, j_hm] = 0

    # capped ratios: recorded here, only cleared directly if FEC is unreachable (below)
    cap = (np.abs(rs[..., i_mr]) >= 2.99) | (np.abs(rs[..., i_cr]) >= 2.99)
    capd = (np.abs(ds[:, j_mr]) >= 2.99) | (np.abs(ds[:, j_cr]) >= 2.99) if ds.size else np.zeros(0, bool)
    rep["capped"] = int(len({(states[so[c]], r) for c, r in zip(*np.where(cap))}) + capd.sum())
    dk = [tuple(k) for k in w["dmap"]["keys"]] if w.get("dmap") is not None else []
    rep["capped_races"] = sorted({f"{RACES[r]} {states[so[c]]}" for c, r in zip(*np.where(cap)) if RACES[r] != "house"}
                                 | {f"house {dk[k][0]}-{int(dk[k][1])}" for k in np.where(capd)[0] if k < len(dk)})

    # rule 2 (guarded): compare every race that still has money with FEC's summary
    try:
        fec = fetch()
    except Exception as e:
        fec = None
        n_money = int(len({(states[so[c]], r) for c, r in zip(*np.where(rs[..., i_hm] > 0.5))}) + (ds[:, j_hm] > 0.5).sum())
        if rep["capped"] <= MONEY_MAX_DROP * max(n_money, 1):
            rs[..., i_mr][cap] = 0; rs[..., i_cr][cap] = 0; rs[..., i_hm][cap] = 0
            for k in np.where(capd)[0]:
                zero_ds(k)
            rep["applied"] = True
            rep["reason"] = f"FEC summary unavailable ({e}); cleared {rep['capped']} capped ratios instead"
        else:
            rep["reason"] = f"FEC summary unavailable ({e}); {rep['capped']} capped ratios is too many to clear blindly, nothing removed"
    if fec is not None:
        si, hi = RACES.index("sen"), RACES.index("house")
        pairs, flags = [], []
        for s, st in enumerate(states):                         # Senate: one value per state
            m = (so == s) & (rs[:, si, i_hm] > 0.5)
            if m.any() and ("S", st, 0) in fec and ("sen", st) not in C.INDEPENDENTS:   # FEC "D" isn't the independent
                a, b = float(rs[m, si, i_mr][0]), fec[("S", st, 0)]
                pairs.append((a, b))
                if abs(a - b) > MONEY_TOL:
                    flags.append(("sen", st, None, a, b))
        dm = w.get("dmap")
        keys = [tuple(k) for k in dm["keys"]] if dm is not None else []
        for k, (st, dn) in enumerate(keys):                     # House: one value per district
            if ds.size and ds[k, j_hm] > 0.5 and ("H", st, int(dn)) in fec:
                a, b = float(ds[k, j_mr]), fec[("H", st, int(dn))]
                pairs.append((a, b))
                if abs(a - b) > MONEY_TOL:
                    flags.append(("house", st, int(dn), a, b))
        rep["checked"] = len(pairs)
        corr = float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) >= 20 else float("nan")
        share = len(flags) / max(len(pairs), 1)
        rep.update(corr=round(corr, 3) if np.isfinite(corr) else None, flagged_share=round(share, 3),
                   disagree=[dict(race=r, state=st, district=dn, bundle_x=round(float(np.exp(abs(a) * 3)), 1) * (1 if a >= 0 else -1),
                                  fec_x=round(float(np.exp(abs(b) * 3)), 1) * (1 if b >= 0 else -1)) for r, st, dn, a, b in flags])
        if len(pairs) < 20:
            rep["reason"] = f"only {len(pairs)} races could be compared: check not trusted, nothing removed"
        elif not (corr >= MONEY_MIN_CORR):
            rep["reason"] = f"bundle and FEC ratios barely agree (corr {corr:.2f}): mapping suspect, nothing removed"
        elif share > MONEY_MAX_DROP:
            rep["reason"] = f"would remove money from {share:.0%} of races (> {MONEY_MAX_DROP:.0%}): suspect, nothing removed"
        else:
            for r, st, dn, a, b in flags:
                if r == "sen":
                    s = states.index(st)
                    zero_rs(so == s, si)
                else:
                    k = keys.index((st, dn))
                    zero_ds(k)
                    cs = dm["pc"][np.asarray(dm["pd"]) == k]      # counties holding a piece of this district
                    zero_rs(np.isin(np.arange(len(so)), cs), hi)
            rep["applied"] = True
            rep["reason"] = f"removed money from {len(flags)} of {len(pairs)} races (corr {corr:.2f})"
    bundle["inst"]["race_static"], bundle["inst"]["dist_static"] = rs, ds
    return rep


class Engine:
    def __init__(self, bundle, model_path):
        self.b = bundle
        kermit_core.CAL.update(bundle["cal"])                  # calibrated noise levels used by Bayes
        w = World.__new__(World)
        w.__dict__.update(bundle["world"])
        self.w = w
        g = bundle["geo"]
        gaz = pd.DataFrame({"lat": g["lat"], "lon": g["lon"]}, index=np.asarray(w.county_fips).astype(str))
        self.rel = build_relations(w, gaz, [tuple(a) for a in g["adj"]], w.dmap)
        cfg = bundle["model_cfg"]
        self.model = Kermit(cfg["n_static"], d=cfg["d"], h=cfg["heads"], n_mix=cfg["mix"], n_latent=cfg["n_latent"],
                            n_factors=cfg["n_factors"], dropout=0.0)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.so = np.asarray(w.state_of)
        self.S, self.Cn = w.n_states, w.n_counties
        self.states = list(w.states)
        self.T = np.exp(np.asarray(bundle["inst"]["static"])[:, 1].astype(float))
        self.up = np.asarray(bundle["up"])
        self.dkeys = [tuple(k) for k in w.dmap["keys"]] if w.dmap is not None else []
        self.fit, self.coefs = bundle["bayes"]["fit"], bundle["bayes"]["coefs"]
        self.money_report = clean_money(bundle)


    # ---------- one instance ----------
    def instance(self, obs, cutoff, ctx_over=None, rs_zero=(), ds_zero=()):
        dos, _ = featurise(self.w, obs, cutoff, Bugs())
        b = self.b["inst"]
        ctx = np.array(b["context"], np.float32).copy()
        ctx[CONTEXT_COLS.index("cutoff_hours")] = cutoff
        for k, v in (ctx_over or {}).items():
            ctx[CONTEXT_COLS.index(k)] = v
        rs = np.array(b["race_static"], np.float32).copy()
        for c in rs_zero:
            rs[..., RS_IDX[c]] = 0
        ds = np.array(b["dist_static"], np.float32).copy()
        for c in ds_zero:
            if c in DS_IDX:
                ds[:, DS_IDX[c]] = 0
        nd = ds.shape[0]
        return dict(county=dos["county"], state=dos["state"], nation=dos["nation"], static=b["static"], race_static=rs,
                    dist_static=ds, context=ctx, y=np.full((self.Cn, R_), np.nan, np.float32),
                    y_cens=np.zeros((self.Cn, R_), np.float32), y_turnout=np.full(self.Cn, np.nan, np.float32),
                    y_d=np.full(nd, np.nan, np.float32), ycens_d=np.zeros(nd, np.float32))

    def kermit_strict(self, obs, cutoff, **kw):
        """One forward pass per distinct hide-set; each race (and the district head, from the House pass) from its own."""
        runs, out, base_inst = {}, None, None
        for race in RACES:
            key = tuple(sorted(C.STRICT[race]))
            if key not in runs:
                o_ = obs[~((obs.unit_level == "state") & obs.source.isin(key))] if key else obs
                inst = self.instance(o_, cutoff, **kw)
                runs[key] = predict(self.model, to_dev(inst), self.rel)
                if not key:
                    base_inst = inst
            o, ri = runs[key], RACES.index(race)
            if out is None:
                out = {k: v.copy() for k, v in o.items()}
            for k in ("mu", "sig", "L"):
                out[k][:, ri] = o[k][:, ri]
            if race == "house":
                for k in ("mu_d", "sig_d", "L_d"):
                    out[k] = o[k].copy()
        return out

    def run(self, obs, cutoff, n_draws=C.N_DRAWS, live_model="blend", **kw):
        """-> dict with county blend (mu, L, dg), district (mean, draws), and the instance used (full polls)."""
        inst = self.instance(obs, cutoff, **kw)
        prior = np.asarray(inst["static"])[:, :1].astype(float)
        mb, Lb, db = bayes_predict(inst, self.so, self.S, self.fit, self.coefs)
        bm, bd = county_to_district(mb, Lb, db, inst, self.w.dmap, n_draws)
        if live_model == "bayes":
            return dict(mu=mb, L=Lb, dg=db, dmean=bm, ddraws=bd, inst=inst)
        o = self.kermit_strict(obs, cutoff, **kw)
        mu, L, dg = combine_fixed(o["mu"], o["L"], o["sig"] ** 2, mb, Lb, db, prior)
        km, kd = kermit_district_draws(o, inst, n_draws)
        dm, dd = blend_districts(km, kd, bm, bd)
        return dict(mu=mu, L=L, dg=dg, dmean=dm, ddraws=dd, inst=inst)

    # ---------- summaries ----------
    def state_mean(self, mu):
        """[S, R] expected Dem share of each statewide race (turnout-weighted, like the backtest)."""
        prior = np.asarray(self.b["inst"]["static"])[:, 0]
        p = sig(prior[:, None] + mu)
        return np.stack([np.bincount(self.so, self.T * p[:, r], self.S) / np.maximum(np.bincount(self.so, self.T, self.S), 1e-9)
                         for r in range(R_)], 1)

    def state_draws(self, res, races=("sen", "gov"), n=C.N_DRAWS, seed=0):
        """Joint draws of every state's share for the given races (shared factors -> correlated states and races)."""
        rng = np.random.default_rng(seed)
        prior = np.asarray(self.b["inst"]["static"])[:, 0]
        K = res["L"].shape[-1]
        z = rng.standard_normal((n, K)).astype(np.float32)
        den = np.maximum(np.bincount(self.so, self.T, self.S), 1e-9)
        out = {}
        onehot = np.zeros((self.Cn, self.S), np.float32)
        onehot[np.arange(self.Cn), self.so] = self.T
        for race in races:
            ri = RACES.index(race)
            lg = prior[None] + res["mu"][None, :, ri] + z @ res["L"][:, ri].T.astype(np.float32) \
                + rng.standard_normal((n, self.Cn)).astype(np.float32) * np.sqrt(res["dg"][:, ri])[None].astype(np.float32)
            out[race] = (sig(lg) @ onehot) / den[None]
        return out


def _ci(x, lo, hi):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def build_outputs(eng, res, draws, drivers, meta, called=None, progress=None):
    """forecast.json content (+ live fields when progress is given)."""
    called = called or {}
    progress = progress or {}
    st_mean = eng.state_mean(res["mu"])
    races = {"senate": [], "house": [], "governor": []}
    chambers = {}
    names, inc = eng.b.get("names", {}), eng.b.get("inc_state", {})
    from polls import STATE_NAME
    for race, label in (("sen", "senate"), ("gov", "governor")):
        ri = RACES.index(race)
        win = np.zeros(draws[race].shape[0])
        ind_win = np.zeros(draws[race].shape[0])
        n_up = 0
        for s, st in enumerate(eng.states):
            if not eng.up[s, ri]:
                continue
            n_up += 1
            key = (race, st)
            d = draws[race][:, s]
            nm = C.NOT_MODELLED.get(key)
            cf = called.get(f"{race}|{st}")
            if nm:
                p = 0.0
            elif cf:
                p = 1.0 if cf == "D" else 0.0
            else:
                p = float((d > 0.5).mean())
            dwin = (d > 0.5) if not (nm or cf) else np.full(len(d), p > 0.5)
            ind = C.INDEPENDENTS.get(key)
            if ind:
                ind_win = ind_win + dwin if race == "sen" else ind_win
            else:
                win += dwin
            lo, hi = _ci(d, 5, 95)
            r = dict(id=st, state=st, name=STATE_NAME.get(st, st), p_dem_win=round(p, 4),
                     dem_share=None if nm else round(float(st_mean[s, ri]), 4),
                     lo90=None if nm else round(lo, 4), hi90=None if nm else round(hi, 4),
                     uncontested=bool(nm), note=nm, incumbent_party=inc.get(f"{race}|{st}"),
                     candidates=names.get(f"{race}|{st}"), drivers=drivers.get((race, st), []),
                     independent=C.INDEPENDENTS.get(key))
            if progress:
                pr = progress.get(f"{race}|{st}", {})
                r.update(reported_pct=pr.get("reported_pct"), status=pr.get("status", "not_started"), called_for=cf)
            races[label].append(r)
        if race == "sen":
            seats = C.SENATE_NOT_UP["D"] + win
            dist = pd.Series(seats).value_counts(normalize=True).sort_index()
            chambers["senate"] = dict(seats_total=100, majority_at=C.SENATE_MAJORITY,
                                      dem_seats_mean=round(float(seats.mean()), 1),
                                      dem_seats_lo80=int(np.percentile(seats, 10)), dem_seats_hi80=int(np.percentile(seats, 90)),
                                      p_dem_majority=round(float((seats >= C.SENATE_MAJORITY).mean()), 4),
                                      seat_distribution=[dict(seats=int(k), p=round(float(v), 5)) for k, v in dist.items()],
                                      not_up=dict(C.SENATE_NOT_UP), independent_seats_mean=round(float(ind_win.mean()), 2),
                                      with_independents=dict(p_dem_majority=round(float((seats + ind_win >= C.SENATE_MAJORITY).mean()), 4),
                                                             dem_seats_mean=round(float((seats + ind_win).mean()), 1)),
                                      note="Democrats need 51 seats: a 50-50 Senate is controlled by the Republican vice president. Independents without a Democrat on the ballot (green on the map) are not counted as Democratic seats.")
        else:
            chambers["governor"] = dict(races_up=n_up, dem_wins_mean=round(float(win.mean()), 1),
                                        note="Expected number of governorships Democrats win among the races on the ballot.")
    # House: drop the non-voting DC delegate and stray "at-large" units in multi-district states (artefacts of the
    # 2024 district file); districts with no county data are added as fixed safe seats (config.HOUSE_FIXED)
    from collections import Counter
    n_per = Counter(st for st, _ in eng.dkeys)
    keep = [k for k, (st, dn) in enumerate(eng.dkeys) if st != "DC" and not (int(dn) == 0 and n_per[st] > 1)]
    dd, dm = res["ddraws"][:, keep], res["dmean"][keep]
    keys = [eng.dkeys[k] for k in keep]
    dwin = dd > 0.5
    for j, (st, dn) in enumerate(keys):
        cf = called.get(f"house|{st}-{int(dn)}")
        if cf:
            dwin[:, j] = cf == "D"
    fixed_d = sum(1 for did, (party, _) in C.HOUSE_FIXED.items()
                  if (called.get("house|" + did.split("-")[0] + "-" + str(int(did.split("-")[1]))) or party) == "D")
    seats = dwin.sum(1) + fixed_d
    dist = pd.Series(seats).value_counts(normalize=True).sort_index()
    chambers["house"] = dict(seats_total=435, majority_at=C.HOUSE_MAJORITY, dem_seats_mean=round(float(seats.mean()), 1),
                             dem_seats_lo80=int(np.percentile(seats, 10)), dem_seats_hi80=int(np.percentile(seats, 90)),
                             p_dem_majority=round(float((seats >= C.HOUSE_MAJORITY).mean()), 4),
                             seat_distribution=[dict(seats=int(k), p=round(float(v), 5)) for k, v in dist.items()],
                             note=f"Districts modelled: {len(keys)}; fixed safe seats: {len(C.HOUSE_FIXED)}.")
    dsi = DS_IDX
    DS = np.asarray(eng.b["inst"]["dist_static"])
    for j, (st, dn) in enumerate(keys):
        k = keep[j]
        cf = called.get(f"house|{st}-{int(dn)}")
        did = f"{st}-{int(dn):02d}" if dn else f"{st}-AL"
        lo, hi = _ci(dd[:, j], 5, 95)
        incv = DS[k, dsi["inc"]] if DS[k, dsi["has_inc"]] > 0.5 else 0
        r = dict(id=did, state=st, district=int(dn), name=f"{STATE_NAME.get(st, st)} {int(dn) if dn else 'at-large'}",
                 p_dem_win=round(float(dwin[:, j].mean()), 4), dem_share=round(float(dm[j]), 4),
                 lo90=round(lo, 4), hi90=round(hi, 4), uncontested=False,
                 incumbent_party="D" if incv > 0.5 else "R" if incv < -0.5 else None,
                 drivers=drivers.get(("house", f"{st}-{int(dn)}"), []))
        if progress:
            pr = progress.get(f"house|{st}-{int(dn)}", {})
            r.update(reported_pct=pr.get("reported_pct"), status=pr.get("status", "not_started"), called_for=cf)
        races["house"].append(r)
    for did, (party, note) in C.HOUSE_FIXED.items():
        st, dn = did.split("-")[0], int(did.split("-")[1])
        cf = called.get(f"house|{st}-{dn}")
        r = dict(id=did, state=st, district=dn, name=f"{STATE_NAME.get(st, st)} {dn}", p_dem_win=1.0 if (cf or party) == "D" else 0.0,
                 dem_share=None, lo90=None, hi90=None, uncontested=True, note=note, incumbent_party=party, drivers=[])
        if progress:
            pr = progress.get(f"house|{st}-{dn}", {})
            r.update(reported_pct=pr.get("reported_pct"), status=pr.get("status", "not_started"), called_for=cf)
        races["house"].append(r)
    return dict(demo=False, model_version=eng.b["version"], generated_at=meta["generated_at"], data_cutoff=meta["data_cutoff"],
                election_date=str(C.ELECTION_DATE), chambers=chambers, races=races, inputs=meta.get("inputs", {}))


# ---------------- drivers ----------------
def driver_groups(eng, obs, live):
    """(name, how to hide it). Only groups with something to hide in this election are kept."""
    f = eng.b["fund"]
    neu = eng.b["fund_neutral"]
    G = []
    for src, name in (("poll_sen", "Senate polls"), ("poll_gov", "Governor polls"), ("poll_generic", "National generic-ballot polls"),
                      ("poll_house", "House district polls"), ("special", "Special elections since 2024"),
                      ("ev_read", "North Carolina early vote"), ("count_eday", "Votes counted so far")):
        if (obs.source == src).any() and (live or src != "count_eday"):
            G.append((src, name, dict(drop_src=src)))
    G.append(("incumbency", "Incumbency", dict(rs_zero=GROUP_COLS["incumbency"], ds_zero=GROUP_COLS["incumbency"])))
    G.append(("past", "Past results of this race", dict(rs_zero=GROUP_COLS["past"], ds_zero=GROUP_COLS["past"])))
    G.append(("money", "Campaign money (pre-general reports)", dict(rs_zero=GROUP_COLS["money"], ds_zero=GROUP_COLS["money"])))
    G.append(("approval", f"Presidential approval ({f['approval']:.0f}%)", dict(ctx_over=dict(approval=neu["approval"]))))
    G.append(("economy", "Economy: jobs and prices", dict(ctx_over=dict(unemp_change=neu["unemp_change"], inflation=neu["inflation"]))))
    return G


def drivers(eng, obs, cutoff, base_res, poll_sum, live=False, ctx_over=None, n_small=300, live_model="blend"):
    """Top-5 inputs per race by how far hiding them moves the (blended) forecast, in points of Dem share."""
    st0 = eng.state_mean(base_res["mu"])
    d0 = base_res["dmean"]
    shifts = {}
    for gid, name, how in driver_groups(eng, obs, live):
        o_ = obs[obs.source != how["drop_src"]] if "drop_src" in how else obs
        kw = dict(ctx_over={**(ctx_over or {}), **how.get("ctx_over", {})}, rs_zero=how.get("rs_zero", ()),
                  ds_zero=how.get("ds_zero", ()))
        r = eng.run(o_, cutoff, n_draws=n_small, live_model=live_model, **kw)
        shifts[(gid, name)] = ((st0 - eng.state_mean(r["mu"])) * 100, (d0 - r["dmean"]) * 100)
    out = {}
    RS = np.asarray(eng.b["inst"]["race_static"])
    DS = np.asarray(eng.b["inst"]["dist_static"])
    for race in ("sen", "gov"):
        ri = RACES.index(race)
        for s, st in enumerate(eng.states):
            if not eng.up[s, ri]:
                continue
            m = eng.so == s
            items = []
            for (gid, name), (sv, _) in shifts.items():
                v = float(sv[s, ri])
                items.append((abs(v), dict(label=name, detail=_detail(gid, race, st, poll_sum, RS[m, ri]), shift_pp=round(v, 2))))
            out[(race, st)] = [x for a, x in sorted(items, key=lambda t: -t[0])[:5] if a >= 0.05]
    for k, (st, dn) in enumerate(eng.dkeys):
        items = []
        for (gid, name), (_, dv) in shifts.items():
            v = float(dv[k])
            items.append((abs(v), dict(label=name, detail=_detail(gid, "house", f"{st}-{int(dn)}", poll_sum, None, DS[k]),
                                       shift_pp=round(v, 2))))
        out[("house", f"{st}-{int(dn)}")] = [x for a, x in sorted(items, key=lambda t: -t[0])[:5] if a >= 0.05]
    return out


def _detail(gid, race, key, poll_sum, rs_rows=None, ds_row=None):
    if gid in ("poll_sen", "poll_gov"):
        r = gid[5:]
        ps = poll_sum.get((r, key))
        if ps:
            return f"{ps['n']} poll{'s' if ps['n'] != 1 else ''} of this race, latest {ps['latest']}" if r == race else \
                   f"{ps['n']} {'Senate' if r == 'sen' else 'governor'} poll{'s' if ps['n'] != 1 else ''} in this state"
        return "polls of other states' races" if r == race else ""
    if gid == "poll_generic":
        ps = poll_sum.get(("generic", "US"))
        if ps:
            a = ps["avg"]
            return f"{ps['n']} polls, recent average {'D' if a >= 0 else 'R'}+{abs(a):.1f}"
    if gid == "poll_house":
        ps = poll_sum.get(("house", key))
        return f"{ps['n']} poll{'s' if ps['n'] != 1 else ''} of this district" if ps else "polls of other districts"
    if gid == "incumbency":
        v = (ds_row[DS_IDX["inc"]] if ds_row is not None else (rs_rows[:, RS_IDX["inc"]].mean() if rs_rows is not None and len(rs_rows) else 0))
        return "Democratic incumbent running" if v > 0.3 else "Republican incumbent running" if v < -0.3 else "open seat"
    if gid == "money":
        v = ds_row[DS_IDX["money_ratio"]] if ds_row is not None else (rs_rows[:, RS_IDX["money_ratio"]].mean() if rs_rows is not None and len(rs_rows) else 0)
        x = float(np.exp(abs(v) * 3))
        if abs(v) < 1e-6:
            return ""
        return f"{'Democrat' if v > 0 else 'Republican'} raised {x:.1f}x the {'Republican' if v > 0 else 'Democrat'}"
    return ""
