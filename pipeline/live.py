"""Election night: county-level counts from the New York Times results feeds (AP data; the same JSON its own pages
read, county level, no key needed), checked, turned into the model's count observations the way the notebook did
for 2024 (night_obs_2024: each new batch of counted votes in a county = one observation).

The top race in a midterm is the House, so county counts = House votes summed over the district pieces in each county
(uncontested House races left out, as in training). Senate/governor feeds give each race's own progress and AP calls.
"""
import os, json, re, datetime as dt
import numpy as np
import pandas as pd
import requests

import config as C
from kermit_core import OBS_COLS

FIPS2POSTAL = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
               "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
               "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
               "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
               "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
               "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
POSTAL2FIPS = {v: k for k, v in FIPS2POSTAL.items()}
STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
HIST = os.path.join(STATE_DIR, "live_history.csv.gz")      # cumulative county totals seen at each run
BATCH = os.path.join(STATE_DIR, "live_batches.csv.gz")     # the count observations built so far


def fetch(url):
    """JSON, or None if the feed isn't up yet (404/403 before election night)."""
    r = requests.get(url, headers={"User-Agent": C.USER_AGENT}, params={"t": int(dt.datetime.utcnow().timestamp())}, timeout=60)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    return r.json()


def _votes(c):
    v = c.get("votes")
    if isinstance(v, dict):
        v = v.get("total", v.get("votes"))
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _party(cid, cand, meta):
    for src in (cand, meta.get(cid, {}) if isinstance(meta, dict) else {}):
        p = src.get("party") if isinstance(src, dict) else None
        if isinstance(p, dict):
            p = p.get("abbreviation") or p.get("name") or p.get("nyt_id")
        p = str(p or "").upper()
        if p.startswith("D"):
            return "D"
        if p.startswith("R") or p.startswith("GOP"):
            return "R"
    return None


def _state_of(race, units):
    st = race.get("state")
    if isinstance(st, dict):
        for k in ("postal", "abbreviation", "nyt_abbrev", "state_postal"):
            if st.get(k) and str(st[k]).upper() in POSTAL2FIPS:
                return str(st[k]).upper()
    if isinstance(st, str) and st.upper() in POSTAL2FIPS:
        return st.upper()
    for u in units:
        f = str(u.get("fips_state") or "")[:2].zfill(2)
        if f in FIPS2POSTAL:
            return FIPS2POSTAL[f]
    m = re.search(r"\b([A-Z]{2})\b", str(race.get("nyt_id") or race.get("race_id") or ""))
    return m.group(1) if m and m.group(1) in POSTAL2FIPS else None


def _district(race):
    for k in ("district", "seat_number"):
        if race.get(k) not in (None, ""):
            try:
                return int(race[k])
            except (TypeError, ValueError):
                pass
    seat = race.get("seat")
    txt = " ".join(str(x) for x in ([seat.get("name"), seat.get("number"), seat.get("nyt_id")] if isinstance(seat, dict) else [seat])
                   + [race.get("seat_name"), race.get("nyt_id"), race.get("race_id")] if x is not None)
    if re.search(r"at[\s\-]?large", txt, re.I):
        return 0
    m = re.search(r"(?:district|seat)\D{0,3}(\d{1,2})\b", txt, re.I) or re.search(r"-(\d{1,2})(?:-|$)", txt)
    return int(m.group(1)) if m else None


def _county_fips(u):
    fc = str(u.get("fips_county") or u.get("fips") or "").strip()
    fs = str(u.get("fips_state") or "").strip()
    if len(fc) >= 5 and fc[:5].isdigit():
        f = fc[:5]
    elif fc.isdigit() and fs.isdigit():
        f = fs.zfill(2) + fc.zfill(3)
    else:
        return None
    return "02000" if f.startswith("02") else f


def parse(js, race_type):
    """-> (county DataFrame [race_key, fips, D, R], race list [dict]) from one NYT results file."""
    rows, races = [], []
    if not js:
        return pd.DataFrame(columns=["race_key", "fips", "D", "R"]), races
    meta_top = js.get("candidate_metadata", {}) if isinstance(js, dict) else {}
    race_list = js.get("races") if isinstance(js, dict) else None
    if race_list is None:
        race_list = [d for d in _walk(js) if isinstance(d.get("reporting_units"), list)]
    for race in race_list:
        units = race.get("reporting_units") or []
        meta = {**meta_top, **(race.get("candidate_metadata") or {})}
        st = _state_of(race, units)
        if st is None:
            continue
        dist = _district(race) if race_type == "house" else None
        key = f"{st}-{dist}" if race_type == "house" else st
        state_unit = next((u for u in units if str(u.get("level", "")).lower() in ("state", "district")), None)
        tot = {"D": 0.0, "R": 0.0}
        parties = set()
        winner = None
        for u in units:
            lvl = str(u.get("level", "")).lower()
            per = {"D": 0.0, "R": 0.0}
            for c in u.get("candidates") or []:
                cid = c.get("nyt_id") or c.get("candidate_id") or c.get("candidate_key")
                p = _party(cid, c, meta)
                if p:
                    parties.add(p)
                    per[p] += _votes(c)
                if c.get("winner") is True and p and u is state_unit:
                    winner = p
            if u is state_unit:
                tot = per
            elif lvl not in ("national", "state", "district"):
                f = _county_fips(u)
                if f:
                    rows.append(dict(race_key=key, fips=f, D=per["D"], R=per["R"]))
        for k in ("winner", "winners", "result_winner"):
            wv = race.get(k)
            if isinstance(wv, (list, str, dict)) and winner is None:
                ids = wv if isinstance(wv, list) else [wv]
                for wid in ids:
                    wid = wid.get("nyt_id") if isinstance(wid, dict) else wid
                    p = _party(wid, {}, meta)
                    if p:
                        winner = p
        eevp = None
        if state_unit:
            for k in ("eevp", "expected_vote_pct", "pct_expected_vote"):
                if state_unit.get(k) is not None:
                    eevp = float(state_unit[k]); break
            if eevp is None and state_unit.get("precincts_total"):
                eevp = 100.0 * float(state_unit.get("precincts_reporting") or 0) / float(state_unit["precincts_total"])
        races.append(dict(race=race_type, key=key, state=st, district=dist, D=tot["D"], R=tot["R"],
                          contested={"D", "R"} <= parties, reported_pct=eevp, called_for=winner,
                          updated_at=race.get("updated_at") or js.get("updated_at")))
    return pd.DataFrame(rows, columns=["race_key", "fips", "D", "R"]), races


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def county_totals(house_counties, house_races):
    """House votes per county, contested races only (as in training)."""
    if not len(house_counties):
        return pd.DataFrame(columns=["fips", "D", "R"])
    ok = {r["key"] for r in house_races if r["contested"]}
    h = house_counties[house_counties.race_key.isin(ok)]
    return h.groupby("fips")[["D", "R"]].sum().reset_index()


def check_and_batch(cur, bundle, t_now, log):
    """Sanity checks, then new batches = growth since the last run. Returns (batch obs to add, good, report)."""
    w = bundle["world"]
    fips = list(np.asarray(w["county_fips"]).astype(str))
    f2u = {f: i for i, f in enumerate(fips)}
    so = np.asarray(w["state_of"])
    Texp = np.asarray(w["turnout_exp"], float) * C.MIDTERM_TURNOUT
    prev = pd.read_csv(HIST, dtype={"fips": str}) if os.path.exists(HIST) else pd.DataFrame(columns=["ts", "fips", "D", "R"])
    last = prev.sort_values("ts").groupby("fips").tail(1).set_index("fips") if len(prev) else pd.DataFrame(columns=["D", "R"])
    bad = {}

    def flag(why):
        bad[why] = bad.get(why, 0) + 1

    keep, obs = [], []
    reporting = cur[(cur.D + cur.R) > 0]
    for r in reporting.itertuples():
        if r.fips not in f2u:
            flag("county not in the model"); continue
        u = f2u[r.fips]
        tot = r.D + r.R
        if tot > C.LIVE_MAX_COUNTY_RATIO * Texp[u]:
            flag("more votes than plausible turnout"); continue
        if tot > 1000 and not (0.02 <= r.D / tot <= 0.98):
            flag("one-sided count (likely a parsing error)"); continue
        pD, pR = (float(last.loc[r.fips, "D"]), float(last.loc[r.fips, "R"])) if r.fips in last.index else (0.0, 0.0)
        dD, dR = r.D - pD, r.R - pR
        if dD < 0 or dR < 0:
            if (pD + pR) > 0 and (pD + pR - tot) / (pD + pR) > C.LIVE_MAX_DROP:
                flag("count went down a lot (skipped this run)"); continue
            keep.append((r.fips, r.D, r.R))          # small correction: new baseline, no batch
            continue
        keep.append((r.fips, r.D, r.R))
        if dD + dR >= 20:
            s = so[u]
            obs.append(("county", u, "count_eday", float(t_now), dD / (dD + dR), dD + dR,
                        float(w["prior_logit"][u] + w["mode_offset_prior"][s, 0]), float(w["turnout_exp"][u])))
    # state-level plausibility
    if keep:
        k = pd.DataFrame(keep, columns=["fips", "D", "R"])
        k["s"] = k.fips.map(lambda f: so[f2u[f]])
        cnt = k.groupby("s")[["D", "R"]].sum().sum(1)
        exp_s = pd.Series(np.bincount(so, Texp, len(w["states"])))
        over = cnt[cnt > 1.3 * exp_s.reindex(cnt.index)]
        if len(over):
            bad_states = set(over.index)
            flag(f"state totals implausible: {[w['states'][s] for s in bad_states]}")
            keep = [x for x in keep if so[f2u[x[0]]] not in bad_states]
            obs = [o for o in obs if so[o[1]] not in bad_states]
    n_rep = len(reporting)
    good_share = len(keep) / n_rep if n_rep else 1.0
    report = dict(counties_reporting=int(n_rep), counties_ok=len(keep), new_batches=len(obs), problems=bad,
                  good_share=round(good_share, 3))
    good = good_share >= C.LIVE_MIN_GOOD_SHARE
    if not good:
        log.append(f"LIVE CHECK FAILED: only {good_share:.0%} of reporting counties passed ({bad}); live update skipped")
    return pd.DataFrame(obs, columns=OBS_COLS), keep, good, report


def commit_state(keep, new_obs, ts):
    os.makedirs(STATE_DIR, exist_ok=True)
    h = pd.DataFrame(keep, columns=["fips", "D", "R"]).assign(ts=ts)
    old = pd.read_csv(HIST, dtype={"fips": str}) if os.path.exists(HIST) else None
    pd.concat([old, h], ignore_index=True).to_csv(HIST, index=False, compression="gzip")
    oldb = pd.read_csv(BATCH) if os.path.exists(BATCH) else None
    pd.concat([oldb, new_obs], ignore_index=True).to_csv(BATCH, index=False, compression="gzip")


def all_batches(extra=None):
    b = pd.read_csv(BATCH) if os.path.exists(BATCH) else pd.DataFrame(columns=OBS_COLS)
    if extra is not None and len(extra):
        b = pd.concat([b, extra], ignore_index=True)
    return b


def pull(bundle, t_now, log):
    """Everything for one live update. Never raises: problems end up in the log and in good=False."""
    feeds, races, counties = {}, [], {}
    for race, url in C.NYT_FEEDS.items():
        try:
            js = fetch(url)
            feeds[race] = js is not None
            c, r = parse(js, race)
            counties[race] = c
            races += r
        except Exception as e:
            feeds[race] = f"error: {e}"
            log.append(f"NYT {race} feed failed: {e}")
    if not any(v is True for v in feeds.values()):
        log.append("no live feed up yet")
        return dict(available=False, feeds=feeds, races=races, new_obs=pd.DataFrame(columns=OBS_COLS), keep=[], report={})
    house_r = [r for r in races if r["race"] == "house"]
    cur = county_totals(counties.get("house", pd.DataFrame()), house_r)
    if not len(cur):
        log.append("house feed has no county rows: running on pre-election inputs + race progress only")
    new_obs, keep, good, report = check_and_batch(cur, bundle, t_now, log)
    return dict(available=True, feeds=feeds, races=races, new_obs=new_obs if good else new_obs.iloc[:0],
                keep=keep if good else [], report=report, good=good)
