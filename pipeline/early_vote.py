"""North Carolina early/mail vote, refreshed daily from the NC State Board of Elections per-ballot file (the only
early-vote source the model was trained to read: county x day returns by party, compared with the same county at
the same point in the previous comparable election). Same maths as the notebook's nc_early_vote_obs()."""
import io, zipfile
import numpy as np
import pandas as pd
import requests

import config as C
from kermit_core import _obs, OBS_COLS, logit


def nc_obs(bundle, log):
    nc = bundle.get("nc") or {}
    prev, name2fips = nc.get("prev"), nc.get("name2fips")
    if prev is None or not name2fips:
        log.append("NC early vote: no baseline in the bundle, kept the bundle's reads")
        return None
    try:
        r = requests.get(C.NCSBE_ZIP, headers={"User-Agent": C.USER_AGENT}, timeout=600)
        r.raise_for_status()
    except Exception as e:
        log.append(f"NC early vote download failed ({e}); kept the bundle's reads")
        return None
    want = {"county_desc": "county_desc", "ballot_rtn_status": "status", "ballot_rtn_dt": "rtn_dt", "voter_party_code": "party"}
    parts = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        inner = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        with z.open(inner) as fh:
            for ch in pd.read_csv(fh, dtype=str, encoding="latin-1", chunksize=400_000,
                                  usecols=lambda c: c.strip().lower() in want):
                ch.columns = [want[c.strip().lower()] for c in ch.columns]
                ch = ch[ch.status.astype(str).str.upper().str.startswith("ACCEPTED")]
                ch["day"] = pd.to_datetime(ch.rtn_dt, errors="coerce")
                p3 = ch.party.astype(str).str.upper().str[:3]
                ch["D"], ch["R"] = (p3 == "DEM").astype(int), (p3 == "REP").astype(int)
                parts.append(ch.groupby(["county_desc", "day"])[["D", "R"]].sum().reset_index())
    if not parts:
        log.append("NC early vote: file had no accepted ballots")
        return None
    ev = pd.concat(parts).groupby(["county_desc", "day"])[["D", "R"]].sum().reset_index()
    eday = pd.Timestamp(C.ELECTION_DATE)
    ev["fips"] = ev.county_desc.astype(str).str.upper().str.strip().map(name2fips)
    ev = ev.dropna(subset=["fips", "day"])
    ev = ev[(ev.day < eday) & (ev.day >= eday - pd.Timedelta(days=90))]
    ev["days_out"] = (eday - ev.day).dt.days
    d = ev.groupby(["fips", "days_out"])[["D", "R"]].sum().sort_index(level=[0, 1], ascending=[True, False])
    cur = d.groupby(level=0).cumsum().reset_index()
    j = cur.merge(prev, on=["fips", "days_out"], how="left")
    j["lean_prev"] = j.groupby("fips").lean_prev.transform(lambda s: s.bfill().ffill())
    f2u = {f: i for i, f in enumerate(np.asarray(bundle["world"]["county_fips"]).astype(str))}
    j["u"] = j.fips.map(f2u)
    j = j.dropna(subset=["u", "lean_prev"])
    j = j[(j.D + j.R) >= 10]
    # checks: counties 100, D share sane, cumulative totals never fall
    share = j.D / (j.D + j.R)
    if j.fips.nunique() < 80 or not share.between(0.02, 0.98).all():
        log.append(f"NC early vote failed checks ({j.fips.nunique()} counties); kept the bundle's reads")
        return None
    log.append(f"NC early vote: {int((cur.D + cur.R).groupby(cur.fips).max().sum()):,} party ballots returned, "
               f"{j.fips.nunique()} counties, latest {int(cur.days_out.min())} days out")
    return _obs("county", j.u.astype(int).values, "ev_read", -24.0 * j.days_out.values, share.values,
                (j.D + j.R).values.astype(float), j.lean_prev.values)
