"""Polls for the 2026 forecast: VoteHub (generic ballot, approval) + Wikipedia race pages (Senate, governor), merged
with the polls already in the bundle, checked, and turned into the model's observations exactly like the notebook's
poll_obs(). Parsing helpers are copied from the extra-data pull notebook (c1/c3), which built the training polls."""
import io, re, time, calendar, datetime as dt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config as C
from kermit_core import OBS_COLS, _obs, sig, logit

S = requests.Session()
S.headers["User-Agent"] = C.USER_AGENT

STATES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
          "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
          "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
          "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
          "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
          "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
          "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
          "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
          "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
STATE_NAME = {v: k for k, v in STATES.items()}
POSTAL = set(STATES.values())
_NAME_RE = re.compile("|".join(sorted(map(re.escape, STATES), key=len, reverse=True)))
TIDY_COLS = ["source", "race", "year", "state", "district", "pollster", "start_date", "end_date", "sample_size",
             "population", "dem_pct", "rep_pct", "dem_name", "rep_name"]
AGGREGATOR = (r"average|aggregat|realclear|\bRCP\b|fivethirtyeight|\b538\b|270towin|decision desk|race to the wh|"
              r"silver bulletin|votehub|predictit|polymarket|kalshi")


def get(url, tries=6, timeout=60, **kw):
    """GET with retries; on 429 (rate limited) waits as long as the server asks (or 10, 20, 40 s ...)."""
    for k in range(tries):
        try:
            r = S.get(url, timeout=timeout, **kw)
            if r.status_code == 429 and k < tries - 1:
                wait = float(r.headers.get("Retry-After") or 10 * 2 ** k)
                time.sleep(min(wait, 120))
                continue
            r.raise_for_status()
            return r
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


# ---------------- parsing helpers (from the pull notebook) ----------------
def to_postal(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "US", None
    t = str(s).strip()
    if t == "" or t.upper() in {"US", "USA", "NATIONAL", "--", "NATIONWIDE", "UNITED STATES", "GENERIC"}:
        return "US", None
    dist = None
    m = re.search(r"(?:CD|district)[\s\-]*(\d+)|-(\d{1,2})$|(\d+)(?:st|nd|rd|th)\s+congressional", t, re.I)
    if m:
        dist = int(next(g for g in m.groups() if g))
    if re.search(r"D\.?\s?C\.?$|District of Columbia", t) and not re.search(r"Washington$", t):
        return "DC", None
    m2 = re.match(r"^([A-Z]{2})\b", t)
    if m2 and m2.group(1) in POSTAL:
        return m2.group(1), dist
    m3 = _NAME_RE.search(t)
    if m3:
        return STATES[m3.group(0)], dist
    return None, dist


_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS.update({"sept": 9})


def _parse_part(p):
    p = p.strip().rstrip(",.")
    mon = day = yr = None
    for tok in re.findall(r"[A-Za-z]+\.?|\d+", p):
        tl = tok.lower().rstrip(".")
        if tl in _MONTHS:
            mon = _MONTHS[tl]
        elif tok.isdigit():
            v = int(tok)
            if v > 31:
                yr = v
            elif day is None:
                day = v
    return mon, day, yr


def parse_date_range(s, year):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT, pd.NaT
    t = re.sub(r"\[.*?\]", "", str(s)).replace("\xa0", " ")
    parts = re.split(r"\s*(?:–|—|-|\bto\b)\s*", t)
    parts = [p for p in parts if re.search(r"\d|[A-Za-z]{3}", p)]
    if not parts:
        return pd.NaT, pd.NaT
    rm, rd, ry = _parse_part(parts[-1])
    lm, ld, ly = _parse_part(parts[0]) if len(parts) > 1 else (rm, rd, ry)
    rm = rm or lm
    lm = lm or rm
    ry = ry or ly or year
    ly = ly or (ry - 1 if (lm and rm and lm > rm) else ry)
    if rm is None:
        return pd.NaT, pd.NaT
    rd = rd or 15
    ld = ld or rd if len(parts) == 1 else (ld or 1)
    try:
        end = pd.Timestamp(dt.date(ry, rm, min(rd, calendar.monthrange(ry, rm)[1])))
        start = pd.Timestamp(dt.date(ly, lm, min(ld, calendar.monthrange(ly, lm)[1])))
    except (ValueError, TypeError):
        return pd.NaT, pd.NaT
    return min(start, end), end


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", re.sub(r"\[.*?\]", "", str(x)))
    return float(m.group(1)) if m else np.nan


def sample_info(x):
    t = re.sub(r"\[.*?\]", "", str(x))
    m = re.search(r"(\d[\d,]*)", t)
    n = float(m.group(1).replace(",", "")) if m else np.nan
    p = re.search(r"\b(LV|RV|A|V)\b", t, re.I)
    return n, (p.group(1).lower() if p else None)


def find_col(cols, pat, exclude=None):
    for c in cols:
        if re.search(pat, c, re.I) and not (exclude and re.search(exclude, c, re.I)):
            return c
    return None


def surname(c):
    t = re.sub(r"\(.*?\)|\[.*?\]|\b(jr|sr|ii|iii|iv)\b\.?|incumbent|\*|†|✓", " ", str(c), flags=re.I)
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", t)
    return toks[-1].lower() if toks else None


def party_of_header(h, name2party):
    hl = h.lower()
    for nm, p in name2party.items():             # a name mapped on purpose (independents) wins over the party tag
        if p == "D" and len(nm) > 2 and re.search(rf"\b{re.escape(nm)}\b", hl) and re.search(r"\(i\)|independent", hl):
            return p
    if re.search(r"democrat|\(d\)|\(dfl\)|\bdem\.?$", hl):
        return "D"
    if re.search(r"republican|\(r\)|\bgop\b|\brep\.?$", hl):
        return "R"
    for nm, p in name2party.items():
        if len(nm) > 2 and re.search(rf"\b{re.escape(nm)}\b", hl):
            return p
    return None


def flat_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for tup in df.columns:
            parts = []
            for p in tup:
                p = re.sub(r"\[.*?\]", "", str(p)).strip()
                if p and not p.startswith("Unnamed") and p not in parts:
                    parts.append(p)
            cols.append(" ".join(parts))
    else:
        cols = [re.sub(r"\[.*?\]", "", str(c)).strip() for c in df.columns]
    seen, out = {}, []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}.{seen[c] - 1}")
    df = df.copy()
    df.columns = out
    return df


def wiki_html(title):
    j = get(C.WIKI_API, params=dict(action="parse", page=title, prop="text", format="json", formatversion=2, redirects=1)).json()
    if "error" in j:
        raise KeyError(title)
    time.sleep(1.0)                      # stay well under Wikipedia's rate limit
    return j["parse"]["text"]


def wiki_tables_with_headings(html):
    soup = BeautifulSoup(html, "lxml")
    out, stack = [], []
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name in ("h2", "h3", "h4"):
            lvl = int(el.name[1]) - 2
            stack = stack[:lvl] + [re.sub(r"\[\s*edit\s*\]", "", el.get_text(" ", strip=True)).strip()]
            continue
        if "wikitable" not in (el.get("class") or []) or el.find_parent("table") is not None:
            continue
        try:
            df = pd.read_html(io.StringIO(str(el)))[0]
        except Exception:
            continue
        out.append((" | ".join(stack), flat_cols(df)))
    return out


def wiki_links(html, pattern):
    soup = BeautifulSoup(html, "lxml")
    seen = []
    for a in soup.find_all("a", href=True):
        m = re.match(r"^/wiki/([^#?]+)$", a["href"])
        if m:
            t = requests.utils.unquote(m.group(1)).replace("_", " ")
            if re.search(pattern, t) and t not in seen:
                seen.append(t)
    return seen


def name_party_map(tables):
    m = {}
    for _, df in tables:
        cols = list(df.columns)
        pc = [c for c in cols if re.match(r"party", c, re.I)]
        cc = find_col(cols, r"candidate|nominee")
        if not pc or cc is None:
            continue
        for _, r in df.iterrows():
            ptxt = " ".join(str(r[c]) for c in pc)
            p = "D" if re.search(r"Democratic|DFL", ptxt) else "R" if re.search(r"Republican", ptxt) else None
            sn = surname(r[cc])
            if p and sn and sn != "nan":
                m.setdefault(sn, p)
    return m


def parse_poll_table(df, year, race, state, name2party=None):
    cols = list(df.columns)
    c_src = find_col(cols, r"poll source|pollster|polling (firm|organi)|^source")
    c_date = find_col(cols, r"date")
    if c_src is None or c_date is None:
        return []
    c_n = find_col(cols, r"sample")
    c_moe = find_col(cols, r"error|moe")
    skip = r"poll|date|sample|error|moe|lead|margin|other|undecided|^und|^n$|spread"
    used = {c_src, c_date, c_n, c_moe}
    dcols = [c for c in cols if c not in used and not re.search(skip, c, re.I) and party_of_header(c, name2party or {}) == "D"]
    rcols = [c for c in cols if c not in used and not re.search(skip, c, re.I) and party_of_header(c, name2party or {}) == "R"]
    if not dcols or not rcols:
        return []
    out = []
    for _, r in df.iterrows():
        src = re.sub(r"\[.*?\]", "", str(r[c_src])).strip()
        if src in ("", "nan") or re.search(AGGREGATOR, src, re.I):
            continue
        dv = [(c, pct(r[c])) for c in dcols]
        rv = [(c, pct(r[c])) for c in rcols]
        dv = [x for x in dv if np.isfinite(x[1])]
        rv = [x for x in rv if np.isfinite(x[1])]
        if len(dv) != 1 or len(rv) != 1 or not (20 <= dv[0][1] + rv[0][1] <= 101):
            continue
        st, en = parse_date_range(r[c_date], year)
        n, popl = sample_info(r[c_n]) if c_n else (np.nan, None)
        out.append(dict(source="wikipedia", race=race, year=year, state=state, district=None, pollster=src,
                        start_date=st, end_date=en, sample_size=n, population=popl,
                        dem_pct=dv[0][1], rep_pct=rv[0][1], dem_name=dv[0][0], rep_name=rv[0][0]))
    return out


# ---------------- sources ----------------
def votehub(poll_type, subject=None):
    params = dict(poll_type=poll_type)
    if subject:
        params["subject"] = subject
    js = get(C.VOTEHUB, params=params, timeout=120).json()
    return js if isinstance(js, list) else js.get("polls", js.get("data", []))


def votehub_generic():
    rows = []
    for p in votehub("generic-ballot"):
        a = {str(x.get("choice", "")).strip().lower()[:3]: x.get("pct") for x in p.get("answers") or []}
        d_, r_ = a.get("dem"), a.get("rep")
        if d_ is None or r_ is None:
            continue
        rows.append(dict(source="votehub", race="generic", year=2026, state="US", district=None, pollster=p.get("pollster"),
                         start_date=pd.to_datetime(p.get("start_date"), errors="coerce"),
                         end_date=pd.to_datetime(p.get("end_date"), errors="coerce"),
                         sample_size=pd.to_numeric(p.get("sample_size"), errors="coerce"), population=p.get("population"),
                         dem_pct=float(d_), rep_pct=float(r_), dem_name="Democrats", rep_name="Republicans"))
    return pd.DataFrame(rows, columns=TIDY_COLS)


def votehub_approval():
    rows = []
    for p in votehub("approval", "donald-trump"):
        a = {str(x.get("choice", "")).strip().lower(): x.get("pct") for x in p.get("answers") or []}
        if a.get("approve") is not None:
            rows.append(dict(end_date=pd.to_datetime(p.get("end_date"), errors="coerce"), approve=float(a["approve"]),
                             pollster=p.get("pollster")))
    return pd.DataFrame(rows)


def wikipedia_races(up_states, log):
    """Senate + governor general-election polls from each 2026 race page (only states with that race on the ballot)."""
    rows = []
    for race, hub in C.WIKI_HUBS.items():
        pat = r"^2026 United States Senate (special )?election in " if race == "sen" else r"^2026 .+ gubernatorial election$"
        try:
            titles = wiki_links(wiki_html(hub), pat)
        except Exception as e:
            log.append(f"wikipedia {race} hub page failed: {e}")
            continue
        n0, pages = len(rows), 0
        for t in titles:
            st, _ = to_postal(t.replace("2026", ""))
            if st not in up_states.get(race, set()):
                continue
            try:
                tabs = wiki_tables_with_headings(wiki_html(t))
            except Exception:
                continue
            pages += 1
            n2p = name_party_map(tabs)
            ind = C.INDEPENDENTS.get((race, st))
            if ind:
                n2p[surname(ind)] = "D"            # the independent is the non-Republican side of this race
            for head, df in tabs:
                if re.search(r"primary|runoff|hypothetical", head, re.I) and not re.search(r"general", head, re.I):
                    continue
                if re.search(r"hypothetical", head, re.I):
                    continue
                rows += parse_poll_table(df, 2026, race, st, n2p)
        log.append(f"wikipedia {race}: {pages} race pages, {len(rows) - n0} poll rows")
    return pd.DataFrame(rows, columns=TIDY_COLS)


# ---------------- merge + checks ----------------
def _norm_pollster(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())[:14]


def clean(P, up_states, today, log):
    """Sanity checks. Every dropped row is counted by reason in the log."""
    P = P.copy()
    n0 = len(P)
    drops = {}

    def drop(mask, why):
        k = int(mask.sum())
        if k:
            drops[why] = drops.get(why, 0) + k
        return P[~mask]

    for c in ("dem_pct", "rep_pct", "sample_size"):
        P[c] = pd.to_numeric(P[c], errors="coerce")
    P["end_date"] = pd.to_datetime(P.end_date, errors="coerce")
    P["start_date"] = pd.to_datetime(P.start_date, errors="coerce").fillna(P.end_date)
    P = drop(P.race.isna() | ~P.race.isin(["sen", "gov", "house", "generic", "pres"]), "unknown race")
    P = drop(P.race.eq("pres"), "no presidential race in 2026")
    P = drop(P.end_date.isna(), "no date")
    P = drop(P.end_date.dt.date > today, "dated in the future")
    P = drop(P.end_date.dt.date < C.POLL_MIN_DATE, "before the cycle")
    P = drop(P.end_date.dt.date > C.ELECTION_DATE, "after election day")
    P = drop(~(P.dem_pct.between(5, 95) & P.rep_pct.between(5, 95)), "share out of range")
    P = drop(~(P.dem_pct + P.rep_pct).between(40, 101), "D+R not between 40 and 101")
    P = drop(P.pollster.isna() | P.pollster.astype(str).str.contains(AGGREGATOR, case=False, regex=True), "aggregator / no pollster")
    statewide = P.race.isin(["sen", "gov"])
    ok_state = pd.Series([st in up_states.get(r, set()) for r, st in zip(P.race, P.state)], index=P.index)
    P = drop(statewide & ~ok_state, "race not on this state's ballot")
    P = drop(P.race.eq("generic") & P.state.ne("US"), "generic ballot not national")
    P = drop(P.race.eq("house") & ~P.state.isin(POSTAL), "house poll without a state")
    big = P.sample_size > C.POLL_MAX_SAMPLE
    P.loc[big, "sample_size"] = C.POLL_MAX_SAMPLE
    # nominee match-up only: per race keep the (D surname, R surname) pair most polled since mid-August
    P["dsn"], P["rsn"] = P.dem_name.map(surname), P.rep_name.map(surname)
    keep = pd.Series(True, index=P.index)
    for (r, st), g in P[P.race.isin(["sen", "gov"])].groupby(["race", "state"]):
        late = g[g.end_date >= "2026-08-15"]
        base = late if len(late) else g
        pair = base.groupby(["dsn", "rsn"]).size().idxmax() if base[["dsn", "rsn"]].notna().all(axis=1).any() else None
        if pair is not None:
            keep[g.index] = (g.dsn == pair[0]) & (g.rsn == pair[1])
    P = drop(~keep, "not the nominee match-up")
    # duplicates across sources: same race, place, pollster, end date -> keep the best population, then best source
    pop_rank = P.population.astype(str).str.lower().map({"lv": 0, "rv": 1, "v": 2, "a": 3}).fillna(4)
    src_rank = P.source.map({"votehub": 0, "wikipedia": 1}).fillna(2)
    P = P.assign(_pr=pop_rank, _sr=src_rank, _pk=P.pollster.map(_norm_pollster),
                 _dk=P.district.fillna(-1).astype(float))
    n_before = len(P)
    P = P.sort_values(["_pr", "_sr"]).drop_duplicates(["race", "state", "_dk", "_pk", "end_date"])
    if n_before - len(P):
        drops["duplicate"] = drops.get("duplicate", 0) + n_before - len(P)
    # outliers: margin far from the median of the same race (>= 4 polls)
    P["margin"] = P.dem_pct - P.rep_pct
    med = P.groupby(["race", "state", "_dk"]).margin.transform("median")
    cnt = P.groupby(["race", "state", "_dk"]).margin.transform("size")
    P = drop((cnt >= 4) & ((P.margin - med).abs() > C.POLL_OUTLIER_PTS), "outlier vs other polls of the race")
    P = P.drop(columns=[c for c in P.columns if c.startswith("_")] + ["dsn", "rsn", "margin"])
    log.append(f"polls: {n0} rows in, {len(P)} kept; dropped: " + (", ".join(f"{k} {v}" for k, v in drops.items()) or "none"))
    return P.reset_index(drop=True), drops


def gather(bundle, today, log):
    """All 2026 polls: bundle + fresh pulls. Never fails: a source that errors is logged and skipped."""
    w = bundle["world"]
    up = bundle["up"]
    states = list(w["states"])
    up_states = {r: {states[s] for s in range(len(states)) if up[s, ri]} for ri, r in enumerate(bundle["races"])}
    parts = []
    B = bundle["polls_2026"].copy()
    B = B[B.race.isin(["sen", "gov", "house", "generic"])].reindex(columns=TIDY_COLS + ["hypothetical"])
    if "hypothetical" in B:
        B = B[~B.hypothetical.astype(str).str.lower().isin(["true", "1"])]
    parts.append(B.reindex(columns=TIDY_COLS))
    fresh = {}
    for name, fn in (("votehub generic ballot", votehub_generic), ("wikipedia races", lambda: wikipedia_races(up_states, log))):
        try:
            d = fn()
            fresh[name] = len(d)
            parts.append(d)
        except Exception as e:
            log.append(f"{name} FAILED: {e} (using what the bundle already has)")
            fresh[name] = -1
    P = pd.concat(parts, ignore_index=True)
    P, drops = clean(P, up_states, today, log)
    approval = None
    try:
        A = votehub_approval()
        A = A[(A.end_date >= "2026-08-01") & (A.end_date <= "2026-11-01") & A.approve.between(20, 70)]
        if len(A) >= 5:
            approval = float(A.approve.mean())
            log.append(f"approval: {approval:.1f}% from {len(A)} VoteHub polls since Aug 1")
    except Exception as e:
        log.append(f"approval FAILED: {e}")
    return P, dict(fresh=fresh, drops=drops, approval=approval,
                   by_race=P.groupby("race").size().to_dict(),
                   newest=str(P.end_date.max().date()) if len(P) else None)


def poll_obs(P, bundle):
    """Tidy polls -> observation rows (identical maths to the notebook's poll_obs)."""
    w = bundle["world"]
    states = list(w["states"])
    s_idx = {s: i for i, s in enumerate(states)}
    sp, npr = np.asarray(bundle["state_prior"]), float(bundle["national_prior"])
    if not len(P):
        return pd.DataFrame(columns=OBS_COLS)
    days = (pd.Timestamp(C.ELECTION_DATE) - pd.to_datetime(P.end_date)).dt.days.values
    v = (P.dem_pct / (P.dem_pct + P.rep_pct)).values
    n = (P.sample_size.fillna(C.POLL_DEFAULT_SAMPLE).clip(100, 20000) * (P.dem_pct + P.rep_pct) / 100).values
    t = -24.0 * days - 1
    out = []
    dist_na = P.district.isna().values if "district" in P else np.ones(len(P), bool)
    for race in ("sen", "gov"):
        m = ((P.race == race) & (P.state != "US") & P.state.isin(s_idx)).values & dist_na
        if m.any():
            u = P.state[m].map(s_idx).values
            out.append(_obs("state", u, f"poll_{race}", t[m], v[m], n[m], sp[u]))
    m = ((P.race == "generic") & (P.state == "US")).values
    if m.any():
        out.append(_obs("nation", np.zeros(m.sum()), "poll_generic", t[m], v[m], n[m], np.full(m.sum(), npr)))
    m = ((P.race == "house") & P.state.isin(s_idx)).values
    if m.any():
        x = bundle["house_xw"]
        single = x.groupby("state_po").district.nunique()
        ph = P[m].copy().reset_index(drop=True)
        ph["district"] = pd.to_numeric(ph.district, errors="coerce").fillna(0).astype(int)
        ph.loc[ph.state.map(single).eq(1), "district"] = 0
        ph = ph.assign(v=v[m], n=n[m], t=t[m])
        j = ph.merge(x, left_on=["state", "district"], right_on=["state_po", "district"])
        if len(j):
            dp = np.array([bundle["dprior"].get(f"{a}|{b}", np.nan) for a, b in zip(j.state, j.district)])
            ok = np.isfinite(dp)
            out.append(_obs("county", j.u.values[ok], "poll_house", j.t.values[ok], j.v.values[ok],
                            (j.n * j.share_dc).values[ok], dp[ok]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=OBS_COLS)


def poll_summary(P):
    """Per race-state: count and latest date (for the drivers' detail text)."""
    out = {}
    for (r, st), g in P.groupby(["race", "state"]):
        out[(r, st)] = dict(n=len(g), latest=str(pd.to_datetime(g.end_date).max().date()),
                            avg=float((g.dem_pct - g.rep_pct).tail(10).mean()))
    for (st, d), g in P[P.race == "house"].groupby(["state", P.district.fillna(0).astype(int)]):
        out[("house", f"{st}-{d}")] = dict(n=len(g), latest=str(pd.to_datetime(g.end_date).max().date()))
    return out
