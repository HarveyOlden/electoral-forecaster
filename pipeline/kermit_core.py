"""KERMIT v7 inference core: the model, the featuriser and the Bayesian baseline, copied verbatim from the v7
notebook cells (c01/c04/c10/c11/c12) so GitHub runs exactly the code that was validated. Do not edit by hand:
fix things in the notebook and re-copy. Training-only code is left out."""
import math, dataclasses as dc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CENSOR_LOGIT = 4.0
N_LATENT, N_FACTORS = 8, 8

# ---------------- c04: constants, World, priors, observations, featuriser ----------------
GROUPS = ["noncollege_white", "college_white", "black", "hispanic", "other"]
MODES = ["eday", "early", "mail"]
RACES = ["pres", "sen", "gov", "house"]
R_ = len(RACES)
# Sources = (race x kind). Night counts are the TOP race on the ballot (president, or House in midterms):
# the only race with real count data, and the ballots are the same for every race anyway.
XGROUPS = ["white", "black", "hispanic", "other"]       # demographic-poll groups
XG_W = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], float)   # XGROUPS x GROUPS
SOURCES = (["poll_pres", "poll_sen", "poll_gov", "poll_house", "poll_generic", "ev_read",
            "count_eday", "count_early", "count_mail"] + [f"xtab_{g}" for g in XGROUPS] + [f"appr_{g}" for g in XGROUPS]
           + ["special"])
SOURCE_RACE = [0, 1, 2, 3, 3, -1, -1, -1, -1] + [-1] * 8 + [3]   # which race each source is about (-1 = ballot-level)
SOURCE_KIND = [0, 0, 0, 0, 1, 2, 3, 3, 3] + [4] * 4 + [5] * 4 + [6]   # poll / generic / early vote / count / crosstab / group approval / special election
LEVEL_SOURCES = dict(county=["poll_house", "ev_read", "count_eday", "count_early", "count_mail"],
                     state=["poll_pres", "poll_sen", "poll_gov"],
                     nation=["poll_pres", "poll_generic"] + [f"xtab_{g}" for g in XGROUPS] + [f"appr_{g}" for g in XGROUPS]
                     + ["special"])
LEVEL_IDX = {lv: [SOURCES.index(s) for s in ss] for lv, ss in LEVEL_SOURCES.items()}
DEMO_SOURCES = [f"xtab_{g}" for g in XGROUPS] + [f"appr_{g}" for g in XGROUPS]


def scol(level, src):
    """column of a source inside a level's dossier"""
    return LEVEL_SOURCES[level].index(src)
CUMULATIVE = {"ev_read"}
TIME_EDGES = np.array([-2880, -1440, -720, -336, -168, -72, -24, 0, 1, 2, 3, 4, 6, 9, 12, 24, 48, 96, 1e9], float)
CHANNELS = ["residual", "se", "n_obs", "observed", "coverage"]
RACE_STATIC_COLS = ["up", "contested", "inc", "last_resid", "has_last", "has_inc", "last_unc", "last2_resid", "has_last2",
                    "money_ratio", "coh_ratio", "has_money"]
# v7: House districts are units of their own. What the model is told about each district (known before the election):
DIST_STATIC_COLS = ["contested", "inc", "has_inc", "last_resid", "has_last", "last_unc", "last2_resid", "has_last2",
                    "money_ratio", "coh_ratio", "has_money", "dprior", "log_votes"]
DS_IDX = {c: k for k, c in enumerate(DIST_STATIC_COLS)}
STATIC_COLS = (["prior_logit", "log_turnout_exp"] + [f"g_{k}" for k in GROUPS] + [f"glag_{k}" for k in GROUPS]
               + [f"dg_{k}" for k in GROUPS] + ["pop_growth", "lag_gap_years"]
               + [f"modeshare_{m}" for m in MODES] + [f"modeoff_{m}" for m in MODES] + ["close_hour"]
               + ["lag_swing", "has_lag2", "h_mexican", "h_pr", "h_cuban"])
CONTEXT_COLS = ["incumbent_dem", "midterm", "approval", "unemp_change", "inflation", "cutoff_hours",
                "house_maj_dem", "senate_maj_dem"]
# governor schedule (even years): these 11 vote in presidential years; NH + VT every two years; 5 vote in odd years
GOV_PRES_YEAR = {"DE", "IN", "MO", "MT", "NH", "NC", "ND", "UT", "VT", "WA", "WV"}
GOV_EVERY_2 = {"NH", "VT"}
GOV_ODD_YEAR = {"KY", "LA", "MS", "NJ", "VA"}


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dc.dataclass
class Bugs:
    independent_errors: bool = False
    inconsistent_ev: bool = False
    leak_mode_share: bool = False
    poll_selection_on_outcome: bool = False
    percent_units: bool = False
    time_leak: bool = False
    no_renormalise: bool = False
    double_count_cumulative: bool = False


@dc.dataclass
class World:
    state_of: np.ndarray
    groups: np.ndarray
    turnout_exp: np.ndarray          # expected votes in a PRESIDENTIAL year (midterms scale it down per instance)
    prior_logit: np.ndarray
    mode_share_prior: np.ndarray
    mode_offset_prior: np.ndarray
    mode_order: list
    close_hour: np.ndarray
    demo_pairs: list = None
    group_lean: np.ndarray = None
    district_of: np.ndarray = None   # [C] synthetic House district cluster (global id)
    states: list = None
    county_fips: np.ndarray = None
    lag_swing: np.ndarray = None     # [C] previous presidential swing (prior minus the result before it), logit
    has_lag2: np.ndarray = None      # [C] 1 if that earlier result exists
    hisp_origin: np.ndarray = None   # [C, 3] Mexican / Puerto Rican / Cuban shares of population
    lag_swings: list = None          # real county swing patterns from consecutive presidential pairs (synthetic draws)
    dmaps: list = None               # v7: district maps (pieces of counties); synthetic elections draw one per election
    dmap: dict = None                # v7: the map of a real election

    @property
    def n_states(self):
        return len(self.mode_order)

    @property
    def n_counties(self):
        return len(self.state_of)

    @property
    def n_districts(self):
        return int(self.district_of.max()) + 1 if self.district_of is not None else self.n_states



CAL = dict(
    race_nat_sd=dict(sen=0.04, gov=0.07, house=0.03),     # race's national deviation from the presidential environment
    race_state_sd=dict(sen=0.15, gov=0.25, house=0.06),   # candidate / state-party effect per state race
    race_cty_sd=dict(sen=0.06, gov=0.09, house=0.12),     # county-level race-specific noise
    house_cluster_frac=0.6,                               # share of House county variance that is district-wide
    inc_beta=dict(sen=0.06, gov=0.08, house=0.10),        # incumbency advantage (logit)
    p_open=dict(sen=0.2, gov=0.35, house=0.12),
    persist=dict(sen=0.4, gov=0.3, house=0.6),            # how much of the last race's effect carries over (incumbent running)
    env_shift_sd=0.10,                                    # national environment difference: last race vs prior year
    uncontested=dict(sen=0.02, gov=0.01, house=0.08),     # share of races with one major party absent
    house_weak=0.10,                                      # share of districts with a paper challenger
    extreme_prob=0.15,                                    # share of elections with heavy-tailed county noise
    poll_sys_sd=0.04,                                     # industry-wide poll miss (logit), shared by every race
    poll_state_sd=dict(pres=0.03, sen=0.05, gov=0.06, house=0.08, generic=0.02),
    poll_rate=dict(pres=10.0, sen=8.0, gov=6.0, house=0.25, generic=60.0, pres_nat=60.0),
    midterm_turnout=0.72,
    p_maj_same=0.55,                                      # P(chamber majority = president's party) before the election
    trend_persist=0.3,                                    # share of a county's previous relative swing that repeats
    trend_persist_sd=0.2,                                 # ...randomised per election, so the model must read it off the data
    origin_swing_sd=0.2,                                  # Mexican / Puerto Rican / Cuban-specific swings (logit)
    xtab_cycle_sd=0.12,                                   # shared miss of all crosstabs of one group in one cycle (logit)
    xtab_pollster_sd=0.12,                                # per-poll house effect (logit; not measurable from CES alone)
    xtab_deff=1.7,                                        # design effect of weighted subsamples
    xtab_rate=1.5, appr_rate=1.0,                         # crosstabs / group-approval readings per group per election
    # v7
    piece_sd=0.4,                                         # spread of district pieces' lean inside one county (logit)
    p_newmap=0.25,                                        # share of elections on new district lines (no district history)
    money=dict(house=dict(beta=1.0, gamma=0.5, sd=0.5, coh_rho=0.7, coh_sd=0.4, p=0.9),   # log D/R money (/3) vs the
               sen=dict(beta=1.0, gamma=0.4, sd=0.4, coh_rho=0.7, coh_sd=0.4, p=0.95)),   # race's own effect + incumbency
    special_rate=6.0,                                     # House special elections per cycle
    special_sd=0.25,                                      # one special's noise around the national shift (logit)
    special_cycle_sd=0.10,                                # shared miss of all specials in one cycle (logit)
)




def agg_logit(lg, wts, unit_of, n_units, mask=None):
    m = np.ones(len(lg), bool) if mask is None else mask
    v = np.bincount(unit_of[m], wts[m], n_units)
    d = np.bincount(unit_of[m], wts[m] * sig(lg[m]), n_units)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v > 0, logit(d / np.maximum(v, 1e-12)), np.nan)


def state_prior(w: World):
    return agg_logit(w.prior_logit, w.turnout_exp, w.state_of, w.n_states)


def national_prior(w: World):
    return float(agg_logit(w.prior_logit, w.turnout_exp, np.zeros(w.n_counties, int), 1)[0])



OBS_COLS = ["unit_level", "unit", "source", "t", "value", "weight", "expected_logit", "mode_expected_votes"]


def _obs(level, unit, source, t, value, weight, expected, mev=np.nan):
    n = len(unit)
    return pd.DataFrame(dict(unit_level=[level] * n, unit=np.asarray(unit, int), source=[source] * n, t=t, value=value,
                             weight=weight, expected_logit=expected, mode_expected_votes=mev))



# ---------------- featuriser: observations -> dossiers (same code runs on real data) ----------------
def featurise(w: World, obs: pd.DataFrame, cutoff: float, bugs: Bugs, turnout_exp=None):
    """Dossiers for counties [C, Src, T, Ch], states [S, ...] and the nation [1, ...]."""
    Texp = w.turnout_exp if turnout_exp is None else turnout_exp
    limit = cutoff + (2.0 if bugs.time_leak else 0.0)
    o = obs[obs.t <= limit].copy()
    if "mode_expected_votes" not in o:
        o["mode_expected_votes"] = np.nan
    o["bin"] = np.clip(np.searchsorted(TIME_EDGES, o.t.values, side="right") - 1, 0, len(TIME_EDGES) - 2)
    if not bugs.double_count_cumulative and len(o):
        cum = o.source.isin(CUMULATIVE)
        latest = o[cum].sort_values("t").groupby(["unit_level", "unit", "source", "bin"]).tail(1)
        o = pd.concat([o[~cum], latest])
    o["resid"] = logit(o.value.values.astype(float)) - o.expected_logit.values.astype(float)
    p = np.clip(o.value.values.astype(float), 1e-3, 1 - 1e-3)
    o["prec"] = o.weight.values.astype(float) * p * (1 - p)
    o["wr"] = o.resid * o.prec
    out = {}
    for level, n_units in [("county", w.n_counties), ("state", w.n_states), ("nation", 1)]:
        srcs = LEVEL_SOURCES[level]
        D = np.zeros((n_units, len(srcs), len(TIME_EDGES) - 1, len(CHANNELS)), np.float32)
        sub = o[(o.unit_level == level) & o.source.isin(srcs)]
        if len(sub):
            g = sub.groupby(["unit", "source", "bin"]).agg(wr=("wr", "sum"), prec=("prec", "sum"), n=("resid", "size"),
                                                           votes=("weight", "sum")).reset_index()
            u, b = g.unit.astype(int).values, g.bin.astype(int).values
            si = g.source.map(srcs.index).values
            prec = np.maximum(g.prec.values, 1e-12)
            D[u, si, b, 0] = g.wr.values / prec
            D[u, si, b, 1] = 1 / np.sqrt(prec)
            D[u, si, b, 2] = g.n.values
            D[u, si, b, 3] = 1.0
            if level == "county":
                isc = g.source.str.startswith("count_").values
                D[u[isc], si[isc], b[isc], 4] = g.votes.values[isc] / Texp[u[isc]]
                D[..., 4] = np.cumsum(D[..., 4], axis=2)
        out[level] = D
    audit = dict(max_t_included=float(o.t.max()) if len(o) else -np.inf, cutoff=cutoff)
    return out, audit




# ---------------- c10: model ----------------
DEV = "cpu"
AMP = torch.bfloat16 if DEV == "cuda" else torch.float32
N_SRC, N_T, N_RS, N_DS = len(SOURCES), len(TIME_EDGES) - 1, len(RACE_STATIC_COLS), len(DIST_STATIC_COLS)
# token types: global | states | counties | House districts (v7). Relation types between any two tokens:
REL_NAMES = ["unrelated", "self", "global", "county_county_same_state", "county_own_state", "county_other_state",
             "state_state", "dist_dist_same_state", "dist_dist_other", "dist_own_state", "dist_other_state",
             "county_dist_piece", "county_dist_same_state", "county_dist_other"]
N_REL = len(REL_NAMES)


def build_relations(w, gaz, adj_pairs, dm=None):
    """Relation of every token pair: a type (embedding per head), county adjacency, log distance, and for county <->
    district pairs the piece weight (district -> county: share of the district's votes in that county; county ->
    district: share of the county's votes in that district)."""
    S, C = w.n_states, w.n_counties
    D = dm["n_d"] if dm is not None else 0
    N = 1 + S + C + D
    oc, od = 1 + S, 1 + S + C
    so = np.asarray(w.state_of)
    T = {n: k for k, n in enumerate(REL_NAMES)}
    typ = np.zeros((N, N), np.uint8)
    typ[0, :], typ[:, 0] = T["global"], T["global"]
    typ[1:oc, 1:oc] = T["state_state"]
    typ[oc:od, oc:od] = np.where(so[:, None] == so[None, :], T["county_county_same_state"], T["unrelated"])
    typ[oc:od, 1:oc] = np.where(so[:, None] == np.arange(S)[None, :], T["county_own_state"], T["county_other_state"])
    typ[1:oc, oc:od] = typ[oc:od, 1:oc].T
    piece = np.zeros((N, N), np.float16)
    if D:
        ds_ = np.asarray(dm["d_state"])
        typ[od:, od:] = np.where(ds_[:, None] == ds_[None, :], T["dist_dist_same_state"], T["dist_dist_other"])
        typ[od:, 1:oc] = np.where(ds_[:, None] == np.arange(S)[None, :], T["dist_own_state"], T["dist_other_state"])
        typ[1:oc, od:] = typ[od:, 1:oc].T
        blk = np.where(so[:, None] == ds_[None, :], T["county_dist_same_state"], T["county_dist_other"]).astype(np.uint8)
        blk[dm["pc"], dm["pd"]] = T["county_dist_piece"]
        typ[oc:od, od:] = blk
        typ[od:, oc:od] = blk.T
        piece[oc + dm["pc"], od + dm["pd"]] = dm["wcd"]
        piece[od + dm["pd"], oc + dm["pc"]] = dm["wdc"]
    np.fill_diagonal(typ, T["self"])
    adj = np.zeros((N, N), bool)
    f2u = {f: i for i, f in enumerate(w.county_fips)}
    for a, b in adj_pairs:
        if a in f2u and b in f2u and a != b:
            adj[oc + f2u[a], oc + f2u[b]] = True
    ld = np.zeros((N, N), np.float16)
    if gaz is not None:
        g = gaz.reindex(w.county_fips)
        lat, lon = np.radians(g.lat.fillna(g.lat.mean()).values), np.radians(g.lon.fillna(g.lon.mean()).values)

        def hav(la, lo):
            a_ = np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2
            return np.log1p(2 * 6371 * np.arcsin(np.sqrt(np.clip(a_, 0, 1))) / 50)
        if D:                                                       # district centre = vote-weighted county centres
            lat = np.concatenate([lat, np.bincount(dm["pd"], dm["wdc"] * lat[dm["pc"]], D)])
            lon = np.concatenate([lon, np.bincount(dm["pd"], dm["wdc"] * lon[dm["pc"]], D)])
        ld[oc:, oc:] = hav(lat, lon).astype(np.float16)
        ld[1:oc, 1:oc] = hav(np.array([lat[:C][so == s].mean() for s in range(S)]),
                             np.array([lon[:C][so == s].mean() for s in range(S)])).astype(np.float16)
    t_ = lambda a, dt: torch.tensor(a, device=DEV).to(dt)
    return dict(typ=t_(typ, torch.uint8), adj=t_(adj, torch.bool), logdist=t_(ld, torch.float16), piece=t_(piece, torch.float16),
                state_of=torch.tensor(so, dtype=torch.long, device=DEV), n_states=S, n_dist=D)


class MaskedSelfAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, mask=None, bias=None):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        am = None
        if mask is not None:
            am = (mask | ~mask.any(-1, keepdim=True))[:, None, None, :]
        if bias is not None:
            am = bias.unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(out.transpose(1, 2).reshape(B, L, d))


class FFN(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.net(x)


class AxialBlock(nn.Module):
    """Attention along each row (one source over time), then down each column (all sources at one time)."""
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.ln_t, self.att_t = nn.LayerNorm(d), MaskedSelfAttn(d, h)
        self.ln_s, self.att_s = nn.LayerNorm(d), MaskedSelfAttn(d, h)
        self.ffn = FFN(d, drop)

    def forward(self, x, m):
        U, S, T, d = x.shape
        xt = x.reshape(U * S, T, d)
        x = (xt + self.att_t(self.ln_t(xt), m.reshape(U * S, T))).reshape(U, S, T, d)
        xs = x.permute(0, 2, 1, 3).reshape(U * T, S, d)
        x = (xs + self.att_s(self.ln_s(xs), m.permute(0, 2, 1).reshape(U * T, S))).reshape(U, T, S, d).permute(0, 2, 1, 3)
        return self.ffn(x)


class RowEncoder(nn.Module):
    """Reads one dossier (race x source rows, time bins, 5 channels) into one vector. Each row's embedding is
    its own + its race's + its kind's (poll / generic / early vote / count), so e.g. Senate and governor polls
    share what 'a poll' means while staying distinguishable."""
    def __init__(self, d, h, n_blocks=2, drop=0.1):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.src_emb = nn.Parameter(torch.randn(N_SRC, d) * 0.02)
        self.race_emb = nn.Parameter(torch.randn(R_ + 1, d) * 0.02)
        self.kind_emb = nn.Parameter(torch.randn(max(SOURCE_KIND) + 1, d) * 0.02)
        self.register_buffer("src_race", torch.tensor([r if r >= 0 else R_ for r in SOURCE_RACE]))
        self.register_buffer("src_kind", torch.tensor(SOURCE_KIND))
        for lv, ix in LEVEL_IDX.items():                  # each level only carries the sources that can occur there
            self.register_buffer(f"idx_{lv}", torch.tensor(ix))
        self.time_emb = nn.Parameter(torch.randn(N_T, d) * 0.02)
        self.blocks = nn.ModuleList([AxialBlock(d, h, drop) for _ in range(n_blocks)])
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.null = nn.Parameter(torch.zeros(d))
        self.ln = nn.LayerNorm(d)

    @staticmethod
    def prep(D):
        obs = D[..., 3]
        r = D[..., 0].clamp(-3, 3) * obs
        se = ((torch.log(D[..., 1].clamp_min(1e-6)) + 6) / 3).clamp(-3, 3) * obs
        n = torch.log1p(D[..., 2].clamp_min(0)) / 2
        return torch.stack([r, se, n, obs, D[..., 4].clamp(0, 2)], -1), obs > 0.5

    def forward(self, D, level):
        X, m = self.prep(D)
        emb = (self.src_emb + self.race_emb[self.src_race] + self.kind_emb[self.src_kind])[getattr(self, f"idx_{level}")]
        x = self.cell(X) + emb[None, :, None] + self.time_emb[None, None]
        for b in self.blocks:
            x = b(x, m)
        U, S, T, d = x.shape
        xf, mf = self.ln(x).reshape(U, S * T, d), m.reshape(U, S * T)
        sc = ((xf @ self.q) / math.sqrt(d)).masked_fill(~mf, -1e4)
        pooled = (torch.softmax(sc.float(), -1).to(xf.dtype)[..., None] * xf).sum(1)
        return torch.where(mf.any(-1, keepdim=True), pooled, self.null.to(pooled.dtype))


class MixerLayer(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.ln, self.att, self.ffn = nn.LayerNorm(d), MaskedSelfAttn(d, h), FFN(d, drop)

    def forward(self, x, bias):
        return self.ffn(x + self.att(self.ln(x), bias=bias))


class LatentSwing(nn.Module):
    def __init__(self, d, h, n_latent):
        super().__init__()
        self.lat = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.read = nn.MultiheadAttention(d, h, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, h, batch_first=True)
        self.write = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        z = self.lat[None].to(x.dtype)
        xn = self.ln1(x)
        z = z + self.read(z, xn, xn, need_weights=False)[0]
        zn = self.ln2(z)
        z = z + self.self_(zn, zn, zn, need_weights=False)[0]
        return x + self.write(self.ln3(x), z, z, need_weights=False)[0]


class Kermit(nn.Module):
    """Outputs per county AND race: mu (shift vs the frozen prior), sig (own noise) and L (loadings on K shared
    factors). The factors are shared across races, so the joint covariance couples e.g. a county's Senate and
    governor results, and a whole state's races, the way a common environment does."""
    def __init__(self, n_static, d=64, h=4, n_mix=3, n_latent=8, n_factors=8, dropout=0.1):
        super().__init__()
        self.d, self.K = d, n_factors
        n_in = n_static + R_ * N_RS
        self.hi = RACES.index("house")
        self.tok_drop = nn.Dropout(dropout)
        self.register_buffer("st_mu", torch.zeros(n_static))
        self.register_buffer("st_sd", torch.ones(n_static))
        self.rows = RowEncoder(d, h, drop=dropout)
        self.static_c = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.static_s = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.Sequential(nn.Linear(len(CONTEXT_COLS), d), nn.GELU(), nn.Linear(d, d))
        self.type_unit = nn.Parameter(torch.randn(4, d) * 0.02)
        self.rel_type = nn.Parameter(torch.zeros(N_REL, h))
        self.rel_piece = nn.Parameter(torch.full((h,), 2.0))      # county <-> its district pieces attend to each other
        self.static_d = nn.Sequential(nn.Linear(N_DS, d), nn.GELU(), nn.Linear(d, d))
        self.dstat = nn.Sequential(nn.Linear(N_DS, d), nn.GELU(), nn.Linear(d, d))
        self.rel_adj = nn.Parameter(torch.zeros(h))
        self.rel_dist = nn.Parameter(torch.full((h,), -0.5))
        self.mix = nn.ModuleList([MixerLayer(d, h, dropout) for _ in range(n_mix)])
        self.latent = LatentSwing(d, h, n_latent)
        self.race_out = nn.Parameter(torch.randn(R_, d) * 0.02)
        self.race_stat = nn.Sequential(nn.Linear(N_RS, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 + n_factors))
        self.thead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    @staticmethod
    def ctx_feats(c):
        cut = c[5]
        return torch.stack([c[0], c[1], (c[2] - 45) / 10, c[3], (c[4] - 3) / 3,
                            torch.sign(cut) * torch.log1p(cut.abs()) / 5, c[6], c[7]])

    def forward(self, inst, rel):
        so, S = rel["state_of"], rel["n_states"]
        C = inst["static"].shape[0]
        st = (inst["static"] - self.st_mu) / self.st_sd
        rs = inst["race_static"].clone()
        rs[..., 3] = rs[..., 3].clamp(-4, 4)
        stc = torch.cat([st, rs.reshape(C, -1)], 1)
        Tw = torch.exp(inst["static"][:, 1:2])
        st_state = torch.zeros(S, stc.shape[1], device=stc.device).index_add_(0, so, stc * Tw)
        st_state = st_state / torch.zeros(S, 1, device=stc.device).index_add_(0, so, Tw).clamp_min(1e-9)
        ds = inst["dist_static"]
        with torch.autocast(device_type="cuda", dtype=AMP, enabled=(DEV == "cuda")):
            c_tok = self.rows(inst["county"], "county") + self.static_c(stc) + self.type_unit[2]
            s_tok = self.rows(inst["state"], "state") + self.static_s(st_state) + self.type_unit[1]
            g_tok = self.ctx(self.ctx_feats(inst["context"]))[None] + self.rows(inst["nation"], "nation") + self.type_unit[0]
            d_tok = self.static_d(ds) + self.type_unit[3]
            x = self.tok_drop(torch.cat([g_tok, s_tok, c_tok.to(g_tok.dtype), d_tok.to(g_tok.dtype)], 0)[None])
            bias = (F.embedding(rel["typ"].int(), self.rel_type.float()).permute(2, 0, 1)
                    + self.rel_adj[:, None, None] * rel["adj"][None] + self.rel_dist[:, None, None] * rel["logdist"][None].float()
                    + self.rel_piece[:, None, None] * rel["piece"][None].float())
            for i, layer in enumerate(self.mix):
                x = layer(x, bias)
                if i == len(self.mix) - 2:
                    x = self.latent(x)
            hc = x[0, 1 + S:1 + S + C]
            hd = x[0, 1 + S + C:]
            z = hc[:, None, :] + self.race_out[None].to(hc.dtype) + self.race_stat(rs.to(hc.dtype))
            o = self.head(z).float()
            t = self.thead(hc).float()
            od = self.head(hd + self.race_out[self.hi].to(hd.dtype) + self.dstat(ds.to(hd.dtype))).float()
        return dict(mu=o[..., 0], sig=F.softplus(o[..., 1]) + 1e-2, L=o[..., 2:] * 0.1,
                    tmu=t[:, 0], tsig=F.softplus(t[:, 1]) + 1e-3,
                    mu_d=od[:, 0], sig_d=F.softplus(od[:, 1]) + 1e-2, L_d=od[:, 2:] * 0.1)




def to_dev(inst, nation_key="nation"):
    """nation_key picks the national dossier variant of a real election: "nation" (with CES demographic polls,
    training), "nation_nodemo" (main evaluation), "nation_smalldemo" (2026-sized noisy demographic polls)."""
    f = lambda k, dt=torch.float32: torch.tensor(np.asarray(inst[k]), dtype=dt, device=DEV)
    return dict(county=f("county"), state=f("state"), nation=f(nation_key if nation_key in inst else "nation"),
                static=f("static"), race_static=f("race_static"), context=f("context"), y=f("y"), y_cens=f("y_cens"),
                y_turnout=f("y_turnout"),
                dist_static=f("dist_static") if "dist_static" in inst else torch.zeros(0, N_DS, device=DEV),
                y_d=f("y_d") if "y_d" in inst else torch.zeros(0, device=DEV),
                ycens_d=f("ycens_d") if "ycens_d" in inst else torch.zeros(0, device=DEV))




@torch.no_grad()
def predict(model, inst, rel):
    model.eval()
    return {k: v.float().cpu().numpy() for k, v in model(inst, rel).items()}



# ---------------- c11: Bayesian baseline + district helpers ----------------
def fund_prior(ctx, fit):
    S, mu = fit["cov"], fit["mu"]
    f = np.array([ctx[3], ctx[4], ctx[2]])
    A = S[1:, 1:] + 1e-6 * np.diag(np.maximum(np.diag(S[1:, 1:]), 1e-6))   # guard: a constant input makes S singular
    w_ = np.linalg.solve(A, S[0, 1:])
    e_var = max(S[0, 0] - S[0, 1:] @ w_, 1e-4)
    s_inc = float(np.clip((fit["mean_mid"] if ctx[1] else fit["mean_pres"]) + w_ @ (f - mu[1:]), -1.0, 1.0))
    return (s_inc if ctx[0] else -s_inc), e_var


def dossier_avg(D, level, src):
    """Precision-weighted mean residual of one source over all time bins (per unit) -> (mean [U], precision [U])."""
    k = scol(level, src)
    cells = D[:, k]
    obs = cells[..., 3] > 0.5
    prec = np.where(obs, 1 / np.maximum(cells[..., 1], 1e-6) ** 2, 0.0)
    P = prec.sum(1)
    return np.where(P > 0, (prec * cells[..., 0]).sum(1) / np.maximum(P, 1e-12), 0.0), P


RACE_X = lambda RS: np.stack([RS[..., 2] * RS[..., 5], RS[..., 3] * RS[..., 4], RS[..., 4]], -1)   # inc, last, has_last


def fit_race_coefs(insts):
    """Race offsets from REAL training elections: (race result - prior - year's national mean) ~ incumbency +
    last result. Ridge-regularised; one row per county result."""
    coefs = {r: np.zeros(3) for r in RACES}
    seen = set()
    by = {r: ([], []) for r in RACES}
    for i in insts:
        if i["year"] in seen or i["kind"] != "pre":
            continue
        seen.add(i["year"])
        prior = i["static"][:, 0]
        X = RACE_X(i["race_static"])
        for ri, r in enumerate(RACES[1:], start=1):
            y = i["y"][:, ri]
            m = np.isfinite(y) & (np.abs(np.nan_to_num(y)) < CENSOR_LOGIT) & (i["race_static"][:, ri, 0] > 0)
            if m.sum() < 30:
                continue
            d = y[m] - prior[m]
            d = d - np.average(d, weights=np.exp(i["static"][m, 1]))
            by[r][0].append(X[m, ri])
            by[r][1].append(d)
    for r, (xs, ds) in by.items():
        if xs:
            X, d = np.concatenate(xs), np.concatenate(ds)
            coefs[r] = np.linalg.solve(X.T @ X + 10.0 * np.eye(3), X.T @ d)
    return coefs


def bayes_counts_top(D, static, ctx, state_of, n_states, fit, sig_rep=0.06, sig_state=0.09, sig_cty=0.08, sig_mode=0.10):
    """v3 election-night baseline for the top race: counted votes are fact, outstanding vote predicted mode by mode."""
    C = len(state_of)
    prior = static[:, 0].astype(float)
    Texp = np.exp(static[:, 1].astype(float))
    off = static[:, [STATIC_COLS.index(f"modeoff_{m}") for m in MODES]].astype(float)
    msh = static[:, [STATIC_COLS.index(f"modeshare_{m}") for m in MODES]].astype(float)
    ci = [scol("county", f"count_{m}") for m in MODES]
    cnt = D[:, ci].astype(float)
    obs = cnt[..., 3] > 0.5
    cov = np.maximum.accumulate(np.where(obs, cnt[..., 4], 0.0), axis=2)
    dv = np.where(obs, np.diff(np.concatenate([np.zeros((C, 3, 1)), cov], 2), axis=2) * Texp[:, None, None], 0.0)
    share = 1 / (1 + np.exp(-(cnt[..., 0] + prior[:, None, None] + off[:, :, None])))
    Vm = dv.sum(2)
    V, Dv = Vm.sum(1), (dv * share).sum((1, 2))
    r = np.where(V > 0, (dv * cnt[..., 0]).sum((1, 2)) / np.maximum(V, 1e-9), 0.0)
    m0, v0 = fund_prior(ctx, fit)
    W = V.sum() / Texp.sum()
    rbar, pd_ = ((V * r).sum() / V.sum(), W / ((1 - min(W, 0.999)) * sig_rep ** 2 + 1e-6)) if V.sum() > 0 else (0.0, 0.0)
    v_n = 1 / (1 / v0 + pd_)
    n_post = v_n * (m0 / v0 + rbar * pd_)
    s_post, v_s = np.zeros(n_states), np.full(n_states, sig_state ** 2)
    for s in range(n_states):
        m = state_of == s
        if V[m].sum() > 0:
            Fs = min(V[m].sum() / Texp[m].sum(), 0.999)
            ps = Fs / ((1 - Fs) * sig_rep ** 2 + 1e-6)
            v_s[s] = 1 / (1 / sig_state ** 2 + ps)
            s_post[s] = v_s[s] * ((V[m] * (r[m] - n_post)).sum() / V[m].sum()) * ps
    f = np.clip(V / Texp, 0, 1)
    base = n_post + s_post[state_of]
    swing_rem = base + f * (r - base)
    Vrem_m = np.maximum(Texp[:, None] * msh - Vm, 0.0)
    p_rem_m = 1 / (1 + np.exp(-(prior[:, None] + off + swing_rem[:, None])))
    Vrem = Vrem_m.sum(1)
    p_tot = (Dv + (Vrem_m * p_rem_m).sum(1)) / np.maximum(V + Vrem, 1e-9)
    mu = logit(np.clip(p_tot, 1e-4, 1 - 1e-4)) - prior
    rem = Vrem / np.maximum(V + Vrem, 1e-9)
    rem_floor = np.maximum(rem, 0.10)
    diag = (rem_floor * sig_cty) ** 2 + (rem * sig_mode) ** 2 + 0.015 ** 2
    L = np.zeros((C, 1 + n_states))
    L[:, 0] = np.sqrt(v_n) * rem_floor
    L[np.arange(C), 1 + state_of] = np.sqrt(v_s[state_of] + (0.5 * sig_mode) ** 2) * rem_floor
    return mu, L, diag


def bayes_predict(inst, state_of, n_states, fit, coefs, use_polls=True):
    C, S, so = len(state_of), n_states, state_of
    ctx, static, RS = inst["context"], inst["static"], inst["race_static"]
    Texp = np.exp(static[:, 1].astype(float))
    m0, v0 = fund_prior(ctx, fit)
    n_post, v_n = m0, v0
    if use_polls:
        P_tot, r_tot = 0.0, 0.0
        for src in ("poll_generic", "poll_pres"):
            r_, P_ = dossier_avg(inst["nation"], "nation", src)
            if P_[0] > 0:
                P_tot += P_[0]; r_tot += P_[0] * r_[0]
        if P_tot > 0:
            v_obs = 1 / P_tot + CAL["poll_sys_sd"] ** 2           # sampling + one shared industry-wide miss
            v_n = 1 / (1 / v0 + 1 / v_obs)
            n_post = v_n * (m0 / v0 + (r_tot / P_tot) / v_obs)
    X = RACE_X(RS)
    mu = np.zeros((C, R_))
    L = np.zeros((C, R_, 1 + S))
    diag = np.zeros((C, R_))
    for ri, race in enumerate(RACES):
        off = X[:, ri] @ coefs.get(race, np.zeros(3))
        nat_sd = 0.0 if race == "pres" else CAL["race_nat_sd"][race]
        st_sd = 0.07 if race == "pres" else float(np.sqrt(0.07 ** 2 + CAL["race_state_sd"][race] ** 2))
        cty_sd = 0.06 if race == "pres" else CAL["race_cty_sd"][race]
        s_post, v_s = np.zeros(S), np.full(S, st_sd ** 2)
        if use_polls and race in ("pres", "sen", "gov"):
            r_s, P_s = dossier_avg(inst["state"], "state", f"poll_{race}")
            off_s = np.bincount(so, Texp * off, S) / np.maximum(np.bincount(so, Texp, S), 1e-9)
            has = P_s > 0
            v_obs = np.where(has, 1 / np.maximum(P_s, 1e-12) + CAL["poll_state_sd"][race] ** 2, np.inf)
            k = np.where(has, st_sd ** 2 / (st_sd ** 2 + v_obs), 0.0)
            s_post = k * np.where(has, r_s - n_post - off_s, 0.0)
            v_s = st_sd ** 2 * (1 - k)
        c_dev = np.zeros(C)
        if use_polls and race == "house":
            r_c, P_c = dossier_avg(inst["county"], "county", "poll_house")
            has = P_c > 0
            v_obs = np.where(has, 1 / np.maximum(P_c, 1e-12) + CAL["poll_state_sd"]["house"] ** 2, np.inf)
            k = np.where(has, cty_sd ** 2 / (cty_sd ** 2 + v_obs), 0.0)
            c_dev = k * np.where(has, r_c - n_post - off, 0.0)
        mu[:, ri] = n_post + off + s_post[so] + c_dev
        L[:, ri, 0] = np.sqrt(v_n + nat_sd ** 2)
        L[np.arange(C), ri, 1 + so] = np.sqrt(v_s[so])
        diag[:, ri] = cty_sd ** 2
    top = 0 if ctx[1] < 0.5 else RACES.index("house")
    ci = [scol("county", f"count_{m}") for m in MODES]
    if (inst["county"][:, ci, :, 3] > 0.5).any():
        mt, Lt, dt = bayes_counts_top(inst["county"], static, ctx, so, S, fit)
        mu[:, top], L[:, top, :], diag[:, top] = mt, Lt, dt
    return mu, L, diag


def prior_predict(inst, state_of, n_states, fit):
    C, S = len(state_of), n_states
    L = np.zeros((C, R_, 1 + S))
    diag = np.zeros((C, R_))
    for ri, race in enumerate(RACES):
        nat_sd = 0.0 if race == "pres" else CAL["race_nat_sd"][race]
        L[:, ri, 0] = np.sqrt(fit["cov"][0, 0] + 0.01 + nat_sd ** 2)
        L[np.arange(C), ri, 1 + state_of] = 0.07 if race == "pres" else np.sqrt(0.07 ** 2 + CAL["race_state_sd"][race] ** 2)
        diag[:, ri] = 0.06 ** 2 if race == "pres" else CAL["race_cty_sd"][race] ** 2
    return np.zeros((C, R_)), L, diag




def _dist_labels(inst):
    yd, cd = np.asarray(inst["y_d"], float), np.asarray(inst["ycens_d"], float)
    lab = np.isfinite(yd) | (cd != 0)
    return yd, cd, lab


def kermit_district_draws(out_d, inst, n=2000, seed=0):
    """KERMIT's own district head: mean share and joint draws [n, D] (shared factors -> correlated districts)."""
    ds = np.asarray(inst["dist_static"])
    mu = ds[:, DS_IDX["dprior"]] + np.asarray(out_d["mu_d"])
    L, sg = np.asarray(out_d["L_d"]), np.asarray(out_d["sig_d"])
    r = np.random.default_rng(seed)
    lg = mu[None] + r.standard_normal((n, L.shape[1])) @ L.T + r.standard_normal((n, len(mu))) * sg[None]
    return sig(mu), sig(lg)


def county_to_district(mu, L, dg, inst, dm, n=2000, seed=0):
    """Any county-level model (e.g. Bayes) -> districts: district's last result + the swing of its county pieces
    (weighted by the district's votes in each), or the pieces summed when the district has no history on these lines."""
    hi = RACES.index("house")
    ds, rs = np.asarray(inst["dist_static"]), np.asarray(inst["race_static"])
    prior = np.asarray(inst["static"])[:, 0]
    C, D = len(prior), dm["n_d"]
    pc, pdi, wdc = dm["pc"], dm["pd"], dm["wdc"]
    c_has = (rs[:, hi, 4] > 0.5) & (rs[:, hi, 6] == 0)
    p_prev = sig(prior + rs[:, hi, 3])
    a, M = np.zeros(D), np.zeros((D, C))
    for k in range(D):
        sel = pdi == k
        cs, w_ = pc[sel], wdc[sel]
        if ds[k, DS_IDX["has_last"]] > 0.5 and ds[k, DS_IDX["last_unc"]] == 0 and c_has[cs].any():
            ws = w_ * c_has[cs]
            ws = ws / ws.sum()
            a[k] = sig(ds[k, DS_IDX["dprior"]] + ds[k, DS_IDX["last_resid"]]) - (ws * p_prev[cs]).sum()
            np.add.at(M[k], cs, ws)
        else:
            np.add.at(M[k], cs, w_ / w_.sum())
    r = np.random.default_rng(seed)
    lgc = prior[None] + mu[None, :, hi] + r.standard_normal((n, L.shape[-1])) @ L[:, hi].T + r.standard_normal((n, C)) * np.sqrt(dg[:, hi])[None]
    draws = np.clip(a[None] + sig(lgc) @ M.T, 1e-4, 1 - 1e-4)
    return np.clip(a + M @ sig(prior + mu[:, hi]), 1e-4, 1 - 1e-4), draws


def blend_districts(km, kd, bm, bd):
    """Inverse-variance blend of two district forecasts (mean shares + draws), errors treated as correlated."""
    ks, bs = kd.std(0) + 1e-4, bd.std(0) + 1e-4
    w = (1 / ks ** 2) / (1 / ks ** 2 + 1 / bs ** 2)
    m = w * km + (1 - w) * bm
    dr = w * kd + (1 - w) * bd
    dev = dr - dr.mean(0)
    target = w * ks + (1 - w) * bs
    return m, np.clip(dr.mean(0) + dev * (target / (dev.std(0) + 1e-6)), 1e-4, 1 - 1e-4)




# ---------------- c12: the fixed blend ----------------
def sd_of(L, d):
    return np.sqrt(d + (L ** 2).sum(-1))


def combine_fixed(mk, Lk, dk, mb, Lb, db, prior):
    """Inverse-variance blend; errors treated as correlated; KERMIT kept where it puts a county beyond 98/2."""
    sk, sb = sd_of(Lk, dk), sd_of(Lb, db)
    w = np.where(np.abs(prior + mk) > CENSOR_LOGIT, 1.0, (1 / sk ** 2) / (1 / sk ** 2 + 1 / sb ** 2))
    mu = w * mk + (1 - w) * mb
    L = np.concatenate([w[..., None] * Lk, (1 - w)[..., None] * Lb], -1)
    dg = w ** 2 * dk + (1 - w) ** 2 * db
    sc = np.sqrt((w * sk + (1 - w) * sb) ** 2 / np.maximum(dg + (L ** 2).sum(-1), 1e-12))
    return mu, L * sc[..., None], dg * sc ** 2



