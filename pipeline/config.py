"""Pipeline settings. The only file you should need to edit."""
import datetime as dt

ELECTION_DATE = dt.date(2026, 11, 3)
# Election-day midnight US Eastern (EST, UTC-5 after DST ends on Nov 1) = hour 0 of the model's clock
T0_UTC = dt.datetime(2026, 11, 3, 5, 0, tzinfo=dt.timezone.utc)
# Live window: from just before the first polls close (6pm ET) until the morning of Nov 5 ET
LIVE_START_UTC = dt.datetime(2026, 11, 3, 22, 30, tzinfo=dt.timezone.utc)
LIVE_END_UTC = dt.datetime(2026, 11, 5, 17, 0, tzinfo=dt.timezone.utc)

# What the live tab uses: "blend" (KERMIT + Bayes, like the forecast), "bayes" (polling model + counted votes only),
# or "off" (live tab shows the explainer, no live numbers). Flip this in the repo if the night goes strange.
LIVE_MODEL = "blend"

# --- sources ---
VOTEHUB = "https://api.votehub.com/polls"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HUBS = {"sen": "2026 United States Senate elections", "gov": "2026 United States gubernatorial elections"}
NYT_FEEDS = {race: f"https://static01.nyt.com/elections-assets/pages/data/2026-11-03/results-{name}.json"
             for race, name in (("sen", "senate"), ("house", "house"), ("gov", "governor"))}
NYT_TEST_FEED = "https://static01.nyt.com/elections-assets/pages/data/2024-11-05/results-senate.json"   # self-test
NCSBE_ZIP = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2026_11_03/absentee_20261103.zip"
USER_AGENT = "KERMIT-election-forecast/1.0 (https://harveyolden.github.io/electoral-forecaster/; scheduled GitHub Actions job)"

# --- model ---
N_DRAWS = 4000                 # joint simulations for seat counts and ranges
STRICT = {"pres": ["poll_sen", "poll_gov"], "sen": ["poll_gov"], "gov": ["poll_sen"], "house": ["poll_sen", "poll_gov"]}
SENATE_NOT_UP = {"D": 34, "R": 31}        # seats not on the 2026 ballot (incl. independents who caucus with Democrats)
SENATE_MAJORITY = 51                      # 50-50 goes to the Republican vice president
HOUSE_MAJORITY = 218
HOUSE_FIXED = {   # districts missing from the model's district map (no county data in the 2024 source file): safe seats
    "FL-20": ("D", "Not modelled (no county data for this district in the source files); a safe Democratic seat."),
    "OK-03": ("R", "Not modelled (no county data for this district in the source files); a safe Republican seat."),
}
NOT_MODELLED = {}
# Independents running with no Democrat on the ballot: modelled as the non-Republican candidate (their polls are read
# as the "Democratic" side), shown green on the map, and NOT counted as Democratic seats.
INDEPENDENTS = {("sen", "NE"): "Dan Osborn", ("sen", "ID"): "Todd Achilles", ("sen", "SD"): "Brian Bengs"}

# --- sanity checks: polls ---
POLL_MIN_DATE = dt.date(2025, 11, 4)
POLL_MAX_SAMPLE = 20000
POLL_DEFAULT_SAMPLE = 600
POLL_OUTLIER_PTS = 25          # drop a poll whose margin is this far from the median of >= 4 polls of the same race
APPROVAL_MAX_JUMP = 5.0        # ignore a recomputed approval average that moved more than this vs the bundle's

# --- sanity checks: published forecast ---
MAX_STATE_JUMP_PTS = 10        # pre-election update holds (not published) if a state's Dem share moves more than this...
MAX_PMAJ_JUMP = 0.40           # ...or a chamber's P(Dem majority) moves more than this since the last published run

MIN_RACES = {"senate": 30, "governor": 30, "house": 400}   # fewer races than this in the output = something broke

# --- sanity checks: live counts ---
LIVE_MAX_COUNTY_RATIO = 1.6    # counted votes > 1.6 x the county's expected midterm turnout = bad row
MIDTERM_TURNOUT = 0.72         # expected midterm votes = presidential-year expectation x this
LIVE_MIN_GOOD_SHARE = 0.6      # fewer than 60% of reporting counties pass the checks -> skip this live update
LIVE_MAX_DROP = 0.02           # a county's count may fall by 2% (corrections); more -> that county is ignored this run
