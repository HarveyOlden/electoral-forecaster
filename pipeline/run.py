"""One pipeline run (what GitHub Actions calls on the schedule).

  python pipeline/run.py                 # auto: pre-election update, or a live update during election night
  python pipeline/run.py --mode pre      # force a pre-election update
  python pipeline/run.py --mode live     # force a live update
  python pipeline/run.py --mode selftest # check everything, publish nothing
  add --force to publish even if the jump check would hold the update

Writes data/forecast.json (pre) or data/live.json (live), data/meta.json and data/pipeline_log.json.
A run that fails a check publishes nothing new (the site keeps the last good files) and exits with an error, so
GitHub emails you and the Actions tab shows a red run with the reason in the log.
"""
import os, sys, json, gzip, pickle, time, argparse, traceback, datetime as dt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
BUNDLE = os.path.join(HERE, "bundle", "bundle.pkl.gz")
MODEL = os.path.join(HERE, "bundle", "model.pt")


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def hours_since_t0(t):
    return (t - C.T0_UTC).total_seconds() / 3600


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(name):
    p = os.path.join(DATA, name)
    try:
        return json.load(open(p))
    except Exception:
        return None


def write_json(name, obj):
    p = os.path.join(DATA, name)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), default=_jdefault)
    os.replace(tmp, p)


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return str(o)


def load_bundle():
    if not (os.path.exists(BUNDLE) and os.path.exists(MODEL)):
        raise SystemExit(f"missing {BUNDLE} or {MODEL}: run the Colab export cell and upload both files to pipeline/bundle/")
    with gzip.open(BUNDLE, "rb") as fh:
        return pickle.load(fh)


def jump_check(new, old, log):
    """Pre-election guard: hold the update if anything moved implausibly far since the last published forecast."""
    if not old or old.get("demo"):
        return True
    bad = []
    for ch in ("senate", "house"):
        a, b = new["chambers"][ch]["p_dem_majority"], old["chambers"].get(ch, {}).get("p_dem_majority")
        if b is not None and abs(a - b) > C.MAX_PMAJ_JUMP:
            bad.append(f"{ch} P(D majority) {b:.2f} -> {a:.2f}")
    for label in ("senate", "governor"):
        prev = {r["id"]: r.get("dem_share") for r in old["races"].get(label, [])}
        for r in new["races"][label]:
            p0 = prev.get(r["id"])
            if p0 is not None and r.get("dem_share") is not None and abs(r["dem_share"] - p0) * 100 > C.MAX_STATE_JUMP_PTS:
                bad.append(f"{label} {r['id']} {100 * p0:.1f} -> {100 * r['dem_share']:.1f}")
    if bad:
        log.append("JUMP CHECK FAILED (update held; rerun with force to publish): " + "; ".join(bad[:10]))
        return False
    return True


def output_checks(out, log):
    """Basic shape/number checks on what we are about to publish."""
    ok = True
    for ch in ("senate", "house"):
        c = out["chambers"][ch]
        if not (0 <= c["p_dem_majority"] <= 1) or not (0 < c["dem_seats_mean"] < c["seats_total"]):
            log.append(f"OUTPUT CHECK FAILED: {ch} numbers out of range ({c['p_dem_majority']}, {c['dem_seats_mean']})"); ok = False
    for label, n_min in C.MIN_RACES.items():
        if len(out["races"][label]) < n_min:
            log.append(f"OUTPUT CHECK FAILED: only {len(out['races'][label])} {label} races"); ok = False
        for r in out["races"][label]:
            v = r.get("dem_share")
            if v is not None and not (0 < v < 1):
                log.append(f"OUTPUT CHECK FAILED: {label} {r['id']} share {v}"); ok = False; break
    return ok


def gather_obs(bundle, today, log, live=False):
    import polls
    P, prep = polls.gather(bundle, today, log)
    pobs = polls.poll_obs(P, bundle)
    ev = None
    if not live and today < C.ELECTION_DATE:
        try:
            import early_vote
            ev = early_vote.nc_obs(bundle, log)
        except Exception as e:
            log.append(f"NC early vote failed: {e}")
    if ev is None:
        ev = bundle["ev_obs"]
    obs = pd.concat([pobs, ev, bundle["spec_obs"]], ignore_index=True)
    approval = bundle["fund"]["approval"]
    if prep["approval"] is not None and abs(prep["approval"] - approval) <= C.APPROVAL_MAX_JUMP:
        approval = prep["approval"]
    elif prep["approval"] is not None:
        log.append(f"approval {prep['approval']:.1f} too far from the bundle's {approval:.1f}: kept the bundle's")
    inputs = dict(polls=prep["by_race"], newest_poll=prep["newest"], poll_sources=prep["fresh"], poll_drops=prep["drops"],
                  nc_early_vote_reads=int((obs.source == "ev_read").sum()), specials=int((obs.source == "special").sum()),
                  approval=round(approval, 1))
    return obs, P, approval, inputs


def run_pre(eng, bundle, t, log, force=False, publish=True, with_drivers=True):
    import polls, forecast as F
    obs, P, approval, inputs = gather_obs(bundle, t.date(), log)
    cutoff = hours_since_t0(t)
    ctx = dict(approval=approval)
    t1 = time.time()
    res = eng.run(obs, cutoff, ctx_over=ctx)
    draws = eng.state_draws(res)
    log.append(f"forecast computed in {time.time() - t1:.0f}s")
    drv = {}
    if with_drivers:
        t1 = time.time()
        drv = F.drivers(eng, obs, cutoff, res, polls.poll_summary(P), ctx_over=ctx)
        log.append(f"drivers computed in {time.time() - t1:.0f}s")
    out = F.build_outputs(eng, res, draws, drv, dict(generated_at=iso(now_utc()), data_cutoff=iso(t), inputs=inputs))
    ok = output_checks(out, log)
    if ok and not force:
        ok = jump_check(out, read_json("forecast.json"), log)
    if publish and ok:
        write_json("forecast.json", out)
        hist = read_json("history.json") or {"note": "headline numbers of every published pre-election forecast", "runs": []}
        hist["runs"].append(dict(generated_at=out["generated_at"],
                                 **{f"{ch}_{k}": out["chambers"][ch][k] for ch in ("senate", "house")
                                    for k in ("p_dem_majority", "dem_seats_mean", "dem_seats_lo80", "dem_seats_hi80")},
                                 governor_dem_wins_mean=out["chambers"]["governor"]["dem_wins_mean"],
                                 races={lab: {r["id"]: r["p_dem_win"] for r in out["races"][lab]} for lab in ("senate", "governor")}))
        write_json("history.json", hist)
    return ok, out


def run_live(eng, bundle, t, log, force=False, publish=True):
    import live, polls, forecast as F
    cutoff = hours_since_t0(t)
    if C.LIVE_MODEL == "off":
        log.append("LIVE_MODEL is off: live tab shows the explainer only")
        if publish:
            write_json("live.json", {"is_live": False, "updated_at": iso(now_utc()), "note": "live forecast switched off"})
        return True, None
    L = live.pull(bundle, cutoff, log)
    if not L["available"]:
        if publish and not (read_json("live.json") or {}).get("is_live"):
            write_json("live.json", {"is_live": False, "updated_at": iso(now_utc())})
        return True, None
    if not L.get("good", True) and not force:
        return False, None                                  # keep the last live.json; red run with the reason
    obs, P, approval, inputs = gather_obs(bundle, t.date(), log, live=True)
    counts = live.all_batches(L["new_obs"])
    obs = pd.concat([obs, counts], ignore_index=True)
    ctx = dict(approval=approval)
    res = eng.run(obs, cutoff, ctx_over=ctx, live_model=C.LIVE_MODEL)
    draws = eng.state_draws(res)
    called, progress = {}, {}
    for r in L["races"]:
        k = f"{r['race']}|{r['key']}"
        if r.get("called_for") in ("D", "R"):
            called[k] = r["called_for"]
        tot = r["D"] + r["R"]
        progress[k] = dict(reported_pct=r.get("reported_pct"),
                           status="called" if r.get("called_for") else ("counting" if tot > 0 else "not_started"))
    drv = F.drivers(eng, obs, cutoff, res, polls.poll_summary(P), live=True, ctx_over=ctx, live_model=C.LIVE_MODEL)
    inputs.update(counted_batches=int(len(counts)), live_checks=L["report"], live_model=C.LIVE_MODEL)
    out = F.build_outputs(eng, res, draws, drv, dict(generated_at=iso(now_utc()), data_cutoff=iso(t), inputs=inputs),
                          called=called, progress=progress)
    out.update(is_live=True, updated_at=iso(now_utc()))
    for ch, lab in (("senate", "sen"), ("house", "house")):
        out["chambers"][ch]["called"] = {p: sum(1 for k, v in called.items() if k.startswith(lab + "|") and v == p) for p in "DR"}
    ok = output_checks(out, log)
    if publish and ok:
        live.commit_state(L["keep"], L["new_obs"], iso(t))
        write_json("live.json", out)
    return ok, out


def selftest(log):
    """Checks everything without publishing: model reproduces Colab's numbers, featuriser matches, every source
    answers, NYT parser works on the 2024 feed, a full forecast runs, and how long it all takes."""
    import torch, polls, live, forecast as F
    from kermit_core import featurise, Bugs, predict, to_dev, bayes_predict
    rep = {}
    t0 = time.time()
    b = load_bundle()
    eng = F.Engine(b, MODEL)
    rep["load_s"] = round(time.time() - t0, 1)
    R = b["reference"]
    inst = dict(county=R["county"].astype(np.float32), state=R["state"].astype(np.float32), nation=R["nation"].astype(np.float32),
                static=b["inst"]["static"], race_static=b["inst"]["race_static"], dist_static=b["inst"]["dist_static"],
                context=b["inst"]["context"], y=np.full((eng.Cn, 4), np.nan, np.float32), y_cens=np.zeros((eng.Cn, 4), np.float32),
                y_turnout=np.full(eng.Cn, np.nan, np.float32))
    t1 = time.time()
    o = predict(eng.model, to_dev(inst), eng.rel)
    rep["forward_pass_s"] = round(time.time() - t1, 2)
    rep["model_matches_colab_max_diff"] = float(np.nanmax(np.abs(o["mu"] - R["mu"])))
    mb, _, _ = bayes_predict(inst, eng.so, eng.S, eng.fit, eng.coefs)
    rep["bayes_matches_colab_max_diff"] = float(np.nanmax(np.abs(mb - R["bayes_mu"])))
    # featuriser: the bundle's own 2026 polls through this repo's poll_obs vs Colab's poll_obs
    p_repo = polls.poll_obs(b["polls_2026"][b["polls_2026"].race.isin(["sen", "gov", "house", "generic"])].copy(), b)
    rep["poll_obs_rows_repo_vs_colab"] = [int(len(p_repo)), int(len(b["poll_obs_at_export"]))]
    cut = float(b["inst"]["context"][5])
    d1, _ = featurise(eng.w, pd.concat([p_repo, b["ev_obs"], b["spec_obs"]]), cut, Bugs())
    rep["state_dossier_max_diff_vs_colab"] = float(np.abs(d1["state"] - R["state"].astype(np.float32)).max())
    # sources
    P, prep = polls.gather(b, now_utc().date(), log)
    rep["polls"] = dict(by_race=prep["by_race"], fresh=prep["fresh"], drops=prep["drops"], approval=prep["approval"], newest=prep["newest"])
    try:
        js = live.fetch(C.NYT_TEST_FEED)
        c, r = live.parse(js, "sen")
        rep["nyt_2024_test"] = dict(races=len(r), county_rows=int(len(c)), called=sum(1 for x in r if x["called_for"]),
                                    example=r[0] if r else None)
    except Exception as e:
        rep["nyt_2024_test"] = f"FAILED: {e}"
    try:
        live_js = {k: (live.fetch(u) is not None) for k, u in C.NYT_FEEDS.items()}
        rep["nyt_2026_feeds_up"] = live_js
    except Exception as e:
        rep["nyt_2026_feeds_up"] = f"FAILED: {e}"
    t1 = time.time()
    ok, out = run_pre(eng, b, now_utc(), log, publish=False)
    rep["full_forecast_with_drivers_s"] = round(time.time() - t1, 1)
    rep["forecast_checks_pass"] = ok
    if out:
        rep["headline"] = {ch: dict(dem_seats_mean=out["chambers"][ch]["dem_seats_mean"], p_dem_majority=out["chambers"][ch]["p_dem_majority"])
                           for ch in ("senate", "house")}
        rep["example_race"] = out["races"]["senate"][0]
    rep["verdict"] = ("OK" if rep["model_matches_colab_max_diff"] < 1e-3 and rep["bayes_matches_colab_max_diff"] < 1e-6 and ok
                      else "PROBLEM - read the numbers above")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "pre", "live", "selftest"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    t = now_utc()
    log = [f"run at {iso(t)} (mode {a.mode}{', force' if a.force else ''})"]
    status = dict(ok=True)
    try:
        if a.mode == "selftest":
            rep = selftest(log)
            print(json.dumps(rep, indent=1, default=_jdefault))
            print("\n".join(log))
            json.dump(dict(report=rep, log=log), open(os.path.join(ROOT, "selftest_result.json"), "w"), indent=1, default=_jdefault)
            return 0 if rep["verdict"] == "OK" else 1
        mode = a.mode
        if mode == "auto":
            if C.LIVE_START_UTC <= t <= C.LIVE_END_UTC:
                mode = "live"
            elif t > C.LIVE_END_UTC:
                log.append("after the live window: nothing to do")
                print("\n".join(log))
                return 0
            else:
                mode = "pre"
        import forecast as F
        bundle = load_bundle()
        eng = F.Engine(bundle, MODEL)
        if mode == "pre":
            ok, _ = run_pre(eng, bundle, t, log, force=a.force)
            if ok and (read_json("live.json") or {}).get("demo"):
                write_json("live.json", {"is_live": False, "updated_at": iso(now_utc())})   # replace the demo file
        else:
            ok, _ = run_live(eng, bundle, t, log, force=a.force)
        status = dict(ok=bool(ok), mode=mode)
        if ok:
            meta = read_json("meta.json") or {}
            lv = read_json("live.json") or {}
            meta.update(demo=False, generated_at=iso(now_utc()), live_available=bool(lv.get("is_live")),
                        files=["forecast.json", "live.json", "backtest.json"])
            write_json("meta.json", meta)                   # written LAST: this is what makes open pages refresh
    except Exception as e:
        log.append(f"CRASH: {e}\n{traceback.format_exc()}")
        status = dict(ok=False)
    status.update(finished=iso(now_utc()), log=log)
    write_json("pipeline_log.json", status)
    print("\n".join(log))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
