"""Studies 149 and 151: does refitting the blend's models monthly instead of quarterly (149), or training on a rolling
2- or 3-year window instead of all data since 2020 (151), give better picks at official cross prices?

Question: the blend's models are refitted once a quarter on every row since 2020 (expanding window). Study 149: does
a monthly refit, which uses up to two more months of recent data, help? Study 151: markets change, so does a model
trained only on the last 2 or 3 years do better than one trained on everything since 2020?

Data: study 99's training frame and code (data/ml_frame.parquet close features, y_night = close -> next panel open;
earn_tonight / news_today from data/local/study99_extra.parquet; behavior groups results/study14_groups.parquet).
Baseline (quarterly refit, expanding window) predictions for seeds 1 and 2 are reused from study 99
(data/local/study99_pred.parquet, variants "base" = LightGBM default seed, called seed 1, and "base_seed2"); study
108 reran the same baseline with seeds 3..5 (data/local/study108_pred.parquet; not needed here, the seed noise
yardstick sd 0.23 Sharpe 2024-26 comes from its five seeds). Official crosses: every cached cross table
(data/local/auctions88 / 94 / 99 / 108 / 110 / 113 / 118 / 121 / 128 / 139.parquet) plus crosses fetched here for new
picks (data/local/auctions149.parquet, study 94's rule: listing-venue 'O' / '6' print, second pass largest print of
any listing exchange; Alpaca SIP, 60 requests a minute).

Design:
  (a) Variants, each with seeds 1 and 2 (seed 1 = LightGBM default seed as study 99's base; seed 2 = seed=2), same
      models and parameters as study 99 (pooled_beh, per_beh with one SMALL model per behavior group with >= 20,000
      training rows and pooled_beh for the rest, jump / drop classifiers; 300 rounds, num_threads=2):
        quarterly  baseline: refit each quarter 2024Q1..2026Q3 on all rows since 2020-01-01 before the quarter start
                   minus a 10-trading-day embargo (study 99's predictions, reused).
        monthly    (149) refit each month 2024-01..2026-09 on all rows since 2020-01-01 before the month start minus
                   the same embargo, predict that month. The first month of each quarter has exactly the baseline's
                   training set, so its predictions are the baseline's (checked: a refit reproduces them).
        roll3y / roll2y  (151) quarterly refits as the baseline, but training rows only from the cut minus 3 / 2
                   years to the cut.
  (b) Blend per variant and seed: (2 x rank(per_beh) + rank(P(jump) - P(drop))) / 3, top 10, buy at the closing
      cross, sell at the next opening cross; returns at crosses as study 94 / 99 (panel return x cross / panel ratio
      per leg, panel price where no usable cross). Costs exec_cost_bps(P, "auction") + 2.5 bp per side.
  (c) Per period (2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09): Sharpe, mean net bp, max drawdown, turnover
      for each single-seed book and for the seed-averaged book (daily average of the two single-seed books' net
      returns, i.e. half the capital in each); paired t of daily net returns against the baseline with the same seed
      (and for the seed-averaged books); share of picks shared with the baseline of the same seed, next to the
      baseline's seed 1 vs seed 2 overlap (what seed change alone does). Differences are judged against the seed
      noise: one seed's Sharpe has sd 0.23 (study 108), so the difference between two 2-seed means has sd ~0.23 if the
      variants' seed draws are independent.

Output: results/study149_151_retrain.csv (tables: check = refit reproduces the baseline; rescore = per variant, seed
("1", "2", "avg book"), pricing and period; seedsum = mean and gap of the two seeds' Sharpe; paired = paired t vs
the baseline; overlap = pick overlap; fallback = share of legs priced at the panel; data_errors). Predictions in
data/local/study149_pred/ (one parquet per variant, seed and refit period; resumable).

    python src/study149_151_retrain.py check     # refit 2024Q1 seed 2 and compare with study 99 (5 min)
    python src/study149_151_retrain.py train     # roll2y, roll3y, monthly x seeds 1, 2 (hours; resumable)
    python src/study149_151_retrain.py fetch     # official crosses for new picks (resumable)
    python src/study149_151_retrain.py           # evaluation
"""
import gc
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # other jobs share the account's limit
from core import stock_cols, exec_cost_bps, ann_stats, RES   # noqa: E402
import bt   # noqa: E402
import study99_cross_retrain as S99   # noqa: E402

LOC = S99.LOC
AUC149 = os.path.join(LOC, "auctions149.parquet")
PRED_DIR = os.path.join(LOC, "study149_pred")
OUT = os.path.join(RES, "study149_151_retrain.csv")
PER = S99.PER
QUARTERS = S99.QUARTERS
MONTHS = pd.period_range("2024-01", "2026-09", freq="M")
SEEDS = [1, 2]
NEW = ["roll2y", "roll3y", "monthly"]          # training order
VARIANTS = ["quarterly"] + NEW
MODELS = ["per_beh", "p_jump", "p_drop"]
OTHER = ["auctions94", "auctions99", "auctions108", "auctions110", "auctions113", "auctions118", "auctions121",
         "auctions128", "auctions139"]


def seed_par(sd):
    return {} if sd == 1 else dict(seed=sd)      # seed 1 = LightGBM default seed (study 99's base)


def periods(v):
    """Refit periods of a new variant (months that start a quarter reuse the baseline for 'monthly')."""
    if v == "monthly":
        return [m for m in MONTHS if m.month not in (1, 4, 7, 10)]
    return list(QUARTERS)


def part_fn(v, sd, p):
    return os.path.join(PRED_DIR, f"{v}_s{sd}_{p}.parquet")


# ------------------------------------------------------------------ (a) train
def fit_period(lgb, D, v, sd, p):
    """Fit the four models for variant v, seed sd, refit period p (Period) and predict p's rows."""
    X, names, dates, tick, y, E, ok, dnp, days, gcache = D
    cut = days[max(0, days.searchsorted(p.start_time) - 11)]
    tr = ok & (dnp < np.datetime64(cut))
    if v.startswith("roll"):
        tr &= dnp >= np.datetime64(pd.Timestamp(cut) - pd.DateOffset(years=int(v[4])))
    te = (dnp >= np.datetime64(p.start_time)) & (dnp <= np.datetime64(p.end_time.normalize()))
    if p.year not in gcache:
        gcache.clear()
        gc.collect()
        gcache[p.year] = S99.group_feats(X, names, dates, tick, p.year)
    gid, GF, gf_names = gcache[p.year]
    sp = seed_par(sd)
    yr = (pd.Series(np.where(tr, y, np.nan)).groupby(dnp).rank(pct=True).values - 0.5).astype("float32")
    res = pd.DataFrame({"date": dates[te], "ticker": tick[te], "p": str(p), "variant": v, "seed": sd})
    ds = lgb.Dataset(S99.mat(tr, [X, GF]), yr[tr], feature_name=names + gf_names, categorical_feature=["g_id"],
                     free_raw_data=True)
    m = lgb.train(dict(S99.PARAMS, **sp), ds, num_boost_round=300)
    del ds
    pb = m.predict(S99.mat(te, [X, GF])).astype("float32")
    res["pooled_beh"] = pb
    del m
    gc.collect()
    per = pb.copy()
    gte = gid[te]
    for gg in sorted(set(gid)):
        mm = gid == gg
        if gg == -1 or (tr & mm).sum() < 20000:
            continue
        ds = lgb.Dataset(S99.mat(tr & mm, [X]), yr[tr & mm], feature_name=names, free_raw_data=True)
        mg = lgb.train(dict(S99.SMALL, **sp), ds, num_boost_round=300)
        del ds
        per[gte == gg] = mg.predict(S99.mat(te & mm, [X]))
        del mg
        gc.collect()
    res["per_beh"] = per
    jn = names + ["earn_tonight", "news_today"]
    Xte = S99.mat(te, [X, E])
    for k, lab in [("jump", y > 0.05), ("drop", y < -0.05)]:
        ds = lgb.Dataset(S99.mat(tr, [X, E]), lab[tr].astype("float32"), feature_name=jn, free_raw_data=True)
        mj = lgb.train(dict(S99.PR_BIN, **sp), ds, num_boost_round=300)
        del ds
        res[f"p_{k}"] = mj.predict(Xte).astype("float32")
        del mj
        gc.collect()
    del Xte
    return res, int(tr.sum()), int(te.sum()), str(dates[tr].min().date())


def load_all():
    P, days, nxt, gapd = S99.calendar()
    del P
    gc.collect()
    X, names, dates, tick, y = S99.load_frame()
    E = pd.read_parquet(S99.EXTRA_FN).values.astype("float32")
    ok = ~np.isnan(y)
    print("rows", len(y), "features", len(names), flush=True)
    return X, names, dates, tick, y, E, ok, np.asarray(dates), days, {}


def check():
    """Refit the baseline 2024Q1 with seed 2 (expanding window) and compare with study 99's base_seed2."""
    import lightgbm as lgb
    os.makedirs(PRED_DIR, exist_ok=True)
    D = load_all()
    t0 = time.time()
    res, ntr, nte, first = fit_period(lgb, D, "quarterly", 2, pd.Period("2024Q1", "Q"))
    old = pd.read_parquet(S99.PRED_FN, filters=[("variant", "==", "base_seed2"), ("q", "==", "2024Q1")])
    m = res.merge(old, on=["date", "ticker"], suffixes=("", "_old"))
    rows = [dict(table="check", variant="quarterly 2024Q1 seed 2 refit vs study 99", set=k, n=len(m),
                 max_abs_diff=float((m[k] - m[k + "_old"]).abs().max()),
                 corr=float(np.corrcoef(m[k], m[k + "_old"])[0, 1])) for k in ["pooled_beh"] + MODELS]
    print(pd.DataFrame(rows).to_string(), "train", ntr, "from", first, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(PRED_DIR, "_check.csv"), index=False)


def train():
    import lightgbm as lgb
    os.makedirs(PRED_DIR, exist_ok=True)
    D = load_all()
    for v in NEW:
        for sd in SEEDS:
            for p in periods(v):
                fn = part_fn(v, sd, p)
                if os.path.exists(fn):
                    continue
                t0 = time.time()
                res, ntr, nte, first = fit_period(lgb, D, v, sd, p)
                res.to_parquet(fn + ".tmp", index=False)
                os.replace(fn + ".tmp", fn)
                print(v, "seed", sd, p, "train", ntr, "from", first, "test", nte, f"{time.time() - t0:.0f}s",
                      flush=True)


# ------------------------------------------------------------------ (b) blends
def preds():
    """{(variant, seed): {model: date x ticker panel}} for complete variant-seeds."""
    base = pd.read_parquet(S99.PRED_FN, columns=["date", "ticker", "variant"] + MODELS,
                           filters=[("variant", "in", ["base", "base_seed2"])])
    base["seed"] = base.variant.map({"base": 1, "base_seed2": 2})
    base["variant"] = "quarterly"
    base["date"] = pd.to_datetime(base.date)
    parts = [base]
    for v in NEW:
        for sd in SEEDS:
            fns = [part_fn(v, sd, p) for p in periods(v)]
            if not all(os.path.exists(f) for f in fns):
                print("incomplete:", v, "seed", sd, sum(os.path.exists(f) for f in fns), "/", len(fns), flush=True)
                continue
            n = pd.concat([pd.read_parquet(f, columns=["date", "ticker"] + MODELS) for f in fns], ignore_index=True)
            n["date"] = pd.to_datetime(n.date)
            if v == "monthly":   # quarter-start months: same training set as the baseline -> baseline predictions
                b = base[(base.seed == sd) & base.date.dt.month.isin([1, 4, 7, 10])]
                n = pd.concat([n, b[["date", "ticker"] + MODELS]], ignore_index=True)
            parts.append(n.assign(variant=v, seed=sd))
    return pd.concat(parts, ignore_index=True)


def all_picks(cols, days):
    pr = preds()
    W = {}
    idx = None
    for (v, sd), g in pr.groupby(["variant", "seed"], sort=False):
        g = g.set_index(["date", "ticker"])
        e = g.per_beh.unstack().reindex(columns=cols).astype("float64")
        j = (g.p_jump - g.p_drop).unstack().reindex(columns=cols).astype("float64")
        if idx is None:
            idx = e.index
        e, j = e.reindex(index=idx), j.reindex(index=idx)
        ok = e.notna() & j.notna()
        S = (2 * e.where(ok).rank(axis=1, pct=True) + j.where(ok).rank(axis=1, pct=True)) / 3
        S = S.loc[(S.index < days[-1])].loc["2024-01":"2026-09"]
        W[(v, int(sd))] = bt.select_topk(S, S.notna(), 10)
    del pr
    gc.collect()
    return W


# ------------------------------------------------------------------ crosses
def crosses():
    parts = [S99.crosses()]
    for fn in [os.path.join(LOC, f + ".parquet") for f in OTHER[2:]] + [AUC149]:
        if os.path.exists(fn):
            c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
            c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
            c["date"] = pd.to_datetime(c.date)
            parts.append(c)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])


def tried():
    parts = [crosses()[["ticker", "date", "kind"]]]
    for fn in [os.path.join(LOC, f + ".parquet") for f in OTHER] + [AUC149]:
        if os.path.exists(fn):
            t = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t["ticker"] = t.ticker.astype(str).str.replace(".", "-", regex=False)
            parts.append(t.assign(date=pd.to_datetime(t.date)))
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def fetch():
    import study94_cross_rescore as S94
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    S94.A.RL = A.RL
    P, days, nxt, gapd = S99.calendar()
    cols = stock_cols(P)
    W = all_picks(cols, days)
    print("blends", list(W), flush=True)
    rows = []
    for Wv in W.values():
        s = Wv.stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(rows).dropna().drop_duplicates()
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    m = need.merge(tried(), how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("needed", len(need), "to fetch", len(todo), flush=True)
    todo = todo.assign(ticker=todo.ticker.str.replace("-", ".", regex=False))
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            r = S94._day_kind(job[0], job[1], job[2], ven)
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
            for x in r:
                x["rule"] = "venue"
            if miss:               # second pass: largest print of any listing exchange
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    if x["ticker"] in lg:
                        x["rule"] = "largest"
                        if not np.isnan(lg[x["ticker"]]["px"]):
                            x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x"]})
            return r
        except RuntimeError as e:  # left uncached; a rerun retries
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC149) if os.path.exists(AUC149) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 12):
        with ThreadPoolExecutor(2) as exe:
            for r in exe.map(one, jobs[i:i + 12]):
                new += r
        if not new:
            continue
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC149 + ".tmp", index=False)
        os.replace(AUC149 + ".tmp", AUC149)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ (c) evaluation
def tstat(d):
    d = d.dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


def overlap(Wa, Wb, a0, b0):
    ha, hb = (Wa > 0).loc[a0:b0], (Wb > 0).loc[a0:b0]
    return float(((ha & hb).sum(axis=1) / ha.sum(axis=1).replace(0, np.nan)).mean())


def main():
    P, days, nxt, gapd = S99.calendar()
    cols = stock_cols(P)
    W = all_picks(cols, days)
    VA = [v for v in VARIANTS if all((v, s) in W for s in SEEDS)]
    print("complete variants", VA, flush=True)
    a = crosses()
    ro, rc, err = S99.cross_ratios(P, cols, days, a)
    print("cross data errors (legs set to the panel price):", err, flush=True)
    rows = [dict(table="data_errors", set=k, n=v["found"], scale_mismatch=v["scale"], ratio_out_of_range=v["ratio"])
            for k, v in err.items()]
    ck = os.path.join(PRED_DIR, "_check.csv")
    if os.path.exists(ck):
        rows += pd.read_csv(ck).to_dict("records")
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = o.shift(-1) / c - 1
    Rn_x = (1 + Rn_p) * ro.fillna(1.0).shift(-1) / rc.fillna(1.0) - 1
    miss = ro.shift(-1).isna() | rc.isna()
    C = exec_cost_bps(P, "auction")[cols] + 2.5
    nets, turns = {}, {}
    for (v, sd), Wx in W.items():
        if v not in VA:
            continue
        h = Wx > 0
        turns[(v, sd)] = 1 - (h & h.shift(1, fill_value=False)).sum(axis=1) / h.sum(axis=1).replace(0, np.nan)
        for pr, R in [("panel", Rn_p), ("cross", Rn_x)]:
            nets[(v, sd, pr)] = bt.run(Wx, R.reindex_like(Wx), C.reindex_like(Wx)).net
        for pl, a0, b0 in PER:
            hh = h.loc[a0:b0]
            mm = miss.reindex_like(Wx).fillna(True).loc[a0:b0] & hh
            rows.append(dict(table="fallback", variant=v, seed=str(sd), period=pl, legs=int(hh.sum().sum()),
                             share_panel=mm.sum().sum() / max(hh.sum().sum(), 1)))
    for v in VA:
        for pr in ["panel", "cross"]:
            nets[(v, "avg book", pr)] = pd.concat([nets[(v, s, pr)] for s in SEEDS], axis=1).mean(axis=1)
        turns[(v, "avg book")] = pd.concat([turns[(v, s)] for s in SEEDS], axis=1).mean(axis=1)
    # rescore
    for v in VA:
        for sd in SEEDS + ["avg book"]:
            for pr in ["panel", "cross"]:
                for pl, a0, b0 in PER:
                    x = nets[(v, sd, pr)].loc[a0:b0].dropna()
                    st = ann_stats(x)
                    rows.append(dict(table="rescore", variant=v, seed=str(sd), pricing=pr, period=pl, n=st["n"],
                                     sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"],
                                     turnover=float(turns[(v, sd)].loc[a0:b0].iloc[1:].mean())))
    R = pd.DataFrame([r for r in rows if r["table"] == "rescore"])
    # seed summary: mean of the two seeds, their gap, and difference vs baseline mean
    for pr in ["panel", "cross"]:
        for pl, a0, b0 in PER:
            g0 = R[(R.variant == "quarterly") & (R.pricing == pr) & (R.period == pl) & R.seed.isin(["1", "2"])]
            for v in VA:
                g = R[(R.variant == v) & (R.pricing == pr) & (R.period == pl) & R.seed.isin(["1", "2"])]
                rows.append(dict(table="seedsum", variant=v, pricing=pr, period=pl, sharpe_mean=g.sharpe.mean(),
                                 sharpe_seed_gap=g.sharpe.iloc[1] - g.sharpe.iloc[0], net_bp_mean=g.net_bp.mean(),
                                 maxdd_mean=g.maxdd.mean(), sharpe_mean_minus_base=g.sharpe.mean() - g0.sharpe.mean(),
                                 diff_in_noise_sd=(g.sharpe.mean() - g0.sharpe.mean()) / 0.23))
    # paired vs baseline with the same seed
    for v in VA[1:]:
        for sd in SEEDS + ["avg book"]:
            for pr in ["panel", "cross"]:
                for pl, a0, b0 in PER:
                    d = (nets[(v, sd, pr)] - nets[("quarterly", sd, pr)]).loc[a0:b0].dropna()
                    rows.append(dict(table="paired", variant=v, seed=str(sd), set="vs quarterly same seed",
                                     pricing=pr, period=pl, n=len(d), diff_bp=1e4 * d.mean(), t_diff=tstat(d),
                                     corr=float(nets[(v, sd, pr)].loc[a0:b0].corr(
                                         nets[("quarterly", sd, pr)].loc[a0:b0]))))
    # seed-noise reference: baseline seed 2 vs seed 1
    for pr in ["panel", "cross"]:
        for pl, a0, b0 in PER:
            for v in VA:
                d = (nets[(v, 2, pr)] - nets[(v, 1, pr)]).loc[a0:b0].dropna()
                rows.append(dict(table="paired", variant=v, seed="2 vs 1", set="seed noise reference",
                                 pricing=pr, period=pl, n=len(d), diff_bp=1e4 * d.mean(), t_diff=tstat(d),
                                 corr=float(nets[(v, 2, pr)].loc[a0:b0].corr(nets[(v, 1, pr)].loc[a0:b0]))))
    # overlap
    for pl, a0, b0 in PER:
        for v in VA:
            for sd in SEEDS:
                if v != "quarterly":
                    rows.append(dict(table="overlap", variant=v, seed=str(sd), set="vs quarterly same seed", period=pl,
                                     share=overlap(W[(v, sd)], W[("quarterly", sd)], a0, b0)))
            rows.append(dict(table="overlap", variant=v, seed="2 vs 1", set="seed noise reference", period=pl,
                             share=overlap(W[(v, 2)], W[(v, 1)], a0, b0)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"check": check, "train": train, "fetch": fetch, "eval": main}[cmd]()
