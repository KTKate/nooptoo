"""Study 185: does training the blend's models on within-day percentile ranks of the stock-level inputs, instead of
their raw values, make the blend robust to shifts in the inputs' cross-stock distribution (2026Q3)?

Question: the blend has been weak since July 2026. Study 181 found that in 2026Q3 the cross-stock spread of 60-day
beta widened sharply (PSI 0.60; median of all ml_frame rows fell from about 0.95 to 0.5), and beta is the ensemble's
largest stock-level input. Tree models split on raw values, so a shift in a feature's distribution moves stocks
across learned split points even if their relative order is unchanged. If every stock-level input is replaced by its
within-day percentile rank across the universe, the models see only the order of stocks each day. Is that blend
better than the raw-input blend at official cross prices, overall and in 2026Q3? (Study 184 tests remapping beta for
the frozen models.)

Data: data/ml_frame.parquet (study-3 close features as of the close of day t, y_night = close -> next panel open;
rows 2020-01-01..2026-09-25), with days_to_e taken from the corrected earnings panel data/local/study158_earn.parquet
in both variants (study 158: the frame's days_to_e is missing for 11-99% of rows in 2026-07..09 because the calendar
it was built with ended in September; elsewhere it matches the frame to 0.1-0.3% of rows). earn_tonight /
news_today from data/local/study99_extra.parquet; behavior groups results/study14_groups.parquet. Official crosses:
every cached data/local/auctions*.parquet (study 166's cached_crosses) plus data/local/auctions185.parquet fetched
here for the legs of new picks with study 94's rule (listing-venue 'O' / '6' print in widening windows, second pass:
the largest 'O' / '6' print of any listing exchange; Alpaca SIP, 60 requests a minute). Industries
data/store/sectors.parquet. ml_frame ends 2026-09-25, so October 2026 days are not scored here.

Design:
  (a) Walk-forward exactly as studies 99 / 108 / 149 (quarterly expanding refits, test 2024Q1..2026Q3, training rows
      from 2020-01-01 to 10 trading days before the quarter start, same LightGBM parameters, 300 rounds,
      num_threads=2): pooled_beh (study-3 inputs + behavior-group features), per_beh (one SMALL model per behavior
      group with >= 20,000 training rows, pooled_beh for the rest), jump / drop classifiers (y_night > +5% / < -5%,
      study-3 inputs + earn_tonight + news_today). Seeds 1 (LightGBM default) and 2 (seed=2), as study 149.
  (b) Variants, both with the corrected days_to_e:
        raw   inputs as in the frame (the baseline, retrained because days_to_e differs from studies 99 / 108).
        rank  every stock-level input replaced by its within-day percentile rank (pandas rank(pct=True), ties
              averaged, missing stays missing) among the frame's rows of that day (the model universe), before
              training and scoring. Stock-level: the 44 frame inputs other than the market-level ones, plus
              earn_tonight and news_today. Market-level, left raw (one value per day across all stocks, checked):
              spy1, spy5, spy20, vix, vix_chg5, vix_term, iwm_spy5, dow, tom. Behavior-group features (g_r1_rel,
              g_r5_rel, g_night1_rel, g_intra1_rel = rank minus group-day mean rank, g_r5_mean = group-day mean of
              the r5 rank) are computed from the ranked inputs; g_id stays categorical. Targets unchanged.
  (c) Blend per variant and seed: (2 x rank(per_beh) + rank(P(jump) - P(drop))) / 3, top 10 with at most 3 per
      industry (paper_overnight.cap_by_industry rule, study 166 capped_topk), equal weight, buy at the closing cross of
      t, sell at the opening cross of t+1 (study 94 / 166 returns: panel night return x open cross / panel open on t+1
      / (close cross / panel close on t), legs without a usable cross at the panel price). Costs
      core.exec_cost_bps(P, "auction") + 2.5 bp per side.
  (d) Per period (2024-01..2025-06, 2025-07..2026-06, 2026Q3 = 2026-07..09, 2024-01..2026-09): Sharpe, mean net bp,
      max drawdown per single-seed book and the 2-seed means; paired t of daily net returns rank vs raw with the same
      seed and for the seed-averaged book (half the capital in each seed's book); seed noise: raw seed 2 vs seed 1 on
      the same table, next to study 108's sd of 0.23 Sharpe for one seed (2024-26), so a difference of two 2-seed means
      has sd about 0.23. Pick overlap rank vs raw (same seed) and seed 2 vs seed 1. Check: the raw seed-1 2024Q1 fit
      against study 99's base (original days_to_e) to show the setup reproduces it.

Output: results/study185_rank_inputs.csv (tables: check, rescore, seedsum, paired, overlap, fallback, data_errors).
Predictions in data/local/study185_pred/ (one parquet per variant, seed and quarter; resumable), ranked inputs cached
in data/local/study185_pred/_xrank.npy. Training log data/local/study185_pred/train.log.

    python src/study185_rank_inputs.py prep      # ranked inputs (one-off, ~3 GB peak, a few minutes)
    python src/study185_rank_inputs.py train     # 2 variants x 2 seeds x 11 quarters (hours; resumable)
    python src/study185_rank_inputs.py fetch     # official crosses for new picks (resumable)
    python src/study185_rank_inputs.py           # evaluation
"""
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # other jobs share the account's limit
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES   # noqa: E402
import bt   # noqa: E402
import study99_cross_retrain as S99   # noqa: E402

LOC = S99.LOC
PRED_DIR = os.path.join(LOC, "study185_pred")
XR_FN = os.path.join(PRED_DIR, "_xrank.npy")
ER_FN = os.path.join(PRED_DIR, "_erank.npy")
DTE_FN = os.path.join(PRED_DIR, "_dte.npy")
EARN158 = os.path.join(LOC, "study158_earn.parquet")
AUC185 = os.path.join(LOC, "auctions185.parquet")
OUT = os.path.join(RES, "study185_rank_inputs.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26H1", "2025-07", "2026-06"), ("2026Q3", "2026-07", "2026-09"),
       ("2024-26", "2024-01", "2026-09")]
QUARTERS = S99.QUARTERS
SEEDS = [1, 2]
VARIANTS = ["raw", "rank"]
MODELS = ["per_beh", "p_jump", "p_drop"]
MARKET = ["spy1", "spy5", "spy20", "vix", "vix_chg5", "vix_term", "iwm_spy5", "dow", "tom"]
NOISE_SD = 0.23          # study 108: sd of one seed's Sharpe, 2024-26 at crosses


def seed_par(sd):
    return {} if sd == 1 else dict(seed=sd)


def part_fn(v, sd, q):
    return os.path.join(PRED_DIR, f"{v}_s{sd}_{q}.parquet")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(PRED_DIR, "train.log"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------ inputs
def corrected_dte(dates, tick):
    """days_to_e from study 158's corrected panel for each frame row (NaN where the panel has no value)."""
    e = pd.read_parquet(EARN158, columns=["date", "ticker", "days_to_e"])
    e["date"] = pd.to_datetime(e.date)
    s = e.set_index(["date", "ticker"]).days_to_e
    s = s[~s.index.duplicated()]
    out = s.reindex(pd.MultiIndex.from_arrays([dates, tick])).values.astype("float32")
    del e, s
    gc.collect()
    return out


def day_rank(col, codes):
    return pd.Series(col).groupby(codes).rank(pct=True).values.astype("float32")


def prep():
    """Corrected days_to_e per row, and the ranked stock-level inputs (frame + extra), cached as .npy."""
    os.makedirs(PRED_DIR, exist_ok=True)
    X, names, dates, tick, y = S99.load_frame()
    j = {n: i for i, n in enumerate(names)}
    dte = corrected_dte(dates, tick)
    old = X[:, j["days_to_e"]]
    both = ~np.isnan(old) & ~np.isnan(dte)
    m26 = np.asarray(dates >= "2026-07-01")
    log("days_to_e check: rows", len(dte), "frame missing", float(np.isnan(old).mean()), "corrected missing",
        float(np.isnan(dte).mean()), "both present and differ", float((both & (np.abs(old - dte) > 0.5)).mean()),
        "2026Q3 frame missing", float(np.isnan(old[m26]).mean()), "corrected", float(np.isnan(dte[m26]).mean()))
    np.save(DTE_FN, dte)
    X[:, j["days_to_e"]] = dte
    # market-level check: one value per day
    codes = pd.factorize(pd.Series(dates))[0]
    for n in MARKET:
        nu = pd.Series(X[:, j[n]]).groupby(codes).nunique().max()
        assert nu <= 1, (n, nu)
    stock = [n for n in names if n not in MARKET]
    log("stock-level inputs ranked:", len(stock), stock)
    log("market-level inputs kept raw:", MARKET)
    for n in stock:
        X[:, j[n]] = day_rank(X[:, j[n]], codes)
    np.save(XR_FN, X)
    E = pd.read_parquet(S99.EXTRA_FN).values.astype("float32")
    for k in range(E.shape[1]):
        E[:, k] = day_rank(E[:, k], codes)
    np.save(ER_FN, E)
    log("ranked inputs saved", X.shape, E.shape)


def load_variant(v):
    X, names, dates, tick, y = S99.load_frame()
    E = pd.read_parquet(S99.EXTRA_FN).values.astype("float32")
    if v == "raw":
        X[:, names.index("days_to_e")] = np.load(DTE_FN)
    else:
        del X, E
        gc.collect()
        X, E = np.load(XR_FN), np.load(ER_FN)
    return X, names, dates, tick, y, E


# ------------------------------------------------------------------ (a) train
def fit_quarter(lgb, D, sd, q):
    X, names, dates, tick, y, E, ok, dnp, days, gcache = D
    cut = days[max(0, days.searchsorted(q.start_time) - 11)]
    tr = ok & (dnp < np.datetime64(cut))
    te = (dnp >= np.datetime64(q.start_time)) & (dnp <= np.datetime64(q.end_time.normalize()))
    if q.year not in gcache:
        gcache.clear()
        gc.collect()
        gcache[q.year] = S99.group_feats(X, names, dates, tick, q.year)
    gid, GF, gf_names = gcache[q.year]
    sp = seed_par(sd)
    yr = (pd.Series(np.where(tr, y, np.nan)).groupby(dnp).rank(pct=True).values - 0.5).astype("float32")
    res = pd.DataFrame({"date": dates[te], "ticker": tick[te], "q": str(q), "seed": sd})
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
    return res, int(tr.sum()), int(te.sum())


def train():
    import lightgbm as lgb
    os.makedirs(PRED_DIR, exist_ok=True)
    if not (os.path.exists(XR_FN) and os.path.exists(ER_FN) and os.path.exists(DTE_FN)):
        raise SystemExit("run prep first")
    only = sys.argv[2:]          # optional: limit to units like raw_s1_2024Q1
    P, days, nxt, gapd = S99.calendar()
    del P
    gc.collect()
    for sd in SEEDS:             # seed-major, so a stopped run leaves both variants of seed 1 first
        for v in VARIANTS:
            todo = [q for q in QUARTERS if not os.path.exists(part_fn(v, sd, q))
                    and (not only or f"{v}_s{sd}_{q}" in only)]
            if not todo:
                continue
            X, names, dates, tick, y, E = load_variant(v)
            ok = ~np.isnan(y)
            D = (X, names, dates, tick, y, E, ok, np.asarray(dates), days, {})
            log("loaded", v, "rows", len(y), "features", len(names), "todo", [str(q) for q in todo])
            for q in todo:
                t0 = time.time()
                res, ntr, nte = fit_quarter(lgb, D, sd, q)
                fn = part_fn(v, sd, q)
                res.to_parquet(fn + ".tmp", index=False)
                os.replace(fn + ".tmp", fn)
                log(v, "seed", sd, q, "train", ntr, "test", nte, f"{time.time() - t0:.0f}s")
            del D, X, E, y
            gc.collect()
    log("train done")


# ------------------------------------------------------------------ (b) blends
def preds():
    parts = []
    for v in VARIANTS:
        for sd in SEEDS:
            fns = [part_fn(v, sd, q) for q in QUARTERS]
            have = [f for f in fns if os.path.exists(f)]
            if len(have) < len(fns):
                print("incomplete:", v, "seed", sd, len(have), "/", len(fns), flush=True)
            if not have:
                continue
            n = pd.concat([pd.read_parquet(f, columns=["date", "ticker", "q"] + MODELS) for f in have],
                          ignore_index=True)
            parts.append(n.assign(variant=v, seed=sd, complete=len(have) == len(fns)))
    pr = pd.concat(parts, ignore_index=True)
    pr["date"] = pd.to_datetime(pr.date)
    return pr


def all_picks(cols, days, grp):
    import study166_167_blend_parts as S166
    pr = preds()
    W, quarters = {}, {}
    for (v, sd), g in pr.groupby(["variant", "seed"], sort=False):
        quarters[(v, int(sd))] = sorted(g.q.unique())
        g = g.set_index(["date", "ticker"])
        e = g.per_beh.unstack().reindex(columns=cols).astype("float64")
        j = (g.p_jump - g.p_drop).unstack().reindex(columns=cols).astype("float64")
        ok = e.notna() & j.notna()
        S = (2 * e.where(ok).rank(axis=1, pct=True) + j.where(ok).rank(axis=1, pct=True)) / 3
        S = S.loc[S.index < days[-1]].loc["2024-01":"2026-09"]
        W[(v, int(sd))] = S166.capped_topk(S, grp)
        if (v, int(sd)) == ("raw", 1):
            S166.check_cap(S, W[(v, int(sd))])
    del pr
    gc.collect()
    return W, quarters


# ------------------------------------------------------------------ crosses
def fetch():
    import study94_cross_rescore as S94
    import study166_167_blend_parts as S166
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    S94.A.RL = A.RL
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    grp, _ = S166.industry_map(cols)
    del P
    W, _ = all_picks(cols, days, grp)
    need = pd.concat([S166.legs(Wx, days) for Wx in W.values()], ignore_index=True)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()           # includes auctions185.parquet
    tried = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(tried, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            r = S94._day_kind(job[0], job[1], job[2], ven)
            for x in r:
                x["rule"] = "venue"
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
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
    old = pd.read_parquet(AUC185) if os.path.exists(AUC185) else pd.DataFrame()
    new, t0 = [], time.time()
    for i in range(0, len(jobs), 12):
        with ThreadPoolExecutor(2) as exe:
            for r in exe.map(one, jobs[i:i + 12]):
                new += r
        if not new:
            continue
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC185 + ".tmp", index=False)
        os.replace(AUC185 + ".tmp", AUC185)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ (d) evaluation
def tstat(d):
    d = d.dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


def overlap(Wa, Wb, a0, b0):
    ha, hb = (Wa > 0).loc[a0:b0], (Wb > 0).loc[a0:b0]
    ha, hb = ha.align(hb, join="inner")
    return float(((ha & hb).sum(axis=1) / ha.sum(axis=1).replace(0, np.nan)).mean())


def check_rows():
    """raw seed 1 2024Q1 vs study 99 base 2024Q1 (differs only in days_to_e on a few rows)."""
    fn = part_fn("raw", 1, "2024Q1")
    if not os.path.exists(fn):
        return []
    a = pd.read_parquet(fn)
    b = pd.read_parquet(S99.PRED_FN, filters=[("variant", "==", "base"), ("q", "==", "2024Q1")])
    m = a.merge(b, on=["date", "ticker"], suffixes=("", "_99"))
    return [dict(table="check", variant="raw seed 1 2024Q1 vs study 99 base", set=k, n=len(m),
                 max_abs_diff=float((m[k] - m[k + "_99"]).abs().max()),
                 corr=float(np.corrcoef(m[k], m[k + "_99"])[0, 1])) for k in ["pooled_beh"] + MODELS]


def main():
    import study166_167_blend_parts as S166
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    grp, _ = S166.industry_map(cols)
    W, quarters = all_picks(cols, days, grp)
    full = [k for k, qs in quarters.items() if len(qs) == len(QUARTERS)]
    print("complete books", full, flush=True)
    index = W[("raw", 1)].index
    Rx, miss, C = S166.returns(P, cols, days, index)
    del P
    gc.collect()
    rows = check_rows()
    nets = {}
    for k, Wx in W.items():
        Wx = Wx.reindex(index=index).fillna(0.0)
        W[k] = Wx
        nets[k] = bt.run(Wx, Rx, C).net.where(Wx.sum(axis=1) > 0)
        h = Wx > 0
        for pl, a0, b0 in PER:
            hh = h.loc[a0:b0]
            if hh.values.sum() == 0:
                continue
            rows.append(dict(table="fallback", variant=k[0], seed=str(k[1]), period=pl, legs=int(hh.values.sum()),
                             share_panel=float((miss.loc[a0:b0] & hh).values.sum() / hh.values.sum())))
    for v in VARIANTS:
        if all((v, s) in nets for s in SEEDS):
            nets[(v, "avg book")] = pd.concat([nets[(v, s)] for s in SEEDS], axis=1).mean(axis=1, skipna=False)
    for k, net in nets.items():
        for pl, a0, b0 in PER:
            x = net.loc[a0:b0].dropna()
            if len(x) < 5:
                continue
            st = ann_stats(x)
            rows.append(dict(table="rescore", variant=k[0], seed=str(k[1]), period=pl, n=st["n"], sharpe=st["sharpe"],
                             net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"],
                             complete=k in full or (k[1] == "avg book" and all((k[0], s) in full for s in SEEDS))))
    R = pd.DataFrame([r for r in rows if r["table"] == "rescore"])
    for pl, _, _ in PER:
        g0 = R[(R.variant == "raw") & (R.period == pl) & R.seed.isin(["1", "2"])]
        for v in VARIANTS:
            g = R[(R.variant == v) & (R.period == pl) & R.seed.isin(["1", "2"])]
            if len(g) < 2:
                continue
            d = g.sharpe.mean() - g0.sharpe.mean() if len(g0) == 2 else np.nan
            rows.append(dict(table="seedsum", variant=v, period=pl, sharpe_mean=g.sharpe.mean(),
                             sharpe_seed2_minus_seed1=g.sharpe.iloc[1] - g.sharpe.iloc[0], net_bp_mean=g.net_bp.mean(),
                             maxdd_mean=g.maxdd.mean(), sharpe_mean_minus_raw=d, diff_in_noise_sd=d / NOISE_SD))
    for sd in SEEDS + ["avg book"]:
        if ("rank", sd) not in nets or ("raw", sd) not in nets:
            continue
        for pl, a0, b0 in PER:
            a_, b_ = nets[("rank", sd)].loc[a0:b0], nets[("raw", sd)].loc[a0:b0]
            d = (a_ - b_).dropna()
            if len(d) < 5:
                continue
            rows.append(dict(table="paired", variant="rank", seed=str(sd), set="vs raw same seed", period=pl,
                             n=len(d), diff_bp=1e4 * d.mean(), t_diff=tstat(d), corr=float(a_.corr(b_))))
    for v in VARIANTS:
        if (v, 1) in nets and (v, 2) in nets:
            for pl, a0, b0 in PER:
                a_, b_ = nets[(v, 2)].loc[a0:b0], nets[(v, 1)].loc[a0:b0]
                d = (a_ - b_).dropna()
                if len(d) < 5:
                    continue
                rows.append(dict(table="paired", variant=v, seed="2 vs 1", set="seed noise reference", period=pl,
                                 n=len(d), diff_bp=1e4 * d.mean(), t_diff=tstat(d), corr=float(a_.corr(b_))))
    for pl, a0, b0 in PER:
        for sd in SEEDS:
            if ("rank", sd) in W and ("raw", sd) in W:
                rows.append(dict(table="overlap", variant="rank", seed=str(sd), set="vs raw same seed", period=pl,
                                 share=overlap(W[("rank", sd)], W[("raw", sd)], a0, b0)))
        for v in VARIANTS:
            if (v, 1) in W and (v, 2) in W:
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
    {"prep": prep, "train": train, "fetch": fetch, "eval": main}[cmd]()
