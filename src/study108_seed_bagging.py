"""Study 108: does averaging several LightGBM seeds (bagging) make the blend steadier or better at official cross prices?

Question: the live blend's models are retrained per quarter with one LightGBM seed. Study 99 retrained the strongest
ensemble member (per_beh, with pooled_beh for stocks outside the large behavior groups) and the study-33 jump / drop
classifiers walk-forward on close features and found that changing only the seed moved the blend from Sharpe 2.24 to
2.52 at cross prices (2024-26; +3.4 bp a night, t 0.65) and kept only 58% of the picks. Does averaging each model's
predictions over several seeds give a blend that is better, or at least less dependent on the seed?

Data: study 99's training frame and code (data/ml_frame.parquet close features, y_night = close -> next panel open;
earn_tonight / news_today from data/local/study99_extra.parquet; behavior groups results/study14_groups.parquet) and
its cached predictions data/local/study99_pred.parquet, variants "base" (LightGBM default seed, called seed 1 here)
and "base_seed2" (seed 2). Official crosses from studies 94 / 88 / 99 (data/local/auctions94.parquet,
auctions88.parquet, auctions99.parquet) plus crosses fetched here for new picks (data/local/auctions108.parquet,
study 94's rule: listing-venue 'O' / '6' print, second pass largest print of any listing exchange; Alpaca SIP, rate
limited).

Design:
  (a) Train seeds 3, 4, 5 exactly as study 99's base variant (walk-forward by quarter, test 2024Q1..2026Q3, rows from
      2020 before the quarter start minus a 10-day embargo, same parameters, num_threads=2, panel target), changing
      only LightGBM's master seed. Seed-major order, so a run stopped early leaves complete seeds.
  (b) Blends: each single-seed blend (2 x rank(per_beh) + rank(P(jump) - P(drop))) / 3 for seeds 1..5; the
      seed-averaged blend (each model's predictions averaged over the seeds, then blended); and 2-seed averages for
      the three ways of splitting seeds 1..4 into two pairs ({1,2} vs {3,4}, {1,3} vs {2,4}, {1,4} vs {2,3}).
  (c) Top 10, buy at the closing cross, sell at the next opening cross; returns at crosses as study 94 / 99 (panel
      return x cross / panel ratio per leg, panel price where no usable cross). Costs exec_cost_bps(P, "auction") +
      2.5 bp per side. Per period (2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09): Sharpe, mean net bp, max
      drawdown per blend; mean, standard deviation, min and max over the single seeds; paired t of the averaged blend
      against the day-by-day mean of the single-seed blends and against seed 1; pick overlap between seeds and with
      the averaged blend; turnover (share of the 10 picks not held the night before).
  (d) Seed noise of averages: for single seeds (all 10 pairs) and for the 3 pairs of 2-seed averages, the absolute
      Sharpe and net-bp difference between the two members, their daily-return correlation, the standard deviation of
      the daily difference, and pick overlap. If averaging removes seed noise these move towards 0 / 1.

Data checks: seeds 1 and 2 rescored here reproduce study 99 exactly (2024-26 at crosses 2.24 / 2.52); 5,248 new
(ticker, day, kind) crosses fetched, 99.6% found; legs at the panel price 0.5-1.9% for every blend.

Result (2024-01..2025-06 / 2025-07..2026-09 / 2024-26, at crosses): single seeds 1..5 Sharpe 2.38 / 2.12 / 2.24,
2.53 / 2.52 / 2.52, 2.13 / 2.13 / 2.12, 2.24 / 1.61 / 1.91, 1.97 / 2.10 / 2.03 (mean 2.25 / 2.10 / 2.16, sd 0.22 /
0.33 / 0.23); net 30.9 / 34.5 / 32.6 bp mean over seeds (sd 3.1 / 5.7 / 3.9). Seed-averaged blend (5 seeds) 2.30 /
2.14 / 2.22, 31.5 / 34.8 / 33.0 bp, max drawdown -19.2% / -24.3% / -24.3% (single seeds -14.7..-22.5% over 2024-26).
Paired vs the day-by-day mean of the single seeds +0.6 / +0.3 / +0.5 bp (t 0.16 / 0.08 / 0.17); vs seed 1 -1.7 bp
(t -0.38). Seeds 1 and 2 (study 99) were the two best of five, so study 99's base retrain is a lucky draw. Pick
overlap between single seeds 58%, between each seed and the average 71%; turnover 0.854 vs 0.860. Seed noise of
2-seed averages vs single seeds (2024-26): pick overlap 68% vs 58%, daily correlation 0.86 vs 0.82, sd of the daily
difference 128 vs 143 bp, |Sharpe difference| 0.15 vs 0.28 (3 pairs only, and not smaller in 2025-07..2026-09: 0.43
vs 0.37). Holding all five single-seed books at 1/5 each (50 slots, daily average of returns) gives 2.45 / 2.25 /
2.34 at crosses, above every single seed but seed 2 and above the averaged blend (lead only, not tested further).
Verdict: reject as an improvement (no gain in mean return); averaging reduces the dependence of the picks on the seed
only partly and does not lower drawdown.

Output: results/study108_seed_bagging.csv (tables: rescore = per blend, pricing, period; seedsum = mean / sd / min /
max over single seeds; paired = paired t-tests; overlap = pick overlap; noise = replicate-pair comparisons;
fallback = share of legs priced at the panel). Seed predictions in data/local/study108_pred.parquet (resumable).

    python src/study108_seed_bagging.py train    # seeds 3..5 (hours; resumable by seed and quarter)
    python src/study108_seed_bagging.py fetch    # official crosses for new picks (resumable)
    python src/study108_seed_bagging.py          # evaluation
"""
import gc
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import stock_cols, exec_cost_bps, ann_stats, RES   # noqa: E402
import bt   # noqa: E402
import study99_cross_retrain as S99   # noqa: E402

LOC = S99.LOC
AUC108 = os.path.join(LOC, "auctions108.parquet")
PRED_FN = os.path.join(LOC, "study108_pred.parquet")
OUT = os.path.join(RES, "study108_seed_bagging.csv")
PER = S99.PER
QUARTERS = S99.QUARTERS
NEW_SEEDS = [3, 4, 5]
MODELS = ["per_beh", "p_jump", "p_drop"]


# ------------------------------------------------------------------ (a) train
def train():
    import lightgbm as lgb
    P, days, nxt, gapd = S99.calendar()
    X, names, dates, tick, y = S99.load_frame()
    E = pd.read_parquet(S99.EXTRA_FN).values.astype("float32")
    ok = ~np.isnan(y)
    dnp = np.asarray(dates)
    print("rows", len(y), "features", len(names), flush=True)
    done = pd.read_parquet(PRED_FN) if os.path.exists(PRED_FN) else None
    have = set(zip(done.q, done.seed)) if done is not None else set()
    out = [done] if done is not None else []
    gcache = {}
    for sd in NEW_SEEDS:
        for q in QUARTERS:
            if (str(q), sd) in have:
                continue
            t0 = time.time()
            cut = days[max(0, days.searchsorted(q.start_time) - 11)]
            tr = ok & (dnp < np.datetime64(cut))
            te = (dnp >= np.datetime64(q.start_time)) & (dnp <= np.datetime64(q.end_time.normalize()))
            if q.year not in gcache:
                gcache.clear()
                gc.collect()
                gcache[q.year] = S99.group_feats(X, names, dates, tick, q.year)
            gid, GF, gf_names = gcache[q.year]
            yr = (pd.Series(np.where(tr, y, np.nan)).groupby(dnp).rank(pct=True).values - 0.5).astype("float32")
            res = pd.DataFrame({"date": dates[te], "ticker": tick[te], "q": str(q), "seed": sd})
            # pooled_beh
            ds = lgb.Dataset(S99.mat(tr, [X, GF]), yr[tr], feature_name=names + gf_names, categorical_feature=["g_id"],
                             free_raw_data=True)
            m = lgb.train(dict(S99.PARAMS, seed=sd), ds, num_boost_round=300)
            del ds
            pb = m.predict(S99.mat(te, [X, GF])).astype("float32")
            res["pooled_beh"] = pb
            del m
            gc.collect()
            # per_beh: one SMALL model per group with >= 20,000 training rows; others keep pooled_beh
            per = pb.copy()
            gte = gid[te]
            for gg in sorted(set(gid)):
                mm = gid == gg
                if gg == -1 or (tr & mm).sum() < 20000:
                    continue
                ds = lgb.Dataset(S99.mat(tr & mm, [X]), yr[tr & mm], feature_name=names, free_raw_data=True)
                mg = lgb.train(dict(S99.SMALL, seed=sd), ds, num_boost_round=300)
                del ds
                per[gte == gg] = mg.predict(S99.mat(te & mm, [X]))
                del mg
                gc.collect()
            res["per_beh"] = per
            # jump / drop classifiers (study-3 inputs + earn_tonight + news_today)
            jn = names + ["earn_tonight", "news_today"]
            Xte = S99.mat(te, [X, E])
            for k, lab in [("jump", y > 0.05), ("drop", y < -0.05)]:
                ds = lgb.Dataset(S99.mat(tr, [X, E]), lab[tr].astype("float32"), feature_name=jn, free_raw_data=True)
                mj = lgb.train(dict(S99.PR_BIN, seed=sd), ds, num_boost_round=300)
                del ds
                res[f"p_{k}"] = mj.predict(Xte).astype("float32")
                del mj
                gc.collect()
            del Xte
            out.append(res)
            pd.concat(out, ignore_index=True).to_parquet(PRED_FN, index=False)
            print("seed", sd, q, "train", int(tr.sum()), "test", int(te.sum()), f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ (b) blends
def seed_panels(cols, days):
    """{seed: {model: date x ticker panel}} for seeds 1 (study 99 base), 2 (base_seed2) and the trained new seeds
    with all 11 quarters."""
    pr = pd.read_parquet(S99.PRED_FN, columns=["date", "ticker", "variant"] + MODELS)
    pr = pr[pr.variant.isin(["base", "base_seed2"])]
    pr["seed"] = pr.variant.map({"base": 1, "base_seed2": 2})
    parts = [pr.drop(columns="variant")]
    if os.path.exists(PRED_FN):
        n = pd.read_parquet(PRED_FN, columns=["date", "ticker", "q", "seed"] + MODELS)
        full = n.groupby("seed").q.nunique()
        n = n[n.seed.isin(full[full == len(QUARTERS)].index)]
        parts.append(n.drop(columns="q"))
    pr = pd.concat(parts, ignore_index=True)
    del parts
    pr["date"] = pd.to_datetime(pr.date)
    out = {}
    for sd, g in pr.groupby("seed"):
        g = g.set_index(["date", "ticker"])
        out[int(sd)] = {k: g[k].unstack().reindex(columns=cols).astype("float64") for k in MODELS}
    idx = out[1]["per_beh"].index
    for sd in out:
        for k in MODELS:
            out[sd][k] = out[sd][k].reindex(index=idx)
            out[sd][k] = out[sd][k].loc[out[sd][k].index < days[-1]]
    return out


def blend(D, seeds):
    """Blend score from the seed-averaged predictions of `seeds`."""
    e = sum(D[s]["per_beh"] for s in seeds) / len(seeds)
    j = sum(D[s]["p_jump"] - D[s]["p_drop"] for s in seeds) / len(seeds)
    ok = e.notna() & j.notna()
    return (2 * e.where(ok).rank(axis=1, pct=True) + j.where(ok).rank(axis=1, pct=True)) / 3


def lab(seeds):
    return f"seed {seeds[0]}" if len(seeds) == 1 else "avg {" + ",".join(map(str, seeds)) + "}"


def configs(seeds):
    cf = [(s,) for s in seeds] + [tuple(seeds)]
    if len(seeds) >= 4:
        for a, b in [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]:
            cf += [a, b]
    return list(dict.fromkeys(cf))


def all_picks(cols, days):
    D = seed_panels(cols, days)
    seeds = sorted(D)
    W = {}
    for c in configs(seeds):
        S = blend(D, c).loc["2024-01":"2026-09"]
        W[lab(c)] = bt.select_topk(S, S.notna(), 10)
    del D
    gc.collect()
    return seeds, W


# ------------------------------------------------------------------ crosses
def crosses():
    a = S99.crosses()
    if os.path.exists(AUC108):
        c = pd.read_parquet(AUC108)[["ticker", "date", "kind", "px"]]
        c["date"] = pd.to_datetime(c.date)
        c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
        a = pd.concat([a, c.dropna(subset=["px"])], ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    return a


def fetch():
    import study94_cross_rescore as S94   # sets the Alpaca rate limiter (ALPACA_PER_MIN, default 150 / min)
    P, days, nxt, gapd = S99.calendar()
    cols = stock_cols(P)
    seeds, W = all_picks(cols, days)
    print("seeds", seeds, "blends", list(W), flush=True)
    rows = []
    for Wv in W.values():
        s = Wv.stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(rows).dropna().drop_duplicates()
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    have = [crosses()[["ticker", "date", "kind"]]]
    for fn in [S99.AUC99, S99.AUC94, AUC108]:            # pairs tried before, found or not
        if os.path.exists(fn):
            t2 = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t2["ticker"] = t2.ticker.astype(str).str.replace(".", "-", regex=False)
            have.append(t2.assign(date=pd.to_datetime(t2.date)))
    have = pd.concat(have).drop_duplicates()
    m = need.merge(have, how="left", indicator=True)
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
            if miss:              # second pass: largest print of any listing exchange
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    x["rule"] = "venue"
                    if x["ticker"] in lg and not np.isnan(lg[x["ticker"]]["px"]):
                        x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x", "rule"]})
                    elif x["ticker"] in lg:
                        x["rule"] = "largest"
            for x in r:
                x.setdefault("rule", "venue")
            return r
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC108) if os.path.exists(AUC108) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(4) as exe:
            for r in exe.map(one, jobs[i:i + 24]):
                new += r
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC108, index=False)
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}" if new else "", f"{time.time() - t0:.0f}s",
              flush=True)


# ------------------------------------------------------------------ (c, d) evaluation
def tstat(d):
    d = d.dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


def overlap(Wa, Wb, a0, b0):
    ha, hb = (Wa > 0).loc[a0:b0], (Wb > 0).loc[a0:b0]
    return float(((ha & hb).sum(axis=1) / ha.sum(axis=1).replace(0, np.nan)).mean())


def main():
    P, days, nxt, gapd = S99.calendar()
    cols = stock_cols(P)
    seeds, W = all_picks(cols, days)
    singles = [lab((s,)) for s in seeds]
    avg = lab(tuple(seeds))
    print("seeds", seeds, flush=True)
    a = crosses()
    ro, rc, err = S99.cross_ratios(P, cols, days, a)
    print("cross data errors (legs set to the panel price):", err, flush=True)
    rows = [dict(table="data_errors", set=k, n=v["found"], scale_mismatch=v["scale"], ratio_out_of_range=v["ratio"])
            for k, v in err.items()]
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = o.shift(-1) / c - 1
    Rn_x = (1 + Rn_p) * ro.fillna(1.0).shift(-1) / rc.fillna(1.0) - 1
    miss = ro.shift(-1).isna() | rc.isna()
    C = exec_cost_bps(P, "auction")[cols] + 2.5
    nets = {}
    for name, Wx in W.items():
        h = Wx > 0
        turn = 1 - (h & h.shift(1, fill_value=False)).sum(axis=1) / h.sum(axis=1).replace(0, np.nan)
        for pr, R in [("panel", Rn_p), ("cross", Rn_x)]:
            net = bt.run(Wx, R.reindex_like(Wx), C.reindex_like(Wx)).net
            nets[(name, pr)] = net
            for pl, a0, b0 in PER:
                x = net.loc[a0:b0].dropna()
                st = ann_stats(x)
                rows.append(dict(table="rescore", variant=name, pricing=pr, period=pl, n=st["n"], sharpe=st["sharpe"],
                                 net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"],
                                 turnover=float(turn.loc[a0:b0].iloc[1:].mean())))
        for pl, a0, b0 in PER:
            hh = h.loc[a0:b0]
            mm = miss.reindex_like(Wx).fillna(True).loc[a0:b0] & hh
            rows.append(dict(table="fallback", variant=name, period=pl, legs=int(hh.sum().sum()),
                             share_panel=mm.sum().sum() / max(hh.sum().sum(), 1)))
    R = pd.DataFrame([r for r in rows if r["table"] == "rescore"])
    # single-seed summary
    for pr in ["panel", "cross"]:
        for pl, a0, b0 in PER:
            g = R[(R.pricing == pr) & (R.period == pl) & R.variant.isin(singles)]
            for k in ["sharpe", "net_bp", "maxdd", "turnover"]:
                rows.append(dict(table="seedsum", pricing=pr, period=pl, set=k, n=len(g), mean=g[k].mean(),
                                 sd=g[k].std(), min=g[k].min(), max=g[k].max(),
                                 averaged=float(R[(R.pricing == pr) & (R.period == pl) & (R.variant == avg)][k].iloc[0])))
    # paired tests
    for pr in ["panel", "cross"]:
        smean = pd.concat([nets[(s, pr)] for s in singles], axis=1).mean(axis=1)
        nets[("single-seed mean", pr)] = smean
        for pl, a0, b0 in PER:
            x = smean.loc[a0:b0].dropna()
            st = ann_stats(x)
            rows.append(dict(table="rescore", variant="single-seed mean (daily average of single-seed returns)",
                             pricing=pr, period=pl, n=st["n"], sharpe=st["sharpe"], net_bp=1e4 * x.mean(),
                             ann=st["ann_ret"], maxdd=st["maxdd"]))
            for name in [avg] + [n for n in W if n.startswith("avg") and n != avg]:
                for ref in ["single-seed mean", "seed 1"]:
                    d = (nets[(name, pr)] - nets[(ref, pr)]).loc[a0:b0].dropna()
                    rows.append(dict(table="paired", variant=name, set=f"vs {ref}", pricing=pr, period=pl, n=len(d),
                                     diff_bp=1e4 * d.mean(), t_diff=tstat(d)))
    # overlap
    for pl, a0, b0 in PER:
        pairs = list(itertools.combinations(singles, 2))
        for x, z in pairs:
            rows.append(dict(table="overlap", variant=x, set=f"vs {z}", period=pl, share=overlap(W[x], W[z], a0, b0)))
        rows.append(dict(table="overlap", variant="single seeds", set="mean over pairs", period=pl,
                         share=np.mean([overlap(W[x], W[z], a0, b0) for x, z in pairs])))
        for x in singles:
            rows.append(dict(table="overlap", variant=x, set=f"vs {avg}", period=pl, share=overlap(W[x], W[avg], a0, b0)))
    # noise: replicate pairs (single seeds vs pairs of 2-seed averages)
    two = [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))] if len(seeds) >= 4 else []
    groups_ = [("single seed", list(itertools.combinations(singles, 2))),
               ("single seed, seeds 1-4 only", list(itertools.combinations(singles[:4], 2))),
               ("2-seed average", [(lab(x), lab(z)) for x, z in two])]
    for pr in ["panel", "cross"]:
        for pl, a0, b0 in PER:
            for gname, prs in groups_:
                if not prs:
                    continue
                v = []
                for x, z in prs:
                    nx, nz = nets[(x, pr)].loc[a0:b0], nets[(z, pr)].loc[a0:b0]
                    d = (nx - nz).dropna()
                    v.append(dict(abs_sharpe_diff=abs(ann_stats(nx.dropna())["sharpe"] - ann_stats(nz.dropna())["sharpe"]),
                                  abs_bp_diff=1e4 * abs(d.mean()), corr=nx.corr(nz), sd_diff_bp=1e4 * d.std(),
                                  overlap=overlap(W[x], W[z], a0, b0)))
                    if gname == "2-seed average":
                        rows.append(dict(table="noise_pairs", variant=x, set=f"vs {z}", pricing=pr, period=pl, **v[-1]))
                v = pd.DataFrame(v).mean()
                rows.append(dict(table="noise", variant=gname, pricing=pr, period=pl, n=len(prs), **v.to_dict()))
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
    {"train": train, "fetch": fetch, "eval": main}[cmd]()
