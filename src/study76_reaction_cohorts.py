"""Study 76: cohort models by earnings behaviour and news reaction (reaction cohorts).

Question: the ensemble (src/ensemble.py, study 14/23) groups stocks by trading profile (behavior) and co-movement.
Do cohorts built from HOW a stock reacts to information (earnings, news days, overnight vs day split, gap reversal)
help the pooled overnight ranker, as extra inputs or as per-cohort models, and do they add to the ensemble?

Cohorts for year Y (2022 .. 2026) use only the two calendar years before Y (like study 14). Returns are indexed by the
decision day t: N_t = open t+1 / close t - 1 (the traded overnight return), D_t = close t / open t - 1 (day session of
t); 'excess' = minus the equal-weight mean of the study-3 universe (data/ml_frame.parquet rows) on t. Per stock, over
the days it is in that universe (at least 400 days required, else cohort -1 = scored by the pooled-with-cohort model):
  e_abs     mean |excess reaction| to earnings (store 'earnings' calendar dates d; report time is missing for ~99%,
            so the reaction night is the larger |excess| of the night into d and the night after d: history only)
  e_sgn     mean signed excess reaction
  e_rev     post-earnings day-session reversal: slope of the excess day-session return right after the reaction night
            on the reaction (sum(day*react)/sum(react^2), clipped to [-2, 2]); negative = reversal
  nw_abs    log(mean |excess N_t| on news days / on no-news days); news day = n_news_t > 0 (news_features.NF.load():
            Benzinga articles in (15:45 t-1, 15:45 t])
  nw_sgn    mean excess N_t on news days - on no-news days
  nd_abs    log(mean |excess D_t| on news days / on no-news days)
  nd_sgn    mean excess D_t on news days - on no-news days
  n_share   overnight share of total return: sum N / (sum |N| + sum |D|) (raw returns)
  gap_rev   corr(excess N_t, excess D_{t+1}): gap followed by day-session reversal (negative) or continuation
Missing values (no earnings or no news in the window) -> cross-sectional median. Standardized, clipped at +-4,
k-means (n_init=10, seed 0) with k in 6..10 chosen each year by the silhouette score on that year's training stocks
(no later data). Variant: k fixed at 8 (the behavior-cluster count) is NOT run; the choice rule is fixed up front.

Models (study-3 design: close features from data/ml_frame.parquet, rank target of y_night, quarterly walk-forward
2022Q1 .. 2026Q3, train on rows before the quarter start minus a 10-day embargo, LightGBM parameters of
src/study10_ml.py with num_threads=2 (resource limit; same trees up to thread nondeterminism)):
  pooled_rc      pooled model + cohort id (categorical) + cohort-relative r1, r5, night1, intra1 + cohort mean r5
                 (exactly the study-14 group-feature construction)
  pooled_rcx     pooled_rc + the stock's own 9 standardized reaction statistics as numeric inputs (variant)
  per_rc         one model per cohort (num_leaves 31, min_data_in_leaf 500 as in study 14; cohorts with < 20,000
                 training rows and cohort -1 use pooled_rc's predictions)
Baseline: results/study10_pred_base.parquet (pooled, same design). No multiprocessing anywhere (LightGBM/OpenMP
then fork hangs): models are trained and scored in this one process.
Evaluation: top 10 by within-day percentile rank, closing-auction buy, opening-auction sell, core.exec_cost_bps
'auction' + 2.5 bp per side (study 10). Periods 2022-23, 2024-25H1, 2025H2-26 (to 2026-09-25). Paired stationary
block bootstrap (block 10 days, 2000 draws) of the daily net return difference vs the baseline: Sharpe difference,
95% interval, one-sided p = share of draws with difference <= 0. Rank IC = mean daily Spearman correlation with N_t.
Ensemble blend (2024-01 .. 2026-09): results/study23_pred.parquet 'ensemble' (15:45 features, the traded score)
combined with the new member's within-day percentile rank: six-member mean (5 * ensemble + member) / 6 and 50/50.
TIMING MISMATCH: the new members use official-close features (look-ahead of the last 15 minutes vs 15:45), so a
control blends the ensemble with the close-feature pooled baseline (study10_pred_base) the same way; only the
difference between the member blend and the control blend is attributable to the cohorts.
Output: results/study76_reaction_cohorts.csv (sections: cohorts, models, blend), results/study76_groups.parquet,
results/study76_pred_<model>.parquet
    python src/study76_reaction_cohorts.py [train|eval|all]   (default all)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store
import bt
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
SMALL = dict(PARAMS, num_leaves=31, min_data_in_leaf=500)
STATS = ["e_abs", "e_sgn", "e_rev", "nw_abs", "nw_sgn", "nd_abs", "nd_sgn", "n_share", "gap_rev"]
END = pd.Timestamp("2026-09-25")
PERIODS = [("2022-23", "2022-01-01", "2023-12-31"), ("2024-25H1", "2024-01-01", "2025-06-30"),
           ("2025H2-26", "2025-07-01", "2026-09-30"), ("2024-26", "2024-01-01", "2026-09-30")]

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
o, c = P["o"][cols].astype("float64"), P["c"][cols].astype("float64")


def load_X():
    X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
    X = X[X.index.get_level_values(0) >= "2020-01-01"]
    X = X.drop(columns=[k for k in X.columns if k.startswith("y_") and k != "y_night"])
    return X


def reaction_stats(U):
    """U: bool panel (days x cols) of the study-3 universe. Returns {Y: DataFrame stocks x STATS (raw)}."""
    import news_features as NF
    N = o.shift(-1) / c - 1
    D = c / o - 1
    Nx = N.where(U)
    Dx = D.where(U)
    Ne = Nx.sub(Nx.mean(axis=1), axis=0)
    De = Dx.sub(Dx.mean(axis=1), axis=0)
    news = NF.load()["n_news"].reindex(index=days, columns=cols) > 0
    E = store.read("earnings")[["symbol", "date"]]
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E = E[E.symbol.isin(cols)].drop_duplicates()
    k = days.searchsorted(E.date.values)
    E = E[(k > 0) & (k < len(days) - 1)]
    k = days.searchsorted(E.date.values)
    ci = {t: i for i, t in enumerate(cols)}
    j = E.symbol.map(ci).values
    Nev, Dev = Ne.values, De.values
    pre, post = Nev[k - 1, j], Nev[k, j]               # night into d (BMO) / night after d (AMC)
    amc = np.abs(post) >= np.abs(pre)
    amc = np.where(np.isnan(pre), True, np.where(np.isnan(post), False, amc))
    react = np.where(amc, post, pre)
    kd = np.where(amc, np.minimum(k + 1, len(days) - 1), k)  # session right after the reaction night
    dayafter = Dev[kd, j]
    ev = pd.DataFrame({"ticker": E.symbol.values, "date": E.date.values, "react": react, "dayafter": dayafter})
    out = {}
    for Y in range(2022, 2027):
        a, b = pd.Timestamp(f"{Y - 2}-01-01"), pd.Timestamp(f"{Y - 1}-12-31")
        sl = (days >= a) & (days <= b)
        ne, de, nr, dr, nw, u = Ne[sl], De[sl], Nx[sl], Dx[sl], news[sl], U[sl]
        okd = (ne.notna() & de.notna()).sum()
        keep = okd.index[okd >= 400]
        ne, de, nr, dr, nw = ne[keep], de[keep], nr[keep], dr[keep], nw[keep]
        st = pd.DataFrame(index=keep)
        x = ev[(ev.date >= a) & (ev.date <= b) & ev.ticker.isin(keep)].dropna()
        g = x.groupby("ticker")
        st["e_abs"] = g.react.apply(lambda s: s.abs().mean())
        st["e_sgn"] = g.react.mean()
        st["e_rev"] = (g.apply(lambda d: (d.dayafter * d.react).sum() / max((d.react ** 2).sum(), 1e-8),
                               include_groups=False)).clip(-2, 2)
        ann, anw = ne.abs().where(nw), ne.abs().where(~nw & ne.notna())
        st["nw_abs"] = np.log(ann.mean() / anw.mean())
        st["nw_sgn"] = ne.where(nw).mean() - ne.where(~nw).mean()
        dnn, dnw = de.abs().where(nw), de.abs().where(~nw & de.notna())
        st["nd_abs"] = np.log(dnn.mean() / dnw.mean())
        st["nd_sgn"] = de.where(nw).mean() - de.where(~nw).mean()
        st["n_share"] = nr.sum() / (nr.abs().sum() + dr.abs().sum())
        st["gap_rev"] = ne.corrwith(de.shift(-1))
        st = st.replace([np.inf, -np.inf], np.nan)
        out[Y] = st
        print(Y, "stocks", len(st), "with earnings", int(st.e_abs.notna().sum()), "with news-day stats",
              int(st.nw_abs.notna().sum()), flush=True)
    return out


def cluster(stats):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    G, Z, info = {}, {}, []
    for Y, st in stats.items():
        s = st.fillna(st.median())
        z = ((s - s.mean()) / s.std()).clip(-4, 4)
        best = None
        for k in range(6, 11):
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(z.values)
            sc = silhouette_score(z.values, lab, sample_size=min(3000, len(z)), random_state=0)
            info.append(dict(section="cohorts", year=Y, k=k, silhouette=sc))
            if best is None or sc > best[0]:
                best = (sc, k, lab)
        G[Y] = pd.Series(best[2], index=z.index)
        Z[Y] = z.astype("float32")
        prof = z.groupby(G[Y]).mean().round(2)
        prof["n"] = G[Y].value_counts().sort_index()
        print(Y, "chosen k", best[1], "silhouette", round(best[0], 3), "\n", prof.to_string(), flush=True)
        for gi, row in prof.iterrows():
            info.append(dict(section="cohorts", year=Y, k=best[1], chosen=True, cohort=gi, n=int(row.n),
                             **{f"z_{k_}": row[k_] for k_ in STATS}))
    return G, Z, info


def add_cohort_features(Xb, gid):
    Xg = Xb.copy()
    Xg["c_id"] = gid.astype("float32")
    d = Xb.index.get_level_values(0)
    gm = Xb.groupby([d, gid])
    for k in ["r1", "r5", "night1", "intra1"]:
        Xg[f"c_{k}_rel"] = (Xb[k] - gm[k].transform("mean")).astype("float32")
    Xg["c_r5_mean"] = gm["r5"].transform("mean").astype("float32")
    return Xg


def train():
    import lightgbm as lgb
    X = load_X()
    feat = [k for k in X.columns if not k.startswith("y_")]
    dates, tick = X.index.get_level_values(0), X.index.get_level_values(1)
    U = pd.Series(True, index=X.index).unstack().reindex(index=days, columns=cols).fillna(False).astype(bool)
    stats = reaction_stats(U)
    G, Z, info = cluster(stats)
    pd.concat({Y: pd.concat([G[Y].rename("cohort"), Z[Y]], axis=1) for Y in G}).to_parquet(
        f"{RES}/study76_groups.parquet")
    pd.DataFrame(info).to_csv(f"{RES}/_study76_cohorts_tmp.csv", index=False)
    y = X["y_night"]
    yr = (y.groupby(level=0).rank(pct=True) - 0.5).values
    ok = y.notna().values
    preds = {"pooled_rc": [], "pooled_rcx": [], "per_rc": []}
    for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
        g = G[q.year]
        gid = pd.Series(tick).map(g).fillna(-1).values
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dates < cut)
        te = np.asarray((dates >= q.start_time) & (dates <= q.end_time))
        if te.sum() == 0:
            continue
        Xg = add_cohort_features(X[feat], gid)
        mdl = lgb.train(PARAMS, lgb.Dataset(Xg[tr], yr[tr], categorical_feature=["c_id"], free_raw_data=True),
                        num_boost_round=300)
        p_rc = pd.Series(mdl.predict(Xg[te]), index=X.index[te])
        preds["pooled_rc"].append(p_rc)
        # variant: own reaction statistics as numeric inputs (year q.year's table, NaN for unclustered stocks)
        zt = Z[q.year].reindex(tick)
        for k in STATS:
            Xg["s_" + k] = zt[k].values
        mdx = lgb.train(PARAMS, lgb.Dataset(Xg[tr], yr[tr], categorical_feature=["c_id"], free_raw_data=True),
                        num_boost_round=300)
        preds["pooled_rcx"].append(pd.Series(mdx.predict(Xg[te]), index=X.index[te]))
        del Xg, mdx
        per = []
        for k in sorted(set(gid)):
            m = gid == k
            if k == -1 or (tr & m).sum() < 20000:
                per.append(p_rc[m[te]])
                continue
            mg = lgb.train(SMALL, lgb.Dataset(X.loc[tr & m, feat], yr[tr & m]), num_boost_round=300)
            per.append(pd.Series(mg.predict(X.loc[te & m, feat]), index=X.index[te & m]))
        preds["per_rc"].append(pd.concat(per))
        print(q, "done; cohorts", len(set(gid)) - (1 if -1 in gid else 0), flush=True)
        for name, v in preds.items():           # save as we go (restartable inspection)
            pd.DataFrame({"pred": pd.concat(v)}).to_parquet(f"{RES}/study76_pred_{name}.parquet")


def paired_boot(a, b, block=10, n=2000, seed=0):
    """a, b: aligned daily return arrays. Sharpe(a) - Sharpe(b): point, 2.5/97.5 pct, one-sided p (diff <= 0)."""
    a, b = np.asarray(a), np.asarray(b)
    T = len(a)
    rng = np.random.default_rng(seed)
    sr = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    d0 = sr(a) - sr(b)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        out[i] = sr(a[idx]) - sr(b[idx])
    return d0, np.percentile(out, 2.5), np.percentile(out, 97.5), float((out <= 0).mean())


def evaluate():
    R = o.shift(-1) / c - 1
    cost = exec_cost_bps(P, "auction")[cols] + 2.5
    Rr = R.rank(axis=1, pct=True)

    def book(S, k=10):
        S = S.reindex(columns=cols)
        S = S.loc[(S.index < days[-1]) & (S.index <= END)]
        W = bt.select_topk(S, S.notna(), k)
        return bt.run(W, R.reindex_like(W), cost.reindex_like(W)), S

    rows = []
    if os.path.exists(f"{RES}/_study76_cohorts_tmp.csv"):
        rows += pd.read_csv(f"{RES}/_study76_cohorts_tmp.csv").to_dict("records")
    base = pd.read_parquet(f"{RES}/study10_pred_base.parquet")["pred"].unstack().rank(axis=1, pct=True)
    models = {"pooled_base": base}
    for m in ["pooled_rc", "pooled_rcx", "per_rc"]:
        models[m] = pd.read_parquet(f"{RES}/study76_pred_{m}.parquet")["pred"].unstack().rank(axis=1, pct=True)
    books = {}
    for m, S in models.items():
        for k in [10, 20]:
            books[(m, k)], S_ = book(S, k)
        ic = S_.corrwith(Rr.reindex_like(S_), axis=1)
        for k in [10, 20]:
            r = books[(m, k)]
            rb = books[("pooled_base", k)]
            for per, a, b in PERIODS:
                x, xb = r.net.loc[a:b], rb.net.loc[a:b]
                ix = x.index.intersection(xb.index)
                st = ann_stats(x)
                row = dict(section="models", model=m, k=k, period=per, n_days=len(x), sharpe=st["sharpe"],
                           ann_ret=st["ann_ret"], maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[a:b].mean(),
                           net_bps=1e4 * x.mean(), rank_ic=float(ic.loc[a:b].mean()))
                if m != "pooled_base":
                    d0, lo, hi, p = paired_boot(x.loc[ix].values, xb.loc[ix].values)
                    row.update(d_sharpe=d0, d_lo=lo, d_hi=hi, p_boot=p)
                rows.append(row)
    # ---- ensemble blend (15:45 ensemble score + close-feature member)
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack()
    ens = ens.loc[ens.index < days[-1]]
    er = ens.rank(axis=1, pct=True)
    blends = {"ensemble": er}
    for m in ["pooled_base", "pooled_rc", "pooled_rcx", "per_rc"]:
        mr = models[m].reindex_like(er)
        ok = er.notna() & mr.notna()
        blends[f"ens6+{m}"] = ((5 * er + mr) / 6).where(ok)
        blends[f"ens50+{m}"] = ((er + mr) / 2).where(ok)
    bb = {n: book(S, 10)[0] for n, S in blends.items()}
    for n, r in bb.items():
        ctrl = None
        if n != "ensemble":
            ctrl = n.split("+")[0] + "+pooled_base"
        for per, a, b in PERIODS[1:]:
            x = r.net.loc[a:b]
            st = ann_stats(x)
            row = dict(section="blend", model=n, k=10, period=per, n_days=len(x), sharpe=st["sharpe"],
                       ann_ret=st["ann_ret"], maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[a:b].mean(),
                       net_bps=1e4 * x.mean())
            if n != "ensemble":
                xe = bb["ensemble"].net.loc[a:b]
                ix = x.index.intersection(xe.index)
                d0, lo, hi, p = paired_boot(x.loc[ix].values, xe.loc[ix].values)
                row.update(d_sharpe=d0, d_lo=lo, d_hi=hi, p_boot=p)
                if ctrl != n:
                    xc = bb[ctrl].net.loc[a:b]
                    d0, lo, hi, p = paired_boot(x.loc[ix].values, xc.loc[ix].values)
                    row.update(d_sharpe_vs_ctrl=d0, dc_lo=lo, dc_hi=hi, p_boot_vs_ctrl=p)
            rows.append(row)
    # pick overlap with baseline / ensemble
    for m in ["pooled_rc", "pooled_rcx", "per_rc"]:
        W1 = bt.select_topk(models[m], models[m].notna(), 10) > 0
        W0 = bt.select_topk(models["pooled_base"], models["pooled_base"].notna(), 10).reindex_like(W1).fillna(0) > 0
        rows.append(dict(section="models", model=m, k=10, period="overlap_with_base",
                         net_bps=float((W1 & W0).sum(axis=1).mean() / 10)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study76_reaction_cohorts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    v = df[df.section == "models"]
    print(v[v.period != "overlap_with_base"].pivot_table(index=["model", "k"], columns="period",
                                                          values=["sharpe", "net_bps", "rank_ic"]).round(3).to_string())
    print(v[v.k == 10].dropna(subset=["d_sharpe"])[["model", "period", "d_sharpe", "d_lo", "d_hi", "p_boot"]]
          .round(3).to_string())
    print(v[v.period == "overlap_with_base"][["model", "net_bps"]].to_string())
    vb = df[df.section == "blend"]
    print(vb[["model", "period", "sharpe", "net_bps", "maxdd", "d_sharpe", "p_boot", "d_sharpe_vs_ctrl",
              "p_boot_vs_ctrl"]].round(3).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "cohorts":                       # quick check of the cohort step only
        Xi = pd.read_parquet(f"{DATA}/ml_frame.parquet", columns=["r1"])
        Xi = Xi[Xi.index.get_level_values(0) >= "2020-01-01"]
        U = pd.Series(True, index=Xi.index).unstack().reindex(index=days, columns=cols).fillna(False).astype(bool)
        cluster(reaction_stats(U))
    if mode in ("train", "all"):
        train()
    if mode in ("eval", "all"):
        evaluate()
