"""Study 14: cohorts of stocks that behave alike, with a model per cohort or cohort features in one model.

Cohorts are re-estimated at the start of each year Y (2022 .. 2026) from the two years before Y only:
  comove    stocks whose daily returns move together: residual returns (stock minus the equal-weight market
            return), correlation matrix, top 15 principal components, k-means with 12 groups on the loadings
  behavior  stocks with similar trading profiles: log dollar volume, volatility, beta to SPY, share of the total
            return earned overnight, mean overnight return, autocorrelation of overnight returns (reversal),
            mean absolute opening gap; standardized, k-means with 8 groups
Stocks without two years of history go to group -1 and are scored by the pooled model.
Models (quarterly walk-forward 2022Q1 .. 2026Q3, 10-day embargo, same features and parameters as study 10):
  pooled            one model for all stocks (study 10 baseline predictions, results/study10_pred_base.parquet)
  per_<cohort>      one model per group, trained on that group's rows only
  pooled+<cohort>   one model for all stocks with the group id and group-relative returns added as features
Evaluation: top 10 of all predictions each day, closing-auction buy, opening-auction sell, auction costs +
2.5 bp per side; rank correlation of predictions with realized overnight returns.
Output: results/study14_clusters.csv, results/study14_groups.parquet
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
o, c, dv = P["o"][cols], P["c"][cols], P["dv"][cols]
r = c / c.shift(1) - 1
night = o / c.shift(1) - 1
spy = P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
feat = [k for k in X.columns if not k.startswith("y_")]
dates, tick = X.index.get_level_values(0), X.index.get_level_values(1)
live = sorted(set(tick))


def groups_for(Y):
    a, b = f"{Y - 2}-01-01", f"{Y - 1}-12-31"
    rr = r.loc[a:b, [t for t in live if t in r.columns]]
    ok = rr.notna().sum() >= 400
    rr = rr.loc[:, ok]
    res = rr.sub(rr.mean(1), axis=0).fillna(0.0)
    z = (res - res.mean()) / res.std().replace(0, np.nan)
    z = z.fillna(0.0).clip(-5, 5)
    load = PCA(15, random_state=0).fit(z.values).components_.T                 # stocks x 15
    load = load / np.linalg.norm(load, axis=1, keepdims=True).clip(1e-9)
    g1 = pd.Series(KMeans(12, n_init=10, random_state=0).fit_predict(load), index=rr.columns)
    nn = night.loc[a:b, rr.columns]
    beh = pd.DataFrame({
        "ldv": np.log(dv.loc[a:b, rr.columns].median().clip(lower=1e5)),
        "vol": rr.std(),
        "beta": rr.apply(lambda x: x.cov(spy.loc[a:b]) / spy.loc[a:b].var()),
        "night_share": nn.sum() / (np.abs(nn).sum() + np.abs(rr - nn).sum()),
        "night_mean": nn.mean(),
        "night_ac": nn.apply(lambda x: x.autocorr()),
        "gap_abs": nn.abs().mean()}).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    beh = (beh - beh.mean()) / beh.std()
    g2 = pd.Series(KMeans(8, n_init=10, random_state=0).fit_predict(beh.clip(-4, 4).values), index=rr.columns)
    return {"comove": g1, "behavior": g2}


G = {Y: groups_for(Y) for Y in range(2022, 2027)}
pd.concat({(Y, k): v for Y, d in G.items() for k, v in d.items()}).rename("group").to_frame().to_parquet(
    f"{RES}/study14_groups.parquet")
print("groups built", {Y: {k: v.value_counts().to_dict() for k, v in d.items()} for Y, d in G.items()}, flush=True)

params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
small = dict(params, num_leaves=31, min_data_in_leaf=500)
y = X["y_night"]
yr = y.groupby(level=0).rank(pct=True) - 0.5
ok = y.notna().values
preds = {}
for kind in ["comove", "behavior"]:
    per, poolg = [], []
    for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
        g = G[q.year][kind]
        gid = pd.Series(tick).map(g).fillna(-1).values
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dates < cut)
        te = (dates >= q.start_time) & (dates <= q.end_time)
        if te.sum() == 0:
            continue
        # pooled with group features
        Xg = X[feat].copy()
        Xg["g_id"] = gid.astype("float32")
        gm = X.groupby([dates, gid])
        for k in ["r1", "r5", "night1", "intra1"]:
            Xg[f"g_{k}_rel"] = (X[k] - gm[k].transform("mean")).astype("float32")
        Xg["g_r5_mean"] = gm["r5"].transform("mean").astype("float32")
        mdl = lgb.train(params, lgb.Dataset(Xg.loc[tr], yr[tr], categorical_feature=["g_id"]), num_boost_round=300)
        poolg.append(pd.Series(mdl.predict(Xg.loc[te]), index=X.index[te]))
        # one model per group; group -1 uses the pooled-with-group model's predictions
        for k in sorted(set(gid)):
            m = gid == k
            if k == -1 or (tr & m).sum() < 20000:
                per.append(pd.Series(mdl.predict(Xg.loc[te & m]), index=X.index[te & m]))
                continue
            mg = lgb.train(small, lgb.Dataset(X.loc[tr & m, feat], yr[tr & m]), num_boost_round=300)
            per.append(pd.Series(mg.predict(X.loc[te & m, feat]), index=X.index[te & m]))
        del Xg
        print(kind, q, flush=True)
    preds[f"per_{kind}"] = pd.concat(per)
    preds[f"pooled+{kind}"] = pd.concat(poolg)
preds["pooled"] = pd.read_parquet(f"{RES}/study10_pred_base.parquet")["pred"]
for k, p in preds.items():
    pd.DataFrame({"pred": p}).to_parquet(f"{RES}/study14_pred_{k.replace('+', '_plus_')}.parquet")

R = o.shift(-1) / c - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5
rows = []
for k, p in preds.items():
    S = p.unstack().reindex(columns=cols)
    S = S.loc[S.index < days[-1]]
    # per-group models are on different scales: compare on within-day percentile ranks
    S = S.rank(axis=1, pct=True)
    W = bt.select_topk(S, S.notna(), 10)
    rt = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
    ic = S.corrwith(R.reindex_like(S).rank(axis=1, pct=True), axis=1)
    for per, a, b in [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
        st = ann_stats(rt.net.loc[a:b])
        rows.append(dict(model=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                         gross_bps=1e4 * rt.gross.loc[a:b].mean(), rank_ic=float(ic.loc[a:b].mean())))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study14_clusters.csv", index=False)
print(df.pivot_table(index="model", columns="period", values=["sharpe", "gross_bps", "rank_ic"]).round(3).to_string())
