"""Study 10b: do news, sentiment, event, short-volume and sector features improve the overnight LightGBM ranker?

Same design as study3_ml.py (quarterly walk-forward, 10-day embargo, rank target, same parameters), trained twice
on identical rows: baseline features vs baseline + factor features. Evaluated as the live strategy is traded:
top 10 by prediction, buy in the closing auction, sell in the opening auction, auction costs + 2.5 bp per side.
Features use close-of-day prices here for both models (the 15:45 substitution of study3_timing.py is run only if
the augmented model is clearly better). News features are cut at 15:45 either way.
New features: n_news, n_news5, sent, sent5, event counts (analyst, upgrade, downgrade, FDA, offering, M&A,
government/regulation, earnings), short_ratio, short_z (FINRA, day t-1), sector code, sector-relative 1-day and
5-day returns, sector mean 5- and 20-day returns (industry momentum), sector news count.
    python src/study10_ml.py [Q_START] [Q_END]       (default 2022Q1 2026Q3)
Output: results/study10_ml.csv, results/study10_pred_{base,aug}.parquet, results/study10_importance.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import news_features as NF
import bt

Q0 = sys.argv[1] if len(sys.argv) > 1 else "2022Q1"
Q1 = sys.argv[2] if len(sys.argv) > 2 else "2026Q3"
P = load_panel()
cols = stock_cols(P)
days = P["c"].index
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
base_feat = [k for k in X.columns if not k.startswith("y_")]
F = NF.load()
idx = X.index
newf = []
for k in ["n_news", "n_news5", "sent", "sent5", "ev_analyst", "ev_up", "ev_down", "ev_fda", "ev_offering", "ev_mna",
          "ev_gov", "ev_earn", "short_ratio", "short_z"]:
    if k in F:
        X["f_" + k] = F[k].stack(future_stack=True).reindex(idx).values.astype("float32")
        newf.append("f_" + k)
sec = F["_sector"]
codes = {s: i for i, s in enumerate(sorted(sec.dropna().unique()))}
X["f_sector"] = pd.Series(idx.get_level_values(1)).map(sec).map(codes).values.astype("float32")
newf.append("f_sector")
g = X.groupby([idx.get_level_values(0), X.f_sector])
for k in ["r1", "r5"]:
    X[f"f_{k}_sec_rel"] = (X[k] - g[k].transform("mean")).astype("float32")
    newf.append(f"f_{k}_sec_rel")
for k in ["r5", "r20"]:
    X[f"f_sec_{k}"] = g[k].transform("mean").astype("float32")
    newf.append(f"f_sec_{k}")
X["f_sec_news"] = g["f_n_news"].transform("sum").astype("float32")
newf.append("f_sec_news")
print("rows", len(X), "base features", len(base_feat), "new", len(newf), flush=True)

params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
y = X["y_night"]
ok = y.notna()
yr = y.groupby(level=0).rank(pct=True) - 0.5
dates = idx.get_level_values(0)
preds = {"base": [], "aug": []}
imp = None
for q in pd.period_range(Q0, Q1, freq="Q"):
    a, b = q.start_time, q.end_time
    cut = days[max(0, days.searchsorted(a) - 1 - 10)]
    tr, te = ok & (dates < cut), (dates >= a) & (dates <= b)
    if te.sum() == 0:
        continue
    for name, feats in [("base", base_feat), ("aug", base_feat + newf)]:
        mdl = lgb.train(params, lgb.Dataset(X.loc[tr, feats], yr[tr]), num_boost_round=300)
        preds[name].append(pd.Series(mdl.predict(X.loc[te, feats]), index=idx[te]))
        if name == "aug":
            imp = pd.Series(mdl.feature_importance("gain"), index=feats)
    print(q, "done", flush=True)
for name in preds:
    pd.DataFrame({"pred": pd.concat(preds[name])}).to_parquet(f"{RES}/study10_pred_{name}.parquet")
(imp / imp.sum()).sort_values(ascending=False).to_csv(f"{RES}/study10_importance.csv")

o, c = P["o"][cols], P["c"][cols]
R = o.shift(-1) / c - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5
rows = []
for name in preds:
    S = pd.concat(preds[name]).unstack().reindex(columns=cols)
    for k in [10, 20]:
        W = bt.select_topk(S, S.notna(), k)
        r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
        for per, a_, b_ in [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"),
                            ("oos", "2025-07", "2026-09")]:
            x = r.loc[a_:b_]
            x = x[x.index < days[-1]]
            st = ann_stats(x.net)
            rows.append(dict(model=name, k=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                             maxdd=st["maxdd"], gross_bps=1e4 * x.gross.mean(), cost_bps=1e4 * x.cost.mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study10_ml.csv", index=False)
print(df.pivot_table(index=["model", "k"], columns="period", values=["sharpe", "gross_bps"]).round(2))
print(imp.sort_values(ascending=False).head(25))
