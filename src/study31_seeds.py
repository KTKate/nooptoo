"""Study 31: how much of the ranker's backtest is random-seed noise, and does averaging seeds help?

The pooled model (study3_ml.py design, close features, 2024Q1 .. 2026Q3) is trained with 5 different random seeds
(bagging and feature sampling). Reports each seed's top-10 net Sharpe (the spread shows run-to-run noise) and the
Sharpe of the seed-averaged prediction. Output: results/study31_seeds.csv, results/study31_pred_avg.parquet
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
feat = [k for k in X.columns if not k.startswith("y_")]
y = X["y_night"]
yr = y.groupby(level=0).rank(pct=True) - 0.5
ok = y.notna().values
dates = X.index.get_level_values(0)
preds = {s: [] for s in range(5)}
for q in pd.period_range("2024Q1", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 11)]
    tr, te = ok & (dates < cut), (dates >= q.start_time) & (dates <= q.end_time)
    ds = lgb.Dataset(X.loc[tr, feat], yr[tr], free_raw_data=False)
    for s in range(5):
        pr = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=100 + s)
        m = lgb.train(pr, ds, num_boost_round=300)
        preds[s].append(pd.Series(m.predict(X.loc[te, feat]), index=X.index[te]))
    print(q, flush=True)
R = P["o"][cols].shift(-1) / P["c"][cols] - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5
rows = []
ranks = []
for s in range(5):
    S = pd.concat(preds[s]).unstack().reindex(columns=cols)
    S = S.loc[S.index < days[-1]]
    ranks.append(S.rank(axis=1, pct=True))
    for k in [10, 20]:
        W = bt.select_topk(S, S.notna(), k)
        r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
        for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
            rows.append(dict(model=f"seed{s}", k=k, period=per, sharpe=ann_stats(r.net.loc[a:b])["sharpe"]))
A = sum(ranks) / 5
pd.DataFrame({"pred": A.stack()}).to_parquet(f"{RES}/study31_pred_avg.parquet")
for k in [10, 20]:
    W = bt.select_topk(A, A.notna(), k)
    r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
    for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
        rows.append(dict(model="avg5", k=k, period=per, sharpe=ann_stats(r.net.loc[a:b])["sharpe"]))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study31_seeds.csv", index=False)
print(df.pivot_table(index=["model", "k"], columns="period", values="sharpe").round(2).to_string())
