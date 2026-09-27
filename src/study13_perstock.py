"""Study 13: one pooled model for the whole market vs a model per stock vs a model per sector.

The overnight ranker (study 3) is a single LightGBM trained on all stocks together (cross-sectional ranks). The
alternatives here use the same features (data/ml_frame.parquet, close-of-day values) and the same quarterly
walk-forward with a 10-day embargo, 2022Q1 .. 2026Q3:
  per_stock   one small LightGBM per stock on its own history, target = its raw overnight return
              (the 300 stocks with the highest median dollar volume in 2021, so each has ~500+ training days)
  per_sector  one LightGBM per sector (Nasdaq sector labels), rank target within the day
  pooled      the study-3 predictions (results/study3_pred_night.parquet) restricted to the same stocks
Evaluation on the 300-stock universe: (a) top 10 by prediction each day, closing-auction buy, opening-auction
sell, auction costs + 2.5 bp per side; (b) rank correlation of prediction and realized overnight return.
Output: results/study13_perstock.csv
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
X = X[X.index.get_level_values(0) >= "2019-01-01"]
feat = [k for k in X.columns if not k.startswith("y_")]
dv = P["dv"][cols].loc["2021"].median()
top = dv.sort_values(ascending=False).index[:300]
tick = X.index.get_level_values(1)
dates = X.index.get_level_values(0)
quarters = pd.period_range("2022Q1", "2026Q3", freq="Q")
sec = pd.read_parquet(f"{DATA}/store/sectors.parquet").drop_duplicates("ticker").set_index("ticker").sector
y = X["y_night"]
yr = y.groupby(level=0).rank(pct=True) - 0.5

small = dict(objective="regression", learning_rate=0.05, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
mid = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)


def cut_for(q):
    return days[max(0, days.searchsorted(q.start_time) - 11)]


# per stock
ps = []
Xs = X[tick.isin(top)]
ts, ds = Xs.index.get_level_values(1), Xs.index.get_level_values(0)
for t in top:
    m = ts == t
    Xt, yt, dt = Xs[m], y[Xs.index[m]], ds[m]
    for q in quarters:
        tr = (dt < cut_for(q)) & yt.notna().values
        te = (dt >= q.start_time) & (dt <= q.end_time)
        if tr.sum() < 250 or te.sum() == 0:
            continue
        mdl = lgb.train(small, lgb.Dataset(Xt.loc[tr, feat], yt[tr].clip(-0.2, 0.2)), num_boost_round=100)
        ps.append(pd.Series(mdl.predict(Xt.loc[te, feat]), index=Xt.index[te]))
print("per-stock done", flush=True)
# per sector (all stocks of the sector for training, predictions kept for the 300)
pe = []
secs = pd.Series(tick).map(sec).values
for s in pd.Series(secs).dropna().unique():
    m = secs == s
    Xm, ym, dm = X[m], yr[m], dates[m]
    for q in quarters:
        tr = (dm < cut_for(q)) & ym.notna().values
        te = (dm >= q.start_time) & (dm <= q.end_time) & pd.Series(Xm.index.get_level_values(1)).isin(top).values
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        mdl = lgb.train(mid, lgb.Dataset(Xm.loc[tr, feat], ym[tr]), num_boost_round=200)
        pe.append(pd.Series(mdl.predict(Xm.loc[te, feat]), index=Xm.index[te]))
    print("sector", s, flush=True)
pooled = pd.read_parquet(f"{RES}/study3_pred_night.parquet")["pred"]
pooled = pooled[pooled.index.get_level_values(1).isin(top)]
preds = {"per_stock": pd.concat(ps), "per_sector": pd.concat(pe), "pooled": pooled}

o, c = P["o"][cols], P["c"][cols]
R = (o.shift(-1) / c - 1)
cost = exec_cost_bps(P, "auction")[cols] + 2.5
rows = []
common = None
for k, p in preds.items():
    S = p.unstack().reindex(columns=list(top))
    common = S.index if common is None else common.intersection(S.index)
for k, p in preds.items():
    S = p.unstack().reindex(index=common, columns=list(top))
    S = S.loc[S.index < days[-1]]
    W = bt.select_topk(S, S.notna(), 10)
    r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
    ic = S.rank(axis=1).corrwith(R.reindex_like(S).rank(axis=1), axis=1)
    for per, a, b in [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
        st = ann_stats(r.net.loc[a:b])
        rows.append(dict(model=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                         gross_bps=1e4 * r.gross.loc[a:b].mean(), rank_ic=float(ic.loc[a:b].mean())))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study13_perstock.csv", index=False)
print(df.pivot_table(index="model", columns="period", values=["sharpe", "gross_bps", "rank_ic"]).round(3).to_string())
