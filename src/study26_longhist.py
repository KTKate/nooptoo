"""Study 26: does a longer training history (from 2011 instead of 2020) improve the overnight ranker?

Panel: data/local/daily_long (2010-01 .. 2019-05, src/fetch_long_history.py) + the store (2019-06 ..), adjusted
exactly as core.load_panel does. Features: ml_features.build on the combined panel (earnings-calendar features are
NaN before 2019-06, when the calendar data starts). Same model and walk-forward as study3_ml.py (quarterly,
10-day embargo, rank target of close -> next open). Two training windows on identical test rows:
  short   training rows from 2020-01 (the current design)
  long    training rows from 2011-01
Test quarters 2022Q1 .. 2026Q3; top-10 portfolio, auction costs + 2.5 bp per side.
Caveat: the long history only has companies still listed in 2026 (survivorship), which mostly affects the
training data, not the 2022-26 test rows.
Output: results/study26_longhist.csv, results/study26_pred_{short,long}.parquet
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import store
from core import stock_cols, exec_cost_bps, ann_stats, RES, DATA, load_panel
import ml_features as M

LOC = os.path.join(DATA, "local", "daily_long")
old = pd.concat([pd.read_parquet(os.path.join(LOC, f)) for f in sorted(os.listdir(LOC)) if f.endswith(".parquet")])
old = old[old.date < "2019-06-01"]
new = store.read("daily", start="2019-06")
d = pd.concat([old, new[old.columns]], ignore_index=True)
d = d[(d.c > 0) & (d.o > 0) & (d.h > 0) & (d.l > 0)].drop_duplicates(["date", "ticker"], keep="last")
d["date"] = pd.to_datetime(d.date)
f = store.adj_factor(d).astype("float64")
d["rawc"] = d["c"]
for k in ["o", "h", "l", "c"]:
    d[k] = d[k] * f
P = {k: d.pivot(index="date", columns="ticker", values=k).astype("float32") for k in ["o", "h", "l", "c", "rawc", "v"]}
del d, old, new
days = P["c"]["SPY"].dropna().index
for k in P:
    P[k] = P[k].reindex(days)
P["dv"] = (P["rawc"] * P["v"]).astype("float32")
bad = (P["l"] > P["h"] * 1.0001) | (P["o"] > P["h"] * 1.02) | (P["o"] < P["l"] * 0.98)
for k in ["o", "h", "l", "c"]:
    P[k] = P[k].mask(bad)
cols = [t for t in stock_cols(P) if t in M.cols]
o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
F, mkt, adv20 = M.build(o, h, l, c, rawc, v, dv, P["c"]["SPY"], P["c"]["^VIX"], P["c"]["^VIX3M"], P["c"]["IWM"])
univ = ((rawc > 5) & (adv20 > 5e6)).loc["2011-01-01":]
X = M.features_frame({k: x.loc["2011-01-01":] for k, x in F.items()}, mkt.loc["2011-01-01":], univ)
yN = (o.shift(-1) / c - 1).stack(future_stack=True).reindex(X.index)
feat = list(X.columns)
print("rows", len(X), "features", len(feat), flush=True)
y = yN.astype("float32")
ok = y.notna().values
yr = y.groupby(level=0).rank(pct=True) - 0.5
dates = X.index.get_level_values(0)
params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
preds = {"short": [], "long": []}
for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 11)]
    te = (dates >= q.start_time) & (dates <= q.end_time)
    if te.sum() == 0:
        continue
    for name, start in [("short", "2020-01-01"), ("long", "2011-01-01")]:
        tr = ok & (dates < cut) & (dates >= start)
        m = lgb.train(params, lgb.Dataset(X.loc[tr, feat], yr[tr]), num_boost_round=300)
        preds[name].append(pd.Series(m.predict(X.loc[te, feat]), index=X.index[te]))
    print(q, flush=True)
Pc = load_panel()
cols2 = stock_cols(Pc)
R = Pc["o"][cols2].shift(-1) / Pc["c"][cols2] - 1
cost = exec_cost_bps(Pc, "auction")[cols2] + 2.5
rows = []
for name, p in preds.items():
    p = pd.concat(p)
    pd.DataFrame({"pred": p}).to_parquet(f"{RES}/study26_pred_{name}.parquet")
    S = p.unstack().reindex(columns=cols2)
    S = S.loc[S.index < Pc["c"].index[-1]]
    for k in [10, 20]:
        W = __import__("bt").select_topk(S, S.notna(), k)
        r = __import__("bt").run(W, R.reindex_like(W), cost.reindex_like(W))
        for per, a, b in [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
            st = ann_stats(r.net.loc[a:b])
            rows.append(dict(train=name, k=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                             gross_bps=1e4 * r.gross.loc[a:b].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study26_longhist.csv", index=False)
print(df.pivot_table(index=["train", "k"], columns="period", values="sharpe").round(2).to_string())
