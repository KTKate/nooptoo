"""Study 24: tuning the overnight ranker's settings and target with a clean selection protocol.

Selection uses 2022-2023 only (models retrained quarterly with data before each quarter, as in study3_ml.py);
the chosen settings are then run once on 2024-01 .. 2026-09 and compared with the default. Everything with close
features (the 15:45 substitution is applied later only to a winner).
Grid:
  target     rank (default) | residual (overnight return minus beta x SPY overnight, ranked) | raw clipped return
             | top-quintile classifier (binary, logloss)
  leaves     31 | 63 (default) | 127
  min_leaf   500 | 2000 (default) | 8000
  rounds     150 | 300 (default) | 600 at learning rate 0.03
  window     all history (default) | last 3 years only
Evaluation: top-10 overnight portfolio, auction costs + 2.5 bp per side, Sharpe per period; rank IC.
Output: results/study24_tuning.csv
"""
import itertools
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
dates, tick = X.index.get_level_values(0), X.index.get_level_values(1)
y = X["y_night"]
spyN = (P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1)
beta = X["beta"] if "beta" in X.columns else pd.Series(1.0, index=X.index)
resid = y - beta.fillna(1.0) * spyN.reindex(dates).values
T = {"rank": y.groupby(level=0).rank(pct=True) - 0.5,
     "residual": resid.groupby(level=0).rank(pct=True) - 0.5,
     "raw": y.clip(-0.1, 0.1),
     "topq": (y.groupby(level=0).rank(pct=True) > 0.8).astype(float)}
ok = y.notna().values
R = P["o"][cols].shift(-1) / P["c"][cols] - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5


def run(target, leaves, min_leaf, rounds, window, quarters):
    pr = dict(objective="binary" if target == "topq" else "regression", learning_rate=0.03, num_leaves=leaves,
              min_data_in_leaf=min_leaf, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=4)
    out = []
    for q in quarters:
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dates < cut)
        if window == "3y":
            tr &= dates >= cut - pd.Timedelta(days=3 * 365)
        te = (dates >= q.start_time) & (dates <= q.end_time)
        if te.sum() == 0:
            continue
        m = lgb.train(pr, lgb.Dataset(X.loc[tr, feat], T[target][tr]), num_boost_round=rounds)
        out.append(pd.Series(m.predict(X.loc[te, feat]), index=X.index[te]))
    return pd.concat(out)


def score(p, periods):
    S = p.unstack().reindex(columns=cols)
    S = S.loc[S.index < days[-1]]
    W = bt.select_topk(S, S.notna(), 10)
    r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
    ic = S.rank(axis=1).corrwith(R.reindex_like(S).rank(axis=1), axis=1)
    return {per: (ann_stats(r.net.loc[a:b])["sharpe"], float(ic.loc[a:b].mean())) for per, a, b in periods}


dev_q = pd.period_range("2022Q1", "2023Q4", freq="Q")
dev_p = [("2022-23", "2022-01", "2023-12")]
rows = []
default = ("rank", 63, 2000, 300, "all")
# one-at-a-time around the default (cheaper than the full grid), then the best combination
cands = [default]
for i, vals in enumerate([["residual", "raw", "topq"], [31, 127], [500, 8000], [150, 600], ["3y"]]):
    for v in vals:
        c = list(default)
        c[i] = v
        cands.append(tuple(c))
for c in cands:
    s = score(run(*c, dev_q), dev_p)
    rows.append(dict(zip(["target", "leaves", "min_leaf", "rounds", "window"], c), stage="dev",
                     sharpe=s["2022-23"][0], ic=s["2022-23"][1]))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows)
base = D.iloc[0]
best = {}
for i, k in enumerate(["target", "leaves", "min_leaf", "rounds", "window"]):
    sub = D[D[k] != base[k]]
    sub = pd.concat([D.iloc[[0]], sub[[all(sub.iloc[j][kk] == base[kk] for kk in ["target", "leaves", "min_leaf", "rounds", "window"] if kk != k) for j in range(len(sub))]]])
    best[k] = sub.sort_values("ic", ascending=False).iloc[0][k]      # choose by rank IC (less noisy than Sharpe)
chosen = tuple(best[k] for k in ["target", "leaves", "min_leaf", "rounds", "window"])
print("chosen on 2022-23:", chosen, flush=True)
test_q = pd.period_range("2024Q1", "2026Q3", freq="Q")
test_p = [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
for name, c in [("default", default), ("chosen", chosen)]:
    p = run(*c, test_q)
    pd.DataFrame({"pred": p}).to_parquet(f"{RES}/study24_pred_{name}.parquet")
    s = score(p, test_p)
    for per in s:
        rows.append(dict(zip(["target", "leaves", "min_leaf", "rounds", "window"], c), stage=f"test_{name}", period=per,
                         sharpe=s[per][0], ic=s[per][1]))
    print(name, s, flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/study24_tuning.csv", index=False)
