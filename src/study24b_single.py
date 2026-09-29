"""Study 24b: single changes from study 24 (residual target alone, 600 rounds alone) on 2024-26. Study 24: tuning the overnight ranker's settings and target with a clean selection protocol.

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


test_q = pd.period_range("2024Q1", "2026Q3", freq="Q")
test_p = [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
rows = []
for name, c in [("residual_only", ("residual", 63, 2000, 300, "all")), ("rounds600_only", ("rank", 63, 2000, 600, "all"))]:
    p = run(*c, test_q)
    pd.DataFrame({"pred": p}).to_parquet(f"{RES}/study24b_pred_{name}.parquet")
    s_ = score(p, test_p)
    for per in s_:
        rows.append(dict(variant=name, period=per, sharpe=s_[per][0], ic=s_[per][1]))
    print(name, s_, flush=True)
pd.DataFrame(rows).to_csv(f"{RES}/study24b_single.csv", index=False)
