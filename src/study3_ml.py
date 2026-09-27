"""Study 3: walk-forward gradient-boosted cross-sectional model (LightGBM).

Decision at the close of day t (practically: ~15:50 with near-final prices).
Targets (cross-sectionally ranked each day, so the model learns relative returns):
  night : c_t -> o_{t+1}           (overnight hold)
  day1  : o_{t+1} -> c_{t+1}       (next-day intraday, entry at next open)
  cc1   : c_t -> c_{t+1}
  cc5   : c_t -> c_{t+5}           (intraweek)
Features: price/volume history, overnight vs intraday decomposition, volatility,
range, liquidity, market state (SPY, VIX), and the earnings calendar
(days to next report, days since last report, last surprise).
Walk-forward: retrain every quarter on all data up to the quarter start minus a
10-day embargo; first test quarter 2022Q1. Only the stock universe with
price > $5 and 20d median dollar volume > $5M is used.
Output: results/study3_pred_<target>[_from<Q_START>].parquet, data/models/<target>_<quarter>.txt
Live retrain for a new quarter (after `python src/update_data.py daily`; delete data/ml_frame.parquet first so
the feature frame is rebuilt with the new days):  Q_START=2026Q4 Q_END=2026Q4 python src/study3_ml.py night
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from core import load_panel, stock_cols, half_spread_bps, cost_bps, split_stats, RES, DATA
import bt

from ml_features import *  # noqa: F401,F403  (P, cols, days, F, mkt, univ, T, H, build, features_frame)


def stack():
    fn = f"{DATA}/ml_frame.parquet"
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    X = features_frame(F, mkt, univ)
    for k, t in T.items():
        X["y_" + k] = t.stack().reindex(X.index).astype("float32")
    X.to_parquet(fn)
    return X


X = stack()
X = X[X.index.get_level_values(0) >= "2020-01-01"]
feat = [k for k in X.columns if not k.startswith("y_")]
print("rows", len(X), "features", len(feat), flush=True)

params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=4)

dates = X.index.get_level_values(0)
Q_START = os.environ.get("Q_START", "2022Q1")          # later start: only (re)train and save models
Q_END = os.environ.get("Q_END", "2026Q3")              # a quarter without test data only trains and saves a model
quarters = pd.period_range(Q_START, Q_END, freq="Q")
targets = sys.argv[1:] or list(T)
preds = {}
for tgt in targets:
    y = X["y_" + tgt]
    ok = y.notna()
    # rank target cross-sectionally, centred -> learns relative ordering, robust to outliers
    yr = y.groupby(level=0).rank(pct=True) - 0.5
    out = []
    for q in quarters:
        a, b = q.start_time, q.end_time
        cut = days[max(0, days.searchsorted(a) - H[tgt] - 10)]
        tr = ok & (dates < cut)
        te = (dates >= a) & (dates <= b)
        ds = lgb.Dataset(X.loc[tr, feat], yr[tr])
        mdl = lgb.train(params, ds, num_boost_round=300)
        os.makedirs(f"{DATA}/models", exist_ok=True)
        mdl.save_model(f"{DATA}/models/{tgt}_{q}.txt")
        if te.sum() == 0:
            print(tgt, q, "train", int(tr.sum()), "no test data: model saved for live use", flush=True)
            continue
        p = pd.Series(mdl.predict(X.loc[te, feat]), index=X.index[te])
        out.append(p)
        print(tgt, q, "train", int(tr.sum()), "test", int(te.sum()), flush=True)
        if q == quarters[-1]:
            imp = pd.Series(mdl.feature_importance("gain"), index=feat).sort_values(ascending=False)
            imp.to_csv(f"{RES}/study3_importance_{tgt}.csv")
    if not out:
        continue
    preds[tgt] = pd.concat(out)
    suffix = "" if Q_START == "2022Q1" else f"_from{Q_START}"
    pd.DataFrame({"pred": preds[tgt]}).to_parquet(f"{RES}/study3_pred_{tgt}{suffix}.parquet")
