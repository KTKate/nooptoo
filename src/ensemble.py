"""Five-model ensemble for the overnight ranker (study 14 follow-up, results/study23_*).

Members (same 53 base features as study3_ml.py / ml_features.py, rank target of the close -> next open return):
  pooled          one model for all stocks (the current paper model design; data/models/night_<q>.txt)
  pooled_beh      one model with the behavior-group id and group-relative returns as extra features
  pooled_com      the same with the co-movement group
  per_beh         one model per behavior group (8)
  per_com         one model per co-movement group (12)
The ensemble score is the mean of each member's within-day percentile rank. Groups for year Y come from
results/study14_groups.parquet (built from the two years before Y); stocks without a group get id -1 and are
scored by the pooled-with-group-features model in the per-group members.

    python src/ensemble.py train 2024Q1 2026Q4     # trains and saves the members per quarter to data/models/ens_*
Models for a quarter use all rows before the quarter start minus a 10-day embargo.
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "data", "models")
RES = os.path.join(ROOT, "results")
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
SMALL = dict(PARAMS, num_leaves=31, min_data_in_leaf=500)
KINDS = {"beh": "behavior", "com": "comove"}


def groups(year):
    g = pd.read_parquet(os.path.join(RES, "study14_groups.parquet"))["group"]
    y = min(int(year), int(g.index.get_level_values(0).max()))
    return {k: g.xs((y, kind), level=(0, 1)) for k, kind in KINDS.items()}


def add_group_features(X, gid):
    """X: long frame indexed (date, ticker) with the base features; gid: array of group ids aligned with X."""
    Xg = X.copy()
    Xg["g_id"] = gid.astype("float32")
    dates = X.index.get_level_values(0)
    gm = X.groupby([dates, gid])
    for k in ["r1", "r5", "night1", "intra1"]:
        Xg[f"g_{k}_rel"] = (X[k] - gm[k].transform("mean")).astype("float32")
    Xg["g_r5_mean"] = gm["r5"].transform("mean").astype("float32")
    return Xg


def gid_for(X, g):
    return pd.Series(X.index.get_level_values(1)).map(g).fillna(-1).values


def train(q0, q1):
    from core import DATA
    X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
    X = X[X.index.get_level_values(0) >= "2020-01-01"]
    feat = [k for k in X.columns if not k.startswith("y_")]
    from core import load_panel
    days = load_panel()["c"].index
    y = X["y_night"]
    yr = y.groupby(level=0).rank(pct=True) - 0.5
    ok = y.notna().values
    dates = X.index.get_level_values(0)
    os.makedirs(MODELS, exist_ok=True)
    for q in pd.period_range(q0, q1, freq="Q"):
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dates < cut)
        G = groups(q.year)
        for k in KINDS:
            gid = gid_for(X, G[k])
            Xg = add_group_features(X[feat], gid)
            m = lgb.train(PARAMS, lgb.Dataset(Xg.loc[tr], yr[tr], categorical_feature=["g_id"]), num_boost_round=300)
            m.save_model(os.path.join(MODELS, f"ens_pooled_{k}_{q}.txt"))
            for gg in sorted(set(gid)):
                mm = gid == gg
                if gg == -1 or (tr & mm).sum() < 20000:
                    continue
                mg = lgb.train(SMALL, lgb.Dataset(X.loc[tr & mm, feat], yr[tr & mm]), num_boost_round=300)
                mg.save_model(os.path.join(MODELS, f"ens_per_{k}_{q}_g{int(gg)}.txt"))
            del Xg
        print("trained", q, flush=True)


def _load(name):
    return lgb.Booster(model_file=os.path.join(MODELS, name))


def predict(X, q, pooled_model=None):
    """X: long base-feature frame (date, ticker) for the scoring rows of quarter q. Returns a DataFrame of member
    predictions and the ensemble score (mean within-day percentile rank)."""
    q = pd.Period(q, freq="Q")
    G = groups(q.year)
    out = pd.DataFrame(index=X.index)
    pm = pooled_model or _load(f"night_{q}.txt")
    out["pooled"] = pm.predict(X[pm.feature_name()])
    for k in KINDS:
        gid = gid_for(X, G[k])
        Xg = add_group_features(X[pm.feature_name()], gid)
        mp = _load(f"ens_pooled_{k}_{q}.txt")
        out[f"pooled_{k}"] = mp.predict(Xg[mp.feature_name()])
        per = out[f"pooled_{k}"].copy()                     # group -1 and small groups: pooled-with-group model
        for gg in sorted(set(gid)):
            fn = os.path.join(MODELS, f"ens_per_{k}_{q}_g{int(gg)}.txt")
            if gg == -1 or not os.path.exists(fn):
                continue
            mm = gid == gg
            mg = lgb.Booster(model_file=fn)
            per[mm] = mg.predict(X.loc[mm, mg.feature_name()])
        out[f"per_{k}"] = per
    dates = X.index.get_level_values(0)
    ranks = out.groupby(dates).rank(pct=True)
    out["ensemble"] = ranks.mean(axis=1)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
