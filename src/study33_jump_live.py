"""Study 33: the study-32 jump/drop classifiers scored at 15:45, alone and combined with the ensemble.

Study 32 (close features) found that ranking by P(jump > +5%) - P(drop < -5%) beat the rank model in 2022-23, val and
holdout. Here the classifiers are retrained per quarter 2024Q1 .. 2026Q4 (training data ends 10 trading days before
the quarter; models saved as data/models/jump_{jump,drop}_<q>.txt) and scored on the 15:45 feature frame of study 23
(SNAP_MODE none). Extra inputs known at 15:45: earnings report tonight or before tomorrow's open, and the number of
news articles since the previous 15:45 (news_features.py).
Variants (top 10, closing-auction buy, opening-auction sell, auction costs + 2.5 bp per side):
  ensemble             study 23 score (results/study23_pred.parquet)
  jmd                  P(jump) - P(drop)
  blend                average of the daily percentile ranks of ensemble and jmd
  blend_2to1           (2 * ensemble rank + jmd rank) / 3
  ensemble_no_drop10   ensemble after removing the 10 highest P(drop)
Output: results/study33_pred.parquet, results/study33_jump_live.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import store
import news_features as NF
from core import exec_cost_bps, ann_stats, RES, DATA
import bt
import study23_ensemble_timing as S

P, cols, days = S.P, S.cols, S.days
MD = f"{DATA}/models"
PR = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)


def extra_panels():
    """earn_tonight and news_today panels (date x ticker), both known at 15:45."""
    E = store.read("earnings")
    di = days.searchsorted(pd.to_datetime(E.date))
    earn = pd.DataFrame(0.0, index=days, columns=cols)
    ci = {t: i for i, t in enumerate(cols)}
    for d, t in zip(di, E.symbol):
        if t in ci and 0 < d < len(days):
            earn.iat[d, ci[t]] = 1.0
            earn.iat[d - 1, ci[t]] = 1.0
    news = NF.load()["n_news"].reindex(index=days, columns=cols)
    return earn, news


EARN, NEWS = extra_panels()


def add_extra(X):
    X = X.copy()
    r = days.get_indexer(X.index.get_level_values(0))
    c = pd.Index(cols).get_indexer(X.index.get_level_values(1))
    ok = (r >= 0) & (c >= 0)
    for k, panel in [("earn_tonight", EARN), ("news_today", NEWS)]:
        v = np.full(len(X), np.nan, dtype="float32")
        v[ok] = panel.values[r[ok], c[ok]]
        X[k] = v
    return X


def train(q0, q1):
    X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
    X = X[X.index.get_level_values(0) >= "2020-01-01"]
    feat = [k for k in X.columns if not k.startswith("y_")]
    y = X["y_night"]
    X = add_extra(X[feat])
    ok = y.notna().values
    dates = X.index.get_level_values(0)
    for q in pd.period_range(q0, q1, freq="Q"):
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dates < cut)
        for k, t in [("jump", y > 0.05), ("drop", y < -0.05)]:
            m = lgb.train(PR, lgb.Dataset(X[tr], t[tr].astype(float)), num_boost_round=300)
            m.save_model(f"{MD}/jump_{k}_{q}.txt")
        print("trained", q, flush=True)


_M = {}


def model(k, q):
    if (k, q) not in _M:
        _M[(k, q)] = lgb.Booster(model_file=f"{MD}/jump_{k}_{q}.txt")
    return _M[(k, q)]


def score(X, q):
    """P(jump), P(drop) for a feature frame X (study-3 columns) with the models of quarter q."""
    X = add_extra(X)
    m = model("jump", q)
    X = X[m.feature_name()]
    return pd.DataFrame({"p_jump": m.predict(X), "p_drop": model("drop", q).predict(X)}, index=X.index)


def one_day(d):
    return score(S.frame_1545(d), pd.Period(d, freq="Q"))


def rank(S_):
    return S_.rank(axis=1, pct=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    if os.environ.get("TRAIN", "1") == "1":
        train("2024Q1", "2026Q4")
    with Pool(3) as pool:
        rows = pool.map(one_day, S.test_days, chunksize=8)
    pred = pd.concat(rows)
    pred.to_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    pj = pred["p_jump"].unstack().reindex(index=ens.index, columns=cols)
    pdr = pred["p_drop"].unstack().reindex(index=ens.index, columns=cols)
    keep = ens.index < days[-1]
    ens, pj, pdr = ens[keep], pj[keep], pdr[keep]
    ok = ens.notna() & pj.notna()
    jmd = (pj - pdr).where(ok)
    ens = ens.where(ok)
    worst = bt.select_topk(pdr.where(ok), ok, 10) > 0
    V = {"ensemble": ens, "jmd": jmd, "blend": (rank(ens) + rank(jmd)) / 2,
         "blend_2to1": (2 * rank(ens) + rank(jmd)) / 3, "ensemble_no_drop10": ens.where(~worst)}
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(ens)
    cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(ens)
    out = []
    for name, sc in V.items():
        W = bt.select_topk(sc, sc.notna(), 10)
        r = bt.run(W, R, cost)
        hit = ((R > 0.05) & (W > 0)).sum(1) / 10
        drop = ((R < -0.05) & (W > 0)).sum(1) / 10
        for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]:
            st = ann_stats(r.net.loc[a:b])
            out.append(dict(variant=name, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                            gross_bps=1e4 * r.gross.loc[a:b].mean(), cost_bps=1e4 * r.cost.loc[a:b].mean(),
                            jump_rate=hit.loc[a:b].mean(), drop_rate=drop.loc[a:b].mean()))
    df = pd.DataFrame(out)
    df.to_csv(f"{RES}/study33_jump_live.csv", index=False)
    print(df.round(4).to_string())
