"""Study 36: study-32 jump/drop classifiers with extra inputs known at 15:45 (news_features.py panels): n_news5,
sent, sent5, ev_fda, ev_offering, ev_mna, ev_gov, ev_earn, ev_up, ev_down, ev_analyst, short_ratio, short_z.
Same walk-forward and evaluation as study 32; baseline = results/study32_pred.parquet.
Output: results/study36_jumps_extra.csv

Study 32 description follows.

Study 32: predicting big overnight jumps (and drops) directly.

Descriptive (printed by the owner-question analysis): of 7,250 liquid overnight jumps > +10% since 2020, 65% had a
news article between 15:45 and the next open (4% for a random night) and 43% fell on an earnings night, so most big
jumps come from news released after the decision time.
Model: LightGBM classifiers, quarterly walk-forward 2022Q1 .. 2026Q3 (study3_ml.py design, close features from
data/ml_frame.parquet) plus two known-at-15:45 inputs: an earnings report tonight or before tomorrow's open (calendar
date is known in advance), and the number of news articles for the stock during the day. Targets:
  jump   overnight return > +5%
  drop   overnight return < -5%
Evaluation: precision of the top 10 by P(jump) per night vs the base rate; overnight portfolios (auction costs +
2.5 bp per side): top 10 by P(jump); top 10 by P(jump) - P(drop); and the regular rank model with the 10 highest
P(drop) names removed.
Output: results/study32_jumps.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import store
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import news_features as NF
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
feat = [k for k in X.columns if not k.startswith("y_")]
idx = X.index
E = store.read("earnings")
E["date"] = pd.to_datetime(E.date)
di = days.searchsorted(E.date)
earn = pd.DataFrame(0.0, index=days, columns=cols)
ci = {t: i for i, t in enumerate(cols)}
for d, t in zip(di, E.symbol):
    if t in ci and 0 < d < len(days):
        earn.iat[d, ci[t]] = 1.0          # report on day d (after the close of d-1 or before the open of d, or after d's close)
        earn.iat[d - 1, ci[t]] = 1.0
X["earn_tonight"] = earn.stack().reindex(idx).values.astype("float32")
F = NF.load()
X["news_today"] = F["n_news"].reindex(index=days, columns=cols).stack(future_stack=True).reindex(idx).values.astype("float32")
EXTRA = ["n_news5", "sent", "sent5", "ev_fda", "ev_offering", "ev_mna", "ev_gov", "ev_earn", "ev_up", "ev_down",
         "ev_analyst", "short_ratio", "short_z"]
for k in EXTRA:
    X[k] = F[k].reindex(index=days, columns=cols).stack(future_stack=True).reindex(idx).values.astype("float32")
feat2 = feat + ["earn_tonight", "news_today"] + EXTRA
y = X["y_night"]
ok = y.notna().values
dates = idx.get_level_values(0)
T = {"jump": (y > 0.05).astype(float), "drop": (y < -0.05).astype(float)}
pr = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
preds = {k: [] for k in T}
for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 11)]
    tr, te = ok & (dates < cut), (dates >= q.start_time) & (dates <= q.end_time)
    if te.sum() == 0:
        continue
    for k in T:
        m = lgb.train(pr, lgb.Dataset(X.loc[tr, feat2], T[k][tr]), num_boost_round=300)
        preds[k].append(pd.Series(m.predict(X.loc[te, feat2]), index=idx[te]))
    print(q, flush=True)
pj = pd.concat(preds["jump"]).unstack().reindex(columns=cols)
pd_ = pd.concat(preds["drop"]).unstack().reindex(columns=cols)
pj, pd_ = pj.loc[pj.index < days[-1]], pd_.loc[pd_.index < days[-1]]
pd.DataFrame({"p_jump": pj.stack(), "p_drop": pd_.stack()}).to_parquet(f"{RES}/study36_pred.parquet")
b32 = pd.read_parquet(f"{RES}/study32_pred.parquet")
pj32 = b32.p_jump.unstack().reindex_like(pj)
pd32 = b32.p_drop.unstack().reindex_like(pj)
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(pj)
cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(pj)
base = pd.read_parquet(f"{RES}/study10_pred_base.parquet")["pred"].unstack().reindex(index=pj.index, columns=cols)
rows = []
per = [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
Wj = bt.select_topk(pj, pj.notna(), 10)
hitj = ((R > 0.05) & (Wj > 0)).sum(1) / 10
rate = (R > 0.05).where(pj.notna()).mean(1)
worst = bt.select_topk(pd_, pd_.notna(), 10) > 0
variants = {"top10_p_jump": Wj, "top10_p_jump_minus_p_drop": bt.select_topk(pj - pd_, pj.notna(), 10),
            "study32_jmd": bt.select_topk(pj32 - pd32, pj32.notna(), 10),
            "rank_model": bt.select_topk(base, base.notna(), 10),
            "rank_model_without_top10_p_drop": bt.select_topk(base.where(~worst), base.notna() & ~worst, 10)}
for name, W in variants.items():
    r = bt.run(W, R, cost)
    for p, a, b in per:
        rows.append(dict(variant=name, period=p, sharpe=ann_stats(r.net.loc[a:b])["sharpe"],
                         gross_bps=1e4 * r.gross.loc[a:b].mean(),
                         jump_hit_rate=float(hitj.loc[a:b].mean()) if name == "top10_p_jump" else np.nan,
                         base_rate=float(rate.loc[a:b].mean())))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study36_jumps_extra.csv", index=False)
print(df.round(4).to_string())
