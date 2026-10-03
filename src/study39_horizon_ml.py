"""Studies 39 (5-day) and 40 (20-day): do fundamentals and analyst data help at holding periods longer than a night?

Question: studies 37-38 found no gain for the overnight models. Slow-moving information may matter more over weeks.
Design: LightGBM rank models (same parameters as study 3, 2 threads) predicting the within-day rank of the
close-to-close return over the next h trading days (h = 5 or 20). Quarterly walk-forward 2022Q1..2026Q3, training
rows every 5th day (overlapping targets make daily rows redundant), embargo h + 10 trading days.
Feature sets: price (the 53 study-3 inputs from data/ml_frame.parquet), price+fund (12 SEC fundamentals, study 37),
price+analyst (18 analyst/announcement inputs, study 38), all.
Trading (horizon_lib): every h days buy the top 20 in the closing auction, hold h days, auction costs + 2.5 bp per
side on names that change; liquid universe (price > $5, ADV > $20M). Judged by excess over the universe average
(the panel has survivorship bias); five rebalance offsets are averaged for h = 20 to reduce start-date luck.
Caveat: features use the day's close; live trading would use 15:45 values.
    python src/study39_horizon_ml.py 5     (study 39)
    python src/study39_horizon_ml.py 20    (study 40)
Output: results/study{39,40}_horizon_ml.csv, results/study{39,40}_pred.parquet
"""
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import analyst_features as AF
import horizon_lib as H
from core import RES, DATA

h = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SID = 39 if h <= 5 else 40
P, cols, days = H.P, H.cols, H.days
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
base = [k for k in X.columns if not k.startswith("y_")]
X = X[base]
idx = X.index


def put(name, panel, group):
    X[name] = panel.reindex(index=days, columns=cols).stack(future_stack=True).reindex(idx).values.astype("float32")
    group.append(name)


fund, anl = [], []
FP = pickle.load(open(f"{DATA}/local/fundamentals_panels.pkl", "rb"))
stale = FP["days_since_filing"] > 200
put("fu_log_mcap", np.log(FP["market_cap"].where(~stale & (FP["market_cap"] > 0))), fund)
for k in ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "net_margin",
          "book_to_market", "cash_to_mcap", "revenue_growth_yoy"]:
    put("fu_" + k, FP[k].where(~stale).reindex(index=days, columns=cols).rank(axis=1, pct=True), fund)
put("fu_days_since_filing", FP["days_since_filing"], fund)
del FP, stale
AFP = AF.load()
for k in ["pt_n", "pt_up", "pt_dn", "pt_net20", "pt_gap", "pt_firms", "pt_gap_chg20", "ev_buyback", "ev_buyback20",
          "ev_guid_up", "ev_guid_up20", "ev_guid_dn", "ev_guid_dn20", "ev_guid_aff", "ev_guid_aff20", "ev_exec", "ev_exec20"]:
    put("an_" + k, AFP[k], anl)
put("an_pt_gap_rank", AFP["pt_gap"].reindex(index=days, columns=cols).rank(axis=1, pct=True), anl)
del AFP
y = H.fwd(h).stack(future_stack=True).reindex(idx)
yr = y.groupby(level=0).rank(pct=True) - 0.5
dates = idx.get_level_values(0)
di = days.get_indexer(dates)
sub = (di % 5 == 0)
ok = yr.notna().values
print("rows", len(X), "train-eligible", int((ok & sub).sum()), "features", len(base), len(fund), len(anl), flush=True)
pr = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=1000, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
sets = {"price": base, "price+fund": base + fund, "price+analyst": base + anl, "all": base + fund + anl}
preds = {k: [] for k in sets}
imp = {}
for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 1 - h - 10)]
    tr = ok & sub & (dates < cut)
    te = (dates >= q.start_time) & (dates <= q.end_time)
    if te.sum() == 0:
        continue
    for name, feats in sets.items():
        m = lgb.train(pr, lgb.Dataset(X.loc[tr, feats], yr.values[tr]), num_boost_round=300)
        preds[name].append(pd.Series(m.predict(X.loc[te, feats]), index=idx[te]))
        if name == "all":
            imp = pd.Series(m.feature_importance("gain"), index=feats)
    print(q, flush=True)
pred = pd.DataFrame({k: pd.concat(v) for k, v in preds.items()})
pred.to_parquet(f"{RES}/study{SID}_pred.parquet")
rows = []
offsets = range(0, h, max(1, h // 5)) if h > 5 else [0]
for name in sets:
    S = pred[name].unstack().reindex(columns=cols)
    for off in offsets:
        bt = H.backtest(S, h, k=20, offset=off)
        for r in H.summarize(name, bt, h):
            rows.append(dict(r, offset=off))
df = pd.DataFrame(rows)
g = df.groupby(["variant", "period"]).mean(numeric_only=True).drop(columns="offset")
g.to_csv(f"{RES}/study{SID}_horizon_ml.csv")
pd.set_option("display.width", 220)
print(g.round(3).to_string())
print((imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())
