"""Study 60: day-session long model (buy at the opening auction, sell at the closing auction) with every input known
before 09:28 ET.

Earlier: study 25 (premarket model, m5snap universe, 2024+) and study 29 (study-3 next-day model) found no long-side
edge; the next-day model's top 10 earns 2-4 bp gross against about 9 bp of auction costs.
New here: study 25's premarket, news, sentiment and earnings inputs (data/local/m5pre/_study25_features.pkl, known at
09:25) plus the 53 study-3 inputs of the previous close (data/ml_frame.parquet, row of day t-1), the analyst panels
(analyst_features.py, day t-1 window, i.e. known by 15:45 of t-1) and fundamentals (fundamentals_panels.pkl, as of
day t). Target: within-day rank of the open-to-close return R. LightGBM regression, walk-forward by quarter, training
on all earlier quarters from 2024-01 (premarket data starts then), test 2024Q3..2026Q3, 5-day embargo.
Trading: top 10 (and 20) by prediction, price > $5, ADV > $5M; buy at the open, sell at the close, cost column of
study 25 (auction model + 2.5 bp per side). Compared with the same model without the new inputs (study-25 inputs only).
Rows with an open-to-close move of 100% or more are dropped (split or data errors in the study-25 frame).
Output: results/study60_day_session.csv
"""
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import analyst_features as AF
from core import load_panel, stock_cols, ann_stats, RES, DATA

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
x = pd.read_pickle(f"{DATA}/local/m5pre/_study25_features.pkl")
x["date"] = pd.to_datetime(x.date)
x = x[x.R.notna() & x.ticker.isin(cols) & (x.R.abs() < 1)].reset_index(drop=True)   # |R| >= 100%: split/data errors
pre = [c for c in x.columns if c not in ("ticker", "date", "R", "cost", "o930", "ygap", "prevc", "pm_first", "pm_last",
                                          "pm_hi", "pm_lo", "px")]
prev = pd.Series(days[:-1], index=days[1:])
x["prev"] = x.date.map(prev)
M = pd.read_parquet(f"{DATA}/ml_frame.parquet")
M = M[M.index.get_level_values(0) >= "2023-12-01"]
close_f = [c for c in M.columns if not c.startswith("y_")]
M = M[close_f].add_prefix("c_").reset_index().rename(columns={"level_0": "prev", "level_1": "ticker"})
M.columns = ["prev", "ticker"] + list(M.columns[2:])
x = x.merge(M, on=["prev", "ticker"], how="left")
close_f = ["c_" + c for c in close_f]
del M
new = []


def put(name, panel, key):
    s = panel.reindex(columns=cols).stack(future_stack=True)
    x[name] = s.reindex(pd.MultiIndex.from_arrays([x[key], x.ticker])).values.astype("float32")
    new.append(name)


A = AF.load()
for k in ["pt_n", "pt_up", "pt_dn", "pt_net20", "pt_gap", "pt_firms", "pt_gap_chg20", "ev_guid_up20", "ev_guid_dn20",
          "ev_buyback20", "ev_exec20"]:
    put("an_" + k, A[k], "prev")
del A
FP = pickle.load(open(f"{DATA}/local/fundamentals_panels.pkl", "rb"))
stale = FP["days_since_filing"] > 200
for k in ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "revenue_growth_yoy"]:
    put("fu_" + k, FP[k].where(~stale).rank(axis=1, pct=True), "date")
del FP, stale
x["y"] = x.groupby("date").R.rank(pct=True) - 0.5
print("rows", len(x), "days", x.date.nunique(), "inputs pre/close/new", len(pre), len(close_f), len(new), flush=True)
pr = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
sets = {"premarket_only": pre, "premarket+close": pre + close_f, "all": pre + close_f + new}
for k in sets:
    x["p_" + k] = np.nan
for q in pd.period_range("2024Q3", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 6)]
    tr = x.date < cut
    te = (x.date >= q.start_time) & (x.date <= q.end_time)
    if te.sum() == 0:
        continue
    for k, f in sets.items():
        m = lgb.train(pr, lgb.Dataset(x.loc[tr, f], x.y[tr]), num_boost_round=300)
        x.loc[te, "p_" + k] = m.predict(x.loc[te, f])
    print(q, flush=True)
liq = (x.px > 5) & (x.adv20 > 5e6)
rows = []
for k in sets:
    for n in [10, 20]:
        sel = x[liq & x["p_" + k].notna()].sort_values("p_" + k, ascending=False).groupby("date").head(n)
        g = sel.groupby("date")
        net = (g.R.mean() - 2 * g.cost.mean() / 1e4)
        gross = g.R.mean()
        for p, a, b in [("2024H2-25H1", "2024-07", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]:
            rows.append(dict(variant=k, k=n, period=p, sharpe=ann_stats(net.loc[a:b])["sharpe"],
                             gross_bp=1e4 * gross.loc[a:b].mean(), net_bp=1e4 * net.loc[a:b].mean(), days=len(net.loc[a:b])))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study60_day_session.csv", index=False)
print(df.round(3).to_string())
