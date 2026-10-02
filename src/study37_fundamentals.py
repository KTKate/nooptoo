"""Study 37: do point-in-time fundamentals improve the overnight models?

Data: data/local/fundamentals_panels.pkl (src/fetch_fundamentals.py; SEC companyfacts, a filing counts from the first
trading day after its filed date; ratios use the previous day's raw close). Inputs, as within-day percentile ranks
(robust to unit errors and outliers) plus log market cap:
  market_cap (log, and rank), ev_sales, pe_ttm, fcf_yield, sbc_to_revenue, op_margin, net_margin, book_to_market,
  cash_to_mcap, revenue_growth_yoy, days_since_filing
Values with days_since_filing > 200 are treated as missing (late or non-filers).
Models (study3/study10 design, close features, quarterly walk-forward 2022Q1..2026Q3, 10-day embargo):
  rank model + fundamentals     vs results/study10_pred_base.parquet (same design without them)
  jump/drop classifiers + fund. vs results/study32_pred.parquet (top 10 by P(jump) - P(drop))
Evaluation: top 10, closing-auction buy, opening-auction sell, auction costs + 2.5 bp per side; paired block
bootstrap of the daily net difference.
Output: results/study37_fundamentals.csv, results/study37_pred.parquet, results/study37_importance.csv
"""
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import store
import news_features as NF
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
X = pd.read_parquet(f"{DATA}/ml_frame.parquet")
X = X[X.index.get_level_values(0) >= "2020-01-01"]
base = [k for k in X.columns if not k.startswith("y_")]
idx = X.index
FP = pickle.load(open(f"{DATA}/local/fundamentals_panels.pkl", "rb"))
stale = FP["days_since_filing"] > 200
fund = []


def put(name, panel):
    X[name] = panel.reindex(index=days, columns=cols).stack(future_stack=True).reindex(idx).values.astype("float32")
    fund.append(name)


put("fu_log_mcap", np.log(FP["market_cap"].where(~stale & (FP["market_cap"] > 0))))
for k in ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "net_margin",
          "book_to_market", "cash_to_mcap", "revenue_growth_yoy"]:
    put("fu_" + k, FP[k].where(~stale).reindex(index=days, columns=cols).rank(axis=1, pct=True))
put("fu_days_since_filing", FP["days_since_filing"])
del FP
# study-32 extra inputs for the jump models
E = store.read("earnings")
di = days.searchsorted(pd.to_datetime(E.date))
earn = pd.DataFrame(0.0, index=days, columns=cols)
ci = {t: i for i, t in enumerate(cols)}
for d, t in zip(di, E.symbol):
    if t in ci and 0 < d < len(days):
        earn.iat[d, ci[t]] = 1.0
        earn.iat[d - 1, ci[t]] = 1.0
X["earn_tonight"] = earn.stack().reindex(idx).values.astype("float32")
X["news_today"] = NF.load()["n_news"].reindex(index=days, columns=cols).stack(future_stack=True).reindex(idx).values.astype("float32")
jbase = base + ["earn_tonight", "news_today"]
print("rows", len(X), "fundamental inputs", len(fund), "coverage", float(X[fund[1]].notna().mean()), flush=True)

y = X["y_night"]
ok = y.notna().values
yr = (y.groupby(level=0).rank(pct=True) - 0.5).values
dates = idx.get_level_values(0)
rp = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
bp = dict(rp, objective="binary")
out = {"rank": [], "jump": [], "drop": []}
imp = {}
for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
    cut = days[max(0, days.searchsorted(q.start_time) - 11)]
    tr = ok & (dates < cut)
    te = (dates >= q.start_time) & (dates <= q.end_time)
    if te.sum() == 0:
        continue
    m = lgb.train(rp, lgb.Dataset(X.loc[tr, base + fund], yr[tr]), num_boost_round=300)
    out["rank"].append(pd.Series(m.predict(X.loc[te, base + fund]), index=idx[te]))
    imp["rank"] = pd.Series(m.feature_importance("gain"), index=base + fund)
    for k, t in [("jump", y > 0.05), ("drop", y < -0.05)]:
        m = lgb.train(bp, lgb.Dataset(X.loc[tr, jbase + fund], t[tr].astype(float)), num_boost_round=300)
        out[k].append(pd.Series(m.predict(X.loc[te, jbase + fund]), index=idx[te]))
        imp[k] = pd.Series(m.feature_importance("gain"), index=jbase + fund)
    print(q, flush=True)
pred = pd.DataFrame({k: pd.concat(v) for k, v in out.items()})
pred.to_parquet(f"{RES}/study37_pred.parquet")
pd.DataFrame({k: v / v.sum() for k, v in imp.items()}).to_csv(f"{RES}/study37_importance.csv")

R = P["o"][cols].shift(-1) / P["c"][cols] - 1
C = exec_cost_bps(P, "auction")[cols] + 2.5
b10 = pd.read_parquet(f"{RES}/study10_pred_base.parquet")["pred"]
b32 = pd.read_parquet(f"{RES}/study32_pred.parquet")
scores = {"rank_fund": pred["rank"], "rank_base": b10, "jmd_fund": pred["jump"] - pred["drop"],
          "jmd_base": b32.p_jump - b32.p_drop}
net = {}
for k, s in scores.items():
    S = s.unstack().reindex(columns=cols)
    S = S.loc[S.index < days[-1]]
    W = bt.select_topk(S, S.notna(), 10)
    net[k] = bt.run(W, R.reindex_like(W), C.reindex_like(W)).net


def boot(x, block=10, n=4000):
    x = x.dropna().values
    T = len(x)
    rng = np.random.default_rng(0)
    m = np.array([x[((rng.integers(0, T, T // block + 1)[:, None] + np.arange(block)).ravel()[:T]) % T].mean()
                  for _ in range(n)])
    return x.mean() / m.std(), float((m <= 0).mean())


rows = []
per = [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
for k, r in net.items():
    d = dict(variant=k, **{p: ann_stats(r.loc[a:b])["sharpe"] for p, a, b in per})
    other = {"rank_fund": "rank_base", "jmd_fund": "jmd_base"}.get(k)
    if other:
        diff = (r - net[other]).dropna()
        d["bp_per_day_vs_base"] = 1e4 * diff.mean()
        d["t_vs_base"], d["p_vs_base"] = boot(diff)
        for p, a, b in per:
            d[f"t_{p}"] = boot(diff.loc[a:b])[0]
    rows.append(d)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study37_fundamentals.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string())
print(pd.read_csv(f"{RES}/study37_importance.csv", index_col=0).loc[fund].round(4).to_string())
