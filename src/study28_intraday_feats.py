"""Study 28: do intraday features known at 15:45 add to the overnight (close -> next open) model?

Intraday features per stock-day (raw 5-minute bars, data/local/m5snap, 09:30-10:00 and 15:30-16:00, ET;
only bars that have closed by 15:45 are used, i.e. 15:30, 15:35, 15:40 in the afternoon):
  r_late    : 15:45 price / 15:30 open - 1
  r_open    : 10:00 price (09:55 bar close) / 09:30 open - 1
  vwap_dev  : 15:45 price / VWAP(15:30-15:45) - 1
  rvol_late : log(volume 15:30-15:45 / its mean over the previous 20 trading days)
  rvol_open : log(volume 09:30-10:00 / its mean over the previous 20 trading days)
  gap       : 09:30 open / previous official close - 1
  ln_n_late, ln_n_open : log(1 + number of trades) in the two windows
  rn_late, rn_open     : log(trades / their mean over the previous 20 trading days)
Base signal: results/study3_pred_night_1545_none.parquet (the live model's 15:45 predictions, out of sample
quarter by quarter), whose rows define the universe.
A. Stacking LightGBM (num_leaves 15, min_data_in_leaf 500, 200 rounds, num_threads 2) predicting the daily
   cross-sectional rank of R from [rank of base prediction, intraday features]. Static: train 2024-01..2025-06,
   test 2025-07..2026-09. Walk-forward: expanding window, retrained each quarter from 2024Q3.
   Two feature encodings: raw values and daily cross-sectional percentile ranks.
B. Univariate decile spreads (top minus bottom decile, overnight return) and rank IC per half-year.
Evaluation: top-10 portfolio, buy closing auction, sell next opening auction, per-side cost
exec_cost_bps(P, "auction") + 2.5 bp.
Output: results/study28_portfolio.csv, results/study28_univariate.csv, results/study28_summary.csv
"""
import os
import resource
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew, kurtosis
import core
import bt
from core import RES, ann_stats, deflated_sharpe

LOCAL = os.path.join(core.DATA, "local")
FEATS = ["r_late", "r_open", "vwap_dev", "rvol_late", "rvol_open", "gap", "ln_n_late", "ln_n_open",
         "rn_late", "rn_open"]
PRIOR_TRIALS = 1170

# ---------------------------------------------------------------- base prediction and universe
pred = pd.read_parquet(f"{RES}/study3_pred_night_1545_none.parquet")["pred"]
pred = pred[pred.index.get_level_values(0) >= "2024-01-01"]
tick = sorted(pred.index.get_level_values(1).unique())
pdays = sorted(pred.index.get_level_values(0).unique())

P = core.load_panel()
cols = [t for t in tick if t in P["c"].columns]
R = (P["o"].shift(-1) / P["c"] - 1)[cols]
COST = (core.exec_cost_bps(P, "auction")[cols] + 2.5)
prevc = P["rawc"][cols].shift(1)
R = R.loc["2023-11-01":]
COST = COST.loc["2023-11-01":]
prevc = prevc.loc["2023-11-01":]
del P
import gc; gc.collect()

# ---------------------------------------------------------------- intraday aggregates, month by month
rows = []
for f in sorted(x for x in os.listdir(os.path.join(LOCAL, "m5snap")) if x.endswith(".parquet")):
    d = pd.read_parquet(os.path.join(LOCAL, "m5snap", f), columns=["ts", "ticker", "o", "c", "v", "n", "vwap"])
    d = d[d.ticker.isin(set(cols))]
    hm = d.ts.dt.hour * 100 + d.ts.dt.minute
    d["date"] = d.ts.dt.normalize()
    d["pv"] = d.vwap.astype("float64") * d.v
    op = d[(hm >= 930) & (hm <= 955)]
    lt = d[(hm >= 1530) & (hm <= 1540)]
    g1 = op.groupby(["date", "ticker"]).agg(v_open=("v", "sum"), n_open=("n", "sum"))
    o930 = op[hm.loc[op.index] == 930].set_index(["date", "ticker"])["o"].rename("o930")
    c955 = op[hm.loc[op.index] == 955].set_index(["date", "ticker"])["c"].rename("c955")
    g2 = lt.groupby(["date", "ticker"]).agg(v_late=("v", "sum"), n_late=("n", "sum"), pv_late=("pv", "sum"))
    o1530 = lt[hm.loc[lt.index] == 1530].set_index(["date", "ticker"])["o"].rename("o1530")
    c1540 = lt[hm.loc[lt.index] == 1540].set_index(["date", "ticker"])["c"].rename("c1540")
    rows.append(pd.concat([g1, o930, c955, g2, o1530, c1540], axis=1).astype("float64"))
    del d, op, lt
agg = pd.concat(rows)
del rows
W = {k: agg[k].unstack().reindex(columns=cols) for k in agg.columns}
del agg
days = W["v_open"].index


def rel20(x):
    """log(x / mean of x over the previous 20 trading days), at least 10 observations."""
    m = x.rolling(20, min_periods=10).mean().shift(1)
    return np.log(x / m).replace([np.inf, -np.inf], np.nan)


F = {
    "r_late": W["c1540"] / W["o1530"] - 1,
    "r_open": W["c955"] / W["o930"] - 1,
    "vwap_dev": W["c1540"] / (W["pv_late"] / W["v_late"]) - 1,
    "rvol_late": rel20(W["v_late"]),
    "rvol_open": rel20(W["v_open"]),
    "gap": W["o930"] / prevc.reindex(days) - 1,
    "ln_n_late": np.log1p(W["n_late"]),
    "ln_n_open": np.log1p(W["n_open"]),
    "rn_late": rel20(W["n_late"]),
    "rn_open": rel20(W["n_open"]),
}
del W
for k in ["r_late", "r_open", "vwap_dev", "gap"]:
    F[k] = F[k].clip(-0.5, 0.5)

# ---------------------------------------------------------------- long frame on the prediction universe
X = pd.DataFrame({"pred": pred})
X["R"] = R.stack().reindex(X.index)
for k in FEATS:
    X[k] = F[k].stack().reindex(X.index).astype("float32")
del F
gc.collect()
X = X[X.R.notna()].copy()
dt = X.index.get_level_values(0)
X["y"] = X.groupby(level=0)["R"].rank(pct=True)
X["pred_rk"] = X.groupby(level=0)["pred"].rank(pct=True)
for k in FEATS:
    X[k + "_rk"] = X.groupby(level=0)[k].rank(pct=True)
print("rows", len(X), "feature coverage:", X[FEATS].notna().mean().round(3).to_dict())

PARAMS = dict(objective="regression", num_leaves=15, min_data_in_leaf=500, learning_rate=0.05,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=2, verbose=-1, seed=7)
ENC = {"raw": ["pred_rk"] + FEATS, "rank": ["pred_rk"] + [k + "_rk" for k in FEATS]}


def fit_predict(tr, te, fs):
    m = lgb.train(PARAMS, lgb.Dataset(X.loc[tr, fs], X.loc[tr, "y"]), num_boost_round=200)
    return pd.Series(m.predict(X.loc[te, fs]), index=X.index[te]), m


scores = {"base": X["pred"]}
imp = {}
tr = dt <= "2025-06-30"
te = dt >= "2025-07-01"
for enc, fs in ENC.items():
    s, m = fit_predict(tr, te, fs)
    scores[f"static_{enc}"] = s
    imp[f"static_{enc}"] = pd.Series(m.feature_importance("gain"), index=fs)
    qs = pd.period_range("2024Q3", pd.Period(dt.max(), "Q"), freq="Q")
    parts = []
    for q in qs:
        trq = dt < q.start_time
        teq = (dt >= q.start_time) & (dt <= q.end_time)
        if teq.sum() == 0:
            continue
        parts.append(fit_predict(trq, teq, fs)[0])
    scores[f"wf_{enc}"] = pd.concat(parts)

# ---------------------------------------------------------------- portfolio evaluation
PERIODS = [("2024H1", "2024-01", "2024-06"), ("2024H2", "2024-07", "2024-12"), ("2025H1", "2025-01", "2025-06"),
           ("2025H2", "2025-07", "2025-12"), ("2026H1", "2026-01", "2026-06"), ("2026Q3", "2026-07", "2026-09"),
           ("val_2024-01_2025-06", "2024-01", "2025-06"), ("wf_2024-07_2025-06", "2024-07", "2025-06"),
           ("oos_2025-07_2026-09", "2025-07", "2026-09")]
Rl = X["R"]
out, nets = [], {}
for name, s in scores.items():
    S = s.unstack().reindex(columns=cols)
    Wt = bt.select_topk(S, S.notna(), 10)
    r = bt.run(Wt, R.reindex_like(Wt), COST.reindex_like(Wt))
    nets[name] = r.net
    ic = pd.DataFrame({"s": s, "R": Rl.reindex(s.index)}).groupby(level=0).apply(
        lambda g: g.s.rank().corr(g.R.rank()))
    for per, a, b in PERIODS:
        rn = r.net.loc[a:b]
        if len(rn.dropna()) < 20:
            continue
        st = ann_stats(rn)
        out.append(dict(variant=name, period=per, days=st["n"], net_sharpe=st["sharpe"], net_annret=st["ann_ret"],
                        maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[a:b].mean(),
                        cost_bps=1e4 * r.cost.loc[a:b].mean(), rank_ic=ic.loc[a:b].mean(),
                        ic_t=ic.loc[a:b].mean() / ic.loc[a:b].std() * np.sqrt(ic.loc[a:b].count())))
port = pd.DataFrame(out)
port.to_csv(f"{RES}/study28_portfolio.csv", index=False)

# ---------------------------------------------------------------- B: univariate decile spreads
hy = pd.PeriodIndex(dt, freq="Q").map(lambda q: f"{q.year}H{1 if q.quarter <= 2 else 2}")
uni = []
for k in FEATS:
    d = X[[k, "R"]].copy()
    d["hy"] = hy
    d = d[d[k].notna()]
    d["dec"] = d.groupby(level=0)[k].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    dm = d.groupby([d.index.get_level_values(0), "dec"])["R"].mean().unstack()
    spread = (dm[9] - dm[0])
    ic = d.groupby(level=0).apply(lambda g: g[k].rank().corr(g.R.rank()))
    h = pd.Series(pd.PeriodIndex(spread.index, freq="Q").map(
        lambda q: f"{q.year}H{1 if q.quarter <= 2 else 2}"), index=spread.index)
    for per in sorted(h.unique()) + ["all"]:
        msk = (h == per) if per != "all" else h.notna()
        sp, icp = spread[msk], ic.reindex(spread.index)[msk]
        uni.append(dict(feature=k, period=per, days=len(sp), d10_minus_d1_bps=1e4 * sp.mean(),
                        spread_t=sp.mean() / sp.std() * np.sqrt(len(sp)), rank_ic=icp.mean(),
                        ic_t=icp.mean() / icp.std() * np.sqrt(icp.count())))
unidf = pd.DataFrame(uni)
unidf.to_csv(f"{RES}/study28_univariate.csv", index=False)

# ---------------------------------------------------------------- summary: OOS deltas and deflated Sharpe
n_var = 4 + len(FEATS)          # four stacked portfolios plus ten univariate features examined
summ = []
for name, rn in nets.items():
    for per, a, b in [("oos_2025-07_2026-09", "2025-07", "2026-09"), ("wf_2024-07_2025-06", "2024-07", "2025-06")]:
        x = rn.loc[a:b].dropna()
        if len(x) < 20 or (name.startswith("static") and per.startswith("wf")):
            continue
        sr = x.mean() / x.std()
        dsr = deflated_sharpe(sr, n_var + PRIOR_TRIALS, len(x), skew=skew(x), kurt=kurtosis(x, fisher=False))
        diff = x - nets["base"].loc[a:b].reindex(x.index)
        summ.append(dict(variant=name, period=per, days=len(x), net_sharpe=sr * np.sqrt(252),
                         sharpe_minus_base=sr * np.sqrt(252) - nets["base"].loc[a:b].pipe(
                             lambda y: y.mean() / y.std() * np.sqrt(252)),
                         diff_mean_bps=1e4 * diff.mean(), diff_t=diff.mean() / diff.std() * np.sqrt(len(diff)),
                         n_variants=n_var, trials=n_var + PRIOR_TRIALS, deflated_sharpe=dsr))
summ = pd.DataFrame(summ)
summ.to_csv(f"{RES}/study28_summary.csv", index=False)

pd.set_option("display.width", 250)
print(port.round(3).to_string())
print(unidf[unidf.period == "all"].round(3).to_string())
print(unidf.pivot(index="feature", columns="period", values="d10_minus_d1_bps").round(1).to_string())
print(summ.round(3).to_string())
for k, v in imp.items():
    print(k, (v / v.sum()).round(3).sort_values(ascending=False).to_dict())
print("peak RSS GB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
