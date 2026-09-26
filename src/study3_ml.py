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
Output: results/study3_preds.parquet and results/study3_ml.csv
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from core import load_panel, stock_cols, half_spread_bps, cost_bps, split_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
cs = cost_bps(half_spread_bps(P)[cols])
lr = np.log(c / c.shift(1))
night = np.log(o / c.shift(1))
intra = np.log(c / o)
adv20 = dv.rolling(20, min_periods=10).median()
univ = (rawc > 5) & (adv20 > 5e6)

F = {}
for n in [1, 2, 3, 5, 10, 20, 60, 120]:
    F[f"r{n}"] = lr.rolling(n, min_periods=max(1, n // 2)).sum()
F["r250_20"] = lr.rolling(250, min_periods=150).sum() - F["r20"]
F["night1"] = night
F["intra1"] = intra
F["night5"] = night.rolling(5).sum()
F["intra5"] = intra.rolling(5).sum()
F["night20"] = night.rolling(20, min_periods=15).mean()
F["intra20"] = intra.rolling(20, min_periods=15).mean()
F["vol5"] = lr.rolling(5).std()
F["vol20"] = lr.rolling(20, min_periods=15).std()
F["vol60"] = lr.rolling(60, min_periods=40).std()
F["vr"] = F["vol5"] / F["vol60"]
F["range1"] = np.log(h / l)
F["range20"] = F["range1"].rolling(20, min_periods=15).mean()
F["clv"] = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)   # close location in day range
F["clv5"] = F["clv"].rolling(5).mean()
F["vz1"] = np.log(v / v.rolling(20, min_periods=10).mean())
F["vz5"] = np.log(v.rolling(5).mean() / v.rolling(60, min_periods=40).mean())
F["ldv"] = np.log(adv20)
F["lpx"] = np.log(rawc)
F["hi20"] = np.log(c / c.rolling(20, min_periods=15).max())
F["lo20"] = np.log(c / c.rolling(20, min_periods=15).min())
F["hi250"] = np.log(c / c.rolling(250, min_periods=150).max())
spy = np.log(P["c"]["SPY"] / P["c"]["SPY"].shift(1))
beta = (lr.rolling(60, min_periods=40).cov(spy)).div(spy.rolling(60, min_periods=40).var(), axis=0)
F["beta"] = beta
F["resid1"] = lr - beta.mul(spy, axis=0)
F["resid5"] = F["resid1"].rolling(5).sum()
# earnings calendar features (dates are announced in advance; surprise known after report)
E = pd.read_parquet(f"{DATA}/earnings.parquet")
E = E[E.symbol.isin(cols)]
days = c.index
di = days.searchsorted(pd.to_datetime(E.date).values)
ev = pd.DataFrame({"di": di, "t": E.symbol.values, "s": E.surprise.values})
ev = ev[ev.di < len(days)]
flag = pd.DataFrame(0.0, index=days, columns=cols)
surp = pd.DataFrame(np.nan, index=days, columns=cols)
for d_, t_, s_ in ev.itertuples(index=False):
    flag.iat[d_, flag.columns.get_loc(t_)] = 1.0
    if d_ + 1 < len(days):
        surp.iat[d_ + 1, surp.columns.get_loc(t_)] = s_   # usable from close of D+1
idx = np.arange(len(days), dtype=float)
last_e = flag.mul(idx, axis=0).replace(0, np.nan).ffill()
next_e = flag.mul(idx, axis=0).replace(0, np.nan).bfill()
F["days_since_e"] = (-last_e).add(idx, axis=0).clip(upper=90)
F["days_to_e"] = next_e.sub(idx, axis=0).clip(upper=15)  # dates usually announced 2-4 weeks ahead
F["last_surp"] = surp.ffill(limit=70).clip(-100, 100)
mkt = pd.DataFrame(index=days)
mkt["spy1"] = spy
mkt["spy5"] = spy.rolling(5).sum()
mkt["spy20"] = spy.rolling(20).sum()
mkt["vix"] = P["c"]["^VIX"]
mkt["vix_chg5"] = np.log(P["c"]["^VIX"] / P["c"]["^VIX"].shift(5))
mkt["vix_term"] = P["c"]["^VIX"] / P["c"]["^VIX3M"]
mkt["iwm_spy5"] = np.log(P["c"]["IWM"] / P["c"]["IWM"].shift(5)) - mkt["spy5"]
mkt["dow"] = days.dayofweek
mkt["tom"] = (pd.Series(days.month, index=days) != pd.Series(days.month, index=days).shift(-1)).astype(float)

T = {
    "night": o.shift(-1) / c - 1,
    "day1": c.shift(-1) / o.shift(-1) - 1,
    "cc1": c.shift(-1) / c - 1,
    "cc5": c.shift(-5) / c - 1,
}
H = {"night": 1, "day1": 2, "cc1": 1, "cc5": 5}  # days until target fully realized (for embargo)


def stack():
    fn = f"{DATA}/ml_frame.parquet"
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    m = univ.stack()
    m = m[m]
    X = pd.DataFrame(index=m.index)
    for k, f in F.items():
        X[k] = f.stack().reindex(m.index).astype("float32")
    # cross-sectional ranks of a few key features (scale-free)
    for k in ["r1", "r5", "r20", "night1", "intra1", "vz1", "vol20", "resid5"]:
        X[k + "_cs"] = X[k].groupby(level=0).rank(pct=True).astype("float32")
    dates = X.index.get_level_values(0)
    for k in mkt.columns:
        X[k] = mkt[k].reindex(dates).values.astype("float32")
    for k, t in T.items():
        y = t.stack().reindex(m.index)
        X["y_" + k] = y.astype("float32")
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
quarters = pd.period_range("2022Q1", "2026Q3", freq="Q")
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
        if te.sum() == 0:
            continue
        ds = lgb.Dataset(X.loc[tr, feat], yr[tr])
        mdl = lgb.train(params, ds, num_boost_round=300)
        p = pd.Series(mdl.predict(X.loc[te, feat]), index=X.index[te])
        out.append(p)
        print(tgt, q, "train", int(tr.sum()), "test", int(te.sum()), flush=True)
        if q == quarters[-1]:
            imp = pd.Series(mdl.feature_importance("gain"), index=feat).sort_values(ascending=False)
            imp.to_csv(f"{RES}/study3_importance_{tgt}.csv")
    preds[tgt] = pd.concat(out)
    pd.DataFrame({"pred": preds[tgt]}).to_parquet(f"{RES}/study3_pred_{tgt}.parquet")
