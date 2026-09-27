"""Study 3 timing test: can the overnight LightGBM model be traded with information available at 15:45?

Market-on-close orders must be entered by about 15:50, so the day-t features cannot use the closing print.
For every day t in 2024-01..2026-09 the feature panels are cut at t (260-day history), and row t is replaced by:
  close   -> price at 15:45 (close of the 15:40-15:45 bar, Alpaca SIP), on the dividend-adjusted scale
  high/low-> the daily high/low widened to include the 15:45 price (small look-ahead: the last 15 minutes
             can move the day's range)
  volume  -> 0.85 x daily volume (the closing auction and last minutes are not yet known)
  SPY/IWM -> 15:44 minute close; VIX -> daily close (small look-ahead, the model uses it as a regime input)
The quarterly models saved by study3_ml.py (Q_START=2024Q1 night) predict on these rows. The trade is
unchanged: buy in the closing auction of t, sell in the opening auction of t+1.
SNAP_MODE (environment, default base) selects the Alpaca-Yahoo consistency filter of the 15:45 prices
(study8_exec_retest.snap_panels); the base filter drops whole tickers using the full 2024-26 history, so
SNAP_MODE=none is the look-ahead-free check. Non-base modes write files with a _<mode> suffix.
Output: results/study3_timing.csv, results/study3_pred_night_1545.parquet
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import alpaca_data as A
import ml_features as M
from core import exec_cost_bps, ann_stats, RES, DATA
import bt

P, cols = M.P, M.cols
days = M.days
earn = M.earnings_features()
from study8_exec_retest import snap_panels
MODE = os.environ.get("SNAP_MODE", "base")
SUF = "" if MODE == "base" else f"_{MODE}"
snap = snap_panels(MODE)
p1545 = snap["c15:40"].reindex(columns=cols)
m1 = A.read("m1", start="2023-12", tickers=["SPY", "IWM"])
m1 = m1[m1.ts.dt.strftime("%H:%M") == "15:44"]
m1["date"] = m1.ts.dt.normalize()
etf1545 = m1.pivot(index="date", columns="ticker", values="c")

o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
adjf = c / rawc
test_days = [d for d in days if d >= pd.Timestamp("2024-01-02") and d in p1545.index and d in etf1545.index]
live_cols = [t for t in cols if P["dv"][t].loc["2023-06":].max() > 3e6]   # others can never pass the universe filter


def one_day(d):
    i = days.get_loc(d)
    sl = slice(max(0, i - 260), i + 1)
    oo, hh, ll, cc, rr, vv, dd = (x[live_cols].iloc[sl].copy() for x in (o, h, l, c, rawc, v, dv))
    px = p1545.loc[d].reindex(live_cols)
    ok = px.notna()
    cc.iloc[-1] = np.where(ok, px * adjf.loc[d, live_cols], np.nan)
    rr.iloc[-1] = np.where(ok, px, np.nan)
    hh.iloc[-1] = np.maximum(hh.iloc[-1], cc.iloc[-1])
    ll.iloc[-1] = np.minimum(ll.iloc[-1], cc.iloc[-1])
    vv.iloc[-1] = vv.iloc[-1] * 0.85
    dd.iloc[-1] = dd.iloc[-1] * 0.85
    spy = P["c"]["SPY"].iloc[sl].copy()
    spy.iloc[-1] = etf1545.at[d, "SPY"] * (P["c"]["SPY"].loc[d] / P["rawc"]["SPY"].loc[d])
    iwm = P["c"]["IWM"].iloc[sl].copy()
    iwm.iloc[-1] = etf1545.at[d, "IWM"] * (P["c"]["IWM"].loc[d] / P["rawc"]["IWM"].loc[d])
    F, mkt, adv20 = M.build(oo, hh, ll, cc, rr, vv, dd, spy, P["c"]["^VIX"].iloc[sl], P["c"]["^VIX3M"].iloc[sl],
                            iwm, earn=earn)
    last = {k: f.iloc[[-1]] for k, f in F.items()}
    univ = ((rr > 5) & (adv20 > 5e6)).iloc[[-1]] & ok.values
    X = M.features_frame(last, mkt.iloc[[-1]], univ)
    q = pd.Period(d, freq="Q")
    mdl = lgb.Booster(model_file=f"{DATA}/models/night_{q}.txt")
    X = X[mdl.feature_name()]
    return pd.Series(mdl.predict(X), index=X.index)


from multiprocessing import Pool
with Pool(4) as pool:
    rows = pool.map(one_day, test_days, chunksize=8)
pred = pd.concat(rows)
pd.DataFrame({"pred": pred}).to_parquet(f"{RES}/study3_pred_night_1545{SUF}.parquet")

R = o.shift(-1) / c - 1
c_auc = exec_cost_bps(P, "auction")[cols]
base = pd.read_parquet(f"{RES}/study3_pred_night_from2024Q1.parquet")["pred"]   # same models
out = []
for name, pr in [("close_features", base), ("features_at_1545", pred)]:
    S = pr.unstack().reindex(index=days, columns=cols)
    S = S.loc["2024-01-02":]
    for k in [5, 10, 20]:
        W = bt.select_topk(S, S.notna(), k)
        r = bt.run(W, R.reindex_like(W), c_auc.reindex_like(W))
        for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
            st = ann_stats(r.net.loc[a:b])
            out.append(dict(signal=name, k=k, period=per, net_sharpe=st["sharpe"], net_annret=st["ann_ret"],
                            maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[a:b].mean(),
                            cost_bps=1e4 * r.cost.loc[a:b].mean()))
df = pd.DataFrame(out)
df.to_csv(f"{RES}/study3_timing{SUF}.csv", index=False)
print(df.round(3).to_string())
# overlap of the top-10 lists
S1 = base.unstack().reindex(columns=cols).loc["2024-01-02":]
S2 = pred.unstack().reindex(columns=cols)
S1 = S1.reindex(S2.index)
W1, W2 = bt.select_topk(S1, S1.notna(), 10) > 0, bt.select_topk(S2, S2.notna(), 10) > 0
print("mean overlap of top-10 lists:", (W1 & W2).sum(1).mean())
