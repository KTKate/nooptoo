"""Study 23: the five-model ensemble (src/ensemble.py) scored with 15:45 features, the way it would trade.

Same day-by-day feature substitution as study3_timing.py (15:45 price, day range widened to include it, 85% of the
day's volume, SPY/IWM at 15:44), SNAP_MODE none (no hindsight filter). For every day 2024-01 .. 2026-09 the members
and the ensemble score are computed with the models of that quarter; then top-k portfolios, closing-auction buy,
opening-auction sell, auction costs + 2.5 bp per side, against the pooled model alone.
Output: results/study23_pred.parquet, results/study23_ensemble_timing.csv
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
MODE = os.environ.get("SNAP_MODE", "none")
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
    return ensemble.predict(X, q)



import ensemble
from multiprocessing import Pool
if __name__ == "__main__":
    with Pool(3) as pool:
        rows = pool.map(one_day, test_days, chunksize=8)
    pred = pd.concat(rows)
    pred.to_parquet(f"{RES}/study23_pred.parquet")
    R = o.shift(-1) / c - 1
    cost = exec_cost_bps(P, "auction")[cols] + 2.5
    out = []
    for m in ["pooled", "pooled_beh", "pooled_com", "per_beh", "per_com", "ensemble"]:
        S = pred[m].unstack().reindex(columns=cols)
        S = S.loc[S.index < days[-1]]
        for k in [5, 10, 20]:
            W = bt.select_topk(S, S.notna(), k)
            r = bt.run(W, R.reindex_like(W), cost.reindex_like(W))
            for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]:
                st = ann_stats(r.net.loc[a:b])
                out.append(dict(model=m, k=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                                gross_bps=1e4 * r.gross.loc[a:b].mean(), cost_bps=1e4 * r.cost.loc[a:b].mean()))
    df = pd.DataFrame(out)
    df.to_csv(f"{RES}/study23_ensemble_timing.csv", index=False)
    print(df.pivot_table(index=["model", "k"], columns="period", values="sharpe").round(2).to_string())
