"""Robustness of the overnight LightGBM strategy (buy top-k in the closing auction, sell in the opening auction).

Uses the 15:45-feature predictions (study3_timing.py) when present, else the close-feature predictions.
Checks: cost sensitivity (auction fraction of the quoted half-spread, extra bps per side for the measured
Yahoo-open vs opening-cross gap), top-k neighborhood, half-year subperiods, market-hedged alpha,
block-bootstrap Sharpe CI, and the deflated Sharpe ratio for the number of variants tried in this project.
Output: results/study3_robust.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, half_spread_panel, ann_stats, deflated_sharpe, block_bootstrap_sharpe, RES
import bt

N_TRIALS = 250          # approximate count of strategy variants evaluated across studies 0-8
P = load_panel()
cols = stock_cols(P)
o, c = P["o"][cols], P["c"][cols]
R = o.shift(-1) / c - 1
hs = half_spread_panel(P, "15:45")[cols]
fn = f"{RES}/study3_pred_night_1545.parquet"
src = "features_at_1545" if os.path.exists(fn) and "--close" not in sys.argv else "close_features"
pred = pd.read_parquet(fn if src == "features_at_1545" else f"{RES}/study3_pred_night.parquet")["pred"]
S = pred.unstack().reindex(columns=cols)
S = S.loc["2024-01-02":]
spyN = (P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1).reindex(S.index)
rows = []


def add(tag, r, **kw):
    for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]:
        x = r.net.loc[a:b]
        st = ann_stats(x)
        rows.append(dict(test=tag, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                         gross_bps=1e4 * r.gross.loc[a:b].mean(), cost_bps=1e4 * r.cost.loc[a:b].mean(), **kw))


# base case and cost sensitivity (k = 10)
W10 = bt.select_topk(S, S.notna(), 10)
for frac in [0.1, 0.25, 0.5, 1.0]:
    for extra in [0.0, 2.5, 5.0, 10.0]:
        cost = (0.3 + 1.0 + frac * hs + extra).reindex_like(W10)
        add("cost", bt.run(W10, R.reindex_like(W10), cost), auction_frac=frac, extra_bps=extra, k=10)
base_cost = (0.3 + 1.0 + 0.1 * hs + 2.5)
for k in [3, 5, 7, 10, 15, 20, 30]:
    W = bt.select_topk(S, S.notna(), k)
    add("k", bt.run(W, R.reindex_like(W), base_cost.reindex_like(W)), k=k)
r = bt.run(W10, R.reindex_like(W10), base_cost.reindex_like(W10))
# half-year subperiods
for (y, h), x in r.net.groupby([r.index.year, (r.index.month - 1) // 6]):
    st = ann_stats(x)
    rows.append(dict(test="half_year", period=f"{y}H{h + 1}", sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                     maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[x.index].mean(), k=10))
# hedged alpha vs SPY overnight
X = pd.concat([r.net, spyN], axis=1).dropna()
b = np.polyfit(X.iloc[:, 1], X.iloc[:, 0], 1)
res = X.iloc[:, 0] - b[0] * X.iloc[:, 1]
rows.append(dict(test="hedged_vs_spy_night", period="2024-26", sharpe=res.mean() / res.std() * np.sqrt(252),
                 beta=b[0], alpha_bps=1e4 * b[1], k=10))
# bootstrap CI and deflated Sharpe (daily SR)
x = r.net.loc["2024-01":"2026-09"].dropna()
lo, med, hi = block_bootstrap_sharpe(x, block=10, n=2000)
sr_d = x.mean() / x.std()
dsr = deflated_sharpe(sr_d, N_TRIALS, len(x), skew=float(x.skew()), kurt=float(x.kurt() + 3))
xo = r.net.loc["2025-07":"2026-09"].dropna()
dsr_oos = deflated_sharpe(xo.mean() / xo.std(), N_TRIALS, len(xo), skew=float(xo.skew()), kurt=float(xo.kurt() + 3))
rows.append(dict(test="bootstrap_ci", period="2024-26", sharpe=med, ci_lo=lo, ci_hi=hi, k=10))
rows.append(dict(test="deflated_sharpe_prob", period="2024-26", value=dsr, n_trials=N_TRIALS, k=10))
rows.append(dict(test="deflated_sharpe_prob", period="oos", value=dsr_oos, n_trials=N_TRIALS, k=10))
df = pd.DataFrame(rows)
df["signal"] = src
df.to_csv(f"{RES}/study3_robust{'' if src == 'features_at_1545' else '_closefeat'}.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.round(3).to_string())
