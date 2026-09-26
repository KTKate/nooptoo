"""Re-price overnight (close -> next open) trades at the primary-exchange opening cross.

For the two overnight leads from study 1 (overnight_mom20, intraday_loser_overnight) in the M and S
liquidity tiers, take the top-10 picks on a random sample of days in 2024-01..2026-09 and compare
  Yahoo open (first regular trade) vs official opening cross (SIP 'O' print).
Output: results/overnight_auction_check.csv
"""
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, RES
import bt

A.RL = A.RateLimiter(50)
P = load_panel()
cols = stock_cols(P)
o, c, rawc, dv = (P[k][cols] for k in ["o", "c", "rawc", "dv"])
adv = dv.rolling(20, min_periods=10).median().shift(1)
px = rawc.shift(1)
tiers = {"M": (px > 5) & (adv > 5e6) & (adv <= 5e7), "S": (px > 2) & (adv > 1e6) & (adv <= 5e6)}
night = o / c.shift(1) - 1
on_mom20 = np.log1p(night).rolling(20, min_periods=15).mean()
oc = c / o - 1
sigs = {"overnight_mom20": on_mom20, "intraday_loser_overnight": -oc}
rng = np.random.default_rng(7)
days = c.loc["2024-01-02":"2026-09-23"].index
sample = sorted(rng.choice(np.arange(len(days)), 120, replace=False))
rows = []
for sn, S in sigs.items():
    for tn, E in tiers.items():
        W = bt.select_topk(S, E & S.notna(), 10)
        for i in sample:
            d = days[i]
            nxt = c.index[c.index.get_loc(d) + 1]
            for t in W.columns[W.loc[d] > 0]:
                rows.append(dict(strat=sn, tier=tn, date=d, next=nxt, ticker=t, close=float(rawc.at[d, t]),
                                 yopen=float(o.at[nxt, t] * rawc.at[nxt, t] / c.at[nxt, t])))
T = pd.DataFrame(rows)
ap = A.auction_prices(zip(T.ticker, T.next))
T = T.merge(ap.rename(columns={"date": "next"}), on=["ticker", "next"], how="left")
T["r_yahoo"] = T.yopen / T.close - 1
T["r_cross"] = T.open_off / T.close - 1
T.to_csv(f"{RES}/overnight_auction_check.csv", index=False)
g = T.dropna(subset=["r_cross"]).groupby(["strat", "tier"])
print(g[["r_yahoo", "r_cross"]].mean().mul(1e4).round(1).assign(n=g.size(),
      found=T.groupby(["strat", "tier"]).open_off.apply(lambda x: x.notna().mean())))
