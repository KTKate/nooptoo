"""Study 71: the full day cycle on one margin account, 2024-07..2026-09.
Legs (each already net of auction costs + 2.5 bp per side in its own study):
  night      overnight blend top 10 (study 33), 100% of equity, closing-auction buy -> opening-auction sell
  day_long   study-60 day-session long model ('all' inputs), top 10, opening-auction buy -> closing-auction sell
  day_short  study-68 short of the blend's easy-to-borrow picks, opening auction -> closing auction
Daily P&L = night + a * day_long + b * day_short (the day legs are held while the night leg is flat, so a and b set
the day-session gross exposure; margin needed when a + b > 0, gross exposure at most a + b during the day).
The day-short leg of night t-1's picks runs in the day session of t; day_long of date t likewise. Grid a in
{0, 0.25, 0.5}, b in {0, 0.25, 0.5}; chosen on 2024H2-25H1, checked on 2025H2-26.
Output: results/study71_full_cycle.csv
"""
import numpy as np
import pandas as pd
exec(open("study68_day_short_picks.py").read().split("variants = {")[0])
night_leg = bt.run(W, night, cost).net                                         # indexed by entry day t
m = (W > 0) & etbm
short_leg = bt.run(m.astype(float).div(10), day, cost + fee, side=-1).net      # day session of t+1, indexed by t
x = pd.read_parquet(f"{RES}/study60_pred.parquet")
liq = (x.px > 5) & (x.adv20 > 5e6)
sel = x[liq & x.p_all.notna()].sort_values("p_all", ascending=False).groupby("date").head(10)
g = sel.groupby("date")
dl = g.R.mean() - 2 * g.cost.mean() / 1e4                                       # day session of date d
prev = pd.Series(days[:-1], index=days[1:])
dl.index = dl.index.map(prev)                                                   # align to the preceding night
idx = night_leg.loc["2024-07":"2026-09"].dropna().index
rows = []
for a in [0, 0.25, 0.5]:
    for b in [0, 0.25, 0.5]:
        tot = night_leg.reindex(idx) + a * dl.reindex(idx).fillna(0) + b * short_leg.reindex(idx).fillna(0)
        for p, a0, b0 in [("2024H2-25H1", "2024-07", "2025-06"), ("2025H2-26", "2025-07", "2026-09"),
                          ("2024H2-26", "2024-07", "2026-09")]:
            st = ann_stats(tot.loc[a0:b0])
            rows.append(dict(day_long=a, day_short=b, period=p, sharpe=st["sharpe"], ann=st["ann_ret"],
                             maxdd=st["maxdd"], bp_day=1e4 * tot.loc[a0:b0].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study71_full_cycle.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["day_long", "day_short"], columns="period", values=["sharpe", "maxdd", "ann"]).round(2).to_string())
c = pd.concat([night_leg.reindex(idx), dl.reindex(idx), short_leg.reindex(idx)], axis=1, keys=["night", "day_long", "day_short"])
print(c.corr().round(2))
