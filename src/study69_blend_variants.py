"""Studies 59, 65 and 69: variants of the overnight blend (study 33; top 10 by 2 x ensemble rank + jump-minus-drop
rank at 15:45, closing-auction buy, opening-auction sell, auction costs + 2.5 bp per side), 2024-01..2026-09.

69  Night-2 re-entry (study 53: the picks gain another 24-28 bp over the universe on the second night). Each night
    hold today's top 10 plus yesterday's top 10 that are not in today's list (equal weight across all names), or
    yesterday's picks alone.
59  Avoid filters from studies 41-42: drop names with a guidance cut headline in the last 20 news windows
    (analyst_features ev_guid_dn20), and names whose earnings reaction (close before the report to the close after
    it, known by today's close) was below -10% within the last 20 trading days.
65  Calendar: mean net return of the blend by weekday of entry, nights before a market holiday or a weekend, the last
    trading day of the month, and monthly options expiration Fridays. A skip rule is chosen on 2024-01..2025-06 and
    checked on 2025-07..2026-09.
Output: results/study69_blend_variants.csv, results/study65_calendar.csv
"""
import numpy as np
import pandas as pd
import store
import analyst_features as AF
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-1]]
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
per = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
rows = []


def rec(name, net):
    for p, a, b in per:
        st = ann_stats(net.loc[a:b])
        rows.append(dict(study=name.split(":")[0], variant=name, period=p, sharpe=st["sharpe"], ann=st["ann_ret"],
                         maxdd=st["maxdd"], net_bp=1e4 * net.loc[a:b].mean()))


W = bt.select_topk(S, S.notna(), 10)
base = bt.run(W, R, C).net
rec("base: blend top 10", base)
# 69: night-2 re-entry
Wy = W.shift(1).fillna(0)
Wy = Wy.where(ok, 0)
both = ((W > 0) | (Wy > 0)).astype(float)
both = both.div(both.sum(1).replace(0, np.nan), axis=0).fillna(0)
rec("69: today + yesterday's picks", bt.run(both, R, C).net)
y_only = (Wy > 0).astype(float).div(10)
rec("69: yesterday's picks only (night 2)", bt.run(y_only, R, C).net)
W20 = bt.select_topk(S, S.notna(), 20)
rec("69: reference top 20 today", bt.run(W20, R, C).net)
# 59: avoid filters
A = AF.load()
gdn = (A["ev_guid_dn20"].reindex_like(S) > 0)
E = store.read("earnings")
E["date"] = pd.to_datetime(E.date)
c = P["c"][cols]
react = pd.DataFrame(np.nan, index=days, columns=cols)
di = days.searchsorted(E.date)
ci = {t: i for i, t in enumerate(cols)}
for d, t in zip(di, E.symbol):
    if t in ci and 1 <= d < len(days) - 1:
        j = ci[t]
        r = c.iat[d + 1, j] / c.iat[d - 1, j] - 1          # known at the close of d + 1
        react.iat[d + 1, j] = r
bad_e = (react < -0.10).rolling(20, min_periods=1).max().fillna(0).astype(bool).reindex_like(S)
for name, m in [("59: no guidance cut (20 windows)", ~gdn), ("59: no earnings reaction < -10% (20 days)", ~bad_e),
                ("59: both filters", ~gdn & ~bad_e)]:
    Sf = S.where(m)
    rec(name, bt.run(bt.select_topk(Sf, Sf.notna(), 10), R, C).net)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study69_blend_variants.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index="variant", columns="period", values="sharpe", sort=False).round(2).to_string())
# 65: calendar
net = base.loc["2024-01":"2026-09"].dropna()
idx = net.index
pos = days.get_indexer(idx)
nxt = days[np.minimum(pos + 1, len(days) - 1)]
gap_days = (nxt - idx).days
month_end = pd.Series(idx.month, index=idx) != pd.Series(nxt.month, index=idx)
third_fri = (idx.weekday == 4) & (idx.day >= 15) & (idx.day <= 21)
groups = {"weekday": pd.Series(idx.day_name(), index=idx),
          "before_weekend_or_holiday": pd.Series(np.where(gap_days > 3, "holiday (gap > 3 days)",
                                                          np.where(gap_days == 3, "weekend", "next day")), index=idx),
          "month_end": month_end.map({True: "last day of month", False: "other"}),
          "opex_friday": pd.Series(np.where(third_fri, "monthly opex Friday", "other"), index=idx)}
crow = []
for gname, g in groups.items():
    for p, a, b in per[:2]:
        x = net.loc[a:b]
        gg = g.loc[a:b]
        for lev, v in x.groupby(gg):
            crow.append(dict(group=gname, level=lev, period=p, n=len(v), mean_bp=1e4 * v.mean(),
                             t=v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 2 else np.nan))
cdf = pd.DataFrame(crow)
cdf.to_csv(f"{RES}/study65_calendar.csv", index=False)
print(cdf.pivot_table(index=["group", "level"], columns="period", values=["mean_bp", "n"]).round(1).to_string())
