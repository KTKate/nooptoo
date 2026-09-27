"""Study 12: insider open-market purchases and sales (SEC Form 4) as a signal, 2020-01 .. 2026-03.

Signal known at the 15:45 decision of day t: filings dated up to t-1. Per stock and day:
  buyers30  distinct insiders with open-market purchases filed in the last 30 calendar days
  buy_val30 dollar value of those purchases; sellers30 / sell_val30 the same for sales
Portfolios (auction costs + 2.5 bp per side): each day buy the stocks with a new cluster purchase (buyers30 >= 2,
at least one filing on t-1), hold h = 1 (overnight), 5 or 20 days, overlapping cohorts with 1/h of capital each.
Also the overnight and 5-day mean excess returns by bucket. Universes as in study 10 (L+M: > $5M ADV, S: $1-5M).
Output: results/study12_insider.csv
"""
import os
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
d = pd.read_parquet(os.path.join(DATA, "local", "insider.parquet"))
d = d[d.ticker.isin(cols) & (d.value > 0)]
d["di"] = days.searchsorted(d.filed) + 1                          # usable from the next trading day
d = d[d.di < len(days)]
d["day"] = days[d.di.values]


def panel(code, what):
    x = d[d.code == code]
    daily = (x.groupby(["day", "ticker"]).owner.nunique() if what == "n" else x.groupby(["day", "ticker"]).value.sum())
    daily = daily.unstack().reindex(index=days, columns=cols).fillna(0.0)
    new = daily > 0
    return daily.rolling(21, min_periods=1).sum(), new         # ~30 calendar days


buyers30, new_buy = panel("P", "n")
sellers30, new_sell = panel("S", "n")
end = d.day.max()
sel = slice("2020-01-02", end)
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
px = traded_close(P)[cols].shift(1).fillna(P["rawc"][cols].shift(1))
U = {"LM": (px > 5) & (adv > 5e6), "S": (px > 2) & (adv > 1e6) & (adv <= 5e6)}
o, c = P["o"][cols], P["c"][cols]
Rn = o.shift(-1) / c - 1
Rcc = c.shift(-1) / c - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5
PER = [("2020-23", "2020-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", str(end.date()))]
rows = []
for un, E in U.items():
    for sig, m in [("cluster_buy", (buyers30 >= 2) & new_buy), ("any_buy", new_buy),
                   ("cluster_sell", (sellers30 >= 3) & new_sell)]:
        M = (E & m).loc[sel]
        W1 = M.astype("float32").div(M.sum(1).clip(lower=10), axis=0)      # at most 1/10 per name
        for h in [1, 5, 20]:
            if h == 1:
                r = bt.run(W1, Rn.loc[sel], cost.loc[sel])
            else:
                r = bt.run(bt.hold_k_days(W1, h), Rcc.loc[sel], cost.loc[sel], roundtrip=False)
            for per, a, b in PER:
                x = r.loc[a:b]
                st = ann_stats(x.net)
                rows.append(dict(universe=un, signal=sig, hold=h, period=per, sharpe=st["sharpe"],
                                 ann_ret=st["ann_ret"], maxdd=st["maxdd"], gross_bps=1e4 * x.gross.mean(),
                                 names=float(M.loc[a:b].sum(1).mean()), invested=float(W1.loc[a:b].sum(1).mean())))
    print(un, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study12_insider.csv", index=False)
pd.set_option("display.width", 250)
print(df.pivot_table(index=["universe", "signal", "hold"], columns="period", values=["sharpe", "gross_bps", "names"]).round(2).to_string())
