"""Study 0: market-level calendar / session effects on index and leveraged ETFs (daily bars).

Tests: overnight (close->open) vs intraday (open->close) session returns,
day-of-week, turn-of-month (last trading day + first 3), and a simple
overnight-only index hold. Costs: 1 bp half-spread + 1 bp slippage per side
for SPY/QQQ/IWM (auction orders), 3 bp for leveraged ETFs.
Output: results/study0_calendar.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, split_stats, RES

P = load_panel()
tick = ["SPY", "QQQ", "IWM", "TQQQ", "SPXL", "SOXL", "TNA"]
o, c = P["o"][tick], P["c"][tick]
days = c.index
night = o / c.shift(1) - 1
intra = c / o - 1
cc = c / c.shift(1) - 1
cost = pd.Series({t: (2.0 if t in ["SPY", "QQQ", "IWM"] else 4.0) / 1e4 for t in tick})
m = pd.Series(days.month, index=days)
last_day = m != m.shift(-1)
first_n = pd.Series(0, index=days)
k = 0
for i, d in enumerate(days):
    k = 1 if i == 0 or m.iloc[i] != m.iloc[i - 1] else k + 1
    first_n.iloc[i] = k
tom = last_day | (first_n <= 3)

rows = []


def add(name, t, r):
    st = split_stats(r)
    for per in ["dev", "val", "oos"]:
        rows.append(dict(strat=name, ticker=t, period=per, sharpe=st.loc[per, "sharpe"],
                         ann_ret=st.loc[per, "ann_ret"], maxdd=st.loc[per, "maxdd"], n=st.loc[per, "n"]))


for t in tick:
    add("buy_hold", t, cc[t])
    add("overnight_only_net", t, night[t] - 2 * cost[t])
    add("intraday_only_net", t, intra[t] - 2 * cost[t])
    add("overnight_only_gross", t, night[t])
    add("intraday_only_gross", t, intra[t])
    # turn of month: hold close of day before window .. close of last window day
    r_tom = cc[t].where(tom, 0.0)
    entries = (tom & ~tom.shift(1, fill_value=False)).astype(float)
    add("turn_of_month_net", t, r_tom - entries * 2 * cost[t])
    for dw, nm in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        sel = pd.Series(days.dayofweek == dw, index=days)
        add(f"cc_only_{nm}_gross", t, cc[t].where(sel, 0.0))

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study0_calendar.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(df.pivot_table(index=["ticker", "strat"], columns="period", values=["sharpe", "ann_ret"]).round(2))
