"""Survivorship check: how much do the study-1 leads change when delisted stocks are added?

The Yahoo daily store only has stocks listed in 2026-09. Alpaca still serves daily bars for 628 inactive
(delisted or renamed) NYSE/NASDAQ/AMEX common stocks (data/local/delisted_daily.parquet, fetch_delisted.py).
That is far fewer than all delistings since 2019, so the difference below is a lower bound on the bias.
Gross returns only (the question is the signal, not costs). Alpaca's daily open is also a first-trade open.
Output: results/survivorship.csv
"""
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, split_stats, RES
import bt
import os

P = load_panel()
cols = stock_cols(P)
d = pd.read_parquet(os.path.join(A.LOCAL, "delisted_daily.parquet"))
d = d[~d.ticker.isin(P["c"].columns)]
days = P["c"].index
X = {k: d.pivot_table(index="date", columns="ticker", values=k, aggfunc="last").reindex(days) for k in ["o", "h", "l", "c", "v"]}
bad = (X["l"] > X["h"] * 1.0001) | (X["o"] > X["h"] * 1.02) | (X["o"] < X["l"] * 0.98)
for k in ["o", "h", "l", "c"]:
    X[k] = X[k].mask(bad)
print("delisted tickers added", X["c"].shape[1])


def rules(o, c, v):
    dv = c * v
    adv = dv.rolling(20, min_periods=10).median().shift(1)
    px = c.shift(1)
    tiers = {"L": (px > 5) & (adv > 5e7), "M": (px > 5) & (adv > 5e6) & (adv <= 5e7),
             "S": (px > 2) & (adv > 1e6) & (adv <= 5e6)}
    night = o / c.shift(1) - 1
    oc = c / o - 1
    gap = night
    vol20 = np.log(c / c.shift(1)).rolling(20, min_periods=15).std()
    co_next = o.shift(-1) / c - 1
    S = {"overnight_mom20": (np.log1p(night).rolling(20, min_periods=15).mean(), co_next),
         "intraday_loser_overnight": (-oc, co_next),
         "gap_down_fade_volnorm": (-gap / vol20.shift(1), oc),
         "gap_up_go": (gap, oc)}
    return tiers, S


out = []
for name, (o, c, v) in {"survivors": (P["o"][cols], P["c"][cols], P["v"][cols]),
                        "with_delisted": (pd.concat([P["o"][cols], X["o"]], axis=1),
                                          pd.concat([P["c"][cols], X["c"]], axis=1),
                                          pd.concat([P["v"][cols], X["v"]], axis=1))}.items():
    tiers, S = rules(o, c, v)
    zero = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for tn, E in tiers.items():
        for sn, (sig, R) in S.items():
            W = bt.select_topk(sig, E & sig.notna(), 10)
            r = bt.run(W, R, zero)["gross"]
            share = (W[X["c"].columns.intersection(W.columns)] > 0).sum(1).sum() / max((W > 0).sum(1).sum(), 1) \
                if name == "with_delisted" else 0.0
            st = split_stats(r)
            for per in ["dev", "val", "oos"]:
                sub = r.loc[{"dev": slice("2020", "2023"), "val": slice("2024", "2025-06"), "oos": slice("2025-07", "2026-09")}[per]]
                out.append(dict(universe=name, tier=tn, rule=sn, period=per, gross_bps=1e4 * sub.mean(),
                                gross_sharpe=st.loc[per, "sharpe"], delisted_share=share))
    print(name, flush=True)
df = pd.DataFrame(out)
df.to_csv(f"{RES}/survivorship.csv", index=False)
pd.set_option("display.width", 250)
print(df.pivot_table(index=["tier", "rule", "period"], columns="universe", values="gross_bps").round(1))
print(df[df.universe == "with_delisted"].groupby(["tier", "rule"]).delisted_share.first().round(3))
