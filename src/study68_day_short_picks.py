"""Study 68: shorting the overnight blend's picks during the next day session (short at the opening auction, cover
at the closing auction), i.e. reversing the position at the open instead of only closing it.

Study 53: the blend's top 10 rise 47-51 bp over the universe overnight and then fall 42-57 bp during the next day
session (t -2.5 to -4.1). Shorting needs a borrow: Alpaca's current easy-to-borrow list
(data/local/alpaca_assets_active.parquet; today's list, which flatters the past) and the short-sale restriction
(a stock that fell 10% or more the previous day; approximated by the night-1 move, since the open is known then).
Costs: auction cost + 2.5 bp per side, plus a borrow fee of 0 (easy to borrow) or 30%/yr (others, about 12 bp a day).
Variants: all picks / easy-to-borrow only / easy-to-borrow without a >10% overnight gap down / only picks that rose
overnight (open above the close).
Output: results/study68_day_short_picks.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-2]]
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
W = bt.select_topk(S, S.notna(), 10)
o, c = P["o"][cols], P["c"][cols]
night = (o.shift(-1) / c - 1).reindex_like(W)
day = (c.shift(-1) / o.shift(-1) - 1).reindex_like(W)
cost = (exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5).reindex_like(W)
a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)])
etbm = pd.DataFrame(np.broadcast_to(np.array([t in etb for t in cols]), W.shape), index=W.index, columns=cols)
fee = pd.DataFrame(np.where(etbm, 0.0, 0.30 / 252 * 1e4), index=W.index, columns=cols)   # bp per day
picked = W > 0
variants = {"all_picks": picked, "etb": picked & etbm, "etb_no_gapdown10": picked & etbm & (night > -0.10),
            "etb_rose_overnight": picked & etbm & (night > 0)}
rows = []
for k, m in variants.items():
    w = m.astype(float).div(10)                               # each short is 1/10 of capital; fewer names -> cash
    r = bt.run(w, day, cost + fee, side=-1)
    x = r.loc[r.index < days[-2]]
    for p, a_, b in [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]:
        y = x.loc[a_:b]
        rows.append(dict(variant=k, period=p, names=y.n.mean(), gross_bp=1e4 * y.gross.mean(), cost_bp=1e4 * y.cost.mean(),
                         net_bp=1e4 * y.net.mean(), sharpe=ann_stats(y.net)["sharpe"]))
    if k == "etb":
        both = (bt.run(W, night, cost).net + x.net).dropna()
        for p, a_, b in [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]:
            rows.append(dict(variant="blend_night + etb_day_short", period=p, net_bp=1e4 * both.loc[a_:b].mean(),
                             sharpe=ann_stats(both.loc[a_:b])["sharpe"]))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study68_day_short_picks.csv", index=False)
print(df.round(2).to_string())
