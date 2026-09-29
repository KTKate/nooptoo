"""Study 29: shorting the stocks predicted to fall during the next day session (opening-auction short, closing-
auction cover), using the study-3 next-day model (results/study3_pred_day1.parquet, predicted at the previous
close). Checks the borrow constraint: Alpaca's current easy-to-borrow list (data/local/alpaca_assets_active.parquet;
today's list, which flatters the past) and the short-sale restriction (stocks that fell > 10% the prior day).
Result (2022-23 / 2024-01..2025-06 / 2025-07..2026-09, net Sharpe, top 10): all stocks 1.39 / 2.75 / 4.09, easy to
borrow only 0.07 / 0.84 / 1.67, plus no SSR and price > $5 0.06 / 0.53 / 0.96. The edge is in hard-to-borrow names.
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
p = pd.read_parquet(f"{RES}/study3_pred_day1.parquet")["pred"].unstack().reindex(columns=cols)
Rd = P["c"][cols].shift(-1) / P["o"][cols].shift(-1) - 1
cost = exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5
px = traded_close(P)[cols].fillna(P["rawc"][cols])
a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)])
etbm = pd.DataFrame(np.broadcast_to(np.array([t in etb for t in cols]), p.shape), index=p.index, columns=cols)
ssr = (P["c"][cols] / P["c"][cols].shift(1) - 1) < -0.10
per = [("2022-23", "2022", "2023"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
rows = []
for lab, E in [("all", p.notna()), ("etb", p.notna() & etbm), ("etb_nossr_px5", p.notna() & etbm & ~ssr & (px > 5))]:
    for k in [10, 20]:
        W = bt.select_topk(-p, E.reindex_like(p), k)
        r = bt.run(W, Rd.reindex_like(W), cost.reindex_like(W), side=-1)
        x = r.net.loc[r.net.index < days[-2]]
        for name, a_, b in per:
            rows.append(dict(universe=lab, k=k, period=name, sharpe=ann_stats(x.loc[a_:b])["sharpe"],
                             gross_bps=1e4 * r.gross.loc[a_:b].mean(), cost_bps=1e4 * r.cost.loc[a_:b].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study29_day_short.csv", index=False)
print(df.pivot_table(index=["universe", "k"], columns="period", values="sharpe").round(2))
