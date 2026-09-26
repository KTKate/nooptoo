"""Study 3 evaluation: turn walk-forward LightGBM predictions into long-only top-k portfolios.

Predictions are made at the close of day t (features use c_t). Execution variants:
  night : buy in the closing auction of t, sell in the opening auction of t+1
  day1  : buy at the open of t+1, sell at the close of t+1
  cc1   : buy close t, sell close t+1
  cc5   : buy close t, hold 5 days (5 overlapping cohorts, 1/5 of capital each)
Costs per side (core.exec_cost_bps): closing-auction side = fees + 1 bp + 10% of the half-spread;
open side = quoted 09:35 half-spread + 2 bp + fees, because the Yahoo open is the first trade rather than
the opening cross (variant 'auction' prices the open side like the close, an optimistic bound).
Output: results/study3_eval.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, split_stats, RES
import bt

P = load_panel()
cols = stock_cols(P)
o, c = P["o"][cols], P["c"][cols]
c_auc = exec_cost_bps(P, "auction")[cols]
c_open = exec_cost_bps(P, "open")[cols]
R = {"night": o.shift(-1) / c - 1, "day1": c.shift(-1) / o.shift(-1) - 1, "cc1": c.shift(-1) / c - 1,
     "cc5": c.shift(-1) / c - 1}
rows = []
for tgt in ["night", "day1", "cc1", "cc5"]:
    try:
        p = pd.read_parquet(f"{RES}/study3_pred_{tgt}.parquet")["pred"]
    except FileNotFoundError:
        continue
    S = p.unstack().reindex(index=c.index, columns=cols)
    E = S.notna()
    for k in [5, 10, 20]:
        W = bt.select_topk(S, E, k)
        variants = {}
        if tgt == "cc5":
            variants["model"] = bt.run(bt.hold_k_days(W, 5), R[tgt], c_auc, roundtrip=False)
        elif tgt == "cc1":
            variants["model"] = bt.run(W, R[tgt], c_auc, roundtrip=False)
        elif tgt == "night":
            variants["model"] = bt.run(W, R[tgt], (c_auc + c_open.shift(-1)) / 2, roundtrip=True)
            variants["auction"] = bt.run(W, R[tgt], c_auc, roundtrip=True)
        else:
            variants["model"] = bt.run(W, R[tgt], (c_auc + c_open.shift(-1)) / 2, roundtrip=True)
            variants["auction"] = bt.run(W, R[tgt], c_auc, roundtrip=True)
        for vn, r in variants.items():
            r = r.loc["2022":]                      # predictions start 2022Q1 ("dev" = 2022-2023)
            st = split_stats(r["net"])
            stg = split_stats(r["gross"])
            for per in ["dev", "val", "oos"]:
                if per == "dev":
                    sub = r.loc["2022":"2023"]      # predictions start 2022Q1
                else:
                    sub = r.loc[{"val": slice("2024", "2025-06"), "oos": slice("2025-07", "2026-09")}[per]]
                rows.append(dict(target=tgt, k=k, cost=vn, period=per, net_sharpe=st.loc[per, "sharpe"],
                                 gross_sharpe=stg.loc[per, "sharpe"], net_annret=st.loc[per, "ann_ret"],
                                 gross_bps=1e4 * sub.gross.mean(), cost_bps=1e4 * sub.cost.mean(),
                                 maxdd=st.loc[per, "maxdd"]))
    # decile spread (gross) as signal quality
    pct = S.rank(axis=1, pct=True)
    ls = R[tgt].where(pct > 0.9).mean(1) - R[tgt].where(pct < 0.1).mean(1)
    st = split_stats(ls)
    for per in ["val", "oos"]:
        rows.append(dict(target=tgt, k="decileLS", cost="gross", period=per, gross_sharpe=st.loc[per, "sharpe"],
                         gross_bps=1e4 * ls.loc[{"val": slice("2024", "2025-06"),
                                                 "oos": slice("2025-07", "2026-09")}[per]].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study3_eval.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.round(3).to_string())
