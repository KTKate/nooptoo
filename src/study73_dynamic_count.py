"""Study 73: a variable number of stocks instead of a fixed top 10 for the overnight blend.

Rules (each name 1/10 of capital unless stated, rest in cash; costs as study 33):
  fixed k = 5, 10, 15, 20 (equal weight over k)
  threshold on the jump model: every name with P(jump) - P(drop) above x (absolute, not a rank), ranked by the blend,
    x = the 50th/75th/90th percentile of the top-10 names' values in 2024H1 (stricter = fewer names, more cash),
    capped at 20, 10% per name; a rule is chosen on 2024-01..2025-06 and checked on 2025-07..2026-09
  threshold on the ensemble raw score (average within-day percentile of the 5 members is relative, so use the pooled
    member's raw prediction), same procedure
  confidence sizing: top 20, weights proportional to the blend rank above the 20th name
Output: results/study73_dynamic_count.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
e5 = pd.read_parquet(f"{RES}/study23_pred.parquet")
ens = e5["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-1]]
pooled = e5["pooled"].unstack().reindex(index=ens.index, columns=cols)
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
jmd = (pj - pdr).where(ok)
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
per = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
rows = []


def rec(name, W):
    r = bt.run(W, R, C)
    for p, a, b in per:
        x = r.loc[a:b]
        st = ann_stats(x.net)
        rows.append(dict(variant=name, period=p, sharpe=st["sharpe"], ann=st["ann_ret"], maxdd=st["maxdd"],
                         names=x.n.mean(), cash_days=(x.n == 0).mean()))


for k in [5, 10, 15, 20]:
    rec(f"fixed top {k}", bt.select_topk(S, S.notna(), k))
top10 = bt.select_topk(S, S.notna(), 10) > 0
for lab, val in [("jump-minus-drop", jmd), ("pooled raw score", pooled.where(ok))]:
    ref = val.where(top10).loc["2024-01":"2024-06"].stack()
    for q in [0.5, 0.75, 0.9]:
        x = float(ref.quantile(q))
        m = val > x
        Sm = S.where(m)
        W = bt.select_topk(Sm, Sm.notna(), 20)
        W = (W > 0).astype(float) / 10                       # 10% per name, up to 20 names (max 200% gross: cap)
        W = W.div(np.maximum(W.sum(1), 1.0), axis=0)          # no leverage: scale down when more than 10 names
        rec(f"threshold {lab} > {x:.4f} (q{q})", W)
r20 = S.rank(axis=1, ascending=False)
w = (21 - r20).clip(lower=0)
w = w.div(w.sum(1), axis=0).fillna(0)
rec("top 20 weighted by rank", w)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study73_dynamic_count.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index="variant", columns="period", values=["sharpe", "names"], sort=False).round(2).to_string())
