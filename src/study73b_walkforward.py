"""Study 73b: walk-forward choice of the stock-count rule (owner question: why not use the most recent data?).
Each quarter from 2024Q3, choose the rule with the best Sharpe on ALL earlier days since 2024-01 (fixed top
5/10/15/20, or a jump-minus-drop threshold at the 50/75/90th percentile of earlier top-10 values), then trade it in
that quarter. Compared with always using the fixed top 10.
Output: results/study73b_walkforward.csv
"""
import numpy as np
import pandas as pd
exec(open("study73_dynamic_count.py").read().split("per = [")[0])
from core import ann_stats


def rule_weights(name, cut=None):
    if name.startswith("top"):
        return bt.select_topk(S, S.notna(), int(name[3:]))
    Sm = S.where(jmd > cut)
    W = (bt.select_topk(Sm, Sm.notna(), 20) > 0).astype(float) / 10
    return W.div(np.maximum(W.sum(1), 1.0), axis=0)


nets = {}
top10 = bt.select_topk(S, S.notna(), 10) > 0
chosen, out = [], []
for q in pd.period_range("2024Q3", "2026Q3", freq="Q"):
    hist = slice("2024-01-01", str((q.start_time - pd.Timedelta(days=1)).date()))
    cands = {f"top{k}": None for k in [5, 10, 15, 20]}
    ref = jmd.where(top10).loc[hist].stack()
    for p in [0.5, 0.75, 0.9]:
        cands[f"jmd>q{p}"] = float(ref.quantile(p))
    best, bs = None, -9
    for name, cut in cands.items():
        key = (name, cut)
        if key not in nets:
            nets[key] = bt.run(rule_weights(name, cut), R, C).net
        s = ann_stats(nets[key].loc[hist])["sharpe"]
        if s > bs:
            best, bs = key, s
    seg = nets[best].loc[str(q.start_time.date()):str(q.end_time.date())]
    out.append(seg)
    chosen.append(dict(quarter=str(q), rule=best[0], cut=best[1], insample_sharpe=bs))
wf = pd.concat(out)
fixed = bt.run(bt.select_topk(S, S.notna(), 10), R, C).net.loc[wf.index]
res = pd.DataFrame(chosen)
res["wf_sharpe_total"] = ann_stats(wf)["sharpe"]
res["fixed10_sharpe_total"] = ann_stats(fixed)["sharpe"]
res.to_csv(f"{RES}/study73b_walkforward.csv", index=False)
print(res.round(3).to_string())
