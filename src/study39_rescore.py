"""Rerun the study 39/40 backtests from saved predictions (results/study{39,40}_pred.parquet), e.g. after a change
to horizon_lib (2026-10-03: universe extended before 2023-12, the first runs had no 2022-23 trades).
    python src/study39_rescore.py 5 | 20
Output: results/study{39,40}_horizon_ml.csv (overwritten)
"""
import sys
import pandas as pd
import horizon_lib as H
from core import RES

h = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SID = 39 if h <= 5 else 40
pred = pd.read_parquet(f"{RES}/study{SID}_pred.parquet")
rows = []
offsets = range(0, h, max(1, h // 5)) if h > 5 else [0]
for name in pred.columns:
    S = pred[name].unstack().reindex(columns=H.cols)
    for off in offsets:
        for r in H.summarize(name, H.backtest(S, h, k=20, offset=off), h):
            rows.append(dict(r, offset=off))
g = pd.DataFrame(rows).groupby(["variant", "period"]).mean(numeric_only=True).drop(columns="offset")
g.to_csv(f"{RES}/study{SID}_horizon_ml.csv")
pd.set_option("display.width", 220)
print(g.round(3).to_string())
