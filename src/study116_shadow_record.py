"""Study 116: live record of the shadow candidates (picks logged, no orders) since 2026-09-29.

Question: after the first two weeks, does any shadow candidate stand out from zero?
Data: logs/paper/shadow_scores.csv (mean net return per night for adr_loser and smallcap_nonews, close to next open at
official prints minus the auction cost model; day_short: open to close, short side) written by src/shadow.py score,
and logs/paper/shadow_meanrev.csv (SPY/QQQ mean reversion trades) when present.
Design: mean, t-statistic and nights per candidate. With fewer than 20 nights this is a progress check, not a test.
Output: results/study116_shadow_record.csv
"""
import os
import numpy as np
import pandas as pd
from core import RES

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "paper")
s = pd.read_csv(os.path.join(LOG, "shadow_scores.csv"))
rows = []
for c in [k for k in s.columns if k != "day"]:
    x = s[c].dropna()
    rows.append(dict(candidate=c, nights=len(x), mean_bp=1e4 * x.mean(), t=x.mean() / (x.std() / np.sqrt(len(x)))
                     if len(x) > 2 else np.nan, positive=(x > 0).mean(), first=s.day[x.index[0]], last=s.day[x.index[-1]]))
mr = os.path.join(LOG, "shadow_meanrev.csv")
if os.path.exists(mr):
    m = pd.read_csv(mr)
    rows.append(dict(candidate="spy_meanrev (trades)", nights=len(m), mean_bp=1e4 * m.ret.mean(), t=np.nan,
                     positive=(m.ret > 0).mean(), first=str(m.entry.min()), last=str(m.exit.max())))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study116_shadow_record.csv", index=False)
print(df.round(3).to_string(index=False))
