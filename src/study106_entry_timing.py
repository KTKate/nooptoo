"""Study 106: can the live entry job send market-on-close orders before the exchange cutoffs?

Question: a live account would send the blend's buys as market-on-close orders. The free data plan forces the job to
start at 15:46 ET (SIP bars up to 15:30 are then 15 minutes old). NYSE accepts market-on-close orders until 15:50 and
Nasdaq until 15:55; a broker may close its own window a minute or two earlier. How long does scoring take?
Data: logs/paper/job_entry_<day>.log ("entry at" start time, order "created_at" times, "entry finished" time).
Design: in market-order mode (2026-09-30 onward) the job waits until 15:55 before buying, so only the 2026-09-29 run
(market-on-close mode, ensemble only) shows the time from start to the first order. Later runs give the start time
and the 15:55 order time only.
Output: results/study106_entry_timing.csv
"""
import glob
import os
import re
import pandas as pd
from core import RES

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "paper")
rows = []
for f in sorted(glob.glob(os.path.join(LOG, "job_entry_2026-*.log"))):
    t = open(f).read()
    day = re.search(r"job_entry_(\d{4}-\d\d-\d\d)", f).group(1)
    start = re.search(r"entry at (\d\d:\d\d:\d\d)", t)
    done = re.search(r"entry finished at (\d\d:\d\d:\d\d)", t)
    buys = [m for m in re.finditer(r'^(?!cover)(\S+) 200 \{.*?"created_at":"([^"]+)"', t, re.M)]
    first = pd.Timestamp(buys[0].group(2)).tz_convert("America/New_York").strftime("%H:%M:%S") if buys else None
    rows.append(dict(day=day, start=start.group(1) if start else None, first_buy_order=first,
                     finished=done.group(1) if done else None, fallback="falling back" in t))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study106_entry_timing.csv", index=False)
print(df.to_string(index=False))
