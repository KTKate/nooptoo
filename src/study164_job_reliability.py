"""Study 164: reliability of the paper-trading jobs since 2026-09-28.

Question: how often did the daily entry (15:05 ET routine) or exit (09:17 ET routine) fail to trade, and why?
Data: logs/paper/job_entry_<day>.log, job_exit_<day>.log, entry_<day>.json.
Design: for each business day, whether the logs and the entry record exist, the entry finish time, and the count of
error lines (traceback, error, failed) with their text.
Output: results/study164_job_reliability.csv
"""
import os
import re
import pandas as pd
from core import RES

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "paper")
rows = []
for d in pd.bdate_range("2026-09-28", pd.Timestamp.today().normalize() - pd.Timedelta(days=1)):
    d = str(d.date())
    ent, ex, rec = (os.path.join(LOG, f) for f in (f"job_entry_{d}.log", f"job_exit_{d}.log", f"entry_{d}.json"))
    t = open(ent).read() if os.path.exists(ent) else ""
    x = open(ex).read() if os.path.exists(ex) else ""
    errs = [l.strip()[:120] for l in (t + "\n" + x).splitlines() if re.search(r"traceback|error|failed", l, re.I)]
    fin = re.search(r"entry finished at (\S+)", t)
    rows.append(dict(day=d, entry_log=bool(t), entry_record=os.path.exists(rec), entry_finished=fin.group(1) if fin
                     else None, exit_log=bool(x), error_lines=len(errs), errors=" | ".join(sorted(set(errs)))))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study164_job_reliability.csv", index=False)
print(df.drop(columns="errors").to_string(index=False))
print(df.errors[df.error_lines > 0].to_string())
