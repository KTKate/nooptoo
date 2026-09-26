"""Study 7: market intraday momentum (Gao, Han, Li & Zhou 2018) on ETFs with 1-minute Alpaca bars.

Signals known at 15:30 (or at 15:59 bar start for the variants):
  r_first  = price(10:00) / previous close - 1           (first half hour incl. overnight)
  r_12     = price(15:30) / price(15:00) - 1            (12th half hour)
  r_sofar  = price(15:30) / previous close - 1
Trade: enter at the close of the 15:30 bar (one minute after the signal), exit at the 15:59 bar close.
Variants: long/short by sign, long-only when signal > 0 (cash account).
Costs per side 1.8 bp (base) and 0.5 bp (sensitivity) for SPY/QQQ/IWM/DIA; 3 bp and 1 bp for
the leveraged ETFs (TQQQ, SPXL, SOXL, TNA, UPRO).
Periods: dev 2019-07..2023-12, val 2024-01..2025-06, oos 2025-07..2026-09.
Output: results/study7_etf_lasthalf.csv
"""
import numpy as np
import pandas as pd
from study5_noise_area import daily_frames
from core import ann_stats, RES

rows = []
for t in ["SPY", "QQQ", "IWM", "DIA", "TQQQ", "SPXL", "SOXL", "TNA", "UPRO"]:
    close, opn, vwap = daily_frames(t)
    prevc = close[389].shift(1)
    p = lambda hm: close[int((pd.Timestamp(hm) - pd.Timestamp("09:30")).seconds / 60) - 1]
    r_first = p("10:00") / prevc - 1
    r_12 = p("15:30") / p("15:00") - 1
    r_sofar = p("15:30") / prevc - 1
    entry = close[int((pd.Timestamp("15:31") - pd.Timestamp("09:30")).seconds / 60) - 1]
    last = close[389] / entry - 1
    lev = t not in ("SPY", "QQQ", "IWM", "DIA")
    for cname, c in [("base", 3e-4 if lev else 1.8e-4), ("low", 1e-4 if lev else 0.5e-4)]:
        for sname, s in [("first", r_first), ("r12", r_12), ("sofar", r_sofar), ("first+r12", np.sign(r_first) + np.sign(r_12))]:
            ls = np.sign(s) * last - 2 * c * (s != 0)
            lo = (s > 0) * (last - 2 * c)
            for vname, r in [("long_short", ls), ("long_only", lo), ("always_long", last - 2 * c)]:
                if vname == "always_long" and sname != "first":
                    continue
                for per, a, b in [("dev", "2019-07", "2023-12"), ("val", "2024-01", "2025-06"),
                                  ("oos", "2025-07", "2026-09")]:
                    st = ann_stats(r.loc[a:b].dropna())
                    rows.append(dict(ticker=t, signal=sname, variant=vname, cost=cname, period=per,
                                     sharpe=st["sharpe"], ann_ret=st["ann_ret"], bps_day=1e4 * r.loc[a:b].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study7_etf_lasthalf.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(df[df.cost == "low"].pivot_table(index=["ticker", "signal", "variant"], columns="period", values="sharpe").round(2))
