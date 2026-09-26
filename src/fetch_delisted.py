"""Daily Alpaca bars for inactive (delisted / renamed) US-exchange common stocks, 2019-06 .. 2026-09.
Used only to measure survivorship bias of the Yahoo daily store (which holds currently listed names).
Output (git-ignored): data/local/delisted_daily.parquet  (adjustment=all: split + dividend)"""
import os
import pandas as pd
import alpaca_data as A

A.RL = A.RateLimiter(40)
a = pd.read_parquet(os.path.join(A.LOCAL, "alpaca_assets_inactive.parquet"))
a = a[a.exchange.isin(["NYSE", "NASDAQ", "AMEX"]) & a.symbol.str.fullmatch(r"[A-Z]{1,5}")]
a = a[~a.name.str.contains(r"Unit|Right|Warrant|Preferred|Perpetual|Depositary|Notes|ETF|Fund|Trust|Shares|%",
                           case=False, regex=True, na=False)]
syms = sorted(a.symbol)
print("symbols", len(syms), flush=True)
parts = []
for i in range(0, len(syms), 100):
    d = A.bars(syms[i:i + 100], "1Day", "2019-06-01T00:00:00Z", "2026-09-26T00:00:00Z", adjustment="all")
    parts.append(d)
    print(i, len(d), flush=True)
d = pd.concat(parts, ignore_index=True)
d["date"] = d.ts.dt.normalize()
d.drop(columns="ts").to_parquet(os.path.join(A.LOCAL, "delisted_daily.parquet"), index=False)
print(d.ticker.nunique(), len(d))
