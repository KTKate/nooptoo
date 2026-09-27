"""Unadjusted daily bars from Alpaca (adjustment=raw) for the stock universe, 2023-12 .. today.

Yahoo's "raw" close in data/store is split-adjusted as fetched, so a stock that later did a 1:50 reverse split
shows 50x its traded price for all earlier days. A price filter (price > $2 or > $5) built on it uses
information that did not exist on the day. These bars give the price as it traded.
Output: data/local/d1raw.parquet (date, ticker, c_raw, v_raw)
"""
import os
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols

P = load_panel()
cols = stock_cols(P)
live = [t for t in cols if P["dv"][t].loc["2023-12":].max() > 3e5]
fn = os.path.join(A.LOCAL, "d1raw.parquet")
end = P["c"].index[-1].strftime("%Y-%m-%dT23:00:00Z")      # SIP data must be older than 15 minutes
parts = []
for i in range(0, len(live), 200):
    b = A.bars(live[i:i + 200], "1Day", "2023-12-01", end, adjustment="raw")
    if len(b):
        parts.append(pd.DataFrame({"date": b.ts.dt.normalize(), "ticker": b.ticker, "c_raw": b.c, "v_raw": b.v}))
    print(i, len(live), flush=True)
d = pd.concat(parts, ignore_index=True)
d.to_parquet(fn, index=False)
print("rows", len(d), "tickers", d.ticker.nunique())
