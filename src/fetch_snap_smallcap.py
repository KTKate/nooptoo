"""5-minute bars 15:30-16:00 for small-cap stocks missing from the m5snap universe, 2024-01 .. 2026-09.

m5snap (alpaca_data.fetch_windows) covers stocks whose 20-day median dollar volume exceeded $5M on at least
one day in 2023-12 .. 2026-09. For the small-cap tier ($1-5M ADV) that is a look-ahead selection: it keeps
only small caps that later became liquid. This adds every stock that was in the small-cap tier (traded price
> $1, ADV $1-5M) on any day since 2024-01 and is not in m5snap.
Output: data/local/m5snapx/<month>.parquet, done_m5snapx.parquet
"""
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, traded_close

P = load_panel()
cols = stock_cols(P)
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).loc["2024-01-02":]
px = traded_close(P)[cols].shift(1).loc["2024-01-02":]
E = (px > 1) & (adv > 1e6) & (adv <= 5e6)
have = set(A.read("m5snap", columns=["ticker"]).ticker.unique())
ever = E.any()
miss = sorted(t for t in ever.index[ever] if t not in have)
print("small-cap tickers missing from m5snap:", len(miss), flush=True)
A.fetch_windows("m5snapx", start="2024-01", end=P["c"].index[-1].strftime("%Y-%m"), windows=(("15:30", "16:00"),),
                tickers=miss, chunk=400, workers=8)
