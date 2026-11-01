"""Data for studies 201-202 (same-time-of-day repeat, Heston-Korajczyk-Sadka 2010).

Universe: every common stock that ranked in the top 500 by 20-day median dollar volume (price > $5) on any day
since 2019-12, from the Yahoo daily store plus the delisted-stock file (reduces survivorship bias). ETFs and funds
excluded by name.
Fetches (Alpaca SIP, split-adjusted, read-only data API):
  data/local/m5win/   5-minute bars 09:30-10:00 and 15:30-16:00, 2019-12 .. last closed day
  data/local/d1/      daily bars (close = official closing cross; open = first trade, not the opening cross)
    python src/study201_fetch.py
"""
import os
import re
import pandas as pd
import alpaca_data as A
import store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def universe():
    fn = os.path.join(A.LOCAL, "hks_universe.csv")
    if os.path.exists(fn):
        return pd.read_csv(fn).ticker.tolist()
    d = store.read("daily", start="2019-11")[["date", "ticker", "c", "v"]]
    dl = pd.read_parquet(os.path.join(A.LOCAL, "delisted_daily.parquet"))[["date", "ticker", "c", "v"]]
    dl = dl[pd.to_datetime(dl.date) >= "2019-11-01"]
    d = pd.concat([d, dl[~dl.ticker.isin(d.ticker.unique())]], ignore_index=True)
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["ticker", "date"])
    d["med"] = (d.c * d.v).groupby(d.ticker).transform(lambda x: x.rolling(20, min_periods=10).median())
    d = d[d.date >= "2019-12-01"]
    d["rk"] = d.groupby("date").med.rank(ascending=False)
    t = set(d[(d.rk <= 500) & (d.c > 5)].ticker)
    s = pd.read_parquet(os.path.join(ROOT, "data", "store", "symbols.parquet"))
    fund = set(s[s.name.fillna("").str.contains(r"\bETF\b|\bFund\b|iShares|SPDR|ProShares|Direxion|Invesco QQQ|"
                                                   r"Vanguard|Trust Units|ETN\b", flags=re.I)].ticker)
    t = sorted(x for x in t - fund - set(A.ETF1M) if not x.startswith("^"))
    pd.DataFrame({"ticker": t}).to_csv(fn, index=False)
    return t


if __name__ == "__main__":
    tick = universe()
    print("universe", len(tick), flush=True)
    for m in A.months("2019-12", A.last_closed_month_end()):
        n = A.fetch_month("d1", tick, m, "1Day", chunk=100)
        print("d1", m, n, flush=True)
    A.fetch_windows(ds="m5win", start="2019-12", tickers=tick, chunk=400, workers=8)
    print("fetch done", flush=True)
