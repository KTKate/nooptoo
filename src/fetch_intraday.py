"""Download intraday bars from Yahoo for the liquid universe.

Yahoo limits: 60m bars for the last 730 days, 5m bars for the last 60 days.
Universe: ETFs + stocks with 60-day median dollar volume > $20M and price > $3
as of the last date of the daily panel (a current-liquidity filter; noted as a
bias in the report).

Usage: python src/fetch_intraday.py 60m|5m
Output: data/intraday_<iv>.parquet (long: ts, ticker, o,h,l,c,v); timestamps in America/New_York.
"""
import os, sys, time
import pandas as pd
import yfinance as yf
from core import load_panel, ETFS, DATA

IV = sys.argv[1] if len(sys.argv) > 1 else "60m"
PERIOD = {"60m": "730d", "5m": "60d", "1m": "8d"}[IV]


def universe(P, min_dv=2e7):
    dv = P["dv"].iloc[-60:].median()
    px = P["rawc"].iloc[-1]
    names = dv[(dv > min_dv) & (px > 3)].index.tolist()
    return sorted(set(names) | set(e for e in ETFS if not e.startswith("^")))


def main():
    P = load_panel()
    tick = universe(P)
    print(len(tick), "tickers", flush=True)
    out = os.path.join(DATA, f"intra_{IV}")
    os.makedirs(out, exist_ok=True)
    n = 50
    for i in range(0, len(tick), n):
        fn = os.path.join(out, f"c{i:05d}.parquet")
        if os.path.exists(fn):
            continue
        df = None
        for k in range(4):
            try:
                df = yf.download(tick[i:i + n], period=PERIOD, interval=IV, auto_adjust=False, prepost=False,
                                 group_by="column", threads=True, progress=False)
                if df is not None and len(df):
                    break
            except Exception as e:
                print("err", e, flush=True)
            time.sleep(5 * 2 ** k)
        if df is None or not len(df):
            print("failed", i, flush=True)
            continue
        df = df.stack(level=1, future_stack=True).reset_index()
        df = df.rename(columns={df.columns[0]: "ts", "Ticker": "ticker", "Open": "o", "High": "h", "Low": "l",
                                "Close": "c", "Volume": "v"})
        df = df.dropna(subset=["c"])[["ts", "ticker", "o", "h", "l", "c", "v"]]
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
        df.to_parquet(fn)
        print(i, len(df), flush=True)
        time.sleep(1)
    parts = [pd.read_parquet(os.path.join(out, f)) for f in sorted(os.listdir(out))]
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(os.path.join(DATA, f"intraday_{IV}.parquet"))
    print(allp.shape, allp.ticker.nunique())


if __name__ == "__main__":
    main()
