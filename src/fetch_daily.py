"""Download daily OHLCV (raw + adjusted close) for the full US common-stock universe.

Universe: NASDAQ Trader symbol directory (all US exchange-listed securities),
filtered to common stocks (no ETFs, warrants, units, rights, preferreds, notes).
A fixed list of ETFs is added for market/sector features.

Output: data/daily/<chunk>.parquet, then data/daily_all.parquet (long format).
Note: the directory only lists currently-listed symbols, so the dataset has
survivorship bias (stocks delisted before 2026-09 are missing).
"""
import os, sys, time
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
START, END = "2019-06-01", "2026-09-26"

ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE", "ARKK", "TLT", "HYG",
        "GLD", "USO", "UUP", "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "UPRO", "TNA", "TZA",
        "^VIX", "^VIX9D", "^VIX3M"]


def universe():
    d = pd.read_csv(os.path.join(DATA, "nasdaqtraded.txt"), sep="|")
    d = d[(d["Nasdaq Traded"] == "Y") & (d["ETF"] == "N") & (d["Test Issue"] == "N")]
    d = d[~d["Security Name"].str.contains(
        "Warrant|Right|Unit|Preferred|Depositary Shares|Notes|Debenture|%", case=False, regex=True, na=False)]
    d = d[d.Symbol.str.fullmatch(r"[A-Z]{1,5}", na=False)]
    return sorted(d.Symbol.unique().tolist())


def fetch(tickers, tries=4):
    for k in range(tries):
        try:
            df = yf.download(tickers, start=START, end=END, auto_adjust=False, actions=False,
                             group_by="column", threads=True, progress=False)
            if df is not None and len(df):
                return df
        except Exception as e:  # network / rate limit
            print("err", e, file=sys.stderr)
        time.sleep(5 * 2 ** k)
    return None


def to_long(df):
    df = df.stack(level=1, future_stack=True).reset_index()
    df.columns = [c if isinstance(c, str) else c for c in df.columns]
    df = df.rename(columns={"Date": "date", "Ticker": "ticker", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    return df.dropna(subset=["close"])


def main():
    out = os.path.join(DATA, "daily")
    os.makedirs(out, exist_ok=True)
    tick = universe() + ETFS
    print(len(tick), "tickers")
    n = 150
    for i in range(0, len(tick), n):
        fn = os.path.join(out, f"c{i:05d}.parquet")
        if os.path.exists(fn):
            continue
        df = fetch(tick[i:i + n])
        if df is None:
            print("failed chunk", i)
            continue
        to_long(df).to_parquet(fn)
        print(i, flush=True)
        time.sleep(1)
    parts = [pd.read_parquet(os.path.join(out, f)) for f in sorted(os.listdir(out))]
    allp = pd.concat(parts, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None)
    allp.to_parquet(os.path.join(DATA, "daily_all.parquet"))
    print(allp.shape, allp.ticker.nunique())


if __name__ == "__main__":
    main()
