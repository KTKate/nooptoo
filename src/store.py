"""Persistent, append-only market data store.

Layout (all under data/store/, committed to git so a fresh session never re-downloads):
  daily/YYYY-MM.parquet     one file per calendar month, long format
                            date, ticker, o, h, l, c (split-adjusted as fetched), v, q
  intra60/YYYY-MM.parquet   ts, ticker, o, h, l, c, v (60-minute bars, America/New_York)
  earnings/YYYY-MM.parquet  Nasdaq earnings calendar rows
  symbols.parquet           every symbol ever seen in the NASDAQ Trader directory, with
                            first_seen / last_seen (keeps delisted names -> less survivorship bias)
  meta.json                 last update dates per dataset

Dividend adjustment without rewriting history: q_t = f_t / f_{t-1} where
f = adj_close / close. A later dividend multiplies every earlier f by the same
constant, so q for stored rows never changes. The adjustment factor is rebuilt as
f_t = prod_{s>t} 1/q_s (so the latest bar has f = 1).

Splits do change Yahoo's split-adjusted history; the updater detects them by
comparing overlapping bars and re-downloads that ticker's full history only.

Closed months are immutable, so git history grows by roughly one small file per
dataset per month.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.environ.get("NOOP_STORE", os.path.join(ROOT, "data", "store"))
PRICE = ["o", "h", "l", "c"]


def _dir(ds):
    d = os.path.join(STORE, ds)
    os.makedirs(d, exist_ok=True)
    return d


def meta():
    fn = os.path.join(STORE, "meta.json")
    return json.load(open(fn)) if os.path.exists(fn) else {}


def set_meta(**kw):
    m = meta()
    m.update(kw)
    os.makedirs(STORE, exist_ok=True)
    json.dump(m, open(os.path.join(STORE, "meta.json"), "w"), indent=1, sort_keys=True)


def _compact(df):
    for k in df.columns:
        if k in ("o", "h", "l", "c", "q") or k in ("surprise", "eps", "epsForecast"):
            df[k] = df[k].astype("float32")
    if "v" in df:
        df["v"] = df["v"].astype("float64")
    return df


def write_month(ds, df, key="date"):
    """Merge df into monthly partitions of dataset ds. New rows win on (key, ticker) conflicts."""
    tcol = "symbol" if ds == "earnings" else "ticker"
    per = pd.to_datetime(df[key]).dt.strftime("%Y-%m")
    for m, part in df.groupby(per):
        fn = os.path.join(_dir(ds), f"{m}.parquet")
        if os.path.exists(fn):
            old = pd.read_parquet(fn)
            part = pd.concat([old, part], ignore_index=True)
            part = part.drop_duplicates([key, tcol], keep="last")
        part = _compact(part.sort_values([key, tcol]).reset_index(drop=True))
        part.to_parquet(fn, compression="zstd", index=False)


def read(ds, start=None, end=None, tickers=None):
    d = _dir(ds)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if start:
        files = [f for f in files if f[:7] >= str(start)[:7]]
    if end:
        files = [f for f in files if f[:7] <= str(end)[:7]]
    filt = [("ticker", "in", list(tickers))] if tickers is not None else None
    parts = [pd.read_parquet(os.path.join(d, f), filters=filt) for f in files]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def replace_ticker(ds, ticker, df, key="date"):
    """Drop every stored row of ticker (all months) and write df in its place (used after splits)."""
    d = _dir(ds)
    for f in sorted(os.listdir(d)):
        fn = os.path.join(d, f)
        old = pd.read_parquet(fn)
        if (old.ticker == ticker).any():
            old[old.ticker != ticker].to_parquet(fn, compression="zstd", index=False)
    write_month(ds, df, key)


def adj_factor(df):
    """Return dividend adjustment factor f per row (latest row per ticker = 1)."""
    df = df.sort_values(["ticker", "date"])
    q = df["q"].astype("float64")
    lq = np.log(q.where((q > 0.5) & (q < 1.5), 1.0).fillna(1.0))
    # f_t = prod_{s>t} 1/q_s  -> log f_t = -(sum_{s>=t} log q_s - log q_t)
    rev = lq[::-1].groupby(df["ticker"][::-1]).cumsum()[::-1]
    return np.exp(-(rev - lq)).reindex(df.index)
