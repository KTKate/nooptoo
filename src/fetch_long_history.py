"""One-off backfill of daily bars 2010-01 .. 2019-06 into data/local/daily_long/<YYYY-MM>.parquet.

    python src/fetch_long_history.py            # download (resumable) then build monthly files
    python src/fetch_long_history.py verify     # compare 2019-06 overlap with data/store

Reuses update_data._yf_daily / _to_store, so the schema is identical to data/store/daily:
date, ticker, o, h, l, c (split-adjusted as Yahoo returns them), v, q (dividend factor step
from Adj Close / Close; the first row of each ticker has q = 1).

Universe: data/store/symbols.parquet + core.ETFS. These are symbols listed in 2026, so the
history is survivorship-biased: companies delisted before 2026 are missing.

Progress: each downloaded chunk is saved as _chunks/cNNNNN.parquet and recorded in
_progress.json; a rerun skips finished chunks.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import core
import store
from update_data import _yf_daily

OUT = os.path.join(store.ROOT, "data", "local", "daily_long")
CHUNKS = os.path.join(OUT, "_chunks")
PROG = os.path.join(OUT, "_progress.json")
START, END = "2010-01-01", "2019-07-01"   # yfinance end is exclusive -> includes 2019-06-28
CHUNK = 140
PAUSE = 8


def tickers():
    s = pd.read_parquet(os.path.join(store.STORE, "symbols.parquet"))
    return sorted(set(s.ticker) | set(core.ETFS))


def load_prog():
    return json.load(open(PROG)) if os.path.exists(PROG) else {"done": {}, "t0": time.time(), "elapsed": 0.0}


def save_prog(p):
    tmp = PROG + ".tmp"
    json.dump(p, open(tmp, "w"), indent=1)
    os.replace(tmp, PROG)


def download():
    os.makedirs(CHUNKS, exist_ok=True)
    p = load_prog()
    tick = tickers()
    print(len(tick), "tickers", flush=True)
    for i in range(0, len(tick), CHUNK):
        key = f"c{i:05d}"
        if key in p["done"]:
            continue
        t0 = time.time()
        part = tick[i:i + CHUNK]
        d = pd.DataFrame()
        for k in range(3):               # outer retry on top of _yf_daily's own 4 tries with backoff
            d = _yf_daily(part, START, END)
            if len(d):
                break
            time.sleep(60 * (k + 1))
        if not len(d):
            print(key, "failed, will retry on next run", flush=True)
            time.sleep(PAUSE * 10)
            continue
        if len(d):
            d = store._compact(d.drop(columns="_first").reset_index(drop=True))
            d.to_parquet(os.path.join(CHUNKS, key + ".parquet"), compression="zstd", index=False)
        p["done"][key] = {"n_req": len(part), "n_got": int(d.ticker.nunique()) if len(d) else 0, "rows": len(d)}
        p["elapsed"] += time.time() - t0 + PAUSE
        save_prog(p)
        print(key, p["done"][key], f"{time.time() - t0:.0f}s", flush=True)
        time.sleep(PAUSE)


def build():
    files = sorted(os.path.join(CHUNKS, f) for f in os.listdir(CHUNKS) if f.endswith(".parquet"))
    ds = pads.dataset(files, format="parquet")
    for y in range(2010, 2020):
        lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        t = ds.to_table(filter=(pads.field("date") >= lo) & (pads.field("date") < hi)).to_pandas()
        if not len(t):
            continue
        t["date"] = t["date"].astype("datetime64[ms]")
        for m, part in t.groupby(t.date.dt.strftime("%Y-%m")):
            part = store._compact(part.sort_values(["date", "ticker"]).reset_index(drop=True))
            part[["date", "ticker", "o", "h", "l", "c", "v", "q"]].to_parquet(
                os.path.join(OUT, f"{m}.parquet"), compression="zstd", index=False)
        print("built", y, len(t), flush=True)


def verify(n=50, seed=0):
    old = store.read("daily", start="2019-06", end="2019-06")
    new = pd.read_parquet(os.path.join(OUT, "2019-06.parquet"))
    dv = (old.c * old.v).groupby(old.ticker).median()
    liquid = dv[(dv > 2e7) & ~dv.index.isin(core.ETFS)].index
    pick = pd.Series(sorted(liquid)).sample(n, random_state=seed).tolist()
    m = old[old.ticker.isin(pick)].merge(new, on=["date", "ticker"], suffixes=("_s", "_n"))
    m["ok"] = (m.c_n / m.c_s - 1).abs() < 0.005
    per = m.groupby("ticker").ok.all()
    print(f"rows compared {len(m)}, row match {m.ok.mean():.4f}, tickers fully matching {per.sum()}/{n}"
          f" (with overlap: {len(per)})")
    bad = per[~per].index.tolist()
    if bad:
        x = m[m.ticker.isin(bad)].groupby("ticker").apply(lambda g: (g.c_n / g.c_s).median())
        print("mismatch median ratio new/store:", x.round(4).to_dict())
    qm = m[m.date > m.date.min()]
    print("q match (after first day):", float((abs(qm.q_n - qm.q_s) < 1e-4).mean()))
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".parquet"))
    tot, per_year = 0, {}
    for f in files:
        d = pd.read_parquet(os.path.join(OUT, f), columns=["ticker"])
        tot += len(d)
        per_year.setdefault(f[:4], set()).update(d.ticker.unique())
    print("total rows", tot, "months", len(files))
    print("tickers per year", {y: len(s) for y, s in per_year.items()})
    p = load_prog()
    print("requested", len(tickers()), "got any data", sum(v["n_got"] for v in p["done"].values()),
          f"download elapsed {p['elapsed'] / 60:.1f} min")


if __name__ == "__main__":
    if sys.argv[1:] == ["verify"]:
        verify()
    else:
        download()
        build()
        verify()
