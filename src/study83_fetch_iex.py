"""Study 83 helper: IEX trades 09:10:00-09:25:00 ET (historical, feed=iex) for the study-25 universe, aggregated per
ticker-day -> data/local/iex83/YYYY-MM.parquet (last price, time of last trade, trade count, shares, VWAP).
Only those trades are needed: SIP bars up to 09:10 are available on the free plan at 09:25, IEX trades are real time.
    python src/study83_fetch_iex.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
from core import DATA

OUT = os.path.join(DATA, "local", "iex83")


def trades(syms, a, b):
    p = dict(symbols=",".join(syms), start=a, end=b, limit=10000, feed="iex", sort="asc")
    rows = []
    while True:
        j = A.get("trades", p)
        for s, tl in (j.get("trades") or {}).items():
            rows.extend((s, t["t"], t["p"], t["s"]) for t in tl)
        tok = j.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok
    return rows


def one_day(d, syms):
    a = pd.Timestamp(f"{d.date()} 09:10:00").tz_localize("America/New_York").tz_convert("UTC")
    b = pd.Timestamp(f"{d.date()} 09:25:00").tz_localize("America/New_York").tz_convert("UTC")
    rows = []
    for i in range(0, len(syms), 400):
        rows += trades(syms[i:i + 400], a.strftime("%Y-%m-%dT%H:%M:%SZ"), b.strftime("%Y-%m-%dT%H:%M:%SZ"))
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows, columns=["ticker", "t", "p", "s"])
    t["ts"] = pd.to_datetime(t.t, utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.tz_localize(None)
    t = t[t.ts <= pd.Timestamp(f"{d.date()} 09:25:00")].sort_values(["ticker", "ts"])
    t["pv"] = t.p * t.s
    g = t.groupby("ticker")
    x = pd.DataFrame({"iex_last": g.p.last(), "iex_t": g.ts.last(), "iex_n": g.size(), "iex_v": g.s.sum(),
                      "iex_pv": g.pv.sum()}).reset_index()
    x["date"] = d
    return x


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f = pd.read_pickle(os.path.join(DATA, "local", "m5pre", "_study25_features.pkl"))[["date", "ticker"]]
    f["date"] = pd.to_datetime(f.date)
    for m, fm in f.groupby(f.date.dt.strftime("%Y-%m")):
        fn = os.path.join(OUT, f"{m}.parquet")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        jobs = [(d, sorted(g.ticker.unique())) for d, g in fm.groupby("date")]
        with ThreadPoolExecutor(3) as ex:
            parts = list(ex.map(lambda j: one_day(*j), jobs))
        out = pd.concat([p for p in parts if len(p)], ignore_index=True)
        out.to_parquet(fn, index=False)
        print(m, len(out), "ticker-days with IEX trades of", len(fm), f"{time.time() - t0:.0f}s", flush=True)
