"""Alpaca market-data fetcher (read-only; never touches the trading API).

Keys come from the environment: ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY
(APCA_API_KEY_ID / APCA_API_SECRET_KEY also accepted).

Local datasets (git-ignored, several GB; same monthly-partition layout as data/store):
  data/local/m1/YYYY-MM.parquet   1-minute SIP bars, index/sector ETFs, 2019-06 ..
  data/local/m5/YYYY-MM.parquet   5-minute SIP bars, liquid stock universe, 2024-01 ..
  data/local/done_<ds>.parquet    (ticker, month) pairs already fetched -> incremental, never refetch

Bars are split-adjusted as of the fetch date (adjustment=split), like the Yahoo daily store,
and timestamps are America/New_York wall-clock (tz-naive) bar START times. Only 07:00-16:00 is kept.

    python src/alpaca_data.py m1            # ETFs, 1-minute
    python src/alpaca_data.py m5            # stock universe, 5-minute
    python src/alpaca_data.py quotes        # quoted-spread sample for the cost model
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "data", "local")
BASE = "https://data.alpaca.markets/v2/stocks"
KEY = os.environ.get("ALPACA_PAPER_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
SEC = os.environ.get("ALPACA_PAPER_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
HDR = {"APCA-API-KEY-ID": KEY or "", "APCA-API-SECRET-KEY": SEC or "", "Accept-Encoding": "gzip"}

ETF1M = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
         "XLRE", "XLC", "SMH", "XBI", "KRE", "ARKK", "TLT", "HYG", "GLD", "USO", "UUP", "TQQQ", "SQQQ", "SOXL",
         "SOXS", "SPXL", "UPRO", "TNA", "TZA", "SPXU", "SDS", "SSO", "QLD"]


class RateLimiter:
    def __init__(self, per_min=190):
        self.gap = 60.0 / per_min
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - time.monotonic()))


RL = RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "190")))
_tls = threading.local()


def _sess():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers.update(HDR)
    return _tls.s


def get(path, params, tries=8):
    for k in range(tries):
        RL.wait()
        try:
            r = _sess().get(f"{BASE}/{path}", params=params, timeout=60)
        except requests.RequestException as e:
            print("net err", e, flush=True)
            time.sleep(2 ** min(k, 5))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** min(k, 5))
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    raise RuntimeError("too many retries")


def bars(symbols, timeframe, start, end, adjustment="split"):
    """All bars for symbols in [start, end) as a long DataFrame (paginated)."""
    p = dict(symbols=",".join(symbols), timeframe=timeframe, start=start, end=end, limit=10000,
             adjustment=adjustment, feed="sip", sort="asc")
    rows = []
    while True:
        j = get("bars", p)
        for s, bl in (j.get("bars") or {}).items():
            if bl:
                d = pd.DataFrame(bl)
                d["ticker"] = s
                rows.append(d)
        tok = j.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d["ts"] = pd.to_datetime(d.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    d = d.rename(columns={"vw": "vwap"})
    if timeframe.endswith("Min"):
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[(hm >= 700) & (hm < 1600)]
    out = d[["ts", "ticker", "o", "h", "l", "c", "v", "n", "vwap"]].copy()
    for k in ["o", "h", "l", "c", "vwap"]:
        out[k] = out[k].astype("float32")
    out["v"] = out["v"].astype("float64")
    out["n"] = out["n"].astype("int32")
    return out


# ------------------------------------------------------------------ local store
def _done_fn(ds):
    return os.path.join(LOCAL, f"done_{ds}.parquet")


def done_pairs(ds):
    fn = _done_fn(ds)
    return pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["ticker", "month"])


def read(ds, start=None, end=None, tickers=None, columns=None):
    d = os.path.join(LOCAL, ds)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if start:
        files = [f for f in files if f[:7] >= str(start)[:7]]
    if end:
        files = [f for f in files if f[:7] <= str(end)[:7]]
    filt = [("ticker", "in", list(tickers))] if tickers is not None else None
    parts = [pd.read_parquet(os.path.join(d, f), filters=filt, columns=columns) for f in files]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_month(ds, tickers, month, timeframe, chunk, workers=4):
    """Fetch one calendar month for the tickers not yet done; append to the month's partition."""
    dn = done_pairs(ds)
    have = set(dn.ticker[dn.month == month])
    todo = [t for t in tickers if t not in have]
    if not todo:
        return 0
    a = pd.Timestamp(month + "-01")
    b = a + pd.offsets.MonthBegin(1)
    groups = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda g: bars(g, timeframe, a.strftime("%Y-%m-%dT00:00:00Z"),
                                           b.strftime("%Y-%m-%dT00:00:00Z")), groups))
    parts = [p for p in parts if len(p)]
    os.makedirs(os.path.join(LOCAL, ds), exist_ok=True)
    fn = os.path.join(LOCAL, ds, f"{month}.parquet")
    if parts:
        new = pd.concat(parts, ignore_index=True)
        if os.path.exists(fn):
            new = pd.concat([pd.read_parquet(fn), new], ignore_index=True).drop_duplicates(["ts", "ticker"], keep="last")
        new.sort_values(["ticker", "ts"]).to_parquet(fn, compression="zstd", index=False)
    if month < pd.Timestamp.today().strftime("%Y-%m"):   # the running month is refetched next time
        dn = pd.concat([dn, pd.DataFrame({"ticker": todo, "month": month})], ignore_index=True)
        dn.to_parquet(_done_fn(ds), index=False)
    return sum(len(p) for p in parts)


def months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def last_closed_month_end():
    return (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m")


def stock_universe(start="2023-12", min_dv=5e6, min_px=3.0):
    """Common stocks whose 20-day median dollar volume exceeded min_dv (price > min_px) on at least
    one day since `start`. Uses the Yahoo daily store; ETFs excluded (they go to m1)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import store
    d = store.read("daily", start=start)
    d = d.sort_values(["ticker", "date"])
    d["dv"] = d.c * d.v
    med = d.groupby("ticker").dv.transform(lambda x: x.rolling(20, min_periods=5).median())
    ok = d[(med > min_dv) & (d.c > min_px)].ticker.unique()
    return sorted(set(ok) - set(ETF1M) - {t for t in ok if t.startswith("^")})


def run_m1(start="2019-06"):
    for m in months(start, last_closed_month_end()):
        n = fetch_month("m1", ETF1M, m, "1Min", chunk=4)
        print("m1", m, n, flush=True)


def run_m5(start="2024-01", tickers=None):
    tick = tickers or stock_universe()
    print("m5 universe", len(tick), flush=True)
    for m in months(start, last_closed_month_end()):
        t0 = time.time()
        n = fetch_month("m5", tick, m, "5Min", chunk=50)
        print("m5", m, n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "m1"
    if what == "m1":
        run_m1()
    elif what == "m5":
        run_m5()
