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
    """Request pacing shared by every process on this machine (the 200/min limit is per account):
    the next free slot time is kept in a small file guarded by an exclusive lock. per_min is
    this process's own ceiling; the global ceiling is GLOBAL_PER_MIN."""
    GLOBAL_PER_MIN = 190

    def __init__(self, per_min=190):
        self.own_gap = 60.0 / per_min
        self.gap = 60.0 / self.GLOBAL_PER_MIN
        self.lock = threading.Lock()
        self.own_next = 0.0
        self.fn = os.path.join(LOCAL, ".ratelimit")
        os.makedirs(LOCAL, exist_ok=True)

    def wait(self):
        """Take the next global slot, but only once this process's own pacing allows it (a slot is never
        reserved ahead for pacing reasons, which would stall the other processes)."""
        import fcntl
        with self.lock:
            while True:
                now = time.time()
                if now < self.own_next:
                    time.sleep(self.own_next - now)
                    continue
                with open(self.fn, "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.seek(0)
                    txt = f.read().strip()
                    t = max(time.time(), float(txt) if txt else 0.0)
                    f.seek(0)
                    f.truncate()
                    f.write(repr(t + self.gap))
                    f.flush()
                    fcntl.flock(f, fcntl.LOCK_UN)
                self.own_next = t + self.own_gap
                break
        time.sleep(max(0.0, t - time.time()))


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
            for b in bl:
                b["ticker"] = s
            rows.extend(bl)
        tok = j.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame.from_records(rows)
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
    # the free plan only serves SIP data older than 15 minutes
    b = min(b, pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min") - pd.Timedelta(minutes=20))
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


# ------------------------------------------------------------------ official auction prices
def _official(t, day, kind):
    """Primary-exchange opening cross ('O' opening print) or closing cross ('6' closing print) for ticker t
    on day, from SIP trades. ('Q'/'M' official open/close are reported by every market center, so a 2-share
    Arca print can carry them; they are not used.) Searches short windows first (liquid names print
    thousands of trades per minute)."""
    if kind == "open":
        wins, cond = [("09:30:00", "09:30:20"), ("09:30:20", "09:32:00"), ("09:32:00", "09:45:00")], "O"
    else:
        wins, cond = [("15:59:59", "16:00:30"), ("16:00:30", "16:15:00")], "6"
    first = np.nan
    for wa, wb in wins:
        a = pd.Timestamp(f"{day} {wa}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        b = pd.Timestamp(f"{day} {wb}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        p = dict(symbols=t, start=a, end=b, limit=10000, feed="sip")
        for _ in range(3):
            j = get("trades", p)
            for x in (j.get("trades") or {}).get(t, []):
                c = x.get("c", [])
                if np.isnan(first) and not set(c) & {"I", "T", "U", "Z"}:
                    first = x["p"]
                if cond in c:
                    return x["p"], first
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
    return np.nan, first


def auction_prices(pairs, workers=8, with_close=False):
    """pairs: iterable of (ticker, date). Returns DataFrame ticker, date, open_off, close_off, first_trade.
    Cached in data/local/auctions.parquet; only missing pairs are requested."""
    fn = os.path.join(LOCAL, "auctions.parquet")
    have = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(
        columns=["ticker", "date", "open_off", "close_off", "first_trade"])
    have["date"] = pd.to_datetime(have["date"])
    want = pd.DataFrame(list(pairs), columns=["ticker", "date"]).drop_duplicates()
    want["date"] = pd.to_datetime(want["date"])
    todo = want.merge(have[["ticker", "date"]], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"][["ticker", "date"]]

    def one(r):
        d = r[1].strftime("%Y-%m-%d")
        try:
            o, f = _official(r[0], d, "open")
            c = _official(r[0], d, "close")[0] if with_close else np.nan
        except RuntimeError:
            o = c = f = np.nan
        return dict(ticker=r[0], date=r[1], open_off=o, close_off=c, first_trade=f)
    rows = []
    items = list(todo.itertuples(index=False, name=None))
    for i in range(0, len(items), 400):
        with ThreadPoolExecutor(workers) as ex:
            rows += list(ex.map(one, items[i:i + 400]))
        new = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        new.to_parquet(fn, index=False)
        print("auctions", len(rows), "/", len(items), flush=True)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    return want.merge(have, on=["ticker", "date"], how="left")


# ------------------------------------------------------------------ targeted windows (faster than full days)
def trading_days(start, end):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import store
    d = store.read("daily", start=start, end=end, tickers=["SPY"])
    return sorted(pd.to_datetime(d.date).dt.normalize().unique())


def fetch_windows(ds="m5snap", start="2024-01", end=None, windows=(("09:30", "10:00"), ("15:30", "16:00")),
                  tickers=None, timeframe="5Min", chunk=400, workers=16):
    """5-minute bars in fixed intraday windows for the whole universe, one request batch per day.
    Progress is tracked per day in done_<ds>.parquet (column 'month' holds the day string)."""
    tick = tickers or stock_universe()
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    days = [d for d in trading_days(start, end) if d < pd.Timestamp.today().normalize()]
    dn = done_pairs(ds)
    have = set(dn.month)
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in have]
    print(ds, "days to fetch", len(todo), "tickers", len(tick), flush=True)
    by_month = {}
    for d in todo:
        by_month.setdefault(d.strftime("%Y-%m"), []).append(d)
    for m, dl in by_month.items():
        jobs = []
        for d in dl:
            for a, b in windows:
                ta = pd.Timestamp(f"{d.date()} {a}").tz_localize("America/New_York").tz_convert("UTC")
                tb = pd.Timestamp(f"{d.date()} {b}").tz_localize("America/New_York").tz_convert("UTC")
                for i in range(0, len(tick), chunk):
                    jobs.append((tick[i:i + chunk], ta.strftime("%Y-%m-%dT%H:%M:%SZ"), tb.strftime("%Y-%m-%dT%H:%M:%SZ")))
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(lambda j: bars(j[0], timeframe, j[1], j[2]), jobs))
        new = pd.concat([p for p in parts if len(p)], ignore_index=True)
        os.makedirs(os.path.join(LOCAL, ds), exist_ok=True)
        fn = os.path.join(LOCAL, ds, f"{m}.parquet")
        if os.path.exists(fn):
            new = pd.concat([pd.read_parquet(fn), new], ignore_index=True).drop_duplicates(["ts", "ticker"], keep="last")
        new.sort_values(["ticker", "ts"]).to_parquet(fn, compression="zstd", index=False)
        dn = pd.concat([dn, pd.DataFrame({"ticker": "*", "month": [d.strftime("%Y-%m-%d") for d in dl]})],
                       ignore_index=True)
        dn.to_parquet(_done_fn(ds), index=False)
        print(ds, m, len(new), f"{time.time() - t0:.0f}s", flush=True)


def fetch_pairs(pairs, ds="m5full", timeframe="5Min", workers=4, a="09:30", b="16:00"):
    """Full regular-session 5-minute bars for specific (ticker, day) pairs; grouped by day so one request
    covers all tickers of that day. Tracks done pairs in done_<ds>.parquet (month column = day)."""
    want = pd.DataFrame(list(pairs), columns=["ticker", "date"]).drop_duplicates()
    want["day"] = pd.to_datetime(want.date).dt.strftime("%Y-%m-%d")
    dn = done_pairs(ds)
    got = set(zip(dn.ticker, dn.month))
    want = want[[(t, d) not in got for t, d in zip(want.ticker, want.day)]]
    print(ds, "pairs to fetch", len(want), flush=True)
    for m, wm in want.groupby(want.day.str[:7]):
        jobs = []
        for d, wd in wm.groupby("day"):
            ta = pd.Timestamp(f"{d} {a}").tz_localize("America/New_York").tz_convert("UTC")
            tb = pd.Timestamp(f"{d} {b}").tz_localize("America/New_York").tz_convert("UTC")
            t = sorted(wd.ticker)
            for i in range(0, len(t), 100):
                jobs.append((t[i:i + 100], ta.strftime("%Y-%m-%dT%H:%M:%SZ"), tb.strftime("%Y-%m-%dT%H:%M:%SZ")))
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(lambda j: bars(j[0], timeframe, j[1], j[2]), jobs))
        parts = [p for p in parts if len(p)]
        os.makedirs(os.path.join(LOCAL, ds), exist_ok=True)
        fn = os.path.join(LOCAL, ds, f"{m}.parquet")
        if parts:
            new = pd.concat(parts, ignore_index=True)
            if os.path.exists(fn):
                new = pd.concat([pd.read_parquet(fn), new], ignore_index=True).drop_duplicates(["ts", "ticker"], keep="last")
            new.sort_values(["ticker", "ts"]).to_parquet(fn, compression="zstd", index=False)
        dn = pd.concat([dn, pd.DataFrame({"ticker": wm.ticker.values, "month": wm.day.values})], ignore_index=True)
        dn.to_parquet(_done_fn(ds), index=False)
        print(ds, m, len(wm), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "m1"
    if what == "m1":
        run_m1()
    elif what == "m5":
        run_m5()
    elif what == "m5snap":
        fetch_windows()
