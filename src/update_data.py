"""Incremental updater for the persistent store (see store.py).

    python src/update_data.py            # update everything that is stale
    python src/update_data.py daily      # one dataset: symbols | daily | earnings | intra60
    python src/update_data.py migrate    # one-time: import the raw downloads in data/daily etc.

Only missing days are downloaded. A 10-trading-day overlap is re-fetched each
time to detect splits (full re-download for that ticker only) and to catch
late corrections. Tickers never seen before get their full history.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import store

HIST_START = "2019-06-01"
ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
        "XLRE", "XLC", "SMH", "XBI", "KRE", "ARKK", "TLT", "HYG", "GLD", "USO", "UUP", "TQQQ", "SQQQ", "SOXL",
        "SOXS", "SPXL", "UPRO", "TNA", "TZA", "^VIX", "^VIX9D", "^VIX3M"]
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


# ------------------------------------------------------------------ symbols
def update_symbols():
    r = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    txt = r.text
    from io import StringIO
    d = pd.read_csv(StringIO(txt), sep="|")
    d = d[(d["Nasdaq Traded"] == "Y") & (d["ETF"] == "N") & (d["Test Issue"] == "N")]
    d = d[~d["Security Name"].str.contains(
        "Warrant|Right|Unit|Preferred|Depositary Shares|Notes|Debenture|%", case=False, regex=True, na=False)]
    d = d[d.Symbol.str.fullmatch(r"[A-Z]{1,5}", na=False)]
    today = pd.Timestamp.today().normalize()
    new = pd.DataFrame({"ticker": d.Symbol.values, "name": d["Security Name"].values, "last_seen": today})
    fn = os.path.join(store.STORE, "symbols.parquet")
    if os.path.exists(fn):
        old = pd.read_parquet(fn)
        m = old.merge(new, on="ticker", how="outer", suffixes=("_o", ""))
        m["name"] = m["name"].fillna(m["name_o"])
        m["first_seen"] = m["first_seen"].fillna(today)
        m["last_seen"] = m["last_seen"].fillna(m["last_seen_o"])
        new = m[["ticker", "name", "first_seen", "last_seen"]]
    else:
        new["first_seen"] = today
    os.makedirs(store.STORE, exist_ok=True)
    new.to_parquet(fn, index=False)
    print("symbols", len(new), "listed today", (new.last_seen == today).sum())
    return new


def all_tickers():
    s = pd.read_parquet(os.path.join(store.STORE, "symbols.parquet"))
    return sorted(set(s.ticker) | set(ETFS))


# ------------------------------------------------------------------ daily
def last_session():
    """Most recent weekday whose regular session has closed (holidays just cause an empty fetch)."""
    now = pd.Timestamp.now(tz="America/New_York")
    d = now.normalize().tz_localize(None)
    if now.hour < 17 or d.dayofweek >= 5:
        d -= pd.tseries.offsets.BDay(1)
    return d if d.dayofweek < 5 else d - pd.tseries.offsets.BDay(1)


def _yf_daily(tickers, start, end=None, tries=4):
    for k in range(tries):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=False, actions=False, group_by="column",
                             threads=True, progress=False)
            if df is not None and len(df):
                break
        except Exception as e:
            print("yf err", e)
        time.sleep(5 * 2 ** k)
    else:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])
    df = df.stack(level=1, future_stack=True).reset_index()
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "ticker"})
    return _to_store(df.dropna(subset=["Close"]))


def _to_store(df):
    """Yahoo columns -> store schema (o,h,l,c split-adjusted; q dividend factor step)."""
    df = df.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v",
                            "Adj Close": "adj_close", "open": "o", "high": "h", "low": "l", "close": "c",
                            "volume": "v"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(["ticker", "date"])
    f = (df["adj_close"] / df["c"]).astype("float64")
    q = f / f.groupby(df["ticker"]).shift(1)
    # guard against Yahoo glitches (zero/negative/inf adj_close): no dividend step there
    df["q"] = q.where((q > 0.5) & (q < 1.5), 1.0).fillna(1.0)
    df["_first"] = ~df["ticker"].duplicated()
    return df[["date", "ticker", "o", "h", "l", "c", "v", "q", "_first"]]


def update_daily(overlap_days=10, chunk=150, tickers=None):
    st = store.read("daily", start=(pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m"))
    known = set(store.read("daily", start="2000-01", end="2100-01").ticker.unique()) if len(st) else set()
    tick = tickers or all_tickers()
    if not len(st):
        print("store empty: run migrate or full download")
        return
    # the last complete date is SPY's: a stray row of one ticker (e.g. a partial bar) must not mark the store current
    spy = st[st.ticker == "SPY"]
    last = spy.date.max() if len(spy) else st.date.max()
    start = (last - pd.tseries.offsets.BDay(overlap_days)).strftime("%Y-%m-%d")
    if last >= last_session():
        print("daily up to date", last.date())
        return
    new_t = [t for t in tick if t not in known]
    print(f"daily: last {last.date()}, fetching from {start} for {len(tick)} tickers; {len(new_t)} new tickers")
    fresh = []
    for i in range(0, len(tick), chunk):
        d = _yf_daily(tick[i:i + chunk], start)
        if len(d):
            fresh.append(d)
        print(" ", i, flush=True)
    fresh = pd.concat(fresh, ignore_index=True)
    fresh = fresh[fresh.date <= last_session()]         # a run during market hours must not store today's partial bar
    # split detection on the overlap window
    ov = fresh.merge(st[["date", "ticker", "c"]], on=["date", "ticker"], suffixes=("", "_old"))
    ratio = (ov.c / ov.c_old).groupby(ov.ticker).median()
    split = ratio[(ratio - 1).abs() > 0.01].index.tolist()
    print("tickers with changed history (splits):", split)
    # the first fetched row of each ticker has no in-window predecessor: keep stored q for it
    fresh = fresh.merge(st[["date", "ticker", "q"]], on=["date", "ticker"], how="left", suffixes=("", "_old"))
    fresh["q"] = np.where(fresh["_first"] & fresh["q_old"].notna(), fresh["q_old"], fresh["q"])
    fresh = fresh[~fresh.ticker.isin(split + new_t)].drop(columns=["q_old", "_first"])
    store.write_month("daily", fresh)
    for t in split + new_t:
        full = _yf_daily([t], HIST_START)
        full = full[full.date <= last_session()] if len(full) else full   # no partial bar of today
        if len(full):
            store.replace_ticker("daily", t, full.drop(columns="_first"))
    store.set_meta(daily_last=str(fresh.date.max().date()))
    print("daily updated to", fresh.date.max().date())


# ------------------------------------------------------------------ earnings
def _earn_day(ds):
    for k in range(4):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={ds}", headers=UA, timeout=30)
            if r.status_code == 200:
                rows = (r.json().get("data") or {}).get("rows") or []
                for x in rows:
                    x["date"] = ds
                return rows
        except Exception as e:
            print("err", ds, e)
        time.sleep(3 * 2 ** k)
    return []


def _earn_frame(rows):
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    for c in ["eps", "epsForecast", "surprise"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[\$,()]", "", regex=True)
                              .str.replace("N/A", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "symbol", "name", "time", "eps", "epsForecast", "surprise", "marketCap", "fiscalQuarterEnding",
            "noOfEsts"]
    return df[[c for c in keep if c in df]].astype({"marketCap": str, "noOfEsts": str})


def update_earnings(back=10, ahead=30):
    from concurrent.futures import ThreadPoolExecutor
    m = store.meta()
    last = pd.Timestamp(m.get("earnings_last", HIST_START))
    start = min(last - pd.Timedelta(days=back), pd.Timestamp.today() - pd.Timedelta(days=back))
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, pd.Timestamp.today() + pd.Timedelta(days=ahead))]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for rs in ex.map(_earn_day, days) for r in rs]
    df = _earn_frame(rows)
    if len(df):
        store.write_month("earnings", df)
    store.set_meta(earnings_last=str(pd.Timestamp.today().date()))
    print("earnings: fetched", len(days), "days,", len(df), "rows")


# ------------------------------------------------------------------ 60m bars
def intra_universe(min_dv=2e7):
    d = store.read("daily", start=(pd.Timestamp.today() - pd.Timedelta(days=100)).strftime("%Y-%m"))
    d["dv"] = d.c * d.v
    g = d.groupby("ticker")
    dv, px = g.dv.median(), g.c.last()
    names = dv[(dv > min_dv) & (px > 3)].index
    return sorted(set(names) | {e for e in ETFS if not e.startswith("^")})


def update_intra60(chunk=50):
    old = store.read("intra60", start=(pd.Timestamp.today() - pd.Timedelta(days=40)).strftime("%Y-%m"))
    tick = sorted(set(intra_universe()) | (set(old.ticker.unique()) if len(old) else set()))
    last = old.ts.max() if len(old) else None
    period = "730d" if last is None else f"{min(729, (pd.Timestamp.now(tz='America/New_York') - last).days + 3)}d"
    print("intra60: fetching", period, "for", len(tick), "tickers")
    for i in range(0, len(tick), chunk):
        for k in range(4):
            try:
                df = yf.download(tick[i:i + chunk], period=period, interval="60m", auto_adjust=False, prepost=False,
                                 group_by="column", threads=True, progress=False)
                break
            except Exception as e:
                print("err", e)
                time.sleep(5 * 2 ** k)
        if df is None or not len(df):
            continue
        df = df.stack(level=1, future_stack=True).reset_index()
        df = df.rename(columns={df.columns[0]: "ts", "Ticker": "ticker", "Open": "o", "High": "h", "Low": "l",
                                "Close": "c", "Volume": "v"}).dropna(subset=["c"])
        df["ts"] = pd.to_datetime(df.ts, utc=True).dt.tz_convert("America/New_York")
        store.write_month("intra60", df[["ts", "ticker", "o", "h", "l", "c", "v"]], key="ts")
        print(" ", i, flush=True)
    store.set_meta(intra60_last=str(pd.Timestamp.today().date()))


# ------------------------------------------------------------------ migration of earlier raw downloads
def migrate(which=("daily", "earnings", "intra60")):
    root = os.path.join(store.ROOT, "data")
    src = os.path.join(root, "daily")
    if "daily" in which and os.path.isdir(src) and not os.listdir(store._dir("daily")):
        parts = [pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src))]
        d = _to_store(pd.concat(parts, ignore_index=True).dropna(subset=["close"])).drop(columns="_first")
        store.write_month("daily", d)
        store.set_meta(daily_last=str(d.date.max().date()))
        print("migrated daily", len(d))
    if "earnings" in which and os.path.exists(os.path.join(root, "earnings.parquet")) and not os.listdir(store._dir("earnings")):
        e = pd.read_parquet(os.path.join(root, "earnings.parquet"))
        e = e.drop(columns=[c for c in ["lastYearRptDt", "lastYearEPS"] if c in e])
        store.write_month("earnings", e.astype({"marketCap": str, "noOfEsts": str}))
        store.set_meta(earnings_last=str(e.date.max().date()))
        print("migrated earnings", len(e))
    src = os.path.join(root, "intra_60m")
    if "intra60" in which and os.path.isdir(src) and not os.listdir(store._dir("intra60")):
        parts = [pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src))]
        x = pd.concat(parts, ignore_index=True)
        store.write_month("intra60", x, key="ts")
        store.set_meta(intra60_last=str(x.ts.max().date()))
        print("migrated intra60", len(x), x.ticker.nunique())


if __name__ == "__main__":
    what = sys.argv[1:] or ["symbols", "daily", "earnings", "intra60"]
    for w in what:
        {"symbols": update_symbols, "daily": update_daily, "earnings": update_earnings,
         "intra60": update_intra60, "migrate": migrate}[w]()
