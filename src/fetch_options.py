"""Alpaca option contracts + daily option bars (read-only market data; never touches orders).

    python src/fetch_options.py probe                 # which endpoints answer on this plan
    python src/fetch_options.py contracts SPY QQQ ... # contract lists (active + inactive) -> data/local/options/contracts_<U>.parquet
    python src/fetch_options.py bars SPY QQQ ...      # daily bars of the contracts selected by select_needed()

Pacing: at most 55 requests/minute from this process, through the shared file-lock limiter of alpaca_data.
"""
import os
import sys
import time

os.environ.setdefault("ALPACA_PER_MIN", "55")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import alpaca_data as AD  # noqa: E402

OUT = os.path.join(AD.LOCAL, "options")
TRADE = "https://paper-api.alpaca.markets/v2"
DATA = "https://data.alpaca.markets/v1beta1/options"


def req(url, params, tries=8):
    for k in range(tries):
        AD.RL.wait()
        try:
            r = AD._sess().get(url, params=params, timeout=60)
        except Exception as e:  # noqa: BLE001
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


def probe():
    j = req(f"{TRADE}/options/contracts", dict(underlying_symbols="SPY", expiration_date_gte="2024-03-01",
                                               expiration_date_lte="2024-03-01", status="inactive", limit=5,
                                               type="put"))
    cs = j.get("option_contracts", [])
    print("contracts", len(cs), cs[:1])
    sym = cs[0]["symbol"] if cs else "SPY240301P00500000"
    for ep in ["bars", "trades", "quotes"]:
        p = dict(symbols=sym, start="2024-02-26", end="2024-03-02", limit=20)
        if ep == "bars":
            p["timeframe"] = "1Day"
        try:
            j = req(f"{DATA}/{ep}", p, tries=2)
            v = j.get(ep, {})
            print(ep, "OK", {k: len(x) for k, x in v.items()}, str(v)[:300])
        except Exception as e:  # noqa: BLE001
            print(ep, "ERR", str(e)[:300])
    for ep in ["quotes/latest", "snapshots/SPY"]:
        try:
            j = req(f"{DATA}/{ep}", dict(symbols=sym, feed="indicative", limit=2), tries=2)
            print(ep, "OK", str(j)[:400])
        except Exception as e:  # noqa: BLE001
            print(ep, "ERR", str(e)[:300])


def contracts(u):
    fn = os.path.join(OUT, f"contracts_{u}.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    rows = []
    # month by month keeps pages small; inactive = expired
    for m in pd.period_range("2024-02", "2026-11", freq="M"):
        a, b = m.start_time.strftime("%Y-%m-%d"), m.end_time.strftime("%Y-%m-%d")
        for st in ["inactive", "active"]:
            p = dict(underlying_symbols=u, expiration_date_gte=a, expiration_date_lte=b, status=st, limit=10000)
            while True:
                j = req(f"{TRADE}/options/contracts", p)
                rows.extend(j.get("option_contracts", []))
                tok = j.get("next_page_token")
                if not tok:
                    break
                p["page_token"] = tok
        print(u, m, len(rows), flush=True)
    d = pd.DataFrame(rows)
    keep = ["symbol", "underlying_symbol", "type", "style", "strike_price", "expiration_date", "size", "status",
            "close_price", "close_price_date", "open_interest", "open_interest_date"]
    d = d[[k for k in keep if k in d]].drop_duplicates("symbol")
    d["strike_price"] = d.strike_price.astype(float)
    d["expiration_date"] = pd.to_datetime(d.expiration_date)
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(fn, index=False)
    return d


def fetch_bars(syms, start, end, timeframe="1Day"):
    rows = []
    for i in range(0, len(syms), 100):
        p = dict(symbols=",".join(syms[i:i + 100]), timeframe=timeframe, start=start, end=end, limit=10000)
        while True:
            j = req(f"{DATA}/bars", p)
            for s, bl in (j.get("bars") or {}).items():
                for b in bl:
                    b["symbol"] = s
                rows.extend(bl)
            tok = j.get("next_page_token")
            if not tok:
                break
            p["page_token"] = tok
    return pd.DataFrame(rows)


def select_needed(u, cs, spot):
    """Last expiry of each week (Friday, or Thursday on a Friday holiday), puts with strikes 80-102% and
    (SPY/QQQ only, for condors) calls 98-110% of any close in the 45 days before expiry."""
    cs = cs[(cs.expiration_date >= "2024-02-20") & (cs.expiration_date.dt.dayofweek <= 4)].copy()
    wk = cs.expiration_date.dt.to_period("W")
    last = cs.groupby(wk).expiration_date.transform("max")
    cs = cs[cs.expiration_date == last]
    spot = spot.copy()
    spot.index = pd.to_datetime(spot.index)
    lo = spot.rolling(45, min_periods=1).min()
    hi = spot.rolling(45, min_periods=1).max()
    ref = pd.DataFrame({"lo": lo, "hi": hi}).sort_index()
    r = ref.reindex(cs.expiration_date.values, method="ffill")
    k = cs.strike_price.values
    put = (cs.type.values == "put") & (k >= 0.80 * r.lo.values) & (k <= 1.02 * r.hi.values)
    call = (cs.type.values == "call") & (k >= 0.98 * r.lo.values) & (k <= 1.10 * r.hi.values) & (u in ("SPY", "QQQ"))
    return cs[put | call]


def bars_for(u):
    fn = os.path.join(OUT, f"bars_{u}.parquet")
    cs = contracts(u)
    daily = pd.read_parquet(os.path.join(AD.ROOT, "data", "store", "daily"))  # noqa
    return fn, cs


def run_bars(u):
    fn = os.path.join(OUT, f"bars_{u}.parquet")
    cs = contracts(u)
    spot = pd.read_parquet(os.path.join(OUT, f"underlying_{u}.parquet")).set_index("date").c.sort_index()
    need = select_needed(u, cs, spot)
    have = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["symbol"])
    done = set(have.symbol.unique())
    donefn = os.path.join(OUT, f"done_{u}.txt")
    if os.path.exists(donefn):
        done |= set(open(donefn).read().split())
    need = need[~need.symbol.isin(done)]
    print(u, "contracts to fetch", len(need), flush=True)
    # group by expiry so a request covers ~2 months of dates for 100 symbols
    parts = [have] if len(have) else []
    pending = []
    for ex, g in need.groupby("expiration_date"):
        start = (ex - pd.Timedelta(days=50)).strftime("%Y-%m-%d")
        end = min(ex + pd.Timedelta(days=1), pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if ex - pd.Timedelta(days=50) >= pd.Timestamp.today().normalize():
            continue
        b = fetch_bars(g.symbol.tolist(), start, end)
        if len(b):
            b["date"] = pd.to_datetime(b.t).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            b = b[["symbol", "date", "o", "h", "l", "c", "v", "n", "vw"]]
            parts.append(b)
        if ex < pd.Timestamp.today().normalize():
            pending += list(g.symbol)
        print(u, ex.date(), len(g), len(b), flush=True)
        if len(parts) > 20:
            parts = [pd.concat(parts, ignore_index=True)]
            parts[0].to_parquet(fn, index=False)
            with open(donefn, "a") as f:     # mark done only once the bars are on disk
                f.write("\n".join(pending) + "\n")
            pending = []
    if parts:
        pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"]).to_parquet(fn, index=False)
        with open(donefn, "a") as f:
            f.write("\n".join(pending) + "\n")


def snap_quotes(u, spot):
    """Current chain snapshot (indicative feed, the only one the free plan allows) -> bid/ask sample."""
    rows = []
    today = pd.Timestamp.today().normalize()
    p = dict(feed="indicative", limit=1000, expiration_date_gte=(today + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
             expiration_date_lte=(today + pd.Timedelta(days=50)).strftime("%Y-%m-%d"),
             strike_price_gte=round(spot * 0.75, 2), strike_price_lte=round(spot * 1.15, 2))
    while True:
        j = req(f"{DATA}/snapshots/{u}", p)
        for s, v in (j.get("snapshots") or {}).items():
            q = v.get("latestQuote") or {}
            db = v.get("dailyBar") or {}
            rows.append(dict(symbol=s, bid=q.get("bp"), ask=q.get("ap"), bs=q.get("bs"), as_=q.get("as"),
                             qt=q.get("t"), day_c=db.get("c"), day_v=db.get("v"), iv=v.get("impliedVolatility"),
                             delta=(v.get("greeks") or {}).get("delta")))
        tok = j.get("next_page_token")
        if not tok:
            break
        p["page_token"] = tok
    d = pd.DataFrame(rows)
    d["underlying"] = u
    d["spot"] = spot
    return d


def underlying_daily(u):
    b = AD.bars([u], "1Day", "2019-06-01T00:00:00Z",
                (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                adjustment="raw")
    b["date"] = b.ts.dt.normalize()
    return b


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "spreads":
        out = []
        for u in sys.argv[2:]:
            dd = underlying_daily(u)
            dd.to_parquet(os.path.join(OUT, f"underlying_{u}.parquet"), index=False)
            out.append(snap_quotes(u, float(dd.c.iloc[-1])))
            print(u, len(out[-1]), flush=True)
        pd.concat(out).to_parquet(os.path.join(OUT, f"snap_quotes_{pd.Timestamp.today():%Y%m%d}.parquet"), index=False)
    if cmd == "probe":
        probe()
    elif cmd == "contracts":
        for u in sys.argv[2:]:
            print(u, len(contracts(u)))
    elif cmd == "bars":
        for u in sys.argv[2:]:
            run_bars(u)
