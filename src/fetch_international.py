"""Study 16 data: US-listed ADRs (+ large foreign-domiciled US listings) and country ETFs.

Outputs (data/local/intl/):
  nasdaqtraded.txt, screener.json   raw directory files (downloaded if missing)
  universe.csv                      ticker, name, country, region, kind (adr_name / foreign_large / etf), in_store
  daily.parquet                     long daily OHLCV from Yahoo (auto_adjust=False), with q like data/store
  m5win.parquet                     Alpaca 5-minute bars in 11:25-11:35 and 15:30-16:00 ET windows (liquid names)

    python src/fetch_international.py universe
    python src/fetch_international.py daily
    python src/fetch_international.py m5 [TICKERS...]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "local", "intl")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ETFS = ["EWJ", "EWG", "EWU", "FXI", "MCHI", "EWZ", "EWW", "EWY", "EWT", "INDA", "EWA", "EWC", "EFA", "EEM",
        "VGK", "EWH", "EWS"]
ETF_REGION = {"EWJ": "Japan/APAC", "EWG": "Europe", "EWU": "Europe", "FXI": "China/HK", "MCHI": "China/HK",
              "EWZ": "LatAm", "EWW": "LatAm", "EWY": "Japan/APAC", "EWT": "Japan/APAC", "INDA": "Japan/APAC",
              "EWA": "Japan/APAC", "EWC": "Other", "EFA": "Europe", "EEM": "Other", "VGK": "Europe",
              "EWH": "China/HK", "EWS": "Japan/APAC"}

KEEP_FOREIGN = {"AMX", "AZN", "BBVA", "BCS", "BP", "BSBR", "CHT", "CIB", "DB", "DEO", "E", "EMBJ", "EQNR", "FER",
                "FMX", "HDB", "HMC", "HSBC", "IBN", "ING", "JHX", "KB", "KEP", "MUFG", "NVO", "NVS", "PAC", "PUK",
                "RDY", "RIO", "SKM", "SNN", "SQM", "STLA", "STM", "TLK", "TM", "TSM", "TTE", "UBS", "UMC", "WIT",
                "ALC", "ITUB"}

EUROPE = {"United Kingdom", "Ireland", "France", "Germany", "Switzerland", "Netherlands", "Sweden", "Finland",
          "Denmark", "Spain", "Belgium", "Luxembourg", "Italy", "Norway", "Austria", "Portugal", "Greece", "Jersey",
          "Guernsey", "Isle of Man", "Monaco", "Cyprus", "Hungary", "Poland", "Czech Republic", "Iceland"}
APAC = {"Japan", "South Korea", "Taiwan", "Australia", "India", "Singapore", "New Zealand", "Indonesia",
        "Philippines", "Malaysia", "Thailand", "Vietnam"}
CHINA = {"China", "Hong Kong", "Macau"}
LATAM = {"Brazil", "Mexico", "Argentina", "Chile", "Colombia", "Peru", "Panama", "Uruguay"}


def region_of(country):
    if country in EUROPE:
        return "Europe"
    if country in APAC:
        return "Japan/APAC"
    if country in CHINA:
        return "China/HK"
    if country in LATAM:
        return "LatAm"
    return "Other"


def _get(url, fn, hdr=None):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        r = requests.get(url, headers=hdr or {}, timeout=120)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    return p


def build_universe():
    src = os.path.join(ROOT, "data", "nasdaqtraded.txt")
    if not os.path.exists(src):
        src = _get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", "nasdaqtraded.txt")
    scr = _get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true",
               "screener.json", {"User-Agent": "Mozilla/5.0"})
    d = pd.read_csv(src, sep="|", keep_default_na=False)
    d = d[(d["Test Issue"] == "N") & (d["ETF"] == "N")]
    n = d["Security Name"]
    adr = (n.str.contains(r"American Depositar|American Depositor|\bADS\b|\bADR\b|Registry Shares", case=False,
                          regex=True)
           & ~n.str.contains("Preferred|Preference|Warrant|Right|Unit", case=False))
    s = pd.DataFrame(json.load(open(scr))["data"]["rows"])
    s["mcap"] = pd.to_numeric(s.marketCap, errors="coerce")
    x = d.merge(s[["symbol", "country", "mcap"]], left_on="Symbol", right_on="symbol", how="left")
    x["adr"] = adr.values
    x["country"] = x.country.fillna("").replace("", np.nan)
    # ADR-named names with unknown / offshore-shell country: ask Yahoo (few dozen calls)
    need = x[x.adr & (x.country.isna() | x.country.isin(["Cayman Islands", "Bermuda", "British Virgin Islands"]))]
    if len(need):
        import yfinance as yf
        for i, t in zip(need.index, need.Symbol):
            try:
                c = yf.Ticker(t.replace(".", "-")).info.get("country")
            except Exception:
                c = None
            if c:
                x.loc[i, "country"] = c
            time.sleep(0.3)
    x["region"] = x.country.map(region_of)
    # large foreign-domiciled listings named "Common Stock" (TM, NVO, AZN, HSBC, TSM ...): most are ADRs
    # whose primary market is abroad. Keep only home regions with a separate trading session.
    big = (~x.adr) & x.region.isin(["Europe", "Japan/APAC", "China/HK", "LatAm"]) & (x.mcap > 1e10)
    # ... but only those with a real primary listing abroad (hand-checked; drops tax-domiciled US companies
    # such as ACN, ETN, JCI, TT, MTD and US-primary foreign names such as SPOT, NBIS, CRH, FLUT)
    big &= x.Symbol.isin(KEEP_FOREIGN)
    u = x[x.adr | big].copy()
    u["kind"] = np.where(u.adr, "adr_name", "foreign_large")
    u = u.rename(columns={"Symbol": "ticker", "Security Name": "name"})[["ticker", "name", "country", "region",
                                                                        "kind", "mcap"]]
    e = pd.DataFrame({"ticker": ETFS, "name": ETFS, "country": "", "region": [ETF_REGION[t] for t in ETFS],
                      "kind": "etf", "mcap": np.nan})
    u = pd.concat([u, e], ignore_index=True)
    import store
    st = set(store.read("daily", start="2026-08", end="2026-08").ticker)
    u["in_store"] = u.ticker.isin(st)
    u.to_csv(os.path.join(OUT, "universe.csv"), index=False)
    print(u.groupby(["kind", "region"]).size())
    return u


def fetch_daily(start="2019-05-01", end="2026-09-28"):
    import yfinance as yf
    u = pd.read_csv(os.path.join(OUT, "universe.csv"))
    tick = sorted(set(u.ticker) | {"SPY"})
    parts = []
    for i in range(0, len(tick), 50):
        ch = tick[i:i + 50]
        ys = [t.replace(".", "-") for t in ch]
        for k in range(4):
            try:
                df = yf.download(ys, start=start, end=end, auto_adjust=False, progress=False, threads=4,
                                 group_by="ticker")
                break
            except Exception as ex:
                print("retry", ex, flush=True)
                time.sleep(5)
        for t, y in zip(ch, ys):
            if y not in df.columns.get_level_values(0):
                continue
            g = df[y].dropna(subset=["Close"])
            if len(g) == 0:
                continue
            f = (g["Adj Close"] / g["Close"]).astype("float64")
            q = f / f.shift(1)
            parts.append(pd.DataFrame({"date": g.index, "ticker": t, "o": g.Open.values, "h": g.High.values,
                                       "l": g.Low.values, "c": g.Close.values, "v": g.Volume.values,
                                       "q": q.values}))
        print("daily", i + len(ch), "/", len(tick), flush=True)
        time.sleep(1)
    d = pd.concat(parts, ignore_index=True)
    for k in ["o", "h", "l", "c", "q"]:
        d[k] = d[k].astype("float32")
    d.to_parquet(os.path.join(OUT, "daily.parquet"), compression="zstd", index=False)
    print(len(d), d.ticker.nunique())


def fetch_m5(tickers, start="2019-06-01", end="2026-09-26"):
    """5-minute bars in the 11:25-11:35 and 15:30-16:00 ET windows. One request per ticker-chunk per month,
    filtered locally (Alpaca has no time-of-day filter; 5-minute bars for a month are ~1.6k per ticker)."""
    import alpaca_data as A
    fn = os.path.join(OUT, "m5win.parquet")
    have = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["ts", "ticker"])
    done = set(have.ticker.unique())
    tickers = [t for t in tickers if t not in done]
    months = pd.date_range(start, end, freq="MS")
    out = [have] if len(have) else []
    for i in range(0, len(tickers), 10):
        ch = tickers[i:i + 10]
        got = []
        for m in months:
            a = m.strftime("%Y-%m-%dT00:00:00Z")
            b = (m + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%dT00:00:00Z")
            df = A.bars(ch, "5Min", a, b)
            if len(df):
                hm = df.ts.dt.hour * 100 + df.ts.dt.minute
                got.append(df[((hm >= 1125) & (hm < 1135)) | (hm >= 1530)])
        if got:
            out.append(pd.concat(got, ignore_index=True))
        pd.concat(out, ignore_index=True).to_parquet(fn, compression="zstd", index=False)
        print("m5", i + len(ch), "/", len(tickers), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "universe"
    if what == "universe":
        build_universe()
    elif what == "daily":
        fetch_daily()
    elif what == "m5":
        fetch_m5(sys.argv[2:])
