"""Free auxiliary datasets for the factor study (study 10).

  python src/fetch_extra.py shortvol    # FINRA Reg SHO daily short-sale volume (off-exchange, all venues
                                        # FINRA reports), 2019-06 .. today -> data/local/shortvol/<month>.parquet
  python src/fetch_extra.py sectors     # sector / industry per ticker (Nasdaq screener) -> data/store/sectors.parquet

The FINRA file for day t is published after the close of t, so it is usable from day t+1 on.
"""
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "data", "local")
UA = {"User-Agent": "nooptoo-research-script"}


def shortvol(start="2019-06-01"):
    out = os.path.join(LOCAL, "shortvol")
    os.makedirs(out, exist_ok=True)
    days = pd.bdate_range(start, pd.Timestamp.today().normalize() - pd.Timedelta(days=1))

    def one(d):
        u = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
        for k in range(4):
            try:
                r = requests.get(u, timeout=30, headers=UA)
                if r.status_code == 404:
                    return None                                  # holiday
                if r.status_code == 200:
                    x = pd.read_csv(io.StringIO(r.text), sep="|", usecols=[0, 1, 2, 4])
                    x = x[x.Date.astype(str).str.fullmatch(r"\d{8}")]
                    return x
            except Exception:
                pass
            time.sleep(2 ** k)
        return None
    for m, dl in pd.Series(days, index=days).groupby(days.to_period("M")):
        fn = os.path.join(out, f"{m}.parquet")
        if os.path.exists(fn) and str(m) < pd.Timestamp.today().strftime("%Y-%m"):
            continue
        with ThreadPoolExecutor(8) as ex:
            parts = [p for p in ex.map(one, dl) if p is not None and len(p)]
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        d.columns = ["date", "ticker", "short_vol", "total_vol"]
        d["date"] = pd.to_datetime(d.date.astype(str))
        d.to_parquet(fn, compression="zstd", index=False)
        print("shortvol", m, len(d), flush=True)


def sectors():
    """Current sector and industry per listed stock from the Nasdaq screener (one request). Classification
    as of today, applied to the whole history (industries rarely change; a small look-ahead)."""
    r = requests.get("https://api.nasdaq.com/api/screener/stocks", headers={"User-Agent": "Mozilla/5.0"},
                     params=dict(tableonly="true", limit=25, offset=0, download="true"), timeout=60)
    d = pd.DataFrame(r.json()["data"]["rows"])[["symbol", "sector", "industry", "marketCap"]]
    d = d.rename(columns={"symbol": "ticker"})
    d["ticker"] = d.ticker.str.strip().str.replace("/", "-", regex=False)
    d.to_parquet(os.path.join(ROOT, "data", "store", "sectors.parquet"), index=False)
    print("sectors", len(d), d.sector.value_counts().to_dict())


if __name__ == "__main__":
    {"shortvol": shortvol, "sectors": sectors}[sys.argv[1]]()
