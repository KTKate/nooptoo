"""Download the Nasdaq earnings calendar (one request per weekday) 2019-06 .. 2026-09.

Each row: report date, symbol, reported EPS, consensus EPS, % surprise, timing
(often 'time-not-supplied' historically). Output: data/earnings.parquet
"""
import os, time, json
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "earnings_raw")
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def day(d):
    fn = os.path.join(OUT, f"{d}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    for k in range(4):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={d}", headers=H, timeout=30)
            if r.status_code == 200:
                j = r.json()
                json.dump(j, open(fn, "w"))
                return j
        except Exception as e:
            print("err", d, e)
        time.sleep(3 * 2 ** k)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for d in pd.bdate_range("2019-06-01", "2026-09-30"):
        ds = d.strftime("%Y-%m-%d")
        j = day(ds)
        if not j or not j.get("data") or not j["data"].get("rows"):
            continue
        for r in j["data"]["rows"]:
            r["date"] = ds
            rows.append(r)
    df = pd.DataFrame(rows)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(r"[\$,()]", "", regex=True)
                             .str.replace("N/A", ""), errors="coerce")
    for c in ["eps", "epsForecast", "surprise"]:
        df[c] = num(df[c])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(os.path.join(ROOT, "data", "earnings.parquet"))
    print(df.shape, df.symbol.nunique())


if __name__ == "__main__":
    main()
