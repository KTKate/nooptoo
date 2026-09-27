"""Alpaca news (Benzinga feed) 2019-06 .. last closed day, one parquet per month in data/local/news/.

About 1,000 articles per trading day; limit 50 per request, so roughly 35,000 requests (~3 hours at the shared
190/min pace). Months already on disk are skipped (the current month is always refetched).
Columns: id, ts (UTC), symbols ("|"-joined), headline, summary, category (from the Benzinga URL path), source.
    python src/fetch_news.py [start_month] [end_month]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import alpaca_data as A

URL = "https://data.alpaca.markets/v1beta1/news"
OUT = os.path.join(A.LOCAL, "news")
os.makedirs(OUT, exist_ok=True)


def get(params, tries=8):
    import time
    for k in range(tries):
        A.RL.wait()
        try:
            r = A._sess().get(URL, params=params, timeout=60)
        except Exception as e:                                  # network error: back off and retry
            print("net err", e, flush=True)
            time.sleep(2 ** min(k, 5))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** min(k, 5))
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    raise RuntimeError("too many retries")


def category(url):
    # https://www.benzinga.com/news/earnings/25/03/... -> news/earnings ; /analyst-ratings/... -> analyst-ratings
    try:
        parts = url.split("benzinga.com/")[1].split("/")
        return "/".join(p for p in parts[:2] if not p[:2].isdigit())
    except Exception:
        return ""


def one_day(day):
    a, b = day.strftime("%Y-%m-%dT00:00:00Z"), (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    p = dict(start=a, end=b, limit=50, sort="asc", include_content="false")
    rows = []
    while True:
        j = get(p)
        for n in j.get("news", []):
            rows.append(dict(id=n["id"], ts=n["created_at"], symbols="|".join(n.get("symbols") or []),
                             headline=n.get("headline", ""), summary=n.get("summary", ""),
                             category=category(n.get("url", "")), source=n.get("source", "")))
        tok = j.get("next_page_token")
        if not tok:
            return rows
        p["page_token"] = tok


def month(m):
    fn = os.path.join(OUT, f"{m}.parquet")
    cur = pd.Timestamp.today().strftime("%Y-%m")
    if os.path.exists(fn) and m < cur:
        return 0
    start = pd.Timestamp(m + "-01")
    end = min(start + pd.offsets.MonthBegin(1), pd.Timestamp.today().normalize())
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one_day, days):
            rows += r
    d = pd.DataFrame(rows)
    if len(d):
        d["ts"] = pd.to_datetime(d.ts, utc=True)
        d = d.drop_duplicates("id")
    d.to_parquet(fn, compression="zstd", index=False)
    print(m, len(d), flush=True)
    return len(d)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "2019-06"
    b = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m")
    for m in [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")][::-1]:     # newest first
        month(m)
