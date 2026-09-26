"""Calibrate the per-side spread cost model with Alpaca SIP quotes.

Sample: tickers stratified by 20-day median dollar volume (5 buckets), random trading days in
2024-01..2026-09, and four times of day (09:31, 09:35, 12:00, 15:45 ET). For each (ticker, day, time)
the first page of NBBO quotes from that minute gives the median quoted half-spread in bps.

Then a log-linear model  log(hs) = a + b1 log(adv) + b2 log(price) + b3 log(vol20) + time-of-day dummies
is fitted with inputs known before the trade (lagged one day). core.half_spread_model() uses it.
Output: results/spread_sample.parquet, results/spread_model.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "40")
import alpaca_data as A
from core import load_panel, stock_cols, RES

A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
TIMES = ["09:31", "09:35", "12:00", "15:45"]


def sample_quotes(n_per_bucket=40, n_days=4, seed=1):
    P = load_panel()
    cols = stock_cols(P)
    c, dv = P["rawc"][cols], P["dv"][cols]
    adv = dv.rolling(20, min_periods=10).median().shift(1)
    rng = np.random.default_rng(seed)
    days = c.loc["2024-01-02":"2026-09-24"].index
    buckets = [(5e8, 1e13), (1e8, 5e8), (2e7, 1e8), (5e6, 2e7), (1e6, 5e6)]
    rows = []
    for d in rng.choice(days, n_days, replace=False):
        a = adv.loc[d]
        px = c.shift(1).loc[d]
        for lo, hi in buckets:
            pool = a[(a > lo) & (a <= hi) & (px > 2)].index
            for t in rng.choice(pool, min(n_per_bucket, len(pool)), replace=False):
                rows.append((pd.Timestamp(d), t))
    def one(job):
        d, t, hm = job
        st = pd.Timestamp(f"{d.date()} {hm}").tz_localize("America/New_York").tz_convert("UTC")
        try:
            j = A.get("quotes", dict(symbols=t, start=st.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     end=(st + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     limit=1000, feed="sip"))
        except RuntimeError as e:
            print("err", t, d, e)
            return None
        q = pd.DataFrame((j.get("quotes") or {}).get(t, []))
        if not len(q):
            return None
        q = q[(q.bp > 0) & (q.ap > q.bp)]
        if not len(q):
            return None
        mid = (q.ap + q.bp) / 2
        hs = ((q.ap - q.bp) / 2 / mid * 1e4)
        return dict(date=d, ticker=t, hm=hm, hs_med=hs.median(), hs_mean=hs.mean(), nq=len(q), mid=mid.median())
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(d, t, hm) for d, t in rows for hm in TIMES]
    with ThreadPoolExecutor(8) as ex:
        out = [x for x in ex.map(one, jobs) if x is not None]
    s = pd.DataFrame(out)
    s.to_parquet(f"{RES}/spread_sample.parquet", index=False)
    return s


def fit(s=None):
    if s is None:
        s = pd.read_parquet(f"{RES}/spread_sample.parquet")
    P = load_panel()
    c, dv = P["rawc"], P["dv"]
    lr = np.log(P["c"] / P["c"].shift(1))
    adv = dv.rolling(20, min_periods=10).median().shift(1)
    vol = lr.rolling(20, min_periods=10).std().shift(1)
    px = c.shift(1)

    def look(M):
        return np.array([M.at[d, t] if t in M.columns else np.nan for d, t in zip(s.date, s.ticker)])
    s = s.copy()
    s["ladv"], s["lpx"], s["lvol"] = np.log(look(adv)), np.log(look(px)), np.log(look(vol))
    s["y"] = np.log(s.hs_med.clip(lower=0.1))
    s = s.replace([np.inf, -np.inf], np.nan).dropna(subset=["ladv", "lpx", "lvol", "y"])
    X = np.column_stack([np.ones(len(s)), s.ladv, s.lpx, s.lvol] +
                        [(s.hm == hm).astype(float) for hm in TIMES[:-1]])
    beta, *_ = np.linalg.lstsq(X, s.y.values, rcond=None)
    res = s.y.values - X @ beta
    names = ["const", "ladv", "lpx", "lvol"] + [f"t{hm}" for hm in TIMES[:-1]]
    m = dict(zip(names, beta.tolist()))
    m["resid_sd"] = float(res.std())
    m["n"] = int(len(s))
    json.dump(m, open(f"{RES}/spread_model.json", "w"), indent=1)
    s["fit"] = X @ beta
    s["bucket"] = pd.cut(np.exp(s.ladv), [0, 5e6, 2e7, 1e8, 5e8, 1e13])
    print(m)
    print(s.groupby(["bucket", "hm"], observed=True).hs_med.median().unstack().round(1))
    return m, s


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    else:
        fit(sample_quotes())
