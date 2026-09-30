"""Study 35: buying after-hours drops and selling in the next opening auction.

Study 34: picks that fell 2-5% between the close and 17:00-20:00 rose 1.2-1.5% more by the open (t 4-5), and in the
extended session the last trade is usually at the ask (the bid is 45-60 bp below it), so buying there costs little
beyond the last price. Universe: liquid stocks of the day (traded close > $5, 20-day median dollar volume > $5M).
Data: SIP 5-minute bars 16:00-20:00 (raw) for the whole universe, cached per month in data/local/m5ah/.
Event at T in {17:00, 18:00, 19:00, 20:00}: price_T / close - 1 below -x, with volume in the hour before T.
Outcome: next open / price_T - 1, next open = traded close * (adjusted open(t+1) / adjusted close(t)).
Split by earnings night (Nasdaq calendar), and priced at the real ask for a sample (study35b).
Output: results/study35_events.parquet, results/study35_ah_reversal.csv
"""
import os
import numpy as np
import pandas as pd
import alpaca_data as A
import store
from concurrent.futures import ThreadPoolExecutor
from core import load_panel, stock_cols, traded_close, RES, DATA

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
tc = traded_close(P)[cols]
adv = P["dv"][cols].rolling(20).median().shift(1)
DIR = f"{DATA}/local/m5ah"
os.makedirs(DIR, exist_ok=True)
test_days = [d for d in days if pd.Timestamp("2024-01-02") <= d < days[-1]]


def fetch_day(d):
    tick = [t for t in cols if tc.at[d, t] > 5 and adv.at[d, t] > 5e6]
    a = pd.Timestamp(f"{d.date()} 16:00").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    b = pd.Timestamp(f"{d.date()} 20:00").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for i in range(0, len(tick), 400):
        p = dict(symbols=",".join(tick[i:i + 400]), timeframe="5Min", start=a, end=b, limit=10000, adjustment="raw",
                 feed="sip", sort="asc")
        while True:
            j = A.get("bars", p)
            for s, bl in (j.get("bars") or {}).items():
                for x in bl:
                    rows.append((d, s, x["t"], x["c"], x["v"]))
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
    return rows


def load_bars():
    months = sorted({d.strftime("%Y-%m") for d in test_days})
    parts = []
    for m in months:
        fn = f"{DIR}/{m}.parquet"
        if not os.path.exists(fn):
            dd = [d for d in test_days if d.strftime("%Y-%m") == m]
            with ThreadPoolExecutor(4) as ex:
                out = list(ex.map(fetch_day, dd))
            x = pd.DataFrame([r for o in out for r in o], columns=["day", "ticker", "t", "c", "v"])
            x["ts"] = pd.to_datetime(x.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
            x.drop(columns="t").to_parquet(fn)
            print("fetched", m, len(x), flush=True)
        parts.append(pd.read_parquet(fn))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    bars = load_bars()
    bars["end"] = bars.ts + pd.Timedelta(minutes=5)
    bars["hm"] = bars.end.dt.strftime("%H:%M")
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1)
    E = store.read("earnings")
    di = days.searchsorted(pd.to_datetime(E.date))
    earn = pd.DataFrame(False, index=days, columns=cols)
    ci = {t: i for i, t in enumerate(cols)}
    for d, t in zip(di, E.symbol):
        if t in ci and 0 < d < len(days):
            earn.iat[d, ci[t]] = True        # report before the open of d
            earn.iat[d - 1, ci[t]] = True    # or after the close of d - 1 / of d
    evs = []
    for T in ["17:00", "18:00", "19:00", "20:00"]:
        y = bars[bars.hm <= T]
        last = y.groupby(["day", "ticker"]).c.last()
        h0 = (pd.Timestamp("2000-01-01 " + T) - pd.Timedelta(hours=1)).strftime("%H:%M")
        vol = y[y.hm > h0].groupby(["day", "ticker"]).v.sum().reindex(last.index).fillna(0)
        e = pd.DataFrame({"pT": last, "vol_hour": vol}).reset_index()
        e["c0"] = tc.stack().reindex(pd.MultiIndex.from_frame(e[["day", "ticker"]])).values
        e["R"] = R.stack().reindex(pd.MultiIndex.from_frame(e[["day", "ticker"]])).values
        e["earn"] = earn.stack().reindex(pd.MultiIndex.from_frame(e[["day", "ticker"]])).values
        e["rT"] = e.pT / e.c0 - 1
        e["o1"] = e.c0 * (1 + e.R)
        e["fwd"] = e.o1 / e.pT - 1
        e["T"] = T
        e = e[(e.rT < -0.02) & (e.vol_hour > 0) & e.fwd.notna()]
        e = e[np.abs(np.log(e.pT / e.c0)) - np.abs(np.log1p(e.R)) < 0.4]     # drop split mismatches
        evs.append(e)
    ev = pd.concat(evs, ignore_index=True)
    ev.to_parquet(f"{RES}/study35_events.parquet")
    rows = []
    for T in ["17:00", "18:00", "19:00", "20:00"]:
        for lo, hi in [(-0.03, -0.02), (-0.05, -0.03), (-0.1, -0.05), (-0.2, -0.1), (-1, -0.2)]:
            for en in [None, True, False]:
                m = (ev["T"] == T) & (ev.rT > lo) & (ev.rT <= hi)
                if en is not None:
                    m &= ev.earn == en
                x = ev.fwd[m]
                if len(x) < 30:
                    continue
                per = {p: x[(ev.day[m] >= a) & (ev.day[m] <= b)].mean() for p, a, b in
                       [("y2024", "2024-01-01", "2024-12-31"), ("y2025", "2025-01-01", "2025-12-31"), ("y2026", "2026-01-01", "2026-12-31")]}
                rows.append(dict(T=T, lo=lo, hi=hi, earn={None: "all", True: "earnings", False: "no earnings"}[en],
                                 n=len(x), mean_fwd_bp=1e4 * x.mean(), median_fwd_bp=1e4 * x.median(),
                                 t=x.mean() / (x.std() / np.sqrt(len(x))), **{k: 1e4 * v for k, v in per.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study35_ah_reversal.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string())
