"""Study 21: computer-driven crypto strategies at Alpaca vs buy and hold, net of Alpaca fees.
Costs per side = retail taker fee 25 bps (Alpaca tier $0-100K 30d volume) + half the median quoted spread
(sampled from Alpaca latest crypto quotes; data/local/crypto/quotes_sample.parquet). Long-flat only (no shorts,
no leverage at Alpaca crypto). Periods: dev < 2024-01, val 2024-01..2025-06, hold 2025-07..now.
Data: data/local/crypto/bars_1Day.parquet, bars_1Hour.parquet (python src/fetch_crypto.py bars).
Outputs results/study21_*.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import deflated_sharpe  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "data", "local", "crypto")
RES = os.path.join(ROOT, "results")
FEE = 0.0025
PERIODS = {"dev": ("2000-01-01", "2023-12-31"), "val": ("2024-01-01", "2025-06-30"),
           "hold": ("2025-07-01", "2099-01-01")}
ROWS = []


def cost_table():
    q = pd.read_parquet(os.path.join(CD, "quotes_sample.parquet"))
    spr = ((q.ap - q.bp) / ((q.ap + q.bp) / 2)).groupby(q.symbol).median()
    return (FEE + spr / 2).to_dict()


COST = cost_table()


def cst(s):
    return COST.get(s, FEE + 0.003)


def stats(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(sharpe=np.nan, ann_ret=np.nan, maxdd=np.nan, n=len(r))
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.0
    return dict(sharpe=r.mean() / r.std() * np.sqrt(365), ann_ret=eq.iloc[-1] ** (1 / yrs) - 1,
                maxdd=(eq / eq.cummax() - 1).min(), n=len(r))


def record(test, name, strat, bench, bench_name):
    """strat, bench: daily net return series (UTC days)."""
    for p, (a, b) in PERIODS.items():
        s, bb = strat[a:b], bench[a:b]
        st, sb = stats(s), stats(bb)
        ROWS.append(dict(test=test, variant=name, period=p, bench=bench_name, **st,
                         bh_sharpe=sb["sharpe"], bh_ann_ret=sb["ann_ret"], bh_maxdd=sb["maxdd"],
                         turnover_yr=np.nan))
    SERIES[(test, name)] = strat


SERIES = {}


def load():
    d = pd.read_parquet(os.path.join(CD, "bars_1Day.parquet"))
    d["t"] = d.t.dt.tz_convert(None).dt.normalize()
    d = d[d.t < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]  # drop today's partial bar
    px = d.pivot(index="t", columns="symbol", values="c").sort_index()
    dv = (d.c * d.v).groupby([d.t, d.symbol]).sum().unstack()
    h = pd.read_parquet(os.path.join(CD, "bars_1Hour.parquet"), columns=["symbol", "t", "o", "c"])
    h["t"] = h.t.dt.tz_convert(None)
    return px, dv, h


def net(w, r, c):
    """w: position held during period (decided before), r: asset return, c: per-side cost."""
    w = w.fillna(0)
    return w * r - w.diff().abs().fillna(w.abs()) * c


# ------------------------------------------------------------------ 1. time of day / day of week
def test_tod(h):
    out = []
    for s in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        x = h[h.symbol == s].set_index("t").sort_index()
        full = pd.date_range(x.index.min(), x.index.max(), freq="h")
        c = x.c.reindex(full).ffill()
        r = c.pct_change().fillna(0)
        hr = r.index.hour
        dow = r.index.dayofweek
        bh = (1 + r).groupby(r.index.normalize()).prod() - 1
        # descriptive: mean hourly return by UTC hour, per period
        for p, (a, b) in PERIODS.items():
            rr = r[a:b]
            g = rr.groupby(rr.index.hour).agg(["mean", "std", "count"])
            g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"]))
            for k, v in g.iterrows():
                out.append(dict(symbol=s, period=p, hour_utc=k, mean_bps=v["mean"] * 1e4, t=v["t"]))
        # UTC hour windows [start, end)
        rules = {"asia_00_08": (hr >= 0) & (hr < 8), "europe_07_13": (hr >= 7) & (hr < 13),
                 "us_13_21": (hr >= 13) & (hr < 21), "us_after_21_24": hr >= 21,
                 "not_us_21_13": (hr >= 21) | (hr < 13), "not_asia_08_24": hr >= 8,
                 "weekend_only": dow >= 5, "weekdays_only": dow < 5,
                 "us_weekdays_13_21": (hr >= 13) & (hr < 21) & (dow < 5)}
        for nm, m in rules.items():
            w = pd.Series(m.astype(float), index=r.index)
            nr = net(w, r, cst(s))
            daily = (1 + nr).groupby(nr.index.normalize()).prod() - 1
            record("1_timing", f"{s}|{nm}", daily, bh, s)
    pd.DataFrame(out).to_csv(os.path.join(RES, "study21_hour_means.csv"), index=False)


# ------------------------------------------------------------------ 2. time-series trend
def test_trend(px):
    for s in ["BTC/USD", "ETH/USD"]:
        p = px[s].dropna()
        r = p.pct_change().fillna(0)
        rv = r.rolling(20).std() * np.sqrt(365)
        sigs = {}
        for L in [5, 10, 20, 30]:
            sigs[f"sma{L}"] = (p > p.rolling(L).mean()).astype(float)
        for L in [1, 3, 5, 10, 20, 30]:
            sigs[f"mom{L}"] = (p.pct_change(L) > 0).astype(float)
        for L in [10, 20, 30]:
            hi, lo = p.rolling(L).max().shift(1), p.rolling(max(L // 2, 2)).min().shift(1)
            st = pd.Series(np.nan, index=p.index)
            st[p > hi] = 1.0
            st[p < lo] = 0.0
            sigs[f"brk{L}"] = st.ffill().fillna(0)
        for nm, sg in sigs.items():
            for vt in [None, 0.4, 0.6]:
                w = sg if vt is None else sg * (vt / rv).clip(upper=1.0)
                if vt is not None:  # rebalance vol-scaled weight only when it moves > 10%
                    w = w.round(1)
                nr = net(w.shift(1), r, cst(s))
                record("2_trend", f"{s}|{nm}|vt{vt}", nr, r, s)


# ------------------------------------------------------------------ 3. cross-sectional
def test_xs(px, dv):
    r = px.pct_change(fill_method=None)
    wk = px.index[px.index.dayofweek == 0]  # rebalance Mondays 00:00 UTC (using closes to Sunday)
    ew = r.where(px.shift(30).notna()).mean(axis=1).fillna(0)  # equal-weight daily-rebalanced basket (no cost)
    btc = r["BTC/USD"].fillna(0)
    adv = dv.rolling(30).median()
    costs = pd.Series({s: cst(s) for s in px.columns})
    for L in [7, 14, 28, 84]:
        mom = px / px.shift(L) - 1
        for k in [3, 5]:
            for side in ["mom", "rev"]:
                rows = {}
                for d in wk:
                    prev = d - pd.Timedelta(days=1)
                    if prev not in px.index:
                        continue
                    ok = px.loc[:prev].iloc[-60:].notna().all() & (adv.loc[prev] > 0)
                    liq = adv.loc[prev][ok].nlargest(15).index  # 15 most liquid on Alpaca (30d median $vol)
                    m = mom.loc[prev, liq].dropna()
                    if len(m) < 2 * k:
                        continue
                    pick = m.nlargest(k).index if side == "mom" else m.nsmallest(k).index
                    rows[d] = pd.Series(1.0 / k, index=pick)
                W = pd.DataFrame(rows).T.reindex(columns=px.columns).fillna(0.0)
                W = W.reindex(px.index).ffill().fillna(0.0)
                # weights drift within week: approximate with fixed weights (daily-rebalanced), cost at rebalance
                gross = (W * r.fillna(0)).sum(axis=1)
                tc = (W.diff().abs().fillna(W.abs()) * costs).sum(axis=1)
                nr = gross - tc
                nr = nr[nr.index >= "2021-03-01"]
                record("3_xsec", f"{side}{L}d_top{k}|vsBTC", nr, btc, "BTC/USD")
                ROWS.append(dict(test="3_xsec_ewbench", variant=f"{side}{L}d_top{k}", period="all",
                                 bench="EW basket", **stats(nr), bh_sharpe=stats(ew[nr.index])["sharpe"],
                                 bh_ann_ret=stats(ew[nr.index])["ann_ret"], bh_maxdd=stats(ew[nr.index])["maxdd"],
                                 turnover_yr=W.diff().abs().sum(axis=1).sum() / (len(W) / 365)))


# ------------------------------------------------------------------ 4. lead-lag with SPY
def test_leadlag(h, px):
    import alpaca_data as A
    m = A.read("m1", tickers=["SPY"], columns=["ts", "ticker", "o", "c"])
    m = m[m.ts >= "2021-01-01"].set_index("ts").sort_index()
    day = m.index.normalize()
    g = m.groupby(day)
    tm = m.index.time

    def at(t, col):
        x = m[tm == pd.Timestamp(t).time()][col]
        x.index = x.index.normalize()
        return x
    spy = pd.DataFrame({"p0700": at("07:00", "o"), "p0930": at("09:30", "o"), "p1545": at("15:45", "o"),
                        "p1600": at("15:59", "c"), "p1030": at("10:30", "o")})
    spy = spy.dropna()
    del m, g
    spy["last15"] = spy.p1600 / spy.p1545 - 1
    spy["gap"] = spy.p0930 / spy.p1600.shift(1) - 1
    spy["pre_to_open"] = spy.p0930 / spy.p0700 - 1
    spy["oc"] = spy.p1600 / spy.p0930 - 1
    spy["first60"] = spy.p1030 / spy.p0930 - 1
    out = []
    for s in ["BTC/USD", "ETH/USD"]:
        x = h[h.symbol == s].set_index("t").sort_index()
        et = x.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
        o = pd.Series(x.o.values, index=et)
        o = o[~o.index.duplicated()]
        # crypto prices at bar starts (ET): 16:00 (close) and 07:00 / 09:00 next morning
        def px_at(hh):
            y = o[o.index.hour == hh]
            y.index = y.index.normalize()
            return y
        c16, c07, c09 = px_at(16), px_at(7), px_at(9)
        df = spy.copy()
        df["cr_on"] = c09.reindex(df.index) / c16.reindex(df.index).shift(1) - 1  # prev 16:00 -> 09:00 today
        df["cr_on07"] = c07.reindex(df.index) / c16.reindex(df.index).shift(1) - 1
        df["cr_next_on"] = df.cr_on.shift(-1)  # tonight's crypto move after today's SPY close
        for p, (a, b) in PERIODS.items():
            d = df[a:b].dropna()
            for xv, yv in [("last15", "cr_next_on"), ("cr_on", "gap"), ("cr_on07", "pre_to_open"),
                           ("cr_on", "oc"), ("cr_on", "first60")]:
                out.append(dict(symbol=s, period=p, x=xv, y=yv, corr=d[xv].corr(d[yv]), n=len(d)))
        # tradable rules, returns booked on the SPY trading date, spread to daily UTC by date (approx)
        c = cst(s)
        bh = r_bh = px[s].pct_change().fillna(0)
        rules = {
            "crypto_on_if_spy_last15_up": ((df.last15 > 0).astype(float), df.cr_next_on, c),
            "crypto_on_if_spy_last15_down": ((df.last15 < 0).astype(float), df.cr_next_on, c),
            "spy_pre0700_to_open_if_crypto_on_up": ((df.cr_on07 > 0).astype(float), df.pre_to_open, 0.0003),
            "spy_open_60min_if_crypto_on_up": ((df.cr_on > 0).astype(float), df.first60, 0.0002),
            "spy_open_60min_if_crypto_on_down": ((df.cr_on < 0).astype(float), df.first60, 0.0002),
        }
        for nm, (w, rr, cc) in rules.items():
            nr = (w * rr - 2 * cc * w).dropna()  # enter and exit each trade
            nr = nr.reindex(pd.date_range(nr.index.min(), nr.index.max(), freq="D")).fillna(0)
            bench = bh if "crypto" in nm.split("_")[0] else spy.p1600.reindex(nr.index).ffill().pct_change().fillna(0)
            record("4_leadlag", f"{s}|{nm}", nr, bench, s if nm.startswith("crypto") else "SPY")
    pd.DataFrame(out).to_csv(os.path.join(RES, "study21_leadlag_corr.csv"), index=False)


def main():
    px, dv, h = load()
    px = px.drop(columns=[c for c in ["USDC/USD", "USDT/USD", "USDG/USD", "PAXG/USD"] if c in px], errors="ignore")
    test_tod(h)
    test_trend(px)
    test_xs(px, dv)
    test_leadlag(h, px)
    res = pd.DataFrame(ROWS)
    nvar = len(SERIES)
    dev = res[(res.period == "dev") & res.test.str.match(r"^\d_[a-z]+$")]
    best = dev.sort_values("sharpe", ascending=False).iloc[0]
    s = SERIES[(best.test, best.variant)]
    s = s[s.index >= "2021-01-01"]
    sr = s.mean() / s.std()
    dsr = deflated_sharpe(sr, nvar + 700, len(s), skew=float(s.skew()), kurt=float(s.kurt()) + 3)
    res.to_csv(os.path.join(RES, "study21_results.csv"), index=False)
    summ = pd.DataFrame([dict(n_variants=nvar, best_test=best.test, best_variant=best.variant,
                              best_dev_sharpe=best.sharpe, full_sharpe_ann=sr * np.sqrt(365), dsr=dsr)])
    summ.to_csv(os.path.join(RES, "study21_summary.csv"), index=False)
    print(summ.T.to_string())
    pd.set_option("display.width", 250)
    for t, g in res.groupby("test"):
        piv = g.pivot_table(index="variant", columns="period", values=["sharpe", "bh_sharpe"])
        print("\n==", t)
        print(piv.round(2).to_string())
    print("costs per side:", {k: round(v * 1e4, 1) for k, v in COST.items() if k in ("BTC/USD", "ETH/USD", "SOL/USD")})


if __name__ == "__main__":
    main()
