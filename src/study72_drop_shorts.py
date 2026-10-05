"""Study 72: shorting isolated no-news drops (pre-registered re-test of the study-44 post hoc finding).

Background: study 44 (src/study44_drop_reversal.py, results/study44_drop_reversal.csv) found AFTER looking at the data
that liquid stocks with a no-news close-to-close drop below -10% on a day with 3 or fewer such drops keep falling for
2-5 days in 2024-26 (excess -390 to -700 bp, t -2.3 to -3.5, 158 events) but much less in 2020-23. This study fixes
every choice below BEFORE the first run (written 2026-10-05, nothing tuned afterwards; any later change would be
reported as a new variant) and adds the frictions a real short book faces.

PRE-REGISTERED DESIGN
Universe on day t (known before the entry): traded price of t-1 > $5 and 20-day median dollar volume through t-1 >
$20M (src/horizon_lib.py TPX / ADV, both lagged one day); close of t-1 and t present; no split dated t-5 .. t+6
(data/local/events/splits_yf.csv, same window as study 44); not a bad print: where Alpaca's traded close exists
(2023-12 ..) its t-1 -> t return must agree with Yahoo's within 5 points (study 44 rule).
Drop measure: d_t = price at 15:45 of t (close of the 15:40-15:45 SIP bar, study8_exec_retest.price_1545('none'),
m5snap + m5snapx, 2024-01 .. ) / Yahoo split-adjusted close of t-1 - 1 where the 15:45 price exists; otherwise
(2020-23, and stock-days without a bar) the close-to-close return r_t (signal at the close: an optimistic timing for
those rows). Data check on the 15:45 rows only: the 15:55-16:00 bar close must be within 15% of the official close,
else the row falls back to r_t (catches unit mismatches; uses same-day data only for validity).
Event (primary): d_t <= -10%; isolated = at most 3 universe stocks with d_t <= -10% that day (counted before the news
filter, as in study 44); no news = no Benzinga article for the symbol with timestamp in (16:00 of t-1, 15:45 of t]
(news_features.NF.load_news) and no earnings calendar date (store 'earnings') on t-1 or t (report times are mostly
missing, so both days are excluded: a t-1 after-close report falls in the window, a t report is either in the
window or due after the close).
Borrow: easy-to-borrow only (Alpaca's CURRENT list, data/local/alpaca_assets_active.parquet, easy_to_borrow and
shortable: today's list flatters the past), fee 0. Variant 'all_fee': every name, 0 fee for easy-to-borrow and 30%/yr
(0.30/252 per trading day held) for the rest.
Entry variants:
  A  short in the closing auction of t, SSR ignored (upper bound: an event with d_t <= -10% has triggered Rule 201,
     which forbids shorts at or below the national best bid for the rest of t and all of t+1)
  A_ssr  A with the SSR approximation: no entry if the low of t is 10%+ below the close of t-1 (blocks nearly all)
  B  short in the closing auction of t+1 (next day's close; SSR is still in force on t+1, so a real order would
     have to be a short limit order priced above the bid; assumed filled at the official close: optimistic)
  C  short in the closing auction of t+2 (first close free of the t-day SSR, unless t+1 also fell 10%)
Exits: cover in the closing auction 2, 3 and 5 trading days after the entry day.
Book: at most 10 names, 10% of capital each (rest cash), filled from that day's eligible events, largest drop first,
a name already held is not added again; daily P&L marked close to close on the adjusted panel (-0.10 x return per
name); costs per side core.exec_cost_bps(P, 'auction') on traded prices + 2.5 bp (s6162_common.auction_cost_bps).
PRIMARY test (decides the verdict): entry B, exit after 3 days, easy-to-borrow only, events as above.
Verdict rule (fixed in advance): Promising = net bp per trade > 0 in both 2020-23 and 2024-26, all-period
date-clustered t > 2 and 2024-26 book Sharpe > 0.5; Candidate = all-period net > 0 with t > 1.5, or 2024-26 alone
significant (t > 2) with 2020-23 > 0; Reject otherwise. Entry A is reported for comparison with study 44 only.
POST HOC (added after the first run showed the 'all_fee' book far above the easy-to-borrow book): borrow variant
'htb_fee' = only names NOT on the current easy-to-borrow list, 30%/yr fee, primary events; and the break-even borrow
fee (annualized) at which its net per trade is zero. Not part of the verdict rule.
Sensitivity (secondary, reported, not used for the verdict): threshold -8% / -12%; isolation <= 5 / no cap;
news allowed (all drops).
Report by year 2020 .. 2026 (2026 to 2026-09-25, end of the news archive) and 2020-23 / 2024-26 / all: trades, net bp
per trade (trade return incl. costs and fee), date-clustered t, book net bp per day, Sharpe, max drawdown, annual
return; SPY buy-and-hold annual return and Sharpe on the same days; correlation of the book with SPY.
Survivorship: the panel holds only tickers alive in 2026; names that collapsed and delisted after a drop are missing,
which biases this short book DOWN (conservative). The 2020-23 drop measure is close-to-close (timing optimistic).
Output: results/study72_drop_shorts.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store
from core import load_panel, stock_cols, ann_stats, traded_close, RES, DATA
from s6162_common import with_traded_price, auction_cost_bps
import horizon_lib as HL

END = pd.Timestamp("2026-09-25")
FEE_HTB = 0.30 / 252
YEARS = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2020, 2027)]
PER = YEARS + [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31"),
               ("all", "2020-01-01", "2026-12-31")]

P = load_panel()
cols = stock_cols(P)
alld = P["c"].index
c = P["c"][cols].astype("float64")
low = P["l"][cols].astype("float64")
rawc = P["rawc"][cols].astype("float64")
r = c / c.shift(1) - 1
univ = (HL.TPX.shift(1) > 5) & (HL.ADV.shift(1) > 20e6) & c.notna() & c.shift(1).notna()
univ = univ.mul(pd.Series((alld >= "2020-01-01") & (alld <= END), index=alld), axis=0).astype(bool)
spl = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
ci = {t: i for i, t in enumerate(cols)}
ns = np.zeros(c.shape, dtype=bool)
for t, d in zip(spl.ticker, spl.date):
    i = alld.searchsorted(d)
    ns[max(0, i - 5): i + 2, ci[t]] = True
univ &= ~pd.DataFrame(ns, index=alld, columns=cols)
TC = traded_close(P)[cols]
r_alp = TC / TC.shift(1) - 1
univ &= ~(r_alp.notna() & r.notna() & ((r_alp - r).abs() > 0.05))
COST = (auction_cost_bps(with_traded_price(P))[cols].astype("float64") / 1e4).fillna(0.01)
a_ = pd.read_parquet(os.path.join(DATA, "local", "alpaca_assets_active.parquet"))
ETB = set(a_.symbol[a_.easy_to_borrow.astype(bool) & a_.shortable.astype(bool)].str.replace(".", "-", regex=False))
etb_row = np.array([t in ETB for t in cols])


def drop_measure():
    from study8_exec_retest import price_1545
    p45 = price_1545("none").reindex(index=alld, columns=cols).astype("float64")
    p55 = price_1545("none", key="c15:55").reindex(index=alld, columns=cols).astype("float64")
    okp = p45.notna() & (np.log(p55 / rawc).abs() < np.log(1.15))
    d45 = p45 / rawc.shift(1) - 1
    d = r.where(~okp, d45)
    print("universe stock-days 2024+: share with a 15:45 price", float(okp[univ].loc["2024":].mean().mean()),
          flush=True)
    return d, okp


def news_mask():
    import news_features as NF
    d = NF.load_news()[["ts", "sym"]]
    d = d[d.sym.isin(cols)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).values
    cut = pd.DatetimeIndex(alld) + pd.Timedelta(hours=15, minutes=45)
    start = pd.DatetimeIndex(np.r_[[pd.Timestamp("1990-01-01")],
                                   (pd.DatetimeIndex(alld[:-1]) + pd.Timedelta(hours=16)).values])
    i = cut.searchsorted(ts, side="left")
    ok = i < len(alld)
    i = np.where(ok, i, 0)
    ok &= ts > start[i].values
    M = np.zeros(c.shape, dtype=bool)
    M[i[ok], pd.Series(d.sym.values[ok]).map(ci).values] = True
    E = store.read("earnings")[["symbol", "date"]]
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E = E[E.date.isin(alld) & E.symbol.isin(cols)]
    k = alld.searchsorted(E.date.values)
    t_ = E.symbol.map(ci).values
    M[k, t_] = True
    m1 = k + 1 < len(alld)
    M[k[m1] + 1, t_[m1]] = True
    return pd.DataFrame(M, index=alld, columns=cols)


def short_book(E, key, lag, h, fee_all):
    """E: event mask on signal day t; entry at close t+lag, cover at close t+lag+h. Returns (daily, trades)."""
    Ev, K, cv, Cv = E.values, key.values, c.values, COST.values
    n = len(alld)
    gross, cost, wsum = np.zeros(n), np.zeros(n), np.zeros(n)
    held, trades = {}, []
    for t in range(n):
        for j, (ex, e0) in list(held.items()):           # P&L of day t for names held since the close of t-1
            rr = cv[t, j] / cv[t - 1, j] - 1
            gross[t] -= 0.10 * (0.0 if np.isnan(rr) else rr)
            wsum[t] += 0.10
            if fee_all and not etb_row[j]:
                cost[t] += 0.10 * FEE_HTB
            if ex == t:
                cost[t] += 0.10 * Cv[t, j]
                del held[j]
        s = t - lag                                      # signal day whose entry is today's close
        if s < 0 or alld[s] > END or t + h >= n:
            continue
        cand = np.where(Ev[s])[0]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(K[s, cand])]
        for j in cand:
            if len(held) >= 10:
                break
            if j in held or np.isnan(cv[t, j]):
                continue
            held[j] = (t + h, t)
            cost[t] += 0.10 * Cv[t, j]
            px = cv[t + 1: t + h + 1, j]
            ret = (px[-1] if not np.isnan(px[-1]) else px[~np.isnan(px)][-1] if (~np.isnan(px)).any() else cv[t, j]) \
                / cv[t, j] - 1
            fee = h * FEE_HTB if (fee_all and not etb_row[j]) else 0.0
            trades.append(dict(signal=alld[s], entry=alld[t], ticker=cols[j], drop=K[s, j], gross=-ret,
                               net=-ret - Cv[t, j] - Cv[min(t + h, n - 1), j] - fee))
    daily = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "wsum": wsum}, index=alld)
    return daily, pd.DataFrame(trades)


def cl_t(tr):
    if len(tr) < 5:
        return np.nan
    x = tr.groupby("entry").net.mean()
    return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan


def main():
    d, okp = drop_measure()
    NM = news_mask()
    spy = P["c"]["SPY"].pct_change()
    rows = []
    specs = {"primary": (-0.10, 3, True), "thr-8%": (-0.08, 3, True), "thr-12%": (-0.12, 3, True),
             "iso<=5": (-0.10, 5, True), "iso_none": (-0.10, None, True), "news_allowed": (-0.10, 3, False)}
    ssr = (low / c.shift(1) - 1) <= -0.10
    for sname, (thr, iso, nonews) in specs.items():
        base = univ & (d <= thr) & d.notna()
        cnt = base.sum(axis=1)
        ev = base.copy()
        if iso is not None:
            ev &= (cnt <= iso).values[:, None]
        if nonews:
            ev &= ~NM
        nev = int(ev.loc[:END].values.sum())
        print(sname, "events", nev, "of which ETB", int(ev.loc[:END].values[:, etb_row].sum()),
              "with 15:45 measure", int((ev & okp).values.sum()), flush=True)
        entries = {"A": (ev, 0), "A_ssr": (ev & ~ssr, 0), "B": (ev, 1), "C": (ev, 2)}
        if sname != "primary":
            entries = {k: v for k, v in entries.items() if k in ("A", "B")}
        for en, (E, lag) in entries.items():
            for borrow in ["etb", "all_fee", "htb_fee"]:
                if borrow != "etb" and sname != "primary":
                    continue
                Eb = E & (etb_row[None, :] if borrow == "etb" else ~etb_row[None, :] if borrow == "htb_fee" else True)
                for h in [2, 3, 5]:
                    daily, tr = short_book(Eb, d.where(Eb), lag, h, borrow != "etb")
                    for per, a, b in PER:
                        bk = daily.loc[a:b].loc[:END + pd.Timedelta(days=10)]
                        x = tr[(tr.signal >= a) & (tr.signal <= b)] if len(tr) else tr
                        row = dict(spec=sname, entry=en, borrow=borrow, hold=h, period=per, n_events=int(
                            E.loc[a:b].loc[:END].values.sum()), n_trades=len(x))
                        if len(x):
                            st = ann_stats(bk.net)
                            sp = spy.reindex(bk.index)
                            row.update(net_bp_trade=1e4 * x.net.mean(), gross_bp_trade=1e4 * x.gross.mean(),
                                       med_bp_trade=1e4 * x.net.median(), hit=(x.net > 0).mean(), t_clust=cl_t(x),
                                       book_net_bp_day=1e4 * bk.net.mean(), book_names=10 * bk.wsum.mean(),
                                       sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                                       corr_spy=bk.net.corr(sp) if bk.net.std() > 0 else np.nan,
                                       spy_ann=ann_stats(sp)["ann_ret"], spy_sharpe=ann_stats(sp)["sharpe"])
                            if borrow == "htb_fee":     # extra annual fee that would make the mean trade zero
                                row["breakeven_fee_ann"] = 0.30 + x.net.mean() / h * 252
                        rows.append(row)
                    if sname == "primary" and en == "B" and borrow == "etb" and h == 3 and len(tr):
                        print("primary trades, worst / best:\n",
                              tr.sort_values("net").iloc[np.r_[0:5, -5:0]].to_string(), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study72_drop_shorts.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pr = res[(res.spec == "primary") & (res.borrow == "etb")]
    print(pr.pivot_table(index=["entry", "hold"], columns="period", values="net_bp_trade").round(0).to_string())
    print(pr.pivot_table(index=["entry", "hold"], columns="period", values="n_trades").to_string())
    print(pr.pivot_table(index=["entry", "hold"], columns="period", values="sharpe").round(2).to_string())
    print(pr[pr.period.isin(["2020-23", "2024-26", "all"])][
        ["entry", "hold", "period", "n_trades", "net_bp_trade", "t_clust", "sharpe", "maxdd", "ann_ret", "corr_spy",
         "spy_ann", "spy_sharpe"]].round(3).to_string())
    hb = res[(res.borrow == "htb_fee") & res.period.isin(["2020-23", "2024-26", "all"])]
    print(hb[["entry", "hold", "period", "n_trades", "net_bp_trade", "t_clust", "sharpe", "maxdd",
              "breakeven_fee_ann"]].round(3).to_string())
    o2 = res[(res.period.isin(["2020-23", "2024-26", "all"])) & ~((res.spec == "primary") & (res.borrow == "etb"))]
    print(o2[["spec", "entry", "borrow", "hold", "period", "n_trades", "net_bp_trade", "t_clust", "sharpe",
              "maxdd"]].round(3).to_string())


if __name__ == "__main__":
    main()
