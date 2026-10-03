"""Study 44: multi-day reversal after large one-day drops, with and without news.

Question: do liquid stocks that fall hard on day t rebound over the next 1-5 days, and is the rebound limited to drops
without news (liquidity-driven selling) as the literature suggests? Is a 10-name book tradable after costs?

Events: day t, liquid universe (traded price of t-1 > $5: Yahoo raw close times later splits, s6162_common.traded_price;
20-day median dollar volume through t-1 > $20M), close-to-close return r_t (adjusted) below -5%, -10%, -15%, or a
z-score log(1 + r_t) / sd(20 daily log returns through t-1) below -3. Ticker-days with a split dated t-1 .. t+5
(splits_yf.csv) are dropped, and so are events whose Yahoo return disagrees by more than 5 points with the Alpaca
traded-close return (d1raw, 2023-12 ..) where both exist (bad prints).
News split: 'news' = >= 1 Benzinga article (news_features.NF.load_news) for the symbol with timestamp in
(16:00 of t-1, 15:45 of t], or an earnings date (store 'earnings') on t-1 or t (report times are mostly missing, so an
after-close report on t is counted as news too); 'nonews' = neither; 'all' = both.
Entry: closing auction of t. The signal uses the official close; in practice it would be computed at 15:45 from the
15:45 price (close enough for -5% and larger moves, but some names cross the threshold only in the last 15 minutes
and some fall back above it). Exits: open of t+1 (overnight only), close of t+1, t+2, t+3, t+5.
Statistics: excess return = event return - equal-weight average of the same return over the liquid universe of day t
(the panel holds only tickers alive in 2026: survivorship flatters buying losers; the universe average carries the same
bias but not fully, since crashed-then-delisted losers are exactly what is missing).
  excess_bp: event-weighted mean; excess_datew_bp: mean of entry-date averages (each date counts once; the t-stats
           below refer to this one); excess_trim_bp: event mean with 1%/99% winsorizing.
  ev_t   : t of the mean excess on entry-date averages (dates clustered), Newey-West with h-1 lags for overlap.
  cal_t  : calendar-time daily portfolio: each day, the average daily excess return of all events still held
           (entered in the last h days); t of that daily series. cal_bp = its mean in bp per day.
Tradable book: up to 10 names, 10% of capital each (rest cash), filled at the close of t from that day's events,
largest drop first (most negative z for the sigma rule), held until the exit; daily P&L on closes (overnight exit:
close t -> open t+1); costs per side = core.exec_cost_bps(P, "auction") on traded prices + 2.5 bp at entry and exit.
Reported: net bp per day, Sharpe, max drawdown, and 'hedged' (net minus the gross weight times the universe average
daily return), against SPY buy and hold.
Short book (post hoc, added after seeing continuation): the same slots shorted (no borrow fee, no easy-to-borrow
filter: an upper bound). Groups for r < -10% only: isolated (<= 3 such drops in the universe that day; isolated_nonews
also without news) vs crowded (> 10 that day, market sell-offs); the 3/10 cut-offs were chosen after the first run.
Overlap: share of events that are in the overnight blend's top 10 on the same evening (study 68 construction, 2024+).
Periods: 2020-23, 2024-26, 2024H1-25H1, 2025H2-26 (to 2026-09-25, end of the news archive).
Output: results/study44_drop_reversal.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store
import bt
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA
from s6162_common import with_traded_price, auction_cost_bps

END = pd.Timestamp("2026-09-25")
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30"),
       ("2024H1-25H1", "2024-01-01", "2025-06-30"), ("2025H2-26", "2025-07-01", "2026-09-30")]
EXITS = {"on": 0, "c1": 1, "c2": 2, "c3": 3, "c5": 5}      # 'on' = open of t+1

P = load_panel()
PT = with_traded_price(P)
cols = stock_cols(P)
alld = P["c"].index
o, c = P["o"][cols].astype("float64"), P["c"][cols].astype("float64")
TP = PT["rawc"][cols]
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
r = c / c.shift(1) - 1
lr = np.log(c / c.shift(1))
vol = lr.rolling(20, min_periods=15).std().shift(1)
z = np.log1p(r) / vol
univ = (TP.shift(1) > 5) & (adv > 20e6) & c.notna() & c.shift(1).notna()
univ = univ.mul(pd.Series(alld <= END, index=alld), axis=0).astype(bool)
# splits t-1 .. t+5
spl = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
nearsplit = np.zeros(c.shape, dtype=bool)
ci = {t: i for i, t in enumerate(cols)}
for t, d in zip(spl.ticker, spl.date):
    i = alld.searchsorted(d)
    nearsplit[max(0, i - 5): i + 2, ci[t]] = True        # split on day i affects events dated i-5 .. i+1
univ &= ~pd.DataFrame(nearsplit, index=alld, columns=cols)
COST = auction_cost_bps(PT)[cols].astype("float64") / 1e4   # per side, decimal
# Alpaca traded close check (2023-12 ..)
TC = traded_close(P)[cols]
r_alp = TC / TC.shift(1) - 1
bad = (r_alp.notna() & r.notna() & ((r_alp - r).abs() > 0.05))


def news_mask():
    import news_features as NF
    d = NF.load_news()[["ts", "sym"]]
    d = d[d.sym.isin(cols)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).values
    cut = pd.DatetimeIndex(alld) + pd.Timedelta(hours=15, minutes=45)
    start = pd.DatetimeIndex(np.r_[[pd.Timestamp("1990-01-01")], (pd.DatetimeIndex(alld[:-1]) + pd.Timedelta(hours=16)).values])
    i = cut.searchsorted(ts, side="left")          # first day whose 15:45 cutoff >= ts
    ok = i < len(alld)
    i = np.where(ok, i, 0)
    ok &= ts > start[i].values
    M = np.zeros(c.shape, dtype=bool)
    ti = pd.Series(d.sym.values[ok]).map(ci).values
    M[i[ok], ti] = True
    E = store.read("earnings")[["symbol", "date"]]
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E = E[E.date.isin(alld) & E.symbol.isin(cols)]
    k = alld.searchsorted(E.date.values)
    t_ = E.symbol.map(ci).values
    M[k, t_] = True
    k1 = k + 1
    m1 = k1 < len(alld)
    M[k1[m1], t_[m1]] = True
    return pd.DataFrame(M, index=alld, columns=cols)


def nw_t(x, lags):
    x = x.dropna().values
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


def blend_W():
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < alld[-2]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    return bt.select_topk(S, S.notna(), 10) > 0


def slot_book(E, key, h):
    """E: event mask (date x ticker), key: ranking (lower = first). Returns daily DataFrame (gross, cost, net, wsum)."""
    Ev = E.values
    K = key.values
    cv, ov = c.values, o.values
    Cv = COST.values
    n = len(alld)
    gross = np.zeros(n)
    cost = np.zeros(n)
    wsum = np.zeros(n)
    held = {}                                       # ticker idx -> exit day idx
    for t in range(n - 1):
        # P&L of day t for positions held from t-1 (close-to-close), exits at close t
        if h > 0:
            for j, ex in list(held.items()):
                rr = cv[t, j] / cv[t - 1, j] - 1
                gross[t] += 0.10 * (0.0 if np.isnan(rr) else rr)
                wsum[t] += 0.10
                if ex == t:
                    cost[t] += 0.10 * (Cv[t, j] if not np.isnan(Cv[t, j]) else 0.01)
                    del held[j]
        if alld[t] > END:
            break
        cand = np.where(Ev[t])[0]
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(K[t, cand])]
        if h == 0:                                  # overnight: up to 10 names, close t -> open t+1
            for j in cand[:10]:
                rr = ov[t + 1, j] / cv[t, j] - 1
                gross[t + 1] += 0.10 * (0.0 if np.isnan(rr) else rr)
                wsum[t + 1] += 0.10
                cost[t + 1] += 0.10 * 2 * (Cv[t, j] if not np.isnan(Cv[t, j]) else 0.01)
            continue
        free = 10 - len(held)
        for j in cand:
            if free <= 0:
                break
            if j in held:
                continue
            held[j] = min(t + h, n - 1)
            cost[t] += 0.10 * (Cv[t, j] if not np.isnan(Cv[t, j]) else 0.01)
            free -= 1
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "wsum": wsum}, index=alld)


def main():
    NM = news_mask()
    rows = []
    ud = (r.where(univ.shift(1, fill_value=False)).mean(axis=1))          # universe avg daily return (members of d-1)
    D = (r.sub(ud, axis=0))                                               # daily excess
    u_on = (o.shift(-1) / c - 1).where(univ).mean(axis=1)
    spy = P["c"]["SPY"].pct_change()
    BW = blend_W().reindex(index=alld, columns=cols).fillna(False).astype(bool)
    thr = {"r<-5%": (r < -0.05, r), "r<-10%": (r < -0.10, r), "r<-15%": (r < -0.15, r), "z<-3": (z < -3, z)}
    nbad = int((univ & (r < -0.05) & bad).sum().sum())
    print("events r<-5% dropped by Alpaca/Yahoo disagreement (2023-12 ..):", nbad, flush=True)
    for tn, (sig, key) in thr.items():
        base = sig & univ & ~bad & key.notna()
        groups = [("all", base), ("nonews", base & ~NM), ("news", base & NM)]
        if tn == "r<-10%":                       # added after the first run (post hoc): crowded vs isolated drop days
            cnt = base.sum(axis=1).values[:, None]
            groups += [("isolated_nonews", base & ~NM & (cnt <= 3)), ("isolated_all", base & (cnt <= 3)),
                       ("crowded_all", base & (cnt > 10))]
        for nn, m in groups:
            m = m.fillna(False)
            for ex, h in EXITS.items():
                if h == 0:
                    R = o.shift(-1) / c - 1
                else:
                    R = c.shift(-h) / c - 1
                bench = R.where(univ).mean(axis=1)
                X = R.sub(bench, axis=0).where(m)
                # calendar-time daily portfolio
                if h == 0:
                    cal = X.mean(axis=1)                       # dated by entry
                    nact = m.sum(axis=1)
                else:
                    A = sum(m.shift(j, fill_value=False).astype("int16") for j in range(1, h + 1))
                    nact = A.sum(axis=1)
                    cal = (A * D.fillna(0)).sum(axis=1) / nact.replace(0, np.nan)
                book = slot_book(m, key.where(m), h)
                ub = (book.wsum * (u_on.shift(1) if h == 0 else ud)).fillna(0)
                for per, a_, b_ in PER:
                    x = X.loc[a_:b_].stack().dropna()
                    if len(x) < 10:
                        rows.append(dict(rule=tn, news=nn, exit=ex, period=per, n_events=len(x)))
                        continue
                    edate = x.groupby(level=0).mean()
                    rw = R.where(m).loc[a_:b_].stack().dropna()
                    cl = cal.loc[a_:b_].dropna()
                    bk = book.loc[a_:b_]
                    bk = bk.loc[:END]
                    st = ann_stats(bk.net)
                    sh = ann_stats(bk.net - ub.loc[bk.index])
                    row = dict(rule=tn, news=nn, exit=ex, period=per, n_events=len(x), n_dates=len(edate),
                               raw_bp=1e4 * rw.mean(), excess_bp=1e4 * x.mean(), excess_med_bp=1e4 * x.median(),
                               excess_trim_bp=1e4 * x.clip(x.quantile(.01), x.quantile(.99)).mean(),
                               excess_datew_bp=1e4 * edate.mean(),
                               hit=(x > 0).mean(), ev_t=nw_t(edate, max(h - 1, 0)),
                               cal_bp_day=1e4 * cl.mean(), cal_t=cl.mean() / cl.std() * np.sqrt(len(cl)) if len(cl) > 5 else np.nan,
                               book_names=10 * bk.wsum.mean(), book_gross_bp=1e4 * bk.gross.mean(),
                               book_cost_bp=1e4 * bk.cost.mean(), book_net_bp=1e4 * bk.net.mean(),
                               book_sharpe=st["sharpe"], book_maxdd=st["maxdd"], book_ann=st["ann_ret"],
                               hedged_net_bp=1e4 * (bk.net - ub.loc[bk.index]).mean(), hedged_sharpe=sh["sharpe"],
                               short_book_net_bp=1e4 * (-bk.gross - bk.cost).mean(),
                               short_book_sharpe=ann_stats(-bk.gross - bk.cost)["sharpe"],
                               short_book_maxdd=ann_stats(-bk.gross - bk.cost)["maxdd"],
                               spy_sharpe=ann_stats(spy.loc[a_:b_].loc[:END])["sharpe"])
                    if a_ >= "2024-01-01":
                        mm = m.loc[a_:b_]
                        bw = BW.loc[a_:b_]
                        row["share_in_blend"] = (mm & bw).sum().sum() / max(mm.sum().sum(), 1)
                        row["blend_picks_that_are_events"] = (mm & bw).sum().sum() / max(bw.sum().sum(), 1)
                    rows.append(row)
            print(tn, nn, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study44_drop_reversal.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    v = res[res.period.isin(["2020-23", "2024-26"])].pivot_table(
        index=["rule", "news", "exit"], columns="period", values=["n_events", "excess_datew_bp", "ev_t", "cal_t", "book_net_bp", "book_sharpe"])
    print(v.round(2).to_string())
    # largest events (data check)
    m = (r < -0.10) & univ & ~bad
    R = c.shift(-5) / c - 1
    x = R.where(m).stack().dropna().sort_values()
    print("\nlargest 5-day returns after a >10% drop:\n", pd.concat([x.tail(8), r.stack().reindex(x.tail(8).index).rename("r_t")], axis=1))
    print("\nworst:\n", pd.concat([x.head(5), r.stack().reindex(x.head(5).index).rename("r_t")], axis=1))


if __name__ == "__main__":
    main()
