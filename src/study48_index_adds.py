"""Study 48: S&P 500 additions and deletions at 1-5 day horizons (beyond study 17).

(a) Announced changes (data/local/events/sp500_events.csv from src/fetch_events_sp500.py: Wikipedia changes table,
    announcement date = the cited S&P press-release date; announcements come after the close). Day a0 = first
    trading day after the announcement date; e0 = effective date (index funds trade in the closing auction of e0-1).
    Windows per event (adjusted prices, total return):
      jump        close a0-1 -> open a0 (the announcement night; not tradable, reference only)
      o_c1..o_c5  opening auction of a0 (first auction after the announcement) -> close of a0+k-1 ("close +k")
      c_c1..c_c5  closing auction of a0 -> close of a0+k
      o_eff       open a0 -> closing auction of e0-1 (the index trade); its hold length is often > 5 days, so
                  'o_eff_le5' keeps only events with e0-1 <= a0+4 (inside the owner's 5-day scope)
      eff_c5      close e0-1 -> close e0+4 (after the index trade)
    Excess = window return - equal-weight mean of the same window over the liquid universe of a0-1 (traded price >
    $5, 20-day median dollar volume > $20M). Also vs SPY. Date-clustered t: events averaged per announcement date
    first (several changes are announced together), t over dates. Periods by announcement date: 2020-23 / 2024-26.
    Data checks: point-in-time market cap at a0-1 must exist (ticker identity; a recycled ticker or missing
    fundamentals drops the event from the 'checked' group), Yahoo vs Alpaca traded-close returns must agree within
    5 points on each day of the window (2023-12+), no split dated a0-1 .. a0+6. Largest moves are printed.
    Tradable: per-trade net return (minus auction cost + 2.5 bp per side) minus SPY over the same window, and a
    daily book (10 slots of 10% capital, cash otherwise): Sharpe, max drawdown. Deletions are traded short (no
    borrow fee: they are large caps, an optimistic assumption).
(b) Candidates before the announcement: membership by date reconstructed from the constituent list of the
    2026-08-10 Wikipedia revision (data/local/events/sp500_wiki_1.html) and the changes table (walked back and
    forward). Eligible non-member on the formation day: point-in-time market cap (fundamentals_panels.pkl,
    SEC filings, price of the previous day) and positive ttm_net_income, liquid universe, >= 252 trading days of
    price history (S&P's 12-month seasoning), not a second share class of a member (BRK-A next to BRK-B), no
    pending addition already announced. Ranked by market cap; top 5 / 10 / 20, and 'above_min' = all with a market
    cap above the 20th percentile of the caps of stocks added in the previous 2 years (a data-driven stand-in for
    S&P's published minimum, which changed several times).
    'fresh_*' also drops perennial names: top-20 candidates at each of the 4 previous quarterly dates and never
    added (MLPs, foreign domicile, low float, ...; past information only).
    Announcement days T: first Friday of Mar/Jun/Sep/Dec (quarterly rebalance; 'quarterly_rule', tradable) and
    'quarterly_actual' (the Friday on which that month's market-cap changes were announced when within 10 days of
    the rule date: hindsight on the date, 2 of 25 differ). Windows:
    pre5 close T-5 -> close T (formation at T-5), pre10 close T-10 -> close T, ann_night close T -> open T+1,
    post5 close T -> close T+5. Placebo: the same construction on every other Friday at least 10 trading days away
    from a quarterly announcement. Excess vs the liquid universe, t over announcement dates; hit rate = share of
    candidates announced as additions within T-2 .. T+5 calendar days (add_hit_rate; excess_hit_bp /
    excess_nonhit_bp split the candidates by it, which is known only afterwards).
    Tradable: buy the top N at the closing auction of T-5, sell at the closing auction of T (or the opening auction
    of T+1), net of costs, minus SPY over the window; Sharpe of the daily series (cash between events).
Output: results/study48_index_adds.csv
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA

EV = os.path.join(DATA, "local", "events")
END = pd.Timestamp("2026-09-30")
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31")]

P = load_panel()
cols = stock_cols(P)
alld = P["c"].index
N = len(alld)
ci = {t: i for i, t in enumerate(cols)}
O = P["o"][cols].values.astype("float64")
C = P["c"][cols].values.astype("float64")
SPYc = P["c"]["SPY"].values.astype("float64")
SPYo = P["o"]["SPY"].values.astype("float64")


def traded_price(P):
    """Yahoo raw close times splits dated after each day (as s6162_common.traded_price)."""
    s = pd.read_csv(os.path.join(EV, "splits_yf.csv"), parse_dates=["date"])
    fac = pd.DataFrame(1.0, index=alld, columns=cols)
    for t, g in s[s.ticker.isin(cols)].groupby("ticker"):
        f = np.ones(N)
        for d, r in zip(g.date, g.ratio):
            f[alld < d] *= r
        fac[t] = f
    return (P["rawc"][cols] * fac).astype("float32")


TP = traded_price(P)
Q = dict(P)
Q["rawc"] = TP.reindex(columns=P["c"].columns)
COST = ((exec_cost_bps(Q, "auction")[cols] + 2.5) / 1e4).values.astype("float64")    # per side, decimal
del Q
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
UNIV = ((TP.shift(1) > 5) & (adv > 20e6) & P["c"][cols].notna()).values
nhist = P["c"][cols].notna().cumsum().values
TC = traded_close(P)[cols]
r_y = P["c"][cols].pct_change().values
r_a = (TC / TC.shift(1) - 1).values
BADDAY = np.isfinite(r_a) & np.isfinite(r_y) & (np.abs(r_a - r_y) > 0.05)
del TC
spl = pd.read_csv(os.path.join(EV, "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
NEARSPLIT = np.zeros((N, len(cols)), bool)
for t, d in zip(spl.ticker, spl.date):
    i = alld.searchsorted(d)
    NEARSPLIT[max(0, i - 7): i + 2, ci[t]] = True        # event day a0 with a split in a0-1 .. a0+6

FP = pd.read_pickle(os.path.join(DATA, "local", "fundamentals_panels.pkl"))
MCAP = FP["market_cap"].reindex(index=alld, columns=cols).values.astype("float64")
NI = FP["ttm_net_income"].reindex(index=alld, columns=cols).values.astype("float64")
del FP


def px(i, kind):
    return (O if kind == "o" else C)[i]


def win(a, ka, b, kb):
    """vector over tickers: price(b, kb) / price(a, ka) - 1"""
    return px(b, kb) / px(a, ka) - 1


def spy_win(a, ka, b, kb):
    pa = SPYo[a] if ka == "o" else SPYc[a]
    pb = SPYo[b] if kb == "o" else SPYc[b]
    return pb / pa - 1


def bench(a, ka, b, kb, umask):
    r = win(a, ka, b, kb)
    m = umask & np.isfinite(r)
    return np.nanmean(np.clip(r[m], -0.9, 3.0)) if m.sum() > 50 else np.nan


def nw_t(x, lags=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n) if v > 0 else np.nan


def book(trades, slots=10, w=0.10):
    """trades: (a, ka, b, kb, j, side, prio). Enter at price (a, ka), exit at (b, kb); at most `slots` open
    positions (w of capital each). Daily net P&L series (costs at entry and exit, auction cost + 2.5 bp)."""
    tr = sorted(trades, key=lambda t: (t[0], t[1] == "c", -t[6]))
    pnl = np.zeros(N)
    busy = np.zeros(N + 1, int)
    acc = []
    for a, ka, b, kb, j, side, pr in tr:
        if not (0 <= a <= b < N) or not np.isfinite(px(a, ka)[j]) or not np.isfinite(px(b, kb)[j]):
            continue
        first = a if ka == "o" else a + 1
        last = b if kb == "c" else b - 1
        if busy[a:b + 1].max() >= slots:
            continue
        busy[a:b + 1] += 1
        acc.append((a, ka, b, kb, j))
        prev = px(a, ka)[j]
        for t in range(first, last + 1):
            x = C[t, j]
            if np.isfinite(x):
                pnl[t] += side * w * (x / prev - 1)
                prev = x
        if kb == "o":
            pnl[b] += side * w * (O[b, j] / prev - 1)
        ca, cb = COST[a, j], COST[b, j]
        pnl[a] -= w * (ca if np.isfinite(ca) else 0.01)
        pnl[b] -= w * (cb if np.isfinite(cb) else 0.01)
    return pd.Series(pnl, index=alld), acc


ROWS = []


def summarize(part, group, period, name, R, X, XS, dates, hold, net_spy=None, extra=None):
    """R raw, X excess vs universe, XS excess vs SPY (arrays per event), dates: cluster key per event."""
    m = np.isfinite(X)
    if m.sum() < 3:
        ROWS.append(dict(part=part, group=group, period=period, window=name, n=int(m.sum())))
        return
    df = pd.DataFrame({"d": np.asarray(dates)[m], "x": X[m], "xs": XS[m]})
    g = df.groupby("d").mean()
    row = dict(part=part, group=group, period=period, window=name, n=int(m.sum()), n_dates=len(g), hold_days=hold,
               raw_bp=1e4 * np.nanmean(R[m]), excess_bp=1e4 * X[m].mean(), excess_med_bp=1e4 * np.median(X[m]),
               excess_datew_bp=1e4 * g.x.mean(), t_date=nw_t(g.x.values), hit=float((X[m] > 0).mean()),
               vs_spy_bp=1e4 * XS[m].mean(), t_vs_spy=nw_t(g["xs"].values))
    if net_spy is not None:
        ns = net_spy[m]
        gn = pd.Series(ns, index=df.d).groupby(level=0).mean()
        row.update(net_vs_spy_bp=1e4 * np.nanmean(ns), t_net_vs_spy=nw_t(gn.values))
    if extra:
        row.update(extra)
    ROWS.append(row)


# ============================================================ membership by date
def membership():
    t1 = open(os.path.join(EV, "sp500_wiki_1.html")).read()
    tabs = pd.read_html(io.StringIO(t1))
    cur = set(tabs[0]["Symbol"].astype(str).str.replace(".", "-", regex=False))
    anchor = pd.Timestamp("2026-08-10")
    ev = pd.read_csv(os.path.join(EV, "sp500_events.csv"), parse_dates=["eff", "ann"])
    ev = ev[ev.eff.notna()].sort_values("eff")
    M = np.zeros((N, len(cols)), bool)
    # walk back from the anchor
    S = set(cur)
    back = ev[ev.eff <= anchor].sort_values("eff", ascending=False)
    fwd = ev[ev.eff > anchor].sort_values("eff")
    # members on day d (d < eff of a change => change not yet applied)
    chg_b = list(back.groupby("eff", sort=False))
    k = 0
    state_by_day = {}
    effs_b = [e for e, _ in chg_b]
    for i in range(N - 1, -1, -1):
        d = alld[i]
        if d > anchor:
            continue
        while k < len(chg_b) and effs_b[k] > d:
            _, g = chg_b[k]
            for _, r in g.iterrows():
                if r.side == "add":
                    S.discard(r.ticker)
                else:
                    S.add(r.ticker)
            k += 1
        state_by_day[i] = frozenset(S)
    S = set(cur)
    for i in range(N):
        d = alld[i]
        if d <= anchor:
            continue
        S2 = set(cur)
        for _, r in fwd[fwd.eff <= d].iterrows():
            if r.side == "add":
                S2.add(r.ticker)
            else:
                S2.discard(r.ticker)
        state_by_day[i] = frozenset(S2)
    for i, s in state_by_day.items():
        for t in s:
            if t in ci:
                M[i, ci[t]] = True
    return M, ev


MEM, EVALL = membership()
nm = MEM.sum(axis=1)
print("members in panel per day: min %d median %d max %d; at 2020-01-02: %d" %
      (nm.min(), np.median(nm), nm.max(), nm[alld.searchsorted(pd.Timestamp("2020-01-02"))]), flush=True)

# ============================================================ (a) announced changes
sp = EVALL[(EVALL.eff >= "2020-01-01") & EVALL.ann.notna() & EVALL.ticker.isin(cols)].copy()
sp["j"] = sp.ticker.map(ci)
sp["a0"] = alld.searchsorted(sp.ann, side="right")          # first trading day after the announcement day
sp["e0"] = alld.searchsorted(sp.eff, side="left")
sp = sp[(sp.e0 < N) & (sp.a0 + 6 < N) & (sp.a0 >= 1)].copy()
sp["e0"] = np.maximum(sp.e0, sp.a0 + 1)                     # effective before a0+1 cannot be traded
ok_data = []
for a, j in zip(sp.a0, sp.j):
    ok_data.append(np.isfinite(C[a - 1, j]) and np.isfinite(O[a, j]))
sp = sp[ok_data].copy()
sp["mcap"] = [MCAP[a - 1, j] for a, j in zip(sp.a0, sp.j)]
sp["bad"] = [bool(BADDAY[a - 1:a + 6, j].any()) for a, j in zip(sp.a0, sp.j)]
sp["split"] = [bool(NEARSPLIT[a, j]) for a, j in zip(sp.a0, sp.j)]
sp["nhist"] = [int(nhist[a - 1, j]) for a, j in zip(sp.a0, sp.j)]
# identity: a plausible point-in-time market cap, or (no fundamentals) at least 60 days of prior price history;
# spin-offs added on listing (CARR, OTIS, GEHC, ...) fail this and are left out of 'checked'
sp["checked"] = (((sp.mcap > 1e9) | (sp.mcap.isna() & (sp.nhist >= 60))) & ~sp.bad & ~sp.split)
sp["period"] = np.where(sp.ann < "2024-01-01", "2020-23", "2024-26")
print("announced events with data:", sp.groupby(["side", "period"]).size().to_dict(),
      "checked:", sp[sp.checked].groupby(["side", "period"]).size().to_dict(), flush=True)
print("dropped by checks:\n", sp[~sp.checked][["ticker", "side", "ann", "mcap", "nhist", "bad", "split"]].to_string(), flush=True)

WIN = {}
for k in range(1, 6):
    WIN[f"o_c{k}"] = (lambda a, e, k=k: (a, "o", a + k - 1, "c"))
for k in range(1, 6):
    WIN[f"c_c{k}"] = (lambda a, e, k=k: (a, "c", a + k, "c"))
WIN["jump"] = lambda a, e: (a - 1, "c", a, "o")
WIN["o_eff"] = lambda a, e: (a, "o", e - 1, "c")
WIN["eff_c5"] = lambda a, e: (e - 1, "c", e + 4, "c")

cache_b = {}
ev_rows = []
for _, r in sp.iterrows():
    rec = dict(ticker=r.ticker, side=r.side, ann=r.ann, a0=r.a0, e0=r.e0, j=r.j, period=r.period, checked=r.checked,
               mcap=r.mcap, mcap_change=r.mcap_change)
    um = UNIV[r.a0 - 1]
    for wn, f in WIN.items():
        a, ka, b, kb = f(r.a0, r.e0)
        if b >= N or b < a or (a == b and ka == kb):
            rec[wn] = np.nan
            continue
        key = (a, ka, b, kb)
        if key not in cache_b:
            cache_b[key] = (bench(a, ka, b, kb, um), spy_win(a, ka, b, kb))
        bm, sy = cache_b[key]
        x = px(b, kb)[r.j] / px(a, ka)[r.j] - 1
        c_ = COST[a, r.j] + COST[b, r.j]
        rec[wn] = x
        rec[wn + "_x"] = x - bm
        rec[wn + "_s"] = x - sy
        rec[wn + "_cost"] = c_
        rec[wn + "_hold"] = b - a + (1 if ka == "o" else 0) - (1 if kb == "o" else 0)
    ev_rows.append(rec)
E = pd.DataFrame(ev_rows)
E["o_eff_le5"] = E["o_eff"].where(E.e0 - 1 <= E.a0 + 4)
for s_ in ["_x", "_s", "_cost", "_hold"]:
    E["o_eff_le5" + s_] = E["o_eff" + s_].where(E.e0 - 1 <= E.a0 + 4)
print("effective-date lag (trading days a0 -> e0-1):", (E.e0 - 1 - E.a0).describe().round(1).to_dict(), flush=True)

for side in ["add", "del"]:
    sgn = 1 if side == "add" else -1
    for grp, gm in [("checked", E.checked), ("checked_mcapchange", E.checked & E.mcap_change.astype(bool)),
                    ("checked_other", E.checked & ~E.mcap_change.astype(bool)), ("all", E.checked | True)]:
        for per in ["2020-23", "2024-26", "all"]:
            x = E[(E.side == side) & gm & ((E.period == per) if per != "all" else True)]
            for wn in list(WIN) + ["o_eff_le5"]:
                if wn + "_x" not in x:
                    continue
                net_spy = sgn * x[wn + "_s"].values - x[wn + "_cost"].values      # trade direction: long adds, short dels
                summarize("a_announced", f"{side}_{grp}", per, wn, x[wn].values, x[wn + "_x"].values,
                          x[wn + "_s"].values, x.ann.values, float(np.nanmean(x[wn + "_hold"])), net_spy)

# largest moves (data check)
chk = E[E.checked].assign(o_c5=E.o_c5).sort_values("o_c5")
print("\nlargest |o_c5| moves (checked events):")
print(pd.concat([chk.head(5), chk.tail(5)])[["ticker", "side", "ann", "mcap", "jump", "o_c1", "o_c5", "o_c5_x"]].round(3).to_string())

# tradable books (checked events only)
spyd = pd.Series(SPYc, index=alld).pct_change()


def book_rows(name, trades, period_key=None):
    s, acc = book(trades)
    for per, a_, b_ in PER:
        x = s.loc[a_:b_].loc[:END]
        st = ann_stats(x)
        ROWS.append(dict(part="a_book" if name.startswith("a_") else "b_book", group=name, period=per, window="book",
                         n=sum(1 for t in acc if a_ <= str(alld[t[0]].date()) <= b_), book_net_bp_day=1e4 * x.mean(),
                         book_sharpe=st["sharpe"], book_maxdd=st["maxdd"], book_ann=st["ann_ret"],
                         invested=float((x != 0).mean()), spy_sharpe=ann_stats(spyd.loc[a_:b_].loc[:END])["sharpe"]))
        print(f"{name:34s} {per}: n={ROWS[-1]['n']:3d} SR={st['sharpe']:5.2f} dd={st['maxdd']:6.3f} "
              f"bp/day={1e4 * x.mean():5.2f}", flush=True)


Ea = E[E.checked & (E.side == "add")]
Ed = E[E.checked & (E.side == "del")]
book_rows("a_add_open_to_c1", [(a, "o", a, "c", j, 1, 0) for a, j in zip(Ea.a0, Ea.j)])
book_rows("a_add_open_to_c5", [(a, "o", a + 4, "c", j, 1, 0) for a, j in zip(Ea.a0, Ea.j)])
book_rows("a_add_open_to_eff_or_c5", [(a, "o", min(e - 1, a + 4), "c", j, 1, 0) for a, e, j in zip(Ea.a0, Ea.e0, Ea.j)])
book_rows("a_add_close_to_c5", [(a, "c", a + 5, "c", j, 1, 0) for a, j in zip(Ea.a0, Ea.j)])
book_rows("a_del_short_open_to_c5", [(a, "o", a + 4, "c", j, -1, 0) for a, j in zip(Ed.a0, Ed.j)])
book_rows("a_del_buy_eff-1_to_eff+4", [(e - 1, "c", e + 4, "c", j, 1, 0) for e, j in zip(Ed.e0, Ed.j) if e + 4 < N])

# ============================================================ (b) candidates before the announcement
# quarterly announcement Fridays
qf = []
for y in range(2020, 2027):
    for mo in [3, 6, 9, 12]:
        d = pd.Timestamp(year=y, month=mo, day=1)
        while d.dayofweek != 4:
            d += pd.Timedelta(days=1)
        if d <= END - pd.Timedelta(days=10):
            qf.append(d)
# check against announced dates
annd = set(EVALL.ann.dropna().dt.normalize())
print("\nquarterly Fridays with an announcement within 0-3 days:",
      sum(any((x - d).days in (0, 1, 2, 3) for x in annd) for d in qf), "of", len(qf), flush=True)
for d in qf:
    near = sorted({x.date() for x in annd if -7 <= (x - d).days <= 7})
    if not any(x == d.date() for x in near):
        print("  ", d.date(), "announcements nearby:", near)

# add caps for the data-driven minimum
adds = sp[(sp.side == "add") & sp.mcap.notna()][["ann", "mcap"]]
base_cls = {}
for t in cols:
    base_cls.setdefault(t.split("-")[0], []).append(ci[t])


def candidates(T_i, f_i):
    """eligible non-members on formation day f_i (signal known at the close of f_i), ranked by market cap."""
    mc, ni = MCAP[f_i], NI[f_i]
    ok = UNIV[f_i] & ~MEM[f_i] & np.isfinite(mc) & (ni > 0) & (nhist[f_i] >= 252)
    # second share class of a member
    memb = MEM[f_i]
    for b, js in base_cls.items():
        if len(js) > 1 and memb[js].any():
            ok[js] = False
    # pending additions announced before the formation close
    d_f = alld[f_i]
    pend = EVALL[(EVALL.side == "add") & (EVALL.ann < d_f) & (EVALL.eff > d_f)]
    for t in pend.ticker:
        if t in ci:
            ok[ci[t]] = False
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-mc[idx])]
    a2 = adds[(adds.ann < d_f) & (adds.ann >= d_f - pd.Timedelta(days=730))]
    thr = np.nanpercentile(a2.mcap, 20) if len(a2) >= 10 else np.nan
    return idx, thr


def added_soon(T_d, j, lo=-2, hi=5):
    t = cols[j]
    x = EVALL[(EVALL.side == "add") & (EVALL.ticker == t) & EVALL.ann.notna()]
    return bool(((x.ann - T_d).dt.days.between(lo, hi)).any())


# 'actual' quarterly dates: the Friday on which that month's market-cap changes were announced, if any within
# 10 days of the rule date (the date is hindsight; the rule date is the tradable one)
mc_ann = EVALL[EVALL.mcap_change.astype(bool) & EVALL.ann.notna()].ann.dt.normalize().unique()
qa = []
for d in qf:
    near = [x for x in mc_ann if abs((x - d).days) <= 10 and pd.Timestamp(x).dayofweek == 4]
    qa.append(pd.Timestamp(min(near, key=lambda x: abs((x - d).days))) if near else d)
print("actual quarterly dates differing from the rule:", [(a.date(), b.date()) for a, b in zip(qf, qa) if a != b])

_top20 = {}


def top20_at(T):
    T_i = alld.searchsorted(T)
    if T not in _top20:
        idx, _ = candidates(T_i, T_i - 5)
        _top20[T] = set(idx[:20])
    return _top20[T]


def perennial(d_f):
    """top-20 candidates at each of the 4 previous quarterly rule dates (before d_f) and never added: S&P has
    passed them over for a year (MLPs, foreign domicile, low float, ...). Uses past information only."""
    prev = [T for T in qf if T < d_f - pd.Timedelta(days=3)][-4:]
    if len(prev) < 4:
        return set()
    return set.intersection(*[top20_at(T) for T in prev])


BW = {"pre5": (-5, "c", 0, "c"), "pre10": (-10, "c", 0, "c"), "pre3": (-3, "c", 0, "c"),
      "ann_night": (0, "c", 1, "o"), "pre4_through_night": (-4, "c", 1, "o"), "post5": (0, "c", 5, "c")}


def run_dates(Ts, label):
    out = []
    for T in Ts:
        T_i = alld.searchsorted(T)
        if T_i >= N or alld[T_i] != T or T_i + 6 >= N or T_i < 260:
            continue
        for wn, (da, ka, db, kb) in BW.items():
            f_i = T_i + da
            idx, thr = candidates(T_i, f_i)
            per_ = perennial(alld[f_i])
            fresh = np.array([j for j in idx if j not in per_], dtype=int)
            um = UNIV[f_i]
            bm = bench(T_i + da, ka, T_i + db, kb, um)
            sy = spy_win(T_i + da, ka, T_i + db, kb)
            r = win(T_i + da, ka, T_i + db, kb)
            cost = COST[T_i + da] + COST[T_i + db]
            sel = {"top5": idx[:5], "top10": idx[:10], "top20": idx[:20], "rank21_50": idx[20:50],
                   "fresh_top5": fresh[:5], "fresh_top10": fresh[:10],
                   "above_min": idx[MCAP[f_i, idx] >= thr] if np.isfinite(thr) else idx[:0]}
            for g, js in sel.items():
                js = js[np.isfinite(r[js])]
                if len(js) == 0:
                    continue
                hits = np.array([added_soon(T, j) for j in js])
                rr = np.clip(r[js], -0.9, 3.0)
                out.append(dict(Td=T, label=label, window=wn, group=g, n=len(js), raw=rr.mean(), x=rr.mean() - bm,
                                xspy=rr.mean() - sy, net_s=rr.mean() - cost[js].mean() - sy, hit=hits.mean(),
                                x_nonhit=np.mean(rr[~hits]) - bm if (~hits).any() else np.nan,
                                x_hit=np.mean(rr[hits]) - bm if hits.any() else np.nan,
                                names=",".join(cols[j] for j in js[:10]) if wn == "pre5" else ""))
    return pd.DataFrame(out)


QD = run_dates(qf, "quarterly_rule")
QA = run_dates(sorted(set(qa)), "quarterly_actual")
fr = [d for d in alld[(alld >= "2020-01-01") & (alld <= END - pd.Timedelta(days=10))] if d.dayofweek == 4]
qi = np.array([alld.searchsorted(d) for d in qf + qa])
pl = [d for d in fr if np.min(np.abs(qi - alld.searchsorted(d))) >= 10]
PD = run_dates(pl, "placebo_fridays")
print("quarterly dates:", QD.Td.nunique(), "placebo dates:", PD.Td.nunique(), flush=True)
for g in ["top10", "fresh_top10"]:
    print(f"\n{g} candidates on the pre5 formation day (quarterly rule dates):")
    x = QD[(QD.window == "pre5") & (QD.group == g)]
    print(x[["Td", "hit", "names"]].to_string())

for lab, D in [("quarterly_rule", QD), ("quarterly_actual", QA), ("placebo_fridays", PD)]:
    for (wn, g), y in D.groupby(["window", "group"]):
        for per, a_, b_ in PER + [("all", "2020-01-01", "2026-12-31")]:
            z = y[(y.Td >= a_) & (y.Td <= b_)]
            if len(z) < 3:
                continue
            hold = BW[wn][2] - BW[wn][0]
            ROWS.append(dict(part="b_candidates", group=f"{lab}_{g}", period=per, window=wn, n=int(z.n.sum()),
                             n_dates=len(z), hold_days=hold, raw_bp=1e4 * z.raw.mean(), excess_bp=1e4 * z.x.mean(),
                             excess_datew_bp=1e4 * z.x.mean(), excess_med_bp=1e4 * z.x.median(), t_date=nw_t(z.x.values),
                             hit=float((z.x > 0).mean()), vs_spy_bp=1e4 * z.xspy.mean(), t_vs_spy=nw_t(z.xspy.values),
                             net_vs_spy_bp=1e4 * z.net_s.mean(), t_net_vs_spy=nw_t(z.net_s.values),
                             add_hit_rate=z.hit.mean(), excess_nonhit_bp=1e4 * z.x_nonhit.mean(),
                             excess_hit_bp=1e4 * z.x_hit.mean()))

# tradable book: top N at the close of T-5 to the close of T (or open T+1 from close T-4)
for g, n in [("top5", 5), ("top10", 10), ("top20", 20), ("fresh_top5", 5), ("fresh_top10", 10)]:
    for wn, (da, ka, db, kb) in [("pre5", BW["pre5"]), ("pre4_through_night", BW["pre4_through_night"])]:
        trades = []
        for T in qf:
            T_i = alld.searchsorted(T)
            if T_i + 6 >= N or alld[T_i] != T or T_i < 260:
                continue
            idx, _ = candidates(T_i, T_i + da)
            if g.startswith("fresh"):
                pr = perennial(alld[T_i + da])
                idx = np.array([j for j in idx if j not in pr], dtype=int)
            for j in idx[:n]:
                trades.append((T_i + da, ka, T_i + db, kb, j, 1, 0))
        s, acc = book(trades, slots=n, w=1.0 / n)
        for per, a_, b_ in PER:
            x = s.loc[a_:b_].loc[:END]
            st = ann_stats(x)
            ROWS.append(dict(part="b_book", group=f"quarterly_rule_{g}", period=per, window=wn, n=len(acc),
                             book_net_bp_day=1e4 * x.mean(), book_sharpe=st["sharpe"], book_maxdd=st["maxdd"],
                             book_ann=st["ann_ret"], invested=float((x != 0).mean()),
                             spy_sharpe=ann_stats(spyd.loc[a_:b_].loc[:END])["sharpe"]))

res = pd.DataFrame(ROWS)
res.to_csv(os.path.join(RES, "study48_index_adds.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
cols_show = ["n", "n_dates", "hold_days", "excess_datew_bp", "excess_med_bp", "t_date", "vs_spy_bp", "net_vs_spy_bp",
             "t_net_vs_spy"]
a = res[(res.part == "a_announced") & res.group.isin(["add_checked", "del_checked"])]
print("\n(a) announced changes")
print(a.set_index(["group", "window", "period"])[cols_show].round(1).to_string())
b = res[(res.part == "b_candidates")]
print("\n(b) candidates")
print(b.set_index(["group", "window", "period"])[cols_show + ["add_hit_rate", "excess_nonhit_bp", "excess_hit_bp"]]
      .round(2).to_string())
print("\nbooks")
print(res[res.part.str.endswith("book")]
      [["part", "group", "period", "window", "n", "book_net_bp_day", "book_sharpe", "book_maxdd", "invested",
        "spy_sharpe"]].round(3).to_string())
