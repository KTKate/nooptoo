"""Study 46: pre-earnings run-up inside the week before a report (never holding through the announcement).

For each report (store 'earnings', Nasdaq calendar, 2020-01 .. panel end; one row per symbol and date, rows with
another report of the same symbol within 10 trading days dropped, as in study 74) buy at the closing auction k = 1, 3
or 5 trading days before the exit and sell at the closing auction of the exit day X = the last close before the
announcement.
Exit day (two versions):
  safe (primary): X = the trading day before the calendar date d (d not a trading day: the trading day before it).
      Correct for both timings (BMO on d, AMC on d: the session of d is skipped), needs no timing information.
  gap: timing recovered from the prices as in study 74 (|gap after session d| >= 2x |gap before| and >= 1% -> AMC
      on d -> X = d; the mirror -> BMO -> X = d-1; ambiguous rows dropped). Uses later prices to recover a timing that
      was public beforehand, but it keeps only events with a clear reaction (selection): robustness only.
Universe at the entry close e = X - k: traded price of e (s6162_common.traded_price) > $5 and 20-day median dollar
volume through e-1 > $20M; adjusted closes of e and X present; no split between e and X.
Return r = C[X] / C[e] - 1 (adjusted closes). Excess = r - equal-weight mean of the same window over the liquid
universe of e. Costs: auction cost (core.exec_cost_bps 'auction' on traded prices) + 2.5 bp, per side.
Splits (known at the entry close): pt_net20 (analyst target raises - lowers over the last 20 windows to 15:45 of e:
>0 / 0 / <0 / none), previous report (calendar row 30-120 trading days before: EPS beat / miss / inline, surprise %
tercile), sector (current classification, mild look-ahead), size (ADV tercile), market (SPY 20-day return to e > 0 or
<= 0), prior run-up ret5 to e (tercile). Tercile breakpoints fixed on 2020-23.
Statistics: mean per event, t clustered by entry date (also by the ISO week of the exit, since windows of nearby dates
overlap).
Rules: at most 2, chosen on 2020-23 (safe exit) as the largest clustered t of the NET excess (long: ex - cost; short:
-ex - cost) among cells with >= 300 events (different split per rule), then run unchanged on 2024-26 as a book: each
trade 10% of equity, at most 10 open names (first come, same-day ties by ADV), cash otherwise; shorts only in names on
Alpaca's current easy-to-borrow list (flatters the past). Compared with SPY and hedged with the universe mean.
Survivorship: the panel holds only tickers alive in 2026 (excess vs the universe carries the same bias).
Output: results/study46_preearnings.csv (sections: coverage, cells, rules, count).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import store
import analyst_features as AF
from core import load_panel, stock_cols, ann_stats, RES, DATA
from s6162_common import traded_price, auction_cost_bps

OUT = os.path.join(RES, "study46_preearnings.csv")
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31")]
KS = (1, 3, 5)
ROWS = []


def emit(section, **kw):
    ROWS.append(dict(section=section, **kw))


def cl_t(x, g):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    m = x.mean()
    e = pd.Series(x - m).groupby(g).sum().values
    G = len(e)
    se = np.sqrt((e ** 2).sum() * G / max(G - 1, 1)) / n
    return m, (m / se if se > 0 else np.nan), n


# ------------------------------------------------------------------ panels
P = load_panel()
cols = stock_cols(P)
alld = P["c"].index
ND = len(alld)
C = P["c"][cols].astype("float64").values
O = P["o"][cols].astype("float64").values
TPfull = traded_price(P)
TP = TPfull[cols].values
ADV = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).values
COSTA = (auction_cost_bps(dict(P, rawc=TPfull))[cols] / 1e4).values
SPY = P["c"]["SPY"].astype("float64").values
UNIV = (TP > 5) & (ADV > 20e6) & np.isfinite(C)
spl = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
SPLIT = np.zeros((ND, len(cols)), dtype=bool)
ci = {t: i for i, t in enumerate(cols)}
for t, d in zip(spl.ticker, spl.date):
    i = alld.searchsorted(d)
    if i < ND:
        SPLIT[i, ci[t]] = True
SPLITCUM = np.cumsum(SPLIT, axis=0)
a_ = pd.read_parquet(os.path.join(DATA, "local", "alpaca_assets_active.parquet"))
ETB = set(a_.symbol[a_.easy_to_borrow.astype(bool) & a_.shortable.astype(bool)].str.replace(".", "-", regex=False))

_BENCH = {}


_RB = {}


def rev_bench(k):
    """Reversal control: per entry day e, universe EW mean of C[e+k]/C[e]-1 within each daily quintile of the
    5-day return to e (breakpoints of the universe on e). Returns (means [ND x 5], breakpoints [ND x 4])."""
    if k not in _RB:
        M = np.full((ND, 5), np.nan)
        Q = np.full((ND, 4), np.nan)
        for e in range(5, ND - k):
            r = C[e + k] / C[e] - 1
            r5 = C[e] / C[e - 5] - 1
            m = UNIV[e] & np.isfinite(r) & (np.abs(r) < 1) & np.isfinite(r5)
            if m.sum() < 100:
                continue
            q = np.quantile(r5[m], [0.2, 0.4, 0.6, 0.8])
            b = np.searchsorted(q, r5[m])
            Q[e] = q
            M[e] = [r[m][b == z].mean() for z in range(5)]
        _RB[k] = (M, Q)
    return _RB[k]


def bench(k):
    """Universe EW mean of C[e+k]/C[e]-1 for entry index e (universe of e, |r| < 100%)."""
    if k not in _BENCH:
        out = np.full(ND, np.nan)
        for e in range(ND - k):
            r = C[e + k] / C[e] - 1
            m = UNIV[e] & np.isfinite(r) & (np.abs(r) < 1)
            if m.sum() > 50:
                out[e] = r[m].mean()
        _BENCH[k] = out
    return _BENCH[k]


# ------------------------------------------------------------------ events
def events():
    E = store.read("earnings")
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E = E[E.symbol.isin(ci)].drop_duplicates(["symbol", "date"])
    Eall = E.copy()
    E = E[(E.date >= "2020-01-01") & (E.date <= alld[-1])].copy()
    E["di"] = alld.searchsorted(E.date.values)                       # first trading day >= d
    E["istd"] = (E.di < ND) & (alld[np.minimum(E.di, ND - 1)] == E.date)
    E = E[E.di < ND].sort_values(["symbol", "di"])
    gp = E.groupby("symbol").di.diff()
    gn = -E.groupby("symbol").di.diff(-1)
    E = E[~((gp < 10) | (gn < 10))].copy()
    E = E[E.di >= 30].copy()
    E["j"] = E.symbol.map(ci)
    di, j = E.di.values, E.j.values
    # gap timing (study 74), only for report dates that are trading days with a next day
    ok = E.istd.values & (di + 1 < ND)
    g_pre = np.where(ok, O[di, j] / C[di - 1, j] - 1, np.nan)
    g_post = np.where(ok, O[np.minimum(di + 1, ND - 1), j] / C[di, j] - 1, np.nan)
    ap, aq = np.abs(g_pre), np.abs(g_post)
    tg = np.select([(aq >= 2 * ap) & (aq >= 0.01), (ap >= 2 * aq) & (ap >= 0.01)], ["AMC", "BMO"], "amb")
    tg[~(np.isfinite(g_pre) & np.isfinite(g_post))] = "amb"
    E["t_gap"] = tg
    E["X_safe"] = di - 1
    E["X_gap"] = np.where(tg == "AMC", di, np.where(tg == "BMO", di - 1, -1))
    # previous report
    pe = Eall[Eall.date.isin(alld)].copy()
    pe["pdi"] = alld.get_indexer(pe.date)
    pe["p_surp"] = pe.surprise.astype(float)
    pe["p_beat"] = np.sign(pe.eps.astype(float) - pe.epsForecast.astype(float))
    pe = pe[["symbol", "pdi", "p_surp", "p_beat"]].sort_values("pdi")
    E = E.sort_values("di")
    E["key"] = E.di - 30
    E = pd.merge_asof(E, pe, left_on="key", right_on="pdi", by="symbol", direction="backward")
    stale = (E.di - E.pdi) > 120
    E.loc[stale, ["p_surp", "p_beat"]] = np.nan
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    E["sector"] = sec.sector.reindex(E.symbol).replace("", np.nan).fillna("none").values
    return E.drop(columns=["key"]).reset_index(drop=True)


def trades(E, k, xcol):
    x = E[E[xcol] >= 0].copy()
    x["X"] = x[xcol]
    x["e"] = x.X - k
    x = x[x.e >= 21].copy()
    e, X, j = x.e.values, x.X.values, x.j.values
    x["u"] = UNIV[e, j] & np.isfinite(C[X, j]) & (SPLITCUM[X, j] == SPLITCUM[e, j])
    x = x[x.u].copy()
    e, X, j = x.e.values, x.X.values, x.j.values
    x["edate"] = alld[e]
    x["xdate"] = alld[X]
    x["week"] = x.xdate.dt.strftime("%G-%V")
    x["r"] = C[X, j] / C[e, j] - 1
    x["bm"] = bench(k)[e]
    x["ex"] = x.r - x.bm
    x["exspy"] = x.r - (SPY[X] / SPY[e] - 1)
    x["cost"] = COSTA[e, j] + COSTA[X, j]
    x["cost"] = x.cost.fillna(0.002)
    x["adv"] = ADV[e, j]
    x["ret5"] = C[e, j] / C[e - 5, j] - 1
    x["mkt20"] = SPY[e] / SPY[e - 20] - 1
    M, Q = rev_bench(k)
    qz = np.array([np.searchsorted(Q[a], v) if np.isfinite(Q[a]).all() and np.isfinite(v) else -1
                   for a, v in zip(e, x.ret5.values)])
    x["ex_rev"] = np.where(qz >= 0, x.r - M[e, np.maximum(qz, 0)], np.nan)     # excess vs same-ret5-quintile stocks
    x["k"] = k
    x["etb"] = x.symbol.isin(ETB)
    x = x[np.isfinite(x.r) & (x.r.abs() < 1)]
    return x


def add_analyst(x, Aa):
    fc = Aa["pt_net20"].columns
    v = Aa["pt_net20"].reindex(index=alld, columns=fc).values
    fj = fc.get_indexer(x.symbol)
    x["pt_net20"] = np.where(fj >= 0, v[x.e.values, np.maximum(fj, 0)], np.nan)
    return x


def buckets(x):
    dev = x[x.edate <= "2023-12-31"]

    def terc(col):
        q = dev[col].dropna().quantile([1 / 3, 2 / 3]).values
        return pd.Series(np.select([x[col] <= q[0], x[col] <= q[1], x[col] > q[1]], ["T1", "T2", "T3"], "none"),
                         index=x.index)
    B = {"all": pd.Series("all", index=x.index)}
    B["pt_net20"] = pd.Series(np.select([x.pt_net20.isna(), x.pt_net20 < 0, x.pt_net20 == 0], ["none", "<0", "0"],
                                        ">0"), index=x.index)
    B["prev_beat"] = pd.Series(np.select([x.p_beat > 0, x.p_beat < 0, x.p_beat == 0], ["beat", "miss", "inline"],
                                         "none"), index=x.index)
    B["prev_surp"] = terc("p_surp")
    B["sector"] = x.sector
    B["size_adv"] = terc("adv")
    B["market20"] = pd.Series(np.where(x.mkt20 > 0, "spy20_up", "spy20_down"), index=x.index)
    B["ret5"] = terc("ret5")
    B["timing_gap(info)"] = x.t_gap
    return B


def cell_rows(x, B, k, xv):
    cells = []
    for s, lab in B.items():
        for bk in sorted(lab.unique()):
            for per, a0, b0 in PER:
                m = (lab == bk) & (x.edate >= a0) & (x.edate <= b0)
                y = x[m]
                if len(y) < 30:
                    continue
                mr, tr, n = cl_t(y.r, y.edate)
                me, te, _ = cl_t(y.ex, y.edate)
                _, tw, _ = cl_t(y.ex, y.week)
                ms, tsp, _ = cl_t(y.exspy, y.edate)
                mv, tv, _ = cl_t(y.ex_rev, y.edate)
                ml, tl, _ = cl_t(y.ex - y.cost, y.edate)
                mS, tS, _ = cl_t(-y.ex - y.cost, y.edate)
                emit("cells", exit=xv, k=k, split=s, bucket=bk, period=per, n=n, dates=y.edate.nunique(),
                     raw_bp=1e4 * mr, t_raw=tr, ex_bp=1e4 * me, t_ex=te, t_ex_week=tw, exspy_bp=1e4 * ms,
                     t_exspy=tsp, ex_rev_bp=1e4 * mv, t_ex_rev=tv, cost_rt_bp=1e4 * y.cost.mean(), net_long_bp=1e4 * ml, t_net_long=tl,
                     net_short_bp=1e4 * mS, t_net_short=tS, hit_ex=(y.ex > 0).mean())
                cells.append(dict(exit=xv, k=k, split=s, bucket=bk, period=per, n=n, ex=me, t=te,
                                  net_long=ml, t_long=tl, net_short=mS, t_short=tS))
    return cells


def book(y, side, d0, d1, max_names=10, w=0.10):
    """Calendar-time book: each trade w of equity from close e to close X, at most max_names open."""
    y = y[(y.edate >= d0) & (y.edate <= d1)]
    if side < 0:
        y = y[y.etb]
    y = y.sort_values(["e", "adv"], ascending=[True, False])
    held = np.zeros(ND, dtype=int)
    gross = np.zeros(ND)
    cost = np.zeros(ND)
    hedge = np.zeros(ND)
    taken = []
    rv = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    um = np.full(ND, np.nan)
    for d in range(1, ND):
        m = UNIV[d - 1] & np.isfinite(rv[d]) & (np.abs(rv[d]) < 1)
        if m.sum() > 50:
            um[d] = rv[d][m].mean()
    for e, X, j, cst in zip(y.e.values, y.X.values, y.j.values, y.cost.values):
        if held[e + 1:X + 1].max(initial=0) >= max_names:
            continue
        held[e + 1:X + 1] += 1
        taken.append(1)
        for d in range(e + 1, X + 1):
            gross[d] += side * w * np.nan_to_num(rv[d, j])
            hedge[d] += side * w * np.nan_to_num(um[d])
        cost[e] += w * cst / 2
        cost[X] += w * cst / 2
    idx = (alld >= d0) & (alld <= min(pd.Timestamp(d1), alld[-1]))
    df = pd.DataFrame({"gross": gross, "cost": cost, "hedge": hedge, "n": held}, index=alld)[idx]
    df["net"] = df.gross - df.cost
    df["net_hedged"] = df.net - df.hedge
    return df, len(taken)


def main():
    Aa = AF.load()
    E = events()
    print("events", len(E), "timing:", E.t_gap.value_counts().to_dict(), flush=True)
    for per, a0, b0 in PER:
        z = E[(E.date >= a0) & (E.date <= b0)]
        emit("coverage", period=per, split="reports", n=len(z),
             bucket=f"AMC {int((z.t_gap == 'AMC').sum())} BMO {int((z.t_gap == 'BMO').sum())} amb {int((z.t_gap == 'amb').sum())}")
    allcells, frames = [], {}
    for xv in ("safe", "gap"):
        for k in KS:
            x = add_analyst(trades(E, k, "X_" + xv), Aa)
            frames[(xv, k)] = x
            for per, a0, b0 in PER:
                z = x[(x.edate >= a0) & (x.edate <= b0)]
                emit("coverage", period=per, split=f"trades_{xv}_k{k}", n=len(z),
                     bucket=f"pt_net20 known {z.pt_net20.notna().mean():.2f}; prev report {z.p_beat.notna().mean():.2f}")
            B = buckets(x)
            allcells += cell_rows(x, B, k, xv)
            a = [r for r in ROWS if r["section"] == "cells" and r["exit"] == xv and r["k"] == k and r["split"] == "all"]
            for r in a:
                print(xv, k, r["period"], "n", r["n"], "ex", round(r["ex_bp"], 1), "t", round(r["t_ex"], 2),
                      "net_long", round(r["net_long_bp"], 1), "cost", round(r["cost_rt_bp"], 1), flush=True)
    cells = pd.DataFrame(allcells)
    # ---------------- rule choice on 2020-23 (safe exit)
    dev = cells[(cells.exit == "safe") & (cells.period == "2020-23") & (cells.n >= 300) & (cells.bucket != "none")
                & ~cells.split.str.contains("info")]
    cand = pd.concat([dev.assign(side=1, tt=dev.t_long, net=dev.net_long),
                      dev.assign(side=-1, tt=dev.t_short, net=dev.net_short)])
    cand = cand.sort_values("tt", ascending=False)
    print("top candidates 2020-23 (net, clustered t):\n",
          cand.head(12)[["k", "split", "bucket", "side", "n", "ex", "net", "tt"]].to_string(), flush=True)
    rules, used = [], set()
    for _, rw in cand.iterrows():
        if len(rules) == 2:
            break
        if rw.split in used or rw.tt <= 0:
            continue
        used.add(rw.split)
        rules.append((int(rw.k), rw.split, rw.bucket, int(rw.side), rw.tt))
    for k, s, bk, side, tt in rules:
        x = frames[("safe", k)]
        mask = buckets(x)[s] == bk
        y = x[mask]
        for per, a0, b0 in PER:
            z = y[(y.edate >= a0) & (y.edate <= b0)]
            if side < 0:
                z = z[z.etb]
            mn, tn, n = cl_t(side * z.ex - z.cost, z.edate)
            mv, tv, _ = cl_t(side * z.ex_rev - z.cost, z.edate)
            df, ntr = book(y, side, a0, b0)
            st, sh = ann_stats(df.net), ann_stats(df.net_hedged)
            spy = ann_stats(P["c"]["SPY"].pct_change().loc[df.index])
            emit("rules", exit="safe", k=k, split=s, bucket=bk, side=side, period=per, n=n, net_bp=1e4 * mn,
                 t_net=tn, net_vs_rev_control_bp=1e4 * mv, t_net_vs_rev_control=tv, choice_t_2020_23=tt, book_trades=ntr, book_avg_names=df.n[df.n > 0].mean(),
                 book_invested=(df.n > 0).mean(), book_ann=st["ann_ret"], book_sharpe=st["sharpe"],
                 book_maxdd=st["maxdd"], book_sharpe_hedged=sh["sharpe"], book_ann_hedged=sh["ann_ret"],
                 spy_ann=spy["ann_ret"], spy_sharpe=spy["sharpe"],
                 corr_spy=float(df.net.corr(P["c"]["SPY"].pct_change().loc[df.index])))
            print("RULE", k, s, bk, side, per, "n", n, "net", round(1e4 * mn, 1), "t", round(tn, 2), "book",
                  round(st["ann_ret"], 3), round(st["sharpe"], 2), "hedged", round(sh["sharpe"], 2), "spy",
                  round(spy["ann_ret"], 3), flush=True)
    n_dev = len(cells[cells.period == "2020-23"])
    emit("count", split="variants_tried", n=n_dev,
         bucket=f"{n_dev} cells (2 exits x 3 k x splits/buckets) in 2020-23; {len(cand)} (cell, side) rule candidates"
                f" on the safe exit; {len(rules)} rules")
    pd.DataFrame(ROWS).to_csv(OUT, index=False)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
