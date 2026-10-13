"""Studies 135, 136 and 137: regimes, edge decay and what to expect live for the overnight blend at official crosses.

135  Regimes and sector mix. Question: in which market conditions is the live (3-per-industry capped) blend weakest,
     and does any pattern hold in both periods? Has the mix of sectors in the picks shifted, in particular towards
     tech hardware?
136  What to expect live. Question: for a $10,000 account run like the backtest, what range of outcomes over 1, 3, 6
     and 12 months is consistent with the backtest, how likely is a loss or a deep drawdown, and how do those change
     if live returns are only half the backtest mean? Where does the paper virtual book so far fall in that range?
137  Edge decay. Question: has the blend's net return per night fallen over 2024-2026?

Data: core.load_panel (split- and dividend-adjusted o/c, rawc; ^VIX and SPY are columns). Blend as study 69 lines
22-46 (top 10 of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3; results/study23_pred.parquet, study33_pred.parquet),
and the live version with at most 3 names per industry (paper_overnight.cap_by_industry, data/store/sectors.parquet,
names without an industry are their own group). Official crosses: caches of studies 88/94/99 (study99.crosses) plus
data/local/auctions108/110/113/118/121/128.parquet. Night return at crosses as study 94: panel night return x
(open cross / panel open on t+1) / (close cross / panel close on t), legs without a usable cross at the panel price.
Costs core.exec_cost_bps(P, "auction") + 2.5 bp per side. Sectors and industries: data/store/sectors.parquet (one
snapshot, today's classification applied to all dates). Tech hardware as studies 90/92 (TECH_HW industries, plus
'Electrical Products' and 'Industrial Machinery/Components' inside the Technology sector). Live virtual book:
logs/paper/virtual_days.csv (read only; nights from 2026-09-29, after the backtest window).

Design 135: regime variables known at the close of the entry day t: VIX level, VIX 5-day change (VIX(t) - VIX(t-5)),
SPY close above / below its 50-day average, SPY 20-day realized volatility (std of daily close-to-close returns).
Terciles use cut points from the whole 2024-01..2026-09 sample, so a tercile means the same level in both periods.
Per bucket and period: nights, mean net bp, annualized Sharpe, hit rate (share of nights with net > 0), and a Welch t
of the bucket mean against the other nights of the same period. Sector mix: share of capped picks per sector by
calendar quarter; tech hardware share per quarter, linear trend over the 11 quarters (slope per quarter and OLS t on
quarterly shares) and the share per period.
Design 136: stationary block bootstrap (Politis and Romano; geometric blocks with mean length 10 nights, circular)
of the capped blend's daily net returns at crosses, 10,000 paths of 252 nights, seed 136, on 2024-01..2026-09 and on
2025-07..2026-09 only. Per horizon (21, 63, 126, 252 nights): compounded return median, 5th and 25th percentiles,
chance of a loss, chance that the drawdown from the running peak (starting equity included) is worse than -10%,
-20%, -30% at any point within the horizon. Haircut: the same paths with every daily return lowered by half the
sample mean (mean halved, volatility unchanged). Virtual book: compounded net over its nights (and, as a second
measure, the log's own equity column, which differs from the compounded net column), placed as a percentile among the bootstrap's compounded returns over the same number of nights (full and haircut).
Design 137: net bp per night and Sharpe per calendar quarter 2024Q1..2026Q3 (capped and uncapped, at crosses). Trend:
OLS of the 33 monthly mean net returns on time (slope in bp per night per year, OLS t, and a Newey-West t with 3
lags). Chow-style split: nights in the first half vs the second half of 2024-01..2026-09 (split at the median
date), mean, Sharpe, Welch t of the difference, and the Chow F for a break in the mean (same as the squared pooled t).
Virtual book: nights, mean net bp, compounded net.

Data checks: the series reproduce study 94 at crosses (uncapped Sharpe 2.48 / 2.27 / 2.38, mean 37.5 bp; capped
2.65 / 2.23 / 2.45, mean 38.2 bp, for 2024-25H1 / 2025H2-26 / 2024-26). Cross data errors (legs at the panel price):
334 + 3 opening, 337 + 2 closing of ~77,000 found crosses; legs without a usable cross 0.7-0.8% of picks.

Result 135 (capped, at crosses; 2024-25H1 / 2025H2-26): no regime is weak in both periods except SPY below its
50-day average (23 vs 44 bp above; 12 vs 42 bp above; Sharpe 1.3 / 0.7 vs 3.2 / 2.5; t -0.6 / -0.8, not significant;
139 of 685 nights). VIX: the high tercile (> 17.9) was weakest in the first period (19 bp) but fine in the second
(44 bp), where the low tercile (<= 15.5) was weakest (2 bp, 68 nights). Rising VIX (5-day change > +0.8) 21 bp vs
41 bp; low SPY 20-day vol 6 bp vs 41 bp: one period each. Sector mix moved: Technology 16% of picks in 2024Q1 to
34-35% in 2026Q2-Q3, Health Care 26% to 7-10%, Basic Materials 2% to 10-22%. Tech hardware share 7% (2024Q1) to 30%
(2026Q3; 3.0 names a night), trend +1.8 points a quarter (t 3.4, Newey-West t 2.7); 8.4% of picks in 2024-25H1 vs
17.7% in 2025H2-26. Nights with 3+ tech hardware names: -51 bp (30 nights) in the first period, +66 bp (84 nights) in
the second, so no stable penalty yet.
Result 137: quarterly net bp (capped) 44, 63, 16, 20, 25, 68, 38, 70, 13, 64, -3 (2024Q1..2026Q3); 2026Q3 is the
only negative quarter (Sharpe -0.20). Monthly trend -6.3 bp a year (t -0.46, NW t -0.65; uncapped -4.1, t -0.30).
First vs second half (split 2025-05-15): 34.8 vs 41.6 bp (t 0.36, Chow F 0.13); uncapped 32.5 vs 42.4. No
measurable decay; quarter-to-quarter swings of 0-70 bp are normal for this series.
Result 136 (capped, 2024-26 sample; daily mean 38.2 bp, sd 247 bp): median return 1m +7.0%, 3m +24%, 6m +55%, 12m
+140%; 5th percentile -10.5% / -10.3% / -2.9% / +25%; chance of a loss 27% / 14% / 6% / 1.4%; drawdown worse than
-10% within the horizon 28% / 69% / 91% / 99%, worse than -20% 2% / 12% / 27% / 50%, worse than -30% 0% / 1% / 4% /
10%. 2025H2-26 sample is similar, slightly wider (12m loss 2.8%, dd < -30% 18%). Half mean (19 bp): median 12m +49%,
chance of a loss 39% / 32% / 25% / 16% (2025H2-26 sample: 40% / 35% / 28% / 20%), 12m drawdown worse than -20% 79%,
worse than -30% 35%. Virtual book: 8 nights (2026-09-29..10-09), mean -11.7 bp, compounded net -1.05% (equity
column -0.53%): 28th percentile of 8-night backtest outcomes (31st on the equity column), 37th under the haircut.
Caveats: the bootstrap reuses 2024-26 nights only (no 2020-23 or 2022-type market), assumes daily returns scale with
a $10,000 book and ignores capacity, fills off the cross and model refits; regime buckets are small (50-150 nights)
and none of the regime differences is significant at 5%.

Output: results/study135_regimes.csv (table = regime / sector_share / techhw_trend), results/study136_expectations.csv
(table = bootstrap / virtual_book), results/study137_decay.csv (table = quarter / trend / halves / virtual_book).

    python src/study135_137_expectations.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(DATA, "local")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
EXTRA = ["auctions108", "auctions110", "auctions113", "auctions118", "auctions121", "auctions128"]
TECH_HW = ["Semiconductors", "Electronic Components", "Computer peripheral equipment", "Computer Manufacturing",
           "Computer Communications Equipment", "Retail: Computer Software & Peripheral Equipment"]
TECH_HW_TECHSECTOR = ["Electrical Products", "Industrial Machinery/Components"]
HOR = [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]
NPATH, BLOCK, SEED = 10000, 10, 136


def tstat_ols(y, x):
    """OLS slope, its t, and a Newey-West (3 lags) t."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXi) * (e @ e) / (n - 2))[1]
    S = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, 4):
        w = 1 - lag / 4
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi * n / (n - 2)
    return b[1], b[1] / se, b[1] / np.sqrt(V[1, 1])


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def st(x):
    x = x.dropna()
    s = ann_stats(x)
    return dict(n=s["n"], net_bp=1e4 * x.mean(), sharpe=s["sharpe"], hit=(x > 0).mean() if len(x) else np.nan)


# ------------------------------------------------------------------ series
def build():
    import study99_cross_retrain as S99
    from paper_overnight import cap_by_industry
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr, ok
    S = S.loc["2024-01":"2026-09"]
    W = bt.select_topk(S, S.notna(), 10)
    Wc = pd.DataFrame(0.0, index=S.index, columns=cols)
    for d, row in S.iterrows():
        out = list(cap_by_industry(row.dropna().sort_values(ascending=False, kind="stable"), 10, 3).index)
        if out:
            Wc.loc[d, out] = 1.0 / len(out)
    # crosses
    parts = [S99.crosses()]
    for f in EXTRA:
        fn = os.path.join(LOC, f + ".parquet")
        if os.path.exists(fn):
            c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
            c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
            c["date"] = pd.to_datetime(c.date)
            parts.append(c)
    X = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    ro, rc, err = S99.cross_ratios(P, cols, days, X)
    print("cross data errors (legs at panel price):", err, flush=True)
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=S.index)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex(index=S.index) / rc.fillna(1.0).reindex(index=S.index) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=S.index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=S.index)
    nets = {}
    for name, Wx in [("uncapped", W), ("capped", Wc)]:
        nets[name] = bt.run(Wx, Rx, C).net.where(Wx.sum(axis=1) > 0).dropna()
        h = Wx > 0
        print(name, "fallback share", round(float((miss & h).sum().sum() / h.sum().sum()), 4),
              {lab: round(ann_stats(nets[name].loc[a:b])["sharpe"], 2) for lab, a, b in PER},
              "mean bp", round(1e4 * nets[name].mean(), 1), flush=True)
    mkt = pd.DataFrame({"vix": P["c"]["^VIX"], "spy": P["c"]["SPY"]})
    return dict(P=P, cols=cols, Wc=Wc, nets=nets, mkt=mkt)


# ------------------------------------------------------------------ 135
def study135(K):
    net = K["nets"]["capped"]
    m = K["mkt"]
    v, s = m.vix.ffill(), m.spy
    sr = s.pct_change()
    reg = pd.DataFrame({"vix_level": v, "vix_chg5": v - v.shift(5), "spy_vs_ma50": s / s.rolling(50).mean() - 1,
                        "spy_vol20": sr.rolling(20).std() * np.sqrt(252)}).reindex(net.index)
    rows = []
    buckets = {}
    for k in ["vix_level", "vix_chg5", "spy_vol20"]:
        q = reg[k].quantile([1 / 3, 2 / 3]).values
        lab = pd.cut(reg[k], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"]).astype(str)
        buckets[k] = (lab, {"low": f"<= {q[0]:.2f}", "mid": f"{q[0]:.2f}..{q[1]:.2f}", "high": f"> {q[1]:.2f}"})
    lab = np.where(reg.spy_vs_ma50 > 0, "above", "below")
    buckets["spy_trend50"] = (pd.Series(lab, index=reg.index).where(reg.spy_vs_ma50.notna()), {"above": "SPY > MA50",
                                                                                              "below": "SPY <= MA50"})
    for k, (lab, desc) in buckets.items():
        for p, a, b in PER:
            x, lb = net.loc[a:b], lab.loc[a:b]
            for lev in desc:
                sel = x[lb == lev]
                rows.append(dict(table="regime", regime=k, bucket=lev, cut=desc[lev], period=p, **st(sel),
                                 t_vs_rest=welch(sel, x[lb != lev])))
    # sector mix of the capped picks by quarter
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    pk = K["Wc"].stack()
    pk = pk[pk > 0].reset_index()
    pk.columns = ["date", "ticker", "w"]
    pk["sector"] = pk.ticker.map(sec.sector).replace("", np.nan).fillna("(none)")
    ind = pk.ticker.map(sec.industry)
    pk["techhw"] = ind.isin(TECH_HW) | (ind.isin(TECH_HW_TECHSECTOR) & (pk.sector == "Technology"))
    pk["q"] = pk.date.dt.to_period("Q").astype(str)
    sh = pd.crosstab(pk.q, pk.sector, normalize="index")
    for q, r in sh.iterrows():
        for sct, val in r.items():
            rows.append(dict(table="sector_share", period=q, bucket=sct, share=val, n=int((pk.q == q).sum())))
    for p, a, b in PER:
        x = pk[(pk.date >= a) & (pk.date <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
        for sct, val in x.sector.value_counts(normalize=True).items():
            rows.append(dict(table="sector_share", period=p, bucket=sct, share=val, n=len(x)))
        rows.append(dict(table="techhw_trend", period=p, bucket="tech hardware share", share=x.techhw.mean(), n=len(x),
                         names_per_night=x.groupby("date").techhw.sum().mean()))
    tq = pk.groupby("q").techhw.mean()
    for q, val in tq.items():
        rows.append(dict(table="techhw_trend", period=q, bucket="tech hardware share", share=val,
                         n=int((pk.q == q).sum()), names_per_night=pk[pk.q == q].groupby("date").techhw.sum().mean()))
    slope, t, tnw = tstat_ols(tq.values, np.arange(len(tq), dtype=float))
    rows.append(dict(table="techhw_trend", period="2024Q1..2026Q3", bucket="linear trend of quarterly share",
                     slope_per_q=slope, t=t, t_nw=tnw, n=len(tq)))
    # tech hardware nights vs others (3+ tech hardware names)
    nth = pk.groupby("date").techhw.sum().reindex(net.index).fillna(0)
    for p, a, b in PER:
        x, nn = net.loc[a:b], nth.loc[a:b]
        for lev, sel in [("0-1 tech hw names", nn <= 1), ("2 tech hw names", nn == 2), ("3+ tech hw names", nn >= 3)]:
            rows.append(dict(table="regime", regime="techhw_names", bucket=lev, period=p, **st(x[sel]),
                             t_vs_rest=welch(x[sel], x[~sel])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 137
def virtual_book(equity=False):
    v = pd.read_csv(os.path.join(ROOT, "logs", "paper", "virtual_days.csv"), parse_dates=["day"])
    if equity:      # the log's own equity column (pnl in dollars on $10,000)
        return v.equity.iloc[-1] / 1e4 - 1
    return v.set_index("day").net


def study137(K):
    rows = []
    vb = virtual_book()
    for name in ["capped", "uncapped"]:
        net = K["nets"][name]
        q = net.groupby(net.index.to_period("Q"))
        for p, x in q:
            rows.append(dict(table="quarter", variant=name, period=str(p), **st(x),
                             maxdd=ann_stats(x)["maxdd"]))
        mm = net.groupby(net.index.to_period("M")).mean()
        yrs = np.arange(len(mm)) / 12.0
        slope, t, tnw = tstat_ols(1e4 * mm.values, yrs)
        rows.append(dict(table="trend", variant=name, period="2024-01..2026-09 monthly means", n=len(mm),
                         slope_bp_per_year=slope, t=t, t_nw=tnw,
                         fitted_start_bp=1e4 * mm.mean() - slope * yrs.mean(),
                         fitted_end_bp=1e4 * mm.mean() + slope * (yrs[-1] - yrs.mean())))
        mid = net.index[len(net) // 2]
        h1, h2 = net[net.index < mid], net[net.index >= mid]
        for lab, x in [("first half", h1), ("second half", h2)]:
            rows.append(dict(table="halves", variant=name, period=f"{lab} ({x.index[0].date()}..{x.index[-1].date()})",
                             **st(x)))
        d = net.copy()
        g = (d.index >= mid).astype(float)
        sp = np.sqrt(((len(h1) - 1) * h1.var() + (len(h2) - 1) * h2.var()) / (len(d) - 2))
        tp = (h2.mean() - h1.mean()) / (sp * np.sqrt(1 / len(h1) + 1 / len(h2)))
        rows.append(dict(table="halves", variant=name, period="second minus first", n=len(d),
                         net_bp=1e4 * (h2.mean() - h1.mean()), t_welch=welch(h2, h1), chow_F=tp ** 2,
                         sharpe=ann_stats(h2)["sharpe"] - ann_stats(h1)["sharpe"]))
        del g
    rows.append(dict(table="virtual_book", variant="live virtual book (cap used live)",
                     period=f"{vb.index[0].date()}..{vb.index[-1].date()}", n=len(vb), net_bp=1e4 * vb.mean(),
                     hit=(vb > 0).mean(), cum_net=(1 + vb).prod() - 1, sharpe=ann_stats(vb)["sharpe"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 136
def boot_paths(r, npath, T, block, rng):
    n = len(r)
    idx = np.empty((npath, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, npath)
    newb = rng.random((npath, T)) < 1.0 / block
    starts = rng.integers(0, n, (npath, T))
    for j in range(1, T):
        idx[:, j] = np.where(newb[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return r[idx]


def study136(K):
    rows = []
    net = K["nets"]["capped"]
    vb = virtual_book()
    nv, cumv = len(vb), (1 + vb).prod() - 1
    for p, a, b in [PER[2], PER[1]]:
        r = net.loc[a:b].values
        mu = r.mean()
        rng = np.random.default_rng(SEED)
        base = boot_paths(r, NPATH, 252, BLOCK, rng)
        for case, shift in [("backtest", 0.0), ("half mean (haircut)", mu / 2)]:
            x = base - shift
            eq = np.cumprod(1 + x, axis=1)
            peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
            dd = np.minimum.accumulate(np.minimum(eq / peak - 1, 0.0), axis=1)
            for h, lab in HOR:
                fin = eq[:, h - 1] - 1
                rows.append(dict(table="bootstrap", sample=p, case=case, horizon=lab, nights=h,
                                 daily_mean_bp=1e4 * (mu - shift), daily_sd_bp=1e4 * r.std(ddof=1),
                                 median=np.median(fin), p5=np.quantile(fin, 0.05), p25=np.quantile(fin, 0.25),
                                 p75=np.quantile(fin, 0.75), p95=np.quantile(fin, 0.95), p_loss=(fin < 0).mean(),
                                 p_dd10=(dd[:, h - 1] < -0.10).mean(), p_dd20=(dd[:, h - 1] < -0.20).mean(),
                                 p_dd30=(dd[:, h - 1] < -0.30).mean(), median_maxdd=np.median(dd[:, h - 1])))
            fin = eq[:, nv - 1] - 1
            for meas, val in [("compounded net column", cumv), ("equity column", virtual_book(True))]:
                rows.append(dict(table="virtual_book", sample=p, case=case, horizon=f"{nv} nights", nights=nv,
                                 measure=meas, virtual_cum=val, virtual_mean_bp=1e4 * vb.mean(), median=np.median(fin),
                                 p5=np.quantile(fin, 0.05), p25=np.quantile(fin, 0.25), p_loss=(fin < 0).mean(),
                                 percentile=(fin < val).mean() + 0.5 * (fin == val).mean()))
        del base
    return pd.DataFrame(rows)


def main():
    K = build()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for fn, f in [("study135_regimes.csv", study135), ("study137_decay.csv", study137),
                  ("study136_expectations.csv", study136)]:
        df = f(K)
        df.to_csv(os.path.join(RES, fn), index=False)
        print("\n==", fn)
        if fn.startswith("study135"):
            print(df[df.table != "sector_share"].dropna(axis=1, how="all").round(3).to_string(index=False))
            sh = df[df.table == "sector_share"].pivot_table(index="bucket", columns="period", values="share")
            print(sh.round(2).to_string())
        else:
            print(df.dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
