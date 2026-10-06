"""Study 57: pairs within industries chosen by fundamental similarity + 1-year return correlation; 1-5 day reversion
of the spread after a 2-standard-deviation move (research, no orders).

Pair formation (first trading day of each quarter 2020Q1..2026Q3, data before that day only):
  universe: traded price > $10 and 20-day median dollar volume > $50M on the previous day, >= 120 daily returns in
  the past 252 days, industry known (data/store/sectors.parquet, today's classification).
  fundamentals (data/local/fundamentals_panels.pkl, point in time): percentile ranks (all stocks that day) of
  log market cap, EV/sales, operating margin, revenue growth y/y; distance = mean absolute rank difference over the
  fields both stocks have (market cap plus at least one other required).
  correlation: daily log returns over the past 252 trading days (>= 120 common days).
  Within each industry, all pairs of universe members are candidates; 200 pairs are kept per quarter by
    corr  : highest correlation (correlation only; the study-20 selection, but 1-year window, quarterly)
    fund  : smallest fundamental distance among pairs with correlation >= 0.5
    combo : best average of the correlation rank and the (inverse) distance rank
Signal: spread return s = r_A - r_B (log, dollar neutral) over k = 1 or 3 days; sigma_k = std of the k-day spread
  sums over the past 252 days (to t-1, >= 120 days). Entry when |s| > 2 sigma_k: long the laggard, short the
  leader, in the closing auction of t. The leader must be shortable and easy to borrow on Alpaca's CURRENT list
  (data/local/alpaca_assets_active.parquet; today's list, which flatters the past), borrow fee 0.
  timing "close": s uses the closing price of t (the auction sets it; not strictly executable, as in study 20).
  timing "1545"  : today's term of s uses the 15:45 price (5-minute snapshots, 2024+ only); exit checks at 15:45 too.
Exit: closing auction of t+H (H = 1, 3, 5), or (exit "revert") at the first close t+j < t+H where at least half
  of the entry gap has closed (laggard - leader log return since entry >= |s|/2).
Book: at most 10 open trades, 0.1 of capital per leg (gross <= 2), ranked by |s|/sigma when slots are short; a stock
  is in at most one open trade. Costs per side per leg: exec_cost_bps(P, 'auction') + 2.5 bp.
Periods: 2020-23 vs 2024-26 (to 2026-09-25). Output: results/study57_pairs.csv (sections: check, variant, pairs).
Survivorship: the daily panel holds only tickers that still trade in 2026.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, RES, DATA  # noqa

EXTRA = 2.5
N_PAIRS = 200
MAX_OPEN = 10
W_LEG = 0.1
END = "2026-09-25"
PERIODS = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", END)]


def setup():
    P = load_panel()
    cols = stock_cols(P)
    dates = P["c"].index
    dates = dates[dates <= END]
    c = P["c"][cols].reindex(dates).astype("float64")
    rawc = P["rawc"][cols].reindex(dates).astype("float64")
    tc = traded_close(P)[cols].reindex(dates)
    px = tc.where(tc.notna(), rawc)
    adv = P["dv"][cols].reindex(dates).rolling(20, min_periods=10).median()
    cost = ((exec_cost_bps(P, "auction")[cols].reindex(dates).astype("float64") + EXTRA) / 1e4).fillna(50e-4)
    lr = np.log(c / c.shift(1))
    checks = []
    big = (lr.abs() > np.log(1.5))
    checks.append(dict(section="check", item="stock_days_abs_logret_gt_40pct", value=int(big.sum().sum()),
                       note="masked (set NaN) in signals and correlations; trades still earn the real return"))
    lr_sig = lr.mask(big)
    from study8_exec_retest import price_1545
    p15 = price_1545("none").reindex(index=dates, columns=cols).astype("float64") * (c / rawc)
    bad15 = (p15 / c - 1).abs() > 0.5
    checks.append(dict(section="check", item="p1545_far_from_close_masked", value=int(bad15.sum().sum()), note=">50% from close"))
    p15 = p15.mask(bad15)
    lr15 = np.log(p15 / c.shift(1)).mask(big)
    dev15 = (np.log(p15 / c)).stack()
    checks.append(dict(section="check", item="p1545_vs_close_median_abs_bp", value=1e4 * dev15.abs().median(),
                       note=f"n={len(dev15)}, first date {p15.dropna(how='all').index.min().date()}"))
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).set_index("ticker")
    ind = sec["industry"].reindex(cols)
    a = pd.read_parquet(os.path.join(DATA, "local", "alpaca_assets_active.parquet"))
    etb = set(a[a.easy_to_borrow & a.shortable].symbol)
    F = pd.read_pickle(os.path.join(DATA, "local", "fundamentals_panels.pkl"))
    fund = {}
    for k in ["market_cap", "ev_sales", "op_margin", "revenue_growth_yoy"]:
        x = F[k].reindex(index=dates, columns=cols).astype("float64")
        if k == "market_cap":
            x = np.log(x.where(x > 0))
        fund[k] = x.rank(axis=1, pct=True)
    return dict(P=P, cols=np.array(cols), dates=dates, c=c, cf=c.ffill(limit=5), px=px, adv=adv, cost=cost,
                lr=lr, lr_sig=lr_sig, lr15=lr15, p15=p15, ind=ind, etb=np.array([t in etb for t in cols]),
                fund=fund, checks=checks)


def form(D, i_f):
    """Candidate pairs at formation index i_f (data to i_f-1). Returns dict selection -> DataFrame(a, b, rho, dist)."""
    dates = D["dates"]
    lo = max(0, i_f - 252)
    win = D["lr_sig"].iloc[lo:i_f]
    ok = ((D["px"].iloc[i_f - 1] > 10) & (D["adv"].iloc[i_f - 1] > 5e7) & (win.notna().sum() >= 120)
          & D["ind"].notna())
    names = ok[ok].index
    fr = {k: v.iloc[i_f - 1][names] for k, v in D["fund"].items()}
    ind = D["ind"][names]
    rows = []
    for g, mem in ind.groupby(ind):
        m = mem.index
        if len(m) < 2:
            continue
        cm = win[m].corr(min_periods=120).values
        FV = np.column_stack([fr[k][m].values for k in ["market_cap", "ev_sales", "op_margin", "revenue_growth_yoy"]])
        iu = np.triu_indices(len(m), 1)
        for i, j in zip(*iu):
            rho = cm[i, j]
            if not np.isfinite(rho):
                continue
            both = np.isfinite(FV[i]) & np.isfinite(FV[j])
            dist = np.abs(FV[i] - FV[j])[both].mean() if (both[0] and both.sum() >= 2) else np.nan
            rows.append((m[i], m[j], rho, dist, g))
    df = pd.DataFrame(rows, columns=["a", "b", "rho", "dist", "industry"])
    out = {"corr": df.nlargest(N_PAIRS, "rho")}
    f = df[(df.rho >= 0.5) & df.dist.notna()]
    out["fund"] = f.nsmallest(N_PAIRS, "dist")
    g = df[df.dist.notna()].copy()
    g["score"] = (g.rho.rank(ascending=False) + g.dist.rank()) / 2
    out["combo"] = g.nsmallest(N_PAIRS, "score")
    return out, len(names), len(df), int(df.dist.notna().sum())


def candidates(D, pairs_by_q, sel, k, timing):
    """All entry signals: list of (t, z, s, long_idx, short_idx)."""
    ci = {t: i for i, t in enumerate(D["cols"])}
    lr = D["lr_sig"].values
    lr15 = D["lr15"].values
    out = []
    for (i_f, i_end), pr in pairs_by_q:
        df = pr[sel]
        if len(df) == 0:
            continue
        ia = np.array([ci[x] for x in df.a])
        ib = np.array([ci[x] for x in df.b])
        lo = max(0, i_f - 260)
        d = lr[lo:i_end, ia] - lr[lo:i_end, ib]                        # spread log returns (close)
        dd = pd.DataFrame(d)
        sk = dd.rolling(k, min_periods=k).sum()
        sig = sk.rolling(252, min_periods=120).std().shift(1).values
        if timing == "close":
            s = sk.values
        else:
            today = lr15[lo:i_end, ia] - lr15[lo:i_end, ib]
            prev = dd.shift(1).rolling(k - 1, min_periods=k - 1).sum().values if k > 1 else 0.0
            s = today + prev
        z = s / sig
        rows = np.arange(lo, i_end)
        sel_t = (rows >= i_f)
        zz = np.where(sel_t[:, None], z, np.nan)
        tt, pp = np.where(np.abs(zz) > 2)
        for t_rel, p in zip(tt, pp):
            sv = s[t_rel, p]
            # s > 0: A led -> short A, long B
            L, S = (ib[p], ia[p]) if sv > 0 else (ia[p], ib[p])
            out.append((rows[t_rel], abs(zz[t_rel, p]), abs(sv), L, S))
    out.sort(key=lambda x: (x[0], -x[1]))
    return out


def simulate(D, cands, H, exit_rule, timing):
    T = len(D["dates"])
    cf = D["cf"].values
    p15 = D["p15"].values
    cost = D["cost"].values
    etb = D["etb"]
    pnl = np.zeros(T)
    gross = np.zeros(T)
    nopen = np.zeros(T)
    trades = []
    busy_until = {}                    # stock -> last day (exit index) of its open trade
    open_by_day = np.zeros(T + 10, int)
    by_t = {}
    for cnd in cands:
        by_t.setdefault(cnd[0], []).append(cnd)
    n_skip_etb = 0
    for t in sorted(by_t):
        if t + 1 >= T:
            continue
        for (_, z, sv, L, S) in by_t[t]:
            if not etb[S]:
                n_skip_etb += 1
                continue
            if open_by_day[t] >= MAX_OPEN:
                break
            if busy_until.get(L, -1) >= t or busy_until.get(S, -1) >= t:
                continue
            if not (np.isfinite(cf[t, L]) and np.isfinite(cf[t, S])):
                continue
            e = min(t + H, T - 1)
            if exit_rule == "revert":
                for j in range(t + 1, e):
                    if timing == "1545" and np.isfinite(p15[j, L]) and np.isfinite(p15[j, S]):
                        sp = np.log(p15[j, L] / cf[t, L]) - np.log(p15[j, S] / cf[t, S])
                    else:
                        sp = np.log(cf[j, L] / cf[t, L]) - np.log(cf[j, S] / cf[t, S])
                    if sp >= 0.5 * sv:
                        e = j
                        break
            aL = cf[t:e + 1, L] / cf[t, L]
            aS = cf[t:e + 1, S] / cf[t, S]
            daily = W_LEG * (np.diff(aL) - np.diff(aS))
            pnl[t + 1:e + 1] += daily
            gross[t + 1:e + 1] += daily
            c_in = W_LEG * (cost[t, L] + cost[t, S])
            c_out = W_LEG * (cost[e, L] + cost[e, S])
            pnl[t] -= c_in
            pnl[e] -= c_out
            open_by_day[t:e] += 1
            nopen[t:e] += 1
            busy_until[L] = e
            busy_until[S] = e
            trades.append(dict(t=t, e=e, gross=(aL[-1] - aS[-1]), net=(aL[-1] - aS[-1]) - (c_in + c_out) / W_LEG))
    return pd.Series(pnl, index=D["dates"]), pd.DataFrame(trades), n_skip_etb


def summarize(name, meta, pnl, tr, dates):
    rows = []
    tr = tr.assign(date=dates[tr.t.values]) if len(tr) else tr
    for per, a, b in PERIODS:
        if meta["timing"] == "1545" and per == "2020-23":
            continue
        x = pnl.loc[a:b]
        if meta["timing"] == "1545":
            x = x.loc["2024-01-02":]
        s = ann_stats(x)
        tt = tr[(tr.date >= a) & (tr.date <= b)] if len(tr) else tr
        yrs = len(x) / 252
        rows.append(dict(section="variant", variant=name, **meta, period=per, days=len(x), sharpe=s["sharpe"],
                         ann_ret=s["ann_ret"], ann_vol=s["ann_vol"], maxdd=s["maxdd"], tstat=s["tstat"],
                         trades=len(tt), trades_per_year=len(tt) / yrs if yrs else np.nan,
                         net_bp_per_trade=1e4 * tt.net.mean() if len(tt) else np.nan,
                         gross_bp_per_trade=1e4 * tt.gross.mean() if len(tt) else np.nan,
                         hit=(tt.net > 0).mean() if len(tt) else np.nan,
                         avg_hold_days=(tt.e - tt.t).mean() if len(tt) else np.nan))
    return rows


def main():
    D = setup()
    dates = D["dates"]
    qstarts = [i for i in range(1, len(dates)) if dates[i].month != dates[i - 1].month and dates[i].month in (1, 4, 7, 10)
               and dates[i] >= pd.Timestamp("2020-01-01")]
    pairs_by_q, plist = [], []
    for n, i_f in enumerate(qstarts):
        i_end = qstarts[n + 1] if n + 1 < len(qstarts) else len(dates)
        pr, nu, nc, nd = form(D, i_f)
        pairs_by_q.append(((i_f, i_end), pr))
        for sel, df in pr.items():
            plist.append(dict(section="pairs", formed=dates[i_f].date(), selection=sel, n_pairs=len(df), universe=nu,
                              cand_pairs=nc, cand_with_fund=nd, mean_rho=df.rho.mean(), mean_dist=df.dist.mean(),
                              n_industries=df.industry.nunique(),
                              overlap_with_corr=len(set(zip(df.a, df.b)) & set(zip(pr["corr"].a, pr["corr"].b)))))
        print("formed", dates[i_f].date(), nu, nc, flush=True)
    rows, series = [], {}
    for timing in ["close", "1545"]:
        for sel in ["corr", "fund", "combo"]:
            for k in [1, 3]:
                cands = candidates(D, pairs_by_q, sel, k, timing)
                for H in [1, 3, 5]:
                    for ex in (["time"] if H == 1 else ["time", "revert"]):
                        pnl, tr, nskip = simulate(D, cands, H, ex, timing)
                        name = f"{sel}_k{k}_H{H}_{ex}_{timing}"
                        meta = dict(selection=sel, k=k, hold=H, exit=ex, timing=timing, signals=len(cands),
                                    skipped_not_etb=nskip)
                        rows += summarize(name, meta, pnl, tr, dates)
                        series[name] = pnl
                print(timing, sel, k, "done", flush=True)
    V = pd.DataFrame(rows)
    # selection-bias check: best close-timing variant on 2020-23, deflated Sharpe over all variants tried
    from scipy.stats import skew, kurtosis
    dv = V[(V.period == "2020-23")].sort_values("sharpe", ascending=False)
    best = dv.iloc[0].variant
    rb = series[best].loc["2020-01-01":"2023-12-31"]
    sr = rb.mean() / rb.std()
    dsr = deflated_sharpe(sr, V.variant.nunique(), len(rb), skew(rb), kurtosis(rb, fisher=False))
    checks = D["checks"] + [dict(section="check", item="best_2020_23_variant", value=sr * np.sqrt(252), note=best),
                            dict(section="check", item="deflated_sharpe_best_2020_23", value=dsr,
                                 note=f"trials={V.variant.nunique()}")]
    spy = D["P"]["c"]["SPY"].astype("float64").pct_change()
    for per, a, b in PERIODS:
        checks.append(dict(section="check", item=f"SPY_buy_hold_sharpe_{per}", value=ann_stats(spy.loc[a:b])["sharpe"], note=""))
    out = pd.concat([pd.DataFrame(checks), V, pd.DataFrame(plist)], ignore_index=True)
    out.to_csv(os.path.join(RES, "study57_pairs.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(checks).to_string())
    print(V[["variant", "period", "sharpe", "ann_ret", "maxdd", "trades_per_year", "net_bp_per_trade",
             "gross_bp_per_trade", "hit", "avg_hold_days"]].round(3).to_string())


if __name__ == "__main__":
    main()
