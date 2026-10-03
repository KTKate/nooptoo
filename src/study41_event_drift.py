"""Study 41: multi-day drift after company announcements and analyst target changes.

Question: after an announcement that is known before the close of day t, does the stock keep moving relative to the
market over the next 1..60 trading days, measured from the CLOSE of t (no overnight-into-announcement return)?

Data: daily panels via src/horizon_lib.py (adjusted close C, survivorship-biased: only tickers alive in 2026);
analyst/announcement flags from src/analyst_features.py (Benzinga headlines, cut at 15:45 ET, so a flag on day t is
known before the close of t). Events: guidance raised (ev_guid_up>0), guidance cut (ev_guid_dn>0), guidance affirmed
(ev_guid_aff>0), buyback (ev_buyback>0), CEO/CFO change (ev_exec>0), target raise (pt_up>0 and pt_dn==0), target cut
(pt_dn>0 and pt_up==0), big target-gap change (pt_gap_chg20 in the top / bottom 5% of the day's universe, counted on
the first day it enters that tail; pt_gap only exists from 2023-12, so this type has no 2020-23 sample).
Universe: horizon_lib's liquid filter (price > $5, 20d median dollar volume > $20M) on day t. horizon_lib's traded
close is NaN before 2023-12, so the price filter uses traded close where known, else the Yahoo close scaled by each
ticker's split factor at the first traded-close day (later splits; splits before 2023-12 are not corrected).

Design:
  A. Event study. Excess return of an event = C[t+h]/C[t]-1 minus the equal-weight universe mean of the same window,
     h in {1,5,10,20,60}. Events are averaged per event date; the t-stat is on that date series with Newey-West lag h
     (clusters events of the same day and handles overlapping windows). Periods 2020-23 and 2024-26.
  B. Calendar-time portfolio for each type and h: each day hold (equal weight) every name with an event in the last h
     days (entered at the close of the event day, exited at the close of t+h); daily excess = portfolio return minus the
     universe equal-weight return; mean (bp/day, annualized) and t (Newey-West lag 5).
  C. Tradable version for the best types (chosen on 2020-23 calendar-time t, h>=5): same portfolio net of costs
     (horizon_lib COST per side, charged once at entry and once at exit), long only, compared with SPY.
Outputs: results/study41_event_drift.csv (section column: event / caltime / tradable)
"""
import os
import numpy as np
import pandas as pd
import horizon_lib as H
from core import RES

PERIODS = (("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31"), ("all", "2020-01-01", "2026-12-31"))
HS = (1, 5, 10, 20, 60)
_RV = None


def univ_ext(min_px=5, min_adv=20e6):
    """horizon_lib.universe() extended before 2023-12 with split-scaled Yahoo closes."""
    return (traded_px() > min_px) & (H.ADV > min_adv) & H.C.notna()


def traded_px():
    """Traded close where known (2023-12+), else Yahoo close x (traded/Yahoo ratio on the first traded-close day)."""
    raw = H.P["rawc"][H.cols].astype("float64")
    ratio = (H.TC / raw.where(raw > 0)).bfill().iloc[0].fillna(1.0)
    return H.TC.fillna(raw * ratio)


def nw_t(x, lag):
    """Mean and Newey-West t-stat of a series."""
    x = pd.Series(x).dropna().values.astype("float64")
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    m = x.mean()
    e = x - m
    v = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return m, m / np.sqrt(v / n), n


def ct_portfolio(entries, h, univ, cost=None, max_names=None, priority=None):
    """Calendar-time portfolio. entries: bool date x ticker (enter at the close of that date), hold h trading days.
    Re-entry while held extends the exit date (no extra cost). max_names: capacity; new entries ranked by priority.
    Returns daily DataFrame: n, gross, cost, net, bench (universe EW, universe of the previous close), spy, excess."""
    global _RV
    if _RV is None:
        _RV = (H.C / H.C.shift(1) - 1).astype("float64").values
    Rv = _RV
    uv = univ.shift(1, fill_value=False).values
    bench = np.nanmean(np.where(uv, Rv, np.nan), axis=1)
    E = entries.reindex(index=H.days, columns=H.cols).fillna(False).values
    Cv = H.COST.values if (cost is None or cost is False) else cost.values
    pr = None if priority is None else priority.reindex(index=H.days, columns=H.cols).values
    held = {}  # col -> exit index
    out = []
    spy = H.SPY.pct_change().values
    for i in range(len(H.days)):
        names = list(held)
        n = len(names)
        g = np.nan_to_num(Rv[i, names]).mean() if n else 0.0
        # closes of day i: exits then entries
        exits = [c for c in names if held[c] <= i]
        for c in exits:
            del held[c]
        new = [c for c in np.flatnonzero(E[i]) if c not in held]
        ext = [c for c in np.flatnonzero(E[i]) if c in held]
        for c in ext:
            held[c] = i + h
        if max_names is not None and len(held) + len(new) > max_names:
            room = max(max_names - len(held), 0)
            if pr is not None and new:
                new = sorted(new, key=lambda c: -np.nan_to_num(pr[i, c], nan=-1e9))
            new = new[:room]
        for c in new:
            held[c] = i + h
        n_after = len(held)
        cst = 0.0
        if cost is not False:
            if exits and n:
                cst += np.nan_to_num(Cv[i, exits], nan=0.01).sum() / n
            if new and n_after:
                cst += np.nan_to_num(Cv[i, new], nan=0.01).sum() / n_after
        out.append((n, g, cst, bench[i], spy[i]))
    df = pd.DataFrame(out, index=H.days, columns=["n", "gross", "cost", "bench", "spy"])
    df["net"] = df.gross - df.cost
    df["excess"] = (df.gross - df.bench).where(df.n > 0)
    df["excess_net"] = (df.net - df.bench).where(df.n > 0)
    return df


def ann_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if len(r) > 20 and r.std() > 0 else np.nan


if __name__ == "__main__":
    import analyst_features as AF
    U = univ_ext()
    A = AF.load()
    ev = {}
    for k, name in [("ev_guid_up", "guid_up"), ("ev_guid_dn", "guid_dn"), ("ev_guid_aff", "guid_aff"),
                    ("ev_buyback", "buyback"), ("ev_exec", "exec_change")]:
        ev[name] = A[k].reindex(index=H.days, columns=H.cols) > 0
    up = A["pt_up"].reindex(index=H.days, columns=H.cols).fillna(0)
    dn = A["pt_dn"].reindex(index=H.days, columns=H.cols).fillna(0)
    ev["pt_raise"] = (up > 0) & (dn == 0)
    ev["pt_cut"] = (dn > 0) & (up == 0)
    g = A["pt_gap_chg20"].reindex(index=H.days, columns=H.cols).where(U)
    rk = g.rank(axis=1, pct=True)
    hi, lo = rk >= 0.95, rk <= 0.05
    ev["gapchg_top5"] = hi & ~hi.shift(1, fill_value=False)
    ev["gapchg_bot5"] = lo & ~lo.shift(1, fill_value=False)
    del A, up, dn, g, rk, hi, lo
    for k in ev:
        ev[k] = ev[k] & U
        ev[k].loc[:"2019-12-31"] = False

    rows = []
    # A. event study
    for h in HS:
        F = H.fwd(h).astype("float64")
        ex = F.sub(F.where(U).mean(axis=1), axis=0)
        for name, M in ev.items():
            x = ex.where(M)
            per_date = x.mean(axis=1)
            cnt = x.notna().sum(axis=1)
            for p, a, b in PERIODS:
                pdm, cn, xs = per_date.loc[a:b], cnt.loc[a:b], x.loc[a:b]
                m, t, nd = nw_t(pdm, h)
                vals = xs.stack()
                rows.append(dict(section="event", type=name, h=h, period=p, n_events=int(cn.sum()), n_dates=nd,
                                 mean_bp=1e4 * vals.mean() if len(vals) else np.nan, date_mean_bp=1e4 * m, t=t,
                                 hit=(vals > 0).mean() if len(vals) else np.nan,
                                 median_bp=1e4 * vals.median() if len(vals) else np.nan))
        del F, ex
    # B. calendar-time
    ct = {}
    for name, M in ev.items():
        for h in HS:
            df = ct_portfolio(M, h, U, cost=False)
            ct[(name, h)] = df
            for p, a, b in PERIODS:
                d = df.loc[a:b]
                m, t, nd = nw_t(d.excess, 5)
                rows.append(dict(section="caltime", type=name, h=h, period=p, n_dates=nd, avg_names=d.n[d.n > 0].mean(),
                                 mean_bp=1e4 * m if nd else np.nan, ann_excess=252 * m if nd else np.nan, t=t))
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    # C. tradable: best two types by 2020-23 calendar-time t (h >= 5; types with a 2020-23 sample)
    cand = res[(res.section == "caltime") & (res.period == "2020-23") & (res.h >= 5) & (res.n_dates > 200)]
    best = cand.sort_values("t", ascending=False).drop_duplicates("type").head(2)
    print("chosen on 2020-23:\n", best[["type", "h", "mean_bp", "t"]])
    for _, r in best.iterrows():
        df = ct_portfolio(ev[r.type], int(r.h), U)
        for p, a, b in PERIODS:
            d = df.loc[a:b]
            net = d.net.where(d.n > 0, 0.0)
            m, t, nd = nw_t(d.excess_net, 5)
            rows.append(dict(section="tradable", type=r.type, h=int(r.h), period=p, n_dates=nd,
                             avg_names=d.n[d.n > 0].mean(), invested=(d.n > 0).mean(),
                             mean_bp=1e4 * m, ann_excess=252 * m, t=t, cost_bp_day=1e4 * d.cost.mean(),
                             net_ann=(1 + net).prod() ** (252 / len(net)) - 1, net_sharpe=ann_sharpe(net),
                             spy_ann=(1 + d.spy).prod() ** (252 / len(d)) - 1, spy_sharpe=ann_sharpe(d.spy),
                             bench_sharpe=ann_sharpe(d.bench)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study41_event_drift.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    ev_t = res[res.section == "event"].pivot_table(index=["type", "period"], columns="h", values=["date_mean_bp", "t"])
    print(ev_t.round(1))
    print(res[res.section == "event"].pivot_table(index="type", columns="period", values="n_events", aggfunc="max"))
    print(res[res.section == "caltime"].pivot_table(index=["type", "period"], columns="h", values=["ann_excess", "t"]).round(2))
    print(res[res.section == "tradable"].round(3).to_string())
