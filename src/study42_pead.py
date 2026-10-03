"""Study 42: post-earnings announcement drift (PEAD) over 20 and 60 trading days.

Question: after a quarterly report, do stocks with the best (worst) surprise keep outperforming (underperforming) the
market when bought at the close AFTER the reaction window?

Data: Nasdaq earnings calendar (store 'earnings': symbol, date, eps, epsForecast; `time` is 'time-not-supplied' for
99% of rows, so announcement timing is unknown), daily panels via src/horizon_lib.py (survivorship-biased).
Surprise measures, both known at the close of t+1 (t = first trading day on/after the calendar date):
  eps   : (eps - epsForecast) / price at close t-1. eps is as reported (not split-adjusted), so the price is the traded
          close where known (2023-12+) and before that the Yahoo close scaled by the split factor of splits after
          2023-12; splits during 2020-23 are not corrected (a handful of large caps get a too-large |surprise|).
  react : C[t+1]/C[t-1] - 1 minus the universe equal-weight return of the same window (2-day window covers both
          before-open and after-close reports).
Design: universe = liquid filter (price > $5, ADV > $20M; study41.univ_ext) at the entry close t+1. Reports are sorted
into quintiles within the calendar month of entry (descriptive; uses the month's breakpoints). Holding windows start
at the close of t+1 and end at the close of t+1+h, h in {20, 60}; excess = return minus universe EW mean of the same
window. Q5, Q1: per-entry-date means, Newey-West t (lag h). Q5-Q1: monthly differences, Newey-West t (lag h/21).
Periods 2020-23, 2024-26.
Tradable: long only, top quintile by trailing breakpoints (80th percentile of the measure over reports entered in the
previous 63 trading days, no look-ahead), entered at the close of t+1, held h days, max 20 names (new names ranked by
the measure when full), equal weight, horizon_lib COST per side at entry and exit (study41.ct_portfolio).
Output: results/study42_pead.csv (section: quintile / tmb / tradable)
"""
import os
import numpy as np
import pandas as pd
import store
import horizon_lib as H
from core import RES
from study41_event_drift import univ_ext, traded_px, nw_t, ct_portfolio, ann_sharpe, PERIODS

HS = (20, 60)

if __name__ == "__main__":
    U = univ_ext()
    PX = traded_px()

    E = store.read("earnings")
    print(E.time.value_counts())
    E = E[E.symbol.isin(set(H.cols)) & (E.date >= "2019-12-01")].copy()
    E["ti"] = np.searchsorted(H.days.values, E.date.values.astype("datetime64[ns]"))
    E = E[(E.ti >= 1) & (E.ti + 1 < len(H.days))]
    E = E.sort_values(["symbol", "ti", "eps"], na_position="last").drop_duplicates(["symbol", "ti"])
    ci = pd.Index(H.cols).get_indexer(E.symbol)
    ti = E.ti.values
    Cv = H.C.values.astype("float64")
    UM = U.values
    E["entry"] = H.days[ti + 1]
    E["inU"] = UM[ti + 1, ci]
    r2 = Cv[ti + 1, ci] / Cv[ti - 1, ci] - 1
    F2 = H.fwd(2).astype("float64")
    bench2 = F2.where(U).mean(axis=1).values
    E["react"] = r2 - bench2[ti - 1]
    px = PX.values[ti - 1, ci]
    E["eps_s"] = (E.eps - E.epsForecast) / np.abs(px)
    for h in HS:
        F = H.fwd(h).astype("float64")
        b = F.where(U).mean(axis=1).values
        E[f"x{h}"] = F.values[ti + 1, ci] - b[ti + 1]
        del F
    E = E[E.inU & (E.entry >= "2020-01-01")].copy()
    E["month"] = E.entry.dt.to_period("M")
    print("reports in universe:", len(E), " with eps surprise:", E.eps_s.notna().sum())

    rows = []
    for meas in ["eps", "react"]:
        col = "eps_s" if meas == "eps" else "react"
        d = E[E[col].notna()].copy()
        d = d[d.groupby("month")[col].transform("count") >= 25]
        d["q"] = d.groupby("month")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
        for h in HS:
            x = f"x{h}"
            for p, a, b in PERIODS:
                s = d[(d.entry >= a) & (d.entry <= b) & d[x].notna()]
                for q in range(1, 6):
                    sq = s[s.q == q]
                    m, t, nd = nw_t(sq.groupby("entry")[x].mean(), h)
                    rows.append(dict(section="quintile", measure=meas, h=h, period=p, q=q, n=len(sq), n_dates=nd,
                                     mean_bp=1e4 * sq[x].mean(), date_mean_bp=1e4 * m, t=t, hit=(sq[x] > 0).mean(),
                                     sig_median=sq[col].median()))
                mm = s.groupby(["month", "q"])[x].mean().unstack()
                tmb = (mm[5] - mm[1]).sort_index()
                m, t, nm = nw_t(tmb, int(np.ceil(h / 21)))
                rows.append(dict(section="tmb", measure=meas, h=h, period=p, n=len(s), n_dates=nm, mean_bp=1e4 * m, t=t,
                                 hit=(tmb > 0).mean()))
    res = pd.DataFrame(rows)

    # tradable: trailing 63-day breakpoints
    for meas in ["eps", "react"]:
        col = "eps_s" if meas == "eps" else "react"
        d = E[E[col].notna()][["entry", "symbol", col]]
        byday = d.groupby("entry")[col].apply(list).reindex(H.days)
        thr = pd.Series(np.nan, index=H.days)
        buf = []
        vals = byday.values
        for i in range(len(H.days)):
            win = [v for L in vals[max(0, i - 63):i] if isinstance(L, list) for v in L]
            if len(win) >= 100:
                thr.iloc[i] = np.quantile(win, 0.8)
        d = d.assign(thr=thr.reindex(d.entry).values)
        top = d[d[col] >= d.thr]
        ent = pd.DataFrame(False, index=H.days, columns=H.cols)
        pri = pd.DataFrame(np.nan, index=H.days, columns=H.cols)
        for e, s, v in top[["entry", "symbol", col]].itertuples(index=False):
            ent.at[e, s] = True
            pri.at[e, s] = v
        for h in HS:
            df = ct_portfolio(ent, h, U, max_names=20, priority=pri)
            for p, a, b in PERIODS:
                q = df.loc[a:b]
                net = q.net.where(q.n > 0, 0.0)
                m, t, nd = nw_t(q.excess_net, 5)
                rows.append(dict(section="tradable", measure=meas, h=h, period=p, n=int(len(top[(top.entry >= a) & (top.entry <= b)])),
                                 n_dates=nd, avg_names=q.n[q.n > 0].mean(), invested=(q.n > 0).mean(), mean_bp=1e4 * m,
                                 ann_excess=252 * m, t=t, cost_bp_day=1e4 * q.cost.mean(),
                                 net_ann=(1 + net).prod() ** (252 / len(net)) - 1, net_sharpe=ann_sharpe(net),
                                 spy_ann=(1 + q.spy).prod() ** (252 / len(q)) - 1, spy_sharpe=ann_sharpe(q.spy),
                                 bench_sharpe=ann_sharpe(q.bench)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study42_pead.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    qq = res[res.section == "quintile"]
    print(qq.pivot_table(index=["measure", "h", "period"], columns="q", values="date_mean_bp").round(0))
    print(qq.pivot_table(index=["measure", "h", "period"], columns="q", values="t").round(2))
    print(qq.pivot_table(index=["measure", "h", "period"], columns="q", values="n").round(0))
    print(res[res.section == "tmb"].round(2).to_string())
    print(res[res.section == "tradable"].round(3).to_string())
