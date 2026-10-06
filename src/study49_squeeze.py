"""Study 49: short-squeeze setups held 1-5 days.

Inputs, all on day t:
  short_ratio, short_z  FINRA Reg SHO daily short volume / total volume of day t-1 and its z-score against the
                        prior 60 days (news_features panels; published after the close of t-1, known before t).
                        This is short-sale VOLUME (largely market-maker hedging), not short interest.
  n_news, sent          Benzinga articles for the symbol in (15:45 of t-1, 15:45 of t] and their mean sentiment.
  breakout              close of t at a 20-day closing high (t-19 .. t) and volume of t > 2x the mean volume of
                        t-20 .. t-1 (panel P).
  r                     close-to-close return of t.
The breakout and large-up-day conditions use the official close and full-day volume of t; a live version would use
the 15:45 price and volume projected to the close (some names cross the thresholds only in the last 15 minutes).
Entry: closing auction of t. Exits: opening auction of t+1 ('on') or closing auction of t+1..t+5 ('c1'..'c5').
Setups (variants):
  A  short_z > {1.5, 2, 3} & breakout & news filter {pos: n_news > 0 and sent > 0, any: n_news > 0, none}
  B  short_ratio > {cross-sectional 90th percentile of the day, 0.6} & r > {5%, 10%}
  ref breakout alone; r > 5% alone.
Universe (two versions): traded price of t-1 > $5 (Yahoo raw close times later splits) and 20-day median dollar
volume through t-1 > $5M ('adv5') or > $20M ('adv20'). Dropped: a split dated t-5 .. t+6; Yahoo vs Alpaca traded
close returns disagreeing by > 5 points on day t (bad prints, 2023-12+).
Statistics: excess = event return - equal-weight mean of the same return over the same universe on day t.
  excess_datew_bp mean of entry-date averages; t_date its Newey-West t (h-1 lags); median and 1/99% trimmed means.
Tradable book: up to 10 names at 10% each (cash otherwise), entered at the close of t from that day's events
(highest short_z first for A, largest r first for B and refs), held to the exit; cost per side auction
(core.exec_cost_bps on traded prices) + 2.5 bp. Net bp/day, Sharpe, max drawdown, vs SPY buy and hold.
Selection: at most 2 (setup, universe, exit) cells chosen on 2020-23 only (highest book net Sharpe among A/B cells,
not the references, with >= 100 events and t_date > 2), then read on 2024-26.
Overlap: share of events that are in the overnight blend's top 10 on the same evening (study 68 score, 2024+).
Periods: 2020-23, 2024-26, 2024H1-25H1, 2025H2-26 (to 2026-09-25, end of the news archive).
Output: results/study49_squeeze.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt
import news_features as NF
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA

END = pd.Timestamp("2026-09-25")
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30"),
       ("2024H1-25H1", "2024-01-01", "2025-06-30"), ("2025H2-26", "2025-07-01", "2026-09-30")]
EXITS = {"on": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5}

P = load_panel()
cols = stock_cols(P)
alld = P["c"].index
N = len(alld)
ci = {t: i for i, t in enumerate(cols)}
o, c = P["o"][cols].astype("float64"), P["c"][cols].astype("float64")
v = P["v"][cols].astype("float32")

s = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
s = s[s.ticker.isin(cols)]
fac = np.ones((N, len(cols)))
nearsplit = np.zeros((N, len(cols)), bool)
for t, d, rt in zip(s.ticker, s.date, s.ratio):
    fac[alld < d, ci[t]] *= rt
    i = alld.searchsorted(d)
    nearsplit[max(0, i - 6): i + 6, ci[t]] = True
TP = P["rawc"][cols] * fac
Q = dict(P)
Q["rawc"] = TP.reindex(columns=P["c"].columns).astype("float32")
COST = ((exec_cost_bps(Q, "auction")[cols] + 2.5) / 1e4).astype("float64")
del Q, fac
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
r = c / c.shift(1) - 1
base_ok = (TP.shift(1) > 5) & c.notna() & c.shift(1).notna() & ~pd.DataFrame(nearsplit, index=alld, columns=cols)
base_ok = base_ok.mul(pd.Series(alld <= END, index=alld), axis=0).astype(bool)
TC = traded_close(P)[cols]
r_a = TC / TC.shift(1) - 1
bad = r_a.notna() & r.notna() & ((r_a - r).abs() > 0.05)
del TC, r_a
UNIVS = {"adv5": base_ok & (adv > 5e6) & ~bad, "adv20": base_ok & (adv > 20e6) & ~bad}

F = NF.load()
SR = F["short_ratio"].reindex(index=alld, columns=cols).astype("float64")
SZ = F["short_z"].reindex(index=alld, columns=cols).astype("float64")
NN = F["n_news"].reindex(index=alld, columns=cols).astype("float64")
SE = F["sent"].reindex(index=alld, columns=cols).astype("float64")
del F

hi20 = c >= c.rolling(20, min_periods=20).max() * (1 - 1e-9)
vr = v / v.shift(1).rolling(20, min_periods=15).mean()
brk = hi20 & (vr > 2) & (r > 0)
sr90 = SR.where(UNIVS["adv5"]).quantile(0.9, axis=1)
pos = (NN > 0) & (SE > 0)
anyn = NN > 0

SET = {}
negSZ, negr = -SZ, -r
for z in [1.5, 2, 3]:
    for nn, nm in [("pos", pos), ("any", anyn), ("none", None)]:
        m = (SZ > z) & brk
        if nm is not None:
            m &= nm
        SET[f"A_z>{z}_brk_news{nn}"] = (m, negSZ)
for rn, rm in [("sr>p90", SR.gt(sr90, axis=0)), ("sr>0.6", SR > 0.6)]:
    for up in [0.05, 0.10]:
        SET[f"B_{rn}_r>{int(up * 100)}%"] = (rm & (r > up), negr)
SET["ref_breakout"] = (brk, negr)
SET["ref_r>5%"] = (r > 0.05, negr)
del SE, NN, vr, hi20, v


def nw_t(x, lags):
    x = x.dropna().values
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    vv = e @ e / n
    for L in range(1, lags + 1):
        vv += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(vv / n) if vv > 0 else np.nan


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
    """As study 44: up to 10 names at 10%, entered at the close of t, held h days (h = 0: to the next open)."""
    Ev = E.values
    K = key.values
    cv, ov = c.values, o.values
    Cv = COST.values
    gross = np.zeros(N)
    cost = np.zeros(N)
    wsum = np.zeros(N)
    held = {}
    for t in range(N - 1):
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
        if h == 0:
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
            held[j] = min(t + h, N - 1)
            cost[t] += 0.10 * (Cv[t, j] if not np.isnan(Cv[t, j]) else 0.01)
            free -= 1
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "wsum": wsum}, index=alld)


def main():
    rows = []
    spy = P["c"]["SPY"].pct_change()
    BW = blend_W().reindex(index=alld, columns=cols).fillna(False).astype(bool)
    RX = {ex: (o.shift(-1) / c - 1 if h == 0 else c.shift(-h) / c - 1) for ex, h in EXITS.items()}
    for un, U in UNIVS.items():
        bench = {ex: R.where(U).clip(-0.9, 3).mean(axis=1) for ex, R in RX.items()}
        ud = r.where(U.shift(1, fill_value=False)).mean(axis=1)
        u_on = RX["on"].where(U).mean(axis=1)
        for sn, (sig, key) in SET.items():
            m = (sig & U).fillna(False)
            for ex, h in EXITS.items():
                R = RX[ex]
                X = R.sub(bench[ex], axis=0).where(m)
                book = slot_book(m, key.where(m), h)
                ub = (book.wsum * (u_on.shift(1) if h == 0 else ud)).fillna(0)
                for per, a_, b_ in PER:
                    x = X.loc[a_:b_].stack().dropna()
                    if len(x) < 10:
                        rows.append(dict(setup=sn, univ=un, exit=ex, period=per, n_events=len(x)))
                        continue
                    edate = x.groupby(level=0).mean()
                    bk = book.loc[a_:b_].loc[:END]
                    st = ann_stats(bk.net)
                    sh = ann_stats(bk.net - ub.loc[bk.index])
                    row = dict(setup=sn, univ=un, exit=ex, period=per, n_events=len(x), n_dates=len(edate),
                               ev_per_day=len(x) / max(len(bk), 1),
                               raw_bp=1e4 * R.where(m).loc[a_:b_].stack().dropna().mean(),
                               excess_bp=1e4 * x.mean(), excess_med_bp=1e4 * x.median(),
                               excess_trim_bp=1e4 * x.clip(x.quantile(.01), x.quantile(.99)).mean(),
                               excess_datew_bp=1e4 * edate.mean(), t_date=nw_t(edate, max(h - 1, 0)),
                               hit=(x > 0).mean(), book_names=10 * bk.wsum.mean(), book_gross_bp=1e4 * bk.gross.mean(),
                               book_cost_bp=1e4 * bk.cost.mean(), book_net_bp=1e4 * bk.net.mean(),
                               book_sharpe=st["sharpe"], book_maxdd=st["maxdd"], book_ann=st["ann_ret"],
                               hedged_net_bp=1e4 * (bk.net - ub.loc[bk.index]).mean(), hedged_sharpe=sh["sharpe"],
                               spy_sharpe=ann_stats(spy.loc[a_:b_].loc[:END])["sharpe"],
                               spy_maxdd=ann_stats(spy.loc[a_:b_].loc[:END])["maxdd"])
                    if a_ >= "2024-01-01":
                        mm = m.loc[a_:b_]
                        bw = BW.loc[a_:b_]
                        row["share_in_blend"] = (mm & bw).sum().sum() / max(mm.sum().sum(), 1)
                        row["blend_picks_that_are_events"] = (mm & bw).sum().sum() / max(bw.sum().sum(), 1)
                    rows.append(row)
            print(un, sn, "done", flush=True)
    res = pd.DataFrame(rows)
    # selection on 2020-23
    d = res[(res.period == "2020-23") & (res.n_events >= 100) & (res.t_date > 2) & ~res.setup.str.startswith("ref")
            ].sort_values("book_sharpe",
                                                                                               ascending=False)
    chosen = d.head(2)[["setup", "univ", "exit"]].values.tolist()
    res["chosen"] = [([a, b, e] in chosen) for a, b, e in zip(res.setup, res.univ, res.exit)]
    print("chosen on 2020-23:", chosen, flush=True)
    res.to_csv(os.path.join(RES, "study49_squeeze.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 600)
    pv = res[res.period.isin(["2020-23", "2024-26"])].pivot_table(
        index=["univ", "setup", "exit"], columns="period",
        values=["n_events", "excess_datew_bp", "excess_med_bp", "t_date", "book_net_bp", "book_sharpe"])
    print(pv.round(2).to_string())
    print(res[res.chosen][["setup", "univ", "exit", "period", "n_events", "excess_datew_bp", "excess_med_bp", "t_date",
                           "book_net_bp", "book_sharpe", "book_maxdd", "hedged_sharpe", "spy_sharpe", "spy_maxdd",
                           "share_in_blend", "blend_picks_that_are_events"]].round(3).to_string())
    # data check: largest event returns
    m = (SET["A_z>2_brk_newsany"][0] | SET["B_sr>0.6_r>10%"][0]) & UNIVS["adv5"]
    x = RX["c5"].where(m).stack().dropna().sort_values()
    print("\nlargest 5-day returns after a setup:\n", pd.concat([x.tail(8), r.stack().reindex(x.tail(8).index).rename("r_t")], axis=1))
    print("worst:\n", pd.concat([x.head(5), r.stack().reindex(x.head(5).index).rename("r_t")], axis=1))


if __name__ == "__main__":
    main()
