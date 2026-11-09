"""Study 207: which published stock-return signals still work, and do any work at short holds?

Source: Chen and Zimmermann, Open Source Asset Pricing, data release 2026.10 (Hugging Face dataset
OpenSourceAP/CrossSection, branch Data-Release-2026.10; the same files the `openassetpricing` package reads). Monthly
portfolio returns end 2025-12 (option-based predictors end 2022-12, a few others earlier). Files cached in
data/local/osap/ by `python src/study207_published_signals.py fetch`:
  SignalDoc.csv (paper, sample start/end years, sign), PredictorPortsFull.parquet (original-paper long-short "LS"),
  PredictorAltPorts_QuintilesEW / QuintilesVW (quintile 05 = predicted-high side, LS = 05 - 01),
  PredictorAltPorts_LiqScreen_Price_gt_5 (original method on stocks above $5).

Step 1 (no stock mapping): for each of the 212 predictors, mean monthly long-short return (percent) in the paper's
sample (SampleStartYear..SampleEndYear), after publication (from January of the year after the paper's year) and
in 2020-01..2025-12 ("recent"), with t-stats. Survivor = recent LS mean > 0 with t > 2 (at least 48 recent months).
Long-only: top quintile minus the average of the five quintiles (EW and VW), as a stand-in for "top quintile minus
market" (the market of stocks that have the signal). Multiple testing: expected chance passes at one-sided t > 2
(about 2.3% of those tested) and two-sided 5%; Benjamini-Hochberg at FDR 5% on one-sided p; Harvey-Liu-Zhu t > 3.
Output: results/study207_step1.csv (all predictors), results/study207_step1_summary.csv.

Step 2: rebuild up to 5 step-1 survivors that need only data we have, on our universe, and trade them long-only.
  Universe on day t: the 500 stocks with the highest 20-day median dollar volume up to t-1, traded close of t-1 above
  $5 (data/store daily panel from 2019-06, src/core.load_panel; ETFs excluded). The panel holds only tickers alive in
  2026 (survivorship): every result is also reported as excess over the same universe's equal-weight average.
  Signals are computed from closes up to day t (the decision is taken just before the closing auction of t, using
  the 15:45-ish price; using the close itself is a small look-ahead that matters only for the 1-day reversal and is
  absent from the slower signals; see the docstring of signals()).
  Portfolio: every h trading days buy the top N (equal weight) in the closing auction, sell in the closing auction
  h days later; names kept across rebalances are not traded. Holds h in {5, 21, 63} days, N in {10, 20}; with
  h > 1 the book is run as h staggered sub-books (one rebalanced each day), so returns are daily and overlapping
  holds do not inflate t.  Cost per side: 1.3 + 0.1 x half-spread(core.half_spread_model "15:45") + 2.5 bp.
  Pre-registered protocol (written before any step-2 run): choose, for each rebuilt signal, the (h, N) with the best
  net Sharpe on 2020-01..2023-12 only, write results/study207_frozen.json, then run 2024-01..2026-10 once.
  Pass (per signal, on the confirmation period): net Sharpe > 1 and above SPY's Sharpe over the same days, and
  t > 2 on the daily net excess over the universe's equal-weight average. No changes after the confirmation run.
    python src/study207_published_signals.py fetch | step1 | discover | confirm
Output: results/study207_step2.csv (grid and confirmation), results/study207_frozen.json,
        results/study207_daily.csv (confirmation daily returns)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as ss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OSAP = os.path.join(ROOT, "data", "local", "osap")
FROZEN = os.path.join(RES, "study207_frozen.json")
HF = "https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/Data-Release-2026.10/"
FILES = {"SignalDoc.csv": "SignalDoc.csv",
         "PredictorPortsFull.parquet": "Portfolios/Full Sets OP/PredictorPortsFull.parquet",
         "PredictorAltPorts_QuintilesEW.parquet": "Portfolios/Full Sets Alt/PredictorAltPorts_QuintilesEW.parquet",
         "PredictorAltPorts_QuintilesVW.parquet": "Portfolios/Full Sets Alt/PredictorAltPorts_QuintilesVW.parquet",
         "PredictorAltPorts_LiqScreen_Price_gt_5.parquet":
             "Portfolios/Full Sets Alt/PredictorAltPorts_LiqScreen_Price_gt_5.parquet"}
RECENT = ("2020-01-01", "2025-12-31")


def fetch():
    import requests
    from urllib.parse import quote
    os.makedirs(OSAP, exist_ok=True)
    for k, path in FILES.items():
        r = requests.get(HF + quote(path), timeout=300)
        r.raise_for_status()
        open(os.path.join(OSAP, k), "wb").write(r.content)
        print(k, len(r.content))


# ------------------------------------------------------------------------------------------------ step 1
def mt(x):
    x = x.dropna()
    n = len(x)
    if n < 3 or x.std() == 0:
        return n, np.nan, np.nan
    return n, x.mean(), x.mean() / x.std() * np.sqrt(n)


def step1():
    doc = pd.read_csv(os.path.join(OSAP, "SignalDoc.csv"))
    doc = doc[doc["Cat.Signal"] == "Predictor"].set_index("Acronym")
    op = pd.read_parquet(os.path.join(OSAP, "PredictorPortsFull.parquet"))
    op["date"] = pd.to_datetime(op.date)
    ls = op[op.port == "LS"].pivot(index="date", columns="signalname", values="ret")
    p5 = pd.read_parquet(os.path.join(OSAP, "PredictorAltPorts_LiqScreen_Price_gt_5.parquet"))
    p5["date"] = pd.to_datetime(p5.date)
    ls5 = p5[p5.port == "LS"].pivot(index="date", columns="signalname", values="ret")
    q = {}
    for w in ["EW", "VW"]:
        z = pd.read_parquet(os.path.join(OSAP, f"PredictorAltPorts_Quintiles{w}.parquet"))
        z["date"] = pd.to_datetime(z.date)
        z = z[z.port.isin(["01", "02", "03", "04", "05"])].pivot_table(index="date", columns=["signalname", "port"],
                                                                        values="ret")
        top = z.xs("05", axis=1, level=1)
        avg = z.T.groupby(level=0).mean().T            # mean of the 5 quintiles = covered-stock market
        q[w] = top - avg.reindex(columns=top.columns)
    a, b = RECENT
    rows = []
    for s in ls.columns:
        if s not in doc.index:
            continue
        d = doc.loc[s]
        x = ls[s].dropna()
        ins = x[(x.index.year >= d.SampleStartYear) & (x.index.year <= d.SampleEndYear)]
        post = x[x.index.year > d.Year]
        rec = x[(x.index >= a) & (x.index <= b)]
        r = dict(signal=s, paper=f"{d.Authors} ({int(d.Year)})", category=d["Cat.Data"], economic=d["Cat.Economic"],
                 predictability=d["Predictability in OP"], sample_end=int(d.SampleEndYear), pub_year=int(d.Year),
                 data_end=str(x.index.max().date()))
        for tag, y in [("insample", ins), ("postpub", post), ("recent", rec)]:
            n, m, t = mt(y)
            r.update({f"{tag}_n": n, f"{tag}_mean": m, f"{tag}_t": t})
        n, m, t = mt(ls5[s][(ls5.index >= a) & (ls5.index <= b)]) if s in ls5 else (0, np.nan, np.nan)
        r.update(recent_px5_mean=m, recent_px5_t=t)
        for w in ["EW", "VW"]:
            if s in q[w]:
                y = q[w][s]
                n, m, t = mt(y[(y.index >= a) & (y.index <= b)])
            else:
                m = t = np.nan
            r.update({f"recent_topq_{w}_mean": m, f"recent_topq_{w}_t": t})
        rows.append(r)
    r = pd.DataFrame(rows)
    tested = r.recent_n >= 48
    r["tested"] = tested
    # one-sided p (H1: mean > 0), Student t with n-1 df
    r["p_one"] = np.where(tested, ss.t.sf(r.recent_t, r.recent_n - 1), np.nan)
    m = int(tested.sum())
    pv = r.loc[tested, "p_one"].sort_values()
    bh_k = (pv.values <= 0.05 * np.arange(1, m + 1) / m).nonzero()[0]
    cut = pv.values[bh_k.max()] if len(bh_k) else -1
    r["survivor_t2"] = tested & (r.recent_mean > 0) & (r.recent_t > 2)
    r["pass_bh5"] = tested & (r.p_one <= cut)
    r["pass_t3"] = tested & (r.recent_t > 3)
    r["topq_EW_t2"] = tested & (r.recent_topq_EW_mean > 0) & (r.recent_topq_EW_t > 2)
    r["topq_VW_t2"] = tested & (r.recent_topq_VW_mean > 0) & (r.recent_topq_VW_t > 2)
    r["decay_post_vs_in"] = r.postpub_mean / r.insample_mean - 1
    r = r.sort_values("recent_t", ascending=False)
    r.to_csv(os.path.join(RES, "study207_step1.csv"), index=False)
    t = r[r.tested]
    summ = dict(
        predictors=len(r), tested_2020_2025=m,
        mean_insample=t.insample_mean.mean(), mean_postpub=t.postpub_mean.mean(), mean_recent=t.recent_mean.mean(),
        median_ratio_post_in=(t.postpub_mean / t.insample_mean).median(),
        median_ratio_recent_in=(t.recent_mean / t.insample_mean).median(),
        recent_positive=int((t.recent_mean > 0).sum()),
        pass_t2=int(t.survivor_t2.sum()), chance_t2_one_sided=round(m * ss.norm.sf(2), 1), chance_5pct=round(m * 0.05, 1),
        pass_bh5=int(t.pass_bh5.sum()), pass_t3=int(t.pass_t3.sum()),
        recent_negative_t2=int(((t.recent_mean < 0) & (t.recent_t < -2)).sum()),
        topq_EW_t2=int(t.topq_EW_t2.sum()), topq_VW_t2=int(t.topq_VW_t2.sum()),
        topq_EW_t3=int((t.recent_topq_EW_t > 3).sum()), topq_VW_t3=int((t.recent_topq_VW_t > 3).sum()),
        both_ls_t2_and_topqVW_t2=int((t.survivor_t2 & t.topq_VW_t2).sum()))
    pd.Series(summ).to_csv(os.path.join(RES, "study207_step1_summary.csv"), header=["value"])
    print(pd.Series(summ).to_string())
    cols = ["signal", "paper", "category", "insample_mean", "postpub_mean", "recent_mean", "recent_t", "recent_px5_t",
            "recent_topq_EW_t", "recent_topq_VW_t", "pass_bh5", "pass_t3"]
    print(r[r.survivor_t2 | r.topq_VW_t2 | r.topq_EW_t2][cols].round(2).to_string(index=False))


# ------------------------------------------------------------------------------------------------ step 2
SIGNALS = ["MomVol", "EarningsStreak", "Mom6m", "Coskewness"]
HS, NS = [5, 21, 63], [10, 20]
DISC = ("2020-01-01", "2023-12-31")
CONF = ("2024-01-01", "2026-10-31")


def panel():
    """Wide panels on the store's trading days: adjusted close C, daily return R (|R| > 60% treated as a data error and
    set to 0), universe U (decision day d: top 500 by 20-day median dollar volume of d-20..d-1 among stocks whose
    traded close of d-1 is above $5), auction cost per side CS (fraction)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import load_panel, stock_cols, traded_close, half_spread_model
    P = load_panel()
    allc = stock_cols(P)
    adv_all = P["dv"][allc].rolling(20, min_periods=10).median().shift(1)
    # candidates: ever in the top 1000 by dollar volume (the $5 filter can only promote names from below rank 500)
    cand = (adv_all.rank(axis=1, ascending=False) <= 1000).any()
    cols = list(cand[cand].index)
    adv = adv_all[cols].astype("float64")
    del adv_all
    C = P["c"][cols].astype("float64")
    # traded close (Alpaca raw) from 2023-12; earlier the Yahoo close scaled by the first traded/Yahoo ratio
    # (same construction as src/horizon_lib.py)
    TC = traded_close(P)[cols].astype("float64")
    raw = P["rawc"][cols].astype("float64")
    TPX = TC.fillna(raw * (TC / raw.where(raw > 0)).bfill().iloc[0].fillna(1.0))
    R = C / C.shift(1) - 1
    bad = R.abs() > 0.6
    print("daily returns beyond +-60% treated as errors (candidate names):", int(bad.sum().sum()))
    R = R.mask(bad)
    ok = (TPX.shift(1) > 5) & C.notna() & adv.notna()
    U = adv.where(ok).rank(axis=1, ascending=False, method="first") <= 500
    keep = U.any()
    keep = keep[keep].index
    vol = np.log(C / C.shift(1)).rolling(20, min_periods=10).std().shift(1)
    hs = half_spread_model(adv[keep], raw[keep].shift(1), vol[keep].clip(lower=1e-3), "15:45")
    CS = (1.3 + 0.1 * hs + 2.5) / 1e4
    spy = P["c"]["SPY"].astype("float64")
    return dict(P=P, C=C[keep], Call=C, R=R[keep], U=U[keep], CS=CS, spy=spy / spy.shift(1) - 1, cols=list(keep))


def signals(D):
    """Scores on decision day d (computed with the close of d; the trade is the closing auction of d, so the live
    version would use a price a few minutes before the close; all four signals change slowly with one day's price).
      Mom6m          C[d-21] / C[d-126] - 1 (months t-6..t-1, skipping the latest month)
      MomVol         Mom6m, kept only for the top third of the universe by 6-month average turnover; turnover =
                     dollar volume / market cap, market cap = today's market cap (data/store/sectors.parquet) scaled
                     by the adjusted price ratio (shares held constant: no point-in-time share counts available)
      Coskewness     daily version: E[ri~ rm~^2] / (sd(ri) var(rm)) over 252 days, rm = SPY; score = -coskewness
                     (the paper uses 60 months of monthly returns, impossible from 2019-06 data)
      EarningsStreak surprise = (eps - consensus) / close before the announcement (data/store/earnings); kept only
                     when it has the same sign as the previous surprise; usable from the first trading day after the
                     announcement date (time of day unknown) for 63 trading days or until the next announcement
    """
    C, U = D["C"], D["U"]
    days = C.index
    out = {}
    mom = C.shift(21) / C.shift(126) - 1
    out["Mom6m"] = mom
    mc = pd.read_parquet(os.path.join(ROOT, "data", "store", "sectors.parquet"))
    mc = pd.to_numeric(mc.set_index("ticker").marketCap, errors="coerce")
    mc = mc[mc > 0].reindex(C.columns)
    last = C.ffill().iloc[-1]
    mcap = C / last * mc
    dv = D["P"]["dv"][C.columns].astype("float64")
    turn = (dv / mcap).rolling(126, min_periods=100).mean()
    tr = turn.where(U).rank(axis=1, pct=True)
    out["MomVol"] = mom.where(tr > 2 / 3)
    ri = D["R"]
    rm = D["spy"].reindex(days)
    M = ri.notna()
    rmM = M.mul(rm, axis=0).where(M)
    W, mp = 252, 240
    rl = lambda x: x.rolling(W, min_periods=mp).mean()
    a, b = rl(ri), rl(rmM)
    e_rm2 = rl(rmM ** 2)
    e_rirm = rl(ri * rmM)
    e_rirm2 = rl(ri * rmM ** 2)
    num = e_rirm2 - 2 * b * e_rirm - a * e_rm2 + 2 * a * b ** 2
    sdi = np.sqrt((rl(ri ** 2) - a ** 2).clip(lower=0))
    varm = (e_rm2 - b ** 2).clip(lower=1e-12)
    out["Coskewness"] = -(num / (sdi * varm))
    out["EarningsStreak"] = earnings_streak(D)
    return out


def earnings_streak(D):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import store
    C = D["Call"]
    days = C.index
    e = store.read("earnings")[["symbol", "date", "eps", "epsForecast"]].dropna()
    e = e.rename(columns={"symbol": "ticker"})
    e = e[e.ticker.isin(D["C"].columns)].sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    # price: close of the last trading day before the announcement date
    pos_before = np.searchsorted(days.values, e.date.values, side="left") - 1
    e = e[pos_before >= 0]
    pos_before = pos_before[pos_before >= 0]
    raw = D["P"]["rawc"]
    px = np.array([raw.iat[i, raw.columns.get_loc(t)] if t in raw.columns else np.nan
                   for i, t in zip(pos_before, e.ticker)])
    e["surp"] = (e.eps - e.epsForecast) / px
    e = e[e.surp.abs() < 0.5]
    e["prev"] = e.groupby("ticker").surp.shift(1)
    e["streak"] = np.sign(e.surp) == np.sign(e.prev)
    e["streak"] &= e.surp != 0
    # usable from the first trading day strictly after the announcement date
    e["start"] = np.searchsorted(days.values, e.date.values, side="right")
    e["nxt"] = e.groupby("ticker").start.shift(-1).fillna(len(days)).astype(int)
    a = np.full((len(days), D["C"].shape[1]), np.nan)
    ci = {t: i for i, t in enumerate(D["C"].columns)}
    for t, st, nx, sp, k in zip(e.ticker, e.start, e.nxt, e.surp, e.streak):
        if not k or st >= len(days):
            continue
        a[st:min(nx, st + 63), ci[t]] = sp
    return pd.DataFrame(a, index=days, columns=D["C"].columns)


def run(D, score, h, N):
    """Staggered top-N long book: h sub-books, sub-book o rebalances on decision days o, o+h, ...; each picks the top N
    of the universe by score in the closing auction and holds h days. Daily return = mean over sub-books. Cost per
    traded name and side CS[d]; charged on the next day's return."""
    R, U, CS = D["R"], D["U"], D["CS"]
    days = R.index
    Rv = R.fillna(0).values
    Sv = score.reindex(index=days, columns=R.columns).where(U).values
    Cv = CS.fillna(0.01).values
    T = len(days)
    first = int(np.argmax(np.isfinite(Sv).sum(1) >= N))
    gross = np.zeros(T)
    cost = np.zeros(T)
    for o in range(h):
        prev = np.array([], dtype=int)
        for d in range(first + o, T - 1, h):
            s = Sv[d]
            v = np.isfinite(s)
            if v.sum() >= N:
                idx = np.where(v)[0]
                pick = idx[np.argsort(-s[idx], kind="stable")[:N]]
            else:
                pick = np.array([], dtype=int)
            traded = np.setxor1d(pick, prev)
            cost[d + 1] += Cv[d, traded].sum() / N / h
            e = min(d + h, T - 1)
            if len(pick):
                gross[d + 1:e + 1] += Rv[d + 1:e + 1][:, pick].sum(1) / N / h
            prev = pick
    out = pd.DataFrame({"gross": gross, "cost": cost}, index=days)
    out["net"] = out.gross - out.cost
    um = U.shift(1).fillna(False).astype(bool)
    out["univ"] = R.where(um).mean(1)
    out["spy"] = D["spy"].reindex(days)
    out["excess"] = out.net - out.univ
    out["gross_excess"] = out.gross - out.univ
    return out.iloc[first + h:]        # all h sub-books live


def st2(o):
    def sh(x):
        x = x.dropna()
        return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    x = o.excess.dropna()
    return dict(days=len(o), start=str(o.index[0].date()), end=str(o.index[-1].date()),
                gross_bp=1e4 * o.gross.mean(), cost_bp=1e4 * o.cost.mean(), net_bp=1e4 * o.net.mean(),
                net_ann=(1 + o.net).prod() ** (252 / len(o)) - 1, net_sharpe=sh(o.net), spy_sharpe=sh(o.spy),
                univ_sharpe=sh(o.univ), excess_bp=1e4 * x.mean(), excess_sharpe=sh(x),
                excess_t=x.mean() / x.std() * np.sqrt(len(x)), gross_excess_t=o.gross_excess.mean() /
                o.gross_excess.std() * np.sqrt(len(o)))


def discover():
    D = panel()
    S = signals(D)
    rows = []
    for k in SIGNALS:
        cov = S[k].where(D["U"]).notna().sum(1)
        print(k, "names with a score in the universe, median 2020-23:", cov.loc[DISC[0]:DISC[1]].median())
        for h in HS:
            for N in NS:
                o = run(D, S[k].loc[:DISC[1]], h, N).loc[DISC[0]:DISC[1]]
                rows.append(dict(tag="discover_2020_2023", signal=k, h=h, N=N, **st2(o)))
    g = pd.DataFrame(rows)
    print(g.round(3).to_string(index=False))
    fz = {}
    for k in SIGNALS:
        b = g[g.signal == k].sort_values("net_sharpe", ascending=False).iloc[0]
        fz[k] = dict(h=int(b.h), N=int(b.N), discovery_net_sharpe=float(b.net_sharpe),
                     discovery_excess_t=float(b.excess_t))
    fz["frozen_on"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    fz["pass_rule"] = ("confirmation 2024-01..2026-10: net Sharpe > 1 and > SPY Sharpe on the same days, "
                       "t > 2 on daily net excess over the universe equal-weight average")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(os.path.join(RES, "study207_step2.csv"), index=False)
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    D = panel()
    S = signals(D)
    rows, daily = [], []
    for k in SIGNALS:
        h, N = fz[k]["h"], fz[k]["N"]
        o = run(D, S[k].loc[:CONF[1]], h, N).loc[CONF[0]:CONF[1]]
        r = st2(o)
        r["pass"] = bool(r["net_sharpe"] > 1 and r["net_sharpe"] > r["spy_sharpe"] and r["excess_t"] > 2)
        rows.append(dict(tag="confirm_2024_2026", signal=k, h=h, N=N, **r))
        for y, oy in o.groupby(o.index.year):
            rows.append(dict(tag=f"confirm_{y}", signal=k, h=h, N=N, **st2(oy)))
        daily.append(o.assign(signal=k, h=h, N=N))
    # secondary (not used for the verdict): the whole grid on the confirmation period
    for k in SIGNALS:
        for h in HS:
            for N in NS:
                o = run(D, S[k].loc[:CONF[1]], h, N).loc[CONF[0]:CONF[1]]
                rows.append(dict(tag="grid_confirm_secondary", signal=k, h=h, N=N, **st2(o)))
    c = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(RES, "study207_step2.csv"))
    old = old[old.tag.str.startswith("discover")]
    pd.concat([old, c]).to_csv(os.path.join(RES, "study207_step2.csv"), index=False)
    pd.concat(daily).to_csv(os.path.join(RES, "study207_daily.csv"))
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    {"fetch": fetch, "step1": step1, "discover": discover, "confirm": confirm}[sys.argv[1]]()
