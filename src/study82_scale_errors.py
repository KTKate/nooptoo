"""Study 82: Yahoo price-scale errors (spin-off "splits" and other restatements) vs Alpaca, and whether they matter.

Study 64 found that in ~5% of 2020 top-500 stock-days the Yahoo close (data/store daily) sits on a different price
scale from Alpaca's split-adjusted bars, by a constant factor per ticker over a period (T 1.324, GE, MMM, IBM, MRK,
CMCSA ...): Yahoo books many spin-offs as splits and restates the whole earlier history with that ratio.

Data
  Yahoo: core.load_panel(): rawc = Yahoo close (split- and spin-off-"split"-adjusted to the store date), c = rawc x
    the dividend factor (q), o = Yahoo open x the same factor. dv = rawc x volume.
  Alpaca: 1Day SIP bars 2019-06 .. latest, adjustment='split' and adjustment='all' (split + dividend + spin-off),
    for every stock that ever passed the training universe (Yahoo rawc > 5 and 20-day median dv > $5M, 2019-06+)
    plus every ticker of data/local/d1s63.parquet -> data/local/d1s82_split.parquet, data/local/d1s82_all.parquet
    (python src/study82_scale_errors.py fetch).

(1) Extent. k_t = Yahoo rawc / Alpaca split close per ticker-day (and Yahoo c / Alpaca 'all' close). Flags:
    |k - 1| > 0.5% (off the Alpaca scale), |k / median_ticker(k) - 1| > 0.5% (off the ticker's own long-run level).
    Persistent ticker-periods = runs where the 21-day rolling median of k is off 1 by > 0.5% (single-day close
    disagreements, e.g. Yahoo last trade vs official close, are not periods).
    Level shifts (adjustment events) = days where the rolling median of k over the next 5 closes differs from the one
    over the previous 5 closes by > 0.5% and the day-to-day change k_t / k_{t-1} itself is > 0.5%.
(2) Does it matter?
    (a) Nights (close t -> open t+1, the overnight model's return) with a level shift between t and t+1 in the
        Yahoo/Alpaca-split ratio, or with a level shift in Yahoo c / Alpaca 'all' (a Yahoo restatement that Alpaca's
        full adjustment does not share), for the blend's daily top 10 (2024-26; study 69 baseline, auction cost + 2.5
        bp/side), the training frame (2020-23, universe rawc > 5 and adv20 > $5M), and the final_series ML ranker.
        Blend Sharpe recomputed (i) dropping flagged picks (cash), (ii) replacing their return with the Alpaca
        'all' night return, (iii) replacing every pick's return whose Yahoo night differs from Alpaca's by > 1%.
    (b) Universe filters 2020-23 (rawc > 5 and adv20 = 20-day median rawc x volume > $5M) recomputed with Alpaca's
        split close: tickers / ticker-days in and out.

    python src/study82_scale_errors.py fetch    # Alpaca daily bars (cached)
    python src/study82_scale_errors.py          # writes results/study82_scale_errors.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA

warnings.filterwarnings("ignore")
LOC = os.path.join(DATA, "local")
FN = {a: os.path.join(LOC, f"d1s82_{a}.parquet") for a in ["split", "all"]}
TH = 0.005


def fetch():
    import alpaca_data as A
    P = load_panel()
    cols = stock_cols(P)
    raw, dv = P["rawc"][cols], P["dv"][cols]
    adv20 = dv.rolling(20, min_periods=10).median()
    u = ((raw > 5) & (adv20 > 5e6)).loc["2019-06-01":]
    tk = set(u.columns[u.any().values])
    tk |= set(pd.read_parquet(os.path.join(LOC, "d1s63.parquet"), columns=["ticker"]).ticker.unique())
    tk = sorted(t.replace("-", ".") for t in tk if not t.startswith("^"))
    last = P["c"].index[-1].strftime("%Y-%m-%d")
    print("tickers", len(tk), "through", last, flush=True)
    for adj in ["split", "all"]:
        if os.path.exists(FN[adj]):
            print("cached", FN[adj])
            continue
        parts = []
        for i in range(0, len(tk), 200):
            p = A.bars(tk[i:i + 200], "1Day", "2019-06-01T00:00:00Z", f"{last}T23:59:00Z", adjustment=adj)
            if len(p):
                parts.append(p)
            print(adj, i, flush=True)
        d = pd.concat(parts, ignore_index=True)
        d["ticker"] = d.ticker.str.replace(".", "-", regex=False)
        d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
        d[["date", "ticker", "o", "h", "l", "c", "v"]].to_parquet(FN[adj], index=False)
        print(adj, len(d), d.ticker.nunique(), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["fetch"]:
    fetch()
    sys.exit(0)

rows = []


def add(part, item, **kw):
    rows.append(dict(part=part, item=item, **kw))


def piv(d, k, idx, cols):
    return d.pivot(index="date", columns="ticker", values=k).reindex(index=idx, columns=cols).astype("float64")


def shift_events(k, stab=0.002):
    """Scale breaks of the ratio k = Yahoo / Alpaca between close t-1 and close t (day t flagged).
    The 5-day median of log k over t..t+4 must differ from the one over t-5..t-1 by more than TH, the step
    log k_t - log k_{t-1} must be at least half that shift, and k must be *stable* on both sides (the range of
    log k inside each window below `stab`), so that day-to-day close disagreements on thin names (Yahoo last
    trade vs the official close) are not counted as scale breaks. Returns (flag, shift size in logs)."""
    lk = np.log(k)
    pre = lk.rolling(5, min_periods=3).median().shift(1)
    post = lk[::-1].rolling(5, min_periods=3).median()[::-1]
    rng = lambda x: x.rolling(5, min_periods=3).max() - x.rolling(5, min_periods=3).min()
    tight = (rng(lk).shift(1) < stab) & (rng(lk[::-1])[::-1] < stab)
    sh = post - pre
    step = lk - lk.ffill().shift(1)
    ev = (sh.abs() > np.log1p(TH)) & (step.abs() > 0.5 * sh.abs()) & tight
    return ev.fillna(False), sh


def runs(flag, k, top):
    """Ticker-periods: maximal runs of a boolean day x ticker flag -> rows."""
    out = []
    for t in flag.columns[flag.any().values]:
        f = flag[t].values
        d = flag.index
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j + 1 < len(f) and f[j + 1]:
                    j += 1
                seg = k[t].iloc[i:j + 1]
                out.append(dict(ticker=t, start=d[i].date(), end=d[j].date(), n_days=j - i + 1,
                                median_ratio=float(seg.median()), days_in_top500=int(top[t].iloc[i:j + 1].sum())))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(out)


def main():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index[P["c"].index >= "2019-06-01"]
    A_s = pd.read_parquet(FN["split"])
    A_a = pd.read_parquet(FN["all"])
    for d in (A_s, A_a):
        d["date"] = d.date.astype("datetime64[ns]")
    cols = sorted(set(cols) & set(A_s.ticker))
    sc, so, sv = (piv(A_s, k, days, cols) for k in ["c", "o", "v"])
    ac, ao = (piv(A_a, k, days, cols) for k in ["c", "o"])
    del A_s, A_a
    raw = P["rawc"][cols].reindex(days).astype("float64")
    yc = P["c"][cols].reindex(days).astype("float64")
    yo = P["o"][cols].reindex(days).astype("float64")
    yo_raw = yo * raw / yc
    v = P["v"][cols].reindex(days).astype("float64")
    dv = P["dv"][cols].reindex(days).astype("float64")
    adv20 = dv.rolling(20, min_periods=10).median()
    rk = adv20.shift(1).where(raw.shift(1) > 5).rank(axis=1, ascending=False)
    top500 = (rk <= 500)
    univ = (raw > 5) & (adv20 > 5e6)
    k = (raw / sc).where((raw > 0) & (sc > 0))                      # Yahoo raw / Alpaca split
    kA = (yc / ac).where((yc > 0) & (ac > 0))                       # Yahoo adjusted / Alpaca all
    ko = (yo_raw / so).where((yo_raw > 0) & (so > 0))
    print("coverage: Yahoo-Alpaca ticker-days", int(k.notna().sum().sum()), "tickers", len(cols), flush=True)
    print("close agreement |k-1| quantiles", (k - 1).abs().stack().quantile([.5, .9, .95, .99]).round(5).to_dict())
    print("open agreement |ko/k-1| quantiles", (ko / k - 1).abs().stack().quantile([.5, .9, .95, .99]).round(5).to_dict())
    print("vol ratio Yahoo/Alpaca split median", float((v / sv).stack().median()))

    # ---------------------------------------------------------------- (1) extent
    medk = k.median()
    medkA = kA.median()
    km = np.exp(np.log(k).rolling(21, center=True, min_periods=10).median())
    kAm = np.exp(np.log(kA).rolling(21, center=True, min_periods=10).median())
    for nm, kk, kmed, kroll in [("split: Yahoo rawc / Alpaca split", k, medk, km),
                                ("all: Yahoo c / Alpaca all", kA, medkA, kAm)]:
        f1 = (kk - 1).abs() > TH
        f2 = (kk / kmed - 1).abs() > TH
        fp = (kroll - 1).abs() > TH
        ok = kk.notna()
        for per, a, b in [("2020", "2020", "2020"), ("2021", "2021", "2021"), ("2022", "2022", "2022"),
                          ("2023", "2023", "2023"), ("2024", "2024", "2024"), ("2025-26", "2025", "2026"),
                          ("2020-23", "2020", "2023"), ("2024-26", "2024", "2026")]:
            for scope, m in [("all", ok), ("top500", ok & top500), ("train_univ", ok & univ)]:
                mm = m.loc[a:b]
                n = mm.sum().sum()
                add("extent", nm, period=per, scope=scope, n_ticker_days=int(n),
                    share_off_1=float((f1.loc[a:b] & mm).sum().sum() / n),
                    share_off_own_median=float((f2.loc[a:b] & mm).sum().sum() / n),
                    share_persistent_off_1=float((fp.loc[a:b] & mm).sum().sum() / n),
                    tickers_persistent=int((fp.loc[a:b] & mm).any().sum()),
                    tickers=int(mm.any().sum()))
    # ticker-periods (persistent, split basis, from 2020)
    fp = ((km - 1).abs() > TH) & k.notna()
    R = runs(fp.loc["2020":], k.loc["2020":], top500.loc["2020":])
    R = R[R.n_days >= 5].sort_values("days_in_top500", ascending=False)
    R.to_csv(os.path.join(SCR, "study82_periods.csv"), index=False) if SCR else None
    print("\npersistent ticker-periods (split basis, >= 5 days):", len(R), "tickers", R.ticker.nunique())
    print(R.head(40).to_string(index=False))
    for _, r in R.head(60).iterrows():
        add("period", r.ticker, period=f"{r.start}..{r.end}", n_ticker_days=int(r.n_days),
            ratio=r.median_ratio, days_in_top500=r.days_in_top500)
    # scale breaks and their direction: at a break, either Yahoo restated its history (so Alpaca's own series
    # shows the uncompensated spin-off price drop and Yahoo's does not -> Yahoo is the economically right one and
    # our returns are fine) or the Yahoo series itself steps (a restatement only partly applied in data/store ->
    # a fake return in our panel).
    spl = pd.read_csv(os.path.join(LOC, "events", "splits_yf.csv"), parse_dates=["date"])
    spl["ticker"] = spl.ticker.str.replace(".", "-", regex=False)
    yspl = set(zip(spl.ticker, spl.date))
    ev, mag = shift_events(k)
    univ1 = univ.shift(1).fillna(False).astype(bool)
    rY = np.log(raw / raw.shift(1))                                  # Yahoo raw close-to-close
    rA = np.log(sc / sc.shift(1))                                    # Alpaca split close-to-close
    E = []
    s_ = ev.loc["2019-07":].stack()
    for (d, t) in s_[s_].index:
        ry, ra, sh = float(rY.at[d, t]), float(rA.at[d, t]), float(mag.at[d, t])
        E.append(dict(date=d, ticker=t, shift_pct=100 * np.expm1(sh), yahoo_cc_pct=100 * np.expm1(ry),
                      alpaca_cc_pct=100 * np.expm1(ra), side="yahoo_steps" if abs(ry) > abs(ra) else "alpaca_steps",
                      yahoo_split_listed=(t, d) in yspl, in_univ=bool(univ1.at[d, t])))
    E = pd.DataFrame(E)
    E.to_csv(os.path.join(SCR, "study82_breaks.csv"), index=False) if SCR else None
    print("\nscale breaks (stable both sides):", len(E), "tickers", E.ticker.nunique())
    print(E.groupby(["side", "yahoo_split_listed", "in_univ"]).size().to_string())
    for _, g in E[E.in_univ].iterrows():
        add("break_in_univ", g.ticker, period=str(g.date.date()), side=g.side, shift_pct=g.shift_pct,
            yahoo_cc_pct=g.yahoo_cc_pct, alpaca_cc_pct=g.alpaca_cc_pct, yahoo_split_listed=g.yahoo_split_listed)
    for (sd, ys), g in E.groupby(["side", "yahoo_split_listed"]):
        add("breaks", f"{sd}, yahoo split listed={ys}", n=len(g), n_in_univ=int(g.in_univ.sum()),
            median_abs_shift_pct=float(g.shift_pct.abs().median()), tickers=g.ticker.nunique())
    print("\nbreaks in the training universe, largest first:")
    print(E[E.in_univ].reindex(E[E.in_univ].shift_pct.abs().sort_values(ascending=False).index)
          .head(45).round(3).to_string(index=False))
    # spin-off-type Yahoo splits (ratios that are not simple split fractions) since 2019-07
    s2 = spl[(spl.date >= "2019-07-01")]
    r2 = s2.ratio.values
    nonint = ~np.isclose(r2 * 2, np.round(r2 * 2)) & ~np.isclose(1 / r2, np.round(1 / r2)) & \
        ~np.isclose(r2 * 3, np.round(r2 * 3))
    frac = s2[nonint]
    add("breaks", "Yahoo non-integer split ratios listed 2019-07+ (spin-off type)", n=len(frac),
        tickers=frac.ticker.nunique(), n_in_univ=int(sum(bool(univ1.at[d, t]) for t, d in zip(frac.ticker, frac.date)
                                                     if t in univ1.columns and d in univ1.index)))

    # ---------------------------------------------------------------- (2a) nights with a scale change
    # night t -> t+1 is hit when an event (either basis) is dated t+1
    hit = ev.shift(-1).fillna(False).astype(bool)
    hitY = (ev & (rY.abs() > rA.abs())).shift(-1).fillna(False).astype(bool)   # Yahoo-side step: real fake return
    nY = yo.shift(-1) / yc - 1                                      # the model's night return (Yahoo)
    nA = ao.shift(-1) / ac - 1                                      # Alpaca 'all'
    nS = so.shift(-1) / sc - 1                                      # Alpaca split only
    dis = (nY - nA).abs()
    print("\n|Yahoo night - Alpaca-all night| quantiles (train univ)",
          dis.where(univ).stack().quantile([.5, .9, .99, .999]).round(5).to_dict())

    # training frame 2020-23
    tu = univ.loc["2020":"2023"] & nY.loc["2020":"2023"].notna()
    th = hit.loc["2020":"2023"] & tu
    big = (dis > 0.01).loc["2020":"2023"] & tu & nA.loc["2020":"2023"].notna()
    add("train_2020_23", "stock-nights in training universe", n=int(tu.sum().sum()))
    add("train_2020_23", "nights with a Yahoo-side scale step (fake return)", n=int((hitY.loc["2020":"2023"] & tu).sum().sum()))
    add("train_2020_23", "nights with a scale change (break at t+1)", n=int(th.sum().sum()),
        mean_yahoo_bp=1e4 * float(nY.loc["2020":"2023"][th].stack().mean()),
        mean_alpaca_all_bp=1e4 * float(nA.loc["2020":"2023"][th].stack().mean()),
        mean_alpaca_split_bp=1e4 * float(nS.loc["2020":"2023"][th].stack().mean()),
        max_abs_yahoo_bp=1e4 * float(nY.loc["2020":"2023"][th].abs().stack().max()))
    add("train_2020_23", "nights |Yahoo - Alpaca-all| > 1%", n=int(big.sum().sum()),
        share=float(big.sum().sum() / tu.sum().sum()))
    # a constant factor cancels in returns: on persistently off-scale ticker-days (not at a break), how far do the
    # Yahoo and Alpaca-split night returns actually differ?
    off = ((km - 1).abs() > TH) & k.notna()
    offn = off.loc["2020":"2023"] & tu & ~hit.loc["2020":"2023"] & nS.loc["2020":"2023"].notna()
    dS = (nY - nS).abs().loc["2020":"2023"]
    add("train_2020_23", "nights on a persistently off-scale ticker-period", n=int(offn.sum().sum()),
        share=float(offn.sum().sum() / tu.sum().sum()),
        share_night_differs_gt_10bp=float((dS[offn] > 0.001).sum().sum() / max(offn.sum().sum(), 1)),
        median_abs_night_diff_bp=1e4 * float(dS[offn].stack().median()))
    # how big is the offset where it is off, and is the 'all' basis just a dividend-convention difference?
    aoff = (kA - 1).abs() > TH
    clean = (k - 1).abs() < 0.001
    add("extent", "all-basis off > 0.5% while split-basis within 0.1% (dividend convention, top500 2020-23)",
        period="2020-23", scope="top500",
        share=float((aoff & clean & top500).loc["2020":"2023"].sum().sum() /
                    (clean & top500).loc["2020":"2023"].sum().sum()))
    lk = np.log(k).abs().loc["2020":"2023"]
    for lo_, hi_ in [(0.005, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 10)]:
        m = (np.expm1(lk) > lo_ - 1e-12) & (np.expm1(lk) <= hi_) & tu
        add("train_2020_23", f"off-scale rows with |k-1| in ({lo_}, {hi_}]", n=int(m.sum().sum()),
            share=float(m.sum().sum() / tu.sum().sum()), tickers=int(m.any().sum()))
    add("train_2020_23", "rows with lpx feature off by > 0.5% (|log k|)", n=int(((lk > np.log1p(TH)) & tu).sum().sum()),
        share=float(((lk > np.log1p(TH)) & tu).sum().sum() / tu.sum().sum()),
        mean_abs_log_off=float(lk[(lk > np.log1p(TH)) & tu].stack().mean()))

    # blend 2024-26 (study 69 baseline)
    import bt
    pcols = stock_cols(P)
    alld = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=pcols)
    ens = ens.loc[ens.index < alld[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=pcols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=pcols)
    okp = ens.notna() & pj.notna()
    S = (2 * ens.where(okp).rank(axis=1, pct=True) + (pj - pdr).where(okp).rank(axis=1, pct=True)) / 3
    Rn = (P["o"][pcols].shift(-1) / P["c"][pcols] - 1).reindex_like(S).astype("float64")
    C = (exec_cost_bps(P, "auction")[pcols] + 2.5).reindex_like(S)
    W = bt.select_topk(S, S.notna(), 10)
    al = lambda X: X.reindex(index=S.index, columns=pcols)
    hitb, nAb, nSb, kb = al(hit).fillna(False).astype(bool), al(nA), al(nS), al(k)
    hitYb = al(hitY).fillna(False).astype(bool)
    picked = W > 0
    print("blend picks without Alpaca bars:", int((picked & kb.isna()).sum().sum()), "of", int(picked.sum().sum()))
    base = bt.run(W, Rn, C).net
    variants = {"base (published)": Rn}
    hp = picked & hitb
    variants["drop scale-break nights (cash)"] = Rn.where(~hitb, 0.0)
    variants["drop Yahoo-side scale-step nights (cash)"] = Rn.where(~hitYb, 0.0)
    variants["correct Yahoo-side scale-step nights with Alpaca-split"] = Rn.where(~hitYb | nSb.isna(), nSb)
    variants["correct scale-break nights with Alpaca-all"] = Rn.where(~hitb | nAb.isna(), nAb)
    disb = (Rn - nAb).abs()
    bad = (disb > 0.01) & nAb.notna()
    variants["replace every night |Yahoo-Alpaca-all| > 1% with Alpaca-all"] = Rn.where(~bad, nAb)
    variants["drop every night |Yahoo-Alpaca-all| > 1% (cash)"] = Rn.where(~bad, 0.0)
    kbm = np.exp(np.log(kb).rolling(21, center=True, min_periods=10).median())
    offscale = ((kbm - 1).abs() > TH)
    variants["drop picks on an off-scale period (cash)"] = Rn.where(~offscale, 0.0)
    print("blend picks on scale-break nights:", int(hp.sum().sum()), "| Yahoo-side steps:",
          int((picked & hitYb).sum().sum()), "| |Y-A|>1%:", int((picked & bad).sum().sum()),
          "| on off-scale periods:", int((picked & offscale).sum().sum()))
    lst = []
    for d, t in zip(*np.where(hp.values)):
        lst.append(dict(date=S.index[d].date(), ticker=pcols[t], yahoo=Rn.iat[d, t], alpaca_all=nAb.iat[d, t],
                        alpaca_split=nSb.iat[d, t]))
    print(pd.DataFrame(lst).to_string())
    for dd, t in zip(*np.where((picked & bad).values)):
        add("blend_bad_nights", pcols[t], period=str(S.index[dd].date()), yahoo_bp=1e4 * Rn.iat[dd, t],
            alpaca_all_bp=1e4 * nAb.iat[dd, t], alpaca_split_bp=1e4 * nSb.iat[dd, t],
            scale_change=bool(hitb.iat[dd, t]))
    for nm, Rv in variants.items():
        r = bt.run(W, Rv, C).net
        for per, a, b in [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"),
                          ("2024-26", "2024-01", "2026-09")]:
            st = ann_stats(r.loc[a:b])
            add("blend", nm, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], net_bp=1e4 * r.loc[a:b].mean(),
                diff_vs_base_bp=1e4 * (r - base).loc[a:b].mean())
    for nm, m in [("scale-break nights", hp), ("Yahoo-side scale-step nights", picked & hitYb),
                  ("|Yahoo-Alpaca-all| > 1%", picked & bad),
                  ("off-scale periods", picked & offscale)]:
        x = (W * Rn).where(m).sum(1)
        add("blend_contrib", nm, n=int(m.loc["2024":"2026-09"].sum().sum()),
            sum_weighted_ret_bp=1e4 * float(x.loc["2024":"2026-09"].sum()),
            mean_yahoo_bp=1e4 * float(Rn.where(m).loc["2024":"2026-09"].stack().mean()) if m.any().any() else np.nan,
            mean_alpaca_all_bp=1e4 * float(nAb.where(m).loc["2024":"2026-09"].stack().mean()) if m.any().any() else np.nan)
    tot = 1e4 * float(base.loc["2024":"2026-09"].sum())
    add("blend_contrib", "total net (sum of daily returns)", sum_weighted_ret_bp=tot)

    # ML ranker of final_series (study 3)
    fn = f"{RES}/study3_pred_night_1545.parquet"
    if os.path.exists(fn):
        pr = pd.read_parquet(fn)["pred"].unstack()
        pr = pr.reindex(index=pr.index[pr.index >= "2024-01-02"], columns=pcols)
        Wm = bt.select_topk(pr, pr.notna(), 10) > 0
        hm = Wm & hitY.reindex(index=pr.index, columns=pcols).fillna(False).astype(bool)
        bm = Wm & ((Rn.reindex(index=pr.index) - nA.reindex(index=pr.index, columns=pcols)).abs() > 0.01)
        add("ml_overnight_k10", "picks on Yahoo-side scale-step nights", n=int(hm.sum().sum()), n_picks=int(Wm.sum().sum()))
        add("ml_overnight_k10", "picks with |Yahoo-Alpaca-all| > 1%", n=int(bm.sum().sum()), n_picks=int(Wm.sum().sum()))

    # final_series small/mid-cap tiers (study 8/9 timing; price filter on the traded close as in final_series.py):
    # how exposed are they to scale breaks and to Yahoo/Alpaca night disagreements?
    from core import traded_close
    tpx = traded_close(P)[cols].shift(1).reindex(days)
    adv_f = dv.rolling(20, min_periods=10).median().shift(1)
    tiers = {"final_series tierS (px>2, adv 1-5M)": (tpx > 2) & (adv_f > 1e6) & (adv_f <= 5e6),
             "final_series tierM (px>5, adv 5-50M)": (tpx > 5) & (adv_f > 5e6) & (adv_f <= 5e7)}
    for nm, tm in tiers.items():
        m = tm.loc["2024":"2026-09"] & nY.loc["2024":"2026-09"].notna()
        hb = hit.loc["2024":"2026-09"] & m
        hb2 = hitY.loc["2024":"2026-09"] & m
        bg = ((nY - nA).abs() > 0.01).loc["2024":"2026-09"] & m & nA.loc["2024":"2026-09"].notna()
        add("final_series_tiers", nm, n=int(m.sum().sum()), n_break_nights=int(hb.sum().sum()),
            n_yahoo_side_steps=int(hb2.sum().sum()), n_night_disagree_gt_1pct=int(bg.sum().sum()),
            share_night_disagree=float(bg.sum().sum() / max(m.sum().sum(), 1)),
            mean_yahoo_bp_on_disagree=1e4 * float(nY.loc["2024":"2026-09"][bg].stack().mean()) if bg.any().any() else np.nan,
            mean_alpaca_all_bp_on_disagree=1e4 * float(nA.loc["2024":"2026-09"][bg].stack().mean()) if bg.any().any() else np.nan)

    # ---------------------------------------------------------------- (2b) universe filters
    kc = k.ffill(limit=5)
    rawA = raw / kc                                                   # Yahoo close on Alpaca's split scale
    advA = (dv / kc).rolling(20, min_periods=10).median()
    univA = (rawA > 5) & (advA > 5e6)
    has = kc.notna()
    for per, a, b in [("2020-23", "2020", "2023"), ("2024-26", "2024", "2026")]:
        u0, u1 = univ.loc[a:b] & has.loc[a:b], univA.loc[a:b] & has.loc[a:b]
        add("universe", f"train filter rawc>5 & adv20>5M, {per}", n_ticker_days_yahoo=int(u0.sum().sum()),
            n_ticker_days_alpaca_scale=int(u1.sum().sum()), only_yahoo=int((u0 & ~u1).sum().sum()),
            only_alpaca=int((u1 & ~u0).sum().sum()), tickers_yahoo=int(u0.any().sum()),
            tickers_alpaca=int(u1.any().sum()), tickers_only_yahoo=int((u0.any() & ~u1.any()).sum()),
            tickers_only_alpaca=int((u1.any() & ~u0.any()).sum()),
            tickers_with_any_day_differing=int((u0 != u1).any().sum()))
        # top-500 by adv (study 64-style)
        r0 = adv20.shift(1).where(raw.shift(1) > 5).rank(axis=1, ascending=False).loc[a:b] <= 500
        r1 = advA.shift(1).where(rawA.shift(1) > 5).rank(axis=1, ascending=False).loc[a:b] <= 500
        add("universe", f"top-500 by adv20 (price>5), {per}", only_yahoo=int((r0 & ~r1).sum().sum()),
            only_alpaca=int((r1 & ~r0).sum().sum()), n_ticker_days_yahoo=int(r0.sum().sum()),
            tickers_with_any_day_differing=int((r0 != r1).any().sum()))
    u0, u1 = univ.loc["2020":"2023"] & has.loc["2020":"2023"], univA.loc["2020":"2023"] & has.loc["2020":"2023"]
    dif = (u0 != u1).sum().sort_values(ascending=False)
    print("\ntickers with most universe-day differences 2020-23:\n", dif.head(20).to_string())
    # blend picks that would leave the (study 23) universe on the Alpaca scale
    out_ = picked & al(~univA.shift(0)).fillna(False).astype(bool) & al(univ).fillna(False).astype(bool)
    add("universe", "blend picks 2024-26 outside the Alpaca-scale filter", n=int(out_.sum().sum()),
        n_picks=int(picked.sum().sum()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "study82_scale_errors.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 30)
    for p in df.part.unique():
        x = df[df.part == p].dropna(axis=1, how="all")
        print("\n==", p)
        print(x.drop(columns="part").round(4).to_string(index=False, max_rows=80))


SCR = os.environ.get("S82_SCRATCH")
if __name__ == "__main__":
    main()
