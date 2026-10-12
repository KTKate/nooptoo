"""Study 121: overnight short of the bottom of the blend, easy-to-borrow names, at official cross prices.

Question: the live overnight strategy is long only (the blend, study 33 / 69). Earlier short tests (studies 29, 72)
found the short side works only in hard-to-borrow names. Studies 88 / 94 found that the panel open (Yahoo, usually the
first trade) lies above the official opening cross for volatile gapped names; a short covered at the open buys back
at the cross, which could be cheaper or dearer than the panel open, so the short side was never checked at cross
prices. Does shorting the lowest-ranked names overnight (sell in the closing auction of t, buy back in the opening
auction of t+1) make money at official cross prices after costs, in names Alpaca marks easy to borrow and shortable?

Data: blend score exactly as study 69 lines 22-46: S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3 from results/
study23_pred.parquet and results/study33_pred.parquet; p_drop from study33_pred. Panel prices core.load_panel.
Borrow list: data/local/alpaca_assets_active.parquet, easy_to_borrow and shortable (TODAY's list: names that were
hard to borrow in the past, and names that delisted, are judged by their 2026 status, which flatters the past; the
panel also holds only tickers alive in 2026). Official crosses: caches of studies 94 / 88 / 99 / 108 / 110 / 113 plus
crosses fetched here into data/local/auctions121.parquet with study 94's rule (first 'O' / '6' print of the listing
exchange, second pass the largest 'O' / '6' print of any listing exchange; Alpaca SIP; resumable). Traded close for
the price floor from core.traded_close (Alpaca raw daily bars; Yahoo raw close where missing); dollar volume floor on
the 20-day median through t-1 (as study 110).

Design: night t = closing cross of t to opening cross of t+1, short side (return = -night return). Returns at
crosses as study 94: panel night return x (open cross / panel open on t+1) / (close cross / panel close on t); legs
without a usable cross (or a data error by study 94's checks) fall back to the panel price (rate reported).
Dividends and splits between t and t+1 stay handled by the adjusted panel (a short pays the dividend: included).
Selections among easy-to-borrow names with a score: bottom 5 / 10 / 20 by S, top 10 by p_drop; floors: none, traded
price > $5 and ADV > $5M, price > $5 and ADV > $20M. Equal weight, 100% of the short book's capital. Cost
exec_cost_bps(P, "auction") + 2.5 bp per side; borrow fee 0 (easy to borrow), sensitivity 0.3% a year charged per
calendar day held (Alpaca, 360-day year). Short sale restriction check: share of picks with low of t or t-1 10%+
below the prior close (Rule 201 in force at the closing auction, where a short may not print at or below the bid);
'ex-SSR' rows drop those picks (no refill). Bias per pick for the shorts = panel short return - cross short return
(positive = the panel flatters the short). Combined book: long blend top 10 at crosses (study 94: Sharpe 2.48 / 2.27
/ 2.38) + 25% or 50% short book (needs margin: 125% / 150% gross). Variant chosen by cross Sharpe of the standalone
short on 2024-01..2025-06; 2025-07..2026-09 is the holdout. Data checks: raw cross-to-cross returns vs the ratio
method, split days on t+1, legs with |return| > 30%.

Output: results/study121_night_shorts.csv (table = short: per variant, period, pricing: Sharpe, gross / net bp per
night, max drawdown, names per night, bias per pick, fallback share, SSR share, net with fee; table = combined:
long alone and long + 25% / 50% short; table = checks: data checks; table = big_moves: legs with |return| > 30%).

    python src/study121_night_shorts.py fetch    # official crosses for the short picks (resumable)
    python src/study121_night_shorts.py          # study
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # two other jobs share the account's limit
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC121 = os.path.join(LOC, "auctions121.parquet")
OTHER = ["auctions94", "auctions99", "auctions108", "auctions110", "auctions113"]
OUT = os.path.join(RES, "study121_night_shorts.csv")
START, END = "2024-01", "2026-09"
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
FLOORS = {"no floor": (0, 0), "px>5 adv>5M": (5, 5e6), "px>5 adv>20M": (5, 20e6)}
SELS = [("bottom 5 S", 5), ("bottom 10 S", 10), ("bottom 20 S", 20), ("top 10 p_drop", 10)]
FEE_YR = 0.003


# ------------------------------------------------------------------ picks
def build():
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
    S = ((2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3).astype("float64")
    PD = pdr.where(ok).astype("float64")
    del ens, pj, pdr, ok
    a = pd.read_parquet(f"{LOC}/alpaca_assets_active.parquet")
    etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)].str.replace(".", "-", regex=False))
    etbv = np.array([t in etb for t in cols])
    tc = traded_close(P)[cols]
    px = tc.where(tc.notna(), P["rawc"][cols]).reindex_like(S)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex_like(S)
    W0 = bt.select_topk(S, S.notna(), 10)                    # long blend top 10 (study 69 base)
    sets = {}
    for fl, (pf, af) in FLOORS.items():
        E = S.notna() & etbv[None, :] & (px > pf) & (adv > af)
        for name, k in SELS:
            if name.startswith("bottom"):
                sets[(name, fl)] = bt.select_topk(S, E, k, largest=False)
            else:
                sets[(name, fl)] = bt.select_topk(PD, E & PD.notna(), k, largest=True)
    # short sale restriction (Rule 201) in force at the close of t: low of t or of t-1 10%+ below the prior close
    l, c = P["l"][cols], P["c"][cols]
    drop = (l / c.shift(1) <= 0.9)
    ssr = (drop | drop.shift(1, fill_value=False)).reindex_like(S).fillna(False)
    return dict(P=P, cols=cols, days=days, S=S, W0=W0, sets=sets, ssr=ssr, px=px, adv=adv)


def need_list(K):
    nxt = pd.Series(K["days"][1:], index=K["days"][:-1])
    rows = []
    for Wx in list(K["sets"].values()) + [K["W0"]]:
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    n = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()
    n["date"] = pd.to_datetime(n.date)
    return n.reset_index(drop=True)


# ------------------------------------------------------------------ crosses
def crosses():
    """All cached crosses (studies 94 / 88 / 99 / 108 / 110 / 113 / 121), panel-style tickers, one row per key."""
    import study99_cross_retrain as S99
    parts = [S99.crosses()]
    for fn in [os.path.join(LOC, f + ".parquet") for f in OTHER[2:]] + [AUC121]:
        if os.path.exists(fn):
            c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
            c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
            c["date"] = pd.to_datetime(c.date)
            parts.append(c)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])


def tried():
    parts = [crosses()[["ticker", "date", "kind"]]]
    for fn in [os.path.join(LOC, f + ".parquet") for f in OTHER] + [AUC121]:
        if os.path.exists(fn):
            t = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t["ticker"] = t.ticker.astype(str).str.replace(".", "-", regex=False)
            parts.append(t.assign(date=pd.to_datetime(t.date)))
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def fetch():
    import study94_cross_rescore as S94
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    S94.A.RL = A.RL
    K = build()
    need = need_list(K)
    del K
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    m = need.merge(tried(), how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("needed", len(need), "to fetch", len(todo), flush=True)
    todo = todo.assign(ticker=todo.ticker.str.replace("-", ".", regex=False))
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            r = S94._day_kind(job[0], job[1], job[2], ven)
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
            for x in r:
                x["rule"] = "venue"
            if miss:               # second pass: largest print of any listing exchange
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    if x["ticker"] in lg:
                        x["rule"] = "largest"
                        if not np.isnan(lg[x["ticker"]]["px"]):
                            x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x"]})
            return r
        except RuntimeError as e:  # left uncached; a rerun retries
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC121) if os.path.exists(AUC121) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 12):
        with ThreadPoolExecutor(2) as exe:
            for r in exe.map(one, jobs[i:i + 12]):
                new += r
        if not new:
            continue
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC121, index=False)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ study
def per_stats(net, a0, b0):
    x = net.loc[a0:b0].dropna()
    st = ann_stats(x)
    return dict(n=st["n"], sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"])


def main():
    import study99_cross_retrain as S99
    import study94_cross_rescore as S94
    K = build()
    P, cols, days, S, sets, ssr = K["P"], K["cols"], K["days"], K["S"], K["sets"], K["ssr"]
    ro, rc, err = S99.cross_ratios(P, cols, days, crosses())
    print("cross data errors (legs at the panel price):", err, flush=True)
    ro = ro.reindex(index=days, columns=cols)
    rc = rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex_like(S)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    gap_short = -(Rp - Rx).where(~miss)        # panel short return minus cross short return
    nxt = pd.Series(days[1:], index=days[:-1])
    cal = pd.Series((nxt.values - nxt.index.values) / np.timedelta64(1, "D"), index=nxt.index).reindex(S.index)
    fee = FEE_YR * cal / 360                                   # per unit of short weight per night
    rows = []
    # ---- long blend top 10 at crosses (must reproduce study 94: 2.48 / 2.27 / 2.38)
    W0 = K["W0"].loc[START:END]
    long = {pr: bt.run(W0, R.loc[START:END], C.loc[START:END]).net for pr, R in [("panel", Rp), ("cross", Rx)]}
    for lab, a0, b0 in PER:
        rows.append(dict(table="combined", set="long blend top 10 alone", period=lab, pricing="cross",
                         **per_stats(long["cross"], a0, b0)))
    shorts = {}
    variants = []
    for (name, fl), Wx in sets.items():
        Wx = Wx.loc[START:END]
        h = Wx > 0
        hs = h & ~ssr.loc[START:END]
        Wn = hs.astype("float32").div(hs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        for tag, Wv, hv in [("", Wx, h), (" ex-SSR", Wn, hs)]:
            key = f"{name}, {fl}{tag}"
            variants.append(key)
            for pr, R in [("panel", Rp), ("cross", Rx)]:
                res = bt.run(Wv, R.loc[START:END], C.loc[START:END], side=-1)
                net_fee = res.net - Wv.sum(axis=1) * fee.loc[START:END]
                shorts[(key, pr)] = res.net
                shorts[(key, pr, "fee")] = net_fee
                for lab, a0, b0 in PER:
                    st = per_stats(res.net, a0, b0)
                    hh = hv.loc[a0:b0]
                    legs = hh.sum().sum()
                    rows.append(dict(table="short", set=key, period=lab, pricing=pr, **st,
                                     gross_bp=1e4 * res.gross.loc[a0:b0].mean(),
                                     cost_bp=1e4 * res.cost.loc[a0:b0].mean(),
                                     net_bp_fee=1e4 * net_fee.loc[a0:b0].mean(),
                                     sharpe_fee=per_stats(net_fee, a0, b0)["sharpe"],
                                     names_per_night=hh.sum(axis=1).mean(),
                                     nights_empty=int((hh.sum(axis=1) == 0).sum()),
                                     bias_bp_per_pick=1e4 * gap_short.loc[a0:b0].where(hh).stack().mean(),
                                     fallback_share=(miss.loc[a0:b0] & hh).sum().sum() / max(legs, 1),
                                     ssr_share=(ssr.loc[a0:b0] & h.loc[a0:b0]).sum().sum() / max(h.loc[a0:b0].sum().sum(), 1),
                                     median_px=float(np.nanmedian(K["px"].loc[a0:b0].values[hh.values])) if legs else np.nan,
                                     median_adv_m=float(np.nanmedian(K["adv"].loc[a0:b0].values[hh.values])) / 1e6 if legs else np.nan,
                                     corr_with_long=res.net.loc[a0:b0].corr(long[pr].loc[a0:b0])))
    df = pd.DataFrame(rows)
    # ---- choose on the first period (standalone short, cross Sharpe, fee 0)
    sel = df[(df.table == "short") & (df.pricing == "cross") & (df.period == PER[0][0])]
    best = sel.sort_values("sharpe", ascending=False).iloc[0].set
    print("chosen on 2024-01..2025-06:", best, flush=True)
    df["chosen"] = df.set == best
    # ---- combined: long + f x short (chosen variant, plus every variant for reference)
    comb = []
    for key in variants:
        for f in [0.25, 0.5]:
            for feeflag in ["", "fee"]:
                sn = shorts[(key, "cross", "fee")] if feeflag else shorts[(key, "cross")]
                net = long["cross"] + f * sn.reindex(long["cross"].index).fillna(0)
                for lab, a0, b0 in PER:
                    st = per_stats(net, a0, b0)
                    d = (net - long["cross"]).loc[a0:b0].dropna()
                    comb.append(dict(table="combined", set=f"long + {int(f * 100)}% short: {key}" + (" (0.3%/yr fee)" if feeflag else ""),
                                     period=lab, pricing="cross", **st, chosen=key == best,
                                     diff_vs_long_bp=1e4 * d.mean(),
                                     t_vs_long=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan))
    df = pd.concat([df, pd.DataFrame(comb)], ignore_index=True)
    # ---- data checks: raw cross-to-cross vs ratio method, splits on t+1, huge moves
    a = crosses()
    ix = days[days >= "2023-12-01"]
    a = a[a.ticker.isin(cols) & a.date.isin(ix)]
    X = {k: g.pivot(index="date", columns="ticker", values="px").reindex(index=days, columns=cols)
         for k, g in a.groupby("kind")}
    raw = (X["open"].shift(-1) / X["close"] - 1).reindex_like(S)
    sp = pd.read_csv(f"{DATA}/local/events/splits_yf.csv", parse_dates=["date"])
    spk = set(zip(sp.ticker, sp.date))
    allh = sum((Wx.loc[START:END] > 0).astype(int) for Wx in sets.values()) > 0
    s = allh.stack()
    s = s[s].index.to_frame(index=False)
    s.columns = ["date", "ticker"]
    mi = pd.MultiIndex.from_arrays([s.date, s.ticker])
    g = lambda w: w.reindex(index=S.index).loc[START:END].stack(future_stack=True).reindex(mi).values
    ra, rx, rp, ms = g(raw), g(Rx), g(Rp), g(miss.astype(float))
    m = ~np.isnan(ra) & ~np.isnan(rx) & (ms == 0)
    dif = np.abs((1 + ra[m]) / (1 + rx[m]) - 1)
    t1 = s.date.map(nxt).values
    is_split = np.array([(t, d) in spk for t, d in zip(s.ticker.values, t1)])
    chk = [dict(table="checks", set="all short picks: raw cross-to-cross vs ratio method", n=int(m.sum()),
                diff_gt_1pct=int((dif > 0.01).sum()), diff_gt_1pct_split_next_day=int(((dif > 0.01) & is_split[m]).sum()),
                median_abs_diff_bp=1e4 * np.median(dif)),
           dict(table="checks", set="all short picks: split dated t+1", n=len(s), count=int(is_split.sum())),
           dict(table="checks", set="cross data errors open (found / scale / ratio)", n=err["ro"]["found"],
                count=err["ro"]["scale"] + err["ro"]["ratio"]),
           dict(table="checks", set="cross data errors close (found / scale / ratio)", n=err["rc"]["found"],
                count=err["rc"]["scale"] + err["rc"]["ratio"])]
    big = s.assign(R_panel=rp, R_cross=rx, R_raw=ra, split_t1=is_split, miss=ms)
    big = big[(np.abs(big.R_cross) > 0.3) | (np.abs(big.R_panel) > 0.3)]
    big["in_chosen"] = [bool(sets[tuple(best.replace(" ex-SSR", "").split(", ", 1))].loc[d, t] > 0)
                        for d, t in zip(big.date, big.ticker)]
    chk.append(dict(table="checks", set="legs with |night return| > 30% (any variant)", n=len(s), count=len(big)))
    big = big.assign(table="big_moves", set=big.ticker + " " + big.date.dt.strftime("%Y-%m-%d")).drop(columns=["date", "ticker"])
    df = pd.concat([df, pd.DataFrame(chk), big], ignore_index=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 1000)
    show = ["set", "period", "pricing", "sharpe", "gross_bp", "net_bp", "net_bp_fee", "maxdd", "names_per_night",
            "bias_bp_per_pick", "fallback_share", "ssr_share", "median_px", "corr_with_long"]
    x = df[df.table == "short"]
    print(x[show].round(3).to_string(index=False))
    y = df[(df.table == "combined") & (df.set.str.contains("alone") | df.chosen.fillna(False).astype(bool))]
    print(y[["set", "period", "sharpe", "net_bp", "maxdd", "diff_vs_long_bp", "t_vs_long"]].round(3).to_string(index=False))
    print(df[df.table == "checks"].dropna(axis=1, how="all").to_string(index=False))
    print(df[df.table == "big_moves"].dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
