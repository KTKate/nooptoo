"""Study 16: can international exposure (US-listed ADRs, country ETFs) help the close->open strategy?

Data: data/local/intl/ (src/fetch_international.py). Holding return everywhere: close(t) -> open(t+1),
entry in the closing auction (signal at 15:45 ET), exit in the next opening auction.
Cost per side: 1.3 bp + 10% of half_spread_model(adv, px, vol20, "15:45") (lagged inputs) + 2.5 bp.

  a. Europe lead-lag: SPY 11:30->15:45 move vs next-morning gap of European ADRs, controlling for the
     ADR's own 11:30->15:45 move (Alpaca 5-minute windows).
  b. Asia: overnight return of Asian ADRs / Asia ETFs vs the same-day US close-to-close move.
  c. Cross-sectional top-k ADR rules (intraday loser, overnight momentum 20d) vs study1 on US stocks.
  d. Gross overnight dispersion of ADRs vs US stocks of similar liquidity.

Outputs: results/study16_tests.csv (strategy stats), results/study16_regress.csv (regressions),
results/study16_dispersion.csv.
"""
import os
import numpy as np
import pandas as pd

from core import load_panel, ann_stats, half_spread_model, RES, DATA, stock_cols
import store
import bt

INTL = os.path.join(DATA, "local", "intl")
PERIODS = {"early": ("2019-06-01", "2023-12-31"), "val": ("2024-01-01", "2025-06-30"),
           "hold": ("2025-07-01", "2026-09-30")}
ROWS, REG = [], []
N_VARIANTS = 0


def panels():
    d = pd.read_parquet(os.path.join(INTL, "daily.parquet"))
    d["date"] = pd.to_datetime(d.date).dt.tz_localize(None).dt.normalize()
    d = d[(d.c > 0) & (d.o > 0) & (d.h > 0) & (d.l > 0)]
    f = store.adj_factor(d).astype("float64")
    d["rawc"] = d["c"]
    for k in ["o", "h", "l", "c"]:
        d[k] = d[k] * f
    P = {k: d.pivot_table(index="date", columns="ticker", values=k).astype("float64")
         for k in ["o", "h", "l", "c", "rawc", "v"]}
    days = P["c"]["SPY"].dropna().index
    days = days[days >= "2019-05-01"]
    for k in P:
        P[k] = P[k].reindex(days)
    P["dv"] = P["rawc"] * P["v"]
    bad = (P["l"] > P["h"] * 1.0001) | (P["o"] > P["h"] * 1.02) | (P["o"] < P["l"] * 0.98)
    for k in ["o", "h", "l", "c"]:
        P[k] = P[k].mask(bad)
    return P


def derived(P):
    D = {}
    c, o = P["c"], P["o"]
    D["co_next"] = (o.shift(-1) / c - 1).clip(-0.5, 1.0)
    D["oc"] = c / o - 1
    D["ret1"] = c / c.shift(1) - 1
    D["night"] = o / c.shift(1) - 1
    D["on_mom20"] = np.log1p(D["night"]).rolling(20, min_periods=15).mean()
    D["adv"] = P["dv"].rolling(20, min_periods=10).median().shift(1)
    px = P["rawc"].shift(1)
    vol = np.log(c / c.shift(1)).rolling(20, min_periods=10).std().shift(1)
    hs = half_spread_model(D["adv"], px, vol.clip(lower=1e-3), "15:45")
    D["cost"] = pd.DataFrame(1.3 + 0.1 * np.asarray(hs) + 2.5, index=c.index, columns=c.columns)
    D["elig"] = (D["adv"] > 5e6) & (px > 3)
    return D


def stats_row(test, variant, r, extra=None):
    """r: DataFrame with gross, net (daily decimal returns, NaN/0 on days without a trade)."""
    for per, (a, b) in PERIODS.items():
        s = r.loc[a:b]
        traded = s[s["n"] > 0] if "n" in s else s
        g, nt = ann_stats(s["gross"]), ann_stats(s["net"])
        row = dict(test=test, variant=variant, period=per, days=len(s), trade_days=len(traded),
                   avg_names=traded["n"].mean() if "n" in s else np.nan,
                   gross_sharpe=g["sharpe"], net_sharpe=nt["sharpe"], net_tstat=nt["tstat"],
                   gross_bp_trade=1e4 * traded["gross"].mean(), net_bp_trade=1e4 * traded["net"].mean(),
                   net_ann_ret=nt["ann_ret"], maxdd=nt["maxdd"])
        if extra:
            row.update(extra)
        ROWS.append(row)


def ols(y, X, names):
    """OLS with White standard errors. Returns dict name -> (coef, t)."""
    m = pd.concat([y, X], axis=1).dropna()
    yv = m.iloc[:, 0].values
    Xv = np.column_stack([np.ones(len(m)), m.iloc[:, 1:].values])
    XtX = np.linalg.inv(Xv.T @ Xv)
    b = XtX @ Xv.T @ yv
    e = yv - Xv @ b
    V = XtX @ (Xv.T * e ** 2) @ Xv @ XtX
    t = b / np.sqrt(np.diag(V))
    r2 = 1 - (e ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return {n: (b[i], t[i]) for i, n in enumerate(["const"] + names)}, len(m), r2


def reg_rows(test, spec, y, X, names):
    for per, (a, b) in PERIODS.items():
        res, n, r2 = ols(y.loc[a:b], X.loc[a:b], names)
        row = dict(test=test, spec=spec, period=per, n=n, r2=r2)
        for k, (cf, tt) in res.items():
            row[f"b_{k}"] = cf
            row[f"t_{k}"] = tt
        REG.append(row)


def basket(W, D, R=None):
    return bt.run(W, D["co_next"] if R is None else R, D["cost"], roundtrip=True)


def ew_weights(mask):
    W = mask.astype("float64")
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


# ------------------------------------------------------------------------------------------------
def test_a(P, D, U):
    global N_VARIANTS
    fn = os.path.join(INTL, "m15win.parquet")
    if not os.path.exists(fn):
        print("no m15win, skip a")
        return
    import alpaca_data as A
    m1 = A.read("m1", start="2019-06", tickers=["SPY"], columns=["ts", "ticker", "c"])
    m1["d"] = m1.ts.dt.normalize()
    hm = m1.ts.dt.hour * 100 + m1.ts.dt.minute
    s1130 = m1[hm == 1129].set_index("d").c    # close of the 11:29 bar = price at 11:30
    s1545 = m1[hm == 1544].set_index("d").c
    spy_aft = (s1545 / s1130 - 1).rename("spy_aft")
    m5 = pd.read_parquet(fn)
    m5["d"] = m5.ts.dt.normalize()
    h5 = m5.ts.dt.hour * 100 + m5.ts.dt.minute
    p1130 = m5[h5 == 1115].pivot_table(index="d", columns="ticker", values="c")   # 11:15 bar closes 11:30
    p1545 = m5[h5 == 1530].pivot_table(index="d", columns="ticker", values="c")   # 15:30 bar closes 15:45
    idx = P["c"].index
    p1130, p1545 = p1130.reindex(idx), p1545.reindex(idx)
    own = (p1545 / p1130 - 1)
    spy = spy_aft.reindex(idx)
    eu = [t for t in U[(U.region == "Europe") & (U.kind != "etf")].ticker if t in own.columns]
    E = D["elig"][eu] & own[eu].notna() & D["co_next"][eu].notna()
    y = D["co_next"][eu].where(E)
    ownE = own[eu].where(E)
    # equal-weight European ADR basket
    yb, ob = y.mean(axis=1), ownE.mean(axis=1)
    X1 = pd.DataFrame({"spy_aft": spy})
    reg_rows("a_europe", "basket_gap~spy_aft", yb, X1, ["spy_aft"])
    X2 = pd.DataFrame({"spy_aft": spy, "own_aft": ob})
    reg_rows("a_europe", "basket_gap~spy_aft+own_aft", yb, X2, ["spy_aft", "own_aft"])
    # residual under-reaction: spy move minus the ADR's own move (pooled, stock-demeaned daily mean)
    spyP = pd.DataFrame(np.repeat(spy.values[:, None], len(eu), 1), index=idx, columns=eu)
    und = spyP - ownE
    reg_rows("a_europe", "basket_gap~(spy_aft-own_aft)", yb, pd.DataFrame({"gap": und.mean(axis=1)}), ["gap"])
    # pooled stock-level (day fixed effects not used; stock-day obs, White SE; indicative only)
    st = pd.DataFrame({"y": y.stack(), "spy": spyP.where(E).stack(),
                       "own": ownE.stack()}).dropna()
    for per, (a, b) in PERIODS.items():
        s = st.loc[a:b]
        res, n, r2 = ols(s["y"], s[["spy", "own"]], ["spy", "own"])
        REG.append(dict(test="a_europe", spec="pooled_stock_gap~spy+own", period=per, n=n, r2=r2,
                        **{f"b_{k}": v[0] for k, v in res.items()}, **{f"t_{k}": v[1] for k, v in res.items()}))
    # trading rules (long only)
    Wall = ew_weights(E)
    for thr in [0.0, 0.005]:
        W = Wall.mul((spy > thr).astype(float), axis=0)
        stats_row("a_europe", f"EW_EU_ADRs_if_spy_aft>{thr:.3f}", basket(W, D))
        N_VARIANTS += 1
    stats_row("a_europe", "EW_EU_ADRs_every_day(ref)", basket(Wall, D))
    N_VARIANTS += 1
    for k in [5, 10]:
        W = bt.select_topk(und.where(E), E, k)
        stats_row("a_europe", f"top{k}_EU_by_(spy_aft-own_aft)", basket(W, D))
        N_VARIANTS += 1
    # Europe ETFs with the same timing
    for t in ["EWG", "EWU", "VGK", "EFA"]:
        if t in own.columns:
            m = pd.DataFrame({t: (spy > 0.0).astype(float)}, index=idx)
            stats_row("a_europe", f"{t}_if_spy_aft>0", basket(m, D))
            N_VARIANTS += 1
            reg_rows("a_europe", f"{t}_gap~spy_aft+own_aft", D["co_next"][t],
                     pd.DataFrame({"spy_aft": spy, "own_aft": own[t]}), ["spy_aft", "own_aft"])


def test_b(P, D, U):
    global N_VARIANTS
    idx = P["c"].index
    spy_cc = D["ret1"]["SPY"]
    asia_etf = ["EWJ", "FXI", "MCHI", "EWY", "EWT", "EWH"]
    asia_adr = [t for t in U[U.region.isin(["Japan/APAC", "China/HK"]) & (U.kind != "etf")].ticker
                if t in P["c"].columns]
    for name, tick in [("asia_etfs", asia_etf), ("asia_adrs", asia_adr)]:
        E = D["elig"][tick] & D["co_next"][tick].notna()
        y = D["co_next"][tick].where(E).mean(axis=1)
        own = D["ret1"][tick].where(E).mean(axis=1)
        reg_rows("b_asia", f"{name}_gap~spy_cc", y, pd.DataFrame({"spy_cc": spy_cc}), ["spy_cc"])
        reg_rows("b_asia", f"{name}_gap~spy_cc+own_cc", y, pd.DataFrame({"spy_cc": spy_cc, "own_cc": own}),
                 ["spy_cc", "own_cc"])
        Wall = ew_weights(E)
        stats_row("b_asia", f"EW_{name}_every_day(ref)", basket(Wall, D))
        N_VARIANTS += 1
        for lab, cond in [("spy_up(momentum)", spy_cc > 0), ("spy_down(reversal)", spy_cc < 0),
                          ("spy<-1%(reversal)", spy_cc < -0.01)]:
            stats_row("b_asia", f"EW_{name}_if_{lab}", basket(Wall.mul(cond.astype(float), axis=0), D))
            N_VARIANTS += 1
    for t in asia_etf:
        reg_rows("b_asia", f"{t}_gap~spy_cc", D["co_next"][t], pd.DataFrame({"spy_cc": spy_cc}), ["spy_cc"])


def test_c(P, D, U):
    global N_VARIANTS
    tick = [t for t in U[U.kind != "etf"].ticker if t in P["c"].columns]
    E = D["elig"][tick]
    sigs = {"intraday_loser": -D["oc"][tick], "overnight_mom20": D["on_mom20"][tick],
            "day_winner": D["ret1"][tick]}
    for sname, S in sigs.items():
        EE = E & S.notna() & D["co_next"][tick].notna()
        for k in [5, 10]:
            W = bt.select_topk(S, EE, k)
            stats_row("c_xsec_adr", f"{sname}_top{k}", basket(W, D))
            N_VARIANTS += 1
    # by home region (intraday loser, top 5)
    for reg in ["Europe", "Japan/APAC", "China/HK", "LatAm", "Other"]:
        tr = [t for t in U[(U.kind != "etf") & (U.region == reg)].ticker if t in P["c"].columns]
        S = -D["oc"][tr]
        EE = D["elig"][tr] & S.notna() & D["co_next"][tr].notna()
        stats_row("c_xsec_adr_region", f"intraday_loser_top5_{reg}", basket(bt.select_topk(S, EE, 5), D))
        N_VARIANTS += 1
    # same rules, same cost model, on US stocks (store universe) and on US + ADRs pooled
    Pu = load_panel()
    cols = stock_cols(Pu)
    Pu = {k: Pu[k][cols].astype("float64") for k in ["o", "h", "l", "c", "rawc", "v", "dv"]}
    Du = derived(Pu)
    extra = [t for t in tick if t not in cols]
    Dj = {k: pd.concat([Du[k], D[k][extra].reindex(Du[k].index)], axis=1) for k in Du}
    for lab, DD in [("US_only", Du), ("US_plus_ADR", Dj)]:
        for sname, S in [("intraday_loser", -DD["oc"]), ("overnight_mom20", DD["on_mom20"])]:
            EE = DD["elig"] & S.notna() & DD["co_next"].notna()
            for k in [5, 10]:
                W = bt.select_topk(S, EE, k)
                r = bt.run(W, DD["co_next"], DD["cost"], roundtrip=True)
                stats_row("c_us_same_rule", f"{lab}_{sname}_top{k}", r,
                          extra=dict(adr_share=W[[c for c in W.columns if c in extra]].sum(axis=1).mean()))
                N_VARIANTS += 1
    # survivorship yardstick: the ADR list holds only names listed today. Rerun the US rule on US names that
    # are still listed today (NASDAQ Trader directory) and compare with the full (delisted-inclusive) store.
    cur = set(pd.read_csv(os.path.join(INTL, "nasdaqtraded.txt"), sep="|", keep_default_na=False).Symbol)
    alive = [c for c in cols if c in cur]
    for k in [5, 10]:
        S = -Du["oc"][alive]
        EE = Du["elig"][alive] & S.notna() & Du["co_next"][alive].notna()
        r = bt.run(bt.select_topk(S, EE, k), Du["co_next"][alive], Du["cost"][alive], roundtrip=True)
        stats_row("c_us_same_rule", f"US_listed_today_only_intraday_loser_top{k}", r)
    del Du, Dj, Pu
    # US reference from study1 (top10, L and M tiers, dev=2020-23)
    s1 = pd.read_csv(os.path.join(RES, "study1_battery.csv"))
    s1 = s1[s1.strat.isin(["close:intraday_loser_overnight", "close:overnight_mom20", "close:day_winner_overnight"])
            & s1.tier.isin(["L", "M"])]
    for r in s1.itertuples():
        ROWS.append(dict(test="c_us_ref_study1", variant=f"US_{r.tier}_{r.strat}_top10",
                         period={"dev": "early(2020-23)", "val": "val", "oos": "hold"}[r.period],
                         gross_sharpe=r.gross_sharpe, net_sharpe=r.net_sharpe, gross_bp_trade=r.gross_bps_day,
                         net_bp_trade=r.gross_bps_day - r.cost_bps_day))


def test_d(P, D, U):
    Pu = load_panel()
    cols = stock_cols(Pu)
    c, o, dv, rawc = (Pu[k][cols] for k in ["c", "o", "dv", "rawc"])
    us_y = (o.shift(-1) / c - 1).clip(-0.5, 1.0)
    us_adv = dv.rolling(20, min_periods=10).median().shift(1)
    us_px = rawc.shift(1)
    tick = [t for t in U[U.kind != "etf"].ticker if t in P["c"].columns]
    a_y, a_adv, a_px = D["co_next"][tick], D["adv"][tick], P["rawc"][tick].shift(1)
    out = []
    buckets = [("5-20M", 5e6, 2e7), ("20-100M", 2e7, 1e8), (">100M", 1e8, 1e13)]
    for per, (a, b) in PERIODS.items():
        for bn, lo, hi in buckets:
            for lab, y, adv, px in [("ADR", a_y, a_adv, a_px), ("US", us_y, us_adv, us_px)]:
                m = (adv > lo) & (adv <= hi) & (px > 3) & y.notna()
                yy = y.where(m).loc[a:b]
                sd = yy.std(axis=1)
                mad = yy.sub(yy.median(axis=1), axis=0).abs().mean(axis=1)
                out.append(dict(period=per, adv_bucket=bn, group=lab, avg_names=m.loc[a:b].sum(axis=1).mean(),
                                xs_sd_bp=1e4 * sd.mean(), xs_mad_bp=1e4 * mad.mean(),
                                abs_ret_bp=1e4 * yy.abs().stack().mean(), mean_ret_bp=1e4 * yy.stack().mean()))
    pd.DataFrame(out).to_csv(os.path.join(RES, "study16_dispersion.csv"), index=False)
    print(pd.DataFrame(out).round(1).to_string())


def validate_auction(P, D, U, step=3, fetch=True):
    """Re-price a sample of ADR intraday-loser top-5 trades (2024-01..) with primary-exchange closing cross (t)
    and opening cross (t+1) from Alpaca SIP trades. Cached in data/local/intl/auctions_intl.parquet."""
    import alpaca_data as A
    from concurrent.futures import ThreadPoolExecutor
    tick = [t for t in U[U.kind != "etf"].ticker if t in P["c"].columns]
    S = -D["oc"][tick]
    EE = D["elig"][tick] & S.notna() & D["co_next"][tick].notna()
    W = bt.select_topk(S, EE, 5).loc["2024-01-01":]
    days = W.index[::step]
    nxt = dict(zip(P["c"].index[:-1], P["c"].index[1:]))
    pairs = [(t, d, nxt[d]) for d in days if d in nxt for t in W.columns[W.loc[d] > 0]]
    fn = os.path.join(INTL, "auctions_intl.parquet")
    have = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["ticker", "date"])
    got = set(zip(have.ticker, pd.to_datetime(have.date)))
    todo = [p for p in pairs if (p[0], p[1]) not in got]
    if fetch and todo:
        def one(p):
            t, d, d1 = p
            try:
                c = A._official(t, d.strftime("%Y-%m-%d"), "close")[0]
                o = A._official(t, d1.strftime("%Y-%m-%d"), "open")[0]
            except RuntimeError:
                c = o = np.nan
            return dict(ticker=t, date=d, close_off=c, open_next_off=o)
        rows = []
        for i in range(0, len(todo), 200):
            with ThreadPoolExecutor(4) as ex:
                rows += list(ex.map(one, todo[i:i + 200]))
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(fn, index=False)
            print("auction", len(rows), "/", len(todo), flush=True)
        have = pd.read_parquet(fn)
    have["date"] = pd.to_datetime(have.date)
    # Yahoo raw prices for the same trades (unadjusted: splits in Alpaca are adjusted as of fetch date, so use
    # ratios only on the same day pair)
    rawo = P["o"] / P["c"] * P["rawc"]
    h = have.dropna().copy()
    h["y_close"] = [P["rawc"].at[d, t] for t, d in zip(h.ticker, h.date)]
    h["y_open1"] = [rawo.shift(-1).at[d, t] for t, d in zip(h.ticker, h.date)]
    h["r_yahoo"] = h.y_open1 / h.y_close - 1
    h["r_auction"] = h.open_next_off / h.close_off - 1
    h = h[(h.r_auction.abs() < 0.5) & (h.r_yahoo.abs() < 0.5)]
    out = []
    for per, (a, b) in PERIODS.items():
        s = h[(h.date >= a) & (h.date <= b)]
        if len(s):
            out.append(dict(period=per, n=len(s), yahoo_bp=1e4 * s.r_yahoo.mean(), auction_bp=1e4 * s.r_auction.mean(),
                            diff_bp=1e4 * (s.r_auction - s.r_yahoo).mean(),
                            diff_t=(s.r_auction - s.r_yahoo).mean() / (s.r_auction - s.r_yahoo).std() * np.sqrt(len(s))))
    print(pd.DataFrame(out).round(2))
    for r in out:
        ROWS.append(dict(test="c_auction_check", variant="ADR_intraday_loser_top5_sample", period=r["period"],
                         trade_days=r["n"], gross_bp_trade=r["auction_bp"], net_bp_trade=np.nan,
                         yahoo_bp=r["yahoo_bp"], diff_bp=r["diff_bp"], diff_t=r["diff_t"]))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    U = pd.read_csv(os.path.join(INTL, "universe.csv"))
    P = panels()
    D = derived(P)
    test_a(P, D, U)
    test_b(P, D, U)
    test_c(P, D, U)
    test_d(P, D, U)
    validate_auction(P, D, U, fetch=os.environ.get("S16_FETCH", "0") == "1")
    T = pd.DataFrame(ROWS)
    T.to_csv(os.path.join(RES, "study16_tests.csv"), index=False)
    R = pd.DataFrame(REG)
    R.to_csv(os.path.join(RES, "study16_regress.csv"), index=False)
    print(R.round(3).to_string())
    print(T.round(2).to_string())
    print("variants (trading rules) tried:", N_VARIANTS)
