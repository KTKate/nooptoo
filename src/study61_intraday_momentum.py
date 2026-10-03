"""Study 61: does the first half hour predict the rest of the day?

Signals known at 10:00 ET (both computable with IEX real-time data on Alpaca's free plan, see the report):
  s_pc = price(10:00) / previous close - 1        (overnight gap + first 30 minutes, Gao-Han-Li-Zhou 2018)
  s_o  = price(10:00) / 09:30 open - 1            (first 30 minutes only)
Targets:
  rod  = 10:00 (entry) -> close (closing auction)          "rest of day"
  l30  = 15:30 (entry) -> close (closing auction)          "last half hour"

A. SPY and QQQ, 1-minute SIP bars (data/local/m1, 2019-06 .. 2026-09). Price at 10:00 = close of the 09:59 bar;
   entry one minute later (close of the 10:00 bar); last-half-hour entry = close of the 15:30 bar. Exit = official
   close (Yahoo close, split-adjusted only, i.e. P["rawc"], which matches the Alpaca 15:59 bar to 0.4 bp sd).
   The previous close is dividend-adjusted (so ex-dividend days do not look like gap downs).
   Predictive regressions (slope, Newey-West t with 5 lags) and trading rules:
     ls      long if signal > 0, short if < 0
     lo      long if signal > 0, else cash (cash account)
     big_ls  long/short only when |signal| > 70th percentile of |signal| over the previous 250 days
   each on all days and on high-volume days (09:30-10:00 volume > 1.2 x its 20-day mean).
B. Cross-section, the 500 most traded stocks each day (lagged 20-day median dollar volume, price > $5),
   30-minute SIP bars (data/local/m30s61, fetched by src/s6162_common.py). Price at 10:00 = close of the
   09:30-10:00 bar (the last trade before 10:00; no extra minute of delay is available on 30-minute bars).
   Last half hour entry = open of the 15:30 bar. Exit = official close.
   Portfolios (equal weight, 10 names per side): mom_ls (long top 10, short bottom 10 by the signal),
   mom_long (long top 10), rev_long (long bottom 10), each among: all, in_play (09:30-10:00 volume >= 2 x its
   20-day mean), news (>= 1 Benzinga article from 16:00 the day before to 10:00), no_news.
   Alpaca/Yahoo consistency: stock-days whose Alpaca 09:30 open or last price differ from the Yahoo open/close
   by more than 3% are dropped (split timing, bad prints), and ticker-months with median disagreement > 0.5%.

Costs per side: entry in continuous trading = quote-model half-spread at that time of day (core.half_spread_model;
10:00 interpolated between the 09:35 and 12:00 calibration points, 15:30 ~ the 15:45 point) + 2 bp slippage +
0.3 bp fees + 2.5 bp; exit in the closing auction = core.exec_cost_bps(P, "auction") + 2.5 bp. Intraday shorts pay
no borrow fee but need a locate (top-500 names are nearly always easy to borrow today; not knowable historically).
Sensitivity for the ETFs: 1 bp per side ("low"), and gross.

Periods: 2020-23 (A starts 2019-07) and 2024-26 (to 2026-09-25), plus val 2024-01..2025-06 and oos 2025-07.. in the CSV.

    python src/study61_intraday_momentum.py        # writes results/study61_intraday_momentum.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import bt
from core import load_panel, stock_cols, ann_stats, RES
from s6162_common import cont_cost_bps, auction_cost_bps

PER = [("2020-23", "2019-07-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30"),
       ("val", "2024-01-01", "2025-06-30"), ("oos", "2025-07-01", "2026-09-30")]
P = load_panel()
C_AUC = auction_cost_bps(P)
C_1000 = cont_cost_bps(P, "10:00")
C_1530 = cont_cost_bps(P, "15:30")
F = P["c"] / P["rawc"]                                     # Yahoo dividend factor (adjusted / split-only)
PC_DIV = P["rawc"].shift(1) * F.shift(1) / F               # previous close, dividend-adjusted, split-only units
O_S = P["o"] / F                                           # official open, split-only units
rows = []


def nw_t(y, x, lags=5):
    """OLS slope of y on x with Newey-West t."""
    m = pd.concat([y, x], axis=1).dropna()
    if len(m) < 30:
        return np.nan, np.nan, len(m)
    yv, xv = m.iloc[:, 0].values, m.iloc[:, 1].values
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    XtX = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX @ S @ XtX
    return b[1], b[1] / np.sqrt(V[1, 1]), len(m)


def add_rows(part, name, r_net, r_gross=None, extra=None):
    for per, a, b in PER:
        x = r_net.loc[a:b]
        st = ann_stats(x)
        row = dict(part=part, variant=name, period=per, n_days=st["n"], active_days=int((x != 0).sum()),
                   sharpe=st["sharpe"], tstat=st["tstat"], ann_ret=st["ann_ret"], ann_vol=st["ann_vol"], maxdd=st["maxdd"],
                   mean_bps_active=1e4 * x[x != 0].mean() if (x != 0).any() else np.nan)
        if r_gross is not None:
            row["sharpe_gross"] = ann_stats(r_gross.loc[a:b])["sharpe"]
            g = r_gross.loc[a:b]
            row["gross_bps_active"] = 1e4 * g[g != 0].mean() if (g != 0).any() else np.nan
        row.update(extra or {})
        rows.append(row)


# ================================================================== A. SPY / QQQ
def etf_frame(t):
    d = A.read("m1", tickers=[t], columns=["ts", "ticker", "o", "c", "v"])
    d = d[(d.ts.dt.hour * 100 + d.ts.dt.minute >= 930)]
    d["date"] = d.ts.dt.normalize()
    hm = d.ts.dt.strftime("%H:%M")
    g = lambda h, k: d[hm == h].set_index("date")[k]
    f = pd.DataFrame({"o": g("09:30", "o"), "p1000": g("09:59", "c"), "e1000": g("10:00", "c"),
                      "e1530": g("15:30", "c"), "c1559": g("15:59", "c")})
    f["v30"] = d[hm < "10:00"].groupby("date").v.sum()
    f = f.reindex(P["c"].index).dropna(subset=["o", "p1000", "e1000", "e1530"])
    f["pc"] = PC_DIV[t].reindex(f.index)
    f["close"] = P["rawc"][t].reindex(f.index)
    f = f.dropna(subset=["pc", "close"])
    f["rvol"] = f.v30 / f.v30.rolling(20, min_periods=10).mean().shift(1)
    return f


def part_a():
    for t in ["SPY", "QQQ"]:
        f = etf_frame(t)
        print(t, "days", len(f), f.index[0].date(), f.index[-1].date(), flush=True)
        sig = {"s_pc": f.p1000 / f.pc - 1, "s_o": f.p1000 / f.o - 1, "gap": f.o / f.pc - 1}
        tgt = {"rod": f.close / f.e1000 - 1, "l30": f.close / f.e1530 - 1}
        cin = {"rod": C_1000[t].reindex(f.index) / 1e4, "l30": C_1530[t].reindex(f.index) / 1e4}
        cout = C_AUC[t].reindex(f.index) / 1e4
        hv = f.rvol > 1.2
        # predictive regressions
        for sn, s in sig.items():
            for tn, y in tgt.items():
                for per, a, b in PER:
                    sl, tt, n = nw_t(y.loc[a:b], s.loc[a:b])
                    rows.append(dict(part="A_reg", variant=f"{t}|{sn}->{tn}", period=per, n_days=n, slope=sl, nw_t=tt,
                                     corr=y.loc[a:b].corr(s.loc[a:b])))
        # benchmarks
        bh = P["c"][t].pct_change().reindex(f.index)
        add_rows("A_bench", f"{t}|buy_hold_close_to_close", bh)
        for tn, y in tgt.items():
            add_rows("A_bench", f"{t}|always_long_{tn}", y - cin[tn] - cout, y)
        for sn in ["s_pc", "s_o"]:
            s = sig[sn]
            thr = s.abs().rolling(250, min_periods=120).quantile(0.7).shift(1)
            pos_rules = {"ls": np.sign(s), "lo": (s > 0).astype(float),
                         "big_ls": np.sign(s) * (s.abs() > thr)}
            for tn, y in tgt.items():
                for rn, pos in pos_rules.items():
                    for fn, fm in [("all", pd.Series(True, index=f.index)), ("hivol", hv)]:
                        w = pos.where(fm, 0.0).fillna(0.0)
                        gross = w * y
                        trade = (w != 0).astype(float)
                        net = gross - trade * (cin[tn] + cout)
                        low = gross - trade * 2e-4
                        name = f"{t}|{sn}->{tn}|{rn}|{fn}"
                        add_rows("A_rule", name, net, gross)
                        add_rows("A_rule_lowcost", name, low, gross)


# ================================================================== B. cross-section
def load_m30():
    parts = []
    d0 = os.path.join(A.LOCAL, "m30s61")
    for fn in sorted(os.listdir(d0)):
        if not fn.endswith(".parquet"):
            continue
        d = pd.read_parquet(os.path.join(d0, fn), columns=["ts", "ticker", "o", "c", "v"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d["date"] = d.ts.dt.normalize()
        a = d[hm == 930].set_index(["date", "ticker"])
        b = d[hm == 1530].set_index(["date", "ticker"])
        x = pd.DataFrame({"a_o": a.o, "p1000": a.c, "v30": a.v}).join(
            pd.DataFrame({"e1530": b.o, "a_last": b.c}), how="outer")
        parts.append(x)
    x = pd.concat(parts)
    x = x.reset_index()
    x["ticker"] = x.ticker.str.replace(".", "-", regex=False)
    W = {k: x.pivot(index="date", columns="ticker", values=k) for k in ["a_o", "p1000", "v30", "e1530", "a_last"]}
    return W


def news_counts(days, cols, cut_hm=(10, 0)):
    """Articles per (day, symbol) from 16:00 of the previous trading day to cut_hm on the day."""
    import news_features as NF
    d = NF.load_news()
    d = d[d.sym.isin(set(cols))]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    cut = pd.DatetimeIndex(days) + pd.Timedelta(hours=cut_hm[0], minutes=cut_hm[1])
    start = pd.DatetimeIndex(np.r_[[pd.Timestamp("1990-01-01")], (pd.DatetimeIndex(days[:-1]) + pd.Timedelta(hours=16)).values])
    i = cut.searchsorted(ts.values, side="left")          # first day whose cutoff is >= ts
    ok = i < len(days)
    i = np.where(ok, i, 0)
    ok &= ts.values >= start[i]                            # inside [prev 16:00, cut)
    dd = pd.DataFrame({"day": pd.DatetimeIndex(days)[i[ok]], "sym": d.sym.values[ok]})
    n = dd.groupby(["day", "sym"]).size().unstack()
    return n.reindex(index=days, columns=cols).fillna(0), ts.min()


def part_b():
    W = load_m30()
    cols = sorted(set(stock_cols(P)) & set(W["a_o"].columns))
    days = P["c"].index[(P["c"].index >= "2020-01-02") & (P["c"].index <= W["a_o"].index.max())]
    W = {k: v.reindex(index=days, columns=cols) for k, v in W.items()}
    rs = lambda k: P[k][cols].reindex(days)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex(days)
    px = P["rawc"][cols].shift(1).reindex(days)
    close = rs("rawc")
    o_s = O_S[cols].reindex(days)
    pc = PC_DIV[cols].reindex(days)
    # consistency filter
    dif = np.maximum((W["a_o"] / o_s - 1).abs(), (W["a_last"] / close - 1).abs())
    tm = dif.groupby(dif.index.to_period("M")).transform("median")
    ok = (dif < 0.03) & (tm < 0.005) & W["p1000"].notna() & W["e1530"].notna()
    print("cross-section: stock-days with bars", int(W["a_o"].notna().sum().sum()), "kept", int(ok.sum().sum()), flush=True)
    rk = adv.where(px > 5).rank(axis=1, ascending=False)
    E = (rk <= 500) & ok
    print("eligible per day", E.sum(1).describe().round(0).to_dict(), flush=True)
    s_pc = W["p1000"] / pc - 1
    s_o = W["p1000"] / W["a_o"] - 1          # Alpaca first trade -> 10:00 (same source)
    rvol = W["v30"] / W["v30"].rolling(20, min_periods=10).mean().shift(1)
    nn, news_start = news_counts(days, cols)
    print("news from", news_start, flush=True)
    filt = {"all": E, "in_play": E & (rvol >= 2), "news": E & (nn > 0), "no_news": E & (nn == 0)}
    tgt = {"rod": close / W["p1000"] - 1, "l30": close / W["e1530"] - 1}
    cin = {"rod": C_1000[cols].reindex(days), "l30": C_1530[cols].reindex(days)}
    cout = C_AUC[cols].reindex(days)
    spy = P["c"]["SPY"].pct_change().reindex(days)
    add_rows("B_bench", "SPY|buy_hold_close_to_close", spy)
    for tn, y in tgt.items():
        Wt = E.astype(float).div(E.sum(1).replace(0, np.nan), axis=0).fillna(0)
        r = bt.run(Wt, y.where(E), (cin[tn] + cout) / 2)          # cost per side split as (in+out)/2, round trip
        add_rows("B_bench", f"universe_ew_always_long_{tn}", r.net, r.gross)
    info = []
    for sn, s in [("s_pc", s_pc), ("s_o", s_o)]:
        # rank IC per day (no costs), all eligible names
        for tn, y in tgt.items():
            ic = s.where(E).rank(axis=1).corrwith(y.where(E).rank(axis=1), axis=1)
            for per, a, b in PER:
                x = ic.loc[a:b].dropna()
                rows.append(dict(part="B_ic", variant=f"{sn}->{tn}", period=per, n_days=len(x), ic=x.mean(),
                                 nw_t=x.mean() / x.std() * np.sqrt(len(x))))
        for tn, y in tgt.items():
            cside = (cin[tn] + cout) / 2
            for fn, Ef in filt.items():
                Ef = Ef & y.notna() & s.notna()
                Wt = bt.select_topk(s, Ef, 10, largest=True)
                Wb = bt.select_topk(s, Ef, 10, largest=False)
                rt = bt.run(Wt, y, cside)
                rb = bt.run(Wb, y, cside)
                ls_net = 0.5 * rt.net - 0.5 * (rb.gross + rb.cost)      # half capital per side
                ls_gross = 0.5 * rt.gross - 0.5 * rb.gross
                base = f"{sn}->{tn}|{fn}"
                n_names = (Wt > 0).sum(1)
                ex = dict(avg_names_long=float(n_names[n_names > 0].mean()) if (n_names > 0).any() else 0)
                add_rows("B_rule", base + "|mom_ls", ls_net, ls_gross, ex)
                add_rows("B_rule", base + "|mom_long", rt.net, rt.gross, ex)
                add_rows("B_rule", base + "|rev_long", rb.net, rb.gross, ex)
                info.append((base, rt.n_missing.sum(), rb.n_missing.sum()))
    print("missing returns while held (should be 0):", sum(a + b for _, a, b in info), flush=True)


if __name__ == "__main__":
    part_a()
    if os.path.isdir(os.path.join(A.LOCAL, "m30s61")) and not os.environ.get("S61_A_ONLY"):
        part_b()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "study61_intraday_momentum.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    reg = df[df.part == "A_reg"].pivot_table(index="variant", columns="period", values=["slope", "nw_t"])
    print(reg[[("slope", "2020-23"), ("nw_t", "2020-23"), ("slope", "2024-26"), ("nw_t", "2024-26")]].round(3))
    for part in ["A_bench", "A_rule", "A_rule_lowcost", "B_bench", "B_rule"]:
        x = df[df.part == part]
        if len(x):
            v = x.pivot_table(index="variant", columns="period", values=["sharpe", "tstat"])
            cols_ = [c for c in [("sharpe", "2020-23"), ("tstat", "2020-23"), ("sharpe", "2024-26"), ("tstat", "2024-26")] if c in v]
            print("\n==", part)
            print(v[cols_].round(2).to_string())
    x = df[df.part == "B_ic"]
    if len(x):
        print(x.pivot_table(index="variant", columns="period", values=["ic", "nw_t"]).round(3))
