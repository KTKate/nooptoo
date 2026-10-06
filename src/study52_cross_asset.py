"""Study 52: do daily cross-asset moves predict sector ETF returns over the next night, next day session and 1-5 days?

Predictors (all available tickers; IEF, ^TNX, LQD, CL=F, BTC-USD are not in the panel):
  rates  TLT  (20y Treasury ETF)          dollar UUP           oil USO          gold GLD
  credit HYG  (high-yield ETF; carries equity beta, so SPY is used as a control in the multivariate model)
  vol    ^VIX (index, official close 16:15 ET)                 crypto BTC/USD (Alpaca hourly bars, 2021+)
  control SPY
Targets: XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC SMH XBI KRE, raw and in excess of SPY (same window).

Timing (what is known at each decision):
  Close decision "C" (signal at 15:45 ET day t, entry in the closing auction of t):
    ETFs  : log(price at 15:45 t / adjusted close t-1)   [1-minute bars, last trade 09:30-15:44 bar]
    BTC   : log(BTC at 15:00 ET t / BTC at 15:00 ET t-1)  [close of the hourly bar ending 15:00 ET]
    VIX   : log(VIX close t-1 / VIX close t-2)  (VIX for day t settles after 16:00: lagged one day)
    5-day versions: from close t-5 (BTC: 15:00 ET five trading days earlier; VIX: t-6 to t-1).
    Targets: night  close t -> open t+1;  day1 open t+1 -> close t+1;  cc1..cc5 close t -> close t+h.
  Open decision "O" (signal at 09:25 ET day t, entry in the opening auction of t):
    ETFs  : log(last premarket trade 07:00-09:24 t / adjusted close t-1) (0 if no premarket trade)
    BTC   : log(BTC at 09:00 ET t / BTC at 16:00 ET t-1)
    VIX   : log(VIX close t-1 / VIX close t-2)
    Target: day0 open t -> close t.
Regressions: univariate OLS per (sector, predictor, target), standardized predictor, Newey-West t (lags h+1),
by period 2020-23 vs 2024-26 (2026 to 09-25, the last 1-minute bar file).
Rules: per-sector multivariate OLS of the excess target on the 8 predictors, refitted each January on all earlier
data (expanding window, targets known before the fit date). 16 candidate rules (4 targets x top1/top3 x
long-only/long-minus-SPY) evaluated on 2021-23 (2020 has no out-of-sample prediction); the 3 best by net excess
Sharpe vs SPY are the chosen rules, then read on 2024-26. Costs per side: exec_cost_bps(P, 'auction') + 2.5 bp.
Output: results/study52_cross_asset.csv (sections: check, reg, reg_summary, rule).
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, exec_cost_bps, ann_stats, DATA, RES  # noqa: E402

SCR = os.environ.get("S52_CACHE", "/tmp/claude-0/-home-user-nooptoo/582a0ce2-853b-5fb5-9a27-c0a484705892/scratchpad")
SECT = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE"]
PRED_ETF = ["TLT", "UUP", "USO", "GLD", "HYG", "SPY"]
PREDS = ["TLT", "UUP", "USO", "GLD", "HYG", "BTC", "VIX", "SPY"]
PERIODS = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-25")]
EXTRA = 2.5
END = "2026-09-25"


# ------------------------------------------------------------------ intraday snapshots from 1-minute bars
def snapshots():
    fn = os.path.join(SCR, "s52_snap.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "local", "m1", "*.parquet"))):
        d = pd.read_parquet(f, columns=["ts", "ticker", "o", "c"])
        d["date"] = d.ts.dt.normalize()
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        g = []
        for name, m, col, how in [("p1545", (hm >= 930) & (hm <= 1544), "c", "last"),
                                  ("p0925", hm <= 924, "c", "last"),
                                  ("c_last", (hm >= 930) & (hm <= 1559), "c", "last"),
                                  ("o930", (hm >= 930) & (hm <= 931), "o", "first")]:
            s = d[m].sort_values("ts").groupby(["date", "ticker"])[col].agg(how).rename(name)
            g.append(s)
        out.append(pd.concat(g, axis=1).reset_index())
    s = pd.concat(out, ignore_index=True)
    s.to_parquet(fn)
    return s


def btc_series():
    h = pd.read_parquet(os.path.join(DATA, "local", "crypto", "bars_1Hour.parquet"), columns=["symbol", "t", "c"])
    h = h[h.symbol == "BTC/USD"].copy()
    end_et = (h.t + pd.Timedelta(hours=1)).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.Series(h.c.values, index=end_et.values).sort_index()      # price at bar END time (ET)


def asof(ser, when):
    """Last value of ser at or before each timestamp in `when`, NaN if older than 3 hours."""
    idx = ser.index.searchsorted(when, side="right") - 1
    v = np.where(idx >= 0, ser.values[np.clip(idx, 0, None)], np.nan)
    age = np.where(idx >= 0, (when - ser.index[np.clip(idx, 0, None)]) / pd.Timedelta(hours=1), np.inf)
    return np.where(age <= 3, v, np.nan)


def nw_t(y, x, lags):
    """OLS y = a + b x; returns b, Newey-West t of b, n."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 60 or x.std() == 0:
        return np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX_i = np.linalg.inv(X.T @ X)
    b = XtX_i @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_i @ S @ XtX_i
    return b[1], b[1] / np.sqrt(V[1, 1]), n


# ------------------------------------------------------------------ build
def build():
    P = load_panel()
    days = P["c"].index
    days = days[(days >= "2019-07-01") & (days <= END)]
    tick = sorted(set(SECT + PRED_ETF))
    c = P["c"][tick].reindex(days).astype("float64")
    o = P["o"][tick].reindex(days).astype("float64")
    rawc = P["rawc"][tick].reindex(days).astype("float64")
    f = c / rawc
    vix = P["c"]["^VIX"].reindex(days).astype("float64")
    cost = (exec_cost_bps(P, "auction")[tick].reindex(days).astype("float64") + EXTRA) / 1e4

    s = snapshots()
    piv = {k: s.pivot(index="date", columns="ticker", values=k).reindex(index=days, columns=tick) for k in
           ["p1545", "p0925", "c_last", "o930"]}
    checks = []
    # data check 1: last regular-session 1-minute trade vs official (raw) daily close
    dev = np.log(piv["c_last"] / rawc)
    for t in tick:
        x = dev[t].dropna()
        checks.append(dict(section="check", item="m1_last_vs_close", ticker=t, n=len(x),
                           median_bp=1e4 * x.median(), p99_abs_bp=1e4 * x.abs().quantile(0.99),
                           n_bad=int((x.abs() > 0.02).sum())))
    bad = dev.abs() > 0.02                                 # day where the minute bars disagree with the daily store
    p1545 = (piv["p1545"] * f).mask(bad)
    p0925 = (piv["p0925"] * f).mask(bad)
    # data check 2: daily returns, extreme values
    r = np.log(c / c.shift(1))
    for t in tick:
        x = r[t].dropna()
        checks.append(dict(section="check", item="daily_ret_extremes", ticker=t, n=len(x),
                           median_bp=1e4 * x.min(), p99_abs_bp=1e4 * x.max(), n_bad=int((x.abs() > 0.15).sum())))
    # data check 3: premarket coverage
    for t in tick:
        x = piv["p0925"][t].reindex(days)
        checks.append(dict(section="check", item="premarket_trade_share", ticker=t, n=int(x.notna().sum()),
                           median_bp=np.nan, p99_abs_bp=np.nan, n_bad=int(x.isna().sum())))

    btc = btc_series()
    d = pd.DatetimeIndex(days)
    b1500 = pd.Series(asof(btc, d + pd.Timedelta(hours=15)), index=days)
    b0900 = pd.Series(asof(btc, d + pd.Timedelta(hours=9)), index=days)
    b1600 = pd.Series(asof(btc, d + pd.Timedelta(hours=16)), index=days)
    checks.append(dict(section="check", item="btc_1500_coverage", ticker="BTC", n=int(b1500.loc["2021-01-04":].notna().sum()),
                       median_bp=np.nan, p99_abs_bp=np.nan, n_bad=int(b1500.loc["2021-01-04":].isna().sum())))

    lc = np.log(c)
    X = {}
    # close decision
    XC1 = pd.DataFrame(index=days)
    XC5 = pd.DataFrame(index=days)
    for t in PRED_ETF:
        XC1[t] = np.log(p1545[t]) - lc[t].shift(1)
        XC5[t] = np.log(p1545[t]) - lc[t].shift(5)
    XC1["BTC"] = np.log(b1500 / b1500.shift(1))
    XC5["BTC"] = np.log(b1500 / b1500.shift(5))
    XC1["VIX"] = np.log(vix.shift(1) / vix.shift(2))
    XC5["VIX"] = np.log(vix.shift(1) / vix.shift(6))
    XO1 = pd.DataFrame(index=days)
    for t in PRED_ETF:
        XO1[t] = (np.log(p0925[t]) - lc[t].shift(1)).fillna(0.0).where(c[t].notna())
    XO1["BTC"] = np.log(b0900 / b1600.shift(1))
    XO1["VIX"] = np.log(vix.shift(1) / vix.shift(2))
    X = {"C1": XC1[PREDS], "C5": XC5[PREDS], "O1": XO1[PREDS]}
    for k, v in X.items():
        checks.append(dict(section="check", item=f"features_{k}_abs_gt_20pct", ticker=",".join(PREDS),
                           n=int(v.notna().sum().sum()), median_bp=np.nan, p99_abs_bp=np.nan,
                           n_bad=int((v.abs() > 0.2).sum().sum())))

    lo = np.log(o)
    Y = {"night": lo.shift(-1) - lc, "day1": lc.shift(-1) - lo.shift(-1), "day0": lc - lo}
    for h in range(1, 6):
        Y[f"cc{h}"] = lc.shift(-h) - lc
    return dict(P=P, days=days, c=c, o=o, X=X, Y=Y, cost=cost, checks=pd.DataFrame(checks))


TARGET_X = {"night": "C", "day1": "C", "cc1": "C", "cc2": "C", "cc3": "C", "cc4": "C", "cc5": "C", "day0": "O"}
HOR = {"night": 1, "day1": 1, "day0": 1, "cc1": 1, "cc2": 2, "cc3": 3, "cc4": 4, "cc5": 5}


def regressions(B):
    rows = []
    for tgt, dec in TARGET_X.items():
        looks = ["1", "5"] if dec == "C" else ["1"]
        Yt = B["Y"][tgt]
        for look in looks:
            Xd = B["X"][dec + look]
            for kind in ["raw", "excess"]:
                for sec in SECT:
                    y = Yt[sec] - (Yt["SPY"] if kind == "excess" else 0)
                    for pr in PREDS:
                        for per, a, b in PERIODS:
                            yy, xx = y.loc[a:b].values, Xd[pr].loc[a:b].values
                            sd = np.nanstd(xx)
                            beta, t, n = nw_t(yy, xx / sd if sd > 0 else xx, HOR[tgt] + 1)
                            rows.append(dict(section="reg", target=tgt, decision=dec, lookback=look, kind=kind,
                                             sector=sec, predictor=pr, period=per, n=n,
                                             beta_bp_per_sd=1e4 * beta, nw_t=t))
    return pd.DataFrame(rows)


def reg_summary(R):
    rows = []
    for (tgt, look, kind, pr), g in R.groupby(["target", "lookback", "kind", "predictor"]):
        w = g.pivot(index="sector", columns="period", values="nw_t")
        bb = g.pivot(index="sector", columns="period", values="beta_bp_per_sd")
        a, b = w["2020-23"], w["2024-26"]
        rows.append(dict(section="reg_summary", target=tgt, lookback=look, kind=kind, predictor=pr,
                         n_sig_2020_23=int((a.abs() > 2).sum()), n_sig_2024_26=int((b.abs() > 2).sum()),
                         n_sig_both_same_sign=int(((a.abs() > 2) & (b.abs() > 2) & (np.sign(a) == np.sign(b))).sum()),
                         corr_t_across_periods=a.corr(b),
                         mean_abs_beta_bp_2020_23=bb["2020-23"].abs().mean(),
                         mean_abs_beta_bp_2024_26=bb["2024-26"].abs().mean(),
                         best_sector_2020_23=a.abs().idxmax(), best_t_2020_23=a.loc[a.abs().idxmax()],
                         same_sector_t_2024_26=b.loc[a.abs().idxmax()]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rules
def predictions(B, tgt):
    """Expanding-window per-sector OLS of excess target on the 8 predictors, refitted each January."""
    dec = TARGET_X[tgt]
    look = "5" if tgt == "cc5" else "1"
    Xd = B["X"][dec + look].fillna(0.0)           # BTC before 2021 -> 0 (contributes nothing)
    Yt = B["Y"][tgt]
    h = HOR[tgt]
    days = B["days"]
    pred = pd.DataFrame(np.nan, index=days, columns=SECT)
    for yr in range(2021, 2027):
        start = pd.Timestamp(f"{yr}-01-01")
        tr_idx = np.where(days < start)[0]
        tr_idx = tr_idx[:len(tr_idx) - h]          # targets fully known before the fit date
        te = (days >= start) & (days < pd.Timestamp(f"{yr + 1}-01-01"))
        Xtr = Xd.iloc[tr_idx].values
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        Z = np.column_stack([np.ones(len(tr_idx)), (Xtr - mu) / sd])
        Zte = np.column_stack([np.ones(te.sum()), (Xd.loc[te].values - mu) / sd])
        for sec in SECT:
            y = (Yt[sec] - Yt["SPY"]).iloc[tr_idx].values
            m = np.isfinite(y) & B["c"][sec].iloc[tr_idx].notna().values
            if m.sum() < 150:
                continue
            coef = np.linalg.lstsq(Z[m], y[m], rcond=None)[0]
            pred.loc[te, sec] = Zte @ coef
    return pred


def run_rule(B, tgt, k, hedged, pred):
    days = B["days"]
    c, o, cost = B["c"], B["o"], B["cost"]
    S = pred.where(c[SECT].notna())
    rk = S.rank(axis=1, ascending=False, method="first")
    W = ((rk <= k) & S.notna()).astype(float).div(k)
    W = W.mul(S.notna().sum(1).ge(k), axis=0)
    spyw = W.sum(1)                                  # 1 when invested
    cs, cspy = cost[SECT], cost["SPY"]
    if tgt in ("night", "day0"):
        if tgt == "night":
            R = (o.shift(-1) / c - 1)
        else:
            R = (c / o - 1)
        g = (W * R[SECT]).sum(1)
        spy_r = spyw * R["SPY"]
        tc = (2 * W * cs).sum(1)
        tspy = 2 * spyw * cspy
        turn = 2 * W.sum(1)
    else:
        h = HOR[tgt]
        Wh = sum(W.shift(i).fillna(0.0) for i in range(h)) / h
        R = c.shift(-1) / c - 1                        # close t -> close t+1, earned by holdings set at t
        g = (Wh * R[SECT]).sum(1)
        spyh = Wh.sum(1)
        spy_r = spyh * R["SPY"]
        dW = Wh.diff().abs()
        dW.iloc[0] = Wh.iloc[0]
        tc = (dW * cs).sum(1)
        tspy = spyh.diff().abs().fillna(0) * cspy
        turn = dW.sum(1)
    net = g - tc
    if hedged:
        net = net - spy_r - tspy
    out = pd.DataFrame({"gross": g, "net": net, "spy_same": spy_r, "excess_net": g - tc - spy_r, "turn": turn})
    return out


def rule_rows(name, df, chosen):
    rows = []
    spy_bh = None
    for per, a, b in [("2021-23", "2021-01-01", "2023-12-31"), ("2024-26", "2024-01-01", END)]:
        x = df.loc[a:b].dropna()
        sn, se, sg, ss = ann_stats(x.net), ann_stats(x.excess_net), ann_stats(x.gross), ann_stats(x.spy_same)
        rows.append(dict(section="rule", rule=name, chosen=chosen, period=per, days=sn["n"],
                         net_sharpe=sn["sharpe"], net_ann=sn["ann_ret"], net_maxdd=sn["maxdd"],
                         gross_sharpe=sg["sharpe"], gross_bp_day=1e4 * x.gross.mean(), net_bp_day=1e4 * x.net.mean(),
                         excess_vs_spy_bp_day=1e4 * x.excess_net.mean(), excess_vs_spy_sharpe=se["sharpe"],
                         excess_t=se["tstat"], spy_same_window_sharpe=ss["sharpe"], turnover_day=x.turn.mean()))
    return rows


def main():
    B = build()
    out = [B["checks"]]
    print(B["checks"].to_string(), flush=True)
    R = regressions(B)
    out.append(R)
    RS = reg_summary(R)
    out.append(RS)
    print(RS.sort_values("n_sig_both_same_sign", ascending=False).head(30).to_string(), flush=True)
    # rules
    rr, nets = [], {}
    preds = {t: predictions(B, t) for t in ["night", "day0", "cc1", "cc5"]}
    for tgt in ["night", "day0", "cc1", "cc5"]:
        for k in [1, 3]:
            for hedged in [False, True]:
                name = f"{tgt}_top{k}_{'minusSPY' if hedged else 'long'}"
                df = run_rule(B, tgt, k, hedged, preds[tgt])
                nets[name] = df
                rr += rule_rows(name, df, False)
    RR = pd.DataFrame(rr)
    dev = RR[RR.period == "2021-23"].sort_values("excess_vs_spy_sharpe", ascending=False)
    # one rule per (target, k): long and minus-SPY share the same excess series, keep distinct picks
    seen, chosen = set(), []
    for _, row in dev.iterrows():
        key = row.rule.rsplit("_", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row.rule)
        if len(chosen) == 3:
            break
    # long and minus-SPY forms of one (target, k) share the same excess-over-SPY series: mark both
    keys = [c.rsplit("_", 1)[0] for c in chosen]
    RR["chosen"] = RR.rule.str.rsplit("_", n=1).str[0].isin(keys)
    # SPY buy and hold reference
    P = B["P"]
    spy = (P["c"]["SPY"].astype("float64").pct_change()).loc["2019-07-01":END]
    for per, a, b in [("2021-23", "2021-01-01", "2023-12-31"), ("2024-26", "2024-01-01", END)]:
        s = ann_stats(spy.loc[a:b])
        RR = pd.concat([RR, pd.DataFrame([dict(section="rule", rule="SPY_buy_and_hold", chosen=False, period=per,
                                                days=s["n"], net_sharpe=s["sharpe"], net_ann=s["ann_ret"],
                                                net_maxdd=s["maxdd"])])])
    out.append(RR)
    print(RR.to_string(), flush=True)
    print("chosen:", chosen)
    pd.concat(out, ignore_index=True).to_csv(os.path.join(RES, "study52_cross_asset.csv"), index=False)


if __name__ == "__main__":
    main()
