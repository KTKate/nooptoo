"""Study 19: portfolio construction variants for the overnight LightGBM ranker (buy at the close auction, sell at
the next opening auction).

Predictions: results/study3_pred_night_1545_none.parquet (15:45 features, 2024-01..2026-09; periods val and oos)
and results/study10_pred_base.parquet (close features; used for 2022-23 only).
Costs per side: core.exec_cost_bps(P, "auction") + 2.5 bp (same for the SPY hedge). Shorts: borrow fee per night
0.3 bp if 20d median dollar volume > $50M, else 2 bp; shorts need prior traded close > $10.
Variants: long-short (dollar / beta neutral), SPY overnight hedge, weighting, persistence, regime filters,
minimum predicted score. Output: results/study19_summary.csv, results/study19_series.csv, results/study19_dsr.csv
"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "1"
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, RES
import bt

P = load_panel()
predA = pd.read_parquet(f"{RES}/study3_pred_night_1545_none.parquet")["pred"].unstack()
predB = pd.read_parquet(f"{RES}/study10_pred_base.parquet")["pred"].unstack()
sc = set(stock_cols(P))
cols = sorted((set(predA.columns) | set(predB.columns)) & sc)
predA = predA.reindex(columns=cols).astype("float32")
predB = predB.loc[:"2023-12-31"].reindex(columns=cols).astype("float32")
idx = P["c"].index
o, c, rawc, dv = (P[k][cols] for k in ["o", "c", "rawc", "dv"])
Rn = o.shift(-1) / c - 1                        # overnight return, close t -> open t+1
Ri1 = c.shift(-1) / o.shift(-1) - 1             # intraday return of day t+1, dated t
Rcc = c.shift(-1) / c - 1                       # close t -> close t+1
C = exec_cost_bps(P, "auction")[cols] + 2.5
adv = dv.rolling(20, min_periods=10).median().shift(1)
tc = traded_close(P)[cols].shift(1)
pxs = pd.concat([rawc.shift(1).loc[:"2023-11-30"], tc.loc["2023-12-01":]])   # raw close before 2023-12
fee = pd.DataFrame(np.where(adv > 5e7, 0.3, 2.0), index=idx, columns=cols) / 1e4
r = c / c.shift(1) - 1
m = P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1
M = pd.DataFrame(np.where(r.notna(), m.values[:, None], np.nan), index=idx, columns=cols)
mr = (r * M).rolling(60, min_periods=40).mean()
beta = ((mr - r.rolling(60, min_periods=40).mean() * M.rolling(60, min_periods=40).mean())
        / M.rolling(60, min_periods=40).var(ddof=0)).shift(1).clip(-1, 5)
del mr, M
vol20 = np.log(c / c.shift(1)).rolling(20, min_periods=10).std().shift(1)
spyN = P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1
spyCC = P["c"]["SPY"].shift(-1) / P["c"]["SPY"] - 1
Cspy = exec_cost_bps(P, "auction")["SPY"] + 2.5
vix = P["c"]["^VIX"].shift(1)
vix_th = vix.shift(1).rolling(250, min_periods=60).quantile(0.9)

# SPY at 15:45 (close of the 15:44 1-minute bar) vs the previous day's 15:59 bar close
import alpaca_data as A
s1 = A.read("m1", tickers=["SPY"], columns=["ts", "c"])
s1["d"] = s1.ts.dt.normalize()
hm = s1.ts.dt.strftime("%H:%M")
p1544 = s1[hm == "15:44"].set_index("d")["c"]
p1559 = s1[hm == "15:59"].set_index("d")["c"]
pc = p1559.reindex(idx).shift(1)
spy1545 = (p1544.reindex(idx) / pc - 1)
del s1


def cap_norm(w, cap=0.20, it=20):
    """Row-normalise nonnegative weights and cap each at `cap`, redistributing the excess."""
    w = np.nan_to_num(np.asarray(w, dtype="float64"))
    s = w.sum(1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
    for _ in range(it):
        over = w > cap + 1e-12
        if not over.any():
            break
        exc = np.where(over, w - cap, 0).sum(1, keepdims=True)
        w = np.where(over, cap, w)
        free = np.where(~over & (w > 0), w, 0)
        fs = free.sum(1, keepdims=True)
        w = w + np.divide(free, fs, out=np.zeros_like(free), where=fs > 0) * exc
    return w


def night(WL, WS=None, hedge=None, skip=None, D=None):
    """Nightly round trip. WL/WS: long/short weights (positive numbers); hedge: SPY short notional (Series)."""
    Rd, Cd = Rn.reindex(D).fillna(0.0), C.reindex(D).fillna(100.0) / 1e4
    WL = WL.reindex(index=D, columns=cols).fillna(0.0)
    g = (WL * Rd).sum(1)
    k = 2 * (WL * Cd).sum(1)
    if WS is not None:
        WS = WS.reindex(index=D, columns=cols).fillna(0.0)
        g -= (WS * Rd).sum(1)
        k += 2 * (WS * Cd).sum(1) + (WS * fee.reindex(D)).sum(1)
    if hedge is not None:
        h = hedge.reindex(D).fillna(0.0)
        g -= h * spyN.reindex(D)
        k += h * (2 * Cspy.reindex(D) / 1e4 + 0.3e-4)
    net = g - k
    if skip is not None:
        net = net.where(~skip.reindex(D).fillna(False).astype(bool), 0.0)
    return net


def trailing_pct(x, win=250, minp=60):
    """Percentile of x_t within x_{t-win..t-1}."""
    v = x.values
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        h = v[max(0, i - win):i]
        h = h[~np.isnan(h)]
        if len(h) >= minp and not np.isnan(v[i]):
            out[i] = (h < v[i]).mean()
    return pd.Series(out, index=x.index)


def variants(S):
    """All variants for one prediction panel S (date x ticker). Returns dict name -> (net series, benchmark)."""
    D = S.index[spyN.reindex(S.index).notna().values]
    S = S.loc[D]
    E = S.notna()
    V = {}
    top = {k: bt.select_topk(S, E, k) for k in [5, 10, 20]}
    shortE = E & (pxs.reindex(D) > 10)
    bot = {k: bt.select_topk(S, shortE, k, largest=False) for k in [5, 10, 20]}
    b = beta.reindex(D).fillna(1.0)
    V["baseline_top10_equal"] = night(top[10], D=D)
    for k in [5, 10, 20]:
        V[f"ls_dollar_k{k}"] = night(top[k], bot[k], D=D)
        bl, bs = (top[k] * b).sum(1), (bot[k] * b).sum(1)
        ratio = (bl / bs.where(bs > 0.2)).clip(0.25, 4).fillna(1.0)
        V[f"ls_beta_k{k}"] = night(top[k], bot[k].mul(ratio, axis=0), D=D)
    bp = (top[10] * b).sum(1)
    for h in [0.5, 1.0, 1.5]:
        V[f"hedge_spy_{h}xbeta"] = night(top[10], hedge=h * bp, D=D)
    # extra: hedge with the trailing overnight beta of the unhedged strategy (120 nights, lagged)
    base = V["baseline_top10_equal"]
    sN = spyN.reindex(D)
    ob = (base.rolling(120, min_periods=60).cov(sN) / sN.rolling(120, min_periods=60).var()).shift(1).clip(0, 4)
    V["hedge_spy_1.0x_ovnbeta"] = night(top[10], hedge=ob.fillna(bp), D=D)
    # weighting (top 10, cap 20%)
    sel = (top[10] > 0).values
    iv = np.where(sel, 1 / vol20.reindex(D).clip(lower=0.003).values, 0)
    V["w_invvol_cap20"] = night(pd.DataFrame(cap_norm(iv), index=D, columns=cols), D=D)
    rk = np.where(sel, S.where(top[10] > 0).rank(axis=1, ascending=True).values, 0)
    V["w_rank_cap20"] = night(pd.DataFrame(cap_norm(rk), index=D, columns=cols), D=D)
    # persistence
    W = top[10]
    Cd, Rd, Rid = C.reindex(D).fillna(100.0) / 1e4, Rn.reindex(D).fillna(0.0), Ri1.reindex(D).fillna(0.0)
    inprev, innext = (W.shift(1) > 0).astype(float), (W.shift(-1) > 0).astype(float)
    g = (W * Rd).sum(1) + (W * innext * Rid).sum(1)
    k = (W * Cd * (1 - inprev)).sum(1) + (W * Cd * (1 - innext)).sum(1)
    V["persist_lookahead"] = g - k                          # needs day t+1 15:45 ranks at the open: not tradable
    dW = (W - W.shift(1).fillna(0.0)).abs()
    V["persist_hold_to_close"] = (W * Rcc.reindex(D).fillna(0.0)).sum(1) - (dW * Cd).sum(1)
    # regime filters (skip night = cash)
    disp = S.std(axis=1)
    skip_d = trailing_pct(disp) < 0.2
    skip_v = vix.reindex(D) > vix_th.reindex(D)
    skip_s = spy1545.reindex(D) < -0.01
    V["skip_low_dispersion"] = night(W, skip=skip_d, D=D)
    V["skip_high_vix"] = night(W, skip=skip_v, D=D)
    V["skip_spy_down1pct"] = night(W, skip=skip_s, D=D)
    V["skip_any_of_3"] = night(W, skip=skip_d | skip_v | skip_s, D=D)
    # minimum predicted score: trailing percentile of the pooled top-10 predictions (250 days, lagged)
    top10vals = S.where(top[10] > 0).values
    for p in [25, 50, 75]:
        th = np.full(len(D), np.nan)
        for i in range(len(D)):
            h = top10vals[max(0, i - 250):i]
            h = h[~np.isnan(h)]
            if len(h) >= 600:
                th[i] = np.percentile(h, p)
        keep = S.values > np.where(np.isnan(th), -np.inf, th)[:, None]
        V[f"minscore_p{p}"] = night(W.where(keep, 0.0), D=D)
    bench = {k: (spyCC if k == "persist_hold_to_close" else spyN) for k in V}
    return V, bench, {"low_disp": skip_d, "high_vix": skip_v, "spy_down": skip_s}


PER = {"A": [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")],
       "B": [("2022-23", "2022-01", "2023-12")]}
rows, series = [], {}
for tag, S in [("A", predA), ("B", predB)]:
    V, bench, skips = variants(S)
    for name, x in V.items():
        if tag == "A":
            series[name] = x
        for per, a, e in PER[tag]:
            y = x.loc[a:e].dropna()
            st = ann_stats(y)
            X = pd.concat([y, bench[name].reindex(y.index)], axis=1).dropna()
            bb, aa = np.polyfit(X.iloc[:, 1], X.iloc[:, 0], 1)
            rows.append(dict(variant=name, pred="1545" if tag == "A" else "close", period=per, sharpe=st["sharpe"],
                             ann_ret=st["ann_ret"], maxdd=st["maxdd"], ann_vol=st["ann_vol"],
                             alpha_ann=252 * aa, beta=bb, days_in_mkt=float((y != 0).mean()), n=len(y)))
    for s, v in skips.items():
        print(tag, "skip fraction", s, round(float(v.mean()), 3))
df = pd.DataFrame(rows)
n_var = df.variant.nunique() - 1                 # excluding the baseline
wide = df.pivot_table(index="variant", columns="period", values="sharpe")
bl = wide.loc["baseline_top10_equal"]
wide["beats_val_and_oos"] = (wide["val"] > bl["val"]) & (wide["oos"] > bl["oos"])
wide["beats_2022_23_too"] = wide["beats_val_and_oos"] & (wide["2022-23"] > bl["2022-23"])
tradable = [v for v in wide.index if v != "persist_lookahead"]
best = wide.loc[tradable, "val"].idxmax()
dsr_rows = []
for v in dict.fromkeys([best, "baseline_top10_equal"]):
    for per, a, e in PER["A"][1:]:
        x = series[v].loc[a:e].dropna()
        dsr_rows.append(dict(variant=v, period=per, n_trials=n_var + 700, sr_daily=x.mean() / x.std(),
                             dsr=deflated_sharpe(x.mean() / x.std(), n_var + 700, len(x), skew=float(x.skew()),
                                                 kurt=float(x.kurt() + 3))))
df.to_csv(f"{RES}/study19_summary.csv", index=False)
wide.to_csv(f"{RES}/study19_sharpe_wide.csv")
pd.DataFrame(series).to_csv(f"{RES}/study19_series.csv")
pd.DataFrame(dsr_rows).to_csv(f"{RES}/study19_dsr.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(df.round(3).to_string())
print(wide.round(2).to_string())
print("variants (excl. baseline):", n_var, "best by val:", best)
print(pd.DataFrame(dsr_rows).round(3).to_string())
