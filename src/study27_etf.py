"""Study 27: simple ETF rotation / timing strategies traded at the closing auction vs SPY/QQQ buy and hold.

Signals use the 15:45 ET price (from 1-minute bars, dividend-adjusted by scaling with the daily adjusted close)
as a stand-in for today's close, so an MOC order can be sent before the 15:50 cutoff. Orders fill at the
official close. VIX-based and realized-vol signals use data through the previous close (1-day lag).
Costs per side: 2 bp (1 bp + 1 bp slippage) for SPY/QQQ/IWM/sector/TLT/GLD, 3 bp for leveraged ETFs.
Cash earns 0. Long only, no margin (gross exposure <= 1).
Periods: dev 2020-06-01..2023-12-31 (data starts 2019-06; 12-month momentum and 200-day averages need a year
of history, so dev starts 2020-06 for every row, including buy and hold), val 2024-01..2025-06,
holdout 2025-07..end of data.
Outputs: results/study27_variants.csv (every variant x period), results/study27_best.csv (per-test best by dev
Sharpe, with deflated Sharpe and bootstrap vs SPY).
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from core import load_panel, ETFS, ann_stats, deflated_sharpe, RES
import alpaca_data as A

FIRST = pd.Timestamp("2020-06-03")  # every lookback (252 days) is available from here; entry at this close
PER = [("dev", "2020-06-04", "2023-12-31"), ("val", "2024-01-01", "2025-06-30"), ("hold", "2025-07-01", "2026-09-30"),
       ("full", "2020-06-04", "2026-09-30")]
SECT = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
LEV = ["TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "UPRO", "TNA", "TZA"]
TR = [t for t in ETFS if not t.startswith("^")]

P = load_panel()
C = P["c"][TR].astype("float64")
O = P["o"][TR].astype("float64")
V = P["c"][["^VIX", "^VIX3M"]].astype("float64")
days = C.index
R = C.pct_change()
COST = pd.Series({t: (3.0 if t in LEV else 2.0) / 1e4 for t in TR})


# ---------------------------------------------------------------- 15:45 snapshot
def snapshot():
    out = []
    for m in A.months("2019-06", str(days[-1])[:7]):
        d = A.read("m1", start=m, end=m, tickers=TR, columns=["ts", "ticker", "c"])
        if d.empty:
            continue
        tod = d.ts.dt.hour * 60 + d.ts.dt.minute
        d = d[(tod >= 570) & (tod < 960)].copy()
        d["tod"] = tod[d.index]
        d["date"] = d.ts.dt.normalize()
        g = d.groupby(["date", "ticker"])
        last = g.tail(1).set_index(["date", "ticker"])
        d = d.join(last["tod"].rename("end"), on=["date", "ticker"])
        # snapshot: last bar at or before 15 minutes before the session's last bar (15:45 normal, 12:45 half days)
        s = d[d.tod <= d.end - 14].groupby(["date", "ticker"]).tail(1).set_index(["date", "ticker"])["c"]
        out.append(pd.DataFrame({"snap": s, "last": last["c"]}).dropna())
    x = pd.concat(out)
    ratio = (x.snap / x["last"]).unstack("ticker")
    ratio.index = pd.DatetimeIndex(ratio.index.astype("datetime64[ns]"))
    ratio = ratio.reindex(index=days, columns=TR)
    return ratio


ratio = snapshot()
cov = ratio.notna().mean()
S = C * ratio
S = S.fillna(C.shift(1))  # no bar: fall back to yesterday's close (no look-ahead)
print("15:45 coverage min/median:", round(cov.min(), 3), round(cov.median(), 3))


def hist_with_snap(t):
    """Matrix helper: closes through t-1 plus snapshot at t, for rolling indicators known at 15:45."""
    return C[t].shift(1), S[t]


def sma_snap(t, n):
    return (C[t].shift(1).rolling(n - 1).sum() + S[t]) / n


# ---------------------------------------------------------------- simulator
def simulate(target, cost=COST):
    """target: DataFrame (days x tickers) of weights decided at close t (NaN row = no rebalance).
    Returns net daily returns, exposure, orders per day."""
    cols = list(target.columns)
    tg = target.values
    rr = R[cols].fillna(0.0).values
    cc = cost[cols].values
    n = len(days)
    w = np.zeros(len(cols))
    ret = np.zeros(n)
    expo = np.zeros(n)
    orders = np.zeros(n)
    for i in range(n):
        if i > 0:
            g = w * rr[i]
            port = g.sum()
            ret[i] = port
            w = w * (1 + rr[i]) / (1 + port) if (1 + port) != 0 else w
        expo[i] = w.sum()
        row = tg[i]
        if not np.all(np.isnan(row)):
            new = np.nan_to_num(row)
            dw = np.abs(new - w)
            trade = dw > 1e-4
            ret[i] -= (dw * cc)[trade].sum()
            orders[i] = trade.sum()
            w = np.where(trade, new, w)
        # exposure held overnight into i+1
    return pd.Series(ret, days), pd.Series(expo, days), pd.Series(orders, days)


ROWS, SER = [], {}


def record(test, name, ret, expo, orders, params=""):
    SER[(test, name)] = ret
    for per, a, b in PER:
        r = ret.loc[a:b]
        st = ann_stats(r)
        e = expo.shift(1).loc[a:b]  # exposure during day
        ROWS.append(dict(test=test, variant=name, params=params, period=per, ann_ret=st["ann_ret"],
                         ann_vol=st["ann_vol"], sharpe=st["sharpe"], maxdd=st["maxdd"],
                         time_in_mkt=(e > 0.01).mean(), avg_expo=e.mean(),
                         orders_per_yr=orders.loc[a:b].sum() / len(r) * 252, n=len(r)))


# ---------------------------------------------------------------- benchmarks
for t in ["SPY", "QQQ", "TLT", "UPRO", "TQQQ", "SPXL", "SOXL", "TNA"]:
    r = R[t].fillna(0.0)
    record("bench", f"{t}_buyhold", r, pd.Series(1.0, days), pd.Series(0.0, days))

# ---------------------------------------------------------------- test 1: rotation
U = SECT + ["TLT", "GLD"]
spy_ma200 = sma_snap("SPY", 200)
trend_on = S["SPY"] > spy_ma200
wk = pd.Series(days.isocalendar().week.values, index=days)
yr = pd.Series(days.year, index=days)
last_wk = (wk != wk.shift(-1)) | (yr != yr.shift(-1))
mo = pd.Series(days.month, index=days)
last_mo = mo != mo.shift(-1)
cols1 = U
n1 = 0
for L in [21, 63, 126, 252]:
    mom = S[U] / C[U].shift(L) - 1
    for N in [1, 2, 3]:
        rk = mom.rank(axis=1, ascending=False)
        pick = (rk <= N).astype(float) / N
        pick = pick.where(mom.notna().sum(axis=1) >= len(U) - 2, np.nan)
        for fq, reb in [("W", last_wk), ("M", last_mo)]:
            for filt in ["none", "tlt", "cash"]:
                tgt = pick.copy()
                if filt != "none":
                    off = ~trend_on
                    alt = pd.DataFrame(0.0, index=days, columns=U)
                    if filt == "tlt":
                        alt["TLT"] = 1.0
                    tgt = pd.DataFrame(np.where(off.values[:, None], alt.values, tgt.values), index=days, columns=U)
                    # trend filter checked daily; rotation only on rebalance days
                    state = trend_on.astype(int)
                    flip = state != state.shift(1)
                    do = reb | flip
                else:
                    do = reb
                tgt = tgt.where(do, np.nan)
                # force entry at the close before the first dev day
                tgt.loc[:FIRST] = np.nan
                tgt.loc[FIRST] = pick.loc[FIRST] if filt == "none" or trend_on.loc[FIRST] else alt.loc[FIRST]
                ret, ex, od = simulate(tgt)
                record("1_rotation", f"rot_L{L}_top{N}_{fq}_{filt}", ret, ex, od)
                n1 += 1

# ---------------------------------------------------------------- test 2: volatility-managed SPY / QQQ
n2 = 0
ts_ratio = (V["^VIX"] / V["^VIX3M"]).shift(1)  # known at previous close
for t in ["SPY", "QQQ"]:
    specs = []
    for win in [20, 60]:
        rv = R[t].rolling(win).std().shift(1) * np.sqrt(252)  # through t-1
        for tv in [0.10, 0.15]:
            specs.append((f"invvol_w{win}_tv{int(tv*100)}", (tv / rv).clip(upper=1.0)))
    for thr in [0.9, 1.0]:
        for lo in [0.0, 0.5]:
            specs.append((f"vixts_thr{thr}_lo{lo}", pd.Series(np.where(ts_ratio < thr, 1.0, lo), days)))
    rv20 = R[t].rolling(20).std().shift(1) * np.sqrt(252)
    for tv in [0.10, 0.15]:
        specs.append((f"combo_w20_tv{int(tv*100)}_thr1.0",
                      (tv / rv20).clip(upper=1.0) * pd.Series(np.where(ts_ratio < 1.0, 1.0, 0.0), days)))
    for nm, e in specs:
        e = e.fillna(1.0)
        for band in [0.05, 0.20]:
            # no-trade band: rebalance only when target differs from current by more than band
            ev = e.values
            cur = 0.0
            tg = np.full(len(days), np.nan)
            for i in range(len(days)):
                if abs(ev[i] - cur) > band or (ev[i] in (0.0, 1.0) and ev[i] != cur):
                    tg[i] = ev[i]
                    cur = ev[i]
            # (current weight drift is small for a single asset; band is on the target)
            tgt = pd.DataFrame({t: tg}, index=days)
            ret, ex, od = simulate(tgt)
            record("2_volmanaged", f"{t}_{nm}_band{band}", ret, ex, od)
            n2 += 1

# ---------------------------------------------------------------- test 3: leveraged ETFs
n3 = 0
for lev, und in [("UPRO", "SPY"), ("TQQQ", "QQQ")]:
    for ma in [100, 200]:
        m = sma_snap(und, ma)
        for off in ["cash", "TLT"]:
            for buf in [0.0, 0.02]:
                # hysteresis: enter above ma*(1+buf), exit below ma*(1-buf)
                up = (S[und] > m * (1 + buf)).values
                dn = (S[und] < m * (1 - buf)).values
                st = np.zeros(len(days))
                s = 0
                for i in range(len(days)):
                    if up[i]:
                        s = 1
                    elif dn[i]:
                        s = 0
                    st[i] = s
                st = pd.Series(st, days)
                tgt = pd.DataFrame(0.0, index=days, columns=[lev, "TLT"])
                tgt[lev] = st
                if off == "TLT":
                    tgt["TLT"] = 1 - st
                ch = st != st.shift(1)
                tgt = tgt.where(ch, np.nan)
                tgt.loc[:FIRST] = np.nan
                tgt.loc[FIRST] = [st.loc[FIRST], (1 - st.loc[FIRST]) if off == "TLT" else 0.0]
                ret, ex, od = simulate(tgt)
                record("3_leveraged", f"{lev}_ma{ma}_{off}_buf{buf}", ret, ex, od)
                n3 += 1
# overnight-only vs full day (full day = buy and hold, in bench); intraday for completeness
for t in ["UPRO", "SPXL", "TQQQ", "SOXL", "TNA"]:
    night = (O[t] / C[t].shift(1) - 1).fillna(0.0) - 2 * COST[t]
    one = pd.Series(1.0, days)
    record("3_leveraged", f"{t}_overnight_only", night, one, pd.Series(2.0, days))
    intra = (C[t] / O[t] - 1).fillna(0.0) - 2 * COST[t]
    record("3_leveraged", f"{t}_intraday_only", intra, one, pd.Series(2.0, days))
    n3 += 2
for lev, und in [("UPRO", "SPY"), ("TQQQ", "QQQ")]:
    on = (S[und] > sma_snap(und, 200)).shift(1, fill_value=False).astype(bool)  # decided at close t-1
    night = ((O[lev] / C[lev].shift(1) - 1).fillna(0.0) - 2 * COST[lev]).where(on, 0.0)
    record("3_leveraged", f"{lev}_overnight_only_ma200", night, on.astype(float), on.astype(float) * 2)
    n3 += 1

# ---------------------------------------------------------------- test 4: short-horizon mean reversion
n4 = 0


def rsi2_snap(t):
    """Wilder RSI(2) with today's close replaced by the 15:45 price."""
    d = C[t].diff()
    g, l_ = d.clip(lower=0), (-d).clip(lower=0)
    ag = g.ewm(alpha=0.5, adjust=False).mean().shift(1)
    al = l_.ewm(alpha=0.5, adjust=False).mean().shift(1)
    dd = S[t] - C[t].shift(1)
    ag2 = 0.5 * ag + 0.5 * dd.clip(lower=0)
    al2 = 0.5 * al + 0.5 * (-dd).clip(lower=0)
    return 100 - 100 / (1 + ag2 / al2.replace(0, 1e-12))


for t in ["SPY", "QQQ"]:
    prev = C[t].shift(1)
    down = (S[t] < prev)
    # consecutive down closes through yesterday, plus today if the 15:45 price is below yesterday's close
    dn_close = (C[t] < C[t].shift(1))
    past = dn_close.shift(1, fill_value=False).astype(bool)
    streak_past = past.astype(int).groupby((~past).cumsum()).cumsum()
    streak = np.where(down, streak_past + 1, 0)
    streak = pd.Series(streak, days)
    rsi = rsi2_snap(t)
    ma5 = sma_snap(t, 5)
    trend = S[t] > sma_snap(t, 200)
    entries = {"dn2": streak >= 2, "dn3": streak >= 3, "rsi5": rsi < 5, "rsi10": rsi < 10}
    for en, ent in entries.items():
        for ex in ["upclose", "ma5"]:
            if ex == "upclose":
                exit_ = (S[t] > prev) if en.startswith("dn") else (rsi > 70)
            else:
                exit_ = S[t] > ma5
            for tf in [False, True]:
                e_ = (ent & trend) if tf else ent
                ev, xv = e_.values, exit_.values
                pos = np.zeros(len(days))
                p = 0
                for i in range(len(days)):
                    if p == 0 and ev[i]:
                        p = 1
                    elif p == 1 and xv[i]:
                        p = 0
                    pos[i] = p
                pos = pd.Series(pos, days)
                tgt = pd.DataFrame({t: pos.where(pos != pos.shift(1), np.nan)}, index=days)
                ret, exs, od = simulate(tgt)
                record("4_meanrev", f"{t}_{en}_{ex}_{'trend' if tf else 'notrend'}", ret, exs, od)
                n4 += 1

# ---------------------------------------------------------------- output
df = pd.DataFrame(ROWS)
df.to_csv(f"{RES}/study27_variants.csv", index=False)
NV = n1 + n2 + n3 + n4
TRIALS = NV + 960
print("variants:", n1, n2, n3, n4, "total", NV, "trials", TRIALS)

spy = SER[("bench", "SPY_buyhold")]


def boot_diff(r, b, a, e, n=2000, block=20, seed=0):
    """Share of block-bootstrap resamples where Sharpe(strategy) > Sharpe(SPY) over [a, e]."""
    x = pd.concat([r.loc[a:e], b.loc[a:e]], axis=1).values
    T = len(x)
    rng = np.random.default_rng(seed)
    nb = T // block + 1
    wins = 0
    for _ in range(n):
        st = rng.integers(0, T - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:T]
        y = x[idx]
        s = y.mean(0) / y.std(0)
        wins += s[0] > s[1]
    return wins / n


best = []
for test in ["1_rotation", "2_volmanaged", "3_leveraged", "4_meanrev"]:
    d = df[(df.test == test) & (df.period == "dev")].sort_values("sharpe", ascending=False)
    v = d.iloc[0].variant
    r = SER[(test, v)].loc["2020-06-04":]
    sr = r.mean() / r.std()
    row = dict(test=test, variant=v, n_variants_in_test=len(d), trials=TRIALS,
               dsr_full=deflated_sharpe(sr, TRIALS, len(r), skew(r), kurtosis(r, fisher=False)),
               p_beats_spy_sharpe_dev=boot_diff(SER[(test, v)], spy, "2020-06-04", "2023-12-31"),
               p_beats_spy_sharpe_2024_26=boot_diff(SER[(test, v)], spy, "2024-01-01", "2026-09-30"))
    for per in ["dev", "val", "hold", "full"]:
        x = df[(df.test == test) & (df.variant == v) & (df.period == per)].iloc[0]
        for k in ["ann_ret", "sharpe", "maxdd"]:
            row[f"{per}_{k}"] = x[k]
    # share of variants in this test beating SPY Sharpe per period
    for per in ["dev", "val", "hold"]:
        spy_sr = df[(df.variant == "SPY_buyhold") & (df.period == per)].sharpe.iloc[0]
        row[f"share_beat_spy_{per}"] = (df[(df.test == test) & (df.period == per)].sharpe > spy_sr).mean()
    best.append(row)
# overall best by full-sample Sharpe (most optimistic) for the DSR
d = df[(df.test != "bench") & (df.period == "full")].sort_values("sharpe", ascending=False).iloc[0]
r = SER[(d.test, d.variant)].loc["2020-06-04":]
best.append(dict(test="overall_best_full_sharpe", variant=d.variant, trials=TRIALS,
                 dsr_full=deflated_sharpe(r.mean() / r.std(), TRIALS, len(r), skew(r), kurtosis(r, fisher=False)),
                 full_sharpe=d.sharpe))
B = pd.DataFrame(best)
B.to_csv(f"{RES}/study27_best.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
print(B.T)
