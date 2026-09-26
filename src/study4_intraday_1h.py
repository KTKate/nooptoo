"""Study 4: intraday strategies on 60-minute bars (Yahoo, last ~730 days: 2024-09 .. 2026-09).

Yahoo 60m bars are stamped 9:30, 10:30, ..., 14:30, 15:30 (the last bar is the
final 30 minutes). Per ticker-day we build:
  pc   previous regular-session close
  o    open (9:30), p1030 = close of the 9:30 bar (price at 10:30),
  p1530 = open of the 15:30 bar, cl = close of the day, vol1 = first-hour volume.
Tests
  A  ETF intraday momentum (Gao-Han-Li-Zhou 2018): sign(p1030/pc - 1) -> last 30 minutes.
     Also sign(p1530/pc -1) (whole-day-so-far) -> last 30 minutes.
  B  ETF trend day: after first hour, go with the first-hour direction until the close.
  C  Stocks in play: first-hour relative volume (vs own 20-day avg first-hour volume)
     top-k with positive first-hour return and gap; buy 10:30, sell at close.
  D  Cross-sectional first-hour momentum/reversal 10:30 -> close.
  E  Cross-sectional last-half-hour: day-so-far return rank -> 15:30..16:00.
First half of the sample (to 2025-09-30) is the development period, the rest is the test.
Costs: per-side = daily half-spread estimate + 2 bp slippage (+ fees); ETFs 1 bp + 1 bp.
Output: results/study4_intraday.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, half_spread_bps, cost_bps, ann_stats, RES, DATA

SPLIT = "2025-09-30"
import store
X = store.read("intra60")
X["date"] = X.ts.dt.tz_localize(None).dt.normalize()
X["hm"] = X.ts.dt.strftime("%H:%M")
X = X[X.hm.isin(["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"])]


def piv(col, hm):
    s = X[X.hm == hm]
    return s.pivot_table(index="date", columns="ticker", values=col, aggfunc="last")


o = piv("o", "09:30")
p1030 = piv("c", "09:30")
vol1 = piv("v", "09:30")
p1530 = piv("o", "15:30")
cl = piv("c", "15:30")
days = o.index
P = load_panel()
tick = [t for t in o.columns if t in P["c"].columns]
o, p1030, vol1, p1530, cl = (x[tick] for x in (o, p1030, vol1, p1530, cl))
# previous close from the daily panel (unadjusted close; intraday bars are unadjusted)
rawc = P["rawc"].reindex(columns=tick)
pc = rawc.shift(1).reindex(days)
hs = half_spread_bps(P).reindex(columns=tick).reindex(days)
cs = cost_bps(hs) / 1e4
etf_cost = 2.0 / 1e4
rows = []


def rec(name, r, extra=None):
    for per, sub in [("dev", r.loc[:SPLIT]), ("test", r.loc[SPLIT:].iloc[1:])]:
        st = ann_stats(sub)
        d = dict(strat=name, period=per, **{k: st[k] for k in ["n", "ann_ret", "sharpe", "tstat", "maxdd", "hit"]})
        d["bps_day"] = 1e4 * sub.mean()
        if extra:
            d.update(extra)
        rows.append(d)


# ---- A/B ETF momentum
for t in ["SPY", "QQQ", "IWM", "DIA", "TQQQ", "SOXL", "SMH", "XLK"]:
    if t not in tick:
        continue
    first = p1030[t] / pc[t] - 1
    sofar = p1530[t] / pc[t] - 1
    last30 = cl[t] / p1530[t] - 1
    rest = cl[t] / p1030[t] - 1
    rec(f"A_{t}_first_hr_sign->last30_LS", np.sign(first) * last30 - 2 * etf_cost)
    rec(f"A_{t}_first_hr_pos->last30_long", (first > 0) * (last30 - 2 * etf_cost))
    rec(f"A_{t}_sofar_sign->last30_LS", np.sign(sofar) * last30 - 2 * etf_cost)
    rec(f"A_{t}_sofar_pos->last30_long", (sofar > 0) * (last30 - 2 * etf_cost))
    rec(f"A_{t}_last30_always_long", last30 - 2 * etf_cost)
    rec(f"B_{t}_first_hr_sign->rest_LS", np.sign(first) * rest - 2 * etf_cost)
    big = first.abs() > first.abs().rolling(20, min_periods=10).mean().shift(1)
    rec(f"B_{t}_big_first_hr_sign->rest_LS", big * (np.sign(first) * rest - 2 * etf_cost))

# ---- stock universe for cross-sectional tests
stocks = [t for t in tick if t not in {"SPY", "QQQ", "IWM", "DIA", "MDY", "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL",
                                          "UPRO", "TNA", "TZA", "SMH", "XBI", "KRE", "ARKK", "TLT", "HYG", "GLD",
                                          "USO", "UUP"} and not t.startswith("XL")]
g = (o / pc - 1)[stocks]
f1 = (p1030 / o - 1)[stocks]
rest = (cl / p1030 - 1)[stocks]
sofar = (p1530 / o - 1)[stocks]
last30 = (cl / p1530 - 1)[stocks]
rv = (vol1 / vol1.rolling(20, min_periods=10).mean().shift(1))[stocks]
C = cs[stocks]
elig = (pc[stocks] > 5) & rest.notna()


def topk(sig, mask, k):
    s = sig.where(mask)
    rk = s.rank(axis=1, ascending=False, method="first")
    W = (rk <= k).astype(float)
    return W.div(W.sum(1).replace(0, np.nan), axis=0).fillna(0)


def pnl(W, R, Cm):
    return (W * R.fillna(0)).sum(1) - (2 * W * Cm.fillna(0.01)).sum(1)


for k in [5, 10, 20]:
    # C stocks in play, long side
    m = elig & (f1 > 0) & (g > 0.02) & (rv > 2)
    rec(f"C_inplay_long_k{k}", pnl(topk(rv, m, k), rest, C), {"avg_n": topk(rv, m, k).gt(0).sum(1).mean()})
    m2 = elig & (f1 > 0) & (rv > 2)
    rec(f"C_inplay_nogap_long_k{k}", pnl(topk(rv, m2, k), rest, C))
    m3 = elig & (f1 < 0) & (g < -0.02) & (rv > 2)
    rec(f"C_inplay_short_k{k}", -(topk(rv, m3, k) * rest.fillna(0)).sum(1) - (2 * topk(rv, m3, k) * C.fillna(0.01)).sum(1))
    # D first-hour momentum / reversal
    rec(f"D_firsthr_winners_long_k{k}", pnl(topk(f1, elig, k), rest, C))
    rec(f"D_firsthr_losers_long_k{k}", pnl(topk(-f1, elig, k), rest, C))
    # E last half hour
    e2 = elig & last30.notna()
    rec(f"E_sofar_winners_last30_k{k}", pnl(topk(sofar, e2, k), last30, C))
    rec(f"E_sofar_losers_last30_k{k}", pnl(topk(-sofar, e2, k), last30, C))

# gross decile spreads (signal quality, no costs)
for nm, sig, R in [("D_firsthr", f1, rest), ("E_sofar", sofar, last30), ("C_relvol", rv * np.sign(f1), rest)]:
    pct = sig.where(elig).rank(axis=1, pct=True)
    ls = R.where(pct > 0.9).mean(1) - R.where(pct < 0.1).mean(1)
    rec(f"{nm}_decile_LS_gross", ls)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study4_intraday.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(df.pivot_table(index="strat", columns="period", values=["sharpe", "bps_day", "ann_ret"]).round(2))
