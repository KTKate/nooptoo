"""Study 2: earnings-event strategies (intraweek holding periods).

Events from the Nasdaq earnings calendar (report date D, EPS surprise %).
Announcement timing (before open vs after close) is mostly missing historically,
so every rule waits until the close of D+1, when both possible reaction gaps
are public. Rules:
  pead_surprise : buy at close D+1 if EPS surprise > 0 and 2-day reaction
                  (c[D+1]/c[D-1]-1) is in the top of the day's events; hold h days.
  ear_drift     : rank by 2-day earnings announcement return (EAR); long top.
  pre_run       : buy at close D-6, sell at close D-1 (never holds through the report).
  hold_through  : buy close D-2, sell close D+1 (earnings announcement premium).
Portfolio: each event gets a fixed slice of capital (1/K) and positions are
overlapping; daily P&L is the sum over live positions. K sets max positions.
Output: results/study2_earnings.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, split_stats, ann_stats, RES, DATA

P = load_panel()
cols = stock_cols(P)
c, rawc, dv = P["c"][cols], P["rawc"][cols], P["dv"][cols]
# every entry and exit happens at a close: market-on-close orders (closing auction). A variant charging the
# continuous-market cost at 15:45 is reported too (COSTKIND=close).
import os
cs = exec_cost_bps(P, os.environ.get("COSTKIND", "auction"))[cols]
days = c.index
adv = dv.rolling(20, min_periods=10).median()
ret = c.pct_change()

import store
E = store.read("earnings")
E = E[E.symbol.isin(cols)].copy()
E["date"] = pd.to_datetime(E["date"])
# map report date to trading-day index (next trading day if report on a holiday)
pos = days.searchsorted(E["date"].values)
E["di"] = pos
E = E[(E.di > 70) & (E.di < len(days) - 25)]
E = E.drop_duplicates(["symbol", "di"])
ci = {t: i for i, t in enumerate(cols)}
E["ti"] = E.symbol.map(ci)
C = c.values
AD = adv.values
PX = rawc.values
CS = cs.values


def val(M, di, ti):
    return M[di, ti]


E["ear"] = C[E.di + 1, E.ti] / C[E.di - 1, E.ti] - 1
E["adv"] = AD[E.di - 1, E.ti]
E["px"] = PX[E.di - 1, E.ti]
E["pre20"] = C[E.di - 1, E.ti] / C[E.di - 21, E.ti] - 1
E = E[np.isfinite(E.ear) & (E.px > 5) & (E.adv > 5e6)]
print("events", len(E), E.date.min(), E.date.max())


def simulate(ev, entry_off, exit_off, K, label):
    """ev: events with di/ti. entry at close of di+entry_off, exit at close di+exit_off.
    Each position weight 1/K; daily return = sum of weighted daily returns of live positions
    (capped: if more than K live, scale down to sum weight 1)."""
    T, N = C.shape
    W = np.zeros((T, N), dtype=np.float32)
    for di, ti in zip(ev.di.values, ev.ti.values):
        a, b = di + entry_off, di + exit_off
        if b >= T or a < 1:
            continue
        W[a + 1:b + 1, ti] += 1.0 / K   # held over returns of days a+1..b
    tot = W.sum(1, keepdims=True)
    scale = np.where(tot > 1, 1 / np.maximum(tot, 1e-9), 1.0)
    W = W * scale
    R = np.nan_to_num(ret.values)
    gross = (W * R).sum(1)
    # costs: weight changes at close (entry/exit) at per-side cost of that day
    Wprev = np.vstack([np.zeros((1, N)), W[:-1]])
    # position held over day t+1 is decided at close t -> trade at close t = W[t+1]-W[t]
    dW = np.abs(np.vstack([W[1:], np.zeros((1, N))]) - W)
    cost = (dW * np.nan_to_num(CS, nan=100.0) / 1e4).sum(1)
    cost = np.concatenate([[0], cost[:-1]])  # charge on the day after decision (same P&L day)
    net = pd.Series(gross - cost, index=days)
    return net, pd.Series(W.sum(1), index=days)


E = E.sort_values(["di", "symbol"]).reset_index(drop=True)


def trailing_q(x, q, n=1000):
    """quantile of the previous n events (strictly earlier trading days), no look-ahead."""
    return x.shift(1).rolling(n, min_periods=200).quantile(q).groupby(E.di).transform("first")


E["q67"] = trailing_q(E.ear.where(E.surprise > 0), 0.67)
E["q90"] = trailing_q(E.ear, 0.9)
E["q10"] = trailing_q(E.ear, 0.1)

rows = []
out = {}
for K in [10, 20]:
    for h in [5, 10, 20]:
        ev = E[(E.surprise > 0)]
        # EAR above the trailing 67th pct of earlier positive-surprise events (known at D+1 close)
        ev = ev[ev.ear > ev.q67]
        n, expo = simulate(ev, 1, 1 + h, K, "pead")
        out[f"pead_pos_top_ear_K{K}_h{h}"] = (n, expo)
        ev2 = E[E.ear > E.q90]
        n, expo = simulate(ev2, 1, 1 + h, K, "ear")
        out[f"ear_top10pct_K{K}_h{h}"] = (n, expo)
        ev3 = E[E.ear < E.q10]
        n, expo = simulate(ev3, 1, 1 + h, K, "ear_loser")
        out[f"ear_bottom10pct_long_K{K}_h{h}"] = (n, expo)
    n, expo = simulate(E, -6, -1, K * 3, "pre")
    out[f"pre_run_5d_K{K*3}"] = (n, expo)
    n, expo = simulate(E[E.adv > 5e7], -2, 1, K, "through")
    out[f"hold_through_liquid_K{K}"] = (n, expo)

for k, (n, expo) in out.items():
    st = split_stats(n)
    for per in ["dev", "val", "oos"]:
        rows.append(dict(strat=k, period=per, sharpe=st.loc[per, "sharpe"], ann_ret=st.loc[per, "ann_ret"],
                         maxdd=st.loc[per, "maxdd"], avg_expo=expo.loc[{"dev": slice("2020", "2023"),
                         "val": slice("2024", "2025-06"), "oos": slice("2025-07", "2026-09")}[per]].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study2_earnings_{os.environ.get('COSTKIND', 'auction')}.csv", index=False)
pd.set_option("display.width", 250)
print(df.pivot_table(index="strat", columns="period", values=["sharpe", "ann_ret", "avg_expo"]).round(2))
