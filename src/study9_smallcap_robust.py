"""Study 9: robustness of the surviving small-cap overnight rules (mirrors study3_robust.py for the ML ranker).

Rules (final_series.py): at 15:45 rank small caps ($1-5M 20d median dollar volume, price > $2) by
  s_intraday_loser : loss from the open to 15:45
  s_day_loser      : loss from the previous close to 15:45
buy the top 10 in the closing auction, sell in the next opening auction.
Checks:
  cost      auction fraction of the 15:45 quoted half-spread (0.1 .. 1.0 = paying the full half-spread in both
            auctions) x extra bp per side (0 .. 10; the measured Yahoo-open vs opening-cross gap for small-cap
            picks is 7.3 bp on the exit side, results/overnight_auction_check.csv)
  k         top-k neighborhood 3 .. 30
  tier      ADV band and price floor neighborhood
  filter    Alpaca-Yahoo consistency filter in study8_exec_retest.snap_panels: base, month (drop only
            ticker-months), strict, none, and a source-consistent signal that uses only Alpaca prices
  half_year subperiods; regime check 2020-2026 with the (unexecutable) closing-price signal
  tails     result without the best 1% of trades (is the edge a few lottery tickets?)
  hedged    alpha after regressing on the IWM overnight return
  bootstrap / deflated Sharpe
Output: results/study9_smallcap_robust.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, half_spread_panel, ann_stats, deflated_sharpe, block_bootstrap_sharpe, RES
from study8_exec_retest import snap_panels
import bt

N_TRIALS = 250
P = load_panel()
cols = stock_cols(P)
days = P["c"].loc["2024-01-02":].index
rs = lambda x: x.reindex(index=days, columns=cols)
Rn = rs(P["o"][cols].shift(-1) / P["c"][cols] - 1)                       # close -> next open, total return
hs = rs(half_spread_panel(P, "15:45")[cols])
adv = rs(P["dv"][cols].rolling(20, min_periods=10).median().shift(1))
px = rs(P["rawc"][cols].shift(1))
yo = rs(P["o"][cols] / (P["c"][cols] / P["rawc"][cols]))                   # raw-scale Yahoo open
prevc = rs(P["rawc"][cols].shift(1))
LAST = P["c"].index[-1]
iwmN = (P["o"]["IWM"].shift(-1) / P["c"]["IWM"] - 1).reindex(days)
rows = []


def cost_of(frac=0.1, extra=2.5):
    return 0.3 + 1.0 + frac * hs + extra


def add(tag, rule, r, periods=(("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"),
                               ("2024-26", "2024-01", "2026-09")), **kw):
    for per, a, b in periods:
        x = r.net.loc[a:b]
        x = x[x.index < LAST]                                                # the last day has no next open
        st = ann_stats(x)
        rows.append(dict(test=tag, rule=rule, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                         maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[x.index].mean(),
                         cost_bps=1e4 * r.cost.loc[x.index].mean(), names=r.n.loc[x.index].mean(), **kw))


def signals(S):
    p1545 = rs(S["c15:40"])
    return p1545, {"s_intraday_loser": -(p1545 / yo - 1), "s_day_loser": -(p1545 / prevc - 1)}


def tier(p1545, lo=1e6, hi=5e6, pmin=2.0):
    return (px > pmin) & (adv > lo) & (adv <= hi) & p1545.notna()


S0 = snap_panels("base")
p1545, sig = signals(S0)
E0 = tier(p1545)
base = {}
for rule, s in sig.items():
    W = bt.select_topk(s, E0 & s.notna(), 10)
    base[rule] = (W, bt.run(W, Rn, cost_of()))
    add("base", rule, base[rule][1], k=10)
    # cost grid
    for frac in [0.1, 0.25, 0.5, 1.0]:
        for extra in [0.0, 2.5, 5.0, 10.0]:
            add("cost", rule, bt.run(W, Rn, cost_of(frac, extra)), auction_frac=frac, extra_bps=extra)
    # k neighborhood
    for k in [3, 5, 7, 15, 20, 30]:
        add("k", rule, bt.run(bt.select_topk(s, E0 & s.notna(), k), Rn, cost_of()), k=k)
    # tier neighborhood
    for lo, hi, pmin in [(5e5, 2e6, 2), (1e6, 3e6, 2), (3e6, 5e6, 2), (2e6, 1e7, 2), (1e6, 5e6, 1),
                         (1e6, 5e6, 5), (5e5, 1e6, 2)]:
        E = tier(p1545, lo, hi, pmin)
        add("tier", rule, bt.run(bt.select_topk(s, E & s.notna(), 10), Rn, cost_of()),
            adv_lo=lo, adv_hi=hi, pmin=pmin)
    # half years
    r = base[rule][1]
    for (y, h), x in r.net.iloc[:-1].groupby([r.index[:-1].year, (r.index[:-1].month - 1) // 6]):
        st = ann_stats(x)
        rows.append(dict(test="half_year", rule=rule, period=f"{y}H{h + 1}", sharpe=st["sharpe"],
                         ann_ret=st["ann_ret"], maxdd=st["maxdd"], gross_bps=1e4 * r.gross.loc[x.index].mean()))
    # tails: per-trade contributions, drop the best 1% of trades
    tr = (W * Rn).stack()
    tr = tr[W.stack() > 0]
    cut = tr.quantile(0.99)
    Rclip = Rn.where(~((W > 0) & (W * Rn >= cut)), 0.0)
    add("tails_drop_top1pct", rule, bt.run(W, Rclip, cost_of()))
    trade_r = Rn.stack()[(W > 0).stack()]
    rows.append(dict(test="trade_stats", rule=rule, period="2024-26", mean_bps=1e4 * trade_r.mean(),
                     median_bps=1e4 * trade_r.median(), hit=(trade_r > 0).mean(), n_trades=len(trade_r),
                     pct_px_lt5=float((rs(P["rawc"][cols])[W > 0].stack() < 5).mean()),
                     median_adv=float(adv[W > 0].stack().median())))
    # hedged vs IWM overnight
    X = pd.concat([r.net, iwmN], axis=1).dropna()
    b = np.polyfit(X.iloc[:, 1], X.iloc[:, 0], 1)
    res = X.iloc[:, 0] - b[0] * X.iloc[:, 1]
    rows.append(dict(test="hedged_vs_iwm_night", rule=rule, period="2024-26",
                     sharpe=res.mean() / res.std() * np.sqrt(252), beta=b[0], alpha_bps=1e4 * b[1]))
    x = r.net.loc["2024-01":].iloc[:-1].dropna()
    lo, med, hi = block_bootstrap_sharpe(x, block=10, n=2000)
    rows.append(dict(test="bootstrap_ci", rule=rule, period="2024-26", sharpe=med, ci_lo=lo, ci_hi=hi))
    rows.append(dict(test="deflated_sharpe_prob", rule=rule, period="2024-26", n_trials=N_TRIALS,
                     value=deflated_sharpe(x.mean() / x.std(), N_TRIALS, len(x), skew=float(x.skew()),
                                           kurt=float(x.kurt() + 3))))
    print(rule, "done", flush=True)

# consistency-filter variants
for mode in ["month", "strict", "none"]:
    S = snap_panels(mode)
    p, sg = signals(S)
    E = tier(p)
    for rule, s in sg.items():
        W = bt.select_topk(s, E & s.notna(), 10)
        add("filter", rule, bt.run(W, Rn, cost_of()), mode=mode,
            overlap=float(((W > 0) & (base[rule][0] > 0)).sum(1).mean()))
# source-consistent signal: Alpaca prices only (09:30 bar open, previous day's 15:55 bar close), no filter
S = snap_panels("none")
pa = rs(S["c15:40"])
sga = {"s_intraday_loser": -(pa / rs(S["o09:30"]) - 1), "s_day_loser": -(pa / rs(S["c15:55"]).shift(1) - 1)}
E = tier(pa)
for rule, s in sga.items():
    W = bt.select_topk(s, E & s.notna(), 10)
    add("filter", rule, bt.run(W, Rn, cost_of()), mode="alpaca_only",
        overlap=float(((W > 0) & (base[rule][0] > 0)).sum(1).mean()))

# regime check with the closing-price signal (not executable; known only after the close) 2020-2026
alld = P["c"].loc["2020-01-02":].index
ra = lambda x: x.reindex(index=alld, columns=cols)
advA = ra(P["dv"][cols].rolling(20, min_periods=10).median().shift(1))
pxA = ra(P["rawc"][cols].shift(1))
EA = (pxA > 2) & (advA > 1e6) & (advA <= 5e6)
RA = ra(P["o"][cols].shift(-1) / P["c"][cols] - 1)
cA = 0.3 + 1.0 + 0.1 * ra(half_spread_panel(P, "15:45")[cols]) + 2.5
sgc = {"s_intraday_loser": -ra(P["c"][cols] / P["o"][cols] - 1), "s_day_loser": -ra(P["c"][cols] / P["c"][cols].shift(1) - 1)}
for rule, s in sgc.items():
    r = bt.run(bt.select_topk(s, EA & s.notna(), 10), RA, cA)
    add("close_signal_by_year", rule, r, periods=[(str(y), str(y), str(y)) for y in range(2020, 2027)])

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study9_smallcap_robust.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 30)
print(df.round(3).to_string())
