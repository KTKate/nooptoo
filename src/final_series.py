"""Daily net return series of the surviving candidates and the benchmark, 2024-01 .. 2026-09, with the
executable timing (signals at 15:45, market-on-close entry, market-on-open exit, auction costs + 2.5 bp/side).

  ml_overnight_k10      LightGBM overnight ranker, features at 15:45 (study3_timing.py), top 10
  s_intraday_loser      small caps ($1-5M ADV): largest loss from the open to 15:45, top 10
  s_day_loser           small caps: largest loss from the previous close to 15:45, top 10
  m_day_winner          mid caps ($5-50M ADV): largest gain from the previous close to 15:45, top 10
  combo                 50% ml_overnight_k10 + 50% s_intraday_loser (same account, 5 names each side)
  spy_buy_hold          SPY total return
Price filters use the close as it traded (core.traded_close), not Yahoo's split-adjusted close.
SNAP_MODE (environment, default none): Alpaca-Yahoo consistency filter for the 15:45 prices. The earlier
default (base) dropped whole tickers using mismatches seen anywhere in 2024-26; those were mostly distressed
small caps that later did reverse splits, so it removed real losses with hindsight (study 9). none has no
look-ahead. The ML predictions come from study3_timing.py run with the same SNAP_MODE.
Output files carry a _<mode> suffix unless the mode is none.
Output: results/final_series.parquet, results/final_summary.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, block_bootstrap_sharpe, RES
from study8_exec_retest import price_1545
import bt
import os

MODE = os.environ.get("SNAP_MODE", "none")
SUF = "" if MODE == "none" else f"_{MODE}"

P = load_panel()
cols = stock_cols(P)
days = P["c"].loc["2024-01-02":].index
o, c, raw, dv = (P[k][cols].reindex(days) for k in ["o", "c", "rawc", "dv"])
Rn = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex(days)
cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(days)
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex(days)
px = traded_close(P)[cols].shift(1).reindex(days)                     # price filter on the traded price
p1545 = price_1545(MODE).reindex(index=days, columns=cols)       # m5snap + small caps it leaves out
yo = (P["o"][cols] / (P["c"][cols] / P["rawc"][cols])).reindex(days)        # raw-scale open
prevc = P["rawc"][cols].shift(1).reindex(days)
tierS = (px > 2) & (adv > 1e6) & (adv <= 5e6) & p1545.notna()
tierM = (px > 5) & (adv > 5e6) & (adv <= 5e7) & p1545.notna()
out = {}
pred = pd.read_parquet(f"{RES}/study3_pred_night_1545{'' if MODE == 'base' else '_' + MODE}.parquet")["pred"].unstack().reindex(index=days, columns=cols)
W_ml = bt.select_topk(pred, pred.notna(), 10)
out["ml_overnight_k10"] = bt.run(W_ml, Rn, cost)
sig = {"s_intraday_loser": (-(p1545 / yo - 1), tierS), "s_day_loser": (-(p1545 / prevc - 1), tierS),
       "m_day_winner": (p1545 / prevc - 1, tierM)}
W = {}
for k, (s, E) in sig.items():
    W[k] = bt.select_topk(s, E & s.notna(), 10)
    out[k] = bt.run(W[k], Rn, cost)
Wc = 0.5 * bt.select_topk(pred, pred.notna(), 5) + 0.5 * bt.select_topk(sig["s_intraday_loser"][0],
                                                                         tierS & sig["s_intraday_loser"][0].notna(), 5)
out["combo"] = bt.run(Wc, Rn, cost)
net = pd.DataFrame({k: v.net for k, v in out.items()})
net["spy_buy_hold"] = (P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1).reindex(days)
net = net.iloc[:-1]                      # the last day has no next open yet
net.to_parquet(f"{RES}/final_series{SUF}.parquet")
rows = []
for k in net.columns:
    for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]:
        x = net[k].loc[a:b]
        st = ann_stats(x)
        d = dict(strategy=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], ann_vol=st["ann_vol"],
                 maxdd=st["maxdd"], hit=st["hit"], total=st["total"])
        if k in out:
            d["gross_bps"] = 1e4 * out[k].gross.loc[a:b].mean()
            d["cost_bps"] = 1e4 * out[k].cost.loc[a:b].mean()
        if per == "2024-26":
            lo, med, hi = block_bootstrap_sharpe(x, block=10, n=1000)
            d.update(ci_lo=lo, ci_hi=hi, dsr_250=deflated_sharpe(x.mean() / x.std(), 250, len(x.dropna()),
                                                                 skew=float(x.skew()), kurt=float(x.kurt() + 3)))
        rows.append(d)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/final_summary{SUF}.csv", index=False)
print(df.round(3).to_string())
print(net.corr().round(2))
