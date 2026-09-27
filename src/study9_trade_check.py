"""Study 9b: are the large overnight winners of the small-cap rules real and fillable?

Study 9 showed that the small-cap overnight rules lose their edge without the best 1% of trades (mean trade
~38 bp, median ~5 bp). This script re-prices trades with official auction prints from Alpaca SIP trades:
close_off (official closing cross of day t) and open_off (official opening cross of t+1), and compares them
with the Yahoo close / open used in the backtest. Sample: all trades in the top 2% and bottom 1% by Yahoo
return, plus a random 300 of the rest, for both rules.
Output: results/study9_trade_check.csv (one row per trade), summary printed.
"""
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, RES
from study8_exec_retest import snap_panels
import bt

A.RL = A.RateLimiter(150)
P = load_panel()
cols = stock_cols(P)
days = P["c"].loc["2024-01-02":].index[:-1]
rs = lambda x: x.reindex(index=days, columns=cols)
raw = rs(P["rawc"][cols])
yo_raw = rs(P["o"][cols] / (P["c"][cols] / P["rawc"][cols]))
Rn = rs(P["o"][cols].shift(-1) / P["c"][cols] - 1)
adv = rs(P["dv"][cols].rolling(20, min_periods=10).median().shift(1))
px = raw.shift(1)
S = snap_panels("base")
p1545 = rs(S["c15:40"])
E = (px > 2) & (adv > 1e6) & (adv <= 5e6) & p1545.notna()
sig = {"s_intraday_loser": -(p1545 / yo_raw - 1), "s_day_loser": -(p1545 / px - 1)}
nxt = pd.Series(P["c"].index[1:], index=P["c"].index[:-1])
rows = []
for rule, s in sig.items():
    W = bt.select_topk(s, E & s.notna(), 10)
    t = Rn.where(W > 0).stack().rename("r_yahoo").reset_index()
    t.columns = ["date", "ticker", "r_yahoo"]
    t["rule"] = rule
    rows.append(t)
T = pd.concat(rows, ignore_index=True)
q_hi, q_lo = T.r_yahoo.quantile(0.98), T.r_yahoo.quantile(0.01)
rng = np.random.default_rng(3)
mid = T[(T.r_yahoo < q_hi) & (T.r_yahoo > q_lo)]
T["bucket"] = np.where(T.r_yahoo >= q_hi, "top2pct", np.where(T.r_yahoo <= q_lo, "bottom1pct", "rest"))
T["sampled"] = T.bucket != "rest"
T.loc[rng.choice(mid.index, 300, replace=False), "sampled"] = True
X = T[T.sampled].copy()
X["next"] = X.date.map(nxt)
X["close_y"] = [float(raw.at[d, k]) for d, k in zip(X.date, X.ticker)]
# closing crosses: fetched directly (the auction cache holds many pairs looked up for the open only)
from concurrent.futures import ThreadPoolExecutor
pairs = list(dict.fromkeys(zip(X.ticker, X.date)))


def close_cross(p):
    try:
        return A._official(p[0], p[1].strftime("%Y-%m-%d"), "close")[0]
    except RuntimeError:
        return np.nan


with ThreadPoolExecutor(8) as ex:
    cp = list(ex.map(close_cross, pairs))
cl = pd.DataFrame({"ticker": [p[0] for p in pairs], "date": [p[1] for p in pairs], "close_off": cp})
print("closing crosses found", cl.close_off.notna().mean(), flush=True)
op = A.auction_prices(zip(X.ticker, X.next))[["ticker", "date", "open_off", "first_trade"]]
X = X.merge(cl, on=["ticker", "date"], how="left")
X = X.merge(op.rename(columns={"date": "next"}), on=["ticker", "next"], how="left")
# Alpaca trade prices are raw (not split adjusted); the Yahoo raw close carries later splits. Compare returns.
X["r_off"] = X.open_off / X.close_off - 1
X["r_first"] = X.first_trade / X.close_off - 1
X.to_csv(f"{RES}/study9_trade_check.csv", index=False)
X["diff_bps"] = 1e4 * (X.r_off - X.r_yahoo)
g = X.groupby(["rule", "bucket"])
print(pd.DataFrame({"n": g.size(), "found": g.r_off.apply(lambda x: x.notna().mean()),
                    "r_yahoo_bps": 1e4 * g.r_yahoo.mean(), "r_off_bps": 1e4 * g.r_off.mean(),
                    "median_diff_bps": g.diff_bps.median(),
                    "share_diff_gt_100bp": g.diff_bps.apply(lambda x: (x.abs() > 100).mean())}).round(2))
