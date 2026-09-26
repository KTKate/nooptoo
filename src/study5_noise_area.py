"""Study 5: intraday momentum with noise-area bands on index ETFs (Zarattini, Aziz & Barbon 2024,
"Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)").

Rule as published (no parameter tuning here):
  sigma(tod) = mean over the previous 14 days of |close(tod) / open - 1|
  UB = max(open_t, close_{t-1}) * (1 + sigma(tod)),  LB = min(open_t, close_{t-1}) * (1 - sigma(tod))
  Every 30 minutes from 10:00 to 15:30: price > UB -> long, price < LB -> short.
  Trailing stop (checked at the same 30-minute marks): long exits below max(UB, session VWAP),
  short exits above min(LB, session VWAP). Everything is flat at the close.
Execution: the decision uses the close of the minute bar that ends at the mark (e.g. 9:59 bar for 10:00);
the fill is the close of the next minute bar (one minute of delay) plus cost. Final exit at the 15:59 bar close.
Costs per side: SPY/QQQ/IWM/DIA 0.5 bp half-spread + 1 bp slippage + 0.3 bp fees = 1.8 bp.
Sizing variants: 1x (cash account; short signals either skipped or implemented with an inverse ETF),
and the paper's volatility sizing  lev = min(cap, 2% / 14-day daily vol).
Periods: dev 2019-07..2023-12, val 2024-01..2025-06, oos 2025-07..2026-09.
Output: results/study5_noise_area.csv, results/study5_daily_<ticker>.parquet
"""
import numpy as np
import pandas as pd
import alpaca_data as A
from core import ann_stats, RES

COST = 1.8e-4
MARKS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]  # 10:00 .. 15:30


def daily_frames(t):
    d = A.read("m1", tickers=[t])
    d = d[(d.ts.dt.time >= pd.Timestamp("09:30").time()) & (d.ts.dt.time < pd.Timestamp("16:00").time())]
    d["date"] = d.ts.dt.normalize()
    d["mm"] = (d.ts.dt.hour - 9) * 60 + d.ts.dt.minute - 30        # 0 .. 389
    close = d.pivot_table(index="date", columns="mm", values="c", aggfunc="last").reindex(columns=range(390)).ffill(axis=1)
    opn = d[d.mm == 0].set_index("date").o.reindex(close.index)
    opn = opn.fillna(close[0])
    pv = (d.vwap.astype(float) * d.v).groupby([d.date, d.mm]).sum().unstack().reindex(columns=range(390)).fillna(0)
    vv = d.groupby([d.date, d.mm]).v.sum().unstack().reindex(columns=range(390)).fillna(0)
    vwap = pv.cumsum(axis=1) / vv.cumsum(axis=1).replace(0, np.nan)
    full = close.notna().sum(axis=1) > 380                              # drop half days / broken days
    return close[full], opn[full], vwap[full]


def simulate(close, opn, vwap, divadj=None):
    days = close.index
    prevc = close[389].shift(1)
    if divadj is not None:
        prevc = prevc * divadj.reindex(days).fillna(1.0)
    move = (close.div(opn, axis=0) - 1).abs()
    sigma = move.rolling(14, min_periods=14).mean().shift(1)             # previous 14 days only
    hi = np.maximum(opn, prevc)
    lo = np.minimum(opn, prevc)
    marks = [int((pd.Timestamp(m) - pd.Timestamp("09:30")).seconds / 60) for m in MARKS]
    out = []
    for d in days:
        if np.isnan(sigma.loc[d, 60]) or np.isnan(prevc.loc[d]):
            continue
        px, vw, sg = close.loc[d].values, vwap.loc[d].values, sigma.loc[d].values
        pos, entry, trades = 0, np.nan, 0
        pnl = {1: 0.0, -1: 0.0}
        for m in marks:
            p = px[m - 1]                     # close of the bar ending at the mark
            ub, lb = hi.loc[d] * (1 + sg[m - 1]), lo.loc[d] * (1 - sg[m - 1])
            want = pos
            if pos == 1 and p < max(ub, vw[m - 1]):
                want = 0
            if pos == -1 and p > min(lb, vw[m - 1]):
                want = 0
            if p > ub:
                want = 1
            elif p < lb:
                want = -1
            if want != pos:
                fill = px[m]                  # one minute later
                if pos != 0:
                    pnl[pos] += pos * (fill / entry - 1) - COST
                    trades += 1
                if want != 0:
                    entry = fill
                    pnl[want] -= COST
                pos = want
        if pos != 0:
            pnl[pos] += pos * (px[389] / entry - 1) - COST
            trades += 1
        out.append((d, pnl[1] + pnl[-1], pnl[1], trades))
    # ret_long: the short signals are ignored (flat instead); this is the cash-account version
    return pd.DataFrame(out, columns=["date", "ret1x", "ret_long", "trades"]).set_index("date")


if __name__ == "__main__":
    rows = []
    for t in ["SPY", "QQQ", "IWM", "DIA"]:
        close, opn, vwap = daily_frames(t)
        r = simulate(close, opn, vwap)
        dret = close[389] / close[389].shift(1) - 1
        vol14 = dret.rolling(14).std().shift(1)
        for cap in [1, 2, 4]:
            lev = np.minimum(cap, 0.02 / vol14).reindex(r.index)
            r[f"vol_cap{cap}"] = r.ret1x * lev
        r["bh"] = dret.reindex(r.index)
        r.to_parquet(f"{RES}/study5_daily_{t}.parquet")
        for col in ["ret1x", "ret_long", "vol_cap1", "vol_cap2", "vol_cap4", "bh"]:
            for per, a, b in [("dev", "2019-07", "2023-12"), ("val", "2024-01", "2025-06"),
                              ("oos", "2025-07", "2026-09"), ("post_pub", "2024-03", "2026-09")]:
                st = ann_stats(r[col].loc[a:b])
                rows.append(dict(ticker=t, variant=col, period=per, **{k: st[k] for k in
                                                                       ["n", "ann_ret", "ann_vol", "sharpe", "maxdd", "hit"]},
                                 trades_day=r.trades.loc[a:b].mean()))
        print(t, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study5_noise_area.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["ticker", "variant"], columns="period", values=["sharpe", "ann_ret", "maxdd"]).round(2))
