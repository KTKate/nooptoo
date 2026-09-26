"""Study 6: 5-minute opening-range breakout on "stocks in play" (Zarattini & Aziz 2023,
"Can Day Trading Really Be Profitable?"), with 5-minute Alpaca SIP bars, 2024-01 .. 2026-09.

Published rule (parameters not tuned here):
  universe at t: price > $5, 14-day average volume > 1M shares, ATR14 > $0.50 (all known at t-1)
  relative volume RV = volume of the 9:30-9:35 bar / mean of the same bar over the previous 14 days
  stocks in play: top 20 by RV with RV >= 1
  direction from the first 5-minute candle: close > open -> long stop at its high,
  close < open -> short stop at its low, doji -> no trade
  stop loss 10% of ATR14 from the entry level; exit at the close if not stopped
  sizing: risk 1% of equity per trade, gross leverage capped at 4x
Execution assumptions (conservative on 5-minute bars):
  entry fills at max(level, bar open) for longs (min for shorts), i.e. gaps through the level fill worse;
  if the stop is inside the entry bar's range the trade is assumed stopped in that same bar;
  stop fills at min(stop, bar open) for longs; exit at the close of the 15:55 bar.
  cost per side = quoted half-spread model at the time of day + 2 bp (stop orders are market orders) + fees.
Variants: long+short with paper sizing (needs a margin account >= $25k: PDT), long-only cash account
with equal 1/k capital per candidate (no leverage), and k in {5, 10, 20}.
Output: results/study6_orb_trades.parquet, results/study6_orb.csv
"""
import os
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, ann_stats, RES, half_spread_model

OUT_TR = f"{RES}/study6_orb_trades.parquet"


def first_bars():
    """Per stock-day: first 5-minute bar (9:30) o h l c v, from the m5 store (cached)."""
    fn = os.path.join(A.LOCAL, "m5_first.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    parts = []
    for f in sorted(os.listdir(os.path.join(A.LOCAL, "m5"))):
        d = pd.read_parquet(os.path.join(A.LOCAL, "m5", f))
        d = d[(d.ts.dt.hour == 9) & (d.ts.dt.minute == 30)]
        d["date"] = d.ts.dt.normalize()
        parts.append(d[["date", "ticker", "o", "h", "l", "c", "v"]])
    fb = pd.concat(parts, ignore_index=True)
    fb.to_parquet(fn, index=False)
    return fb


def candidates(k=20):
    P = load_panel()
    fb = first_bars()
    days = P["c"].index
    fb = fb[fb.date.isin(days)]
    v1 = fb.pivot(index="date", columns="ticker", values="v").reindex(days)
    tick = v1.columns
    raw = P["rawc"][tick]
    f = raw / P["c"][tick]
    h, l, c, vol = P["h"][tick] * f, P["l"][tick] * f, raw, P["v"][tick]
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr = tr.rolling(14, min_periods=14).mean().shift(1)
    avgv = vol.rolling(14, min_periods=14).mean().shift(1)
    px = raw.shift(1)
    rv = v1 / v1.rolling(14, min_periods=10).mean().shift(1)
    elig = (px > 5) & (avgv > 1e6) & (atr > 0.5) & (rv >= 1)
    rk = rv.where(elig).rank(axis=1, ascending=False, method="first")
    sel = (rk <= k).stack()
    sel = sel[sel].index
    out = pd.DataFrame(index=sel).reset_index()
    out.columns = ["date", "ticker"]
    out["rank"] = rk.stack().reindex(sel).values
    out["rv"] = rv.stack().reindex(sel).values
    out["atr"] = atr.stack().reindex(sel).values
    out["pc"] = c.shift(1).stack().reindex(sel).values
    out["adv"] = (P["dv"][tick].rolling(20, min_periods=10).median().shift(1)).stack().reindex(sel).values
    out["vol20"] = np.log(P["c"][tick] / P["c"][tick].shift(1)).rolling(20, min_periods=10).std().shift(1)\
        .stack().reindex(sel).values
    return out


def simulate_trades(cand):
    rows = []
    cand = cand.assign(month=cand.date.dt.strftime("%Y-%m"))
    for m, cm in cand.groupby("month"):
        d = A.read("m5", start=m, end=m, tickers=sorted(cm.ticker.unique()))
        d = d[(d.ts.dt.time >= pd.Timestamp("09:30").time()) & (d.ts.dt.time < pd.Timestamp("16:00").time())]
        d["date"] = d.ts.dt.normalize()
        g = {k: v for k, v in d.groupby(["date", "ticker"])}
        for r in cm.itertuples(index=False):
            b = g.get((r.date, r.ticker))
            if b is None or len(b) < 10 or b.ts.iloc[0].time() != pd.Timestamp("09:30").time():
                continue
            o1, h1, l1, c1 = b.o.iloc[0], b.h.iloc[0], b.l.iloc[0], b.c.iloc[0]
            side = 1 if c1 > o1 else (-1 if c1 < o1 else 0)
            if side == 0:
                continue
            lvl = h1 if side == 1 else l1
            stop_dist = 0.1 * r.atr
            O, H, L, C = b.o.values[1:], b.h.values[1:], b.l.values[1:], b.c.values[1:]
            T = b.ts.values[1:]
            entry = exitp = np.nan
            ei = xi = None
            for i in range(len(O)):
                if side == 1 and H[i] >= lvl:
                    entry = max(lvl, O[i]); ei = i; break
                if side == -1 and L[i] <= lvl:
                    entry = min(lvl, O[i]); ei = i; break
            if ei is None:
                rows.append(dict(date=r.date, ticker=r.ticker, side=side, filled=False, rank=r.rank, rv=r.rv))
                continue
            stop = entry - side * stop_dist
            how = "close"
            for i in range(ei, len(O)):
                if side == 1 and L[i] <= stop:
                    exitp = stop if i == ei else min(stop, O[i]); xi = i; how = "stop"; break
                if side == -1 and H[i] >= stop:
                    exitp = stop if i == ei else max(stop, O[i]); xi = i; how = "stop"; break
            if xi is None:
                xi = len(O) - 1
                exitp = C[-1]
            rows.append(dict(date=r.date, ticker=r.ticker, side=side, filled=True, rank=r.rank, rv=r.rv,
                             atr=r.atr, entry=entry, exit=exitp, how=how, t_entry=pd.Timestamp(T[ei]),
                             ret=side * (exitp / entry - 1), risk=stop_dist / entry, adv=r.adv, vol20=r.vol20,
                             pc=r.pc))
        print(m, len(cm), flush=True)
    return pd.DataFrame(rows)


def portfolio(tr, k, long_only, sizing):
    t = tr[tr["rank"] <= k].copy()
    if long_only:
        t = t[t.side == 1]
    f = t[t.filled]
    # per-side cost: spread model at the entry time (opening minutes are wider) and at the close
    hs_in = half_spread_model(f.adv, f.pc, f.vol20, "09:35")
    hs_out_close = half_spread_model(f.adv, f.pc, f.vol20, "15:45")
    hs_out = np.where(f.how == "stop", hs_in * 0 + half_spread_model(f.adv, f.pc, f.vol20, "12:00"), hs_out_close)
    cost = (hs_in + 2.0 + 0.3 + hs_out + 2.0 + 0.3) / 1e4
    f = f.assign(net=f.ret - cost, cost=cost)
    if sizing == "equal":
        w = pd.Series(1.0 / k, index=f.index)
    else:  # paper: 1% risk per trade, total gross leverage capped at 4x
        w = 0.01 / f.risk
        tot = w.groupby(f.date).transform("sum")
        w = w * np.minimum(1.0, 4.0 / tot)
    daily = (w * f.net).groupby(f.date).sum()
    return f, daily


if __name__ == "__main__":
    if os.path.exists(OUT_TR):
        tr = pd.read_parquet(OUT_TR)
    else:
        cand = candidates(20)
        print("candidates", len(cand), cand.date.min(), cand.date.max(), flush=True)
        tr = simulate_trades(cand)
        tr.to_parquet(OUT_TR, index=False)
    P = load_panel()
    days = P["c"].loc["2024-01-01":].index
    spy = (P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1).reindex(days)
    rows = []
    for k in [5, 10, 20]:
        for lo in [False, True]:
            for sz in ["equal", "risk1pct_4x"]:
                f, daily = portfolio(tr, k, lo, sz)
                daily = daily.reindex(days).fillna(0.0)
                for per, a, b in [("2024", "2024-01", "2024-12"), ("val", "2024-01", "2025-06"),
                                  ("oos", "2025-07", "2026-09"), ("all", "2024-01", "2026-09")]:
                    st = ann_stats(daily.loc[a:b])
                    ff = f[(f.date >= a) & (f.date <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
                    rows.append(dict(k=k, long_only=lo, sizing=sz, period=per, sharpe=st["sharpe"],
                                     ann_ret=st["ann_ret"], maxdd=st["maxdd"], trades=len(ff),
                                     gross_bps_trade=1e4 * ff.ret.mean(), net_bps_trade=1e4 * ff.net.mean(),
                                     cost_bps_trade=1e4 * ff.cost.mean(), stop_rate=(ff.how == "stop").mean()))
    for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09"), ("all", "2024-01", "2026-09")]:
        st = ann_stats(spy.loc[a:b])
        rows.append(dict(k=0, long_only=True, sizing="SPY_buy_hold", period=per, sharpe=st["sharpe"],
                         ann_ret=st["ann_ret"], maxdd=st["maxdd"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study6_orb.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
