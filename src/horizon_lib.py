"""Shared tools for multi-day holding studies (39+): forward returns, rebalanced top-k backtests, statistics.

Trades: buy in the closing auction of the rebalance day, sell in the closing auction h trading days later (cost per
side = core.exec_cost_bps(P, "auction") + 2.5 bp, charged on the names that change at each rebalance).
Survivorship: the daily panel only contains tickers that still trade in 2026 (no delisted names), which inflates
multi-week returns of risky stocks. Results are therefore reported as excess over the equal-weight average of the
same tradable universe on the same days (that average carries the same bias), and the universe is restricted to
liquid names (traded close, extended before 2023-12 by TPX, > $5, 20-day median dollar volume > $20M by default).
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
C = P["c"][cols]
TC = traded_close(P)[cols]
# traded close exists from 2023-12 only; earlier, scale the Yahoo close by each ticker's traded/Yahoo ratio on its
# first traded-close day (fixes later splits, not splits during 2020-23)
_raw = P["rawc"][cols].astype("float64")
TPX = TC.fillna(_raw * (TC / _raw.where(_raw > 0)).bfill().iloc[0].fillna(1.0))
ADV = P["dv"][cols].rolling(20).median()
COST = (exec_cost_bps(P, "auction")[cols] + 2.5) / 1e4
SPY = P["c"]["SPY"]


def universe(min_px=5, min_adv=20e6):
    return (TPX > min_px) & (ADV > min_adv) & C.notna()


def fwd(h):
    """Close-to-close return from t to t+h (adjusted)."""
    return C.shift(-h) / C - 1


def rebalance_days(h, start="2022-01-01", end=None, offset=0):
    d = days[(days >= start) & ((days <= end) if end else True)]
    d = d[: len(d) - h] if len(d) > h else d[:0]
    return d[offset::h]


def backtest(score, h, k=20, univ=None, start="2022-01-01", offset=0, largest=True):
    """score: date x ticker. Every h days buy the top k (equal weight) at the close, hold h days.
    Returns DataFrame per rebalance date: gross, cost, net, bench (universe equal-weight), spy, excess (net - bench)."""
    univ = universe() if univ is None else univ
    R = fwd(h)
    rows, prev = [], set()
    for d in rebalance_days(h, start, offset=offset):
        s = score.loc[d].where(univ.loc[d]).dropna() if d in score.index else pd.Series(dtype=float)
        u = univ.loc[d] & R.loc[d].notna()
        if len(s) < k or u.sum() < 50:
            continue
        pick = (s.nlargest(k) if largest else s.nsmallest(k)).index
        r = R.loc[d, pick].fillna(0)
        new = set(pick) - prev
        out = prev - set(pick)
        c = (COST.loc[d, list(new)].fillna(0.01).sum() + COST.loc[d, list(out)].fillna(0.01).sum()) / k if prev else \
            2 * COST.loc[d, pick].fillna(0.01).mean()
        prev = set(pick)
        rows.append(dict(date=d, gross=r.mean(), cost=c, net=r.mean() - c, bench=R.loc[d][u].mean(),
                         spy=SPY.shift(-h).loc[d] / SPY.loc[d] - 1, turnover=len(new) / k))
    df = pd.DataFrame(rows).set_index("date")
    df["excess"] = df.net - df.bench
    return df


def stats(df, h, col="excess"):
    x = df[col].dropna()
    per = 252 / h
    sr = x.mean() / x.std() * np.sqrt(per) if x.std() > 0 else np.nan
    t = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
    return dict(n=len(x), mean_bp=1e4 * x.mean(), ann=(1 + x.mean()) ** per - 1, sharpe=sr, t=t,
                hit=(x > 0).mean())


def summarize(name, df, h, periods=(("2022-23", "2022-01", "2023-12"), ("2024-25H1", "2024-01", "2025-06"),
                                    ("2025H2-26", "2025-07", "2026-09"))):
    out = []
    for p, a, b in periods:
        x = df.loc[a:b]
        if len(x) < 3:
            continue
        e, n = stats(x, h, "excess"), stats(x, h, "net")
        out.append(dict(variant=name, period=p, n=e["n"], excess_bp=e["mean_bp"], excess_sharpe=e["sharpe"],
                        excess_t=e["t"], net_ann=n["ann"], net_sharpe=n["sharpe"], spy_ann=(1 + x.spy.mean()) ** (252 / h) - 1,
                        turnover=x.turnover.mean(), cost_bp=1e4 * x.cost.mean()))
    return out
