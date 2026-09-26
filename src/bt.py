"""Generic cross-sectional backtester on wide daily panels.

A strategy is: signal S (date x ticker, must be computable at decision time),
eligibility mask E, a per-period holding return R (e.g. open->close of day t),
and a selection rule (top-k / bottom-k by S). Portfolio is equal weight among
selected names, rebalanced each period, costs charged per side on traded
weight changes.
"""
import numpy as np
import pandas as pd


def select_topk(S, E, k, largest=True):
    """Return 0/1 weights (equal weight among the k selected names each day)."""
    s = S.where(E)
    if not largest:
        s = -s
    rk = s.rank(axis=1, ascending=False, method="first")
    W = (rk <= k).astype("float32")
    n = W.sum(axis=1).replace(0, np.nan)
    return W.div(n, axis=0).fillna(0.0)


def select_quantile(S, E, q=0.1, largest=True):
    s = S.where(E)
    if not largest:
        s = -s
    pct = s.rank(axis=1, pct=True)
    W = (pct >= 1 - q).astype("float32")
    n = W.sum(axis=1).replace(0, np.nan)
    return W.div(n, axis=0).fillna(0.0)


def run(W, R, cost_side_bps, roundtrip=True, side=1):
    """W: weights held over period t (row t), R: holding return in period t.
    roundtrip=True: every period position is opened and closed (intraday or
    overnight holds): cost = 2 * cost_side on gross weight.
    roundtrip=False: positions carried; cost = cost_side * |dW|.
    side=+1 long, -1 short (returns are negated for shorts).
    Returns DataFrame with gross, cost, net, n (names held), turnover."""
    R = R.reindex_like(W)
    Rn = R.fillna(0.0)  # missing return while held -> 0 (rare; flagged by n_missing)
    gross = side * (W * Rn).sum(axis=1)
    C = cost_side_bps.reindex_like(W).fillna(100.0) / 1e4
    if roundtrip:
        cost = (2 * W.abs() * C).sum(axis=1)
        turn = 2 * W.abs().sum(axis=1)
    else:
        dW = W.diff().abs()
        dW.iloc[0] = W.iloc[0].abs()
        cost = (dW * C).sum(axis=1)
        turn = dW.sum(axis=1)
    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                        "n": (W > 0).sum(axis=1), "turnover": turn,
                        "n_missing": ((W > 0) & R.isna()).sum(axis=1)})
    return out


def hold_k_days(W1, h):
    """Turn a daily entry signal into overlapping h-day holdings (1/h of capital per cohort)."""
    Wh = sum(W1.shift(i).fillna(0.0) for i in range(h)) / h
    return Wh
