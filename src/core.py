"""Shared data loading, cost model, and statistics for the strategy research.

Conventions
-----------
* Wide panels: DataFrame indexed by trading date, one column per ticker.
* Prices o/h/l/c are split- and dividend-adjusted with the Yahoo adj factor
  (adj_close / close) so multi-day returns are total returns. `rawc` keeps the
  unadjusted close for price filters.
* Costs are charged per side in basis points of traded notional. The model is
  deliberately conservative for a $10k account (market impact ~0):
      cost_side = half_spread + slippage + fees
  half_spread comes from the Abdi-Ranaldo (2017) close-high-low estimator,
  rolling 63-day median per stock (lagged one day), floored by a
  dollar-volume bucket table taken from typical quoted spreads.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# Research split (fixed before any testing):
DEV_START, DEV_END = "2020-01-01", "2023-12-31"      # design / parameter choice
VAL_START, VAL_END = "2024-01-01", "2025-06-30"      # validation (2024-2025H1)
OOS_START, OOS_END = "2025-07-01", "2026-09-25"      # final holdout; looked at once per strategy

ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
        "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE", "ARKK", "TLT", "HYG", "GLD", "USO", "UUP", "TQQQ",
        "SQQQ", "SOXL", "SOXS", "SPXL", "UPRO", "TNA", "TZA", "^VIX", "^VIX9D", "^VIX3M"]

_PANEL = None


def load_panel(cache=True):
    """Return dict of wide float32 panels: o h l c rawc v dv (dollar volume), plus 'etf' ohlc."""
    global _PANEL
    if _PANEL is not None:
        return _PANEL
    pq = os.path.join(DATA, "panel.pkl")
    if cache and os.path.exists(pq):
        _PANEL = pd.read_pickle(pq)
        return _PANEL
    d = pd.read_parquet(os.path.join(DATA, "daily_all.parquet"))
    d = d[d.date >= "2019-06-01"]
    d = d[(d.close > 0) & (d.open > 0) & (d.high > 0) & (d.low > 0)]
    f = (d.adj_close / d.close).astype("float64")
    d["o"], d["h"], d["l"], d["c"] = d.open * f, d.high * f, d.low * f, d.adj_close
    d["rawc"] = d.close
    d["v"] = d.volume
    P = {}
    for k in ["o", "h", "l", "c", "rawc", "v"]:
        P[k] = d.pivot(index="date", columns="ticker", values=k).astype("float32")
    # align on SPY trading days
    days = P["c"]["SPY"].dropna().index
    for k in P:
        P[k] = P[k].reindex(days)
    P["dv"] = (P["rawc"] * P["v"]).astype("float32")
    # data sanity: kill bars with obviously bad OHLC (low > high etc.)
    bad = (P["l"] > P["h"] * 1.0001) | (P["o"] > P["h"] * 1.02) | (P["o"] < P["l"] * 0.98)
    for k in ["o", "h", "l", "c"]:
        P[k] = P[k].mask(bad)
    pd.to_pickle(P, pq)
    _PANEL = P
    return P


def stock_cols(P):
    return [c for c in P["c"].columns if c not in ETFS]


def half_spread_bps(P):
    """Per stock-day half-spread estimate in bps, known before the open of that day."""
    lc, lh, ll = np.log(P["c"]), np.log(P["h"]), np.log(P["l"])
    eta = (lh + ll) / 2
    # Abdi-Ranaldo: s^2 = 4 (c_t - eta_t)(c_t - eta_{t+1}); use t-1 values so it is known at t
    s2 = 4 * (lc - eta) * (lc - eta.shift(-1))
    s2 = s2.shift(1)
    ar = np.sqrt(s2.clip(lower=0).rolling(63, min_periods=20).mean()) * 1e4  # full spread bps
    # floor table on 20d median dollar volume
    adv = P["dv"].rolling(20, min_periods=5).median().shift(1)
    floor = pd.DataFrame(np.select(
        [adv > 5e8, adv > 1e8, adv > 2e7, adv > 5e6, adv > 1e6],
        [1.0, 2.0, 5.0, 12.0, 30.0], 75.0), index=adv.index, columns=adv.columns)
    # price floor: 1 cent tick relative to price (half-tick minimum)
    tick = 0.5 * 0.01 / P["rawc"].shift(1) * 1e4
    hs = np.maximum(np.maximum(ar / 2, floor), tick)
    return hs.clip(upper=500).astype("float32")


def cost_bps(hs, slippage=2.0, fees=0.3):
    """Per-side cost in bps given half-spread table."""
    return hs + slippage + fees


# ---------------------------------------------------------------- stats
def ann_stats(r, periods=252):
    """r: daily portfolio returns (Series, decimal)."""
    r = r.dropna()
    if len(r) < 5 or r.std() == 0:
        return dict(n=len(r), ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, tstat=np.nan, maxdd=np.nan,
                    hit=np.nan, total=np.nan)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sr = r.mean() / r.std() * np.sqrt(periods)
    return dict(n=len(r), ann_ret=(eq.iloc[-1]) ** (periods / len(r)) - 1, ann_vol=r.std() * np.sqrt(periods),
                sharpe=sr, tstat=r.mean() / r.std() * np.sqrt(len(r)), maxdd=dd, hit=(r > 0).mean(),
                total=eq.iloc[-1] - 1)


def split_stats(r, periods=252):
    out = {}
    for name, a, b in [("dev", DEV_START, DEV_END), ("val", VAL_START, VAL_END), ("oos", OOS_START, OOS_END),
                       ("2024-26", VAL_START, OOS_END)]:
        out[name] = ann_stats(r.loc[a:b], periods)
    return pd.DataFrame(out).T


def deflated_sharpe(sr_obs, n_trials, T, skew=0.0, kurt=3.0, sr_var=None):
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio (per-period SR inputs).
    Returns probability that true SR > 0 after accounting for n_trials."""
    from scipy.stats import norm
    emc = 0.5772156649
    if sr_var is None:
        sr_var = 1.0 / T
    sr0 = np.sqrt(sr_var) * ((1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    z = (sr_obs - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2)
    return norm.cdf(z)


def block_bootstrap_sharpe(r, block=10, n=2000, seed=0, periods=252):
    """Stationary-ish block bootstrap CI for annualised Sharpe."""
    r = r.dropna().values
    rng = np.random.default_rng(seed)
    T = len(r)
    out = []
    for _ in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + block, T)))
        x = r[np.array(idx[:T])]
        out.append(x.mean() / x.std() * np.sqrt(periods))
    return np.percentile(out, [2.5, 50, 97.5])
