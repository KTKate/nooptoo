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
  rolling 63-day mean of the daily terms per stock (lagged one day), floored by a
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
    """Return dict of wide float32 panels: o h l c rawc v dv (dollar volume), built from data/store.
    The derived pickle cache is rebuilt automatically when the store has newer data."""
    global _PANEL
    import store
    stamp = store.meta().get("daily_last", "")
    if _PANEL is not None and _PANEL.get("_stamp") == stamp:
        return _PANEL
    pq = os.path.join(DATA, "panel.pkl")
    if cache and os.path.exists(pq):
        P = pd.read_pickle(pq)
        if P.get("_stamp") == stamp:
            _PANEL = P
            return P
    d = store.read("daily", start="2019-06")
    d = d[(d.c > 0) & (d.o > 0) & (d.h > 0) & (d.l > 0)]
    f = store.adj_factor(d).astype("float64")
    d["rawc"] = d["c"]
    for k in ["o", "h", "l", "c"]:
        d[k] = d[k] * f
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
    P["_stamp"] = stamp
    pd.to_pickle(P, pq)
    _PANEL = P
    return P


def stock_cols(P):
    return [c for c in P["c"].columns if c not in ETFS]


def half_spread_bps(P):
    """Per stock-day half-spread estimate in bps, known before the open of that day."""
    lc, lh, ll = np.log(P["c"]), np.log(P["h"]), np.log(P["l"])
    eta = (lh + ll) / 2
    # Abdi-Ranaldo: s^2 = 4 (c_t - eta_t)(c_t - eta_{t+1}). The term dated t uses the high/low of
    # t+1, so it is only known after the close of t+1: shift by 2 to be known before the open of t.
    s2 = 4 * (lc - eta) * (lc - eta.shift(-1))
    s2 = s2.shift(2)
    # average the raw (possibly negative) terms first, then clip: clipping each day first biases the
    # estimate upward by the noise level (tens of bps even for mega caps)
    ar = np.sqrt(s2.rolling(63, min_periods=20).mean().clip(lower=0)) * 1e4  # full spread bps
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


# ---------------------------------------------------------------- quote-calibrated spread model
_SPREAD_MODEL = None


def half_spread_model(adv, px, vol20, hm="12:00"):
    """Quoted half-spread (bps) predicted from lagged 20d median dollar volume, price and 20d daily
    log-return vol, fitted on an Alpaca SIP NBBO sample (src/cost_calib.py, results/spread_model.json).
    hm: time of day, one of 09:31 09:35 12:00 15:45 (the open is much wider than midday).
    Returns the conditional mean exp(mu + s^2/2), floored at half a cent."""
    global _SPREAD_MODEL
    if _SPREAD_MODEL is None:
        import json
        _SPREAD_MODEL = json.load(open(os.path.join(RES, "spread_model.json")))
    m = _SPREAD_MODEL
    # inputs clipped to the range covered by the quote sample; output capped at 300 bps per side
    adv = np.clip(adv, 2e5, None)
    px = np.clip(px, 1.0, None)
    vol20 = np.clip(vol20, 0.005, 0.15)
    mu = (m["const"] + m["ladv"] * np.log(adv) + m["lpx"] * np.log(px) + m["lvol"] * np.log(vol20)
          + m.get(f"t{hm}", 0.0))
    hs = np.exp(mu + 0.5 * m["resid_sd"] ** 2)
    return np.minimum(np.maximum(hs, 0.5 * 0.01 / px * 1e4), 300.0)


def half_spread_panel(P, hm="12:00"):
    """half_spread_model on the wide daily panel, using only information known before day t."""
    adv = P["dv"].rolling(20, min_periods=10).median().shift(1)
    px = P["rawc"].shift(1)
    vol = np.log(P["c"] / P["c"].shift(1)).rolling(20, min_periods=10).std().shift(1)
    return half_spread_model(adv, px, vol.clip(lower=1e-3), hm).astype("float32")


def exec_cost_bps(P, kind, slippage=2.0, fees=0.3, auction_frac=0.1):
    """Per-side cost panel (bps) by execution type, known before day t:
      'auction'  : market-on-open / market-on-close order. No spread is crossed; charge fees + 1 bp +
                   auction_frac x quoted half-spread for joining a thin auction.
      'open'     : marketable order in the first minutes (quoted spread at 09:35) + slippage + fees
      'mid'      : marketable order midday (12:00 spread) + slippage + fees
      'close'    : marketable order at 15:45 + slippage + fees
    """
    if kind == "auction":
        return (fees + 1.0 + auction_frac * half_spread_panel(P, "15:45")).astype("float32")
    hm = {"open": "09:35", "mid": "12:00", "close": "15:45"}[kind]
    return (half_spread_panel(P, hm) + slippage + fees).astype("float32")
