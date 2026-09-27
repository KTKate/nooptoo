"""Feature construction for the LightGBM study (shared by study3_ml.py and study3_timing.py).
All features for day t use data up to the close of t (or the substituted intraday values)."""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
import store
_E = store.read("earnings")
_E = _E[_E.symbol.isin(cols)]


def earnings_features():
    """Earnings-calendar features on the full daily grid (dates are announced in advance; the surprise is
    known after the report, used from the close of D+1)."""
    F = {}
    di = days.searchsorted(pd.to_datetime(_E.date).values)
    ev = pd.DataFrame({"di": di, "t": _E.symbol.values, "s": _E.surprise.values})
    ev = ev[ev.di < len(days)]
    flag = pd.DataFrame(0.0, index=days, columns=cols)
    surp = pd.DataFrame(np.nan, index=days, columns=cols)
    for d_, t_, s_ in ev.itertuples(index=False):
        flag.iat[d_, flag.columns.get_loc(t_)] = 1.0
        if d_ + 1 < len(days):
            surp.iat[d_ + 1, surp.columns.get_loc(t_)] = s_   # usable from close of D+1
    idx = np.arange(len(days), dtype=float)
    last_e = flag.mul(idx, axis=0).replace(0, np.nan).ffill()
    next_e = flag.mul(idx, axis=0).replace(0, np.nan).bfill()
    F["days_since_e"] = (-last_e).add(idx, axis=0).clip(upper=90)
    F["days_to_e"] = next_e.sub(idx, axis=0).clip(upper=15)  # dates usually announced 2-4 weeks ahead
    F["last_surp"] = surp.ffill(limit=70).clip(-100, 100)

    return F


def build(o, h, l, c, rawc, v, dv, spy_c, vix, vix3m, iwm_c, earn=None):
    """Feature panels F (stock level) and mkt (market level) from panels whose last row for day t may be
    replaced by intraday values (e.g. 15:45 prices) to test executable timing."""
    lr = np.log(c / c.shift(1))
    night = np.log(o / c.shift(1))
    intra = np.log(c / o)
    adv20 = dv.rolling(20, min_periods=10).median()
    F = {}
    for n in [1, 2, 3, 5, 10, 20, 60, 120]:
        F[f"r{n}"] = lr.rolling(n, min_periods=max(1, n // 2)).sum()
    F["r250_20"] = lr.rolling(250, min_periods=150).sum() - F["r20"]
    F["night1"] = night
    F["intra1"] = intra
    F["night5"] = night.rolling(5).sum()
    F["intra5"] = intra.rolling(5).sum()
    F["night20"] = night.rolling(20, min_periods=15).mean()
    F["intra20"] = intra.rolling(20, min_periods=15).mean()
    F["vol5"] = lr.rolling(5).std()
    F["vol20"] = lr.rolling(20, min_periods=15).std()
    F["vol60"] = lr.rolling(60, min_periods=40).std()
    F["vr"] = F["vol5"] / F["vol60"]
    F["range1"] = np.log(h / l)
    F["range20"] = F["range1"].rolling(20, min_periods=15).mean()
    F["clv"] = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)   # close location in day range
    F["clv5"] = F["clv"].rolling(5).mean()
    F["vz1"] = np.log(v / v.rolling(20, min_periods=10).mean())
    F["vz5"] = np.log(v.rolling(5).mean() / v.rolling(60, min_periods=40).mean())
    F["ldv"] = np.log(adv20)
    F["lpx"] = np.log(rawc)
    F["hi20"] = np.log(c / c.rolling(20, min_periods=15).max())
    F["lo20"] = np.log(c / c.rolling(20, min_periods=15).min())
    F["hi250"] = np.log(c / c.rolling(250, min_periods=150).max())
    spy = np.log(spy_c / spy_c.shift(1))
    beta = (lr.rolling(60, min_periods=40).cov(spy)).div(spy.rolling(60, min_periods=40).var(), axis=0)
    F["beta"] = beta
    F["resid1"] = lr - beta.mul(spy, axis=0)
    F["resid5"] = F["resid1"].rolling(5).sum()
    idx_days = c.index
    if earn is None:
        earn = earnings_features()
    for k in ["days_since_e", "days_to_e", "last_surp"]:
        F[k] = earn[k].reindex(index=idx_days, columns=c.columns)
    days_ = c.index
    mkt = pd.DataFrame(index=days_)
    mkt["spy1"] = spy
    mkt["spy5"] = spy.rolling(5).sum()
    mkt["spy20"] = spy.rolling(20).sum()
    mkt["vix"] = vix
    mkt["vix_chg5"] = np.log(vix / vix.shift(5))
    mkt["vix_term"] = vix / vix3m
    mkt["iwm_spy5"] = np.log(iwm_c / iwm_c.shift(5)) - mkt["spy5"]
    mkt["dow"] = days_.dayofweek
    mm = pd.Series(days.month, index=days)
    mkt["tom"] = (mm != mm.shift(-1)).astype(float).reindex(days_)     # calendar: known in advance
    return F, mkt, adv20


o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
F, mkt, adv20 = build(o, h, l, c, rawc, v, dv, P["c"]["SPY"], P["c"]["^VIX"], P["c"]["^VIX3M"], P["c"]["IWM"])
univ = (rawc > 5) & (adv20 > 5e6)

T = {
    "night": o.shift(-1) / c - 1,
    "day1": c.shift(-1) / o.shift(-1) - 1,
    "cc1": c.shift(-1) / c - 1,
    "cc5": c.shift(-5) / c - 1,
}
H = {"night": 1, "day1": 2, "cc1": 1, "cc5": 5}  # days until target fully realized (for embargo)


def features_frame(F, mkt, mask):
    """Long feature frame (date, ticker) for the rows where mask is True (same columns as training)."""
    m = mask.stack()
    m = m[m]
    X = pd.DataFrame(index=m.index)
    for k, f in F.items():
        X[k] = f.stack().reindex(m.index).astype("float32")
    for k in ["r1", "r5", "r20", "night1", "intra1", "vz1", "vol20", "resid5"]:
        X[k + "_cs"] = X[k].groupby(level=0).rank(pct=True).astype("float32")
    dates = X.index.get_level_values(0)
    for k in mkt.columns:
        X[k] = mkt[k].reindex(dates).values.astype("float32")
    return X

