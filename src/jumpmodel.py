"""Jump/drop classifiers of studies 32-33 for live use.

Models: data/models/jump_{jump,drop}_<quarter>.txt (study33_jump_live.py trains them; LightGBM binary, targets
overnight return > +5% and < -5%). Inputs: the study-3 feature frame plus
  earn_tonight  1 if an earnings report falls after the previous trading day and on or before the next trading day
                (the report is tonight after the close or tomorrow before the open, or was this morning)
  news_today    number of Alpaca/Benzinga articles tagged with 1-3 symbols since 15:45 ET of the previous trading
                day (NaN when none, as in news_features.py)
Score used by the paper runner (study 33, blend_2to1): (2 * pct rank of the ensemble + pct rank of
P(jump) - P(drop)) / 3.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import DATA

MD = os.path.join(DATA, "models")


def model_quarter(q):
    """Latest quarter <= q with both models present."""
    have = sorted({f[len("jump_jump_"):-4] for f in os.listdir(MD) if f.startswith("jump_jump_")})
    have = [h for h in have if os.path.exists(os.path.join(MD, f"jump_drop_{h}.txt")) and pd.Period(h, freq="Q") <= q]
    if not have:
        raise FileNotFoundError("no jump models in data/models")
    return have[-1]


def earn_tonight(tickers, prev_day, next_day):
    import store
    E = store.read("earnings")
    d = pd.to_datetime(E.date)
    s = set(E.symbol[(d > prev_day) & (d <= next_day)])
    return pd.Series([1.0 if t in s else 0.0 for t in tickers], index=tickers, dtype="float32")


def predict(X, q, earn, news):
    """X: feature frame indexed (date, ticker) for one day; earn, news: Series by ticker. Returns p_jump, p_drop."""
    q = model_quarter(pd.Period(q, freq="Q"))
    t = X.index.get_level_values(1)
    X = X.copy()
    X["earn_tonight"] = earn.reindex(t).fillna(0.0).values.astype("float32")
    X["news_today"] = news.reindex(t).values.astype("float32")
    out = {}
    for k in ["jump", "drop"]:
        m = lgb.Booster(model_file=os.path.join(MD, f"jump_{k}_{q}.txt"))
        out[f"p_{k}"] = m.predict(X[m.feature_name()])
    return pd.DataFrame(out, index=X.index), q


def blend(ens, pj, pdr):
    """(2 * rank(ensemble) + rank(P(jump) - P(drop))) / 3, percentile ranks within the day."""
    return (2 * ens.rank(pct=True) + (pj - pdr).rank(pct=True)) / 3
