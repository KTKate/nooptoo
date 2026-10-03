"""Study 77b: which input combinations do the current overnight models rely on? (descriptive)

Models: data/models/night_2026Q3.txt (pooled overnight regression, 53 inputs) and the study-33 jump classifiers
data/models/jump_jump_2026Q3.txt / jump_drop_2026Q3.txt (53 inputs + earn_tonight, news_today, built exactly as in
src/study33_jump_live.py add_extra: earnings calendar row on d or d+1, number of news articles in today's window).
Sample: 20,000 random rows of data/ml_frame.parquet dated 2026Q2 (seed 77) -- the quarter before the models' quarter,
so the rows are close to (possibly inside) the models' training data; this describes the fitted functions, it is
not a test.
Attribution: main effects from lightgbm pred_contrib (mean |SHAP value| per input, raw score scale: log-odds for the
classifiers); pairwise SHAP interaction values from shap.TreeExplainer.shap_interaction_values (chunks of 1,000
rows). Shares: total = sum over i, j of mean |Phi_ij| (diagonal = pure main effect Phi_ii, each off-diagonal pair
counted twice); main share = mean |Phi_ii| / total; pair share = 2 mean |Phi_ij| / total. The pred_contrib share is
mean |phi_i| / sum_k mean |phi_k| (phi_i = Phi_ii + sum_j!=i Phi_ij, so it includes the input's interactions).
Output: results/study77b_interactions.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import store
import news_features as NF
from core import load_panel, stock_cols, RES, DATA

OUT = os.path.join(RES, "study77b_interactions.csv")
NS = int(os.environ.get("NS", "20000"))
CH = 1000
MODELS = ["night_2026Q3", "jump_jump_2026Q3", "jump_drop_2026Q3"]


def sample():
    X = pd.read_parquet(f"{DATA}/ml_frame.parquet", filters=[("date", ">=", pd.Timestamp("2026-04-01")),
                                                             ("date", "<=", pd.Timestamp("2026-06-30"))])
    X = X[[k for k in X.columns if not k.startswith("y_")]]
    print("2026Q2 rows", len(X), flush=True)
    X = X.sample(n=min(NS, len(X)), random_state=77).sort_index()
    # study33 add_extra (copied: importing study33 would rebuild study 23's frames)
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    del P
    E = store.read("earnings")
    di = days.searchsorted(pd.to_datetime(E.date))
    earn = pd.DataFrame(0.0, index=days, columns=cols)
    ci = {t: i for i, t in enumerate(cols)}
    for d, t in zip(di, E.symbol):
        if t in ci and 0 < d < len(days):
            earn.iat[d, ci[t]] = 1.0
            earn.iat[d - 1, ci[t]] = 1.0
    news = NF.load()["n_news"].reindex(index=days, columns=cols)
    r = days.get_indexer(X.index.get_level_values(0))
    c = pd.Index(cols).get_indexer(X.index.get_level_values(1))
    ok = (r >= 0) & (c >= 0)
    for k, panel in [("earn_tonight", earn), ("news_today", news)]:
        v = np.full(len(X), np.nan, dtype="float32")
        v[ok] = panel.values[r[ok], c[ok]]
        X[k] = v
    print("earn_tonight share", float(np.nanmean(X.earn_tonight)), "news_today mean", float(np.nanmean(X.news_today)),
          flush=True)
    return X


def main():
    import shap
    X = sample()
    rows = []
    for name in MODELS:
        m = lgb.Booster(model_file=f"{DATA}/models/{name}.txt")
        f = m.feature_name()
        Xm = X[f].astype("float64")
        pc = m.predict(Xm, pred_contrib=True)[:, :-1]
        main_pc = np.abs(pc).mean(0)
        ex = shap.TreeExplainer(m)
        acc = np.zeros((len(f), len(f)))
        t0 = time.time()
        for s in range(0, len(Xm), CH):
            iv = ex.shap_interaction_values(Xm.iloc[s:s + CH])
            acc += np.abs(iv).sum(0)
            if s == 0:
                # consistency: row sums of interaction values = SHAP values = pred_contrib
                err = np.abs(iv.sum(2) - pc[:CH]).max()
                print(name, "max |sum_j Phi_ij - pred_contrib|", float(err), "chunk secs", round(time.time() - t0, 1),
                      flush=True)
        acc /= len(Xm)
        tot = acc.sum()
        diag = np.diag(acc)
        print(name, "done", round(time.time() - t0), "s; main share", round(diag.sum() / tot, 3), flush=True)
        rows.append(dict(model=name, kind="summary", feature="all", share=diag.sum() / tot, value=tot,
                         note="share = pure main effects / total; 1-share = pairwise interactions"))
        o = np.argsort(-diag)
        pcs = main_pc / main_pc.sum()
        for rk, i in enumerate(o[:15]):
            rows.append(dict(model=name, kind="main", rank=rk + 1, feature=f[i], value=diag[i], share=diag[i] / tot,
                             pred_contrib_mean_abs=main_pc[i], pred_contrib_share=pcs[i]))
        po = np.argsort(-main_pc)
        for rk, i in enumerate(po[:15]):
            rows.append(dict(model=name, kind="pred_contrib", rank=rk + 1, feature=f[i], value=main_pc[i],
                             share=pcs[i]))
        iu = np.triu_indices(len(f), 1)
        pv = 2 * acc[iu]
        o = np.argsort(-pv)
        for rk, k in enumerate(o[:15]):
            i, j = iu[0][k], iu[1][k]
            rows.append(dict(model=name, kind="pair", rank=rk + 1, feature=f"{f[i]} x {f[j]}", value=pv[k],
                             share=pv[k] / tot))
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.drop(columns=[c for c in ["note"] if c in df]).round(4).to_string())


if __name__ == "__main__":
    main()
