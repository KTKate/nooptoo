"""Study 30: small MLP as an extra member of the overnight ranking ensemble.

Same walk-forward as study3/study10 (quarterly retrain, test 2022Q1..2026Q3, train on all rows before the quarter
start minus a 10-trading-day embargo, target = daily percentile rank of y_night - 0.5).
Features: each of the 53 features rank-normalized per day to [-0.5, 0.5], missing -> 0.
Net: 53 -> 64 -> 32 -> 1, ReLU, dropout 0.1, Adam 1e-3, batch 4096, EPOCHS epochs, MSE, average of 3 seeds.
Evaluation: MLP alone and 6-member rank average (5 study14 LightGBM members + MLP) vs the 5-member average;
top 10 / 20, closing auction buy, opening auction sell, auction cost + 2.5 bp per side.
    python src/study30_mlp.py            (set RETRAIN=0 to reuse results/study30_pred_mlp.parquet)
Output: results/study30_pred_mlp.parquet, results/study30_sharpe.csv, results/study30_ic.csv, results/study30_diff.csv
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

torch.set_num_threads(2)
EPOCHS = int(os.environ.get("EPOCHS", 4))
SEEDS = [0, 1, 2]
BS = 4096
FN = f"{RES}/study30_pred_mlp.parquet"

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
PERIODS = [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]


def build():
    # read one column at a time to keep peak memory low
    fn = f"{DATA}/ml_frame.parquet"
    allc = pq.ParquetFile(fn).schema_arrow.names
    feat = [k for k in allc if not k.startswith("y_") and k not in ("date", "ticker")]
    y = pd.read_parquet(fn, columns=["y_night"])["y_night"]
    keep = np.asarray(y.index.get_level_values(0) >= "2020-01-01")
    y = y[keep]
    idx = y.index
    codes = pd.Series(pd.factorize(idx.get_level_values(0))[0])
    rank = lambda v: (pd.Series(v).groupby(codes).rank(pct=True) - 0.5).values.astype("float32")
    yr = rank(y.values)
    Z = np.zeros((len(y), len(feat)), dtype="float32")
    for j, k in enumerate(feat):
        v = pd.read_parquet(fn, columns=[k])[k].values[keep]
        Z[:, j] = np.nan_to_num(rank(v), nan=0.0)
    return Z, yr, idx, len(feat)


def net(nf):
    return torch.nn.Sequential(torch.nn.Linear(nf, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(32, 1))


def train_predict(Xtr, ytr, Xte, nf):
    Xtr, ytr, Xte = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xte)
    out = np.zeros(len(Xte), dtype="float64")
    for s in SEEDS:
        torch.manual_seed(s)
        m = net(nf)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(s)
        m.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), BS):
                b = perm[i:i + BS]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(m(Xtr[b]).squeeze(1), ytr[b])
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            out += np.concatenate([m(Xte[i:i + 65536]).squeeze(1).numpy() for i in range(0, len(Xte), 65536)])
    return out / len(SEEDS)


if os.environ.get("RETRAIN", "1") == "1" or not os.path.exists(FN):
    Z, yr, idx, nf = build()
    dates = idx.get_level_values(0)
    ok = ~np.isnan(yr)
    preds = []
    for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
        t0 = time.time()
        a, b = q.start_time, q.end_time
        cut = days[max(0, days.searchsorted(a) - 1 - 10)]
        tr = np.asarray(ok & (dates < cut))
        te = np.asarray((dates >= a) & (dates <= b))
        if te.sum() == 0:
            continue
        p = train_predict(Z[tr], yr[tr], Z[te], nf)
        preds.append(pd.Series(p, index=idx[te]))
        print(q, "train", int(tr.sum()), "test", int(te.sum()), f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"pred": pd.concat(preds)}).to_parquet(FN)
    del Z

mlp = pd.read_parquet(FN)["pred"]
names = ["pooled", "per_behavior", "per_comove", "pooled_plus_behavior", "pooled_plus_comove"]
W = lambda s: s.unstack().reindex(columns=cols)
rk = lambda S: S.rank(axis=1, pct=True)
lgb_ranks = [rk(W(pd.read_parquet(f"{RES}/study14_pred_{n}.parquet")["pred"])) for n in names]
ix = lgb_ranks[0].index
lgb_ranks = [r.reindex(ix) for r in lgb_ranks]
M = rk(W(mlp).reindex(ix))
E = sum(lgb_ranks) / 5
E6 = (sum(lgb_ranks) + M) / 6
elig = E.notna() & M.notna()
scores = {"lgb5": E, "mlp": M, "ens6": E6}

o, c = P["o"][cols], P["c"][cols]
R = o.shift(-1) / c - 1
cost = exec_cost_bps(P, "auction")[cols] + 2.5
Y = R.reindex(ix)

rows, nets = [], {}
for name, S in scores.items():
    for k in [10, 20]:
        Wt = bt.select_topk(S, elig, k)
        r = bt.run(Wt, R.reindex_like(Wt), cost.reindex_like(Wt))
        r = r[r.index < days[-1]]
        nets[(name, k)] = r.net
        for per, a_, b_ in PERIODS:
            x = r.loc[a_:b_]
            st = ann_stats(x.net)
            rows.append(dict(model=name, k=k, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                             gross_bps=1e4 * x.gross.mean(), net_bps=1e4 * x.net.mean()))
sh = pd.DataFrame(rows)
sh.to_csv(f"{RES}/study30_sharpe.csv", index=False)
print(sh.pivot_table(index=["model", "k"], columns="period", values="sharpe", sort=False).round(2))

# daily rank IC vs realised overnight return, and rank correlation of MLP with the LightGBM average
Yr = Y.where(elig).rank(axis=1)
ic_rows = []
daily = {}
for name, S in list(scores.items()) + [("mlp_vs_lgb5", None)]:
    A = (S.where(elig) if S is not None else M.where(elig)).rank(axis=1)
    B = Yr if S is not None else E.where(elig).rank(axis=1)
    daily[name] = A.corrwith(B, axis=1)
for name, s in daily.items():
    s = s.dropna()
    s = s[s.index < days[-1]]
    for per, a_, b_ in PERIODS + [("all", "2022-01", "2026-09")]:
        x = s.loc[a_:b_]
        ic_rows.append(dict(series=name, period=per, mean=x.mean(), t=x.mean() / x.std() * np.sqrt(len(x)), n=len(x)))
ic = pd.DataFrame(ic_rows)
ic.to_csv(f"{RES}/study30_ic.csv", index=False)
print(ic.pivot_table(index="series", columns="period", values="mean", sort=False).round(4))
print(ic.pivot_table(index="series", columns="period", values="t", sort=False).round(2))

d_rows = []
for k in [10, 20]:
    d = (nets[("ens6", k)] - nets[("lgb5", k)]).dropna()
    for per, a_, b_ in PERIODS + [("all", "2022-01", "2026-09")]:
        x = d.loc[a_:b_]
        d_rows.append(dict(k=k, period=per, mean_bps=1e4 * x.mean(), t=x.mean() / x.std() * np.sqrt(len(x)),
                           n=len(x)))
dd = pd.DataFrame(d_rows)
dd.to_csv(f"{RES}/study30_diff.csv", index=False)
print(dd.round(3))
