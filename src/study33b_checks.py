"""Study 33b: checks on the study-33 result (jump/drop classifiers blended with the ensemble at 15:45).

For ensemble, jmd, blend and blend_2to1 (top 10, auction costs + 2.5 bp per side), 2024-01 .. 2026-09:
  half-year Sharpe; Sharpe with the 20 best single trades set to zero; share of picks on earnings nights and Sharpe with
  those names excluded before selection; beta to SPY's close-to-open return and the hedged Sharpe; Sharpe at twice the
  cost; deflated Sharpe (1,250 trials); and a paired block bootstrap of the daily net difference against the ensemble.
Output: results/study33b_checks.csv
"""
import numpy as np
import pandas as pd
import store
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, deflated_sharpe, RES
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[(ens.index < days[-1]) & (ens.index <= "2026-09-30")]
pj = pred["p_jump"].unstack().reindex(index=ens.index, columns=cols)
pdr = pred["p_drop"].unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
ens = ens.where(ok)
jmd = (pj - pdr).where(ok)
rk = lambda s: s.rank(axis=1, pct=True)
V = {"ensemble": ens, "jmd": jmd, "blend": (rk(ens) + rk(jmd)) / 2, "blend_2to1": (2 * rk(ens) + rk(jmd)) / 3}
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(ens)
cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(ens)
spy = (P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1).reindex(ens.index)

E = store.read("earnings")
di = days.searchsorted(pd.to_datetime(E.date))
earn = pd.DataFrame(False, index=days, columns=cols)
ci = {t: i for i, t in enumerate(cols)}
for d, t in zip(di, E.symbol):
    if t in ci and 0 < d < len(days):
        earn.iat[d, ci[t]] = True
        earn.iat[d - 1, ci[t]] = True
earn = earn.reindex_like(ens)


def boot_diff(x, block=10, n=4000, seed=0):
    x = x.dropna().values
    rng = np.random.default_rng(seed)
    T = len(x)
    m = []
    for _ in range(n):
        s = rng.integers(0, T, T // block + 1)
        idx = (s[:, None] + np.arange(block)).ravel()[:T] % T
        m.append(x[idx].mean())
    m = np.array(m)
    return x.mean() / (m.std() + 1e-12), float((m <= 0).mean())


rows, nets = [], {}
for name, sc in V.items():
    W = bt.select_topk(sc, sc.notna(), 10)
    r = bt.run(W, R, cost)
    nets[name] = r.net
    d = dict(variant=name, sharpe=ann_stats(r.net)["sharpe"], ann_ret=ann_stats(r.net)["ann_ret"],
             maxdd=ann_stats(r.net)["maxdd"])
    for h, g in r.net.groupby([r.net.index.year, (r.net.index.month > 6) + 1]):
        d[f"{h[0]}H{h[1]}"] = ann_stats(g)["sharpe"]
    tr = (W * R).stack()
    tr = tr[W.stack() > 0]
    top = tr.nlargest(20)
    d["top20_share_of_gross"] = top.sum() / tr.sum()
    Rx = R.copy()
    for (dd, t) in top.index:
        Rx.at[dd, t] = 0.0
    d["sharpe_without_top20"] = ann_stats(bt.run(W, Rx, cost).net)["sharpe"]
    d["earn_night_share"] = float((W > 0)[earn].sum().sum() / (W > 0).sum().sum())
    We = bt.select_topk(sc.where(~earn), sc.notna() & ~earn, 10)
    d["sharpe_no_earn_nights"] = ann_stats(bt.run(We, R, cost).net)["sharpe"]
    b = np.polyfit(spy.fillna(0), r.net.reindex(spy.index).fillna(0), 1)[0]
    d["beta_spy_night"] = b
    d["hedged_sharpe"] = ann_stats(r.net - b * spy.fillna(0))["sharpe"]
    d["sharpe_2x_cost"] = ann_stats(r.gross - 2 * r.cost)["sharpe"]
    x = r.net.dropna()
    d["dsr_1250"] = deflated_sharpe(x.mean() / x.std(), 1250, len(x), skew=float(x.skew()), kurt=float(x.kurt() + 3))
    if name != "ensemble":
        t, p = boot_diff(r.net - nets["ensemble"])
        d["t_vs_ensemble"], d["p_vs_ensemble"] = t, p
        for per, a, bb in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
            d[f"t_vs_ensemble_{per}"] = boot_diff((r.net - nets["ensemble"]).loc[a:bb])[0]
    rows.append(d)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study33b_checks.csv", index=False)
print(df.round(3).T.to_string())
