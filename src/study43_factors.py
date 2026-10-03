"""Study 43: monthly-rebalanced long-only single-factor and two-factor portfolios.

Question: which simple cross-sectional factors give a top-20 / top-quintile long-only portfolio that beats the
equal-weight liquid universe (and SPY) net of costs over a 21-day holding period, and does a combination chosen on
2020-23 hold up in 2024-26?

Data: daily panels via src/horizon_lib.py (survivorship-biased: tickers alive in 2026); point-in-time fundamentals
data/local/fundamentals_panels.pkl (filing counts from the trading day after it is filed; days_since_filing > 200
treated as missing); analyst panels via src/analyst_features.py (pt_gap from 2023-12 only, pt_net20 from 2019).
Factors (higher score = buy): value_evs (-ev_sales, ev_sales > 0), value_bm (book_to_market), value_fcf (fcf_yield),
quality_opm (op_margin), quality_sbc (-sbc_to_revenue), growth (revenue_growth_yoy), an_gap (pt_gap), an_net20
(pt_net20), mom12_1 (C[t-21]/C[t-252]-1), lowvol (-std of 60 daily returns).
Design: universe = liquid filter (price > $5, ADV > $20M; study41.univ_ext, which extends horizon_lib's traded-close
filter before 2023-12). Every 21 trading days from 2020-01 buy the top 20 (horizon_lib.backtest) or the top quintile
of scored universe names (same logic, k = 20% of scored names), equal weight, closing auction, hold 21 days, cost
per side horizon_lib COST on changed names. Combination: the two single factors with the best 2020-23 top-quintile
excess Sharpe (only factors with >= 24 months in 2020-23), 50/50 average of within-day percentile ranks; 2024-26 is
the out-of-sample check.
Output: results/study43_factors.csv (one row per variant x k x period)
"""
import os
import numpy as np
import pandas as pd
import horizon_lib as H
from core import RES, DATA
from study41_event_drift import univ_ext

PERIODS = (("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31"), ("all", "2020-01-01", "2026-12-31"))
HZ = 21


def backtest_q(score, h, frac, univ, start, min_names=100):
    """Top-fraction version of horizon_lib.backtest (same costs and benchmark)."""
    R = H.fwd(h)
    rows, prev = [], set()
    for d in H.rebalance_days(h, start):
        s = score.loc[d].where(univ.loc[d] & R.loc[d].notna()).dropna()
        u = univ.loc[d] & R.loc[d].notna()
        if len(s) < min_names:
            continue
        k = int(len(s) * frac)
        pick = s.nlargest(k).index
        r = R.loc[d, pick].fillna(0)
        new, out = set(pick) - prev, prev - set(pick)
        c = (H.COST.loc[d, list(new)].fillna(0.01).sum() / k + H.COST.loc[d, list(out)].fillna(0.01).sum() / max(len(prev), 1)) \
            if prev else 2 * H.COST.loc[d, pick].fillna(0.01).mean()
        prev = set(pick)
        rows.append(dict(date=d, gross=r.mean(), cost=c, net=r.mean() - c, bench=R.loc[d][u].mean(),
                         spy=H.SPY.shift(-h).loc[d] / H.SPY.loc[d] - 1, turnover=len(new) / k, k=k))
    df = pd.DataFrame(rows).set_index("date")
    df["excess"] = df.net - df.bench
    return df


def summ(name, kind, df):
    out = []
    for p, a, b in PERIODS:
        x = df.loc[a:b]
        if len(x) < 6:
            continue
        e, n, s = H.stats(x, HZ, "excess"), H.stats(x, HZ, "net"), H.stats(x, HZ, "spy")
        out.append(dict(variant=name, k=kind, period=p, n=e["n"], excess_bp=e["mean_bp"], excess_ann=e["ann"],
                        excess_sharpe=e["sharpe"], excess_t=e["t"], excess_hit=e["hit"], net_ann=n["ann"],
                        net_sharpe=n["sharpe"], spy_ann=s["ann"], spy_sharpe=s["sharpe"],
                        bench_ann=(1 + x.bench.mean()) ** (252 / HZ) - 1, turnover=x.turnover.mean(),
                        cost_bp=1e4 * x.cost.mean(), names=x.k.mean() if "k" in x else 20, first=x.index[0].date()))
    return out


if __name__ == "__main__":
    U = univ_ext()
    FP = pd.read_pickle(os.path.join(DATA, "local", "fundamentals_panels.pkl"))
    keep = ["ev_sales", "book_to_market", "fcf_yield", "op_margin", "sbc_to_revenue", "revenue_growth_yoy", "days_since_filing"]
    FP = {k: FP[k].reindex(index=H.days, columns=H.cols).astype("float64") for k in keep}
    import gc
    gc.collect()
    ok = FP.pop("days_since_filing") <= 200
    for k in FP:
        FP[k] = FP[k].where(ok)
    import analyst_features as AF
    A = AF.load()
    S = {"value_evs": -FP["ev_sales"].where(FP["ev_sales"] > 0), "value_bm": FP["book_to_market"],
         "value_fcf": FP["fcf_yield"], "quality_opm": FP["op_margin"], "quality_sbc": -FP["sbc_to_revenue"],
         "growth": FP["revenue_growth_yoy"],
         "an_gap": A["pt_gap"].reindex(index=H.days, columns=H.cols).astype("float64"),
         "an_net20": A["pt_net20"].reindex(index=H.days, columns=H.cols).astype("float64")}
    del FP, A, ok
    gc.collect()
    C = H.C.astype("float64")
    S["mom12_1"] = C.shift(21) / C.shift(252) - 1
    S["lowvol"] = -(C / C.shift(1) - 1).rolling(60, min_periods=50).std()
    # an_net20 is a small integer count: break ties by liquidity (higher ADV first) instead of alphabetically
    S["an_net20"] = S["an_net20"].where(S["an_net20"].notna()) + 1e-12 * H.ADV.rank(axis=1)
    for k in S:
        S[k] = S[k].where(U)
        S[k] = S[k].where(np.isfinite(S[k]))

    rows, bts = [], {}
    for name, sc in S.items():
        b20 = H.backtest(sc, HZ, k=20, univ=U, start="2020-01-01")
        bq = backtest_q(sc, HZ, 0.2, U, "2020-01-01")
        bts[name] = bq
        rows += summ(name, "top20", b20) + summ(name, "topQ", bq)
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    dev = res[(res.k == "topQ") & (res.period == "2020-23") & (res.n >= 24)].sort_values("excess_sharpe", ascending=False)
    print("2020-23 ranking (topQ excess Sharpe):\n", dev[["variant", "n", "excess_bp", "excess_sharpe", "excess_t"]].round(2))
    a, b = dev.variant.iloc[0], dev.variant.iloc[1]
    comb = 0.5 * S[a].rank(axis=1, pct=True) + 0.5 * S[b].rank(axis=1, pct=True)
    name = f"combo_{a}+{b}"
    rows += summ(name, "top20", H.backtest(comb, HZ, k=20, univ=U, start="2020-01-01"))
    rows += summ(name, "topQ", backtest_q(comb, HZ, 0.2, U, "2020-01-01"))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study43_factors.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(res.round(3).to_string())
