"""Study 18: can the market reaction to an earnings report be predicted before the report?

Report dates are published weeks ahead (Nasdaq calendar). For each report with known timing (study 11: first
earnings headline before the open = BMO, after the close = AMC), the reaction night is close(R-1) -> open(R).
At 15:45 of R-1 we know: the stock's past surprises (last 4, sign and size), past reaction nights (last 4),
pre-report returns (5, 20, 60 days), volatility, size, news count and mean sentiment in the prior 30 days,
analyst upgrade/downgrade counts in the prior 30 days, insider purchases and sales in the prior 90 days,
short-sale ratio (z-score), sector, and timing (BMO/AMC).
Model: LightGBM regression on the market-adjusted reaction-night return (clipped at +-30%), yearly walk-forward
(train on all reports before Jan 1 of the test year minus 10 days), test years 2022 .. 2026.
Trades: each day, buy (market-on-close at R-1) the reports with the highest predicted reaction, at most 10
names at 1/10 of capital, sell at the open of R; costs per side = auction cost + 2.5 bp. Also: rank correlation
of prediction and outcome, and hit rate of the sign.
Output: results/study18_earnings_predict.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import store
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, RES, DATA
import news_features as NF

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
o, c = P["o"][cols], P["c"][cols]
night = (o / c.shift(1) - 1)
lr = np.log(c / c.shift(1))
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
px = traded_close(P)[cols].shift(1).fillna(P["rawc"][cols].shift(1))
elig = (px > 2) & (adv > 1e6)
mkt_n = night.where(elig).mean(1)
F = NF.load()

E = store.read("earnings")
E["date"] = pd.to_datetime(E.date)
E = E[E.symbol.isin(cols) & (E.date >= "2019-10-01") & (E.date <= days[-2])].drop_duplicates(["symbol", "date"])
E["di"] = days.searchsorted(E.date)
src = os.path.join(DATA, "local", "news_scored")
N = pd.concat([pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.endswith(".parquet")])
N = N[N.headline.str.contains(r"(?i)(\bEPS\b|earnings|results|revenue|sales)", regex=True)]
N["symbol"] = N.symbols.str.split("|")
N = N.explode("symbol")
N["t"] = pd.to_datetime(N.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
E["wstart"] = E.date - pd.Timedelta(hours=8)
M = pd.merge_asof(E.sort_values("wstart"), N[["symbol", "t"]].sort_values("t"), left_on="wstart", right_on="t",
                  by="symbol", direction="forward")
wend = M.date + pd.Timedelta(days=1, hours=9, minutes=30) + pd.to_timedelta(np.where(M.date.dt.dayofweek == 4, 2, 0), "D")
hit = M.t.notna() & (M.t < wend)
M["timing"] = np.where(~hit, "unknown", np.where(M.t < M.date + pd.Timedelta(hours=9, minutes=30), "BMO",
                       np.where(M.t >= M.date + pd.Timedelta(hours=16), "AMC", "during")))
E = M[M.timing.isin(["BMO", "AMC"])].copy()
E["R"] = E.di + (E.timing == "AMC").astype(int)
E = E[(E.R >= 61) & (E.R < len(days) - 1)]
ci = {t: i for i, t in enumerate(cols)}
E["j"] = E.symbol.map(ci)
E = E[elig.values[E.R.values - 1, E.j.values]]
E["y"] = night.values[E.R.values, E.j.values] - mkt_n.values[E.R.values]
E = E[np.isfinite(E.y)]
E = E.sort_values(["symbol", "date"])
# history of the same stock's reports (known before R-1: previous reports only)
g = E.groupby("symbol")
for k in range(1, 5):
    E[f"surp_l{k}"] = g.surprise.shift(k).clip(-200, 200)
    E[f"react_l{k}"] = g.y.shift(k).clip(-0.3, 0.3)
E["surp_mean4"] = E[[f"surp_l{k}" for k in range(1, 5)]].mean(1)
E["surp_pos4"] = (E[[f"surp_l{k}" for k in range(1, 5)]] > 0).sum(1)
E["react_mean4"] = E[[f"react_l{k}" for k in range(1, 5)]].mean(1)
E["react_abs4"] = E[[f"react_l{k}" for k in range(1, 5)]].abs().mean(1)
R1, J = E.R.values - 1, E.j.values
cum = lr.fillna(0).cumsum().values
for n in [5, 20, 60]:
    E[f"ret{n}"] = cum[R1, J] - cum[R1 - n, J]
E["vol20"] = lr.rolling(20, min_periods=10).std().values[R1, J]
E["ladv"] = np.log(adv.values[R1, J])
E["lpx"] = np.log(px.values[R1, J])
E["n_est"] = pd.to_numeric(E.noOfEsts, errors="coerce")
E["bmo"] = (E.timing == "BMO").astype(float)
roll = lambda k, w: F[k].reindex(index=days, columns=cols).fillna(0).rolling(w, min_periods=1).sum().values[R1, J]
E["news30"] = roll("n_news", 21)
sent_sum = (F["sent"].reindex(index=days, columns=cols).fillna(0) * F["n_news"].reindex(index=days, columns=cols).fillna(0))
E["sent30"] = sent_sum.rolling(21, min_periods=1).sum().values[R1, J] / np.maximum(E.news30.values, 1)
E["up30"], E["down30"] = roll("ev_up", 21), roll("ev_down", 21)
if "short_z" in F:
    E["short_z"] = F["short_z"].reindex(index=days, columns=cols).values[R1, J]
ins = pd.read_parquet(os.path.join(DATA, "local", "insider.parquet"))
ins = ins[ins.ticker.isin(cols) & (ins.value > 0)]
ins["di"] = days.searchsorted(ins.filed) + 1
ins = ins[ins.di < len(days)]
for code, nm in [("P", "ins_buy90"), ("S", "ins_sell90")]:
    x = ins[ins.code == code].groupby(["di", "ticker"]).owner.nunique().unstack()
    x = x.reindex(index=range(len(days)), columns=cols).fillna(0).rolling(63, min_periods=1).sum().values
    E[nm] = x[R1, J]
sec = F["_sector"]
E["sector"] = E.symbol.map(sec).astype("category").cat.codes
feat = [k for k in E.columns if k.startswith(("surp_", "react_", "ret", "vol20", "ladv", "lpx", "n_est", "bmo",
                                              "news30", "sent30", "up30", "down30", "short_z", "ins_", "sector"))]
print("reports", len(E), "features", len(feat), flush=True)
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
E["d"] = days[E.R.values]
preds = []
for Y in range(2022, 2027):
    cut = days[days.searchsorted(pd.Timestamp(f"{Y}-01-01")) - 10]
    tr = E.d < cut
    te = (E.d >= f"{Y}-01-01") & (E.d <= f"{Y}-12-31")
    if te.sum() == 0:
        continue
    mdl = lgb.train(params, lgb.Dataset(E.loc[tr, feat], E.loc[tr, "y"].clip(-0.3, 0.3)), num_boost_round=300)
    preds.append(pd.Series(mdl.predict(E.loc[te, feat]), index=E.index[te]))
    if Y == 2026:
        imp = pd.Series(mdl.feature_importance("gain"), index=feat).sort_values(ascending=False)
E["pred"] = pd.concat(preds)
T = E.dropna(subset=["pred"]).copy()
cost = (exec_cost_bps(P, "auction")[cols] + 2.5).values
T["cost"] = 2 * cost[T.R.values - 1, T.j.values] / 1e4
T["raw"] = night.values[T.R.values, T.j.values]
rows = []
spy = P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1
T["rk"] = T.groupby("d").pred.rank(ascending=False, method="first")
T["rnd"] = T.groupby("d").pred.transform(lambda x: pd.Series(np.random.default_rng(0).permutation(len(x)) + 1, index=x.index))
for name, m in [("top_pred_10", T.rk <= 10), ("top_pred_pos_only", (T.rk <= 10) & (T.pred > 0)),
                ("all_reports", T.rnd <= 10)]:
    pnl = ((T.raw - T.cost) * 0.1)[m].groupby(T.d[m]).sum()
    pnl = pnl.reindex(days, fill_value=0.0).loc["2022-01-01":days[-2]]
    for per, a, b in [("2022-23", "2022-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
        x = pnl.loc[a:b]
        st = ann_stats(x)
        tt = T[(T.d >= pd.Timestamp(a)) & (T.d < pd.Timestamp(b) + pd.offsets.MonthBegin(1))]
        rows.append(dict(rule=name, period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                         corr_spy=float(x.corr(spy.reindex(x.index))),
                         rank_ic=float(tt[["pred", "y"]].corr("spearman").iloc[0, 1]),
                         sign_hit=float((np.sign(tt.pred) == np.sign(tt.y)).mean()), n_reports=len(tt)))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study18_earnings_predict.csv", index=False)
print(df.round(3).to_string())
print((imp / imp.sum()).head(15).round(3).to_string())
