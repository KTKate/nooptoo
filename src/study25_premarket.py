"""Study 25: day session traded only at the auctions (buy/short at the opening auction, exit at the closing
auction) with signals from premarket data known at 09:25 ET (m5pre, 5-minute SIP bars 07:00-09:25).

Opening auction price = Yahoo adjusted open P["o"], closing auction = P["c"]. Cost per side =
core.exec_cost_bps(P, "auction") + 2.5 bp. Shorts: traded price > $10, 2 bp borrow per day when ADV < $50M.
Portfolio: up to K=10 slots of 10% each (rest cash). Validation 2024-01..2025-06, holdout 2025-07..latest.

    python src/study25_premarket.py            # writes results/study25_*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import store
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, RES, DATA

K = 10
VAL = ("2024-01-01", "2025-06-30")
HOLD = ("2025-07-01", "2026-12-31")

P = load_panel()
days_all = P["c"].index
days = days_all[days_all >= "2024-01-02"]
cols = sorted(set(stock_cols(P)) & set(A.read("m5snap", columns=["ticker"]).ticker.unique()))
prev_day = pd.Series(days_all[:-1], index=days_all[1:])


def panel(k):
    return P[k][cols]


# ------------------------------------------------------------------ Alpaca reference prices (m5snap)
def snap_refs():
    rows = []
    for f in sorted(os.listdir(os.path.join(DATA, "local", "m5snap"))):
        d = pd.read_parquet(os.path.join(DATA, "local", "m5snap", f), columns=["ts", "ticker", "o", "c"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        a = d[hm == 1555][["ts", "ticker", "c"]].rename(columns={"c": "c1555"})
        b = d[hm == 930][["ts", "ticker", "o"]].rename(columns={"o": "o930"})
        a["date"], b["date"] = a.ts.dt.normalize(), b.ts.dt.normalize()
        rows.append(a.drop(columns="ts").merge(b.drop(columns="ts"), on=["ticker", "date"], how="outer"))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ premarket features
def pm_features():
    out = []
    for f in sorted(f for f in os.listdir(os.path.join(DATA, "local", "m5pre")) if f.endswith(".parquet")):
        d = pd.read_parquet(os.path.join(DATA, "local", "m5pre", f))
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[hm <= 920].copy()                         # bars that end by 09:25
        d["date"] = d.ts.dt.normalize()
        d["dvb"] = d.v * d.vwap.astype("float64")
        d["late"] = np.where(d.ts.dt.hour * 100 + d.ts.dt.minute <= 900, d.c, np.nan)
        g = d.groupby(["ticker", "date"], sort=False)
        x = g.agg(pm_first=("o", "first"), pm_last=("c", "last"), pm_hi=("h", "max"), pm_lo=("l", "min"),
                  pm_dv=("dvb", "sum"), pm_n=("n", "sum"), pm_bars=("c", "size"), pm_900=("late", "last"),
                  pm_tlast=("ts", "last")).reset_index()
        out.append(x)
        print("pm", f, len(x), flush=True)
    return pd.concat(out, ignore_index=True)


def news_features():
    parts = []
    for f in sorted(os.listdir(os.path.join(DATA, "local", "news_scored"))):
        if f[:7] < "2023-12":
            continue
        n = pd.read_parquet(os.path.join(DATA, "local", "news_scored", f), columns=["ts", "symbols", "sent"])
        n["ts"] = n.ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
        n["ticker"] = n.symbols.str.split("|")
        n = n.explode("ticker")
        parts.append(n[n.ticker.isin(cols)][["ts", "ticker", "sent"]])
    n = pd.concat(parts, ignore_index=True)
    cut = days + pd.Timedelta(hours=9, minutes=25)                     # decision time per day
    start = pd.DatetimeIndex(prev_day.reindex(days).values) + pd.Timedelta(hours=16)
    i = np.searchsorted(cut.values, n.ts.values, side="left")       # first day whose cutoff >= ts
    ok = i < len(days)
    n, i = n[ok], i[ok]
    keep = n.ts.values > start.values[i]
    n = n[keep].assign(date=days.values[i[keep]])
    g = n.groupby(["ticker", "date"]).sent
    return pd.DataFrame({"n_news": g.size(), "sent_mean": g.mean(), "sent_max": g.max(),
                         "sent_min": g.min()}).reset_index()


def build():
    pm = pm_features()
    ref = snap_refs()
    ref = ref.sort_values(["ticker", "date"])
    # previous trading day's 15:55 close
    pc = ref[["ticker", "date", "c1555"]].dropna()
    pc = pc[pc.date.isin(prev_day.values)]
    pc["date"] = pd.to_datetime(pc.date.map(pd.Series(prev_day.index, index=prev_day.values)))
    pc = pc.rename(columns={"c1555": "prevc"})
    x = pm.merge(pc, on=["ticker", "date"], how="inner").merge(ref[["ticker", "date", "o930"]], on=["ticker", "date"], how="left")
    # scale consistency m5pre vs m5snap (fetched at different dates: later splits): per ticker-month median
    x["mon"] = x.date.dt.strftime("%Y-%m")
    rr = (x.pm_last / x.o930).groupby([x.ticker, x.mon]).transform("median")
    fix = (rr < 0.8) | (rr > 1.25)
    for k in ["pm_first", "pm_last", "pm_hi", "pm_lo", "pm_900"]:
        x.loc[fix, k] = x.loc[fix, k] / rr[fix]
    print("ticker-days rescaled", int(fix.sum()), flush=True)
    x = x[x.date.isin(days)]
    x["gap"] = x.pm_last / x.prevc - 1
    x["pm_trend"] = x.pm_last / x.pm_first - 1
    x["pm_range"] = (x.pm_hi - x.pm_lo) / x.prevc
    x["pm_late"] = x.pm_last / x.pm_900 - 1
    x["pm_mins"] = (x.date + pd.Timedelta(hours=9, minutes=25) - x.pm_tlast).dt.total_seconds() / 60
    # daily panel features known before the open of t
    dvm = panel("dv").rolling(20, min_periods=10).mean().shift(1)
    c = panel("c")
    feats = {"adv20": dvm, "ret1": (c / c.shift(1) - 1).shift(1), "ret5": (c / c.shift(5) - 1).shift(1),
             "vol20": np.log(c / c.shift(1)).rolling(20, min_periods=10).std().shift(1),
             "px": traded_close(P)[cols].shift(1),
             "ygap": panel("o") / c.shift(1) - 1,                                   # sanity only, NOT a feature
             "R": panel("c") / panel("o") - 1,
             "cost": exec_cost_bps(P, "auction")[cols] + 2.5}
    for k, v in feats.items():
        s = v.reindex(days).stack()
        s.index.names = ["date", "ticker"]
        x = x.merge(s.rename(k).reset_index(), on=["ticker", "date"], how="left")
    x["relvol"] = x.pm_dv / x.adv20
    nw = news_features()
    x = x.merge(nw, on=["ticker", "date"], how="left")
    x["n_news"] = x.n_news.fillna(0)
    e = store.read("earnings")
    e = e[e.symbol.isin(cols)][["symbol", "date", "time"]].rename(columns={"symbol": "ticker"})
    e["date"] = pd.to_datetime(e.date).dt.normalize()
    x["earn_t"] = x.set_index(["ticker", "date"]).index.isin(pd.MultiIndex.from_frame(e[["ticker", "date"]])).astype("int8")
    ep = e.copy()
    ep = ep[ep.date.isin(prev_day.values)]
    ep["date"] = pd.to_datetime(ep.date.map(pd.Series(prev_day.index, index=prev_day.values)))
    x["earn_prev"] = x.set_index(["ticker", "date"]).index.isin(pd.MultiIndex.from_frame(ep[["ticker", "date"]])).astype("int8")
    x["earn"] = ((x.earn_t + x.earn_prev) > 0).astype("int8")
    x["mkt_gap"] = x.groupby("date").gap.transform("median")
    x["gap_rel"] = x.gap - x.mkt_gap
    x["gap_z"] = x.gap / x.vol20
    # data-consistency filter: premarket gap vs Yahoo open gap more than 30%/43% apart -> price-scale mismatch
    ratio = (1 + x.gap) / (1 + x.ygap)
    bad = (ratio < 0.7) | (ratio > 1 / 0.7)
    print("scale mismatch rows dropped", int(bad.sum()), "of", len(x), flush=True)
    x = x[~bad]
    x = x.drop(columns=["mon", "pm_tlast"])
    return x


# ------------------------------------------------------------------ portfolio evaluation
def evaluate(sel, side):
    """sel: rows of x selected (<= K per day, already ranked). Fixed 1/K slot weight."""
    s = sel.copy()
    gross = side * s.R.fillna(0.0)
    cost = 2 * s.cost / 1e4
    if side < 0:
        cost = cost + np.where(s.adv20 < 5e7, 2e-4, 0.0)
    s["gross"], s["costr"] = gross, cost
    s["net"] = gross - cost
    daily = s.groupby("date")[["gross", "costr", "net"]].sum() / K
    daily = daily.reindex(days_eval).fillna(0.0)
    return s, daily


def pick(x, mask, key, side):
    if side < 0:
        mask = mask & (x.px > 10)
    s = x[mask].copy()
    s["_k"] = key[mask]
    return s.sort_values(["date", "_k"], ascending=[True, False]).groupby("date").head(K)


def lgbm_preds(x, feat):
    import lightgbm as lgb
    x = x.copy()
    x["q"] = pd.PeriodIndex(x.date, freq="Q")
    x["y"] = x.groupby("date").R.rank(pct=True) - 0.5
    out = []
    for q in sorted(x.q.unique())[1:]:
        tr = x[(x.q < q) & x.y.notna()]
        te = x[x.q == q]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              n_jobs=1, verbose=-1)
        m.fit(tr[feat].astype("float32"), tr.y)
        out.append(pd.Series(m.predict(te[feat].astype("float32")), index=te.index))
        ic = pd.DataFrame({"p": out[-1], "y": te.y}).groupby(te.date).apply(lambda d: d.p.corr(d.y, method="spearman")).mean()
        print("lgbm", q, len(tr), f"IC {ic:.4f}", flush=True)
    return pd.concat(out).reindex(x.index) if out else pd.Series(np.nan, index=x.index)


if __name__ == "__main__":
    ff = os.path.join(DATA, "local", "m5pre", "_study25_features.pkl")      # cache (not *.parquet)
    if os.path.exists(ff) and os.environ.get("REBUILD") != "1":
        x = pd.read_pickle(ff)
    else:
        x = build()
        x.to_pickle(ff)
    days_eval = days[(days >= x.date.min()) & (days <= x.date.max())]
    print("rows", len(x), "days", len(days_eval), "gap vs yahoo open gap: corr",
          round(x.gap.corr(x.ygap), 3), "median abs diff bp", round((x.gap - x.ygap).abs().median() * 1e4, 1), flush=True)
    x = x[(x.adv20 > 2e6) & (x.px > 3) & x.R.notna() & x.cost.notna() & x.gap.notna()].reset_index(drop=True)
    ag = x.gap.abs()
    news, sent = x.n_news > 0, x.sent_mean.fillna(0)
    quiet = (x.n_news == 0) & (x.earn == 0)
    V = {}
    for g in [0.02, 0.04, 0.08]:
        for rv in [0.02, 0.05]:
            V[f"go_long_g{g}_rv{rv}"] = (pick(x, (x.gap > g) & (x.relvol > rv) & news, ag, 1), 1)
            V[f"down_cont_short_g{g}_rv{rv}"] = (pick(x, (x.gap < -g) & (x.relvol > rv) & news, ag, -1), -1)
        V[f"up_news_short_g{g}"] = (pick(x, (x.gap > g) & news, ag, -1), -1)
        V[f"fade_up_short_g{g}"] = (pick(x, (x.gap > g) & quiet, ag, -1), -1)
        V[f"fade_down_long_g{g}"] = (pick(x, (x.gap < -g) & quiet, ag, 1), 1)
        V[f"down_rev_long_g{g}"] = (pick(x, (x.gap < -g) & news, ag, 1), 1)
    for g in [0.02, 0.04]:
        V[f"sent_up_pos_long_g{g}"] = (pick(x, (x.gap > g) & (sent > 0.5), ag, 1), 1)
        V[f"sent_up_neg_short_g{g}"] = (pick(x, (x.gap > g) & news & (sent < -0.3), ag, -1), -1)
        V[f"sent_down_neg_short_g{g}"] = (pick(x, (x.gap < -g) & news & (sent < -0.3), ag, -1), -1)
        V[f"sent_down_pos_long_g{g}"] = (pick(x, (x.gap < -g) & (sent > 0.5), ag, 1), 1)
    feat = ["gap", "gap_rel", "gap_z", "mkt_gap", "relvol", "pm_range", "pm_trend", "pm_late", "pm_mins", "pm_n",
            "pm_bars", "n_news", "sent_mean", "sent_max", "sent_min", "earn_t", "earn_prev", "ret1", "ret5",
            "vol20", "adv20", "px"]
    x["pred"] = lgbm_preds(x, feat)
    xm = x[x.pred.notna()]
    for k in [5, 10]:
        V[f"lgbm_long_k{k}"] = (pick(xm, xm.pred.notna(), xm.pred, 1).groupby("date").head(k), 1)
        V[f"lgbm_short_k{k}"] = (pick(xm, xm.pred.notna() & (xm.px > 10), -xm.pred, -1).groupby("date").head(k), -1)
    # robustness of the LGBM short leg: liquid names only (easy to borrow proxy), no SSR days (prior day <= -10%)
    liq = xm.pred.notna() & (xm.px > 10) & (xm.adv20 > 5e7) & (xm.ret1 > -0.1)
    V["lgbm_short_k10_liq50m"] = (pick(xm, liq, -xm.pred, -1), -1)
    V["lgbm_short_k10_liq200m"] = (pick(xm, liq & (xm.adv20 > 2e8), -xm.pred, -1), -1)
    V["lgbm_long_k10_liq50m"] = (pick(xm, liq, xm.pred, 1), 1)
    pk = pd.concat([V[k][0].assign(variant=k) for k in ["lgbm_short_k10", "lgbm_long_k10", "lgbm_short_k10_liq50m"]])
    pk[["variant", "date", "ticker", "pred", "gap", "relvol", "n_news", "earn", "px", "adv20", "ret1", "R", "cost"]].to_csv(
        os.path.join(RES, "study25_picks.csv"), index=False)
    ovn = pd.read_parquet(os.path.join(RES, "final_series.parquet"))["ml_overnight_k10"]
    rows, series = [], {}
    for name, (sel, side) in V.items():
        s, d = evaluate(sel, side)
        series[name] = d.net
        for per, (a, b) in [("val", VAL), ("hold", HOLD), ("all", (VAL[0], HOLD[1]))]:
            dd, ss = d.loc[a:b], s[(s.date >= a) & (s.date <= b)]
            st = ann_stats(dd.net)
            r = dd.net
            rows.append(dict(variant=name, period=per, trades=len(ss), days_traded=int((dd.gross != 0).sum()),
                             sharpe=st["sharpe"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                             net_bp=ss.net.mean() * 1e4, gross_bp=ss.gross.mean() * 1e4, cost_bp=ss.costr.mean() * 1e4,
                             corr_overnight=r.corr(ovn.reindex(r.index)),
                             dsr=deflated_sharpe(r.mean() / r.std(), len(V) + 1100, len(r), r.skew(), r.kurt() + 3)
                             if r.std() > 0 else np.nan))
    # long-short LGBM combos
    for k in [5, 10]:
        r = (series[f"lgbm_long_k{k}"] + series[f"lgbm_short_k{k}"]) / 2
        series[f"lgbm_ls_k{k}"] = r
        for per, (a, b) in [("val", VAL), ("hold", HOLD), ("all", (VAL[0], HOLD[1]))]:
            rr = r.loc[a:b]
            st = ann_stats(rr)
            rows.append(dict(variant=f"lgbm_ls_k{k}", period=per, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                             maxdd=st["maxdd"], corr_overnight=rr.corr(ovn.reindex(rr.index)),
                             dsr=deflated_sharpe(rr.mean() / rr.std(), len(V) + 2 + 1100, len(rr), rr.skew(), rr.kurt() + 3)))
    res = pd.DataFrame(rows)
    res["n_variants"] = len(V) + 2
    res.to_csv(os.path.join(RES, "study25_variants.csv"), index=False)
    pd.DataFrame(series).to_csv(os.path.join(RES, "study25_series.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    w = res.pivot(index="variant", columns="period", values=["sharpe", "net_bp", "trades"])
    print(w.round(2).sort_values(("sharpe", "val"), ascending=False))
