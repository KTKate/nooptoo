"""Study 77 (Part A): models trained ONLY on earnings events, small leaves, walk-forward by quarter.

Question (owner): inputs with no signal alone may matter in combination around rare events. The pooled overnight
models (min_data_in_leaf 2000, all stock-days) cannot learn event-specific combinations, and studies 41/42/74 looked
at one input at a time.

Events and timing: study 74 exactly (src/study74_earnings_day.py imported: Nasdaq calendar 2020-01..2026-09-25,
gap-rule timing, liquid universe on the traded session t, which is the session immediately before the reaction
overnight). Two targets per event:
  pre      day session of t: opening auction -> closing auction, excess vs the equal-weight liquid universe's
           open-to-close (study 74's 'ex'). Entry at the open of t.
  through  close of t -> open of t+1 (the reaction overnight), excess vs the universe's overnight mean (|r| < 50%).
           Entry at the closing auction of t.
Caveat built into the event set: the gap rule keeps only events whose reaction overnight moved >= 1% and >= 2x the
other overnight, i.e. it selects on the MAGNITUDE of the 'through' target (not its sign). Robustness: the same models
on the headline-timed events (first Benzinga EPS headline; no selection on reaction size).
Inputs (all known before the entry):
  price      data/ml_frame.parquet row of t-1 (pre) or of t (through: close-of-t features, the convention of the
             overnight backtests; live they would be 15:45 values) - 53 columns; plus for pre the stock's opening gap
             on t and SPY's opening gap (known at the open).
  news       sent5, n_news5 (5 windows before the window of t), sent_prev (window ending 15:45 t-1);
             through also sent / n_news of today's window (ending 15:45 of t).
  analyst    pt_net20, pt_gap, pt_gap_chg20, ev_guid_up20, ev_guid_dn20, pt_firms at t-1 (lagged one day).
  fundament. percentile ranks (cross-section of the day) of market_cap, ev_sales, pe_ttm, fcf_yield, sbc_to_revenue,
             op_margin, net_margin, revenue_growth_yoy, book_to_market, cash_to_mcap at t-1;
             days_since_filing > 200 -> missing.
  prev rep.  previous calendar report 30-120 trading days earlier: EPS surprise %, beat sign, 2-day reaction.
  sector     Nasdaq sector code (categorical; current classification, mild look-ahead).
  insider    SEC Form 4 open-market purchases (P) minus sales (S) in $ with filing date in the 30 / 90 calendar
             days before the date of t (a filing dated d is usable from the next trading day), / 20d median dollar
             volume; number of distinct buying owners over 90 days. Data end 2026-03-31: missing after.
  timing     AMC / BMO (published in advance; here recovered).
Model: LightGBM regression, objective l2, num_leaves 15, min_data_in_leaf 100, learning_rate 0.03, 200 rounds,
feature_fraction 0.7, bagging 0.7, lambda_l2 10, min_gain_to_split 0 (pre-specified, not tuned).
Targets: within-quarter percentile rank of the excess return - 0.5 (primary) or raw excess clipped at the 1/99%
of the training data. Walk-forward by quarter: train on all events with t at least 3 trading days before the
quarter, test when >= 3,000 training events.
Variants: sample (gap timing; headline timing) x window (pre; through) x inputs (price only; full) x target (rank;
raw, gap sample only), plus 'through' with full inputs but price rows of t-1 (full_lag1: no close-of-t prices in the
inputs, guards against closing-auction microstructure) and, as a baseline, the current pooled overnight models
(data/models/night_<q>.txt, 2024Q1+) scored on the same events.
Evaluation: rank IC per test quarter, mean within-day IC (days with >= 5 events) (Spearman of prediction vs excess return); books: each day with >= 5 events take
the top / bottom 20% of that day's predictions (at most 10 names), 10% weight each, cash otherwise; shorts only
in names on Alpaca's current easy-to-borrow list (today's list: flatters the past); costs per side
core.exec_cost_bps(P,'auction') + 2.5 bp (both legs are auctions). Hedged = book minus the universe mean with the
same weights.
Output: results/study77_event_models.csv (sections: data, ic_q, ic, book, book_year, importance, count)
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import study74_earnings_day as S
from core import ann_stats, RES, DATA

OUT = os.path.join(RES, "study77_event_models.csv")
ROWS = []
S.h = S.isres = S.N = None                         # free study 74's headline frames
alld, cols, univ, Ov, Cv = S.alld, S.cols, S.univ, S.Ov, S.Cv
PER = [("2020-21", "2020-01-01", "2021-12-31"), ("2022-23", "2022-01-01", "2023-12-31"),
       ("2024-26", "2024-01-01", "2026-09-30")]
PR = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
          num_threads=int(os.environ.get("NTHREADS", "2")), seed=77)
NROUND = 200
FU = ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "net_margin",
      "revenue_growth_yoy", "book_to_market", "cash_to_mcap"]
INS_END = None


def emit(section, **kw):
    ROWS.append(dict(section=section, **kw))


# ------------------------------------------------------------------ overnight benchmark
Ron = S.o.shift(-1) / S.c - 1
mkt_on = Ron.where(univ & (Ron.abs() < 0.5)).mean(1)


def frame(timing_col):
    x = S.add_features(S.session_frame(timing_col))
    ti, jj = x.ti.values, x.j.values
    ok = ti + 1 < len(alld)
    x = x[ok].copy()
    ti, jj = x.ti.values, x.j.values
    x["r_on"] = Ov[ti + 1, jj] / Cv[ti, jj] - 1
    x["ex_on"] = x.r_on - mkt_on.values[ti]
    x["mkt_on"] = mkt_on.values[ti]
    x["q"] = pd.PeriodIndex(x.tdate, freq="Q")
    return x.reset_index(drop=True)


def add_inputs(x):
    """Adds price (pre_/thr_ prefixed ml_frame rows), news today, fundamentals, insider, codes."""
    global INS_END
    ti, jj = x.ti.values, x.j.values
    # ml_frame rows
    need = sorted(set(alld[ti - 1]) | set(alld[ti]))
    X = pd.read_parquet(os.path.join(DATA, "ml_frame.parquet"),
                        filters=[("date", ">=", min(need)), ("date", "<=", max(need)),
                                 ("ticker", "in", sorted(set(x.symbol)))])
    X = X[[k for k in X.columns if not k.startswith("y_")]]
    pf = list(X.columns)
    for tag, lag in [("pre_", 1), ("thr_", 0)]:
        k = pd.MultiIndex.from_arrays([alld[ti - lag], x.symbol.values])
        v = X.reindex(k).values
        for i, cname in enumerate(pf):
            x[tag + cname] = v[:, i]
    print("ml_frame coverage pre/thr", round(float(np.isfinite(x["pre_r1"]).mean()), 3),
          round(float(np.isfinite(x["thr_r1"]).mean()), 3), flush=True)
    del X
    # news today (window ending 15:45 of t) for the overnight target
    F = S.NF.load()
    fcols = F["sent"].columns
    fj = fcols.get_indexer(x.symbol)
    for k, nm in [("sent", "sent_today"), ("n_news", "n_news_today")]:
        v = F[k].reindex(index=alld, columns=fcols).values[ti, np.maximum(fj, 0)]
        x[nm] = np.where(fj >= 0, v, np.nan)
    del F
    # fundamentals: daily cross-sectional percentile ranks at t-1
    FP = pd.read_pickle(os.path.join(DATA, "local", "fundamentals_panels.pkl"))
    stale = FP["days_since_filing"].reindex(index=alld, columns=cols) > 200
    for k in FU:
        p = FP[k].reindex(index=alld, columns=cols).where(~stale).rank(axis=1, pct=True)
        x["fu_" + k] = p.values[ti - 1, jj]
        FP[k] = None
    del FP, stale, p
    # insider net buying (filing date strictly before the date of t)
    ins = pd.read_parquet(os.path.join(DATA, "local", "insider.parquet"))
    ins["ticker"] = ins.ticker.str.upper().str.replace(".", "-", regex=False)
    ins["sv"] = np.where(ins.code == "P", 1.0, -1.0) * (ins.shares.abs() * ins.price.abs())
    ins = ins[np.isfinite(ins.sv) & ins.ticker.isin(set(x.symbol))]
    ins["fd"] = ins.filed.dt.normalize()
    INS_END = ins.fd.max()
    D = x.tdate.values
    out = {k: np.zeros(len(x)) for k in ["ins_net30", "ins_net90", "ins_nbuy90"]}
    g = {t: grp.sort_values("fd") for t, grp in ins.groupby("ticker")}
    for t, idx in x.groupby("symbol").indices.items():
        if t not in g:
            continue
        gg = g[t]
        fd = gg.fd.values
        cs = np.concatenate([[0], np.cumsum(gg.sv.values)])
        d = D[idx]
        hi = np.searchsorted(fd, d, side="left")              # filings dated < date of t
        for w, nm in [(30, "ins_net30"), (90, "ins_net90")]:
            lo = np.searchsorted(fd, d - np.timedelta64(w, "D"), side="left")
            out[nm][idx] = cs[hi] - cs[lo]
        lo = np.searchsorted(fd, d - np.timedelta64(90, "D"), side="left")
        buy = (gg.code.values == "P")
        own = gg.owner.values
        out["ins_nbuy90"][idx] = [len(set(own[a:b][buy[a:b]])) for a, b in zip(lo, hi)]
    for k, v in out.items():
        x[k] = v
    x["ins_net30"] /= x.adv
    x["ins_net90"] /= x.adv
    late = x.tdate > INS_END
    x.loc[late, ["ins_net30", "ins_net90", "ins_nbuy90"]] = np.nan
    x["sector_code"] = x.sector.astype("category").cat.codes.replace(-1, np.nan) if x.sector.notna().any() else np.nan
    x["amc"] = (x.timing == "AMC").astype(float)
    return x, pf


def feature_sets(pf):
    common_news = ["sent5", "n_news5", "sent_prev"]
    an = ["pt_net20", "pt_gap", "pt_gap_chg20", "ev_guid_up20", "ev_guid_dn20", "pt_firms"]
    rest = ["fu_" + k for k in FU] + ["p_react", "p_surp", "p_beat", "sector_code", "ins_net30", "ins_net90",
                                      "ins_nbuy90", "amc"]
    return {
        ("pre", "price"): ["pre_" + k for k in pf] + ["gap_t", "spy_gap"],
        ("pre", "full"): ["pre_" + k for k in pf] + ["gap_t", "spy_gap"] + common_news + an + rest,
        ("through", "price"): ["thr_" + k for k in pf],
        ("through", "full"): ["thr_" + k for k in pf] + common_news + ["sent_today", "n_news_today"] + an + rest,
        # robustness: price rows of t-1 instead of the close of t (no close-auction microstructure in the inputs)
        ("through", "full_lag1"): ["pre_" + k for k in pf] + common_news + ["sent_today", "n_news_today"] + an + rest,
    }


def pooled_pred(x, pf):
    """Current pooled overnight models (data/models/night_<q>.txt, 2024Q1+, trained before each quarter) scored on
    the close-of-t rows: does the event model add anything over the general model on these events?"""
    p = pd.Series(np.nan, index=x.index)
    for q in sorted(x.q.unique()):
        fn = os.path.join(DATA, "models", f"night_{q}.txt")
        if not os.path.exists(fn):
            continue
        m = lgb.Booster(model_file=fn)
        te = (x.q == q).values
        p[te] = m.predict(x.loc[te, ["thr_" + k for k in m.feature_name()]].astype("float32").values)
    return p


def ic_daily(x, p, ycol, a0, b0):
    """Mean within-day Spearman IC over days with >= 5 events (what the daily book uses), t over days."""
    y = x.assign(p=p)[p.notna() & x[ycol].notna() & (x.tdate >= a0) & (x.tdate <= b0)]
    y = y[y.groupby("tdate").p.transform("size") >= 5]
    if y.empty:
        return np.nan, np.nan, 0
    r = y.groupby("tdate").apply(lambda g: g[ycol].rank().corr(g.p.rank()), include_groups=False).dropna()
    return r.mean(), r.mean() / r.std() * np.sqrt(len(r)), len(r)


def walk_forward(x, feats, ycol, target):
    pred = pd.Series(np.nan, index=x.index)
    imp = None
    for q in sorted(x.q.unique()):
        te = (x.q == q).values
        cut = alld[max(0, alld.searchsorted(q.start_time) - 3)]
        tr = (x.tdate < cut).values & x[ycol].notna().values
        if tr.sum() < 3000 or te.sum() == 0:
            continue
        xt = x[tr]
        if target == "rank":
            y = xt.groupby("q")[ycol].rank(pct=True) - 0.5
        else:
            lo, hi = xt[ycol].quantile([0.01, 0.99])
            y = xt[ycol].clip(lo, hi)
        cat = ["sector_code"] if "sector_code" in feats else "auto"
        ds = lgb.Dataset(xt[feats].astype("float32"), y.values, categorical_feature=cat, free_raw_data=True)
        m = lgb.train(PR, ds, num_boost_round=NROUND)
        pred[te] = m.predict(x.loc[te, feats].astype("float32"))
        imp = pd.Series(m.feature_importance("gain"), index=feats)
    return pred, imp


def ic_by_q(x, p, ycol):
    ok = p.notna() & x[ycol].notna()
    r = []
    for q, g in x[ok].groupby("q"):
        r.append(dict(q=str(q), date=q.start_time, n=len(g), ic=g[ycol].rank().corr(p[g.index].rank())))
    return pd.DataFrame(r)


def books(x, p, rcol, mcol):
    """Daily long / short / long-short books from predictions p (top / bottom 20% of the day's events, <= 10)."""
    y = x.assign(p=p)[p.notna() & x[rcol].notna()].copy()
    n = y.groupby("tdate").p.transform("size")
    rk = y.groupby("tdate").p.rank(method="first", ascending=False)
    rkl = y.groupby("tdate").p.rank(method="first", ascending=True)
    k = np.minimum(10, np.floor(0.2 * n))
    y["L"] = (rk <= k) & (k >= 1)
    y["Sh"] = (rkl <= k) & (k >= 1) & y.etb
    y["cost"] = 2 * y.cost_a / 1e4
    dd = alld[(alld >= y.tdate.min()) & (alld <= S.END)]
    res = {}
    for side, col, sg in [("long", "L", 1), ("short", "Sh", -1)]:
        z = y[y[col]]
        g = z.groupby("tdate")
        gross = (0.1 * sg * z[rcol]).groupby(z.tdate).sum().reindex(dd).fillna(0)
        cost = (0.1 * z.cost).groupby(z.tdate).sum().reindex(dd).fillna(0)
        hedge = (0.1 * sg * z[mcol]).groupby(z.tdate).sum().reindex(dd).fillna(0)
        res[side] = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "net_h": gross - cost - hedge,
                                  "n": g.size().reindex(dd).fillna(0),
                                  "tr_net": (sg * z[rcol] - z.cost).groupby(z.tdate).sum().reindex(dd).fillna(0),
                                  "tr_ex": (sg * (z[rcol] - z[mcol])).groupby(z.tdate).sum().reindex(dd).fillna(0)})
    res["ls"] = res["long"] + res["short"]
    return res


def main():
    nvar = 0
    for sample, tcol in [("gap", "t_gap"), ("news", "t_news")]:
        x = frame(tcol)
        x, pf = add_inputs(x)
        print(sample, "events", len(x), "quarters", x.q.nunique(), "insider data end", INS_END, flush=True)
        # data checks
        for ycol in ["ex", "ex_on"]:
            qq = x[ycol].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict()
            print(sample, ycol, qq, "n |.|>0.5:", int((x[ycol].abs() > 0.5).sum()), flush=True)
            emit("data", sample=sample, target=ycol, n=len(x), q001=qq[0.001], q999=qq[0.999],
                 n_abs_gt50=int((x[ycol].abs() > 0.5).sum()), mean_bp=1e4 * x[ycol].mean())
        big = x[x.ex_on.abs() > 0.5][["symbol", "tdate", "r_on", "g_pre", "g_post", "timing"]]
        print(big.head(15).to_string(), flush=True)
        FS = feature_sets(pf)
        for f, cols_ in FS.items():
            cov = np.isfinite(x[cols_].astype(float)).mean()
            print(sample, f, "inputs", len(cols_), "lowest coverage:", cov.sort_values().head(5).round(2).to_dict(),
                  flush=True)
        specs = []
        for (win, fs), feats in FS.items():
            for target in (["rank", "raw"] if sample == "gap" and fs != "full_lag1" else ["rank"]):
                specs.append((win, fs, target, feats))
        specs.append(("through", "pooled_night_model", "none", None))
        for win, fs, target, feats in specs:
            ycol, rcol, mcol = ("ex", "r", "mkt") if win == "pre" else ("ex_on", "r_on", "mkt_on")
            if True:
                nvar += 1
                tag = f"{sample}|{win}|{fs}|{target}"
                if feats is None:
                    p, imp = pooled_pred(x, pf), None
                else:
                    p, imp = walk_forward(x, feats, ycol, target)
                ic = ic_by_q(x, p, ycol)
                for _, r in ic.iterrows():
                    emit("ic_q", sample=sample, window=win, features=fs, target=target, quarter=r.q, n=r.n, ic=r.ic)
                b = books(x, p, rcol, mcol)
                for per, a0, b0 in PER:
                    icp = ic[(ic.date >= a0) & (ic.date <= b0)]
                    if len(icp) == 0:
                        continue
                    t_ic = icp.ic.mean() / icp.ic.std() * np.sqrt(len(icp)) if len(icp) > 1 else np.nan
                    icd, icd_t, nd = ic_daily(x, p, ycol, a0, b0)
                    emit("ic", sample=sample, window=win, features=fs, target=target, period=per, quarters=len(icp),
                         n=int(icp.n.sum()), ic_mean=icp.ic.mean(), ic_t=t_ic, ic_pos=(icp.ic > 0).mean(),
                         ic_daily=icd, ic_daily_t=icd_t, ic_days=nd)
                    line = [tag, per, f"IC {icp.ic.mean():.4f} (t {t_ic:.2f}, {len(icp)}q) dIC {icd:.4f} (t {icd_t:.2f})"]
                    for side in ["long", "short", "ls"]:
                        d = b[side].loc[a0:b0]
                        if d.n.sum() == 0:
                            continue
                        s, sh = ann_stats(d.net), ann_stats(d.net_h)
                        ntr = d.n.sum()
                        emit("book", sample=sample, window=win, features=fs, target=target, period=per, side=side,
                             trades=int(ntr), days_active=int((d.n > 0).sum()), avg_names=d.n[d.n > 0].mean(),
                             gross_bp_tr=1e4 * (d.tr_net.sum() + d.cost.sum() / 0.1) / ntr,
                             net_bp_tr=1e4 * d.tr_net.sum() / ntr, ex_bp_tr=1e4 * d.tr_ex.sum() / ntr,
                             cost_bp_rt=1e4 * (d.cost.sum() / 0.1) / ntr, ann_ret=s["ann_ret"], sharpe=s["sharpe"],
                             t=s["tstat"], maxdd=s["maxdd"], sharpe_hedged=sh["sharpe"], maxdd_hedged=sh["maxdd"])
                        line.append(f"{side}: net {1e4 * d.tr_net.sum() / ntr:.0f}bp/tr ex {1e4 * d.tr_ex.sum() / ntr:.0f}"
                                    f" SR {s['sharpe']:.2f} SRh {sh['sharpe']:.2f} n {int(ntr)}")
                    print(" | ".join(line), flush=True)
                for yr in range(2020, 2027):
                    for side in ["long", "short", "ls"]:
                        d = b[side].loc[str(yr)]
                        if d.n.sum() == 0:
                            continue
                        s_ = ann_stats(d.net)
                        emit("book_year", sample=sample, window=win, features=fs, target=target, period=str(yr),
                             side=side, trades=int(d.n.sum()), net_bp_tr=1e4 * d.tr_net.sum() / d.n.sum(),
                             ex_bp_tr=1e4 * d.tr_ex.sum() / d.n.sum(), sharpe=s_["sharpe"], ann_ret=s_["ann_ret"])
                if imp is not None and fs == "full" and target == "rank":
                    for rk, (k, v) in enumerate((imp / imp.sum()).sort_values(ascending=False).head(15).items()):
                        emit("importance", sample=sample, window=win, features=fs, target=target, rank=rk + 1,
                             feature=k, share=v)
                pd.DataFrame(ROWS).to_csv(OUT, index=False)
        # reference: all-event averages (excess) per period
        for per, a0, b0 in PER:
            y = x[(x.tdate >= a0) & (x.tdate <= b0)]
            for ycol in ["ex", "ex_on"]:
                m, t, n = S.cl_t(y[ycol], y.tdate)
                emit("data", sample=sample, target=ycol, period=per, n=n, mean_bp=1e4 * m, t_cl=t)
        del x
    emit("count", n=nvar, note="model variants (sample x window x features x target); each scored with IC + 3 books "
                               "x 3 periods; hyperparameters pre-specified, not tuned")
    pd.DataFrame(ROWS).to_csv(OUT, index=False)
    print("variants", nvar, "saved", OUT, flush=True)


if __name__ == "__main__":
    main()
