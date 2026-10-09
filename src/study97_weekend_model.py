"""Study 97: does a weekend-specific model pick better stocks for holds over a weekend (Friday close to Monday open)?

Question: study 65 found Monday-night entries weakest and Friday-close-to-Monday-open holds span a weekend. Does a
rank model trained only on nights before a non-trading day, or the study-3 inputs plus a weekend flag and
interactions, pick better stocks for those nights than the blend (study 33) or the plain pooled model? And does
simply skipping Friday or Monday nights help the blend?
Data: data/ml_frame.parquet (study-3 close features, target y_night = close -> next open), results/study23_pred.parquet
and results/study33_pred.parquet (15:45 blend scores, 2024-01..2026-09), daily panel (core.load_panel).
Design:
  * "Weekend night" = calendar gap from day t to the next trading day >= 3 days (Fridays, and days before a market
    holiday); known in advance from the exchange calendar.
  * Part A: blend (top 10 by (2 x ensemble rank + jump-minus-drop rank) / 3, closing-auction buy, opening-auction
    sell, auction costs + 2.5 bp per side) mean net return by night type and weekday, per period.
  * Part B: LightGBM rank models (study 3 single-model params, 300 rounds, num_threads=2), walk-forward by quarter,
    test 2022Q1..2026Q3, training on all rows before the quarter start minus a 10-day embargo:
      base    all nights, study-3 inputs (these already include day of week 'dow' and month end 'tom')
      wkflag  all nights, + weekend flag, gap days and 6 flag x input interactions (r1, r5, vol20, night1, beta,
              days_to_e)
      wkonly  weekend nights only, study-3 inputs
    Top-10 portfolios on weekend nights, net of the same auction costs, compared with paired t-tests on the daily net
    returns: wkflag and wkonly vs base (2022-01..2026-09, all on close features), and vs the blend's weekend picks
    (2024-01..2026-09; the blend is scored at 15:45, the models on close features, so this comparison favours the
    models slightly). A 50/50 rank mix of blend and wkonly on weekend nights is also tried.
  * Part C: the blend with Friday nights or Monday nights (or weekend/pre-holiday nights) skipped; rule chosen on
    2024-01..2025-06, checked on 2025-07..2026-09.
Output: results/study97_weekend_model.csv (columns part, variant, period, n, mean_bp, sharpe, diff_bp, t_diff ...);
model predictions cached in data/local/study97_pred.parquet (delete it to retrain).
    python src/study97_weekend_model.py          # trains if the cache is missing, then evaluates
"""
import os
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import bt

PRED_FN = f"{DATA}/local/study97_pred.parquet"
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=2)
INTER = ["r1", "r5", "vol20", "night1", "beta", "days_to_e"]
PER = [("2022-23", "2022-01", "2023-12"), ("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]

P = load_panel()
days = P["c"].index
cols = stock_cols(P)
nxt = pd.Series(days[1:].append(pd.DatetimeIndex([pd.NaT])), index=days)
GAP = (nxt - pd.Series(days, index=days)).dt.days            # calendar days to the next session (NaN on last day)


def train_all():
    import lightgbm as lgb
    f = pq.ParquetFile(f"{DATA}/ml_frame.parquet")
    names = [n for n in f.schema_arrow.names if n not in ("date", "ticker")]
    feat = [n for n in names if not n.startswith("y_")]
    t = f.read(columns=feat + ["y_night", "date", "ticker"])
    dates = pd.DatetimeIndex(t.column("date").to_pandas())
    keep = np.asarray(dates >= "2020-01-01")
    tick = t.column("ticker").to_pandas().values[keep]
    y = t.column("y_night").to_numpy(zero_copy_only=False)[keep].astype("float32")
    X = np.empty((keep.sum(), len(feat)), dtype="float32")
    for j, n in enumerate(feat):
        X[:, j] = t.column(n).to_numpy(zero_copy_only=False)[keep]
    del t
    gc.collect()
    dates = dates[keep]
    gap = GAP.reindex(dates).values.astype("float32")
    wk = (gap >= 3).astype("float32")
    wk[np.isnan(gap)] = np.nan
    extra = [wk, gap] + [X[:, feat.index(k)] * wk for k in INTER]
    feat_wk = feat + ["wk", "gap"] + [f"wk_{k}" for k in INTER]
    ok = ~np.isnan(y)
    yr = pd.Series(y).groupby(dates.values).rank(pct=True).values.astype("float32") - 0.5
    print("rows", len(y), "weekend share", np.nanmean(wk), flush=True)
    done = pd.read_parquet(PRED_FN) if os.path.exists(PRED_FN) else None
    out = [done] if done is not None else []
    have_q = set(done["q"]) if done is not None else set()
    for q in pd.period_range("2022Q1", "2026Q3", freq="Q"):
        if str(q) in have_q:
            continue
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & np.asarray(dates < cut)
        te = np.asarray((dates >= q.start_time) & (dates <= q.end_time))
        res = pd.DataFrame({"date": dates[te], "ticker": tick[te], "q": str(q), "wk": wk[te]})
        for name in ["base", "wkflag", "wkonly"]:
            if name == "wkflag":
                Xtr = np.hstack([X[tr]] + [e[tr, None] for e in extra])
                Xte = np.hstack([X[te]] + [e[te, None] for e in extra])
                fn = feat_wk
            else:
                m = tr & (wk == 1) if name == "wkonly" else tr
                Xtr, Xte, fn = X[m], X[te], feat
            ytr = yr[tr & (wk == 1)] if name == "wkonly" else yr[tr]
            ds = lgb.Dataset(Xtr, ytr, feature_name=fn, free_raw_data=True)
            mdl = lgb.train(PARAMS, ds, num_boost_round=300)
            res[name] = mdl.predict(Xte).astype("float32")
            print(q, name, "train", len(ytr), "test", int(te.sum()), flush=True)
            del ds, mdl, Xtr, Xte
            gc.collect()
        out.append(res)
        pd.concat(out, ignore_index=True).to_parquet(PRED_FN, index=False)
    return pd.concat(out, ignore_index=True)


def stats_row(part, variant, period, x, ref=None):
    x = x.dropna()
    d = dict(part=part, variant=variant, period=period, n=len(x), mean_bp=1e4 * x.mean(),
             sharpe=ann_stats(x)["sharpe"] if len(x) > 5 else np.nan,
             t_mean=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan)
    if ref is not None:
        dd = (x - ref.reindex(x.index)).dropna()
        d.update(n_paired=len(dd), diff_bp=1e4 * dd.mean(),
                 t_diff=dd.mean() / dd.std() * np.sqrt(len(dd)) if len(dd) > 2 and dd.std() > 0 else np.nan)
    return d


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pred = pd.read_parquet(PRED_FN) if os.path.exists(PRED_FN) else None
    if pred is None or pred.q.nunique() < 19:
        pred = train_all()
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.set_index(["date", "ticker"])
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).astype("float32")
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).astype("float32")
    rows = []

    # ---- blend (study 33 / 69)
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    p33 = pd.read_parquet(f"{RES}/study33_pred.parquet")
    pj = p33.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = p33.p_drop.unstack().reindex(index=ens.index, columns=cols)
    okb = ens.notna() & pj.notna()
    SB = (2 * ens.where(okb).rank(axis=1, pct=True) + (pj - pdr).where(okb).rank(axis=1, pct=True)) / 3
    del p33, pj, pdr
    WB = bt.select_topk(SB, SB.notna(), 10)
    blend = bt.run(WB, R.reindex_like(WB), C.reindex_like(WB)).net
    blend = blend.loc[:"2026-09"]

    # ---- Part A: blend by night type
    idx = blend.index
    gap = GAP.reindex(idx)
    typ = pd.Series(np.where(gap >= 3, np.where(idx.weekday == 4, "Friday (weekend)", "pre-holiday weekday"),
                             "next-day night"), index=idx)
    for lab, g in [("night type", typ), ("weekday", pd.Series(idx.day_name(), index=idx))]:
        for p, a, b in PER[1:]:
            x, gg = blend.loc[a:b], g.loc[a:b]
            for lev, v in x.groupby(gg):
                rows.append(stats_row("A blend by " + lab, lev, p, v))
    # weekend nights vs other nights, difference in means (Welch t)
    for p, a, b in PER[1:]:
        x, w = blend.loc[a:b], (gap.loc[a:b] >= 3)
        u, v = x[w], x[~w]
        rows.append(dict(part="A blend weekend minus other nights", variant="gap>=3 minus gap<3", period=p,
                         n=len(u), diff_bp=1e4 * (u.mean() - v.mean()),
                         t_diff=(u.mean() - v.mean()) / np.sqrt(u.var() / len(u) + v.var() / len(v))))

    # ---- Part B: weekend-night top-10 portfolios
    S = {k: pred[k].unstack().reindex(columns=cols) for k in ["base", "wkflag", "wkonly"]}
    wkd = pred["wk"].groupby(level=0).first()
    wkdays = wkd.index[wkd == 1]
    nets = {}
    for k, s in S.items():
        s = s.loc[s.index.isin(wkdays)]
        W = bt.select_topk(s, s.notna(), 10)
        nets[k] = bt.run(W, R.reindex_like(W), C.reindex_like(W)).net
    sb = SB.loc[SB.index.isin(wkdays)]
    nets["blend"] = blend.reindex(sb.index)
    # 50/50 rank mix of the blend score and the weekend-only model (2024+)
    wo = S["wkonly"].reindex(index=sb.index)
    okm = sb.notna() & wo.notna()
    mix = (sb.where(okm).rank(axis=1, pct=True) + wo.where(okm).rank(axis=1, pct=True)) / 2
    Wm = bt.select_topk(mix, mix.notna(), 10)
    nets["blend+wkonly mix"] = bt.run(Wm, R.reindex_like(Wm), C.reindex_like(Wm)).net
    # same for the blend with the wkflag model
    wf = S["wkflag"].reindex(index=sb.index)
    okf = sb.notna() & wf.notna()
    mixf = (sb.where(okf).rank(axis=1, pct=True) + wf.where(okf).rank(axis=1, pct=True)) / 2
    Wf = bt.select_topk(mixf, mixf.notna(), 10)
    nets["blend+wkflag mix"] = bt.run(Wf, R.reindex_like(Wf), C.reindex_like(Wf)).net
    # overlap of picks
    for k in ["wkflag", "wkonly"]:
        s = S[k].loc[S[k].index.isin(wkdays)]
        Wk = bt.select_topk(s, s.notna(), 10) > 0
        s0 = S["base"].loc[s.index]
        W0 = bt.select_topk(s0, s0.notna(), 10) > 0
        ov = (Wk & W0).sum(1) / 10
        rows.append(dict(part="B pick overlap with base", variant=k, period="2022-26", n=len(ov), mean_bp=np.nan,
                         overlap=ov.mean()))
    for p, a, b in PER:
        for k in nets:
            x = nets[k].loc[a:b]
            if x.dropna().empty:
                continue
            rows.append(stats_row("B weekend nights top 10 vs base", k, p, x, nets["base"].loc[a:b]))
            if p != "2022-23":
                rows.append(stats_row("B weekend nights top 10 vs blend", k, p, x, nets["blend"].loc[a:b]))
    # also on all nights (does the flag model hurt the other nights?)
    for k in ["base", "wkflag"]:
        s = S[k]
        W = bt.select_topk(s, s.notna(), 10)
        nets["all_" + k] = bt.run(W, R.reindex_like(W), C.reindex_like(W)).net
    for p, a, b in PER:
        x = nets["all_wkflag"].loc[a:b]
        rows.append(stats_row("B all nights top 10 vs base", "wkflag", p, x, nets["all_base"].loc[a:b]))
        rows.append(stats_row("B all nights top 10 vs base", "base", p, nets["all_base"].loc[a:b]))
    # data check: largest single-name weekend returns among the picks
    W0 = bt.select_topk(S["wkonly"].loc[S["wkonly"].index.isin(wkdays)], S["wkonly"].loc[S["wkonly"].index.isin(wkdays)].notna(), 10)
    picks = (R.reindex_like(W0)).where(W0 > 0).stack()
    print("largest |weekend return| among wkonly picks:\n", picks.abs().sort_values().tail(8))

    # ---- Part C: skip rules on the blend
    wd = pd.Series(idx.weekday, index=idx)
    rules = {"none": pd.Series(True, index=idx), "skip Friday nights": wd != 4, "skip Monday nights": wd != 0,
             "skip Friday and Monday": ~wd.isin([0, 4]), "skip weekend + pre-holiday (gap>=3)": gap < 3}
    for name, keep in rules.items():
        x = blend.where(keep, 0.0)
        for p, a, b in PER[1:]:
            st = ann_stats(x.loc[a:b])
            rows.append(dict(part="C blend skip rule", variant=name, period=p, n=int(keep.loc[a:b].sum()),
                             mean_bp=1e4 * x.loc[a:b].mean(), sharpe=st["sharpe"], ann=st["ann_ret"], maxdd=st["maxdd"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study97_weekend_model.csv", index=False)
    show = ["part", "variant", "period", "n", "mean_bp", "sharpe", "diff_bp", "t_diff", "overlap"]
    print(df[[c for c in show if c in df]].round(3).to_string())
