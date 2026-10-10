"""Study 99: retrain the blend's models with a target that penalizes the gap between the panel open (first trade) and
the official opening cross, and/or with the weekend flag, and score the picks at official cross prices.

Question: studies 88 and 94 found that the daily panel's open (Yahoo, core.load_panel P["o"]) is usually the first
trade, not the listing exchange's opening cross, and for the blend's picks lies +5 / +11 bp above the cross
(2024-01..2025-06 / 2025-07..2026-09), against +0.2 / +1.9 bp for random liquid stocks. The models were trained on
the panel open, so they may favour stocks whose first trade overshoots the cross. Does retraining with a target that
subtracts the expected first-trade-vs-cross gap, and/or adding study 97's weekend flag, pick stocks that do better at
cross prices?

Data: official crosses cached by studies 88 / 94 (data/local/auctions88.parquet, auctions94.parquet: blend picks and
6,000 random liquid stock-days) plus crosses fetched here for the new picks (data/local/auctions99.parquet, same
rule as study 94: listing-venue 'O' / '6' print, second pass largest print of any listing exchange; Alpaca SIP).
Training frame data/ml_frame.parquet (study-3 close features, y_night = close -> next panel open), earnings
calendar and news counts as in study 33 (earn_tonight, news_today), behavior groups (results/study14_groups.parquet).

Design:
  (a) Gap model. Target: gap = panel open / official opening cross - 1 (bp, both on the day's traded scale, study 94
      method; winsorized at +-300 bp) for every cached opening cross. Inputs, all known by the open or before and
      available for every training row: the realized overnight return to the panel open (y_night, signed and
      absolute), log price, log dollar volume, 20-day volatility, the day's range and volume z-score (ml_frame
      columns of the entry day). Premarket volume is not available for the whole training universe and is not used.
      Small LightGBM regression. Out of sample checks: fit on 2024-01..2025-06, test on 2025-07..2026-09 and the
      reverse. For the target adjustment the gap model is cross-fitted so it never sees crosses from the test
      period: test quarters 2025Q3..2026Q3 use a gap model fitted on crosses before the training cut (walk-forward);
      test quarters 2024Q1..2025Q2 have no earlier crosses and use a gap model fitted on 2025-07..2026-09 crosses
      (a look-ahead in the target only, small if the gap model is stable across periods; reported).
      Adjusted overnight target = y_night - predicted gap.
  (b) Retrain, walk-forward by quarter, test 2024Q1..2026Q3, all rows from 2020 before the quarter start minus a
      10-day embargo, same LightGBM parameters as the originals (num_threads=2): the strongest ensemble member at
      15:45 in study 23, per_beh (one model per behavior group, SMALL parameters; stocks without a group or in a
      group with fewer than 20,000 training rows scored by pooled_beh, which is retrained too), and the study-33
      jump / drop classifiers (targets > +5% / < -5%). Variants: base (panel target), adj (gap-adjusted target),
      base_wk and adj_wk (+ weekend flag, calendar gap days and 6 flag x input interactions, as study 97), and
      base_seed2 (base with LightGBM seed 2: how much the picks and returns move from random seeds alone). Rank
      targets use the within-day rank of the (adjusted) return; jump / drop use the (adjusted) return.
      Cost: the 15:45 feature frames of study 23 need ~5 GB RAM and ~8 s per day in one process (the full feature
      build runs on import), over this study's 4 GB limit, so all variants are trained and scored on close
      features. The fair comparison is therefore with the base variant retrained and scored the same way; the live
      15:45 blend (original models) is shown as a reference only (it is scored at 15:45 with all five members).
  (c) Blend per variant: (2 x rank(per_beh) + rank(P(jump) - P(drop))) / 3, top 10, buy at the closing cross, sell at
      the next opening cross; returns at crosses = panel return x cross / panel ratio per leg (study 94, legs
      without a usable cross fall back to the panel price; shares reported). Costs exec_cost_bps(P, "auction") +
      2.5 bp per side. Sharpe, mean net bp, max drawdown per period (2024-01..2025-06, 2025-07..2026-09,
      2024-01..2026-09); paired t of daily net returns against the retrained base blend (fair test) and against the
      live blend at cross prices (study 94: 2.48 / 2.27 / 2.38).

Data checks: ml_frame y_night equals the panel night return on every cross row; the live blend rescored here
reproduces study 94 exactly (2.48 / 2.27 / 2.38 at crosses); 71 opening and 72 closing legs whose Yahoo raw close
could not be put on the traded scale and 1 opening cross outside 0.8..1.25 of the panel open use the panel price;
16,905 + 2,130 new (ticker, day, kind) crosses fetched, 99.2% found; legs at the panel price 0.5-1.8% per variant.
The "ensemble member only" rows (per_beh alone) are priced at the panel only (their picks were not fetched).

Result (2024-01..2025-06 / 2025-07..2026-09 / 2024-26, Sharpe at crosses): base 2.38 / 2.12 / 2.24, adj 2.92 /
2.63 / 2.75, base_wk 2.86 / 2.55 / 2.70, adj_wk 2.20 / 1.81 / 2.00, base_seed2 2.53 / 2.52 / 2.52; live blend 2.48 /
2.27 / 2.38. Paired vs base (2024-26, cross): adj +6.1 bp (t 1.16), base_wk +4.3 (t 0.79), adj_wk -6.3 (t -1.14),
base with another seed +3.4 (t 0.65). A seed change alone keeps only 58% of the picks, as many as the variants keep
(54-61%), so the differences between variants are of the size of seed noise, and the two changes together are the
worst variant. The adjusted target lowers the picks' panel-vs-cross gap by ~1.5 bp a night (7.8 -> 6.3 bp), the
other variants by less. The gap model is weak (out-of-sample R2 ~0) but ranks the gap in the right order (top
quintile +8.8 / +4.9 bp vs bottom +1.1 / +0.1 bp). Verdict: reject; nothing beats the base retrain beyond seed noise.

Output: results/study99_cross_retrain.csv (tables: gapmodel = out-of-sample checks, gapcal = calibration by predicted
quintile, rescore = Sharpe / net bp / drawdown per variant, period and pricing, paired = paired t-tests, fallback =
share of legs at the panel price, overlap = shared picks with the base variant, pickgap = realized panel-vs-cross gap
of each variant's picks). Model predictions cached in data/local/study99_pred.parquet (resumable), gap predictions in
data/local/study99_gaphat.parquet, study-33 extra inputs for ml_frame rows in data/local/study99_extra.parquet.

    python src/study99_cross_retrain.py prep     # extra inputs + gap models (small, no fork after LightGBM)
    python src/study99_cross_retrain.py train    # walk-forward retrain (hours; resumable by quarter and variant)
    python src/study99_cross_retrain.py fetch    # official crosses for new picks (resumable)
    python src/study99_cross_retrain.py          # evaluation
"""
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
ML = os.path.join(DATA, "ml_frame.parquet")
AUC99 = os.path.join(LOC, "auctions99.parquet")
AUC94 = os.path.join(LOC, "auctions94.parquet")
AUC88 = os.path.join(LOC, "auctions88.parquet")
PRED_FN = os.path.join(LOC, "study99_pred.parquet")
GAP_FN = os.path.join(LOC, "study99_gaphat.parquet")
EXTRA_FN = os.path.join(LOC, "study99_extra.parquet")
OUT = os.path.join(RES, "study99_cross_retrain.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
QUARTERS = pd.period_range("2024Q1", "2026Q3", freq="Q")
VARIANTS = ["base", "adj", "base_wk", "adj_wk", "base_seed2"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=2000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
SMALL = dict(PARAMS, num_leaves=31, min_data_in_leaf=500)
PR_BIN = dict(PARAMS, objective="binary")
INTER = ["r1", "r5", "vol20", "night1", "beta", "days_to_e"]
GAP_IN = ["y_night", "abs_night", "lpx", "ldv", "vol20", "range1", "vz1"]
GAP_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=300, lambda_l2=10.0,
                  feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=99)
GAP_ROUNDS = 200
WIN = 300.0


# ------------------------------------------------------------------ common
def calendar():
    P = load_panel()
    days = P["c"].index
    nxt = pd.Series(days[1:].append(pd.DatetimeIndex([pd.NaT])), index=days)
    gapd = (nxt - pd.Series(days, index=days)).dt.days
    return P, days, nxt, gapd


def load_frame(columns=None):
    """ml_frame rows from 2020 as float32 numpy: X (features), names, dates, tickers, y_night."""
    f = pq.ParquetFile(ML)
    names = [n for n in f.schema_arrow.names if n not in ("date", "ticker") and not n.startswith("y_")]
    if columns is not None:
        names = [n for n in names if n in columns]
    dates = pd.DatetimeIndex(f.read(columns=["date"]).column("date").to_pandas())
    keep = np.asarray(dates >= "2020-01-01")
    tick = f.read(columns=["ticker"]).column("ticker").to_pandas().values[keep]
    X = np.empty((int(keep.sum()), len(names)), dtype="float32")
    for j, n in enumerate(names):
        X[:, j] = f.read(columns=[n]).column(n).to_numpy(zero_copy_only=False)[keep]
    y = f.read(columns=["y_night"]).column("y_night").to_numpy(zero_copy_only=False)[keep].astype("float32")
    return X, names, dates[keep], tick, y


def crosses():
    """Cached official crosses (ticker in panel style, date, kind, px), studies 94 / 88 / 99, one row per key."""
    import study94_cross_rescore as S94
    a = pd.read_parquet(AUC94)[["ticker", "date", "kind", "px"]].assign(src=0)
    parts = [a]
    b = pd.read_parquet(AUC88)
    ven = S94.listing_venue()
    lv = b.ticker.map(ven)
    b = b[b.open_px.notna() & (lv.isna() | (b.open_x == lv))]           # same filter as study 94's seed
    parts.append(pd.DataFrame({"ticker": b.ticker, "date": b.date, "kind": "open", "px": b.open_px, "src": 1}))
    if os.path.exists(AUC99):
        c = pd.read_parquet(AUC99)
        parts.append(c[["ticker", "date", "kind", "px"]].assign(src=2))
    x = pd.concat(parts, ignore_index=True)
    x["date"] = pd.to_datetime(x.date)
    x["ticker"] = x.ticker.astype(str).str.replace(".", "-", regex=False)
    x = x.sort_values(["src"]).dropna(subset=["px"]).drop_duplicates(["ticker", "date", "kind"])
    return x.drop(columns="src").reset_index(drop=True)


def cross_ratios(P, cols, days, a):
    """Study 94 cross_ratios on a given cross table: ro, rc = cross / panel price on the traded scale (NaN where no
    cross or data error), plus error counts."""
    import study94_cross_rescore as S94
    tc = traded_close(P)[cols]
    ix = days[days >= "2023-12-01"]
    a = a[a.ticker.isin(cols) & a.date.isin(ix)]
    X = {k: g.pivot(index="date", columns="ticker", values="px").reindex(index=ix, columns=cols)
         for k, g in a.groupby("kind")}
    F = S94.split_factor(cols, ix)
    prc = P["rawc"][cols].reindex(ix) * F
    scale_ok = ((prc / tc.reindex(ix) - 1).abs() < 0.03) | tc.reindex(ix).isna()
    pro = P["o"][cols].reindex(ix) / P["c"][cols].reindex(ix) * prc
    out, err = {}, {}
    for k, x, pr in [("ro", X["open"], pro), ("rc", X["close"], prc)]:
        r = x / pr
        bad_scale = x.notna() & ~scale_ok
        bad_ratio = x.notna() & scale_ok & ((r < 0.8) | (r > 1.25))
        err[k] = dict(found=int(x.notna().sum().sum()), scale=int(bad_scale.sum().sum()),
                      ratio=int(bad_ratio.sum().sum()))
        out[k] = r.where(scale_ok & ~bad_ratio).reindex(index=days)
    return out["ro"], out["rc"], err


# ------------------------------------------------------------------ (a) prep: extra inputs and gap models
def gap_inputs(Xg, names):
    j = {n: i for i, n in enumerate(names)}
    y = Xg[:, j["y_night"]]
    return np.column_stack([y, np.abs(y)] + [Xg[:, j[k]] for k in GAP_IN[2:]]).astype("float32")


def prep():
    import lightgbm as lgb
    P, days, nxt, gapd = calendar()
    cols = stock_cols(P)
    # ---- study-33 extra inputs for every ml_frame row (earn_tonight, news_today), as study33_jump_live.extra_panels
    if not os.path.exists(EXTRA_FN):
        import store
        f = pq.ParquetFile(ML)
        dates = pd.DatetimeIndex(f.read(columns=["date"]).column("date").to_pandas())
        keep = np.asarray(dates >= "2020-01-01")
        dates = dates[keep]
        tick = f.read(columns=["ticker"]).column("ticker").to_pandas().values[keep]
        E = store.read("earnings")
        di = days.searchsorted(pd.to_datetime(E.date))
        earn = np.zeros((len(days), len(cols)), dtype="float32")
        ci = {t: i for i, t in enumerate(cols)}
        for d, t in zip(di, E.symbol):
            if t in ci and 0 < d < len(days):
                earn[d, ci[t]] = 1.0
                earn[d - 1, ci[t]] = 1.0
        news = pd.read_pickle(os.path.join(LOC, "factor_panels.pkl"))["n_news"].reindex(index=days, columns=cols)
        news = news.values.astype("float32")
        gc.collect()
        r = days.get_indexer(dates)
        c = pd.Index(cols).get_indexer(tick)
        ok = (r >= 0) & (c >= 0)
        ev, nv = np.full(len(r), np.nan, "float32"), np.full(len(r), np.nan, "float32")
        ev[ok], nv[ok] = earn[r[ok], c[ok]], news[r[ok], c[ok]]
        pd.DataFrame({"earn_tonight": ev, "news_today": nv}).to_parquet(EXTRA_FN, index=False)
        del news, earn
        gc.collect()
        print("extra inputs saved", len(r), flush=True)

    # ---- gap model data: every cached opening cross, with ml_frame inputs of the entry day t (open on t+1)
    Xg, names, dates, tick, y = load_frame(columns=GAP_IN[2:])
    names = ["y_night"] + names
    Xg = np.column_stack([y, Xg])
    Z = gap_inputs(Xg, names)
    del Xg
    gc.collect()
    a = crosses()
    ro, rc, err = cross_ratios(P, cols, days, a)
    print("cross data errors:", err, flush=True)
    s = ro.stack(future_stack=True).dropna().rename("ro").reset_index()
    s.columns = ["d1", "ticker", "ro"]
    prev = pd.Series(days[:-1], index=days[1:])
    s["date"] = s.d1.map(prev)
    s["gap"] = 1e4 * (1 / s.ro - 1)
    key = pd.MultiIndex.from_arrays([dates, tick])
    pos = pd.Series(np.arange(len(key)), index=key)
    s["row"] = pos.reindex(pd.MultiIndex.from_arrays([s.date, s.ticker])).values
    print("opening crosses", len(s), "with ml_frame row", int(s.row.notna().sum()), flush=True)
    s = s.dropna(subset=["row"]).reset_index(drop=True)
    rn = (P["o"][cols].shift(-1) / P["c"][cols] - 1).stack(future_stack=True)
    rn = rn.reindex(pd.MultiIndex.from_arrays([s.date, s.ticker])).values
    dd = np.abs(Z[s.row.astype(int).values, 0] - rn)
    print("check: ml_frame y_night vs panel night return, share differing > 1 bp:", float(np.nanmean(dd > 1e-4)),
          flush=True)
    del rn
    s["row"] = s.row.astype(int)
    # sample labels: random liquid stock-days of study 94 vs picks
    import study94_cross_rescore as S94
    K = S94.build_picks()
    rnd = set(zip(K["rnd"].date, K["rnd"].ticker))
    del K
    gc.collect()
    s["random"] = [(d, t) in rnd for d, t in zip(s.d1, s.ticker)]
    s["per"] = np.where(s.d1 < "2025-07-01", "2024-25H1", "2025H2-26")
    Zs = Z[s.row.values]
    yw = s.gap.clip(-WIN, WIN).values

    def fit(m):
        return lgb.train(GAP_PARAMS, lgb.Dataset(Zs[m], yw[m], feature_name=GAP_IN), num_boost_round=GAP_ROUNDS)

    rows = []
    for tr_lab, te_lab in [("2024-25H1", "2025H2-26"), ("2025H2-26", "2024-25H1")]:
        mtr, mte = (s.per == tr_lab).values, (s.per == te_lab).values
        mdl = fit(mtr)
        ph = mdl.predict(Zs[mte])
        ya = yw[mte]
        fin = mtr & ~np.isnan(Zs[:, 1])
        lin = np.polyfit(Zs[fin][:, 1], yw[fin], 1)                          # benchmark: |night| only, linear
        pl = np.polyval(lin, np.nan_to_num(Zs[mte][:, 1], nan=np.nanmean(Zs[:, 1])))
        for setn, mm in [("all", np.ones(mte.sum(), bool)), ("picks", ~s.random.values[mte]),
                         ("random", s.random.values[mte])]:
            rows.append(dict(table="gapmodel", train=tr_lab, test=te_lab, set=setn, n=int(mm.sum()),
                             actual_bp=ya[mm].mean(), pred_bp=ph[mm].mean(),
                             corr=np.corrcoef(ya[mm], ph[mm])[0, 1],
                             r2=1 - np.mean((ya[mm] - ph[mm]) ** 2) / np.var(ya[mm]),
                             r2_absnight_linear=1 - np.mean((ya[mm] - pl[mm]) ** 2) / np.var(ya[mm])))
        q = pd.qcut(ph, 5, labels=False, duplicates="drop")
        for k in range(int(q.max()) + 1):
            mm = q == k
            rows.append(dict(table="gapcal", train=tr_lab, test=te_lab, set=f"pred quintile {k + 1}", n=int(mm.sum()),
                             actual_bp=ya[mm].mean(), pred_bp=ph[mm].mean()))
        imp = pd.Series(mdl.feature_importance("gain"), index=GAP_IN)
        print("gap model", tr_lab, "->", te_lab, "importance", (imp / imp.sum()).round(3).to_dict(), flush=True)
    # by realized-night bucket on all data (in-sample description)
    nb = pd.cut(Zs[:, 0], [-1, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 10])
    for b, g in s.groupby(nb, observed=True):
        rows.append(dict(table="gapdesc", set=f"night return {b}", n=len(g), actual_bp=g.gap.clip(-WIN, WIN).mean(),
                         share_random=g.random.mean()))
    pd.DataFrame(rows).to_csv(os.path.join(LOC, "study99_gaprows.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)

    # ---- gap predictions for every ml_frame row, one column per gap-model version
    G = {}
    m2 = (s.per == "2025H2-26").values
    G["g_p1"] = fit(m2).predict(Z).astype("float32")                      # for test quarters 2024Q1..2025Q2
    for q in QUARTERS[QUARTERS >= pd.Period("2025Q3", "Q")]:
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        m = (s.d1 < cut).values
        G[f"g_{q}"] = fit(m).predict(Z).astype("float32")
        print("gap model for", q, "fitted on", int(m.sum()), "crosses; mean pred bp", float(G[f"g_{q}"].mean()), flush=True)
    pd.DataFrame(G).to_parquet(GAP_FN, index=False)
    print("gap predictions saved", flush=True)


def gap_col(q):
    return "g_p1" if q < pd.Period("2025Q3", "Q") else f"g_{q}"


# ------------------------------------------------------------------ (b) train
def groups(year):
    g = pd.read_parquet(os.path.join(RES, "study14_groups.parquet"))["group"]
    y = min(int(year), int(g.index.get_level_values(0).max()))
    return g.xs((y, "behavior"), level=(0, 1))


def group_feats(X, names, dates, tick, year):
    """ensemble.add_group_features for the behavior groups of `year`: g_id, g_{r1,r5,night1,intra1}_rel, g_r5_mean."""
    g = groups(year)
    gid = pd.Series(tick).map(g).fillna(-1).values.astype("float32")
    j = {n: i for i, n in enumerate(names)}
    key = pd.MultiIndex.from_arrays([dates, gid])
    out = [gid]
    df = pd.DataFrame({k: X[:, j[k]] for k in ["r1", "r5", "night1", "intra1"]}, index=key)
    gm = df.groupby(level=[0, 1]).transform("mean")
    for k in ["r1", "r5", "night1", "intra1"]:
        out.append((df[k].values - gm[k].values).astype("float32"))
    out.append(gm["r5"].values.astype("float32"))
    return gid, np.column_stack(out), ["g_id", "g_r1_rel", "g_r5_rel", "g_night1_rel", "g_intra1_rel", "g_r5_mean"]


def mat(rows, parts, chunk=200000):
    """Rows `rows` (bool mask) of the column blocks `parts` as one float32 matrix, filled in chunks (low peak RAM)."""
    idx = np.flatnonzero(rows)
    M = np.empty((len(idx), sum(p.shape[1] for p in parts)), dtype="float32")
    c0 = 0
    for p in parts:
        k = p.shape[1]
        for a in range(0, len(idx), chunk):
            M[a:a + chunk, c0:c0 + k] = p[idx[a:a + chunk]]
        c0 += k
    return M


def train():
    import lightgbm as lgb
    P, days, nxt, gapd = calendar()
    X, names, dates, tick, y = load_frame()
    ex = pd.read_parquet(EXTRA_FN)
    E = ex.values.astype("float32")
    del ex
    G = pd.read_parquet(GAP_FN)
    n_base = len(names)
    gd = gapd.reindex(dates).values.astype("float32")
    wk = (gd >= 3).astype("float32")
    wk[np.isnan(gd)] = np.nan
    j = {n: i for i, n in enumerate(names)}
    W = np.column_stack([wk, gd] + [X[:, j[k]] * wk for k in INTER]).astype("float32")
    wk_names = ["wk", "gapdays"] + [f"wk_{k}" for k in INTER]
    ok = ~np.isnan(y)
    dnp = np.asarray(dates)
    print("rows", len(y), "features", n_base, "weekend share", float(np.nanmean(wk)), flush=True)
    done = pd.read_parquet(PRED_FN) if os.path.exists(PRED_FN) else None
    have = set(zip(done.q, done.variant)) if done is not None else set()
    out = [done] if done is not None else []
    gcache = {}
    for q in QUARTERS:
        if all((str(q), v) in have for v in VARIANTS):
            continue
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & (dnp < np.datetime64(cut))
        te = (dnp >= np.datetime64(q.start_time)) & (dnp <= np.datetime64(q.end_time.normalize()))
        if q.year not in gcache:
            gcache.clear()
            gc.collect()
            gcache[q.year] = group_feats(X, names, dates, tick, q.year)
        gid, GF, gf_names = gcache[q.year]
        gh = G[gap_col(q)].values
        for v in VARIANTS:
            if (str(q), v) in have:
                continue
            t0 = time.time()
            yv = y - gh / 1e4 if v.startswith("adj") else y
            yr = (pd.Series(np.where(tr, yv, np.nan)).groupby(dnp).rank(pct=True).values - 0.5).astype("float32")
            use_wk = v.endswith("_wk")
            sd = dict(seed=2) if v == "base_seed2" else {}      # noise yardstick: base with other random seeds
            res = pd.DataFrame({"date": dates[te], "ticker": tick[te], "q": str(q), "variant": v})

            base_parts = [X] + ([W] if use_wk else [])
            base_names = names + (wk_names if use_wk else [])
            # pooled_beh
            fn = base_names + gf_names
            ds = lgb.Dataset(mat(tr, base_parts + [GF]), yr[tr], feature_name=fn, categorical_feature=["g_id"],
                             free_raw_data=True)
            m = lgb.train(dict(PARAMS, **sd), ds, num_boost_round=300)
            del ds
            pb = m.predict(mat(te, base_parts + [GF])).astype("float32")
            res["pooled_beh"] = pb
            del m
            gc.collect()
            # per_beh: one SMALL model per group with >= 20,000 training rows; others keep pooled_beh
            per = pb.copy()
            gte = gid[te]
            for gg in sorted(set(gid)):
                mm = gid == gg
                if gg == -1 or (tr & mm).sum() < 20000:
                    continue
                ds = lgb.Dataset(mat(tr & mm, base_parts), yr[tr & mm], feature_name=base_names, free_raw_data=True)
                mg = lgb.train(dict(SMALL, **sd), ds, num_boost_round=300)
                del ds
                sel = te & mm
                per[gte == gg] = mg.predict(mat(sel, base_parts))
                del mg
                gc.collect()
            res["per_beh"] = per
            # jump / drop classifiers (study-3 inputs + earn_tonight + news_today)
            jn = names + ["earn_tonight", "news_today"] + (wk_names if use_wk else [])
            jparts = [X, E] + ([W] if use_wk else [])
            Xte = mat(te, jparts)
            for k, lab in [("jump", yv > 0.05), ("drop", yv < -0.05)]:
                ds = lgb.Dataset(mat(tr, jparts), lab[tr].astype("float32"), feature_name=jn, free_raw_data=True)
                mj = lgb.train(dict(PR_BIN, **sd), ds, num_boost_round=300)
                del ds
                res[f"p_{k}"] = mj.predict(Xte).astype("float32")
                del mj
                gc.collect()
            del Xte
            out.append(res)
            pd.concat(out, ignore_index=True).to_parquet(PRED_FN, index=False)
            print(q, v, "train", int(tr.sum()), "test", int(te.sum()), f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ (c) blends, fetch, evaluation
def blends(cols, days):
    """Score panels per variant (date x ticker) and the live 15:45 blend; index = entry day t."""
    pr = pd.read_parquet(PRED_FN)
    pr["date"] = pd.to_datetime(pr.date)
    S = {}
    for v, g in pr.groupby("variant"):
        g = g.set_index(["date", "ticker"])
        e = g.per_beh.unstack().reindex(columns=cols)
        j = (g.p_jump - g.p_drop).unstack().reindex(index=e.index, columns=cols)
        ok = e.notna() & j.notna()
        s = (2 * e.where(ok).rank(axis=1, pct=True) + j.where(ok).rank(axis=1, pct=True)) / 3
        S[v] = s.loc[s.index < days[-1]]
        S[v + " (ensemble member only)"] = e.where(ok).loc[e.index < days[-1]]
    # live blend, exactly as study 94 build_picks
    p33 = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = p33.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = p33.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S["live 15:45 blend (study 33)"] = (2 * ens.where(ok).rank(axis=1, pct=True)
                                        + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    return S


def fetch():
    import study94_cross_rescore as S94
    P, days, nxt, gapd = calendar()
    cols = stock_cols(P)
    S = blends(cols, days)
    rows = []
    for v in [v for v in VARIANTS if v in S]:
        Wv = bt.select_topk(S[v], S[v].notna(), 10).loc["2024-01":"2026-09"]
        s = Wv.stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(rows).dropna().drop_duplicates()
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    have = crosses()[["ticker", "date", "kind"]]
    tried = pd.read_parquet(AUC99) if os.path.exists(AUC99) else None
    if tried is not None:            # pairs tried here already (found or not)
        t2 = tried[["ticker", "date", "kind"]].copy()
        t2["ticker"] = t2.ticker.str.replace(".", "-", regex=False)
        have = pd.concat([have, t2.assign(date=pd.to_datetime(t2.date))])
    # also pairs tried by study 94 without a cross
    a94 = pd.read_parquet(AUC94)[["ticker", "date", "kind"]]
    a94["ticker"] = a94.ticker.str.replace(".", "-", regex=False)
    have = pd.concat([have, a94.assign(date=pd.to_datetime(a94.date))]).drop_duplicates()
    m = need.merge(have, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("needed", len(need), "to fetch", len(todo), flush=True)
    todo = todo.assign(ticker=todo.ticker.str.replace("-", ".", regex=False))
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            r = S94._day_kind(job[0], job[1], job[2], ven)
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
            if miss:              # second pass: largest print of any listing exchange
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    x["rule"] = "venue"
                    if x["ticker"] in lg and not np.isnan(lg[x["ticker"]]["px"]):
                        x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x", "rule"]})
                    elif x["ticker"] in lg:
                        x["rule"] = "largest"
            for x in r:
                x.setdefault("rule", "venue")
            return r
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC99) if os.path.exists(AUC99) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(4) as exe:
            for r in exe.map(one, jobs[i:i + 24]):
                new += r
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC99, index=False)
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}" if new else "", f"{time.time() - t0:.0f}s",
              flush=True)


def main():
    P, days, nxt, gapd = calendar()
    cols = stock_cols(P)
    S = blends(cols, days)
    a = crosses()
    ro, rc, err = cross_ratios(P, cols, days, a)
    print("cross data errors (legs set to the panel price):", err, flush=True)
    rows = [dict(table="data_errors", set=k, n=v["found"], scale_mismatch=v["scale"], ratio_out_of_range=v["ratio"])
            for k, v in err.items()]
    if os.path.exists(os.path.join(LOC, "study99_gaprows.csv")):
        rows += pd.read_csv(os.path.join(LOC, "study99_gaprows.csv")).to_dict("records")
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = o.shift(-1) / c - 1
    fo, fc = ro.fillna(1.0), rc.fillna(1.0)
    Rn_x = (1 + Rn_p) * fo.shift(-1) / fc - 1
    miss = ro.shift(-1).isna() | rc.isna()
    gap_bp = 1e4 * (1 / ro.shift(-1) - 1)                         # panel open vs cross on the sell day (t+1)
    C = exec_cost_bps(P, "auction")[cols] + 2.5
    nets, Ws = {}, {}
    VA = [v for v in VARIANTS if v in S]
    order = ["live 15:45 blend (study 33)"] + VA + [v + " (ensemble member only)" for v in VA]
    for name in order:
        Sx = S[name].loc["2024-01":"2026-09"]
        Wx = bt.select_topk(Sx, Sx.notna(), 10)
        Ws[name] = Wx
        member = "member only" in name          # crosses were not fetched for these picks: panel prices only
        for pr, R in [("panel", Rn_p), ("cross", Rn_x)][:1 if member else 2]:
            net = bt.run(Wx, R.reindex_like(Wx), C.reindex_like(Wx)).net
            nets[(name, pr)] = net
            for lab, a0, b0 in PER:
                x = net.loc[a0:b0].dropna()
                st = ann_stats(x)
                rows.append(dict(table="rescore", variant=name, pricing=pr, period=lab, n=st["n"], sharpe=st["sharpe"],
                                 net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"]))
        h = Wx > 0
        for lab, a0, b0 in PER:
            hh = h.loc[a0:b0]
            mm = (miss.reindex_like(Wx).fillna(True) & hh)
            g = gap_bp.reindex_like(Wx).where(hh).loc[a0:b0].stack(future_stack=True).dropna()
            if member:
                continue
            rows.append(dict(table="fallback", variant=name, period=lab, legs=int(hh.sum().sum()),
                             share_panel=mm.loc[a0:b0].sum().sum() / max(hh.sum().sum(), 1)))
            rows.append(dict(table="pickgap", variant=name, period=lab, n=len(g), mean_bp=g.mean(),
                             mean_winsor_bp=g.clip(-WIN, WIN).mean(), median_bp=g.median()))
            if name != "live 15:45 blend (study 33)":
                ref = "base" if "member" not in name else "base (ensemble member only)"
                ov = ((Wx > 0) & (Ws[ref] > 0)).loc[a0:b0].sum(axis=1) / (Wx > 0).loc[a0:b0].sum(axis=1)
                rows.append(dict(table="overlap", variant=name, period=lab, set=f"vs {ref}", share=ov.mean()))
                ovl = ((Wx > 0) & (Ws["live 15:45 blend (study 33)"] > 0)).loc[a0:b0].sum(axis=1) / (
                    Wx > 0).loc[a0:b0].sum(axis=1)
                rows.append(dict(table="overlap", variant=name, period=lab, set="vs live 15:45 blend", share=ovl.mean()))
    for name in order[1:]:
        refs = ["live 15:45 blend (study 33)"]
        refs.append("base" if "member" not in name else "base (ensemble member only)")
        for ref in refs:
            if ref == name:
                continue
            for pr in ["panel", "cross"]:
                if (name, pr) not in nets or (ref, pr) not in nets:
                    continue
                for lab, a0, b0 in PER:
                    d = (nets[(name, pr)] - nets[(ref, pr)]).loc[a0:b0].dropna()
                    rows.append(dict(table="paired", variant=name, set=f"vs {ref}", pricing=pr, period=lab, n=len(d),
                                     diff_bp=1e4 * d.mean(), t_diff=d.mean() / d.std() * np.sqrt(len(d))))
    # weekend nights only (study 97's question at cross prices)
    wkd = gapd.reindex(nets[("base", "cross")].index) >= 3
    for name in VA:
        for lab, a0, b0 in PER:
            x = nets[(name, "cross")].loc[a0:b0]
            r_ = nets[("base", "cross")].loc[a0:b0]
            m = wkd.loc[a0:b0].values
            d = (x - r_)[m].dropna()
            rows.append(dict(table="weekend_nights", variant=name, pricing="cross", period=lab, n=int(m.sum()),
                             net_bp=1e4 * x[m].mean(), diff_bp=1e4 * d.mean(),
                             t_diff=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"prep": prep, "train": train, "fetch": fetch, "eval": main}[cmd]()
