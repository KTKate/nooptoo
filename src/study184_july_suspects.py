"""Study 184: do the two drifting inputs (days_to_e, beta) explain the blend's July-September 2026 weakness?

Background: the blend (study 33: (2 x within-day rank of the 5-model ensemble + rank of P(jump) - P(drop))) / 3,
top 10 with at most 3 per industry) has been weak since July 2026. Study 180: every model vintage was weak in 2026Q3
and October, so the 2026Q3 refit is not the cause. Study 183: simple no-model rules were only mildly weaker in Q3, so
Q3 looks model-side. Study 181: two stock inputs drifted in 2026Q3: beta (60-day; PSI 0.60; the cross-section
widened, median from about 0.95 to 0.5; the ensemble's largest stock input, 7% of gain) and days_to_e (missing for
11% / 84% / 99% of rows in July / August / September in data/ml_frame.parquet; about 11% of the jump models' gain).

Question: with the frozen models the backtest used on those days (2026Q3 models on 2026Q3 days, 2026Q4 models in
October; the 2026Q2 models as a second check), do the July-September and October results improve when
  (a) a_earn       days_to_e and days_since_e come from the corrected study 158 panel,
  (b) b_qmap       beta is replaced by the value at the same within-day percentile of the 2024-25 beta distribution,
  (c) c_clip       beta is clipped to the 2024-25 1st-99th percentile range,
  (d) d_earn_qmap  (a) and (b) together?
Sanity checks: 2026Q2 as a control (own 2026Q2 models), and (b) / (c) on 2025 days with the 2025 quarterly models,
where the beta distribution is the reference itself, so little should change.

Data
  Frames (study 179/180, data/local/study179_frames.parquet): C = rebuilt daily close features (2026Q2, 2026-07-01..
  09-25, 2026-09-28..10-14), ML = data/ml_frame.parquet close features of the 2026Q2 / Q3 days (what the study 23/33
  backtest saw, with the days_to_e hole), B = 15:45 frame of the 13 October days; earn_tonight and news_today as
  study 180. 2025 frame: data/ml_frame.parquet rows 2025-01-02..12-31 with earn_tonight from the extended-grid
  calendar (study 154 Ctx.earn_tonight_ext) and news_today from data/local/factor_panels.pkl n_news (0 when none), as
  study 180 does for the ML rows; written to data/local/study184_frames2025.parquet.
  Corrected earnings inputs: data/local/study158_earn.parquet (date, ticker; missing where the panel has no row).
  Beta reference: every data/ml_frame.parquet row 2024-01-02..2025-12-31 with beta present (the model universe,
  raw close > $5 and ADV20 > $5M, as the frames).
  Models (read only): data/models night_ / ens_* / jump_{jump,drop}_<q>, scored with study 180's score_set
  (ensemble.predict and the jump models; LightGBM predict with num_threads=2, single process).
  Official crosses: every cached data/local/auctions*.parquet (study 166 cached_crosses) plus data/local/
  auctions184.parquet fetched here for legs not cached (study 94 rule: listing-venue 'O' / '6' print in widening
  windows, then the largest 'O' / '6' print of any listing exchange; about 60 Alpaca requests a minute).
  Tech hardware: study 90/92 definition (data/store/sectors.parquet snapshot, study 154 sectors()).

Design
  Variants change only the named columns of the frame before scoring; all other inputs are unchanged.
    base         the frame as is
    a_earn       days_to_e, days_since_e from the study 158 panel (by date, ticker)
    b_qmap       per day, percentile p = (rank - 0.5) / n of beta among the day's rows with beta, beta := the
                 2024-25 reference quantile at p (missing stays missing)
    c_clip       beta clipped to [1st, 99th] percentile of the reference
    d_earn_qmap  a_earn and b_qmap
    h_hole       (C and B frames only) days_to_e set to missing for every row, the live job's input in 2026Q3-Q4
                 (study 158), to check the effect in the other direction where the frame is already correct.
  Model sets: 2026Q2 days: 2026Q2. 2026Q3 days: 2026Q3 (own) and 2026Q2. October: 2026Q4 (own) and 2026Q2. 2025:
  the own quarter's models 2025Q1..Q4 (variants base, b_qmap, c_clip).
  Picks: blend on the scored universe, top 10 with at most 3 per industry (paper_overnight.cap_by_industry rule), no
  tradable filter (as the backtest and study 180). Night return at crosses as study 94 / 180 (panel night return x
  open cross / panel open on t+1 / (close cross / panel close on t); legs without a usable cross at the panel price),
  equal weight, cost core.exec_cost_bps(P, 'auction') + 2.5 bp per side.
  Per period x frame x model x variant: nights, net bp a night, Sharpe (annualized, 252), hit rate, gross, mean
  overlap with the base picks (of 10), mean tech hardware picks a night, mean daily Spearman of the blend with base,
  paired t of the nightly net difference against base, share of changed beta / days_to_e values. Monthly means too.

Output: results/study184_july_suspects.csv (table = summary / monthly / input_change)
    python src/study184_july_suspects.py frames   # 2025 frame (ml_frame rows + earn_tonight / news_today)
    python src/study184_july_suspects.py score    # all variants -> data/local/study184_picks.parquet
    python src/study184_july_suspects.py fetch    # crosses not cached -> data/local/auctions184.parquet
    python src/study184_july_suspects.py report
"""
import os
import sys
import time

os.environ.setdefault("ALPACA_PER_MIN", "60")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from core import load_panel, stock_cols, ann_stats, RES, DATA   # noqa: E402

LOC = os.path.join(DATA, "local")
ML = os.path.join(DATA, "ml_frame.parquet")
FRAMES = os.path.join(LOC, "study179_frames.parquet")
F2025 = os.path.join(LOC, "study184_frames2025.parquet")
EARN = os.path.join(LOC, "study158_earn.parquet")
PICKS = os.path.join(LOC, "study184_picks.parquet")
CHG = os.path.join(LOC, "study184_input_change.parquet")
AUC184 = os.path.join(LOC, "auctions184.parquet")
OUT = os.path.join(RES, "study184_july_suspects.csv")
Y25 = ("2025-01-01", "2025-12-31")
Q2 = ("2026-04-01", "2026-06-30")
Q3 = ("2026-07-01", "2026-09-25")
OCT = ("2026-09-28", "2026-10-14")
PERIODS = [("2025", *Y25), ("2026Q2", *Q2), ("2026Q3", *Q3), ("2026-10", *OCT)]
VARIANTS = ["base", "a_earn", "b_qmap", "c_clip", "d_earn_qmap", "h_hole"]


def period_of(d):
    s = str(pd.Timestamp(d).date())
    for p, a, b in PERIODS:
        if a <= s <= b:
            return p
    return None


# ------------------------------------------------------------------ 2025 frame
def frames():
    import study150_154_live_check as S154
    nn = pd.read_pickle(os.path.join(LOC, "factor_panels.pkl"))["n_news"]
    ctx = S154.Ctx()
    keep = [c for c in pq.ParquetFile(FRAMES).schema_arrow.names if c not in ("kind", "date", "ticker",
                                                                               "earn_tonight", "news_today")]
    ml = pq.read_table(ML, columns=["date", "ticker"] + keep,
                       filters=[("date", ">=", pd.Timestamp(Y25[0])), ("date", "<=", pd.Timestamp(Y25[1]))]).to_pandas()
    if "date" in ml.columns:
        ml = ml.set_index(["date", "ticker"])
    ml = ml[keep].astype("float32")
    d0, tk = ml.index.get_level_values(0), ml.index.get_level_values(1)
    t = sorted(set(tk))
    E = ctx.earn_tonight_ext(t)
    r, c = E.index.get_indexer(d0), E.columns.get_indexer(tk)
    ml["earn_tonight"] = E.values[r, c].astype("float32")
    nv = nn.reindex(columns=t)
    r2 = nv.index.get_indexer(d0)
    ml["news_today"] = np.nan_to_num(nv.values[r2, c]).astype("float32")
    ml["kind"] = "ML"
    ml.to_parquet(F2025)
    print("2025 frame", ml.shape, "earn_tonight share", round(float(ml.earn_tonight.mean()), 4),
          "news>0", round(float((ml.news_today > 0).mean()), 3), flush=True)


# ------------------------------------------------------------------ variants
def beta_ref():
    b = pq.read_table(ML, columns=["date", "beta"],
                      filters=[("date", ">=", pd.Timestamp("2024-01-01")), ("date", "<=", pd.Timestamp(Y25[1]))]
                      ).column("beta").to_numpy(zero_copy_only=False)
    b = b[np.isfinite(b)]
    grid = np.linspace(0, 1, 2001)
    return grid, np.quantile(b, grid).astype("float64"), len(b)


def qmap(X, grid, qv):
    d = X.index.get_level_values(0)
    p = X.beta.groupby(d).rank(method="average")
    n = X.beta.notna().groupby(d).transform("sum")
    pct = ((p - 0.5) / n).values
    out = np.interp(pct, grid, qv)
    return np.where(X.beta.notna().values, out, np.nan).astype("float32")


def variant(X, v, ref, earn):
    if v == "base":
        return X
    X = X.copy()
    grid, qv, _ = ref
    if v in ("a_earn", "d_earn_qmap"):
        e = earn.reindex(X.index)
        X["days_to_e"] = e.days_to_e.values.astype("float32")
        X["days_since_e"] = e.days_since_e.values.astype("float32")
    if v in ("b_qmap", "d_earn_qmap"):
        X["beta"] = qmap(X, grid, qv)
    if v == "c_clip":
        lo, hi = np.interp([0.01, 0.99], grid, qv)
        X["beta"] = X.beta.clip(lo, hi)
    if v == "h_hole":
        X["days_to_e"] = np.float32(np.nan)
    return X


def changed(a, b):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    same = (a == b) | (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-4)
    return float(1 - same.mean())


# ------------------------------------------------------------------ scoring
def patch_lgb():
    import lightgbm as lgb
    orig = lgb.Booster.predict

    def pred(self, data, *a, **kw):
        kw["num_threads"] = 2
        return orig(self, data, *a, **kw)
    lgb.Booster.predict = pred


def capper():
    ind = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index(
        "ticker").industry.to_dict()

    def cap(pred, n=10, cap=3):            # same rule as paper_overnight.cap_by_industry, sectors read once
        out, cnt = [], {}
        for t in pred.index:
            g = ind.get(t)
            g = g if isinstance(g, str) and g else t
            if cnt.get(g, 0) >= cap:
                continue
            cnt[g] = cnt.get(g, 0) + 1
            out.append(t)
            if len(out) == n:
                break
        return out
    return cap


def jobs():
    """(frame, model q, day range, variants)."""
    vC = VARIANTS
    vML = VARIANTS[:5]
    J = [("C", "2026Q2", Q2, vC), ("C", "2026Q3", Q3, vC), ("C", "2026Q2", Q3, vC),
         ("C", "2026Q4", OCT, vC), ("C", "2026Q2", OCT, vC),
         ("ML", "2026Q2", Q2, vML), ("ML", "2026Q3", Q3, vML), ("ML", "2026Q2", Q3, vML),
         ("B", "2026Q4", OCT, vC), ("B", "2026Q2", OCT, vC)]
    for q in ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]:
        p = pd.Period(q, freq="Q")
        J.append(("ML25", q, (str(p.start_time.date()), str(p.end_time.date())), ["base", "b_qmap", "c_clip"]))
    return J


def score():
    patch_lgb()
    from study179_180_model_check import score_set
    cap = capper()
    ref = beta_ref()
    print("beta reference rows", ref[2], "p1 / p50 / p99", np.round(np.interp([0.01, 0.5, 0.99], ref[0], ref[1]), 3),
          flush=True)
    earn = pd.read_parquet(EARN, columns=["date", "ticker", "days_to_e", "days_since_e"])
    earn["date"] = pd.to_datetime(earn.date)
    earn = earn.set_index(["date", "ticker"])
    F = pd.read_parquet(FRAMES)
    F25 = pd.read_parquet(F2025)
    picks, chg, t0 = [], [], time.time()
    for kind, q, (a, b), vs in jobs():
        src = F25 if kind == "ML25" else F[F.kind == kind]
        d = src.index.get_level_values(0)
        X0 = src[(d >= a) & (d <= b)].drop(columns="kind")
        base_blend = None
        for v in vs:
            X = variant(X0, v, ref, earn)
            s = score_set(X, q)
            if base_blend is None:
                base_blend = s.blend
            dd = s.index.get_level_values(0)
            rk = pd.DataFrame({"v": s.blend.groupby(dd).rank(), "b": base_blend.reindex(s.index).groupby(dd).rank()})
            rho = rk.groupby(dd).apply(lambda g: g.v.corr(g.b))
            for m, g in X.groupby(X.index.get_level_values(0).to_period("M")):
                g0 = X0.loc[g.index]
                chg.append(dict(frame=kind, model=q, variant=v, month=str(m),
                                beta_changed=changed(g.beta, g0.beta), days_to_e_changed=changed(g.days_to_e, g0.days_to_e),
                                days_since_e_changed=changed(g.days_since_e, g0.days_since_e),
                                days_to_e_nan=float(g.days_to_e.isna().mean()), beta_median=float(g.beta.median()),
                                beta_p10=float(g.beta.quantile(0.1)), beta_p90=float(g.beta.quantile(0.9)),
                                rows=len(g)))
            for day, g in s.groupby(level=0):
                g = g.droplevel(0)
                top = cap(g.blend.sort_values(ascending=False, kind="stable"), 10, 3)
                picks += [dict(date=day, frame=kind, model=q, variant=v, ticker=t, rank=i + 1,
                               spearman_base=float(rho.get(day, np.nan))) for i, t in enumerate(top)]
            print(kind, q, a, v, "days", s.index.get_level_values(0).nunique(), "rows", len(s),
                  f"{time.time() - t0:.0f}s", flush=True)
            del X, s
    pd.DataFrame(picks).to_parquet(PICKS)
    pd.DataFrame(chg).to_parquet(CHG)


# ------------------------------------------------------------------ crosses
def all_weights(cols, days):
    from study179_180_model_check import weights
    pk = pd.read_parquet(PICKS)
    idx = days[(days >= Y25[0]) & (days <= OCT[1])]
    Ws = {}
    for k, g in pk.groupby(["frame", "model", "variant"]):
        Ws[k] = weights(g, cols, idx)
    return Ws, idx, pk


def fetch():
    import study94_cross_rescore as S94
    import study166_167_blend_parts as S166
    import alpaca_data as A
    from concurrent.futures import ThreadPoolExecutor
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    Ws, idx, _ = all_weights(cols, days)
    del P
    need = pd.concat([S166.legs(W, days) for W in Ws.values()], ignore_index=True)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC184):
        have = pd.read_parquet(AUC184)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jb = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jb), flush=True)
    S94.AUC_FN = AUC184
    rows, t0 = [], time.time()

    def one(job):
        try:
            return S94._day_kind(job[0], job[1], job[2], ven)
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    for i in range(0, len(jb), 24):
        with ThreadPoolExecutor(2) as ex:
            for r in ex.map(one, jb[i:i + 24]):
                rows += r
        S94._save(pd.concat([have, pd.DataFrame(rows)], ignore_index=True))
        print("jobs", min(i + 24, len(jb)), "/", len(jb), "rows", len(rows),
              f"found {np.mean([not np.isnan(r['px']) for r in rows]) if rows else 0:.3f}", f"{time.time() - t0:.0f}s",
              flush=True)
    if os.path.exists(AUC184) and len(pd.read_parquet(AUC184)):
        S94.refetch_missing()


# ------------------------------------------------------------------ report
def paired_t(a, b):
    d = (a - b).dropna()
    if len(d) < 5 or d.std() == 0:
        return np.nan
    return d.mean() / d.std() * np.sqrt(len(d))


def report():
    import bt
    import study166_167_blend_parts as S166
    import study150_154_live_check as S154
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    _, _, hw = S154.sectors()
    hwc = hw.reindex(cols).fillna(False).astype(bool)
    Ws, idx, pk = all_weights(cols, days)
    Rx, miss, C = S166.returns(P, cols, days, idx)
    del P
    nets, gross, fb = {}, {}, {}
    for k, W in Ws.items():
        r = bt.run(W, Rx, C)
        has = W.sum(axis=1) > 0
        nets[k], gross[k] = r.net.where(has).dropna(), r.gross.where(has).dropna()
        h = W > 0
        fb[k] = float((miss & h).sum().sum() / max(1, h.sum().sum()))
    pk["period"] = pk.date.map(period_of)
    pk["hw"] = hwc.reindex(pk.ticker).fillna(False).values
    rows = []
    for (kind, q, v), x in nets.items():
        base = nets[(kind, q, "base")]
        Hb = Ws[(kind, q, "base")] > 0
        H = Ws[(kind, q, v)] > 0
        sub = pk[(pk.frame == kind) & (pk.model == q) & (pk.variant == v)]
        for per, a, b in PERIODS:
            xx = x.loc[a:b]
            if not len(xx):
                continue
            s = sub[sub.period == per]
            ov = (H & Hb).loc[xx.index].sum(axis=1)
            st = ann_stats(xx)
            pername = per if kind != "ML25" else "2025"
            rows.append(dict(table="summary", period=pername, frame="ML" if kind == "ML25" else kind, model=q,
                             variant=v, nights=len(xx), net_bp=1e4 * xx.mean(), sharpe=st["sharpe"], hit=st["hit"],
                             gross_bp=1e4 * gross[(kind, q, v)].loc[a:b].mean(),
                             diff_vs_base_bp=1e4 * (xx - base.reindex(xx.index)).mean(),
                             t_vs_base=paired_t(xx, base.reindex(xx.index)), overlap_base=float(ov.mean()),
                             techhw_picks=float(s.groupby("date").hw.sum().mean()),
                             spearman_base=float(s.groupby("date").spearman_base.first().mean()),
                             fallback_share=fb[(kind, q, v)]))
            for mth, xm in xx.groupby(xx.index.to_period("M")):
                rows.append(dict(table="monthly", period=str(mth), frame=kind, model=q, variant=v, nights=len(xm),
                                 net_bp=1e4 * xm.mean(), diff_vs_base_bp=1e4 * (xm - base.reindex(xm.index)).mean(),
                                 overlap_base=float(ov.loc[xm.index].mean())))
    # 2025 pooled over the four quarterly model sets
    for v in ["base", "b_qmap", "c_clip"]:
        ks = [k for k in nets if k[0] == "ML25" and k[2] == v]
        x = pd.concat([nets[k] for k in ks]).sort_index()
        bs = pd.concat([nets[(k[0], k[1], "base")] for k in ks]).sort_index()
        ov = pd.concat([((Ws[k] > 0) & (Ws[(k[0], k[1], "base")] > 0)).sum(axis=1).loc[nets[k].index] for k in ks])
        s = pk[(pk.frame == "ML25") & (pk.variant == v)]
        st = ann_stats(x)
        rows.append(dict(table="summary", period="2025", frame="ML", model="2025Q1-Q4 (own quarter)", variant=v,
                         nights=len(x), net_bp=1e4 * x.mean(), sharpe=st["sharpe"], hit=st["hit"],
                         diff_vs_base_bp=1e4 * (x - bs.reindex(x.index)).mean(), t_vs_base=paired_t(x, bs.reindex(x.index)),
                         overlap_base=float(ov.mean()), techhw_picks=float(s.groupby("date").hw.sum().mean()),
                         spearman_base=float(s.groupby("date").spearman_base.first().mean())))
    ch = pd.read_parquet(CHG)
    ch["table"] = "input_change"
    ch = ch.rename(columns={"month": "period"})
    df = pd.concat([pd.DataFrame(rows), ch], ignore_index=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    sm = df[df.table == "summary"]
    print(sm[["period", "frame", "model", "variant", "nights", "net_bp", "sharpe", "hit", "diff_vs_base_bp", "t_vs_base",
              "overlap_base", "techhw_picks", "spearman_base", "fallback_share"]].round(2).to_string(index=False))
    mo = df[(df.table == "monthly") & (df.frame.isin(["C", "ML"]))]
    print(mo.pivot_table(index=["frame", "model", "period"], columns="variant", values="net_bp").round(0).to_string())
    print(ch[ch.variant != "base"].groupby(["frame", "variant"])[["beta_changed", "days_to_e_changed",
                                                                  "days_since_e_changed", "days_to_e_nan"]].mean().round(3).to_string())
    print(ch[ch.variant.isin(["base", "b_qmap"]) & (ch.model.isin(["2026Q2", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]))]
          .drop_duplicates(["frame", "variant", "period"])
          .pivot_table(index=["frame", "period"], columns="variant", values=["beta_median", "beta_p10", "beta_p90"]).round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"frames": frames, "score": score, "fetch": fetch, "report": report}[what]()
    import resource
    print("peak memory GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
