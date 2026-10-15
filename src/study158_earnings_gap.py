"""Study 158: the earnings-calendar gap in days_to_e / days_since_e, in training and live.

Question: study 154 found that ml_features.earnings_features() places reports on the panel's trading-day grid and
drops every report dated after the grid's last day (ev.di < len(days)). Live, the grid ends the previous day, so
days_to_e (trading days to the next report, clipped at 15) is NaN for nearly every name and days_since_e is one day
stale. The training frame data/ml_frame.parquet (built 2026-09-27 on a panel through 2026-09-25 and a calendar that
ended in September) has the same hole at its end. (1) How large is the gap by month 2020-01..2026-09 in the frame
the models were trained on, compared with what the current calendar says? (2) A corrected panel on an extended
calendar. (3) Do the 2026Q3 / 2026Q4 models pick differently, and earn differently, on 2026Q3 days (July-September,
the only losing quarter in study 137, -3 bp a night) with the corrected inputs? (4) Would retraining matter: share
of training rows affected, and where the models send a missing days_to_e.

Data: core.load_panel (daily store through 2026-10-13); store.read("earnings") (Nasdaq calendar as of 2026-10-14,
report dates 2019-06-03..2026-11-13); data/ml_frame.parquet (the training frame of the night_ / ens_* members in
src/ensemble.py and of the jump_{jump,drop}_ classifiers in src/study33_jump_live.py, which add earn_tonight and
news_today to the same frame; all 2024Q1..2026Q4 models were trained from this one file on 2026-09-29/30);
data/models (read only); data/local/factor_panels.pkl n_news (news_today as in study 33 training: 0 when none);
official crosses data/local/auctions*.parquet with study 99/94 cross_ratios; data/store/sectors.parquet.
pandas_market_calendars and exchange_calendars are not installed: the grid past the panel end is business days
minus the NYSE holidays 2026-11-26, 2026-12-25, 2027-01-01, 2027-01-18 (to 2027-01-29). Inside the panel the panel's
own trading days are used.
ml_features.py is not imported (its module body builds the full feature panels); earnings_features is loaded from
the source file as in study 154 and run unchanged on the chosen grid.

Design
  corrected  earnings_features on the panel days + the future grid, all stock columns, current calendar. Written to
             data/local/study158_earn.parquet (date, ticker, days_since_e, days_to_e, last_surp; panel days and the
             future grid; rows with all three missing dropped).
  (1) monthly: per month of the frame's rows (the model universe: price > $5, ADV > $5M), share with days_to_e
      missing in the frame, missing in the corrected panel, corrected days_to_e < 15 (a report within the next 14
      trading days, the range the clipped feature distinguishes), and "wrong" = frame missing but corrected present
      (wrong_near: corrected < 15, i.e. a report within 3 weeks that was in the calendar); value mismatches where
      both are present (days_to_e, days_since_e, last_surp differ), e.g. later calendar revisions.
      Live share: the live job's inputs replicated per day (earnings_features on a grid that ends the previous
      trading day, last row reused for the day, as paper_overnight / study 154 Lthen), for 2026Q3 days and the
      paper-trading days 2026-09-29..2026-10-13.
  (3) scoring: close-based frame rows of 2026-07-01..2026-09-25 (ml_frame, as trained) scored by the 2026Q3 and
      2026Q4 model sets (ensemble.predict members: night_, ens_pooled_beh/com, ens_per_beh/com groups; jump_jump /
      jump_drop with earn_tonight from the extended grid (study 33 panel: report day and the day before) and
      news_today), three input variants that differ only in days_to_e and days_since_e:
        orig  as in the frame (= what the 2026Q3 backtest of studies 23/33 saw: same calendar state)
        live  as the live job computes them
        fix   corrected panel
      blend = (2 rank(ensemble) + rank(P(jump) - P(drop))) / 3, top 10 capped at 3 per industry (cap_by_industry, as
      live and study 137) and uncapped top 10. Per variant: overlap of picks with orig and with fix, Spearman of the
      blend, mean net bp a night at panel prices (o(t+1)/c(t)) and at official crosses (study 94 ratio method, legs
      without a cross at the panel price), cost exec_cost_bps(auction) + 2.5 bp per side as study 137, paired t of
      the nightly difference. The 2026Q4 models were trained on rows through 2026-09-16, so their 2026Q3 returns are
      in sample and shown only for pick changes. Reference: study 23/33 15:45 backtest picks for the same nights.
  (4) training rows affected per model quarter 2024Q1..2026Q4 (rows 2020-01-01 .. 10 trading days before the
      quarter, y_night present): share with frame missing / corrected present, and the share in the last 60 trading
      days of the window. Split points: every split on days_to_e in the 2026Q3/2026Q4 models (dump_model): threshold,
      gain, missing direction; a missing value goes with the far side (values above the threshold, i.e. with 15 =
      "no report within 3 weeks") or with the near side. Gain share of days_to_e per model.

Result (1): the frame matches the current calendar to within 0.1-0.3% of rows a month for 2020-01..2026-05 (about
1.5% of rows have no future report in any calendar; same in both). The gap is the frame's end: days_to_e missing
for 3% (2026-06), 11% (07), 84% (08), 99% (09) of rows, of which 0.8 / 8.5 / 78 / 87 points had a later report in
today's calendar. Almost none of them (0-1%) had a report within 15 trading days: the frame says "missing" where
the truth is mostly the clipped 15. days_since_e and last_surp match exactly. Live: days_to_e missing for 99.7-100%
of the universe every day 2026-07..10-13 (truth: 2-41% with a report within 15 days, 12-14% missing because the
calendar ends 2026-11-13); days_since_e off by one day for 97% of names.
Result (3), 2026Q3 models on 2026Q3 close frames, capped, at crosses: orig 4.2 bp a night, fix 6.9 (+2.6, t 0.75),
live 3.1 (-1.1, t -0.10). Picks changed vs orig: fix 1.3 of 10 (0.5 in July, 2.6 in September), live 2.7; live vs
fix 3.6. The 15:45 backtest reference reproduces study 137 (-3.1 bp); close-frame picks share only 6.4 of 10 with it.
2026Q4 models are in sample on these days (about +200 bp a night), only pick changes are meaningful (fix vs orig
1.4, live vs fix 2.9).
Result (4): training rows with days_to_e missing but a later report known: 0.08% for the 2024Q1..2025Q4 models,
0.11% for 2026Q3, 2.1% for 2026Q4 (45% of its last 60 days); within-15-day cases 0.00-0.06%. days_to_e is 10-12% of
the gain of the jump models and 0.1-1.4% of the ensemble members. In the jump models 85-94% of the days_to_e gain
is in splits whose missing branch is the far side (with values above the threshold, mostly at 1.5), so a missing
value is read mostly like "no report in the next day or two"; 2-17% of the gain sends it to the near side.

LightGBM: one process, predict with num_threads=2 (Booster.predict wrapped), no fork.
Output: results/study158_earnings_gap.csv (table = monthly / live_share / train_affected / splits / importance /
score / overlap_day), data/local/study158_earn.parquet.

    python src/study158_earnings_gap.py
"""
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import store   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
MD = os.path.join(DATA, "models")
ML = os.path.join(DATA, "ml_frame.parquet")
OUT = os.path.join(RES, "study158_earnings_gap.csv")
EARN_OUT = os.path.join(LOC, "study158_earn.parquet")
HOL = pd.DatetimeIndex(["2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18"])
Q3 = ("2026-07-01", "2026-09-30")
EK = ["days_since_e", "days_to_e", "last_surp"]


def load_mlf():
    tree = ast.parse(open(os.path.join(SRC, "ml_features.py")).read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "earnings_features"]
    ns = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=fns, type_ignores=[]), "ml_features.py", "exec"), ns)
    return ns


def ext_grid(days):
    fut = pd.bdate_range(days[-1] + pd.Timedelta(days=1), "2027-01-29")
    return days.append(fut.difference(HOL))


def earn_feats(ns, grid, cols, E):
    ns.update(days=grid, cols=cols, _E=E[E.symbol.isin(cols)])
    return ns["earnings_features"]()


def lookup(panel, dates, tick):
    r = panel.index.get_indexer(dates)
    c = panel.columns.get_indexer(tick)
    ok = (r >= 0) & (c >= 0)
    v = np.full(len(dates), np.nan, dtype="float32")
    v[ok] = panel.values[r[ok], c[ok]]
    return v


def live_inputs(ns, days, E, d, tick):
    """The live job's days_since_e / days_to_e / last_surp for day d: grid ends the previous trading day."""
    i = days.get_loc(d)
    g = days[max(0, i - 300):i]
    e = E[pd.to_datetime(E.date) >= g[0]]
    f = earn_feats(ns, g, list(tick), e)
    return {k: f[k].iloc[-1] for k in EK}


def patch_lgb():
    import lightgbm as lgb
    orig = lgb.Booster.predict

    def pred(self, data, *a, **kw):
        kw.setdefault("num_threads", 2)
        return orig(self, data, *a, **kw)
    lgb.Booster.predict = pred
    return lgb


# ------------------------------------------------------------------ main
def main():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    grid = ext_grid(days)
    ns = load_mlf()
    E = store.read("earnings")
    E = E[E.symbol.isin(cols)]
    print("calendar", E.date.min().date(), E.date.max().date(), len(E), "grid end", grid[-1].date(), flush=True)
    C = earn_feats(ns, grid, cols, E)
    C = {k: v.astype("float32") for k, v in C.items()}
    long = pd.concat({k: v.stack() for k, v in C.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long = long.dropna(how="all").reset_index()
    long.to_parquet(EARN_OUT)
    print("wrote", EARN_OUT, len(long), flush=True)
    del long
    rows = []

    # (1) frame vs corrected by month
    f = pq.ParquetFile(ML)
    fr = f.read(columns=["date", "ticker", "days_to_e", "days_since_e", "last_surp", "y_night"]).to_pandas(ignore_metadata=True)
    fr = fr[fr.date >= "2020-01-01"].reset_index(drop=True)
    dts, tk = pd.DatetimeIndex(fr.date), fr.ticker.values
    for k in EK:
        fr[k + "_c"] = lookup(C[k], dts, tk)
    fr["m"] = dts.to_period("M")
    fr["fmiss"] = fr.days_to_e.isna()
    fr["cmiss"] = fr.days_to_e_c.isna()
    fr["wrong"] = fr.fmiss & ~fr.cmiss
    fr["wrong_near"] = fr.fmiss & (fr.days_to_e_c < 15)
    fr["near_c"] = fr.days_to_e_c < 15
    fr["near_f"] = fr.days_to_e < 15
    both = ~fr.fmiss & ~fr.cmiss
    fr["mis_to"] = both & ((fr.days_to_e - fr.days_to_e_c).abs() > 0.5)
    fr["mis_since"] = (fr.days_since_e.notna() | fr.days_since_e_c.notna()) & \
        ~((fr.days_since_e - fr.days_since_e_c).abs() <= 0.5)
    fr["mis_surp"] = (fr.last_surp.notna() | fr.last_surp_c.notna()) & \
        ~((fr.last_surp - fr.last_surp_c).abs() <= 1e-3)
    g = fr.groupby("m")
    mon = pd.DataFrame({"n": g.size(), "frame_missing": g.fmiss.mean(), "corrected_missing": g.cmiss.mean(),
                        "report_within_15_true": g.near_c.mean(), "report_within_15_frame": g.near_f.mean(),
                        "wrong": g.wrong.mean(), "wrong_near": g.wrong_near.mean(), "days_to_e_mismatch": g.mis_to.mean(),
                        "days_since_e_mismatch": g.mis_since.mean(), "last_surp_mismatch": g.mis_surp.mean()})
    for m, r in mon.iterrows():
        rows.append(dict(table="monthly", period=str(m), **r.to_dict()))
    print(mon.round(3).to_string(), flush=True)

    # (4a) training rows affected per model quarter
    ok = fr.y_night.notna().values
    for q in pd.period_range("2024Q1", "2026Q4", freq="Q"):
        cut = days[max(0, days.searchsorted(q.start_time) - 11)]
        tr = ok & np.asarray(dts < cut)
        last60 = tr & np.asarray(dts >= days[days.searchsorted(cut) - 60])
        rows.append(dict(table="train_affected", period=str(q), n=int(tr.sum()), cut=str(cut.date()),
                         wrong=fr.wrong.values[tr].mean(), wrong_near=fr.wrong_near.values[tr].mean(),
                         frame_missing=fr.fmiss.values[tr].mean(), corrected_missing=fr.cmiss.values[tr].mean(),
                         days_since_e_mismatch=fr.mis_since.values[tr].mean(),
                         wrong_last60=fr.wrong.values[last60].mean(), wrong_near_last60=fr.wrong_near.values[last60].mean(),
                         n_wrong=int(fr.wrong.values[tr].sum())))
    yr = dts.year.values
    for y0, y1 in [(2024, 2026), (2020, 2026)]:
        s = ok & (yr >= y0) & (yr <= y1)
        rows.append(dict(table="train_affected", period=f"{y0}-{y1} all rows", n=int(s.sum()),
                         wrong=fr.wrong.values[s].mean(), wrong_near=fr.wrong_near.values[s].mean()))
    del fr

    # (3) 2026Q3 frame rows, three variants
    lgb = patch_lgb()
    import ensemble
    import jumpmodel as J
    from paper_overnight import cap_by_industry
    X = pq.read_table(ML, filters=[("date", ">=", pd.Timestamp(Q3[0])), ("date", "<=", pd.Timestamp(Q3[1]))]).to_pandas(ignore_metadata=True)
    X = X.set_index(["date", "ticker"]).sort_index()
    X = X[[k for k in X.columns if not k.startswith("y_")]]
    qd = X.index.get_level_values(0).unique()
    xd, xt = X.index.get_level_values(0), X.index.get_level_values(1)
    print("Q3 rows", len(X), "days", len(qd), flush=True)
    V = {"orig": X}
    Xf = X.copy()
    for k in ["days_to_e", "days_since_e"]:
        Xf[k] = lookup(C[k], xd, xt)
    V["fix"] = Xf
    Xl = X.copy()
    lv = {k: np.full(len(X), np.nan, dtype="float32") for k in EK}
    pos = pd.Series(np.arange(len(X)), index=X.index)
    tick_all = cols
    live_days = list(qd) + list(days[(days >= "2026-09-29") & (days <= "2026-10-13")])
    for d in live_days:
        L = live_inputs(ns, days, E, d, tick_all)
        if d in qd:
            ii = pos.loc[d]
            for k in EK:
                lv[k][ii.values] = L[k].reindex(ii.index).values
        # live share: model universe of the day (frame rows; for October the liquid names by ADV / price)
        if d in qd:
            uni = list(X.loc[d].index)
        else:
            i = days.get_loc(d)
            dv = P["dv"][cols].iloc[i - 20:i].median()
            uni = list(dv.index[(dv > 5e6) & (P["rawc"][cols].iloc[i - 1] > 5)])
        tr_to = C["days_to_e"].loc[d].reindex(uni)
        tr_si = C["days_since_e"].loc[d].reindex(uni)
        rows.append(dict(table="live_share", period=str(d.date()), n=len(uni),
                         live_missing=float(L["days_to_e"].reindex(uni).isna().mean()),
                         corrected_missing=float(tr_to.isna().mean()),
                         report_within_15_true=float((tr_to < 15).mean()),
                         frame_missing=float(X.loc[d].days_to_e.isna().mean()) if d in qd else np.nan,
                         days_since_e_stale=float(((L["days_since_e"].reindex(uni) - tr_si).abs() > 0.5).mean())))
    for k in ["days_to_e", "days_since_e"]:
        Xl[k] = lv[k]
    V["live"] = Xl
    ls = pd.DataFrame([r for r in rows if r["table"] == "live_share"])
    print(ls.round(3).tail(14).to_string(), flush=True)

    # extra inputs for the jump models
    ci = {t: i for i, t in enumerate(cols)}
    di = grid.searchsorted(pd.to_datetime(E.date))
    et = np.zeros((len(grid), len(cols)), dtype="float32")
    for d_, t_ in zip(di, E.symbol):
        if 0 < d_ < len(grid):
            et[d_, ci[t_]] = 1.0
            et[d_ - 1, ci[t_]] = 1.0
    ET = pd.DataFrame(et, index=grid, columns=cols)
    NW = pd.read_pickle(os.path.join(LOC, "factor_panels.pkl"))["n_news"]
    earn_t = lookup(ET, xd, xt)
    news_t = lookup(NW.reindex(columns=cols), xd, xt)

    S = {}
    for q in ["2026Q3", "2026Q4"]:
        for vn, Xv in V.items():
            ens = ensemble.predict(Xv, q)
            Xj = Xv.copy()
            Xj["earn_tonight"], Xj["news_today"] = earn_t, news_t
            out = {}
            for k in ["jump", "drop"]:
                m = lgb.Booster(model_file=os.path.join(MD, f"jump_{k}_{q}.txt"))
                out[k] = m.predict(Xj[m.feature_name()])
            sc = pd.DataFrame({"ensemble": ens.ensemble.values, "p_jump": out["jump"], "p_drop": out["drop"]},
                              index=Xv.index)
            sc["blend"] = (2 * sc.ensemble.groupby(level=0).rank(pct=True)
                           + (sc.p_jump - sc.p_drop).groupby(level=0).rank(pct=True)) / 3
            S[(q, vn)] = sc
            print("scored", q, vn, flush=True)

    # returns at panel and crosses (study 137 method)
    import study99_cross_retrain as S99
    parts = [S99.crosses()]
    for fn in sorted(glob.glob(os.path.join(LOC, "auctions*.parquet"))):
        if os.path.basename(fn) in ("auctions94.parquet", "auctions88.parquet", "auctions99.parquet"):
            continue
        c = pd.read_parquet(fn)
        if not {"ticker", "date", "kind", "px"} <= set(c.columns):
            continue
        c = c[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
        c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
        c["date"] = pd.to_datetime(c.date)
        parts.append(c)
    AX = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    ro, rc, err = S99.cross_ratios(P, cols, days, AX)
    del AX
    print("cross errors", err, flush=True)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=qd)
    Rx = (1 + Rp) * ro.shift(-1).reindex(index=qd).fillna(1.0) / rc.reindex(index=qd).fillna(1.0) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=qd)
    Cst = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=qd)

    def weights(blend, capped):
        W = pd.DataFrame(0.0, index=qd, columns=cols)
        pk = {}
        for d in qd:
            s = blend.loc[d].sort_values(ascending=False, kind="stable")
            sel = list(cap_by_industry(s, 10, 3).index) if capped else list(s.index[:10])
            pk[d] = sel
            W.loc[d, sel] = 1.0 / len(sel)
        return W, pk

    # reference: study 23/33 15:45 backtest blend
    pr = pd.read_parquet(f"{RES}/study33_pred.parquet")
    pr = pr[(pr.index.get_level_values(0) >= Q3[0]) & (pr.index.get_level_values(0) <= Q3[1])]
    en = pd.read_parquet(f"{RES}/study23_pred.parquet", columns=["ensemble"])
    en = en[(en.index.get_level_values(0) >= Q3[0]) & (en.index.get_level_values(0) <= Q3[1])]["ensemble"]
    j = pd.concat([en, pr.p_jump, pr.p_drop], axis=1, join="inner").dropna()
    bt1545 = ((2 * j.ensemble.groupby(level=0).rank(pct=True)
               + (j.p_jump - j.p_drop).groupby(level=0).rank(pct=True)) / 3)
    del pr, en, j

    nets = {}
    picks = {}
    for capped in [True, False]:
        cap = "capped" if capped else "uncapped"
        Wr, pkr = weights(bt1545, capped)
        picks[(cap, "ref", "bt1545")] = pkr
        for pn, R in [("panel", Rp), ("cross", Rx)]:
            nets[(cap, "ref", "bt1545", pn)] = bt.run(Wr, R, Cst).net
        for (q, vn), sc in S.items():
            W, pk = weights(sc.blend, capped)
            picks[(cap, q, vn)] = pk
            for pn, R in [("panel", Rp), ("cross", Rx)]:
                nets[(cap, q, vn, pn)] = bt.run(W, R, Cst).net
            h = W > 0
            nets[(cap, q, vn, "fallback")] = float((miss & h).sum().sum() / h.sum().sum())
    for (cap, q, vn, pn), x in list(nets.items()):
        if pn == "fallback":
            continue
        x = x.dropna()
        st = ann_stats(x)
        r = dict(table="score", variant=f"{q}_{vn}", capped=cap, pricing=pn, n=len(x), net_bp=1e4 * x.mean(),
                 sharpe=st["sharpe"], hit=float((x > 0).mean()), fallback=nets.get((cap, q, vn, "fallback")))
        if q != "ref" and vn != "orig":
            dlt = x - nets[(cap, q, "orig", pn)].reindex(x.index)
            r.update(diff_vs_orig_bp=1e4 * dlt.mean(), t_diff=dlt.mean() / (dlt.std() / np.sqrt(len(dlt))))
        if q != "ref":
            r["overlap_vs_orig"] = np.mean([len(set(picks[(cap, q, vn)][d]) & set(picks[(cap, q, "orig")][d])) for d in qd])
            r["overlap_vs_fix"] = np.mean([len(set(picks[(cap, q, vn)][d]) & set(picks[(cap, q, "fix")][d])) for d in qd])
            r["overlap_vs_bt1545"] = np.mean([len(set(picks[(cap, q, vn)][d]) & set(picks[(cap, "ref", "bt1545")][d]))
                                              for d in qd])
            a, b = S[(q, vn)].blend, S[(q, "orig")].blend
            r["spearman_vs_orig"] = float(np.mean([a.loc[d].rank().corr(b.loc[d].rank()) for d in qd]))
        rows.append(r)
    # per-day overlap (capped)
    for q in ["2026Q3", "2026Q4"]:
        for d in qd:
            po, pf, pl = (set(picks[("capped", q, v)][d]) for v in ["orig", "fix", "live"])
            rows.append(dict(table="overlap_day", variant=q, period=str(d.date()),
                             frame_missing=float(X.loc[d].days_to_e.isna().mean()),
                             overlap_fix_orig=len(pf & po), overlap_live_orig=len(pl & po), overlap_live_fix=len(pl & pf),
                             net_orig_bp=1e4 * nets[("capped", q, "orig", "cross")].get(d, np.nan),
                             net_fix_bp=1e4 * nets[("capped", q, "fix", "cross")].get(d, np.nan),
                             net_live_bp=1e4 * nets[("capped", q, "live", "cross")].get(d, np.nan)))
    # by month inside Q3 (capped, crosses)
    for q in ["2026Q3", "2026Q4"]:
        for vn in ["orig", "fix", "live"]:
            x = nets[("capped", q, vn, "cross")].dropna()
            for m, xm in x.groupby(x.index.to_period("M")):
                rows.append(dict(table="score_month", variant=f"{q}_{vn}", capped="capped", pricing="cross",
                                 period=str(m), n=len(xm), net_bp=1e4 * xm.mean()))

    # (4b) split points and missing direction
    for q in ["2026Q3", "2026Q4"]:
        names = ([f"night_{q}.txt", f"ens_pooled_beh_{q}.txt", f"ens_pooled_com_{q}.txt", f"jump_jump_{q}.txt",
                  f"jump_drop_{q}.txt"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(MD, f"ens_per_*_{q}_g*.txt"))))
        for nm in names:
            m = lgb.Booster(model_file=os.path.join(MD, nm))
            fn = m.feature_name()
            gain = pd.Series(m.feature_importance("gain"), index=fn)
            share = {k: float(gain.get(k, 0) / gain.sum()) for k in ["days_to_e", "days_since_e", "last_surp", "earn_tonight"]}
            sp = []

            def walk(n):
                if "split_feature" not in n:
                    return
                if fn[n["split_feature"]] in ("days_to_e", "days_since_e"):
                    sp.append(dict(f=fn[n["split_feature"]], thr=n["threshold"], gain=n["split_gain"],
                                   cnt=n["internal_count"], mt=n["missing_type"], dl=n["default_left"]))
                walk(n["left_child"])
                walk(n["right_child"])
            for t in m.dump_model()["tree_info"]:
                walk(t["tree_structure"])
            sp = pd.DataFrame(sp)
            row = dict(table="importance", variant=nm[:-4], period=q, **{f"gain_share_{k}": v for k, v in share.items()})
            for feat in ["days_to_e", "days_since_e"]:
                s = sp[sp.f == feat] if len(sp) else sp
                if not len(s):
                    continue
                # far side = values above the threshold (larger = further from / longer since the report)
                nan_far = np.where(s.mt.isin(["NaN", "Zero"]), ~s.dl.astype(bool), s.thr < 0)
                row[f"{feat}_splits"] = len(s)
                row[f"{feat}_nan_far_gainshare"] = float(s.gain[nan_far].sum() / s.gain.sum())
                row[f"{feat}_missing_type_nan_share"] = float((s.mt == "NaN").mean())
            rows.append(row)
            if nm.startswith(("night_", "jump_", "ens_pooled")):
                s = sp[sp.f == "days_to_e"] if len(sp) else sp
                if len(s):
                    s = s.assign(nan_side=np.where(s.mt.isin(["NaN", "Zero"]), np.where(s.dl, "near (<=thr)", "far (>thr)"),
                                                   "as 0 (near)"))
                    for (thr, side), gg in s.groupby([s.thr.round(2), "nan_side"]):
                        rows.append(dict(table="splits", variant=nm[:-4], period=q, threshold=thr, nan_side=side,
                                         n=len(gg), gain=float(gg.gain.sum()), count=int(gg.cnt.sum())))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, flush=True)
    sc = df[df.table == "score"]
    print(sc[["variant", "capped", "pricing", "n", "net_bp", "sharpe", "diff_vs_orig_bp", "t_diff", "overlap_vs_orig",
              "overlap_vs_fix", "overlap_vs_bt1545", "spearman_vs_orig", "fallback"]].round(3).to_string(), flush=True)
    print(df[df.table == "train_affected"].dropna(axis=1, how="all").round(4).to_string(), flush=True)
    print(df[df.table == "importance"].dropna(axis=1, how="all").round(3).to_string(), flush=True)
    print(df[df.table == "splits"].dropna(axis=1, how="all").to_string(), flush=True)
    print(df[df.table == "score_month"].dropna(axis=1, how="all").round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()
