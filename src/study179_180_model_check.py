"""Studies 179 and 180: do the backtest models lose on the October 2026 live nights, and did the 2026Q3 refit change
what the blend picks?

Background: the live blend (study 33: (2 x within-day rank of the 5-model ensemble + rank of P(jump) - P(drop)) / 3,
top 10, at most 3 per industry, closing-auction buy, opening-auction sell) lost about 12.5% over 12 paper nights
2026-09-29..10-15 at official auction prints (logs/paper/virtual_days.csv). The backtest at official crosses (study
135 build(), study 94 method) was weak from July 2026 (-3 bp a night vs +42 bp before). Study 178: about half of the
live gap is a market-wide overnight reversal in high-beta / tech hardware stocks in October; the rest is unexplained.

179  October rescore. Question: with the panel now running to 2026-10-15, do backtest-style scorings of the entry days
     2026-09-28..10-14 with the frozen 2026Q4 models lose as much as the live virtual book, and how much do their picks
     overlap with the live picks?
180  What changed in July. Question: on the same test days (2026Q3: 2026-07-01..09-25; 2026Q2 for contrast; October
     2026-09-28..10-14 as well), do older quarterly models (2025Q3, 2026Q1, 2026Q2) do better than the 2026Q3 refit?
     If they do, the refit is implicated; if all do poorly, it is the market. Also: gain importance per model and the
     share of tech hardware picks per model on the same days.

Data
  core.load_panel (Yahoo daily, split/dividend adjusted, ends 2026-10-15; last night 10-14 -> 10-15). Models (read
  only) data/models/night_<q>, ens_*_<q>, jump_{jump,drop}_<q>; ensemble.predict and the jump models as jumpmodel.py.
  Feature frames are built with the study 154 replica (src/study150_154_live_check.Ctx, ml_features' build /
  features_frame / earnings_features on a 260-day window; earnings calendar data/store/earnings on a grid that extends
  past the day, as in training):
    B  backtest 15:45 frame of study 23 / 33: daily open/high/low, SIP 15:40-15:45 5-minute bar close as the price,
       85% of the day's volume, SPY/IWM 15:44 minute, VIX close of the day. Bars from data/local/study154_bars.parquet
       (live days) plus data/local/study179_bars.parquet (fetched here for 2026-09-28, 09-30, 10-14).
    C  daily close features (the full daily row of the day).
  Extra jump-model inputs: earn_tonight (study 33 panel: 1 on the report's trading day and the day before) and
  news_today (articles tagged with 1-3 symbols in (15:45 previous day, 15:45 day]; data/local/factor_panels.pkl
  n_news to 2026-09-25, then data/local/study154_news.parquet and study179_news.parquet fetched here the same way;
  0 when none, as study 154 B/C).
  Second frame source for 180 (check): data/ml_frame.parquet close features (ends 2026-09-25), whose days_to_e is
  missing for 44% of rows in August-September 2026 (study 158 gap) against about 2% in the rebuilt C frames.
  Live: logs/paper/entry_<day>.json (capped picks, model used), logs/paper/virtual_days.csv (net at official prints,
  by entry day; no entry 2026-09-30). Official crosses: every cached data/local/auctions*.parquet plus
  data/local/auctions179.parquet fetched here (study 94 rule: listing-venue 'O' / '6' print in widening windows,
  second pass the largest 'O' / '6' print of any listing exchange; Alpaca SIP trades, 60 requests a minute).
  Tech hardware: study 90/92 definition (data/store/sectors.parquet snapshot).

Design 179: per entry day 2026-09-28..10-14 (13 days), frames B and C scored with the 2026Q4 ensemble and jump models
  (the live job used them from 10-01; 09-29 ran --strategy=ensemble with the 2026Q3 models; the 2026Q4 training data
  ends 2026-09-16, so all 13 days are out of sample). Picks as live: blend on the scored universe, Alpaca-tradable
  names, top 10 with at most 3 per industry. Night return at crosses as study 94 (panel night return x open cross /
  panel open on t+1 / (close cross / panel close on t); legs without a usable cross at the panel price), equal
  weight, cost core.exec_cost_bps(P, 'auction') + 2.5 bp per side. The live picks are repriced the same way
  (live_repriced) to check the pricing against the virtual book. Per night: net, gross, tech hardware count, overlap
  with the live picks; liquid-universe EW overnight return at panel prices as market context. Summary over the 11
  nights with a live entry and over all 13.
Design 180: frames C (main) and ml_frame (check) for every trading day of 2026Q2 and 2026-07-01..09-25, C for the 13
  October days. Model sets q = 2025Q3, 2026Q1, 2026Q2, 2026Q3, 2026Q4 (ensemble members and jump models of the same
  quarter; ensemble groups of q's year as trained). A model set is scored only on days on or after its training cut
  (10 trading days before the quarter start), so newer models never score their own training days: 2026Q2 days get
  2025Q3 / 2026Q1 / 2026Q2, 2026Q3 days add 2026Q3, October adds 2026Q4. Blend, cap and pricing as 179 but without
  the tradable filter (as the backtest, study 135 build()). Per period and model set: nights, mean net bp, Sharpe,
  hit rate, compounded net, paired t of the nightly difference against the period's own-quarter model, mean overlap
  with the own-quarter picks, mean Spearman of the blend scores with the own-quarter scores, tech hardware share of
  the capped picks, of the ensemble top 10 and of the P(jump)-P(drop) top 10, and monthly means. Reference: study 135
  build() capped net (15:45 frame, own-quarter models) on the same nights. Gain importance (normalized to 100%) of
  the pooled members (night_, ens_pooled_beh, ens_pooled_com) and jump_jump / jump_drop for each q; the features
  whose share moved most between 2026Q2 and 2026Q3.

Data checks: the live picks repriced here (panel x cross ratios, model cost) give -101.8 bp a night over the 11 live
  nights against -102.2 bp in the virtual book (nightly correlation 1.00). The C frame with the own-quarter models
  reproduces the backtest: 2026Q2 +63.9 bp vs study 135 build() +63.9 bp; 2026Q3 +4.0 vs -3.1 bp (nightly
  correlation 0.96 with the 15:45 backtest). 1,711 new legs fetched into auctions179 (99.8% found); legs at the panel
  price 0.0-0.7% per variant. days_to_e missing: rebuilt C frames 2-5% (Apr-Aug), 12-14% (Sep-Oct); ml_frame 2-3%
  (Apr-Jun), 11% (Jul), 84% (Aug), 99% (Sep). Peak memory 2.2 GB (report), 1.4 GB (frames).

Result 179 (11 nights with a live entry, net bp a night at crosses; 13 nights 09-28..10-14 in brackets): live
  virtual book -102.2 (compounded -10.9%); B with 2026Q4 models -95.3 [-77.5], C with 2026Q4 -85.9 [-69.2]; with the
  2026Q3 models B -76.6 [-61.8], C -66.6 [-49.7]. Nightly correlation of B 2026Q4 with live 0.95, paired t of the
  difference 0.30. Overlap with the live picks 4.9 of 10 (B) and 4.6 (C) on average (0 on 09-29, when live ran the
  ensemble only; 0-5 on 10-14, when live lost 501 bp and B 556 bp). Tech hardware 3.8 picks a night in B and C, 3.5
  live. Live picks sit at median rank 11 (B) and 13 (C) of about 2,580 names. Liquid universe overnight +2.0 bp a
  night on the live nights, liquid tech hardware -59.9 bp. So the backtest models, scored as in the backtest and priced
  the same way, lose about as much as the live book on the same nights; the live job is not the cause.
Result 180 (C frame, capped top 10 at crosses, net bp a night; own-quarter model first): 2026Q2 days: 2026Q2 +63.9,
  2026Q1 +52.3 (t vs own -0.7), 2025Q3 +13.7 (t -2.4). 2026Q3 days: 2026Q3 +4.0, 2026Q2 +12.9 (t +0.7), 2026Q1 +4.8
  (t +0.1), 2025Q3 +16.4 (t +0.6). October (13 nights): 2026Q4 -69.2, 2026Q3 -49.7, 2026Q2 -66.8, 2026Q1 -36.2, 2025Q3
  -62.4 (all t within +-0.8 of own). ml_frame features give the same picture (2026Q3 days: +4.0, +18.3, +5.8, +24.3).
  Months, C: July -18 (2026Q3) / -13 / +2 / +26 (older), August +41 / +58 / +21 / +33, September +12 / +4 / +11 / -24.
  In 2026Q2 a stale model costs return (2025Q3 50 bp a night below the own model); in 2026Q3 and October every model
  set is weak and the differences are small and not significant. Tech hardware share of capped picks on 2026Q3 days:
  2026Q3 28%, 2026Q2 26%, 2026Q1 23%, 2025Q3 23%; October 34% / 33% / 27% / 17% (2026Q4 34%); in the ensemble top 10
  in October 45% (2026Q3) vs 12% (2025Q3). Gain importance vectors of the 2026Q3 pooled members correlate 0.995-0.999
  with every older quarter (jump models 0.97-0.99); the largest Q2 -> Q3 shifts are jump_jump range1 -2.4 and range20
  +2.3 points, jump_drop vix +2.4, vix_term -1.9 points; 2026Q4 jump_drop moved night1 from 1.9 to 5.9%.
Reading: the 2026Q3 refit is not what made July-October weak: models up to five quarters older did not do
  significantly better on the same days, and the importance profiles barely moved. All model sets lose in October,
  so the weakness is in the market the strategy trades (study 178: high-beta / tech hardware overnight reversal), and
  the part study 178 left unexplained is the models' own selection in that market, shared by every model vintage.
  Newer models lean more into tech hardware in October (34% vs 17% for 2025Q3), which matches the larger losses of
  the newest sets, but the gap between vintages is within noise over 13 nights.
Caveats: 11-13 October nights and 61 Q3 nights are small samples; paired t ignores the correlation of picks across
  nights. Older models are stale but scored with their own year's ensemble groups. sectors.parquet and the Alpaca
  tradable list are current snapshots. B uses the day's full high / low (as study 23), not the live 15:45 values.
  News for October comes from fresh Alpaca queries, not the scored news archive. Costs are the model cost, not the
  realized cost.

Output: results/study179_october_rescore.csv (table = nightly / summary / live_pick_rank),
  results/study180_july_models.csv (table = period_model / monthly / importance / importance_shift / days_to_e)
    python src/study179_180_model_check.py fetchbars   # 15:40 bars and news for the October days not cached
    python src/study179_180_model_check.py frames      # feature frames -> data/local/study179_frames.parquet
    python src/study179_180_model_check.py score       # all model sets -> data/local/study179_scores.parquet
    python src/study179_180_model_check.py fetch       # official crosses of all picks -> auctions179.parquet
    python src/study179_180_model_check.py report      # both studies
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("ALPACA_PER_MIN", "60")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, ann_stats, RES, DATA   # noqa: E402

A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
LOC = os.path.join(DATA, "local")
MD = os.path.join(DATA, "models")
LOG = os.path.join(A.ROOT, "logs", "paper")
BARS179 = os.path.join(LOC, "study179_bars.parquet")
NEWS179 = os.path.join(LOC, "study179_news.parquet")
FRAMES = os.path.join(LOC, "study179_frames.parquet")
SCORES = os.path.join(LOC, "study179_scores.parquet")
PICKS = os.path.join(LOC, "study179_picks.parquet")
AUC179 = os.path.join(LOC, "auctions179.parquet")
OUT179 = os.path.join(RES, "study179_october_rescore.csv")
OUT180 = os.path.join(RES, "study180_july_models.csv")
OCT = ("2026-09-28", "2026-10-14")
Q2 = ("2026-04-01", "2026-06-30")
Q3 = ("2026-07-01", "2026-09-25")
QS = ["2025Q3", "2026Q1", "2026Q2", "2026Q3", "2026Q4"]
PERIODS = [("2026Q2", *Q2), ("2026Q3", *Q3), ("2026-10", *OCT)]


def own_q(period):
    return {"2026Q2": "2026Q2", "2026Q3": "2026Q3", "2026-10": "2026Q4"}[period]


def train_cut(q, days):
    """First day after the training data of quarter q's models (rows before days[i - 11] are training)."""
    return days[max(0, days.searchsorted(pd.Period(q, freq="Q").start_time) - 11)]


def live_entries():
    out = {}
    for f in sorted(glob.glob(os.path.join(LOG, "entry_*.json"))):
        if "replay" in f:
            continue
        r = json.load(open(f))
        if r.get("submitted"):
            out[r["day"]] = r
    return out


# ------------------------------------------------------------------ fetch bars / news
def fetchbars():
    import study150_154_live_check as S154
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    oct_days = days[(days >= OCT[0]) & (days <= OCT[1])]
    have = pd.read_parquet(S154.BARS)
    hn = pd.read_parquet(S154.NEWS)
    mine = pd.read_parquet(BARS179) if os.path.exists(BARS179) else pd.DataFrame(columns=["day"])
    mn = pd.read_parquet(NEWS179) if os.path.exists(NEWS179) else pd.DataFrame(columns=["day"])
    parts, nparts = [mine], [mn]
    for day in oct_days:
        if day not in set(have.day[have.tf == "5Min"]) and day not in set(mine.day):
            _, cand = S154.universe_sets(P, cols, day)
            tick = sorted(set(cand) | {"SPY", "IWM"})
            x = pd.concat([A.bars(tick[i:i + 400], "5Min", S154.utc(day, "15:40"), S154.utc(day, "15:45"))
                           for i in range(0, len(tick), 400)])
            x["day"], x["tf"] = day, "5Min"
            y = A.bars(["SPY", "IWM"], "1Min", S154.utc(day, "15:44"), S154.utc(day, "15:45"))
            y["day"], y["tf"] = day, "1Min"
            parts += [x, y]
            print(day.date(), "bars", len(x), len(y), flush=True)
        if day not in set(hn.day) and day not in set(mn.day):
            prev = days[days < day][-1]
            n = S154.news_counts(S154.utc(prev, "15:45"), S154.utc(day, "15:45"))
            nparts.append(pd.DataFrame({"day": day, "ticker": list(n), "n": list(n.values())}))
            print(day.date(), "news symbols", len(n), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(BARS179)
    pd.concat(nparts, ignore_index=True).to_parquet(NEWS179)


# ------------------------------------------------------------------ frames
def frames():
    import study150_154_live_check as S154
    ctx = S154.Ctx()
    P, days = ctx.P, ctx.days
    bars = pd.concat([ctx.bars, pd.read_parquet(BARS179)], ignore_index=True)
    news_oct = pd.concat([ctx.news, pd.read_parquet(NEWS179)], ignore_index=True)
    nn = pd.read_pickle(os.path.join(LOC, "factor_panels.pkl"))["n_news"]
    test = [d for d in days if (Q2[0] <= str(d.date()) <= Q2[1]) or (Q3[0] <= str(d.date()) <= Q3[1])
            or (OCT[0] <= str(d.date()) <= OCT[1])]
    out = []
    t0 = time.time()
    bc_cache = FRAMES.replace(".parquet", "_bc.parquet")      # B / C frames, kept so the ml_frame step can rerun
    if os.path.exists(bc_cache):
        test = []
        out = [pd.read_parquet(bc_cache)]
    for d in test:
        _, cand = S154.universe_sets(P, ctx.cols, d)
        if d <= nn.index[-1]:
            nw_all = nn.loc[d]
        else:
            n = news_oct[news_oct.day == d]
            nw_all = pd.Series(n.n.values, index=n.ticker.values, dtype="float64")
        kinds = ["C"] + (["B"] if str(d.date()) >= OCT[0] else [])
        for kind in kinds:
            if kind == "B":
                b = bars[bars.day == d]
                hm = b.ts.dt.strftime("%H:%M")
                p5 = b[(b.tf == "5Min") & (hm == "15:40")].groupby("ticker").c.last()
                m1 = b[(b.tf == "1Min") & (hm == "15:44")].groupby("ticker").c.last()
                intr = pd.DataFrame({"p1545": p5.reindex(cand)})
                X = ctx.frame(d, "B", cand, intr=intr, m1=m1)
            else:
                X = ctx.frame(d, "C", cand)
            t = list(X.index.get_level_values(1))
            X = X.astype("float32")
            X["earn_tonight"] = ctx.earn_tonight_ext(t).loc[d].reindex(t).fillna(0.0).values.astype("float32")
            X["news_today"] = nw_all.reindex(t).fillna(0.0).values.astype("float32")
            X["kind"] = kind
            out.append(X)
        print(d.date(), len(X), f"{time.time() - t0:.0f}s", flush=True)
    F = pd.concat(out)
    F.index = F.index.set_names(["date", "ticker"])
    F.to_parquet(bc_cache)
    # ml_frame close features of the same 2026Q2/Q3 days (check), with the same extra inputs as the C frame rows
    import pyarrow.parquet as pq
    ml = pq.read_table(os.path.join(DATA, "ml_frame.parquet"),
                       filters=[("date", ">=", pd.Timestamp(Q2[0])), ("date", "<=", pd.Timestamp(Q3[1]))]).to_pandas()
    if "date" in ml.columns:
        ml = ml.set_index(["date", "ticker"])
    ml = ml[[k for k in F.columns if k not in ("earn_tonight", "news_today", "kind")]]
    ml = ml.astype("float32")
    Ci = F[F.kind == "C"]
    ex = Ci[["earn_tonight", "news_today"]]
    ml = ml.join(ex, how="left")
    d0 = ml.index.get_level_values(0)
    miss = ml.earn_tonight.isna()
    if miss.any():           # rows outside the C universe: earn_tonight from the panel grid, news from n_news
        t = list(sorted(set(ml.index.get_level_values(1))))
        E = ctx.earn_tonight_ext(t)
        r = E.index.get_indexer(d0[miss])
        c = E.columns.get_indexer(ml.index.get_level_values(1)[miss])
        ml.loc[miss, "earn_tonight"] = E.values[r, c].astype("float32")
        nv = nn.reindex(columns=t)
        r2 = nv.index.get_indexer(d0[miss])
        ml.loc[miss, "news_today"] = np.nan_to_num(nv.values[r2, c]).astype("float32")
    ml["kind"] = "ML"
    F = pd.concat([F, ml])
    F.index = F.index.set_names(["date", "ticker"])
    F.to_parquet(FRAMES)
    print("frames", F.groupby("kind").size().to_dict(), flush=True)


# ------------------------------------------------------------------ scoring
def score_set(X, q):
    """Ensemble members, ensemble, P(jump), P(drop) of quarter q's models on a multi-day frame."""
    import lightgbm as lgb
    import ensemble
    mdl = lgb.Booster(model_file=os.path.join(MD, f"night_{q}.txt"))
    ens = ensemble.predict(X, pd.Period(q, freq="Q"), pooled_model=mdl)
    out = pd.DataFrame({"ensemble": ens["ensemble"].values}, index=X.index)
    for k in ["jump", "drop"]:
        m = lgb.Booster(model_file=os.path.join(MD, f"jump_{k}_{q}.txt"))
        out[f"p_{k}"] = m.predict(X[m.feature_name()])
    out["jmd"] = out.p_jump - out.p_drop
    d = out.index.get_level_values(0)
    out["blend"] = (2 * out.ensemble.groupby(d).rank(pct=True) + out.jmd.groupby(d).rank(pct=True)) / 3
    return out


def score():
    from paper_overnight import cap_by_industry, tradable
    P = load_panel()
    days = P["c"].index
    F = pd.read_parquet(FRAMES)
    rows, picks = [], []
    for kind in ["B", "C", "ML"]:
        X = F[F.kind == kind].drop(columns="kind")
        d = X.index.get_level_values(0)
        for q in QS:
            m = d >= train_cut(q, days)
            if kind == "B" and q != "2026Q4" and q != "2026Q3":
                continue
            if not m.any():
                continue
            s = score_set(X[m], q)
            s["kind"], s["q"] = kind, q
            rows.append(s)
            ok_tr = set(tradable(sorted(set(s.index.get_level_values(1)))))
            for day, g in s.groupby(level=0):
                g = g.droplevel(0)
                pred = g.blend.sort_values(ascending=False, kind="stable")
                for filt, pr in [("all", pred), ("tradable", pred[pred.index.isin(ok_tr)])]:
                    if filt == "tradable" and str(day.date()) < OCT[0]:
                        continue
                    top = list(cap_by_industry(pr, 10, 3).index)
                    picks += [dict(date=day, kind=kind, q=q, filt=filt, ticker=t, rank=i + 1) for i, t in enumerate(top)]
                for lab, col in [("ens_top", "ensemble"), ("jmd_top", "jmd")]:
                    picks += [dict(date=day, kind=kind, q=q, filt=lab, ticker=t, rank=i + 1)
                              for i, t in enumerate(g[col].nlargest(10).index)]
            print(kind, q, "days", s.index.get_level_values(0).nunique(), "rows", len(s), flush=True)
    S = pd.concat(rows)
    S["kind"], S["q"] = S.kind.astype("category"), S.q.astype("category")
    S.to_parquet(SCORES)
    pd.DataFrame(picks).to_parquet(PICKS)


# ------------------------------------------------------------------ crosses
def weights(pk, cols, index):
    W = pd.DataFrame(0.0, index=index, columns=cols)
    for d, g in pk.groupby("date"):
        t = [x for x in g.ticker if x in W.columns]
        if t and d in W.index:
            W.loc[d, t] = 1.0 / len(t)
    return W


def all_weights(P, cols):
    days = P["c"].index
    pk = pd.read_parquet(PICKS)
    pk = pk[pk.filt.isin(["all", "tradable"])]
    idx = days[(days >= Q2[0]) & (days <= OCT[1])]
    Ws = {}
    for (kind, q, filt), g in pk.groupby(["kind", "q", "filt"]):
        Ws[(kind, q, filt)] = weights(g, cols, idx)
    lv = []
    for day, r in live_entries().items():
        if OCT[0] <= day <= OCT[1]:
            lv += [dict(date=pd.Timestamp(day), ticker=t) for t in r["ml_top"]]
    Ws[("live", "live", "live")] = weights(pd.DataFrame(lv), cols, idx)
    return Ws, idx


def fetch():
    import study94_cross_rescore as S94
    import study166_167_blend_parts as S166
    from concurrent.futures import ThreadPoolExecutor
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    Ws, idx = all_weights(P, cols)
    del P
    need = pd.concat([S166.legs(W, days) for W in Ws.values()], ignore_index=True)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC179):
        have = pd.read_parquet(AUC179)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC179
    rows, t0 = [], time.time()

    def one(job):
        try:
            return S94._day_kind(job[0], job[1], job[2], ven)
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(2) as ex:
            for r in ex.map(one, jobs[i:i + 24]):
                rows += r
        S94._save(pd.concat([have, pd.DataFrame(rows)], ignore_index=True))
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(rows),
              f"found {np.mean([not np.isnan(r['px']) for r in rows]) if rows else 0:.3f}", f"{time.time() - t0:.0f}s",
              flush=True)
    if os.path.exists(AUC179) and len(pd.read_parquet(AUC179)):
        S94.refetch_missing()


# ------------------------------------------------------------------ report
def sectors():
    import study150_154_live_check as S154
    return S154.sectors()


def paired_t(a, b):
    d = (a - b).dropna()
    if len(d) < 5 or d.std() == 0:
        return np.nan
    return d.mean() / d.std() * np.sqrt(len(d))


def stats(x):
    x = x.dropna()
    return dict(nights=len(x), net_bp=1e4 * x.mean(), sharpe=ann_stats(x)["sharpe"] if len(x) >= 5 else np.nan,
                hit=(x > 0).mean() if len(x) else np.nan, compounded=float((1 + x).prod() - 1))


def report():
    import bt
    import study166_167_blend_parts as S166
    from scipy.stats import spearmanr
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    _, ind, hw = sectors()
    hwc = hw.reindex(cols).fillna(False).astype(bool)
    Ws, idx = all_weights(P, cols)
    Rx, miss, C = S166.returns(P, cols, days, idx)
    nets, gross, fb = {}, {}, {}
    for k, W in Ws.items():
        r = bt.run(W, Rx, C)
        has = W.sum(axis=1) > 0
        nets[k] = r.net.where(has).dropna()
        gross[k] = r.gross.where(has).dropna()
        h = W > 0
        fb[k] = float((miss & h).sum().sum() / max(1, h.sum().sum()))
    # liquid-universe EW overnight at panel prices (market context)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).clip(-0.5, 0.5).reindex(idx)
    liq = ((P["rawc"][cols] > 5) & (P["dv"][cols].shift(1).rolling(20, min_periods=10).median() > 5e6)).reindex(idx)
    uni = Rp.where(liq).mean(axis=1)
    uni_hw = Rp.where(liq & hwc.values).mean(axis=1)
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    live = live_entries()
    pk = pd.read_parquet(PICKS)
    S = pd.read_parquet(SCORES)

    # ---------------- 179
    rows = []
    oct_idx = idx[(idx >= OCT[0]) & (idx <= OCT[1])]
    variants = [("live_repriced", ("live", "live", "live")), ("B_2026Q4", ("B", "2026Q4", "tradable")),
                ("C_2026Q4", ("C", "2026Q4", "tradable")), ("B_2026Q3", ("B", "2026Q3", "tradable")),
                ("C_2026Q3", ("C", "2026Q3", "tradable"))]
    Hl = Ws[("live", "live", "live")] > 0
    for d in oct_idx:
        ds = str(d.date())
        lr = live.get(ds)
        base = dict(table="nightly", day=ds, live_strategy=lr["strategy"] if lr else "",
                    live_model=lr["model"] if lr else "", universe_overnight_bp=1e4 * uni.get(d, np.nan),
                    techhw_universe_overnight_bp=1e4 * uni_hw.get(d, np.nan))
        rows.append(dict(base, variant="live_virtual_book", net_bp=1e4 * vd.net.get(d, np.nan),
                         gross_bp=1e4 * vd.gross.get(d, np.nan),
                         techhw_picks=int(hwc.reindex(list(lr["ml_top"])).sum()) if lr else np.nan,
                         picks=" ".join(lr["ml_top"]) if lr else ""))
        for name, k in variants:
            W = Ws[k]
            H = W.loc[d] > 0
            t = list(H.index[H])
            if k[0] != "live":
                g = pk[(pk.date == d) & (pk.kind == k[0]) & (pk.q == k[1]) & (pk.filt == k[2])].sort_values("rank")
                t = list(g.ticker)
            rows.append(dict(base, variant=name, net_bp=1e4 * nets[k].get(d, np.nan),
                             gross_bp=1e4 * gross[k].get(d, np.nan), techhw_picks=int(hwc.reindex(t).fillna(False).sum()),
                             overlap_live=int((H & Hl.loc[d]).sum()) if lr else np.nan,
                             overlap_B_2026Q4=int((H & (Ws[("B", "2026Q4", "tradable")].loc[d] > 0)).sum()),
                             picks=" ".join(t)))
    nt = pd.DataFrame(rows)
    live_days_oct = [d for d in oct_idx if str(d.date()) in live]
    for name in ["live_virtual_book"] + [v for v, _ in variants]:
        x = nt[nt.variant == name].set_index("day").net_bp / 1e4
        x.index = pd.to_datetime(x.index)
        lvx = vd.net.reindex(x.index)
        for lab, sel in [("11 live nights", x.index.isin(live_days_oct)), ("13 nights 09-28..10-14", np.ones(len(x), bool))]:
            xx = x[sel].dropna()
            st = stats(xx)
            g = nt[(nt.variant == name) & pd.to_datetime(nt.day).isin(xx.index)]
            rows_s = dict(table="summary", variant=name, item=lab, **st,
                          gross_bp=g.gross_bp.mean(), techhw_picks=g.techhw_picks.mean(),
                          overlap_live=g.overlap_live.mean() if "overlap_live" in g else np.nan,
                          corr_with_live=float(xx.corr(lvx.reindex(xx.index))) if name != "live_virtual_book" else 1.0,
                          universe_overnight_bp=g.universe_overnight_bp.mean(),
                          techhw_universe_overnight_bp=g.techhw_universe_overnight_bp.mean())
            if name != "live_virtual_book":
                k = dict(variants)[name]
                rows_s["fallback_share"] = fb[k]
                rows_s["t_vs_live"] = paired_t(xx, lvx.reindex(xx.index))
            rows.append(rows_s)
    # ranks of the live picks in the B / C 2026Q4 scorings
    for ds, r in live.items():
        d = pd.Timestamp(ds)
        if d not in oct_idx:
            continue
        for kind in ["B", "C"]:
            s = S[(S.kind == kind) & (S.q == "2026Q4")]
            if d not in s.index.get_level_values(0):
                continue
            s = s.xs(d, level=0)
            rk = s.blend.rank(ascending=False)
            for t in r["ml_top"]:
                rows.append(dict(table="live_pick_rank", day=ds, variant=f"{kind}_2026Q4", ticker=t,
                                 industry=ind.get(t), techhw=bool(hwc.get(t, False)), rank=rk.get(t, np.nan),
                                 universe=len(s), in_scored_universe=t in s.index))
    df179 = pd.DataFrame(rows)
    df179.to_csv(OUT179, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(nt.pivot_table(index="day", columns="variant", values="net_bp").round(0).to_string())
    print(nt.pivot_table(index="day", columns="variant", values="overlap_live").to_string())
    sm = df179[df179.table == "summary"]
    print(sm[["variant", "item", "nights", "net_bp", "gross_bp", "compounded", "hit", "techhw_picks", "overlap_live",
              "corr_with_live", "t_vs_live", "fallback_share", "universe_overnight_bp",
              "techhw_universe_overnight_bp"]].round(3).to_string(index=False))
    lr_ = df179[df179.table == "live_pick_rank"]
    print(lr_.groupby("variant")["rank"].describe().round(1).to_string())

    # ---------------- 180
    import study135_137_expectations as S135
    ref = S135.build()["nets"]["capped"]
    rows = []
    for per, a, b in PERIODS:
        oq = own_q(per)
        ref_p = ref.loc[a:b]
        for kind in (["C", "ML"] if per != "2026-10" else ["C"]):
            kk = {q: (kind, q, "all") for q in QS if (kind, q, "all") in Ws and q <= oq}
            base = nets[kk[oq]].loc[a:b] if oq in kk else None
            Hb = Ws[kk[oq]].loc[a:b] > 0
            for q, k in kk.items():
                x = nets[k].loc[a:b]
                if not len(x):
                    continue
                H = Ws[k].loc[a:b] > 0
                dd = H.index[H.sum(axis=1) > 0]
                st = stats(x)
                sub = pk[(pk.kind == kind) & (pk.q == q) & (pk.date >= a) & (pk.date <= b)]
                sh_ = {f: float(hwc.reindex(sub[sub.filt == f].ticker).fillna(False).mean()) for f in ["all", "ens_top", "jmd_top"]}
                sc = S[(S.kind == kind) & (S.q == q)]
                so = S[(S.kind == kind) & (S.q == oq)]
                rho = []
                for d in dd[::3]:
                    if d in sc.index.get_level_values(0) and d in so.index.get_level_values(0):
                        u, v = sc.xs(d, level=0).blend, so.xs(d, level=0).blend
                        com = u.index.intersection(v.index)
                        rho.append(spearmanr(u[com], v[com])[0])
                rows.append(dict(table="period_model", period=per, frame=kind, model=q,
                                 role="own quarter" if q == oq else ("older" if q < oq else "newer"), **st,
                                 gross_bp=1e4 * gross[k].loc[a:b].mean(),
                                 t_vs_own=paired_t(x, base) if base is not None and q != oq else np.nan,
                                 diff_vs_own_bp=1e4 * (x - base).mean() if base is not None else np.nan,
                                 overlap_own=float((H & Hb).loc[dd].sum(axis=1).mean()),
                                 spearman_own=float(np.mean(rho)) if rho else np.nan,
                                 techhw_share=sh_["all"], techhw_share_ens_top=sh_["ens_top"],
                                 techhw_share_jmd_top=sh_["jmd_top"], fallback_share=fb[k],
                                 universe_overnight_bp=1e4 * uni.loc[x.index].mean(),
                                 ref_1545_ownq_bp=1e4 * ref_p.reindex(x.index).mean(),
                                 corr_ref_1545=float(x.corr(ref_p.reindex(x.index)))))
                for mth, xm in x.groupby(x.index.to_period("M")):
                    rows.append(dict(table="monthly", period=str(mth), frame=kind, model=q, nights=len(xm),
                                     net_bp=1e4 * xm.mean(), ref_1545_ownq_bp=1e4 * ref.reindex(xm.index).mean(),
                                     techhw_share=float(hwc.reindex(sub[(sub.filt == "all") & (sub.date.dt.to_period("M") == mth)].ticker).fillna(False).mean()),
                                     universe_overnight_bp=1e4 * uni.loc[xm.index].mean()))
    # days_to_e coverage per frame and month
    F = pd.read_parquet(FRAMES, columns=["days_to_e", "kind"])
    m_ = F.index.get_level_values(0).to_period("M")
    for (kind, mth), g in F.groupby([F.kind, m_]):
        rows.append(dict(table="days_to_e", period=str(mth), frame=kind, nan_share=float(g.days_to_e.isna().mean()), n=len(g)))
    del F
    # gain importance
    import lightgbm as lgb
    imp = {}
    for q in QS:
        for fam in ["night", "ens_pooled_beh", "ens_pooled_com", "jump_jump", "jump_drop"]:
            mdl = lgb.Booster(model_file=os.path.join(MD, f"{fam}_{q}.txt"))
            g = pd.Series(mdl.feature_importance("gain"), index=mdl.feature_name())
            g = 100 * g / g.sum()
            imp[(fam, q)] = g
            for f, v in g.items():
                rows.append(dict(table="importance", model=q, family=fam, feature=f, gain_pct=v))
    for fam in ["night", "ens_pooled_beh", "ens_pooled_com", "jump_jump", "jump_drop"]:
        a_, b_ = imp[(fam, "2026Q2")], imp[(fam, "2026Q3")]
        dlt = (b_ - a_.reindex(b_.index)).sort_values()
        for f in list(dlt.index[:6]) + list(dlt.index[-6:]):
            rows.append(dict(table="importance_shift", family=fam, feature=f, gain_2025Q3=imp[(fam, "2025Q3")].get(f),
                             gain_2026Q1=imp[(fam, "2026Q1")].get(f), gain_2026Q2=a_.get(f), gain_2026Q3=b_.get(f),
                             gain_2026Q4=imp[(fam, "2026Q4")].get(f), shift_q2_to_q3=dlt[f]))
        # similarity of the importance vectors
        for q in QS:
            rows.append(dict(table="importance_shift", family=fam, feature="(corr with 2026Q3 vector)", model=q,
                             gain_pct=float(imp[(fam, q)].corr(b_.reindex(imp[(fam, q)].index)))))
    df180 = pd.DataFrame(rows)
    df180.to_csv(OUT180, index=False)
    pm = df180[df180.table == "period_model"]
    print(pm[["period", "frame", "model", "role", "nights", "net_bp", "sharpe", "hit", "compounded", "diff_vs_own_bp",
              "t_vs_own", "overlap_own", "spearman_own", "techhw_share", "techhw_share_ens_top", "techhw_share_jmd_top",
              "ref_1545_ownq_bp", "corr_ref_1545", "fallback_share"]].round(3).to_string(index=False))
    mo = df180[df180.table == "monthly"]
    print(mo.pivot_table(index="period", columns=["frame", "model"], values="net_bp").round(0).to_string())
    print(mo[mo.frame == "C"].pivot_table(index="period", columns="model", values="techhw_share").round(3).to_string())
    print(df180[df180.table == "days_to_e"].pivot_table(index="period", columns="frame", values="nan_share").round(3).to_string())
    sh = df180[(df180.table == "importance_shift")]
    print(sh[sh.feature != "(corr with 2026Q3 vector)"][["family", "feature", "gain_2025Q3", "gain_2026Q1", "gain_2026Q2",
                                                         "gain_2026Q3", "gain_2026Q4", "shift_q2_to_q3"]].round(2).to_string(index=False))
    print(sh[sh.feature == "(corr with 2026Q3 vector)"].pivot_table(index="family", columns="model", values="gain_pct").round(3).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"fetchbars": fetchbars, "frames": frames, "score": score, "fetch": fetch, "report": report}[what]()
    import resource
    print("peak memory GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
