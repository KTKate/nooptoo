"""Study 113: what does real-time data (Alpaca Algo Trader Plus, $99 a month) add to the overnight blend?

Question: on the free plan SIP data must end more than 15 minutes in the past, so the 15:46 entry job sees SIP bars
only up to 15:30. With the paid real-time plan the job could score at about 15:45 with SIP prices to 15:44 and still
meet the 15:50 market-on-close cutoff. How much of the blend's return depends on the last 15 minutes of data, and is
the gain worth $1,188 a year for a $10k or $25k account?

Data: daily panel (core.load_panel), m5snap 5-minute SIP bars (data/local/m5snap: bars start at 15:30, so no bar ends
at 15:30; the 15:30 price is the open of the 15:30-15:35 bar, the first trade at or after 15:30:00), SPY/IWM 1-minute
bars (data/local/m1), earnings calendar and Alpaca/Benzinga news (data/local/news_scored), the saved walk-forward
models (data/models/night_<q>, ens_*_<q>*, jump_{jump,drop}_<q>; read only), official opening and closing crosses
cached by studies 88 / 94 / 99 (data/local/auctions88/94/99.parquet) plus crosses fetched here for new picks
(data/local/auctions113.parquet, study 94 rule: listing-venue 'O' / '6' print, second pass the largest print of any
listing exchange; resumable).

Design: the study-23 15:45 feature frame (same code: ml_features.build / features_frame, SNAP_MODE none, m5snap
universe) is rebuilt day by day for 2024-01 .. 2026-09 in three versions of the last row:
  p1545    study 23 exactly (the paid plan): price = close of the 15:40-15:45 bar, day volume x 0.85, SPY/IWM close
           of the 15:44 minute, news counted to 15:45.
  p1530    everything at 15:30 (the free plan if it used SIP only): price = open of the 15:30-15:35 bar, volume =
           day volume x 0.85 minus the 15:30-15:45 bar volume (clipped to 40-85% of the day), SPY/IWM close of the
           15:29 minute, news counted to 15:30 (window from the previous day's 15:30).
  free_live  what the free-plan job does today (paper_overnight.today_intraday): SIP volume to 15:30, but the price
           is the real-time IEX last trade at about 15:45 and news is real time. IEX trades are not available
           historically, so the 15:45 SIP price stands in for the IEX last trade (optimistic for thin names whose IEX
           trade is stale and falls back to the 15:30 SIP close).
High and low: all versions use the day's high/low widened to include the price (study 23 convention; m5snap has no
bars before 15:30 after 10:00, so the 15:30 version still sees the 15:30-16:00 range in the high/low, a small
look-ahead that favours p1530). Only the volume / price / ETF / news inputs differ. The ensemble (five members,
ensemble.predict) and the jump / drop classifiers (study 33 inputs: earn_tonight, news_today) are scored with the
models of each quarter. Feature frames are built without LightGBM in two worker processes, predictions in a separate
run (no fork after LightGBM). Columns are restricted per day to a superset of the universe (price > $5 and unmodified
20-day median dollar volume > $5M); the result is identical because every feature is per column and the
cross-sectional ranks use the universe only (checked: p1545 reproduces results/study23_pred.parquet and
study33_pred.parquet).
Blend = (2 x rank(ensemble) + rank(P(jump) - P(drop))) / 3, top 10 (uncapped, as study 94's base; and the live
version with at most 3 per industry), buy at the closing cross, sell at the next opening cross; returns at crosses
= panel return x cross / panel price ratio per leg (study 94; legs without a usable cross use the panel price) and at
panel prices. Costs exec_cost_bps(P, "auction") + 2.5 bp per side. Per period (2024-01..2025-06, 2025-07..2026-09,
2024-01..2026-09): Sharpe, mean net bp, max drawdown, daily pick overlap with p1545, paired t of daily net returns
(p1545 minus the other version). Dollars a year = mean daily net difference x 252 x account size (no compounding),
against the $1,188 a year plan cost.

Data checks: the rebuilt p1545 predictions cover exactly the rows of study23_pred / study33_pred; raw predictions
match except ~1-2% of rows (later revisions of the Yahoo panel and earnings calendar, e.g. CTVA, HON, and the day
2026-09-25), and the top-10 picks agree with study 94's on 97.6% of names; p1545 at crosses 2.79 / 2.19 / 2.50 vs
study 94's 2.48 / 2.27 / 2.38. The news count rebuilt with a 15:45 cutoff equals news_features' n_news on every day
in the test window. m5snap volume of the 15:30-15:45 bars is ~4% of the day (median volume share to 15:30 = 0.81).
2,426 new crosses fetched (99.6% found); 1.1% of legs priced at the panel in every version.

Result (top 10 uncapped, Sharpe at crosses, 2024-01..2025-06 / 2025-07..2026-09 / 2024-26): p1545 2.79 / 2.19 /
2.50 (39.2 bp net a night), p1530 2.58 / 2.19 / 2.39 (36.4 bp), free_live 2.94 / 2.36 / 2.66 (41.4 bp). Paired,
2024-26 at crosses: p1545 minus p1530 +2.8 bp a night (t 0.66, 95% CI -5.5..+11.1), picks shared 74%; p1545 minus
free_live -2.2 bp (t -1.23), picks shared 95%. Capped live version: p1545 minus p1530 +5.5 bp (t 1.35), p1545 minus
free_live -2.3 bp (t -1.56). In dollars (2024-26, crosses, uncapped): +$700 a year on $10k, +$1,751 on $25k against
p1530 (break-even ~$17k; capped: ~$8.6k), but negative against free_live. Verdict: the 15 minutes of extra SIP data
matter only for the price; the live free-plan job already takes the price from the real-time IEX last trade, and
volume / range to 15:44 add nothing (slightly negative, within noise). Not worth $1,188 a year, provided the IEX
price is close to the SIP price for the picks (not testable here: no IEX history).

Output: results/study113_signal_time.csv (tables: rescore, paired, overlap, dollars, fallback, checks).
Frames cached in data/local/study113_frames/, predictions in data/local/study113_pred.parquet.

    python src/study113_signal_time.py frames    # feature frames (no LightGBM, 2 processes, resumable by month)
    python src/study113_signal_time.py predict   # ensemble + jump predictions (LightGBM, one process)
    python src/study113_signal_time.py fetch     # official crosses for new picks (resumable)
    python src/study113_signal_time.py           # evaluation
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATA, RES   # noqa: E402

LOC = os.path.join(DATA, "local")
FR = os.path.join(LOC, "study113_frames")
PRED_FN = os.path.join(LOC, "study113_pred.parquet")
AUC113 = os.path.join(LOC, "auctions113.parquet")
OUT = os.path.join(RES, "study113_signal_time.csv")
VARS = ["p1545", "p1530", "free_live"]
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
PLAN = 99.0 * 12


# ------------------------------------------------------------------ light feature code
def ml_light():
    """ml_features without its module-level full feature build (which needs ~5 GB): the same source, with the block
    that builds F / univ / T for the whole panel removed. Returns the namespace (P, cols, days, build,
    features_frame, earnings_features)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_features.py")).read()
    a = src.index("\no, h, l, c, rawc, v, dv = (P[k][cols]")
    b = src.index("\ndef features_frame")
    ns = {"__name__": "ml_light"}
    exec(compile(src[:a] + "\n" + src[b:], "ml_features.py(light)", "exec"), ns)
    return ns


def news_counts(days, cols, hm):
    """n_news panel as news_features.build, with the daily cutoff at hm ET instead of 15:45."""
    import news_features as NF
    d = NF.load_news()
    d = d[d.sym.isin(cols)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    h, m = map(int, hm.split(":"))
    cut = pd.DatetimeIndex(days) + pd.Timedelta(hours=h, minutes=m)
    i = cut.searchsorted(ts.values, side="left")
    d = d.assign(di=i)[i < len(days)]
    d["day"] = days[d.di.values]
    n = d.groupby(["day", "sym"]).size().unstack().reindex(index=days, columns=cols).astype("float32").fillna(0.0)
    n.loc[:d.day.min() - pd.Timedelta(days=1)] = np.nan
    return n


def snap_bits(days, cols):
    """From m5snap: 15:45 price (c of the 15:40 bar), 15:30 price (o of the 15:30 bar), volume of the 15:30, 15:35
    and 15:40 bars; date x ticker on (snap dates, cols)."""
    import pyarrow.parquet as pq
    d = os.path.join(LOC, "m5snap")
    parts = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".parquet"):
            x = pq.read_table(os.path.join(d, f), columns=["ts", "ticker", "o", "c", "v"]).to_pandas()
            hm = x.ts.dt.strftime("%H:%M")
            parts.append(x[hm.isin(["15:30", "15:35", "15:40"])].assign(hm=hm))
    x = pd.concat(parts, ignore_index=True)
    x["date"] = x.ts.dt.normalize()
    sd = pd.DatetimeIndex(sorted(x.date.unique()))
    pv = lambda hm, k: x[x.hm == hm].pivot(index="date", columns="ticker", values=k).reindex(index=sd, columns=cols)
    out = dict(c1540=pv("15:40", "c"), o1530=pv("15:30", "o"),
               v3045=pv("15:30", "v").fillna(0) + pv("15:35", "v").fillna(0) + pv("15:40", "v").fillna(0))
    return out


# ------------------------------------------------------------------ frames
_G = {}


def _init():
    import alpaca_data as A
    M = ml_light()
    P, cols, days = M["P"], M["cols"], M["days"]
    from study8_exec_retest import snap_panels
    snap = snap_panels("none")
    sb = snap_bits(days, cols)
    p1545 = snap["c15:40"].reindex(columns=cols)
    m1 = A.read("m1", start="2023-12", tickers=["SPY", "IWM"])
    m1["hm"] = m1.ts.dt.strftime("%H:%M")
    m1["date"] = m1.ts.dt.normalize()
    etf = {hm: m1[m1.hm == hm].pivot(index="date", columns="ticker", values="c") for hm in ["15:44", "15:29"]}
    test_days = [d for d in days if d >= pd.Timestamp("2024-01-02") and d in p1545.index and d in etf["15:44"].index]
    live_cols = [t for t in cols if P["dv"][t].loc["2023-06":].max() > 3e6]
    _G.update(M=M, P=P, cols=cols, days=days, p1545=p1545, sb=sb, etf=etf, test_days=test_days, live_cols=live_cols,
              earn=M["earnings_features"](),
              adjf=P["c"][cols] / P["rawc"][cols])


def frames_day(d):
    """Feature frames of day d for the three versions (dict name -> long frame), study23.frame_1545 logic."""
    G = _G
    M, P, days, lc = G["M"], G["P"], G["days"], G["live_cols"]
    i = days.get_loc(d)
    sl = slice(max(0, i - 260), i + 1)
    o30 = G["sb"]["o1530"]
    px = {"p1545": G["p1545"].loc[d].reindex(lc),
          "p1530": (o30.loc[d] if d in o30.index else pd.Series(np.nan, index=o30.columns)).reindex(lc)}
    px["free_live"] = px["p1545"]
    adv_un = P["dv"][lc].iloc[sl].rolling(20, min_periods=10).median().iloc[-1]
    vday = P["v"][lc].iloc[i]
    v3045 = G["sb"]["v3045"].reindex(index=[d]).iloc[0].reindex(lc).fillna(0)
    out, info = {}, {}
    for name in VARS:
        p = px[name]
        cand = [t for t, a, b in zip(lc, p.values, adv_un.values) if a > 5 and b > 5e6]   # superset of the universe
        oo, hh, ll, cc, rr, vv, dd = (P[k][cand].iloc[sl].copy() for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
        pp = p.reindex(cand)
        ok = pp.notna()
        cc.iloc[-1] = np.where(ok, pp * G["adjf"].loc[d, cand], np.nan)
        rr.iloc[-1] = np.where(ok, pp, np.nan)
        hh.iloc[-1] = np.maximum(hh.iloc[-1], cc.iloc[-1])
        ll.iloc[-1] = np.minimum(ll.iloc[-1], cc.iloc[-1])
        if name == "p1545":
            fv = pd.Series(0.85, index=cand)
        else:
            vd = vday.reindex(cand)
            fv = ((0.85 * vd - v3045.reindex(cand)) / vd).clip(0.40, 0.85).fillna(0.85)
            info[name] = float(fv.median())
        vv.iloc[-1] = vv.iloc[-1] * fv.values
        dd.iloc[-1] = dd.iloc[-1] * fv.values
        ehm = "15:29" if name == "p1530" else "15:44"
        e = {}
        for t in ["SPY", "IWM"]:
            s = P["c"][t].iloc[sl].copy()
            ex = G["etf"][ehm]
            ex = ex if d in ex.index else G["etf"]["15:44"]           # early-close days: no 15:29 minute bar
            s.iloc[-1] = ex.at[d, t] * (P["c"][t].loc[d] / P["rawc"][t].loc[d])
            e[t] = s
        F, mkt, adv20 = M["build"](oo, hh, ll, cc, rr, vv, dd, e["SPY"], P["c"]["^VIX"].iloc[sl],
                                   P["c"]["^VIX3M"].iloc[sl], e["IWM"], earn=G["earn"])
        last = {k: f.iloc[[-1]] for k, f in F.items()}
        univ = ((rr > 5) & (adv20 > 5e6)).iloc[[-1]] & ok.values
        out[name] = M["features_frame"](last, mkt.iloc[[-1]], univ)
    return d, out, info


def frames():
    from multiprocessing import Pool
    _init()
    os.makedirs(FR, exist_ok=True)
    months = sorted({d.strftime("%Y-%m") for d in _G["test_days"]})
    todo = [m for m in months if not all(os.path.exists(os.path.join(FR, f"{v}_{m}.parquet")) for v in VARS)]
    print("months to build", len(todo), "of", len(months), flush=True)
    t0 = time.time()
    with Pool(2) as pool:
        for m in todo:
            dd = [d for d in _G["test_days"] if d.strftime("%Y-%m") == m]
            res = pool.map(frames_day, dd, chunksize=2)
            for v in VARS:
                pd.concat([r[1][v] for r in res]).to_parquet(os.path.join(FR, f"{v}_{m}.parquet"))
            fv = np.nanmedian([r[2].get("p1530", np.nan) for r in res])
            print(m, len(dd), "days", f"median volume share to 15:30 {fv:.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ predictions
def predict():
    import lightgbm as lgb
    import ensemble
    import store
    from core import load_panel, stock_cols
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    # study-33 extra inputs: earn_tonight (as study33_jump_live.extra_panels), news_today per version's cutoff
    E = store.read("earnings")
    di = days.searchsorted(pd.to_datetime(E.date))
    earn = pd.DataFrame(0.0, index=days, columns=cols)
    ci = {t: k for k, t in enumerate(cols)}
    for dq, t in zip(di, E.symbol):
        if t in ci and 0 < dq < len(days):
            earn.iat[dq, ci[t]] = 1.0
            earn.iat[dq - 1, ci[t]] = 1.0
    import news_features as NF
    n45 = NF.load()["n_news"].reindex(index=days, columns=cols)
    n30 = news_counts(days, cols, "15:30")
    chk = news_counts(days, cols, "15:45")
    a, b = chk.loc["2024":"2026-09"], n45.loc["2024":"2026-09"]
    print("news 15:45 rebuild matches news_features:", float(((a == b) | (a.isna() & b.isna())).values.mean()), flush=True)
    del chk, a, b
    news = {"p1545": n45, "free_live": n45, "p1530": n30}
    have = pd.read_parquet(PRED_FN) if os.path.exists(PRED_FN) else None
    rows = [] if have is None else [have]
    done = set() if have is None else set(zip(have.variant, have.q))
    files = sorted(os.listdir(FR))
    for v in VARS:
        X = pd.concat([pd.read_parquet(os.path.join(FR, f)) for f in files if f.startswith(v + "_")])
        dates = X.index.get_level_values(0)
        qs = dates.to_period("Q")
        for q in sorted(set(qs)):
            if (v, str(q)) in done:
                continue
            t0 = time.time()
            Xq = X[qs == q]
            ens = ensemble.predict(Xq, q)
            Xj = Xq.copy()
            r = days.get_indexer(Xj.index.get_level_values(0))
            c = pd.Index(cols).get_indexer(Xj.index.get_level_values(1))
            for k, panel in [("earn_tonight", earn), ("news_today", news[v])]:
                Xj[k] = panel.values[r, c].astype("float32")
            mj = lgb.Booster(model_file=os.path.join(DATA, "models", f"jump_jump_{q}.txt"))
            md = lgb.Booster(model_file=os.path.join(DATA, "models", f"jump_drop_{q}.txt"))
            Xj = Xj[mj.feature_name()]
            out = ens.assign(p_jump=mj.predict(Xj), p_drop=md.predict(Xj)).astype("float32")
            out = out.reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
            out["variant"], out["q"] = v, str(q)
            rows.append(out)
            pd.concat(rows, ignore_index=True).to_parquet(PRED_FN, index=False)
            print(v, q, len(Xq), f"{time.time() - t0:.0f}s", flush=True)
        del X


# ------------------------------------------------------------------ blends, crosses
def blends(cols, days):
    """Blend score per version (date x ticker), plus the ensemble alone."""
    pr = pd.read_parquet(PRED_FN)
    S, E = {}, {}
    for v, g in pr.groupby("variant"):
        g = g.set_index(["date", "ticker"])
        ens = g["ensemble"].unstack().reindex(columns=cols)
        ens = ens.loc[ens.index < days[-1]]
        pj = g.p_jump.unstack().reindex(index=ens.index, columns=cols)
        pdr = g.p_drop.unstack().reindex(index=ens.index, columns=cols)
        ok = ens.notna() & pj.notna()
        S[v] = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
        E[v] = ens.where(ok)
    return S, E


def capped(Sx, cols):
    """Top 10 with at most 3 names per industry (paper_overnight.cap_by_industry rule, as study 94)."""
    ind = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index(
        "ticker").industry
    Wc = pd.DataFrame(0.0, index=Sx.index, columns=cols)
    for d, row in Sx.iterrows():
        r = row.dropna().sort_values(ascending=False, kind="stable")
        out, cnt = [], {}
        for t in r.index:
            g = ind.get(t)
            g = g if isinstance(g, str) and g else t
            if cnt.get(g, 0) >= 3:
                continue
            cnt[g] = cnt.get(g, 0) + 1
            out.append(t)
            if len(out) == 10:
                break
        if out:
            Wc.loc[d, out] = 1.0 / len(out)
    return Wc


def all_crosses():
    import study99_cross_retrain as S99
    x = S99.crosses()
    if os.path.exists(AUC113):
        c = pd.read_parquet(AUC113)[["ticker", "date", "kind", "px"]]
        c["date"] = pd.to_datetime(c.date)
        c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
        x = pd.concat([x, c.dropna(subset=["px"])], ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    return x


def weights(cols, days):
    S, E = blends(cols, days)
    W = {}
    for v in VARS:
        W[(v, "top10")] = bt_topk(S[v])
        W[(v, "top10_cap3")] = capped(S[v], cols)
    return S, E, W


def bt_topk(Sx):
    import bt
    return bt.select_topk(Sx, Sx.notna(), 10)


def fetch():
    import study94_cross_rescore as S94
    import study99_cross_retrain as S99
    from concurrent.futures import ThreadPoolExecutor
    from core import load_panel, stock_cols
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    nxt = pd.Series(days[1:], index=days[:-1])
    S, E, W = weights(cols, days)
    rows = []
    for Wv in W.values():
        s = Wv.loc["2024-01":"2026-09"].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(rows).dropna().drop_duplicates()
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    have = [all_crosses()[["ticker", "date", "kind"]]]
    for fn in [S99.AUC99, S99.AUC94, AUC113]:          # pairs already tried (found or not)
        if os.path.exists(fn):
            t2 = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t2["ticker"] = t2.ticker.astype(str).str.replace(".", "-", regex=False)
            have.append(t2.assign(date=pd.to_datetime(t2.date)))
    have = pd.concat(have).drop_duplicates()
    m = need.merge(have, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("needed", len(need), "to fetch", len(todo), flush=True)
    if not len(todo):
        return
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
    old = pd.read_parquet(AUC113) if os.path.exists(AUC113) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(4) as exe:
            for r in exe.map(one, jobs[i:i + 24]):
                new += r
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC113, index=False)
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}" if new else "", f"{time.time() - t0:.0f}s",
              flush=True)


# ------------------------------------------------------------------ evaluation
def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def main():
    import bt
    import study99_cross_retrain as S99
    from core import load_panel, stock_cols, exec_cost_bps, ann_stats
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    S, E, W = weights(cols, days)
    rows = []
    # ---- checks: p1545 against studies 23 / 33
    pr = pd.read_parquet(PRED_FN)
    b = pr[pr.variant == "p1545"].set_index(["date", "ticker"])
    s23 = pd.read_parquet(f"{RES}/study23_pred.parquet")
    s33 = pd.read_parquet(f"{RES}/study33_pred.parquet")
    for name, ref, k in [("ensemble vs study23_pred", s23, "ensemble"), ("pooled vs study23_pred", s23, "pooled"),
                         ("p_jump vs study33_pred", s33, "p_jump"), ("p_drop vs study33_pred", s33, "p_drop")]:
        j = b[[k]].join(ref[[k]], how="outer", rsuffix="_ref")
        both = j.dropna()
        rows.append(dict(table="checks", set=name, n=len(j), n_both=len(both),
                         max_abs_diff=float((both[k] - both[k + "_ref"]).abs().max()),
                         share_rows_only_here=float(j[k + "_ref"].isna().mean()),
                         share_rows_only_ref=float(j[k].isna().mean())))
    del pr, b, s23, s33
    for v in VARS:
        n = S[v].notna().sum(axis=1)
        rows.append(dict(table="checks", set=f"universe size per day ({v})", n=int(len(n)), mean=float(n.mean())))
    # ---- returns
    ro, rc, err = S99.cross_ratios(P, cols, days, all_crosses())
    rows.append(dict(table="checks", set="cross data errors (legs at panel price)", n=err["ro"]["found"] + err["rc"]["found"],
                     scale_mismatch=err["ro"]["scale"] + err["rc"]["scale"], ratio_out=err["ro"]["ratio"] + err["rc"]["ratio"]))
    idx = S["p1545"].index
    for v in VARS:
        idx = idx.intersection(S[v].index[S[v].notna().any(axis=1)])
    idx = idx[(idx >= "2024-01-01") & (idx <= "2026-09-30")]
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=idx)
    Rx = (1 + Rp) * ro.shift(-1).fillna(1.0).reindex(index=idx) / rc.fillna(1.0).reindex(index=idx) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=idx)
    cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=idx)
    net = {}
    for (v, sel), Wv in W.items():
        Wv = Wv.reindex(index=idx).fillna(0)
        for pricing, R in [("panel", Rp), ("cross", Rx)]:
            net[(v, sel, pricing)] = bt.run(Wv, R, cost).net
            for lab, a0, b0 in PER:
                x = net[(v, sel, pricing)].loc[a0:b0].dropna()
                st = ann_stats(x)
                rows.append(dict(table="rescore", variant=v, selection=sel, pricing=pricing, period=lab, n=st["n"],
                                 sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"]))
        h = Wv > 0
        for lab, a0, b0 in PER:
            hh = h.loc[a0:b0]
            rows.append(dict(table="fallback", variant=v, selection=sel, period=lab, legs=int(hh.values.sum()),
                             share_panel=float((miss.loc[a0:b0] & hh).values.sum() / max(hh.values.sum(), 1))))
    # ---- overlap and paired tests vs p1545
    for sel in ["top10", "top10_cap3"]:
        hb = W[("p1545", sel)].reindex(index=idx).fillna(0) > 0
        for v in ["p1530", "free_live"]:
            hv = W[(v, sel)].reindex(index=idx).fillna(0) > 0
            ov = (hb & hv).sum(axis=1) / hb.sum(axis=1).replace(0, np.nan)
            for lab, a0, b0 in PER:
                rows.append(dict(table="overlap", variant=v, selection=sel, period=lab,
                                 mean_shared=float(ov.loc[a0:b0].mean()), median_shared=float(ov.loc[a0:b0].median())))
            for pricing in ["panel", "cross"]:
                dlt = net[("p1545", sel, pricing)] - net[(v, sel, pricing)]
                for lab, a0, b0 in PER:
                    x = dlt.loc[a0:b0].dropna()
                    se = x.std() / np.sqrt(len(x))
                    rows.append(dict(table="paired", variant=f"p1545 minus {v}", selection=sel, pricing=pricing,
                                     period=lab, n=len(x), diff_bp=1e4 * x.mean(), t=tstat(x),
                                     ci95_lo_bp=1e4 * (x.mean() - 1.96 * se), ci95_hi_bp=1e4 * (x.mean() + 1.96 * se),
                                     sharpe_diff=ann_stats(net[("p1545", sel, pricing)].loc[a0:b0])["sharpe"]
                                     - ann_stats(net[(v, sel, pricing)].loc[a0:b0])["sharpe"]))
                    if lab == "2024-26":
                        for acct in [10000, 25000]:
                            gain = x.mean() * 252 * acct
                            rows.append(dict(table="dollars", variant=f"p1545 minus {v}", selection=sel,
                                             pricing=pricing, period=lab, account=acct, gain_per_year=gain,
                                             gain_lo95=(x.mean() - 1.96 * se) * 252 * acct,
                                             gain_hi95=(x.mean() + 1.96 * se) * 252 * acct,
                                             plan_cost=PLAN, net_of_plan=gain - PLAN,
                                             breakeven_account=PLAN / (x.mean() * 252) if x.mean() > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"frames": frames, "predict": predict, "fetch": fetch, "eval": main}[cmd]()
