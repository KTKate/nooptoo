"""Studies 150 and 154: what drives the overnight blend's picks now, and does the paper job trade what the backtest would.

154  Live vs backtest consistency. Question: the virtual book lost about 2% on 4 of 5 nights (2026-10-06/07/09/12).
     Is that bad luck, or does the live entry job (src/paper_overnight.py entry --strategy=blend) pick different
     stocks than the backtest model would on the same days?
150  What drives the picks now. Question: tech hardware's share of the capped picks rose from 7% (2024Q1) to 30%
     (study 135). Is that the stocks (their recent volatility and momentum) or a change in the models (drift in what
     the quarterly refits use)? How do the live October picks compare with the 2024-26 backtest picks?

Data
  Live logs (read only): logs/paper/entry_<day>.json (submitted orders, capped and uncapped top 10, model names) and
  members_<day>.json (top 10 of each ensemble member, of the ensemble and of P(jump) - P(drop), news and earnings
  counts) for every submitted entry from 2026-09-29.
  Models (read only): data/models/night_<q>, ens_*_<q>, jump_{jump,drop}_<q> (same files the live job pulls).
  Daily panel core.load_panel (Yahoo store through 2026-10-12), earnings calendar data/store/earnings,
  data/store/sectors.parquet. Alpaca SIP bars for the candidate universe on each live day: 30-minute bars
  09:30-15:30, the 15:40-15:45 5-minute bar, SPY/IWM 15:44 1-minute bars and, for 2026-10-13 (not yet in the daily
  store), the daily bar; cached in data/local/study154_bars.parquet. Alpaca news counts per symbol for
  (15:45 of the previous trading day, 15:45 of the day], articles tagged with 1-3 symbols, cached in
  data/local/study154_news.parquet. Backtest predictions results/study23_pred.parquet and study33_pred.parquet.
  ml_features.py is not imported (its module-level full-panel build peaks above 4 GB); its earnings_features, build
  and features_frame functions are loaded from the source file and run on a 260-day window.

Design 154, per live day, four offline scorings with the same quarterly models the live job used:
  L      live replica: the code path of paper_overnight.ml_scores (30-minute SIP bars to 15:30 for open/high/low/
         volume, the price at 15:45, earnings features from the panel that ends the previous day, VIX of the previous
         day, news NaN when none). The live price is the IEX last trade after 15:40; L uses the SIP 15:40-15:45 bar
         close, as the runner's replay mode does.
  Lfix   L with the earnings features computed on a calendar that extends past the day (as in training).
  B      backtest 15:45 (study 23 / 33 frame): daily-store open/high/low, the SIP 15:40-15:45 close, 85% of the day's
         volume, SPY/IWM at 15:44, VIX close of the day, earnings features on the extended calendar, news 0 when none.
  C      daily close features (cruder): the full daily row of the day.
  Each is ranked as live: (2 rank(ensemble) + rank(P(jump) - P(drop))) / 3 on the tradable names, top 10 with at most 3
  per industry (2026-09-29 ran --strategy=ensemble with the 2026Q3 models; scored that way). Checks: overlap of each
  top 10 with the live picks; Spearman rank correlation across the universe between variants; L against every value
  the live job logged (member top 10s, ensemble top 10, jump-minus-drop top 10, earnings and news counts); per live
  pick: rank in each variant, price at 15:45 vs the order size, ADV, news, earnings flags, panel sanity (last close
  date, zero volume, largest daily move, split list in the job logs).
Design 150: backtest blend picks 2024-01..2026-09 (study 69 lines 22-46, capped at 3 per industry as live) and the
  live October picks; traits per pick: price, 20-day median dollar volume, 20-day volatility, return from the previous
  close to 15:45 (study 23 frame: the 15:40 SIP bar), 5-day return, news count in the window, sector, 60-day beta,
  tech hardware flag (TECH_HW of studies 90/92/135). Each trait is also expressed as the pick's within-day percentile
  among the universe, so a market-wide shift (e.g. all of tech more volatile) is separated from the selection.
  Decomposition of the tech hardware share: share of tech hardware in the universe, and its share among the names
  with the highest volatility/momentum, per half-year. Model side: LightGBM gain importance (normalized to 100%)
  of night_, ens_pooled_beh/com_ and jump_ models of 2024Q1, 2025Q1, 2026Q1 and 2026Q4; and a cross-scoring test:
  the 2024Q1 and 2026Q4 models score the same 2026Q3 feature frames (B frames rebuilt for 20 sampled days), tech
  hardware share of each model's top 10.
Output: results/study154_live_consistency.csv (table = day_summary / variant_corr / member_check / live_pick /
  check), results/study150_pick_traits.csv (table = traits / trait_pct / techhw_decomp / importance / cross_score)
    python src/study150_154_live_check.py fetch   # Alpaca bars and news for the live days (cached)
    python src/study150_154_live_check.py 154
    python src/study150_154_live_check.py 150
"""
import ast
import glob
import json
import os
import sys

os.environ.setdefault("ALPACA_PER_MIN", "60")
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import alpaca_data as A
import store
from core import load_panel, stock_cols, RES, DATA

LOG = os.path.join(A.ROOT, "logs", "paper")
BARS = os.path.join(A.LOCAL, "study154_bars.parquet")
NEWS = os.path.join(A.LOCAL, "study154_news.parquet")
MD = os.path.join(DATA, "models")
TECH_HW = ["Semiconductors", "Electronic Components", "Computer peripheral equipment", "Computer Manufacturing",
           "Computer Communications Equipment", "Retail: Computer Software & Peripheral Equipment"]
TECH_HW_TECHSECTOR = ["Electrical Products", "Industrial Machinery/Components"]
ET = "America/New_York"


def utc(day, hm):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {hm}").tz_localize(ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def load_mlf():
    """earnings_features, build and features_frame from src/ml_features.py without running its module body."""
    tree = ast.parse(open(os.path.join(SRC, "ml_features.py")).read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in ("earnings_features", "build", "features_frame")]
    ns = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=fns, type_ignores=[]), "ml_features.py", "exec"), ns)
    return ns


def live_days():
    out = []
    for f in sorted(glob.glob(os.path.join(LOG, "entry_*.json"))):
        if "replay" in f:
            continue
        r = json.load(open(f))
        if r.get("submitted") and r["day"] >= "2026-09-29":
            mf = os.path.join(LOG, f"members_{r['day']}.json")
            r["members_log"] = json.load(open(mf)) if os.path.exists(mf) else {}
            out.append(r)
    return out


def sectors():
    s = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    ind, sec = s.industry, s.sector
    hw = ind.isin(TECH_HW) | (ind.isin(TECH_HW_TECHSECTOR) & (sec == "Technology"))
    return sec.replace("", np.nan), ind.replace("", np.nan), hw


def extended_grid(days):
    """Trading-day grid past the panel end (business days, US market holidays to year end removed)."""
    hol = pd.DatetimeIndex(["2026-11-26", "2026-12-25", "2027-01-01"])
    fut = pd.bdate_range(days[-1] + pd.Timedelta(days=1), "2027-01-29")
    return days.append(fut.difference(hol))


def universe_sets(P, cols, day):
    """liquid: the live job's fetch list (20-day median dollar volume > $5M and close > $5 before the day);
    cand: every name that can pass the day's filter (9 of the previous 19 days above $5M)."""
    h = P["dv"][cols].loc[P["c"].index < day]
    rc = P["rawc"][cols].loc[P["c"].index < day]
    liquid = [t for t, m, px in zip(cols, h.iloc[-20:].median().values, rc.iloc[-1].values) if m > 5e6 and px > 5]
    cand = list(np.array(cols)[((h.iloc[-19:] > 5e6).sum().values >= 9)])
    return liquid, sorted(set(liquid) | set(cand))


# ------------------------------------------------------------------ fetch
def news_counts(since_utc, until_utc):
    """As paper_overnight.news_counts: articles per symbol tagged with 1-3 symbols in [since, until)."""
    p = dict(start=since_utc, end=until_utc, limit=50, sort="asc", include_content="false")
    out = {}
    for _ in range(400):
        A.RL.wait()
        j = A._sess().get("https://data.alpaca.markets/v1beta1/news", params=p, timeout=60).json()
        for n in j.get("news", []):
            s = n.get("symbols") or []
            if 1 <= len(s) <= 3:
                for t in s:
                    out[t] = out.get(t, 0) + 1
        if not j.get("next_page_token"):
            break
        p["page_token"] = j["next_page_token"]
    return out


def fetch():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    have = pd.read_parquet(BARS) if os.path.exists(BARS) else pd.DataFrame()
    hn = pd.read_parquet(NEWS) if os.path.exists(NEWS) else pd.DataFrame()
    parts, nparts = [have], [hn]
    for r in live_days():
        day = pd.Timestamp(r["day"])
        _, cand = universe_sets(P, cols, day)
        tick = sorted(set(cand) | {"SPY", "IWM"})
        done = set(have.tf[have.day == day]) if len(have) else set()
        for tf, a, b, tt in [("30Min", "09:30", "15:30", tick), ("5Min", "15:40", "15:45", tick),
                             ("1Min", "15:44", "15:45", ["SPY", "IWM"])]:
            if tf in done:
                continue
            x = pd.concat([A.bars(tt[i:i + 400], tf, utc(day, a), utc(day, b)) for i in range(0, len(tt), 400)])
            x["day"], x["tf"] = day, tf
            parts.append(x)
            print(day.date(), tf, len(x), flush=True)
        if day not in days and "1Day" not in done:
            x = A.bars(tick, "1Day", utc(day, "00:00"), utc(day, "23:59"), adjustment="raw")
            x["day"], x["tf"] = day, "1Day"
            parts.append(x)
            print(day.date(), "1Day", len(x), flush=True)
        if not len(hn) or day not in set(hn.day):
            prev = days[days < day][-1]
            n = news_counts(utc(prev, "15:45"), utc(day, "15:45"))
            nparts.append(pd.DataFrame({"day": day, "ticker": list(n), "n": list(n.values())}))
            print(day.date(), "news symbols", len(n), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(BARS)
    pd.concat(nparts, ignore_index=True).to_parquet(NEWS)


# ------------------------------------------------------------------ feature frames
class Ctx:
    def __init__(self):
        self.P = load_panel()
        self.cols = stock_cols(self.P)
        self.days = self.P["c"].index
        self.grid = extended_grid(self.days)
        self.ns = load_mlf()
        E = store.read("earnings")
        self.E = E[E.symbol.isin(self.cols)]
        self.bars = pd.read_parquet(BARS) if os.path.exists(BARS) else None
        self.news = pd.read_parquet(NEWS) if os.path.exists(NEWS) else None
        self._earn_ext = None

    def earn_feats(self, grid, cols):
        self.ns.update(days=grid, cols=cols, _E=self.E[self.E.symbol.isin(cols)])
        return self.ns["earnings_features"]()

    def earn_ext(self, cols):
        key = tuple(cols)
        if self._earn_ext is None or self._earn_ext[0] != key:
            self._earn_ext = (key, self.earn_feats(self.grid, cols))
        return self._earn_ext[1]

    def earn_tonight_ext(self, cols):
        """study 33 panel: 1 on the event's grid day and the day before."""
        E = self.E[self.E.symbol.isin(cols)]
        di = self.grid.searchsorted(pd.to_datetime(E.date))
        m = pd.DataFrame(0.0, index=self.grid, columns=cols)
        ci = {t: i for i, t in enumerate(cols)}
        for d, t in zip(di, E.symbol):
            if 0 < d < len(self.grid):
                m.iat[d, ci[t]] = 1.0
                m.iat[d - 1, ci[t]] = 1.0
        return m

    def intraday(self, day, tick):
        """Live-style aggregate (paper_overnight.today_intraday, replay price) and the 15:40 SIP close."""
        b = self.bars[self.bars.day == day]
        b30 = b[b.tf == "30Min"]
        g = b30.groupby("ticker")
        agg = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "v": g.v.sum(), "c1530": g.c.last()})
        p5 = b[b.tf == "5Min"].groupby("ticker").c.last()
        agg = agg.reindex(sorted(set(agg.index) | set(p5.index)))
        agg["p1545"] = p5
        agg["p"] = agg.p1545.fillna(agg.c1530)
        agg["h"] = np.maximum(agg.h, agg.p)
        agg["l"] = np.minimum(agg.l, agg.p)
        m1 = b[b.tf == "1Min"].groupby("ticker").c.last()
        d1 = b[b.tf == "1Day"].set_index("ticker")
        return agg.reindex(tick), m1, d1

    def news_series(self, day):
        n = self.news[self.news.day == day]
        return pd.Series(n.n.values, index=n.ticker.values, dtype="float64")

    def frame(self, day, kind, tick, intr=None, m1=None, d1=None):
        """Feature frame for day (index (date, ticker)) on the 260 trading days before it plus the day row."""
        P, ns = self.P, self.ns
        hist = P["c"].index < day
        H = {k: P[k][tick].loc[hist].iloc[-260:] for k in ["o", "h", "l", "c", "rawc", "v", "dv"]}
        hd = H["c"].index
        idx = hd.append(pd.DatetimeIndex([day]))
        mk = {k: P["c"][k] for k in ["SPY", "IWM", "^VIX", "^VIX3M"]}
        in_panel = day in P["c"].index
        if kind in ("L", "Lfix"):
            adjf = (H["c"] / H["rawc"]).iloc[-1]
            row = dict(o=intr.o * adjf, h=intr.h * adjf, l=intr.l * adjf, c=intr.p * adjf, rawc=intr.p, v=intr.v,
                       dv=intr.v * intr.p)
            spy, iwm = intr.p.get("SPY", np.nan), intr.p.get("IWM", np.nan)
            vix, vix3 = mk["^VIX"].loc[hd].iloc[-1], mk["^VIX3M"].loc[hd].iloc[-1]
        else:
            if in_panel:
                D = {k: P[k][tick].loc[day] for k in ["o", "h", "l", "c", "rawc", "v", "dv"]}
                adjf = D["c"] / D["rawc"]
                vix, vix3 = mk["^VIX"].loc[day], mk["^VIX3M"].loc[day]
                spy_c, iwm_c = mk["SPY"].loc[day] / P["rawc"]["SPY"].loc[day], mk["IWM"].loc[day] / P["rawc"]["IWM"].loc[day]
            else:                                   # 2026-10-13: Alpaca daily bar, VIX of the previous day
                x = d1.reindex(tick)
                adjf = pd.Series(1.0, index=tick)
                D = dict(o=x.o, h=x.h, l=x.l, c=x.c, rawc=x.c, v=x.v, dv=x.v * x.c)
                vix, vix3 = mk["^VIX"].loc[hd].iloc[-1], mk["^VIX3M"].loc[hd].iloc[-1]
                spy_c = iwm_c = 1.0
            if kind == "B":
                px = intr.p1545
                ok = px.notna()
                c = (px * adjf).where(ok)
                row = dict(o=D["o"], h=np.maximum(D["h"], c), l=np.minimum(D["l"], c), c=c, rawc=px.where(ok),
                           v=D["v"] * 0.85, dv=D["dv"] * 0.85)
                spy, iwm = m1.get("SPY", np.nan) * spy_c, m1.get("IWM", np.nan) * iwm_c
            else:                                   # C: daily close
                row = D
                spy = mk["SPY"].loc[day] if in_panel else d1.c.get("SPY", np.nan)
                iwm = mk["IWM"].loc[day] if in_panel else d1.c.get("IWM", np.nan)

        def ext(k):
            return pd.concat([H[k], pd.DataFrame([pd.Series(row[k]).reindex(tick)], index=[day])]).reindex(idx)
        o, h, l, c, raw, v, dv = (ext(k) for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
        s = lambda k, val: pd.concat([mk[k].loc[hd], pd.Series([val], index=[day])])
        if kind == "L":                         # live: calendar grid ends the day before, today's row = yesterday's
            grid = P["c"].index[hist]
            earn = self.earn_feats(grid, tick)
            earn = {k: pd.concat([f, f.iloc[[-1]].set_axis([day])]) for k, f in earn.items()}
        else:
            grid = self.grid
            earn = self.earn_ext(tick)
        ns.update(days=grid)
        F, mkt, adv20 = ns["build"](o, h, l, c, raw, v, dv, s("SPY", spy), s("^VIX", vix), s("^VIX3M", vix3),
                                    s("IWM", iwm), earn=earn)
        univ = ((raw > 5) & (adv20 > 5e6)).iloc[[-1]] & c.iloc[[-1]].notna()
        return ns["features_frame"]({k: f.iloc[[-1]] for k, f in F.items()}, mkt.iloc[[-1]], univ)


def score(X, day, q_ens, earn, news, jump=True):
    """Ensemble (and blend) scores as paper_overnight.ml_scores; returns a DataFrame by ticker."""
    import lightgbm as lgb
    import ensemble
    import jumpmodel as J
    mdl = lgb.Booster(model_file=os.path.join(MD, f"night_{q_ens}.txt"))
    ens = ensemble.predict(X, pd.Period(q_ens, freq="Q"), pooled_model=mdl)
    tick = X.index.get_level_values(1)
    out = pd.DataFrame({k: ens[k].values for k in ens.columns}, index=tick)
    if jump:
        pj, jq = J.predict(X, pd.Period(day, freq="Q"), earn, news)
        out["p_jump"], out["p_drop"] = pj.p_jump.values, pj.p_drop.values
        out["jmd"] = out.p_jump - out.p_drop
        out["blend"] = J.blend(out.ensemble, out.p_jump, out.p_drop)
        out.attrs["jq"] = jq
    else:
        out["blend"] = out.ensemble
    return out


def picks(sc):
    from paper_overnight import cap_by_industry, tradable
    pred = sc.blend.sort_values(ascending=False)
    pred = pred[pred.index.isin(tradable(pred.index))]
    return cap_by_industry(pred, 10), pred


# ------------------------------------------------------------------ study 154
def run154():
    from scipy.stats import spearmanr
    ctx = Ctx()
    P, days = ctx.P, ctx.days
    sec, ind, hw = sectors()
    rows = []
    splits = {}
    for f in sorted(glob.glob(os.path.join(LOG, "job_entry_*.log"))):
        for line in open(f, errors="ignore"):
            if line.startswith("tickers with changed history"):
                for t in ast.literal_eval(line.split(":", 1)[1].strip()):
                    splits.setdefault(t, []).append(os.path.basename(f)[10:20])
    for r in live_days():
        day = pd.Timestamp(r["day"])
        ml = r["members_log"]
        blend_live = r["strategy"] == "blend"
        q_ens = r["model"].split("night_")[1][:6]
        liquid, cand = universe_sets(P, ctx.cols, day)
        intr_all, m1, d1 = ctx.intraday(day, sorted(set(cand) | {"SPY", "IWM"}))
        prev = days[days < day][-1]
        nxt = ctx.grid[ctx.grid > day][0]
        news = ctx.news_series(day)
        live_top = list(r["ml_top"])
        live_unc = list(r.get("uncapped_top") or [])
        res = {}
        for kind in ["L", "Lfix", "B", "C"]:
            tick = liquid + ["SPY", "IWM"] if kind in ("L", "Lfix") else cand
            intr = intr_all.reindex(liquid + ["SPY", "IWM"]) if kind in ("L", "Lfix") else intr_all
            X = ctx.frame(day, kind, [t for t in tick if t not in ("SPY", "IWM")], intr=intr, m1=m1, d1=d1)
            t = X.index.get_level_values(1)
            if kind in ("L", "Lfix"):
                import jumpmodel as J
                et = J.earn_tonight(t, prev, nxt)
                nw = news
            else:
                et = ctx.earn_tonight_ext(list(t)).loc[day]
                nw = news.reindex(t).fillna(0.0)
            sc = score(X, day, q_ens, et, nw, jump=blend_live)
            sc["earn_tonight"] = et.reindex(sc.index).values
            sc["news"] = news.reindex(sc.index).fillna(0).values
            for k in ["days_to_e", "days_since_e", "lpx", "ldv", "vol20", "r1", "r5", "vz1", "vix"]:
                sc[k] = X[k].values
            top, pred = picks(sc)
            sc["rank"] = pred.rank(ascending=False)
            res[kind] = (sc, list(top.index), list(pred.head(10).index))
            print(day.date(), kind, "universe", len(sc), "overlap with live", len(set(top.index) & set(live_top)), flush=True)
        # day summary
        for kind, (sc, top, unc) in res.items():
            rows.append(dict(table="day_summary", day=r["day"], variant=kind, strategy=r["strategy"],
                             model=r["model"], universe=len(sc), overlap_live=len(set(top) & set(live_top)),
                             overlap_live_uncapped=len(set(unc) & set(live_unc)) if live_unc else np.nan,
                             techhw_picks=int(hw.reindex(top).fillna(False).sum()),
                             live_techhw_picks=int(hw.reindex(live_top).fillna(False).sum()),
                             picks=" ".join(top), live_picks=" ".join(live_top),
                             net_live=np.nan))
        # rank correlations between variants on the common universe
        for a, b in [("L", "B"), ("L", "C"), ("B", "C"), ("L", "Lfix")]:
            x, y = res[a][0], res[b][0]
            com = x.index.intersection(y.index)
            for k in ["blend", "ensemble"] + (["jmd"] if blend_live else []):
                rows.append(dict(table="variant_corr", day=r["day"], variant=f"{a}_vs_{b}", item=k, n=len(com),
                                 value=spearmanr(x.loc[com, k], y.loc[com, k])[0],
                                 overlap=len(set(res[a][1]) & set(res[b][1]))))
        # L against the logged live values
        sc = res["L"][0]
        checks = [(f"member_{m}", sc[m], v) for m, v in ml.get("members", {}).items()]
        if blend_live:
            checks += [("ensemble_top", sc.ensemble, ml["ensemble_top"]), ("jmd_top", sc.jmd, ml["jmd_top"]),
                       ("blend_top", sc.blend, r["uncapped_top"])]
        else:
            checks += [("ensemble_top", sc.ensemble, r["ml_top"])]
        for name, ser, logged in checks:
            lg = pd.Series(logged, dtype="float64")
            mine = ser.reindex(lg.index)
            rows.append(dict(table="member_check", day=r["day"], variant="L", item=name,
                             n=len(lg), overlap=len(set(ser.nlargest(10).index) & set(lg.index)),
                             value=float((mine - lg).abs().max()), missing=int(mine.isna().sum()),
                             median_abs_diff=float((mine - lg).abs().median())))
        if blend_live:
            rows.append(dict(table="member_check", day=r["day"], variant="L", item="earn_tonight_n",
                             value=float(sc.earn_tonight.sum()), live=ml.get("earn_tonight_n")))
            rows.append(dict(table="member_check", day=r["day"], variant="L", item="news_symbols_n",
                             value=float(len(news)), live=ml.get("news_symbols_n")))
        # live picks: ranks and sanity
        qty = {o["symbol"]: o["qty"] for o in r["orders"]}
        eq = r["equity"]
        for t in live_top:
            x = res["L"][0]
            d = dict(table="live_pick", day=r["day"], ticker=t, sector=sec.get(t), industry=ind.get(t),
                     techhw=bool(hw.get(t, False)), live_score=r["ml_top"][t])
            for kind in res:
                s_ = res[kind][0]
                d[f"rank_{kind}"] = s_["rank"].get(t, np.nan)
                d[f"in_top_{kind}"] = t in res[kind][1]
            d["score_L"] = x.blend.get(t, np.nan)
            for k in ["ensemble", "jmd", "p_jump", "p_drop", "earn_tonight", "news", "days_to_e", "days_since_e",
                      "r1", "r5", "vol20", "vz1"]:
                if k in x:
                    d[k] = x[k].get(t, np.nan)
            d["days_to_e_B"] = res["B"][0].days_to_e.get(t, np.nan)
            d["ens_rank_L"] = x.ensemble.rank(ascending=False).get(t, np.nan)
            d["jmd_rank_L"] = x.jmd.rank(ascending=False).get(t, np.nan) if "jmd" in x else np.nan
            d["p1545_sip"] = intr_all.p1545.get(t, np.nan)
            q_ = qty.get(t)
            if q_:      # qty = int(equity * 0.1 // price): the live price lies in (eq/10/(q+1), eq/10/q]
                d["live_px_lo"], d["live_px_hi"] = eq / 10 / (q_ + 1), eq / 10 / q_
            hist = P["rawc"][t].loc[days < day].tail(20) if t in P["rawc"] else pd.Series(dtype=float)
            vv = P["v"][t].loc[days < day].tail(20) if t in P["v"] else pd.Series(dtype=float)
            d["prev_close"] = float(hist.iloc[-1]) if len(hist) else np.nan
            d["last_close_date"] = str(hist.dropna().index[-1].date()) if hist.notna().any() else ""
            d["adv20"] = float((hist * vv).median()) if len(hist) else np.nan
            d["zero_vol_days20"] = int((vv.fillna(0) <= 0).sum())
            d["max_abs_daily_ret20"] = float(P["c"][t].loc[days < day].tail(21).pct_change().abs().max()) if t in P["c"] else np.nan
            d["split_flags"] = " ".join(splits.get(t, []))
            rows.append(d)
    rows += earn_nan_test(ctx, hw)
    df = pd.DataFrame(rows)
    # virtual book next to the day summary
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv"), index_col=0)
    m = df.table == "day_summary"
    df.loc[m, "net_live"] = df.loc[m, "day"].map(vd.net)
    # backtest-variant night returns of the B picks vs live picks (panel open/close, gross)
    c, o = P["c"], P["o"]
    for i in df.index[m]:
        d0 = pd.Timestamp(df.at[i, "day"])
        if d0 not in c.index or c.index[-1] <= d0:
            continue
        d1_ = c.index[c.index.get_loc(d0) + 1]
        for col, src in [("gross_variant", "picks"), ("gross_live_panel", "live_picks")]:
            tk = [t for t in df.at[i, src].split() if t in c.columns]
            df.at[i, col] = float((o.loc[d1_, tk] / c.loc[d0, tk] - 1).mean())
    df.to_csv(f"{RES}/study154_live_consistency.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[m][["day", "variant", "universe", "overlap_live", "overlap_live_uncapped", "techhw_picks",
                 "live_techhw_picks", "net_live", "gross_variant", "gross_live_panel"]].to_string(index=False))
    print(df[df.table == "variant_corr"].pivot_table(index="day", columns=["variant", "item"], values="value").round(3).to_string())
    print(df[df.table == "member_check"][["day", "item", "n", "overlap", "value", "median_abs_diff", "live"]].to_string(index=False))
    en = df[df.table == "earn_nan_test"]
    print(en.groupby("variant")[["overlap", "value", "gross_variant", "gross_live_panel", "techhw_picks",
                                 "live_techhw_picks", "nan_share_days_to_e"]].mean().round(4).to_string())
    lp = df[df.table == "live_pick"]
    print(lp[["day", "ticker", "industry", "rank_L", "rank_Lfix", "rank_B", "rank_C", "ens_rank_L", "jmd_rank_L",
              "news", "earn_tonight", "days_to_e", "days_to_e_B", "p1545_sip", "live_px_lo", "live_px_hi",
              "prev_close", "last_close_date", "adv20", "zero_vol_days20", "max_abs_daily_ret20",
              "split_flags"]].round(3).to_string(index=False))

def earn_nan_test(ctx, hw):
    """Effect of the live earnings-calendar gap (days_to_e missing for nearly every name) on backtest days:
    close-feature frames scored with the quarter's models, once with days_to_e as in training and once with
    days_to_e set to NaN (as live). 2026Q3 every second day plus 40 random days of 2024-01..2026-06."""
    from paper_overnight import cap_by_industry
    from scipy.stats import spearmanr
    days, P = ctx.days, ctx.P
    F = pd.read_pickle(os.path.join(A.LOCAL, "factor_panels.pkl"))["n_news"]
    rng = np.random.default_rng(154)
    early = days[(days >= "2024-01-02") & (days <= "2026-06-29")]
    q3 = days[(days >= "2026-07-01") & (days <= "2026-09-24")][::2]
    sample = [("2024-26H1", pd.Timestamp(d)) for d in sorted(rng.choice(early, 40, replace=False))] + \
             [("2026Q3", d) for d in q3]
    out = []
    for lab, d in sample:
        _, cand = universe_sets(P, ctx.cols, d)
        X = ctx.frame(d, "C", cand)
        t = list(X.index.get_level_values(1))
        et = ctx.earn_tonight_ext(t).loc[d]
        nw = F.loc[d].reindex(t).fillna(0.0)
        q = str(pd.Period(d, freq="Q"))
        a = score(X, d, q, et, nw)
        X2 = X.copy()
        X2["days_to_e"] = np.nan
        b = score(X2, d, q, et, nw)
        ta = list(cap_by_industry(a.blend.sort_values(ascending=False), 10, 3).index)
        tb = list(cap_by_industry(b.blend.sort_values(ascending=False), 10, 3).index)
        i = days.get_loc(d)
        R = P["o"].iloc[i + 1] / P["c"].iloc[i] - 1
        out.append(dict(table="earn_nan_test", day=str(d.date()), variant=lab, item="days_to_e NaN vs as trained",
                        n=len(a), overlap=len(set(ta) & set(tb)), value=spearmanr(a.blend, b.blend)[0],
                        gross_variant=float(R.reindex(ta).mean()), gross_live_panel=float(R.reindex(tb).mean()),
                        techhw_picks=int(hw.reindex(ta).fillna(False).sum()),
                        live_techhw_picks=int(hw.reindex(tb).fillna(False).sum()),
                        nan_share_days_to_e=float(X.days_to_e.isna().mean())))
    return out


# ------------------------------------------------------------------ study 150
def backtest_scores(P, cols):
    """Blend scores of the backtest as study 69 lines 22-46."""
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    return S.loc["2024-01":"2026-09"]


def trait_panels(P, cols, idx):
    """Trait panels (date x ticker) on the dates idx, all known at 15:45 of the date except 'ret_close'."""
    c, rawc, dv = P["c"][cols], P["rawc"][cols], P["dv"][cols]
    lr = np.log(c / c.shift(1))
    spy = np.log(P["c"]["SPY"] / P["c"]["SPY"].shift(1))
    snap = pd.read_pickle(os.path.join(A.LOCAL, "m5snap_panels_686_none.pkl"))["c15:40"].reindex(index=idx, columns=cols)
    adjf = (c / rawc).reindex(idx)
    T = {"price": snap.where(snap.notna(), rawc.reindex(idx)),
         "adv20": dv.shift(1).rolling(20, min_periods=10).median().reindex(idx),
         "vol20": lr.shift(1).rolling(20, min_periods=15).std().reindex(idx),
         "ret_1545": (snap * adjf / c.shift(1).reindex(idx) - 1),
         "ret_close": (c / c.shift(1) - 1).reindex(idx),
         "ret5": (c.shift(1) / c.shift(6) - 1).reindex(idx),
         "beta60": (lr.shift(1).rolling(60, min_periods=40).cov(spy.shift(1))
                    .div(spy.shift(1).rolling(60, min_periods=40).var(), axis=0)).reindex(idx)}
    del snap
    F = pd.read_pickle(os.path.join(A.LOCAL, "factor_panels.pkl"))
    T["news"] = F["n_news"].reindex(index=idx, columns=cols)
    del F
    return T


def run150():
    import lightgbm as lgb
    from paper_overnight import cap_by_industry
    P = load_panel()
    cols = stock_cols(P)
    sec, ind, hw = sectors()
    S = backtest_scores(P, cols)
    rows = []
    pk = []
    for d, row in S.iterrows():
        r = row.dropna().sort_values(ascending=False, kind="stable")
        for t in cap_by_industry(r, 10, 3).index:
            pk.append((d, t))
    pk = pd.MultiIndex.from_tuples(pk, names=["date", "ticker"])
    univ = S.notna()
    T = trait_panels(P, cols, S.index)
    per = lambda d: np.where(d < pd.Timestamp("2025-01-01"), "2024", np.where(d < pd.Timestamp("2026-01-01"), "2025", "2026"))
    di = S.index.get_indexer(pk.get_level_values(0))
    ci = pd.Index(cols).get_indexer(pk.get_level_values(1))
    picks = pd.DataFrame(index=pk)
    picks["period"] = per(pk.get_level_values(0))
    hwv = hw.reindex(cols).fillna(False).values
    picks["techhw"] = hwv[ci]
    picks["sector"] = sec.reindex(pk.get_level_values(1)).fillna("(none)").values
    for k, X in T.items():
        v = X.values
        picks[k] = v[di, ci]
        pct = X.where(univ).rank(axis=1, pct=True).values
        picks[k + "_pct"] = pct[di, ci]
    picks["news"] = picks.news.fillna(0)
    # universe means for reference
    uni = {}
    for k, X in T.items():
        Xu = X.where(univ)
        for p in ["2024", "2025", "2026"]:
            m = per(Xu.index) == p
            uni[(p, k)] = float(np.nanmedian(Xu.values[m]))
    # live October picks
    live = []
    for r in live_days():
        if r["strategy"] != "blend":
            continue
        for t in r["ml_top"]:
            live.append((pd.Timestamp(r["day"]), t))
    lv = pd.DataFrame(live, columns=["date", "ticker"])
    lt = live_traits(P, cols, lv)
    lt["techhw"] = hw.reindex(lt.ticker).fillna(False).values
    lt["sector"] = sec.reindex(lt.ticker).fillna("(none)").values
    lt["period"] = "live_2026-10"
    traits = ["price", "adv20", "vol20", "ret_1545", "ret_close", "ret5", "news", "beta60"]
    groups_ = [(p, picks[picks.period == p]) for p in ["2024", "2025", "2026"]] + [("2024-26", picks), ("live_2026-10", lt)]
    for p, x in groups_:
        for k in traits + ["techhw"]:
            v = x[k].astype(float)
            d = dict(table="traits", period=p, item=k, n=int(v.notna().sum()), mean=v.mean(), median=v.median(),
                     universe_median=uni.get((p, k), np.nan))
            if k in traits and k + "_pct" in x:
                d["pct_in_universe_mean"] = x[k + "_pct"].mean()
            if p == "live_2026-10" and k in traits:       # where the live picks fall in the backtest pick distribution
                ref = picks[k].dropna().values
                d["pct_vs_backtest_picks"] = float(np.mean([np.mean(ref <= z) for z in v.dropna()]))
                ref26 = picks.loc[picks.period == "2026", k].dropna().values
                d["pct_vs_2026_picks"] = float(np.mean([np.mean(ref26 <= z) for z in v.dropna()]))
            rows.append(d)
        for sct, val in x.sector.value_counts(normalize=True).items():
            rows.append(dict(table="sector_share", period=p, item=sct, mean=val, n=len(x)))
    # tech hardware decomposition by half-year
    half = lambda d: [f"{x.year}H{1 if x.month <= 6 else 2}" for x in d]
    vpct = T["vol20"].where(univ).rank(axis=1, pct=True)
    mpct = T["ret5"].where(univ).rank(axis=1, pct=True)
    picks["half"] = half(pk.get_level_values(0))
    hwmask = pd.DataFrame(np.broadcast_to(hwv, univ.shape), index=univ.index, columns=cols)
    for h in sorted(set(picks.half)):
        m = np.array(half(univ.index)) == h
        U = univ.values[m]
        HW = hwmask.values[m]
        V, M5 = vpct.values[m], mpct.values[m]
        x = picks[picks.half == h]
        d = dict(table="techhw_decomp", period=h, item="tech hardware", n=len(x), picks_share=x.techhw.mean(),
                 universe_share=HW[U].mean(), share_top_vol_decile=HW[U & (V > 0.9)].mean(),
                 share_top_ret5_decile=HW[U & (M5 > 0.9)].mean(),
                 hw_mean_vol_pct=np.nanmean(V[U & HW]), hw_mean_ret5_pct=np.nanmean(M5[U & HW]),
                 picks_mean_vol_pct=x.vol20_pct.mean(), picks_mean_ret5_pct=x.ret5_pct.mean())
        # trait-only expectation: picks drawn at random from the universe within the same vol x ret5 x price
        # buckets (deciles of vol20, terciles of ret5) as the actual picks
        vb = np.minimum((V * 10).astype("float"), 9)
        mb = np.minimum((M5 * 3).astype("float"), 2)
        ok = U & np.isfinite(vb) & np.isfinite(mb)
        key = (np.nan_to_num(vb) * 3 + np.nan_to_num(mb)).astype(int)
        uh = pd.DataFrame({"k": key[ok], "hw": HW[ok]}).groupby("k").hw.mean()
        kp = (np.minimum((x.vol20_pct * 10).fillna(-1), 9).astype(int) * 3
              + np.minimum((x.ret5_pct * 3).fillna(-1), 2).astype(int))
        d["trait_expected_share"] = float(kp.map(uh).mean())
        rows.append(d)
    del T, vpct, mpct, hwmask
    # gain importance
    imp = {}
    for q in ["2024Q1", "2025Q1", "2026Q1", "2026Q4"]:
        for fam in ["night", "ens_pooled_com", "ens_pooled_beh", "jump_jump", "jump_drop"]:
            m = lgb.Booster(model_file=os.path.join(MD, f"{fam}_{q}.txt"))
            g = pd.Series(m.feature_importance("gain"), index=m.feature_name())
            g = 100 * g / g.sum()
            imp[(fam, q)] = g
            for f, v in g.items():
                rows.append(dict(table="importance", period=q, item=f, model=fam, mean=v))
    # cross-scoring: old and new model sets on the same days (close-feature frames)
    rows += cross_score(hw)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study150_pick_traits.csv", index=False)
    pd.set_option("display.width", 250)
    t = df[df.table == "traits"]
    print(t.pivot_table(index="item", columns="period", values="median", sort=False).round(4).to_string())
    print(t.pivot_table(index="item", columns="period", values="pct_in_universe_mean", sort=False).round(3).to_string())
    print(t[t.period == "live_2026-10"][["item", "mean", "median", "pct_vs_backtest_picks", "pct_vs_2026_picks"]].round(3).to_string(index=False))
    print(t.pivot_table(index="item", columns="period", values="universe_median", sort=False).round(4).to_string())
    print(df[df.table == "sector_share"].pivot_table(index="item", columns="period", values="mean").round(3).to_string())
    print(df[df.table == "techhw_decomp"].dropna(axis=1, how="all").round(3).to_string(index=False))
    ii = df[df.table == "importance"].pivot_table(index="item", columns=["model", "period"], values="mean")
    for fam in ["night", "ens_pooled_com", "jump_jump", "jump_drop"]:
        x = ii[fam]
        print(fam, "\n", x.loc[x.max(axis=1).sort_values(ascending=False).index[:15]].round(1).to_string())
    print(df[df.table == "cross_score"].dropna(axis=1, how="all").round(3).to_string(index=False))


def live_traits(P, cols, lv):
    """Traits of the live picks from the panel before the day and the cached 15:40 SIP bar."""
    bars = pd.read_parquet(BARS)
    nw = pd.read_parquet(NEWS)
    c, rawc, dv = P["c"], P["rawc"], P["dv"]
    days = c.index
    spy = np.log(c["SPY"] / c["SPY"].shift(1))
    out = []
    for d, t in lv.itertuples(index=False):
        h = days < d
        p = bars[(bars.day == d) & (bars.tf == "5Min") & (bars.ticker == t)].c
        p = float(p.iloc[-1]) if len(p) else np.nan
        lr = np.log(c[t].loc[h] / c[t].loc[h].shift(1))
        sp = spy.loc[h]
        pc = float(rawc[t].loc[h].iloc[-1])
        rc = float(rawc[t].loc[d]) if d in days else np.nan
        n = nw[(nw.day == d) & (nw.ticker == t)].n
        out.append(dict(date=d, ticker=t, price=p, adv20=float(dv[t].loc[h].tail(20).median()),
                        vol20=float(lr.tail(20).std()), ret_1545=p / pc - 1, ret_close=rc / pc - 1,
                        ret5=float(c[t].loc[h].iloc[-1] / c[t].loc[h].iloc[-6] - 1),
                        news=float(n.sum()) if len(n) else 0.0,
                        beta60=float(lr.tail(60).cov(sp.tail(60)) / sp.tail(60).var())))
    return pd.DataFrame(out)


def cross_score(hw):
    """2024Q1 and 2026Q4 model sets on the same close-feature frames (20 days of 2024H1 and 20 of 2026Q3)."""
    from paper_overnight import cap_by_industry
    ctx = Ctx()
    days = ctx.days
    F = pd.read_pickle(os.path.join(A.LOCAL, "factor_panels.pkl"))["n_news"]
    rng = np.random.default_rng(150)
    out = []
    for lab, a, b in [("2024H1 days", "2024-01-02", "2024-06-28"), ("2026Q3 days", "2026-07-01", "2026-09-24")]:
        cand_days = days[(days >= a) & (days <= b)]
        sample = sorted(rng.choice(cand_days, 20, replace=False))
        res = {}
        for d in sample:
            d = pd.Timestamp(d)
            _, cand = universe_sets(ctx.P, ctx.cols, d)
            X = ctx.frame(d, "C", cand)
            t = list(X.index.get_level_values(1))
            et = ctx.earn_tonight_ext(t).loc[d]
            nw = F.loc[d].reindex(t).fillna(0.0)
            for q in ["2024Q1", "2026Q4"]:
                sc = score_q(X, q, et, nw)
                top = cap_by_industry(sc.blend.sort_values(ascending=False), 10, 3)
                res.setdefault(q, []).append((d, list(top.index), sc))
        for q in res:
            share = np.mean([hw.reindex(tp).fillna(False).mean() for _, tp, _ in res[q]])
            ens_share = np.mean([hw.reindex(sc.ensemble.nlargest(10).index).fillna(False).mean() for _, _, sc in res[q]])
            jmd_share = np.mean([hw.reindex(sc.jmd.nlargest(10).index).fillna(False).mean() for _, _, sc in res[q]])
            out.append(dict(table="cross_score", period=lab, item=f"models {q}", n=len(res[q]), mean=share,
                            ens_top10_share=ens_share, jmd_top10_share=jmd_share))
        ov = np.mean([len(set(a_[1]) & set(b_[1])) for a_, b_ in zip(res["2024Q1"], res["2026Q4"])])
        from scipy.stats import spearmanr
        rho = np.mean([spearmanr(a_[2].blend, b_[2].blend.reindex(a_[2].index))[0]
                       for a_, b_ in zip(res["2024Q1"], res["2026Q4"])])
        out.append(dict(table="cross_score", period=lab, item="overlap 2024Q1 vs 2026Q4 top 10", n=len(sample),
                        mean=ov, spearman_blend=rho))
        print(lab, "done", flush=True)
    return out


def score_q(X, q, et, news):
    """Blend score with the ensemble and jump models of quarter q."""
    import lightgbm as lgb
    import ensemble
    import jumpmodel as J
    mdl = lgb.Booster(model_file=os.path.join(MD, f"night_{q}.txt"))
    ens = ensemble.predict(X, pd.Period(q, freq="Q"), pooled_model=mdl)
    tick = X.index.get_level_values(1)
    out = pd.DataFrame({"ensemble": ens["ensemble"].values}, index=tick)
    pj, _ = J.predict(X, pd.Period(q, freq="Q"), et, news)
    out["jmd"] = pj.p_jump.values - pj.p_drop.values
    out["blend"] = J.blend(out.ensemble, pd.Series(pj.p_jump.values, index=tick), pd.Series(pj.p_drop.values, index=tick))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "154"
    if what == "fetch":
        fetch()
    elif what == "154":
        run154()
    elif what == "150":
        run150()
