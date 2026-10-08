"""Study 88: what drives the fall of the overnight picks in the first minutes after the opening auction, and can
it be used? 2024-01 .. 2026-09.

PLACEHOLDER docstring, completed below once the study is run.

    python src/study88_open_dynamics.py fetch   # opening-cross prints (price, size) per pick, incremental
    python src/study88_open_dynamics.py         # study
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A

A.RL = A.RateLimiter(90)
AUC_FN = os.path.join(A.LOCAL, "auctions88.parquet")


# ------------------------------------------------------------------ fetch: opening cross print per pick
def _open_cross(t, day):
    """Primary-exchange opening cross ('O' print) for ticker t on day: price, size, time, exchange, plus the
    first regular trade price and the share volume printed before the cross (all venues, 09:30 onward)."""
    wins = [("09:30:00", "09:30:20"), ("09:30:20", "09:32:00"), ("09:32:00", "09:45:00")]
    first, pre_sz = np.nan, 0
    for wa, wb in wins:
        a = pd.Timestamp(f"{day} {wa}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        b = pd.Timestamp(f"{day} {wb}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        p = dict(symbols=t, start=a, end=b, limit=10000, feed="sip")
        while True:
            j = A.get("trades", p)
            for x in (j.get("trades") or {}).get(t, []):
                c = x.get("c", [])
                if "O" in c:
                    return dict(open_px=x["p"], open_sz=x["s"], open_ts=x["t"], open_x=x.get("x"), first_px=first,
                                pre_sz=pre_sz)
                if not set(c) & {"I", "T", "U", "Z"}:
                    if np.isnan(first):
                        first = x["p"]
                    pre_sz += x["s"]
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
    return dict(open_px=np.nan, open_sz=np.nan, open_ts=None, open_x=None, first_px=first, pre_sz=pre_sz)


def fetch():
    import study85_87_open_execution as S85
    pk = S85.pick_list()
    pk = pk[pk.tdate <= pd.Timestamp("2026-09-30")]
    want = pd.DataFrame({"ticker": pk.symbol.str.replace("-", ".", regex=False).values, "date": pk.tdate.values})
    have = pd.read_parquet(AUC_FN) if os.path.exists(AUC_FN) else pd.DataFrame(columns=["ticker", "date"])
    have["date"] = pd.to_datetime(have["date"])
    todo = want.merge(have[["ticker", "date"]], how="left", indicator=True)
    todo = list(todo[todo._merge == "left_only"][["ticker", "date"]].itertuples(index=False, name=None))
    print("pairs to fetch", len(todo), flush=True)

    def one(r):
        try:
            d = _open_cross(r[0], r[1].strftime("%Y-%m-%d"))
        except RuntimeError as e:
            print("err", r, e, flush=True)
            d = dict(open_px=np.nan, open_sz=np.nan, open_ts=None, open_x=None, first_px=np.nan, pre_sz=np.nan)
        return dict(ticker=r[0], date=r[1], **d)
    rows = []
    for i in range(0, len(todo), 300):
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            rows += list(ex.map(one, todo[i:i + 300]))
        out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        out["open_ts"] = out.open_ts.astype("string")
        out["open_x"] = out.open_x.astype("string")
        out.to_parquet(AUC_FN, index=False)
        print("auctions88", len(rows), "/", len(todo), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch()
    sys.exit()


# ------------------------------------------------------------------ study
import study85_87_open_execution as S85   # noqa: E402  (loads the panel, picks and blend score)
from core import ann_stats, traded_close, RES, DATA   # noqa: E402

P, cols, PER = S85.P, S85.cols, S85.PER
P1, P2 = ("2024-01", "2025-06"), ("2025-07", "2026-09")
UNIV = {}


def day_clustered_ols(y, X, g):
    """OLS with standard errors clustered by day g. Returns (coef, t)."""
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    S = np.zeros((X.shape[1], X.shape[1]))
    for idx in pd.Series(np.arange(len(y))).groupby(g).indices.values():
        s = X[idx].T @ e[idx]
        S += np.outer(s, s)
    V = XtX @ S @ XtX
    return b, b / np.sqrt(np.diag(V))


def snap_tendency():
    """Per stock-day, from 5-minute bars of the whole universe (data/local/m5snap): e5 = open of the 09:35 bar /
    open of the 09:30 bar - 1 and e30 = open of the 10:00 bar / open of the 09:30 bar - 1. Returns wide
    date x ticker frames (Yahoo-style tickers)."""
    parts = []
    for f in sorted(os.listdir(os.path.join(DATA, "local", "m5snap"))):
        if not f.endswith(".parquet") or f[:7] < "2023-09":
            continue
        d = pd.read_parquet(os.path.join(DATA, "local", "m5snap", f), columns=["ts", "ticker", "o"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[hm.isin([930, 935, 1000])]
        d = d.assign(date=d.ts.dt.normalize(), hm=hm[hm.isin([930, 935, 1000])].values)
        w = d.pivot_table(index=["date", "ticker"], columns="hm", values="o", aggfunc="first")
        parts.append(w)
    w = pd.concat(parts)
    e5 = (w[935] / w[930] - 1).unstack()
    e30 = (w[1000] / w[930] - 1).unstack()
    ren = {c: c.replace(".", "-") for c in e5.columns}
    return e5.rename(columns=ren), e30.rename(columns=ren)


def bar_anchor(pk):
    """Study 85's anchor per pair: open of the first 1-minute bar within 09:30-09:35 (m1s85, split-adjusted)."""
    key = {(s_.replace("-", "."), d): i for i, (s_, d) in enumerate(zip(pk.symbol, pk.tdate))}
    f = np.full(len(pk), np.nan)
    for m_ in sorted(pk.tdate.dt.strftime("%Y-%m").unique()):
        fn = os.path.join(A.LOCAL, S85.DS, f"{m_}.parquet")
        if not os.path.exists(fn):
            continue
        d = pd.read_parquet(fn, columns=["ts", "ticker", "o"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[(hm >= 930) & (hm <= 935)].sort_values(["ticker", "ts"]).groupby(["ticker", d.ts.dt.normalize()]).o.first()
        for (t_, dd), v in d.items():
            i = key.get((t_, dd))
            if i is not None:
                f[i] = v
    return f


def build():
    pk, N = S85.build_pairs()
    C = S85.cost_grid(pk)
    n = len(pk)
    ok = pk.bars_ok.values
    nan = np.full(n, np.nan)
    for k, lab in [(1, "r0931"), (5, "r0935"), (15, "r0945"), (30, "r1000")]:
        pk[lab] = np.where(ok, N["o"][:, k] - 1, nan)
    pk["r_close"] = np.where(ok, pk.r_day, nan)
    pk["r1000_close"] = (1 + pk.r_close) / (1 + pk.r1000) - 1
    pk["per"] = np.where(pk.pdate.dt.strftime("%Y-%m") <= P1[1], "P1", "P2")
    # ---- daily-panel features known at the open of tdate (values of pdate)
    c, dv = P["c"][cols], P["dv"][cols]
    lr = np.log(c / c.shift(1))
    oo = P["o"][cols]
    night_all = oo / c.shift(1) - 1
    day_all = c / oo - 1
    feats = {"adv20": dv.rolling(20, min_periods=10).mean(), "vol20": lr.rolling(20, min_periods=10).std(),
             "px": traded_close(P)[cols], "ret1": c / c.shift(1) - 1, "ret5": c / c.shift(5) - 1,
             # daily reversal tendency: 60-day correlation of the overnight return with the following day session
             "nd_corr60": night_all.rolling(60, min_periods=40).corr(day_all)}
    pi_ = c.index.get_indexer(pk.pdate)
    jj = c.columns.get_indexer(pk.symbol)
    for k, v in feats.items():
        pk[k] = v.values[pi_, jj]
    pk["gap_z"] = pk.night / pk.vol20
    # blend score and its rank among the day's 10 picks (1 = highest)
    S = S85.S
    pk["score"] = S.reindex(index=c.index, columns=cols).values[pi_, jj]
    pk["score_rank"] = pk.groupby("pdate").score.rank(ascending=False)
    # ---- universe opening-minutes tendency (m5snap), known before tdate's open
    e5, e30 = snap_tendency()
    e5 = e5.reindex(index=c.index, columns=cols)
    e30 = e30.reindex(index=c.index, columns=cols)
    up = night_all > 0.01
    pk["e5_univ_today_chk"] = e5.values[c.index.get_indexer(pk.tdate), jj]
    UNIV.update(e5=e5, e30=e30, night=night_all, adv=feats["adv20"].shift(1), px=feats["px"].shift(1))
    t5 = e5.rolling(60, min_periods=20).mean()
    t5up = e5.where(up).rolling(120, min_periods=5).mean()
    t30 = e30.rolling(60, min_periods=20).mean()
    pk["past_e5_60d"] = t5.values[pi_, jj]                     # through pdate: known at tdate's open
    pk["past_e5_gapup_120d"] = t5up.values[pi_, jj]
    pk["past_e30_60d"] = t30.values[pi_, jj]
    # past fall of the same stock on its earlier pick days (bars, open -> 09:35), at least 2 earlier
    pk = pk.sort_values(["symbol", "tdate"])
    pk["past_pick_r0935"] = pk.groupby("symbol").r0935.transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    pk = pk.sort_index()
    # ---- study 25 premarket / news / earnings features for tdate
    x = pd.read_pickle(f"{DATA}/local/m5pre/_study25_features.pkl")
    x = x[["ticker", "date", "pm_last", "prevc", "o930", "gap", "pm_trend", "pm_range", "relvol", "pm_mins",
           "n_news", "sent_mean", "earn_t", "earn_prev", "earn"]].rename(columns={"gap": "pm_gap"})
    pk = pk.merge(x, left_on=["symbol", "tdate"], right_on=["ticker", "date"], how="left").drop(columns=["ticker", "date"])
    pk["open_vs_pm"] = pk.o930 / pk.pm_last - 1                  # auction (first trade) vs last premarket price
    pk["has_pm"] = pk.pm_last.notna()
    # earnings and news for the pairs study 25 does not cover
    import store
    e = store.read("earnings")[["symbol", "date"]]
    e["date"] = pd.to_datetime(e.date).dt.normalize()
    ek = set(zip(e.symbol, e.date))
    pk["earn_any"] = [((s, t) in ek) or ((s, p) in ek) for s, t, p in zip(pk.symbol, pk.tdate, pk.pdate)]
    pk["earn_any"] = pk.earn_any.astype(int)
    # ---- opening cross prints (fetch above)
    if os.path.exists(AUC_FN):
        a = pd.read_parquet(AUC_FN)
        a["ticker"] = a.ticker.str.replace(".", "-", regex=False)
        a["date"] = pd.to_datetime(a.date)
        pk = pk.merge(a, left_on=["symbol", "tdate"], right_on=["ticker", "date"], how="left").drop(
            columns=["ticker", "date"])
        pk["auc_dv_adv"] = pk.open_px * pk.open_sz / pk.adv20
        pk["pre_cross_share"] = pk.pre_sz / (pk.pre_sz + pk.open_sz)
        # the bar anchor of study 85 (first 1-minute bar open, split-adjusted) vs the official opening cross (raw
        # price). Same scale unless a later split was applied to the bars: then use the first trade before the cross
        # (first_px, raw) as the anchor's raw value.
        f = bar_anchor(pk)
        araw = np.where(pk.first_px.notna(), pk.first_px, pk.open_px)
        same = np.abs(f / araw - 1) < 0.2
        pk["anchor_vs_cross"] = np.where(same, f / pk.open_px, araw / pk.open_px) - 1
        for k in ["r0931", "r0935", "r0945", "r1000", "r_close"]:
            pk[k + "_bar"] = pk[k]
            pk[k] = (1 + pk[k]) * (1 + pk.anchor_vs_cross) - 1   # from the official cross (NaN if not found)
        pk["cross_found"] = pk.open_px.notna()
        # Yahoo open (used for 'night' and the backtest's sell price) vs the official cross, in traded prices
        tc = traded_close(P)[cols]
        yo = (P["o"][cols] / P["c"][cols] * tc).values[c.index.get_indexer(pk.tdate), jj]
        pk["yahoo_open_vs_cross"] = yo / pk.open_px - 1
    assert len(pk) == n, "merge duplicated rows"
    return pk.reset_index(drop=True), C, N


# ------------------------------------------------------------------ cross-section
FEATS = ["night", "gap_z", "pm_gap", "open_vs_pm", "pm_trend", "relvol", "n_news", "earn_any", "px", "adv20",
         "vol20", "ret1", "nd_corr60", "past_e5_60d", "past_e5_gapup_120d", "past_pick_r0935", "score_rank",
         "auc_dv_adv", "pre_cross_share"]
DISCRETE = {"n_news": [-0.5, 0.5, 1.5, 3.5, 1e9], "earn_any": [-0.5, 0.5, 1.5], "score_rank": [0, 3.5, 7.5, 11]}
TGT = ["r0931", "r0935", "r1000", "r1000_close", "r_close"]


def mean_t(y, g):
    """Mean and day-clustered t-stat."""
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 20:
        return np.nan, np.nan
    b, t = day_clustered_ols(y[m], np.ones((m.sum(), 1)), np.asarray(g)[m])
    return b[0], t[0]


def cross_section(pk):
    rows = []
    ok = pk.bars_ok.values
    for f in FEATS:
        if f not in pk:
            continue
        for per in ["P1", "P2"]:
            d = pk[ok & (pk.per == per).values & pk[f].notna().values]
            if len(d) < 100:
                continue
            if f in DISCRETE:
                b = pd.cut(d[f], DISCRETE[f], labels=False)
            else:
                b = pd.qcut(d[f], 5, labels=False, duplicates="drop")
            for q, dd in d.groupby(b):
                r = dict(table="bucket", feature=f, period=per, bucket=int(q) + 1, lo=dd[f].min(), hi=dd[f].max(),
                         n=len(dd), mean_feature=dd[f].mean())
                for t_ in TGT:
                    m_, tt = mean_t(dd[t_], dd.pdate)
                    r[f"{t_}_bp"] = 1e4 * m_
                    r[f"{t_}_t"] = tt
                rows.append(r)
            # univariate slope: target on the feature's within-period percentile rank (bp per 0 -> 1 of the rank)
            z = d[f].rank(pct=True).values - 0.5
            for t_ in ["r0935", "r1000"]:
                y = d[t_].clip(d[t_].quantile(0.005), d[t_].quantile(0.995)).values
                m = np.isfinite(y)
                bb, tt = day_clustered_ols(y[m], np.c_[np.ones(m.sum()), z[m]], d.pdate.values[m])
                rows.append(dict(table="slope_rank", feature=f, period=per, target=t_, n=int(m.sum()),
                                 coef_bp=1e4 * bb[1], t=tt[1]))
    return rows


MODEL_FEATS = ["night", "gap_z", "pm_gap", "open_vs_pm", "relvol", "n_news", "earn_any", "px", "adv20", "vol20",
               "ret1", "nd_corr60", "past_e5_60d", "score_rank", "auc_dv_adv"]


def design(pk, feats, ref):
    """Within-period percentile ranks of each feature (missing -> 0.5, i.e. median), plus a missing-premarket flag.
    ref: rows used for winsorizing nothing (ranks need no scaling); ranks are computed per pick day to stay causal."""
    X = []
    for f in feats:
        r = pk.groupby("pdate")[f].rank(pct=True) if f not in ("earn_any", "n_news") else (pk[f] > 0).astype(float)
        X.append(r.fillna(0.5).values - 0.5)
    X.append((~pk.has_pm).astype(float).values)
    return np.column_stack([np.ones(len(pk))] + X)


def model(pk, target="r0935", feats=MODEL_FEATS):
    """Pooled OLS of the target on within-day percentile ranks, fit on P1; coefficients by period and P1-fit
    predictions for every row (out of sample in P2)."""
    feats = [f for f in feats if f in pk and pk[f].notna().mean() > 0.5]
    X = design(pk, feats, None)
    y = pk[target].values
    lo, hi = np.nanquantile(y[pk.per.values == "P1"], [0.005, 0.995])
    y = np.clip(y, lo, hi)
    rows = []
    coefs = {}
    for per in ["P1", "P2"]:
        m = (pk.per.values == per) & np.isfinite(y) & pk.bars_ok.values
        b, t = day_clustered_ols(y[m], X[m], pk.pdate.values[m])
        coefs[per] = b
        for nm, bb, tt in zip(["const"] + feats + ["no_premarket"], b, t):
            rows.append(dict(table=f"multi_{target}", feature=nm, period=per, n=int(m.sum()), coef_bp=1e4 * bb, t=tt))
    pred = X @ coefs["P1"]
    # out-of-sample check: daily Spearman IC and top/bottom tercile (by P1 cut points) in each period
    for per in ["P1", "P2"]:
        m = (pk.per.values == per) & np.isfinite(y) & pk.bars_ok.values
        d = pd.DataFrame({"p": pred[m], "y": pk[target].values[m], "g": pk.pdate.values[m]})
        ic = d.groupby("g").apply(lambda z: z.p.corr(z.y, method="spearman"), include_groups=False).dropna()
        q1, q2 = np.quantile(pred[(pk.per.values == "P1") & np.isfinite(y)], [1 / 3, 2 / 3])
        r = dict(table=f"multi_{target}_oos", feature="pred_P1fit", period=per, n=int(m.sum()),
                 ic_mean=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)))
        for lab, mm in [("low", d.p <= q1), ("mid", (d.p > q1) & (d.p <= q2)), ("high", d.p > q2)]:
            mt = mean_t(d.y[mm], d.g[mm])
            r[f"{lab}_bp"], r[f"{lab}_t"], r[f"{lab}_n"] = 1e4 * mt[0], mt[1], int(mm.sum())
        rows.append(r)
    return rows, pred


def universe_context():
    """All liquid stocks (ADV > $20M, price > $5): open -> 09:35 / 10:00 (m5snap) by overnight-gap bucket."""
    e5, e30, nt = UNIV["e5"], UNIV["e30"], UNIV["night"]
    ok = (UNIV["adv"] > 2e7) & (UNIV["px"] > 5) & e5.notna() & nt.notna()
    s = pd.DataFrame({"e5": e5[ok].stack(), "e30": e30[ok].stack(), "night": nt[ok].stack()})
    s = s[s.index.get_level_values(0) >= "2024-01-02"]
    s["per"] = np.where(s.index.get_level_values(0) <= "2025-06-30", "P1", "P2")
    edges = [-1, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.1, 10]
    rows = []
    for per, d in s.groupby("per"):
        b = pd.cut(d.night, edges)
        for q, dd in d.groupby(b, observed=True):
            g = dd.index.get_level_values(0)
            r = dict(table="universe_by_gap", feature="night", period=per, bucket=str(q), n=len(dd),
                     mean_feature=dd.night.mean())
            for k, lab in [("e5", "r0935"), ("e30", "r1000")]:
                m_, t_ = mean_t(dd[k].clip(-0.3, 0.3), g)
                r[f"{lab}_bp"], r[f"{lab}_t"] = 1e4 * m_, t_
            rows.append(r)
    return rows


# ------------------------------------------------------------------ tradability
def tradability(pk, C, pred, pred_name):
    """(a) day short of easy-to-borrow picks, (b) holding long picks past the open; rules chosen on P1 quantiles
    of the P1-fit prediction (pred = predicted open -> 09:35 return)."""
    rows = []
    n = len(pk)
    ok = pk.bars_ok.values
    etb = pk.etb.values & ok
    p1 = (pk.per.values == "P1") & ok & np.isfinite(pred)
    night, rday = pk.night.values, pk.r_day.values
    R = {"09:35": pk.r0935.values, "10:00": pk.r1000.values}
    # (a) day short
    base = -rday - 2 * C["auc_t"]
    rows += S85.stats_rows(88, f"[{pred_name}] short etb open->close, all", base, pk, etb,
                           extra={"gross_bp": -rday, "r0935_bp": R["09:35"]})
    for q in [0.3, 0.5]:
        cut = np.quantile(pred[p1 & pk.etb.values], q)
        sel = etb & (pred <= cut)
        rows += S85.stats_rows(88, f"[{pred_name}] short etb open->close, pred fall in P1 bottom {int(q * 100)}%",
                               base, pk, sel, extra={"gross_bp": -rday, "r0935_bp": R["09:35"]})
        rows += S85.stats_rows(88, f"[{pred_name}] short etb open->close, rest (not bottom {int(q * 100)}%)",
                               base, pk, etb & ~(pred <= cut), extra={"gross_bp": -rday, "r0935_bp": R["09:35"]})
        for hm in ["09:35", "10:00"]:
            g = -R[hm]
            rows += S85.stats_rows(88, f"[{pred_name}] short etb open->{hm} cover, bottom {int(q * 100)}%",
                                   g - C["auc_t"] - C[hm], pk, sel, extra={"gross_bp": g})
            rows += S85.stats_rows(88, f"[{pred_name}] short etb open->{hm} cover, bottom {int(q * 100)}% "
                                   "[auction cost on all legs]", g - 2 * C["auc_t"], pk, sel, extra={"gross_bp": g})
    for hm in ["09:35", "10:00"]:
        g = -R[hm]
        rows += S85.stats_rows(88, f"[{pred_name}] short etb open->{hm} cover, all", g - C["auc_t"] - C[hm], pk, etb,
                               extra={"gross_bp": g})
    # (b) longs: sell all at the opening auction vs hold the predicted non-fallers to 09:35 / 10:00
    base = night - C["auc_p"] - C["auc_t"]
    rows += S85.stats_rows(88, f"[{pred_name}] long sell all at open auction", base, pk, ok, extra={"gross_bp": night})
    for q in [0.5, 0.7, 0.9]:
        cut = np.quantile(pred[p1], q)
        hold = ok & (pred > cut)
        for hm in ["09:35", "10:00"]:
            g_h = (1 + night) * (1 + R[hm]) - 1
            for lab, ch in [("", C[hm]), (" [auction cost on all legs]", C["auc_t"])]:
                net = np.where(hold, g_h - C["auc_p"] - ch, base)
                inc = np.where(hold, net - base, np.nan)
                rows += S85.stats_rows(88, f"[{pred_name}] long hold P1 top {int(round((1 - q) * 100))}% to {hm}{lab}",
                                       net, pk, ok, extra={"held_share_pct": hold / 100.0, "held_increment_bp": inc})
                rows += S85.stats_rows(88, f"[{pred_name}] long hold P1 top {int(round((1 - q) * 100))}% to {hm}{lab}, "
                                       "held names only", net - base, pk, hold, extra={"drift_bp": R[hm]})
    return rows


def limit_on_open(pk, C):
    """Q4: sell the longs with a limit-on-open order instead of market-on-open. A sell limit fills at the auction
    price when auction >= limit; a non-fill is sold at market at 09:31 / 09:35 / 10:00 (continuous cost)."""
    rows = []
    ok = pk.bars_ok.values
    night = pk.night.values
    base = night - C["auc_p"] - C["auc_t"]
    # auction vs premarket: first trade of the 09:30 5-minute bar (m5snap) over the last premarket trade
    ovp = pk.open_vs_pm.values
    lims = {"prior close": night, "prior close -0.5%": (1 + night) / 0.995 - 1,
            "prior close -1%": (1 + night) / 0.99 - 1, "prior close -2%": (1 + night) / 0.98 - 1,
            "last premarket": ovp, "last premarket -0.5%": (1 + ovp) / 0.995 - 1,
            "last premarket -1%": (1 + ovp) / 0.99 - 1, "last premarket -2%": (1 + ovp) / 0.98 - 1}
    for lab, rel in lims.items():                 # rel = auction / limit - 1
        has = ok & np.isfinite(rel)
        fill = rel >= -1e-9
        for hm, k in [("09:31", "r0931"), ("09:35", "r0935"), ("10:00", "r1000")]:
            alt = (1 + night) * (1 + pk[k].values) - 1 - C["auc_p"] - C[hm]
            net = np.where(fill, base, alt)
            for per in ["P1", "P2", "all"]:
                m = has & ((pk.per.values == per) if per != "all" else True)
                nf = m & ~fill
                rows.append(dict(table="limit_on_open", variant=f"LOO at {lab}, non-fill sold {hm}", period=per,
                                 n=int(m.sum()), fill_pct=100 * fill[m].mean(),
                                 nonfill_night_bp=1e4 * np.nanmean(night[nf]),
                                 nonfill_drift_bp=1e4 * np.nanmean(pk[k].values[nf]),
                                 nonfill_lost_bp=1e4 * np.nanmean((net - base)[nf]),
                                 lost_per_name_all_bp=1e4 * np.nanmean((net - base)[m]),
                                 lost_t=mean_t((net - base)[m], pk.pdate.values[m])[1]))
    return rows


def checks(pk):
    ok = pk.bars_ok.values
    print("pairs", len(pk), "bars ok", ok.mean().round(4), "premarket features", pk.has_pm.mean().round(3),
          "etb", pk.etb.mean().round(3))
    for k in ["r0931", "r0935", "r1000", "night", "open_vs_pm", "pm_gap"]:
        v = pk[k][ok]
        print(f"{k:12s} mean {1e4 * v.mean():8.1f} bp  median {1e4 * v.median():7.1f}  p0.5 {1e4 * v.quantile(.005):8.0f}"
              f"  p99.5 {1e4 * v.quantile(.995):8.0f}  n {v.notna().sum()}")
    c = pk[["r0935", "e5_univ_today_chk"]][ok].dropna()
    print("bars r0935 vs m5snap e5 same pair: corr", c.corr().iloc[0, 1].round(3), "mean", (1e4 * c.mean()).round(1).tolist())
    c = pk[["night", "pm_gap"]].dropna()
    print("night vs premarket gap corr", c.corr().iloc[0, 1].round(3))
    if "open_px" in pk:
        a = pk.anchor_vs_cross[ok]
        print("cross found", pk.open_px.notna().mean().round(4), "| bar anchor vs official cross bp: mean",
              round(1e4 * a.mean(), 1), "median", round(1e4 * a.median(), 1), "share |diff|>10bp",
              (a.abs() > 1e-3).mean().round(3))
        print("r0935 from official cross mean bp", round(1e4 * pk.r0935_off[ok].mean(), 1),
              "from bar anchor", round(1e4 * pk.r0935[ok].mean(), 1))
        ratio = pk.open_px / pk.o930
        print("official cross vs m5snap 09:30 open: median |diff| bp", round(1e4 * (ratio - 1).abs().median(), 1),
              "share > 20% apart (split rescale)", ((ratio - 1).abs() > 0.2).mean().round(4))


def main():
    pk, C, N = build()
    checks(pk)
    rows = cross_section(pk)
    rows += universe_context()
    preds = {}
    for t_ in ["r0935", "r1000"]:
        r, p = model(pk, t_)
        rows += r
        preds[t_] = p
    # single-feature rule: the overnight gap alone (prediction = minus its within-day rank)
    preds["night"] = -(pk.groupby("pdate").night.rank(pct=True).values - 0.5)
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/study88_open_dynamics.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    sh = ["feature", "period", "bucket", "n", "mean_feature", "r0931_bp", "r0935_bp", "r0935_t", "r1000_bp",
          "r1000_t", "r1000_close_bp", "r_close_bp"]
    print(d[d.table.isin(["bucket", "universe_by_gap"])][sh].round(4).to_string(index=False))
    print(d[d.table == "slope_rank"][["feature", "period", "target", "n", "coef_bp", "t"]].round(2).to_string(index=False))
    print(d[d.table.str.startswith("multi")].dropna(axis=1, how="all").round(3).to_string(index=False))
    tr = []
    for nm, p in preds.items():
        tr += tradability(pk, C, p, nm)
    t = pd.DataFrame(tr)
    t.insert(0, "table", "tradability")
    lo = pd.DataFrame(limit_on_open(pk, C))
    out = pd.concat([t, lo], ignore_index=True)
    out.to_csv(f"{RES}/study88_tradability.csv", index=False)
    sh = ["variant", "period", "names_per_day", "gross_bp", "r0935_bp", "held_share_pct", "held_increment_bp",
          "drift_bp", "per_name_net_bp", "net_bp_day", "sharpe", "tstat"]
    print(t[t.period != "all"][[c for c in sh if c in t]].round(2).to_string(index=False))
    print(lo.round(2).to_string(index=False))
    pk.to_pickle(os.path.join(os.environ.get("TMPDIR", "/tmp"), "s88_pairs.pkl")) if os.environ.get("S88_DUMP") else None


if __name__ == "__main__":
    main()
