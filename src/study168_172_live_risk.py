"""Studies 168, 171 and 172: concentration, earnings season and same-day news as explanations for the paper book's
October losses (virtual book -10.9% compounded over 11 nights from 2026-09-29; 6 of the last 7 nights -1.9% to -5%;
on 2026-10-14 eight of the ten picks were optical / networking / semiconductor hardware names).

172  Concentration. Question: were the live October nights unusually concentrated in tech hardware, do nights with
     many tech hardware picks do worse at official crosses, and do pre-registered concentration rules help?
     (a) live nights (logs/paper/entry_*.json 'ml_top' = the capped picks, logs/paper/virtual.csv per-name returns at
         the official auction prints): tech hardware names per night and their share of the losses; backtest
         distribution of tech hardware names per night by quarter.
     (b) backtest by tech hardware count (0-1, 2-3, 4-5, 6+): nights, mean net, Sharpe, worst night, share of nights
         losing 2% or more, per period.
     (c) rules (pre-registered in the task, all reported): halve the book when 4+ / 6+ picks are tech hardware; cap
         tech hardware at 3 / 5 names (greedy down the blend ranking, industry cap 3 kept, filling from next-ranked
         names); equal weight across sectors for the same 10 picks. Per period and 2026Q3 alone; paired t of the
         daily difference against the live blend (and, for the halving rules, against the live blend scaled to the
         same average gross).
     (d) average pairwise 60-day overnight-return correlation of the 10 picks per night, by quarter, with a market
         reference (50 random names a night from the 1,000 largest by 20-day dollar volume, seed 172).
168  Earnings season. Question: is the blend weaker in reporting season? Weeks (Monday-Friday) labelled by the share
     of S&P 500 names reporting that week (store 'earnings' dates; S&P 500 = the 2026-08-10 Wikipedia constituent
     list, today's members applied to all dates); season = share above the 2024-01..2026-09 median of weekly shares.
     Second definition (calendar): nights 14..55 days after a quarter end (6 weeks starting 2 weeks after). Blend
     net at crosses in season vs off season per period (Welch t), share of picks reporting within 5 trading days
     (earnings date in t..t+5) and reporting that night (date t and not pre-market, or t+1 and not after-hours),
     and net bp per pick for those picks.
171  Same-day news on picks. Question: do picks that had news during the entry day do differently overnight? Benzinga
     articles (data/local/news_scored, tagged with 1-3 symbols, as news_features) timestamped from 09:30 to 15:45 ET
     on the entry day; tone = mean headline sentiment (P(pos) - P(neg), src/score_news.py): positive >= 0.25,
     negative <= -0.25, neutral otherwise. Archive ends 2026-09-26, so nights through 2026-09-25 only. Net bp per
     pick at crosses for no news / news / positive / neutral / negative, 1 vs 2+ articles, per period. Pre-listed
     candidates: drop news picks, drop negative picks, drop positive picks (each refilled from next-ranked names
     without the flag, industry cap 3), halve the weight of news picks, halve the weight of negative picks (weights
     renormalized to a full book); the one with the best 2024-01..2025-06 Sharpe is the chosen one, tested on
     2025-07..2026-09.

Data: base series = study 135's live capped blend at official cross prices (study135_137_expectations.build logic,
repeated here so that the score matrix, cross returns and costs are also available: top 10 by (2 rank(ensemble) +
rank(p_jump - p_drop)) / 3 with at most 3 names per industry; night return at crosses, legs without a usable cross
at the panel price; cost exec_cost_bps(P, "auction") + 2.5 bp per side). Tech hardware as study 90 (TECH_HW
industries, plus 'Electrical Products' and 'Industrial Machinery/Components' inside the Technology sector;
data/store/sectors.parquet, one current snapshot). Periods 2024-01..2025-06 and 2025-07..2026-09 (+ 2024-26).
Names picked only by a variant may lack a cached cross: those legs use the panel price (share reported per variant).

Data checks: the base series reproduces study 135 (capped Sharpe 2.65 / 2.23 / 2.45, mean 38.2 bp); it ends with the
night of 2026-09-25 (last blend prediction), so "2026Q3" is 61 nights to 09-25. Legs at the panel price 0.4-1.1% for
every variant (new names picked by the caps or news filters almost always have a cached cross). The 11 live nights'
picks in virtual.csv match entry_*.json 'ml_top' on every night.

Result 172 (a) live: tech hardware was 34.5% of picks (3.5 names a night; 4.9 over the last 7 nights) and produced
59% of the name-level losses: tech hardware names averaged -219 bp net vs -41 bp for the rest, -75.5 of the -102 bp a
night. Over the last 7 nights the other names also lost (-166 bp vs -240 bp). Backtest tech hardware names a night:
0.6-1.1 in 2024Q1..2025Q4, 1.7 / 2.4 / 3.0 in 2026Q1 / Q2 / Q3 (nights with 6+: 0-3% before 2026, 16% in 2026Q3,
36% live). 21% of 11-night windows in 2026Q3 had a mean count at least as high as live (7% in 2025H2-26).
(b) by count (2024-25H1 / 2025H2-26): 0-1 names 46 / 46 bp (Sharpe 3.1 / 3.1), 2-3 32 / 10, 4-5 -118 (9 nights) /
+60 (36 nights), 6+ -65 (1) / +39 (13); 2026Q3 every bucket near zero (-14..+5 bp). No count bucket is consistently
bad; share of nights losing 2%+ is higher with 4+ names (19-27% vs 11%), but the means are not lower in the second
period. (c) rules, diff vs live bp (paired t), 2024-25H1 / 2025H2-26 / 2026Q3: halve at 4+ +1.5 (1.0) / -4.3 (-1.2)
/ +1.2 (0.1); halve at 6+ +0.1 / -0.8 / -0.4; cap tech hw 3 -0.5 / -0.9 / +5.7 (1.7); cap 5 -0.1 / +0.1 / +3.8
(2.1); sector equal weight +5.8 (1.2) / -7.8 (-1.5) / +8.8 (1.0). The caps rarely bind before 2026 (98-100% of picks
kept). Reference, study 90's lead (halve when any tech hw) at crosses: Sharpe 3.04 / 2.53 vs 2.66 / 2.23, drawdown
-15% / -13% vs -17% / -23%, diff at the same average gross +4.2 (t 1.4) / +3.6 (t 1.2), but 2026Q3 -4.4 bp (t -1.0).
None of the pre-registered rules is a significant improvement in either period.
(d) average pairwise 60-day overnight correlation of the picks rose from 0.21 (2024Q1) to 0.38 (2026Q3), trend +0.014
a quarter (t 2.2, NW t 5.9), while the market reference fell (0.26 to 0.08), so the excess over the market rose from
-0.06 to +0.30 (t 4.2). Live nights 0.38 on average, 0.57-0.64 on 10-07, 10-08, 10-13, 10-14. Pick correlation
terciles show no consistent link to the next night's net (high tercile 36 / 47 bp vs low 40 / 35 bp).
Result 168: share-of-S&P-500 season (49% of nights): 39 vs 40 bp (t -0.1) in 2024-25H1, 53 vs 23 bp (t 1.0) in
2025H2-26; calendar season 49 vs 31 bp (t 0.8) and 38 vs 36 bp (t 0.05). The blend is not weaker in season. Picks
reporting within 5 trading days: 26% / 19% of picks (34% in season, 11% off season); their net 53 / 38 bp vs 35 /
37 bp for the others; off season in 2025H2-26 they lost -80 bp (t -2.1) but gained +81 bp in 2024-25H1. Live
nights: 1 of 110 picks reported within 5 days and the weeks had 1.4% of S&P 500 names reporting (off season), so
earnings season does not explain the October losses.
Result 171 (nights to 2026-09-25): 21% / 18% of picks had news between 09:30 and 15:45. Net per pick: no news 39 /
33 bp, news 41 / 58 bp; positive tone 22 / 90, neutral 41 / 61, negative 88 / -42 bp (t 0.7 / -1.8, 116 / 87
picks). The tone effects flip sign between periods. The candidate chosen on 2024-25H1 (drop positive-tone picks,
Sharpe 2.93 vs 2.66) lost -4.1 bp a night (t -1.6) on 2025H2-26; dropping news picks -4.1 / -6.9 bp. No news rule.

Caveats: sectors.parquet is today's classification for all dates; tech hardware share and correlation trends are
measured on 11 quarters; the halving rules' gross differs (compare the same-gross columns); live nights are 11, with
returns at official auction prints from virtual.csv; earnings dates come from the Nasdaq calendar (many rows have no
time of day) and S&P 500 membership is today's list; news ends 2026-09-26, so the live October nights have no news
test.

Output: results/study172_concentration.csv (table = live_nights / live_names / count_dist / count_buckets / rules /
correlation), results/study168_season.csv (table = weeks / season / picks_reporting), results/study171_news.csv
(table = buckets / rules).

    python src/study168_172_live_risk.py
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402
import store   # noqa: E402
import study135_137_expectations as S135   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(DATA, "local")
PAPER = os.path.join(ROOT, "logs", "paper")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
PERQ = PER + [("2026Q3", "2026-07", "2026-09")]
NEWS_END = "2026-09-25"
CACHE = os.environ.get("STUDY168_CACHE")          # optional pickle of the base series (development only)


# ------------------------------------------------------------------ base series (as study 135 build)
def build_ext():
    if CACHE and os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    import study99_cross_retrain as S99
    from paper_overnight import cap_by_industry
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr, ok
    S = S.loc["2024-01":"2026-09"].astype("float64")
    Wc = pd.DataFrame(0.0, index=S.index, columns=cols)
    for d, row in S.iterrows():
        out = list(cap_by_industry(row.dropna().sort_values(ascending=False, kind="stable"), 10, 3).index)
        if out:
            Wc.loc[d, out] = 1.0 / len(out)
    parts = [S99.crosses()]
    for f in S135.EXTRA:
        fn = os.path.join(LOC, f + ".parquet")
        if os.path.exists(fn):
            c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
            c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
            c["date"] = pd.to_datetime(c.date)
            parts.append(c)
    X = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    ro, rc, err = S99.cross_ratios(P, cols, days, X)
    print("cross data errors (legs at panel price):", err, flush=True)
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=S.index)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex(index=S.index) / rc.fillna(1.0).reindex(index=S.index) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=S.index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=S.index)
    net = bt.run(Wc, Rx, C).net.where(Wc.sum(axis=1) > 0).dropna()
    print("capped (live) blend at crosses:", {lab: round(ann_stats(net.loc[a:b])["sharpe"], 2) for lab, a, b in PER},
          "mean bp", round(1e4 * net.mean(), 1), flush=True)
    # overnight returns (open t / close t-1) for correlations, and 20-day dollar volume rank for the reference
    on = (o / c.shift(1) - 1).astype("float32")
    dv20 = P["dv"][cols].rolling(20, min_periods=10).median()
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    K = dict(cols=cols, days=days, S=S, Wc=Wc, net=net, Rx=Rx.astype("float32"), C=C.astype("float32"),
             miss=miss, on=on, dv20=dv20.astype("float32"), sector=sec.sector.reindex(cols),
             industry=sec.industry.reindex(cols))
    if CACHE:
        pd.to_pickle(K, CACHE)
    return K


def techhw_of(industry, sector):
    ind, sct = pd.Series(industry), pd.Series(sector)
    return (ind.isin(S135.TECH_HW) | (ind.isin(S135.TECH_HW_TECHSECTOR) & (sct == "Technology"))).values


def groups(lab, cols):
    lab = pd.Series(lab).fillna("").astype(str).str.strip()
    lab = lab.where(lab != "", "__" + pd.Series(list(cols), index=lab.index))
    return pd.factorize(lab)[0]


def select_greedy(S, cols, caps, exclude=None, k=10, depth=300):
    """walk down each day's ranking; caps = [(group codes, max)]; exclude = boolean frame of names not allowed"""
    Sv = S.values
    Ex = None if exclude is None else exclude.reindex(index=S.index, columns=cols).fillna(False).values
    W = np.zeros_like(Sv)
    for i in range(len(S)):
        s = Sv[i]
        okj = np.where(np.isfinite(s))[0]
        order = okj[np.argsort(-s[okj], kind="stable")][:depth]
        cnt = [dict() for _ in caps]
        chosen = []
        for j in order:
            if Ex is not None and Ex[i, j]:
                continue
            if any(cnt[q].get(g[j], 0) >= m for q, (g, m) in enumerate(caps)):
                continue
            for q, (g, m) in enumerate(caps):
                cnt[q][g[j]] = cnt[q].get(g[j], 0) + 1
            chosen.append(j)
            if len(chosen) == k:
                break
        if chosen:
            W[i, chosen] = 1.0 / len(chosen)
    return pd.DataFrame(W, index=S.index, columns=cols)


def tstat(d):
    d = pd.Series(d).dropna()
    return d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan


def welch(a, b):
    return S135.welch(a, b)


def book_stats(name, net, base, gross, miss_share, periods=PERQ, extra=None):
    rows = []
    for p, a, b in periods:
        x = net.loc[a:b].dropna()
        if not len(x):
            continue
        s = ann_stats(x)
        bs = base.loc[x.index]
        g = gross.loc[x.index].mean()
        rows.append(dict(table="rules", variant=name, period=p, n=len(x), net_bp=1e4 * x.mean(), sharpe=s["sharpe"],
                         maxdd=s["maxdd"], worst_bp=1e4 * x.min(), share_le_2pct=(x <= -0.02).mean(),
                         avg_gross=g, diff_bp=1e4 * (x - bs).mean(), t_vs_live=tstat(x - bs),
                         diff_bp_same_gross=1e4 * (x - g * bs).mean(), t_vs_live_same_gross=tstat(x - g * bs),
                         legs_at_panel=miss_share.loc[a:b].mean(), **(extra or {})))
    return rows


def run_book(W, K):
    out = bt.run(W, K["Rx"], K["C"])
    net = out.net.where(W.sum(axis=1) > 0).reindex(K["net"].index)
    h = W > 0
    ms = (K["miss"] & h).sum(axis=1) / h.sum(axis=1).replace(0, np.nan)
    return net, ms.reindex(K["net"].index)


def picks_long(W):
    s = W.stack()
    s = s[s > 0].reset_index()
    s.columns = ["date", "ticker", "w"]
    return s


# ------------------------------------------------------------------ 172
def live_nights():
    v = pd.read_csv(os.path.join(PAPER, "virtual.csv"), parse_dates=["day"])
    vd = pd.read_csv(os.path.join(PAPER, "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    ent = {}
    for fn in sorted(glob.glob(os.path.join(PAPER, "entry_2026-*.json"))):
        if "replay" in fn:
            continue
        j = json.load(open(fn))
        ent[pd.Timestamp(j["day"])] = list(j.get("ml_top") or {})
    return v, vd, ent


def study172(K):
    rows = []
    cols, S, Wc, net = K["cols"], K["S"], K["Wc"], K["net"]
    th = techhw_of(K["industry"].values, K["sector"].values)
    thS = pd.Series(th, index=cols)
    # ---- (a) live nights
    v, vd, ent = live_nights()
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    v["techhw"] = techhw_of(v.symbol.map(sec.industry).values, v.symbol.map(sec.sector).values)
    v["net_name"] = v.ret - v.cost
    for d, g in v.groupby("day"):
        picks = ent.get(d, [])
        same = sorted(picks) == sorted(g.symbol)
        n = len(g)
        contrib = g.net_name / n
        loss = contrib[contrib < 0]
        rows.append(dict(table="live_nights", period=str(d.date()), n=n, techhw_names=int(g.techhw.sum()),
                         net_bp=1e4 * vd.net.get(d, np.nan), techhw_contrib_bp=1e4 * contrib[g.techhw.values].sum(),
                         other_contrib_bp=1e4 * contrib[~g.techhw.values].sum(),
                         techhw_share_of_name_losses=(loss[g.techhw.values[contrib.values < 0]].sum() / loss.sum()
                                                      if len(loss) else np.nan),
                         names_down=int((g.ret < 0).sum()), entry_json_matches=same,
                         techhw_list=" ".join(g.symbol[g.techhw.values])))
        for _, r in g.iterrows():
            rows.append(dict(table="live_names", period=str(d.date()), ticker=r.symbol, techhw=bool(r.techhw),
                             industry=sec.industry.get(r.symbol), sector=sec.sector.get(r.symbol),
                             net_bp=1e4 * r.net_name))
    allc = v.net_name / v.groupby("day").symbol.transform("size")
    neg = allc < 0
    tot = dict(table="live_nights", period="all live nights", n=v.day.nunique(),
               techhw_names=v.groupby("day").techhw.sum().mean(), net_bp=1e4 * vd.net.mean(),
               techhw_contrib_bp=1e4 * allc[v.techhw].sum() / v.day.nunique(),
               other_contrib_bp=1e4 * allc[~v.techhw].sum() / v.day.nunique(),
               techhw_share_of_name_losses=allc[neg & v.techhw].sum() / allc[neg].sum(),
               techhw_share_of_picks=v.techhw.mean(),
               techhw_mean_name_bp=1e4 * v.net_name[v.techhw].mean(),
               other_mean_name_bp=1e4 * v.net_name[~v.techhw].mean())
    rows.append(tot)
    last7 = sorted(v.day.unique())[-7:]
    w7 = v[v.day.isin(last7)]
    a7 = w7.net_name / 10
    rows.append(dict(table="live_nights", period="last 7 live nights", n=7, techhw_names=w7.groupby("day").techhw.sum().mean(),
                     net_bp=1e4 * vd.net.loc[last7].mean(), techhw_contrib_bp=1e4 * a7[w7.techhw].sum() / 7,
                     other_contrib_bp=1e4 * a7[~w7.techhw].sum() / 7,
                     techhw_share_of_name_losses=a7[(a7 < 0) & w7.techhw].sum() / a7[a7 < 0].sum(),
                     techhw_share_of_picks=w7.techhw.mean(), techhw_mean_name_bp=1e4 * w7.net_name[w7.techhw].mean(),
                     other_mean_name_bp=1e4 * w7.net_name[~w7.techhw].mean()))
    live_cnt = v.groupby("day").techhw.sum()
    # ---- backtest count distribution by quarter
    cnt = ((Wc > 0) & thS).sum(axis=1).loc[net.index]
    q = cnt.index.to_period("Q").astype(str)
    bins = [("0-1", 0, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6+", 6, 99)]
    groups_ = [(qq, cnt[q == qq]) for qq in sorted(set(q))] + [(p, cnt.loc[a:b]) for p, a, b in PER] + \
        [("live 2026-09-29..10-14", live_cnt)]
    for lab, x in groups_:
        r = dict(table="count_dist", period=lab, n=len(x), mean_techhw=x.mean(), median_techhw=x.median(),
                 max_techhw=x.max())
        for bl, lo, hi in bins:
            r[f"share_{bl}"] = ((x >= lo) & (x <= hi)).mean()
        rows.append(r)
    for lab, ref in [("2026Q3", cnt.loc["2026-07":"2026-09"]), ("2025H2-26", cnt.loc["2025-07":"2026-09"])]:
        rows.append(dict(table="count_dist", period=f"live mean count vs {lab}", n=len(live_cnt),
                         mean_techhw=live_cnt.mean(),
                         pct_nights_ge_live_mean=(ref >= live_cnt.mean()).mean(),
                         pct_11night_means_ge_live=(ref.rolling(len(live_cnt)).mean() >= live_cnt.mean()).mean()))
    # ---- (b) buckets
    for p, a, b in PER + [("2026Q3", "2026-07", "2026-09")]:
        x, cc = net.loc[a:b], cnt.loc[a:b]
        for bl, lo, hi in bins:
            sel = x[(cc >= lo) & (cc <= hi)]
            s = ann_stats(sel) if len(sel) > 1 else {"sharpe": np.nan}
            rows.append(dict(table="count_buckets", period=p, bucket=bl, n=len(sel),
                             net_bp=1e4 * sel.mean() if len(sel) else np.nan, sharpe=s["sharpe"],
                             worst_bp=1e4 * sel.min() if len(sel) else np.nan,
                             share_le_2pct=(sel <= -0.02).mean() if len(sel) else np.nan,
                             hit=(sel > 0).mean() if len(sel) else np.nan, t_vs_rest=welch(sel, x.drop(sel.index))))
    # ---- (c) rules
    one = pd.Series(1.0, index=net.index)
    base_net, base_ms = run_book(Wc, K)
    assert np.allclose(base_net.values, net.values)
    rows += book_stats("live blend (industry cap 3)", net, net, one, base_ms)
    for m in [4, 6, 1]:
        sc = pd.Series(np.where(cnt >= m, 0.5, 1.0), index=net.index)
        lab = f"halve book when {m}+ tech hw" if m > 1 else "ref: halve book when any tech hw (study 90 lead)"
        rows += book_stats(lab, net * sc, net, sc, base_ms,
                           extra=dict(nights_halved=int((cnt >= m).sum())))
    g_ind = groups(K["industry"].values, cols)
    g_th = np.where(th, -1, np.arange(len(cols)))
    for m in [3, 5]:
        W = select_greedy(S, cols, [(g_ind, 3), (g_th, m)])
        nv, ms = run_book(W, K)
        keep = ((W > 0) & (Wc > 0)).sum(axis=1).loc[net.index].mean() / 10
        rows += book_stats(f"cap tech hw at {m} (+ industry 3)", nv, net, one, ms,
                           extra=dict(nights_changed=int((((W > 0) != (Wc > 0)).any(axis=1)).loc[net.index].sum()),
                                      picks_kept=keep))
    g_sec = groups(K["sector"].values, cols)
    Wv = Wc.values
    We = np.zeros_like(Wv)
    for i in range(len(Wv)):
        js = np.where(Wv[i] > 0)[0]
        if not len(js):
            continue
        gs = g_sec[js]
        u, inv, nn = np.unique(gs, return_inverse=True, return_counts=True)
        We[i, js] = 1.0 / len(u) / nn[inv]
    We = pd.DataFrame(We, index=Wc.index, columns=cols)
    nv, ms = run_book(We, K)
    rows += book_stats("equal weight across sectors (same picks)", nv, net, one, ms)
    # ---- (d) correlation
    on, days, dv20 = K["on"], K["days"], K["dv20"]
    onv = on.values
    pos = days.get_indexer(net.index)
    rng = np.random.default_rng(172)

    def avg_corr(i_end, js, n=60, mp=40):
        Xm = onv[i_end - n + 1:i_end + 1][:, js].astype("float64")
        good = np.isfinite(Xm).sum(0) >= mp
        Xm = Xm[:, good]
        if Xm.shape[1] < 2:
            return np.nan
        Cm = pd.DataFrame(Xm).corr(min_periods=mp).values
        iu = np.triu_indices(Cm.shape[0], 1)
        return np.nanmean(Cm[iu])

    pc, mc = {}, {}
    dvv = dv20.values
    for i, d in zip(pos, net.index):
        js = np.where(Wc.loc[d].values > 0)[0]
        pc[d] = avg_corr(i, js)
        r = dvv[i]
        top = np.where(np.isfinite(r))[0]
        top = top[np.argsort(-r[top])][:1000]
        mc[d] = avg_corr(i, rng.choice(top, 50, replace=False))
    # live nights
    ci = {t: j for j, t in enumerate(cols)}
    for d, g in v.groupby("day"):
        if d not in days:
            continue
        js = [ci[t.replace(".", "-")] for t in g.symbol if t.replace(".", "-") in ci]
        i = days.get_loc(d)
        pc[d] = avg_corr(i, np.array(js))
        r = dvv[i]
        top = np.where(np.isfinite(r))[0]
        top = top[np.argsort(-r[top])][:1000]
        mc[d] = avg_corr(i, rng.choice(top, 50, replace=False))
    pc, mc = pd.Series(pc).sort_index(), pd.Series(mc).sort_index()
    live_ix = pd.DatetimeIndex(sorted(set(v.day) & set(pc.index)))
    pcl, mcl = pc.loc[live_ix], mc.loc[live_ix]
    pc, mc = pc.loc[net.index], mc.loc[net.index]
    qq = pc.index.to_period("Q").astype(str)
    for lab in sorted(set(qq)):
        sel = qq == lab
        rows.append(dict(table="correlation", period=lab, n=int(sel.sum()), pick_corr=pc[sel].mean(),
                         market_corr=mc[sel].mean(), excess_corr=(pc[sel] - mc[sel]).mean(),
                         techhw_names=cnt[sel].mean()))
    for p, a, b in PER:
        rows.append(dict(table="correlation", period=p, n=len(pc.loc[a:b]), pick_corr=pc.loc[a:b].mean(),
                         market_corr=mc.loc[a:b].mean(), excess_corr=(pc - mc).loc[a:b].mean(),
                         techhw_names=cnt.loc[a:b].mean()))
    rows.append(dict(table="correlation", period="live 2026-09-29..10-14", n=len(pcl), pick_corr=pcl.mean(),
                     market_corr=mcl.mean(), excess_corr=(pcl - mcl).mean(), techhw_names=live_cnt.mean(),
                     pct_bt_nights_ge_live=(pc >= pcl.mean()).mean(),
                     pct_2026Q3_nights_ge_live=(pc.loc["2026-07":] >= pcl.mean()).mean()))
    bt_q = pc.groupby(pc.index.to_period("Q")).mean()
    slope, t, tnw = S135.tstat_ols(bt_q.values, np.arange(len(bt_q), dtype=float))
    ex_q = (pc - mc).groupby(pc.index.to_period("Q")).mean()
    s2, t2, tnw2 = S135.tstat_ols(ex_q.values, np.arange(len(ex_q), dtype=float))
    rows.append(dict(table="correlation", period="trend 2024Q1..2026Q3", n=len(bt_q), slope_per_q=slope, t=t,
                     t_nw=tnw, excess_slope_per_q=s2, excess_t=t2, excess_t_nw=tnw2))
    x = net
    pcb = pc
    qs = pcb.quantile([1 / 3, 2 / 3]).values
    for lab, sel in [("pick corr low tercile", pcb <= qs[0]), ("mid", (pcb > qs[0]) & (pcb <= qs[1])),
                     ("high tercile", pcb > qs[1])]:
        for p, a, b in PER:
            xs = x.loc[a:b][sel.loc[a:b]]
            rows.append(dict(table="correlation", period=p, bucket=lab, n=len(xs), net_bp=1e4 * xs.mean(),
                             sharpe=ann_stats(xs)["sharpe"], share_le_2pct=(xs <= -0.02).mean(),
                             t_vs_rest=welch(xs, x.loc[a:b][~sel.loc[a:b]])))
    print("live nights pick corr", pcl.round(2).to_dict(), flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 168
def sp500_list():
    t1 = open(os.path.join(LOC, "events", "sp500_wiki_1.html")).read()
    tabs = pd.read_html(io.StringIO(t1))
    return set(tabs[0]["Symbol"].astype(str).str.replace(".", "-", regex=False))


def study168(K):
    rows = []
    net, Wc, days = K["net"], K["Wc"], K["days"]
    e = store.read("earnings", start="2023-10")
    e["ticker"] = e.symbol.astype(str).str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    e = e.drop_duplicates(["ticker", "date"])
    sp = sp500_list()
    es = e[e.ticker.isin(sp)]
    wk = es.date.dt.to_period("W-FRI")
    wshare = es.groupby(wk).ticker.nunique() / len(sp)
    allw = pd.period_range("2023-12-25", "2026-10-02", freq="W-FRI")
    wshare = wshare.reindex(allw).fillna(0.0)
    med = wshare[(wshare.index.start_time >= "2024-01-01") & (wshare.index.start_time <= "2026-09-30")].median()
    season_w = wshare > med
    for w, val in wshare.items():
        rows.append(dict(table="weeks", period=str((w.start_time + pd.Timedelta(days=2)).date()), share_sp500_reporting=val,
                         season=bool(season_w[w]), median=med))
    nwk = net.index.to_period("W-FRI")
    lab_share = pd.Series(season_w.reindex(nwk).values, index=net.index).fillna(False).astype(bool)
    qend = net.index.to_period("Q").start_time - pd.Timedelta(days=1)
    lab_cal = pd.Series(((net.index - qend).days >= 14) & ((net.index - qend).days < 56), index=net.index)
    print("season share of nights: share rule", round(lab_share.mean(), 3), "calendar", round(lab_cal.mean(), 3),
          "agreement", round((lab_share == lab_cal).mean(), 3), flush=True)
    for defn, lab in [("share of S&P 500 reporting above median", lab_share), ("calendar 14..55 days after QE", lab_cal)]:
        for p, a, b in PERQ:
            x, lb = net.loc[a:b], lab.loc[a:b]
            for lev, sel in [("season", lb), ("off season", ~lb)]:
                xs = x[sel]
                if len(xs) < 2:
                    continue
                rows.append(dict(table="season", definition=defn, period=p, bucket=lev, n=len(xs),
                                 net_bp=1e4 * xs.mean(), sharpe=ann_stats(xs)["sharpe"], hit=(xs > 0).mean(),
                                 share_le_2pct=(xs <= -0.02).mean(), worst_bp=1e4 * xs.min(),
                                 t_vs_other=welch(xs, x[~sel])))
    # picks reporting within 5 trading days / that night
    pk = picks_long(Wc)
    pk = pk[pk.date.isin(net.index)]
    di = days.get_indexer(pk.date)
    ed = e[["ticker", "date", "time"]].rename(columns={"date": "edate"})
    ed = ed[ed.edate.isin(days)]
    ed["ei"] = days.get_indexer(ed.edate)
    m = pk.assign(pi=di).merge(ed, on="ticker", how="left")
    m["gap"] = m.ei - m.pi
    m["w5"] = (m.gap >= 0) & (m.gap <= 5)
    m["tonight"] = ((m.gap == 0) & (m.time != "time-pre-market")) | ((m.gap == 1) & (m.time != "time-after-hours"))
    f = m.groupby(["date", "ticker"])[["w5", "tonight"]].any().reset_index()
    Rx, C = K["Rx"], K["C"]
    ri = Rx.index.get_indexer(f.date)
    cj = Rx.columns.get_indexer(f.ticker)
    f["net_bp"] = 1e4 * (Rx.values[ri, cj] - 2 * C.values[ri, cj] / 1e4)
    f["season"] = lab_share.reindex(f.date).values
    for p, a, b in PERQ:
        x = f[(f.date >= a) & (f.date <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
        for slab, xs in [("all nights", x), ("season", x[x.season]), ("off season", x[~x.season])]:
            rows.append(dict(table="picks_reporting", period=p, bucket=slab, n=len(xs),
                             share_report_5d=xs.w5.mean(), share_report_tonight=xs.tonight.mean(),
                             net_bp_report_5d=xs.net_bp[xs.w5].mean(), net_bp_not_5d=xs.net_bp[~xs.w5].mean(),
                             net_bp_tonight=xs.net_bp[xs.tonight].mean(), net_bp_not_tonight=xs.net_bp[~xs.tonight].mean(),
                             t_5d_vs_not=welch(xs.net_bp[xs.w5], xs.net_bp[~xs.w5]),
                             n_report_5d=int(xs.w5.sum()), n_tonight=int(xs.tonight.sum())))
    # live October: reporting within 5 days
    v, vd, ent = live_nights()
    vv = v[["day", "symbol"]].copy()
    vv["ticker"] = vv.symbol.str.replace(".", "-", regex=False)
    allbd = pd.bdate_range("2026-09-01", "2026-12-31")
    vv["pi"] = allbd.get_indexer(vv.day)
    e2 = e[e.date >= "2026-09-01"][["ticker", "date"]].copy()
    e2["ei"] = allbd.get_indexer(e2.date)
    mm = vv.merge(e2, on="ticker", how="left")
    mm["w5"] = (mm.ei - mm.pi).between(0, 5)
    lw = mm.groupby(["day", "ticker"]).w5.any()
    rows.append(dict(table="picks_reporting", period="live 2026-09-29..10-14", bucket="all nights", n=len(lw),
                     share_report_5d=lw.mean(), n_report_5d=int(lw.sum()),
                     share_sp500_reporting_week_mean=wshare.reindex(pd.DatetimeIndex(vd.index).to_period("W-FRI")).mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 171
def study171(K):
    rows = []
    cols, S, Wc, net = K["cols"], K["S"], K["Wc"], K["net"]
    src = os.path.join(LOC, "news_scored")
    files = [f for f in sorted(os.listdir(src)) if f.endswith(".parquet") and f[:7] >= "2023-12"]
    d = pd.concat([pd.read_parquet(os.path.join(src, f), columns=["id", "ts", "symbols", "sent"]) for f in files],
                  ignore_index=True).drop_duplicates("id")
    d["sym"] = d.symbols.str.split("|")
    d = d[d.sym.str.len() <= 3].explode("sym").reset_index(drop=True)
    d["sym"] = d.sym.str.replace(".", "-", regex=False)
    d = d[d.sym.isin(set(cols))].reset_index(drop=True)
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    tod = ts - ts.dt.normalize()
    d = d[(tod >= pd.Timedelta(hours=9, minutes=30)) & (tod <= pd.Timedelta(hours=15, minutes=45)).values]
    d["day"] = ts.loc[d.index].dt.normalize().values
    g = d.groupby(["day", "sym"])
    nn = g.size().unstack().reindex(index=S.index, columns=cols).fillna(0)
    sm = g.sent.mean().unstack().reindex(index=S.index, columns=cols)
    print("news rows in window", len(d), "last", d.day.max(), flush=True)
    idx = net.index[net.index <= NEWS_END]
    pos_ = sm >= 0.25
    neg_ = sm <= -0.25
    has = nn > 0
    pk = picks_long(Wc)
    pk = pk[pk.date.isin(idx)]
    ri = S.index.get_indexer(pk.date)
    cj = pd.Index(cols).get_indexer(pk.ticker)
    pk["n_news"] = nn.values[ri, cj]
    pk["sent"] = sm.values[ri, cj]
    pk["net_bp"] = 1e4 * (K["Rx"].values[ri, cj] - 2 * K["C"].values[ri, cj] / 1e4)
    pk["tone"] = np.select([pk.n_news == 0, pk.sent >= 0.25, pk.sent <= -0.25], ["none", "positive", "negative"],
                           "neutral")
    # reference: all names in the top 50 of the blend ranking that night (news base rate among candidates)
    rk = S.loc[idx].rank(axis=1, ascending=False)
    top50 = (rk <= 50)
    base_rate = (has.loc[idx] & top50).sum(axis=1) / top50.sum(axis=1)
    for p, a, b in [(p, a, min(b, "2026-09")) for p, a, b in PER]:
        x = pk[(pk.date >= a) & (pk.date <= min(pd.Timestamp(b) + pd.offsets.MonthEnd(0), pd.Timestamp(NEWS_END)))]
        buckets = [("all picks", x.n_news >= 0), ("no news", x.n_news == 0), ("news", x.n_news > 0),
                   ("1 article", x.n_news == 1), ("2+ articles", x.n_news >= 2),
                   ("positive tone", x.tone == "positive"), ("neutral tone", x.tone == "neutral"),
                   ("negative tone", x.tone == "negative")]
        for lab, sel in buckets:
            xs = x.net_bp[sel]
            rows.append(dict(table="buckets", period=p, bucket=lab, n=len(xs), share_of_picks=sel.mean(),
                             net_bp=xs.mean(), se_bp=xs.std() / np.sqrt(len(xs)) if len(xs) > 1 else np.nan,
                             t_vs_rest=welch(xs, x.net_bp[~sel]) if lab != "all picks" else np.nan,
                             top50_news_rate=base_rate.loc[a:b].mean()))
    # rules
    sub = S.loc[idx]
    base = net.loc[idx]
    one = pd.Series(1.0, index=idx)
    g_ind = groups(K["industry"].values, cols)
    Kx = dict(K, net=base)
    _, base_ms = run_book(Wc.loc[idx], Kx)
    per2 = [(p, a, b) for p, a, b in PER] + [("2026Q3 to 09-25", "2026-07", "2026-09")]
    rows += [dict(r, table="rules") for r in book_stats("live blend (industry cap 3)", base, base, one, base_ms, per2)]
    cands = {}
    for lab, ex in [("drop news picks (refill)", has), ("drop negative-tone picks (refill)", neg_),
                    ("drop positive-tone picks (refill)", pos_)]:
        W = select_greedy(sub, cols, [(g_ind, 3)], exclude=ex.loc[idx])
        cands[lab] = W
    for lab, fl in [("halve weight of news picks", has), ("halve weight of negative-tone picks", neg_)]:
        W = Wc.loc[idx] * np.where(fl.loc[idx].values, 0.5, 1.0)
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        cands[lab] = W
    sel_sh = {}
    out = []
    for lab, W in cands.items():
        nv, ms = run_book(W, Kx)
        sel_sh[lab] = ann_stats(nv.loc["2024-01":"2025-06"].dropna())["sharpe"]
        out += book_stats(lab, nv, base, one, ms, per2)
    best = max(sel_sh, key=sel_sh.get)
    for r in out:
        r["chosen_on_first_period"] = r["variant"] == best
        r["first_period_sharpe"] = sel_sh[r["variant"]]
    rows += out
    print("171 chosen:", best, {k: round(v, 2) for k, v in sel_sh.items()}, flush=True)
    return pd.DataFrame(rows)


def show(df, cols_):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 300)
    print(df[[c for c in cols_ if c in df.columns]].round(3).to_string(index=False), flush=True)


def main():
    K = build_ext()
    r172 = study172(K)
    r172.to_csv(os.path.join(RES, "study172_concentration.csv"), index=False)
    for t, c in [("live_nights", ["period", "n", "techhw_names", "net_bp", "techhw_contrib_bp", "other_contrib_bp",
                                  "techhw_share_of_name_losses", "techhw_share_of_picks", "techhw_mean_name_bp",
                                  "other_mean_name_bp", "entry_json_matches", "techhw_list"]),
                 ("count_dist", ["period", "n", "mean_techhw", "share_0-1", "share_2-3", "share_4-5", "share_6+",
                                 "max_techhw", "pct_nights_ge_live_mean", "pct_11night_means_ge_live"]),
                 ("count_buckets", ["period", "bucket", "n", "net_bp", "sharpe", "worst_bp", "share_le_2pct", "hit",
                                    "t_vs_rest"]),
                 ("rules", ["variant", "period", "n", "net_bp", "sharpe", "maxdd", "worst_bp", "share_le_2pct",
                            "avg_gross", "diff_bp", "t_vs_live", "diff_bp_same_gross", "t_vs_live_same_gross",
                            "legs_at_panel", "picks_kept"]),
                 ("correlation", ["period", "bucket", "n", "pick_corr", "market_corr", "excess_corr", "techhw_names",
                                  "pct_bt_nights_ge_live", "pct_2026Q3_nights_ge_live", "slope_per_q", "t", "t_nw", "excess_slope_per_q", "excess_t", "net_bp", "sharpe",
                                  "share_le_2pct", "t_vs_rest"])]:
        print(f"\n== 172 {t}")
        show(r172[r172.table == t], c)
    r168 = study168(K)
    r168.to_csv(os.path.join(RES, "study168_season.csv"), index=False)
    print("\n== 168 season")
    show(r168[r168.table == "season"], ["definition", "period", "bucket", "n", "net_bp", "sharpe", "hit",
                                        "share_le_2pct", "worst_bp", "t_vs_other"])
    show(r168[r168.table == "picks_reporting"], ["period", "bucket", "n", "share_report_5d", "share_report_tonight",
                                                 "net_bp_report_5d", "net_bp_not_5d", "net_bp_tonight",
                                                 "net_bp_not_tonight", "t_5d_vs_not",
                                                 "share_sp500_reporting_week_mean"])
    r171 = study171(K)
    r171.to_csv(os.path.join(RES, "study171_news.csv"), index=False)
    print("\n== 171")
    show(r171[r171.table == "buckets"], ["period", "bucket", "n", "share_of_picks", "net_bp", "se_bp", "t_vs_rest",
                                         "top50_news_rate"])
    show(r171[r171.table == "rules"], ["variant", "period", "n", "net_bp", "sharpe", "maxdd", "diff_bp", "t_vs_live",
                                       "legs_at_panel", "chosen_on_first_period"])


if __name__ == "__main__":
    main()
