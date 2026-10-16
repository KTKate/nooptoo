"""Studies 166 and 167: which inputs of the live overnight blend earn its return, and how to weight them.

166  Blend parts. Question: the live blend (study 33) is S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3, where the
     ensemble is the mean within-day rank of five LightGBM members. Which inputs earn the return? Rebuild the ensemble
     with each member dropped in turn, and score the blend without the jump score (ensemble only) and without the
     ensemble (jump score only). Is any variant better or worse than the live blend, per period?
167  Weights. Question: blend = (w rank(ensemble) + rank(p_jump - p_drop)) / (w + 1) for w = 0.5, 1, 2 (live), 3, 4.
     Study 118 found half the book from the ensemble top 5 and half from the jump top 5 gave Sharpe 2.57 vs 2.38 (not
     significant), a hint that the jump score may deserve more weight. Choose w on 2024-01..2025-06 and report it on
     2025-07..2026-09.
Both: average number of tech hardware picks per night in 2024 and in 2026 (2026-01..09) for each variant, since the
live picks on 2026-10-14 were 8 of 10 tech hardware.

Data: member and ensemble predictions results/study23_pred.parquet (pooled, pooled_beh, per_beh, pooled_com, per_com,
ensemble; out of sample, quarterly refits); p_jump, p_drop results/study33_pred.parquet. src/ensemble.py defines the
ensemble as the plain mean of the members' within-day percentile ranks (checked: identical to the stored 'ensemble'
column), so a dropped-member ensemble is the mean rank of the remaining four. core.load_panel (split- and
dividend-adjusted o/c, rawc). Official crosses: every cached data/local/auctions*.parquet (studies 88-159; 88 filtered
to listing-venue opens as study 94's seed) plus data/local/auctions166.parquet fetched here for the legs of the new
variants' picks with study 94's rule (listing-venue 'O' / '6' print in widening windows, second pass: the largest
'O' / '6' print of any listing exchange). Night return at crosses as study 94 / study135_137_expectations.build():
panel night return x (open cross / panel open on t+1) / (close cross / panel close on t), legs without a usable cross
at the panel price. Costs core.exec_cost_bps(P, "auction") + 2.5 bp per side. Industries data/store/sectors.parquet
(one snapshot). Tech hardware as studies 90/92 (TECH_HW industries, plus 'Electrical Products' and 'Industrial
Machinery/Components' inside the Technology sector).

Design: every variant uses the same universe each day (names with all five members and p_jump present, as the live
blend), ranks are within-day percentile ranks, top 10 by score with at most 3 per industry (the rule of
paper_overnight.cap_by_industry, reimplemented here with the same ordering and checked against it on sampled days),
equal weight, entry at the closing cross of day t, exit at the opening cross of t+1, 2024-01..2026-09. Per period
(2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09): nights, Sharpe, mean net bp, max drawdown (compounded), and a
paired t of the daily net difference against the live blend on common nights; overlap = mean share of the live picks
also picked. Study 167 picks w with the highest 2024-01..2025-06 Sharpe and reports its 2025-07..2026-09 numbers.
The live variant is also compared with study135_137_expectations.build() (capped Sharpe 2.65 / 2.23 / 2.45).

Output: results/study166_167_blend_parts.csv (one row per variant x period; columns study, variant, period, n, sharpe,
net_bp, maxdd, t_vs_live, overlap_live, techhw_2024, techhw_2026, techhw_last).

    python src/study166_167_blend_parts.py fetch    # crosses for the new picks (resumable, ~60 Alpaca requests a minute)
    python src/study166_167_blend_parts.py
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402
import study94_cross_rescore as S94   # noqa: E402  (sets A.RL at import; reset below)

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
LOC = os.path.join(DATA, "local")
AUC166 = os.path.join(LOC, "auctions166.parquet")
OUT = os.path.join(RES, "study166_167_blend_parts.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
MEMBERS = ["pooled", "pooled_beh", "per_beh", "pooled_com", "per_com"]
WEIGHTS = [0.5, 1, 2, 3, 4]
TECH_HW = ["Semiconductors", "Electronic Components", "Computer peripheral equipment", "Computer Manufacturing",
           "Computer Communications Equipment", "Retail: Computer Software & Peripheral Equipment"]
TECH_HW_TECHSECTOR = ["Electrical Products", "Industrial Machinery/Components"]


# ------------------------------------------------------------------ scores and picks
def scores(cols, last_day):
    """Variant name -> (study, wide score frame) on 2024-01..2026-09, all on the live blend's universe."""
    m = pd.read_parquet(f"{RES}/study23_pred.parquet")
    m = m[m.index.get_level_values(0) < last_day]
    R = {k: m[k].unstack().reindex(columns=cols).rank(axis=1, pct=True) for k in MEMBERS}
    ens = m["ensemble"].unstack().reindex(columns=cols)
    del m
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    jd = (pred.p_jump - pred.p_drop).unstack().reindex(index=ens.index, columns=cols)
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    for k in MEMBERS:
        ok &= R[k].notna()
    del pj
    sl = slice("2024-01", "2026-09")
    re_ = ens.where(ok).rank(axis=1, pct=True).loc[sl]
    rj = jd.where(ok).rank(axis=1, pct=True).loc[sl]
    out = {"live": ("both", (2 * re_ + rj) / 3)}
    for k in MEMBERS:
        e = sum(R[j] for j in MEMBERS if j != k) / 4
        rk = e.where(ok).rank(axis=1, pct=True).loc[sl]
        out[f"drop_{k}"] = ("166", (2 * rk + rj) / 3)
    out["ensemble_only"] = ("166", re_)
    out["jump_only"] = ("166", rj)
    for w in WEIGHTS:
        if w != 2:
            out[f"w{w:g}"] = ("167", (w * re_ + rj) / (w + 1))
    return out


def industry_map(cols):
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    ind = sec.industry.reindex(cols)
    grp = pd.Series([g if isinstance(g, str) and g else t for t, g in zip(cols, ind)], index=cols)
    sct = sec.sector.reindex(cols)
    th = ind.isin(TECH_HW) | (ind.isin(TECH_HW_TECHSECTOR) & (sct == "Technology"))
    return grp, th.fillna(False)


def capped_topk(S, grp, n=10, cap=3):
    """paper_overnight.cap_by_industry applied each day: descending stable sort, at most cap names per industry."""
    W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
    g = grp.reindex(S.columns).values
    vals = S.values
    for i in range(len(S)):
        v = vals[i]
        idx = np.where(~np.isnan(v))[0]
        order = idx[np.argsort(-v[idx], kind="stable")]
        cnt, out = {}, []
        for j in order:
            if cnt.get(g[j], 0) >= cap:
                continue
            cnt[g[j]] = cnt.get(g[j], 0) + 1
            out.append(j)
            if len(out) == n:
                break
        if out:
            W.iloc[i, out] = 1.0 / len(out)
    return W


def check_cap(S, W, k=15):
    from paper_overnight import cap_by_industry
    rng = np.random.default_rng(166)
    bad = 0
    for d in rng.choice(S.index, k, replace=False):
        a = set(cap_by_industry(S.loc[d].dropna().sort_values(ascending=False, kind="stable"), 10, 3).index)
        bad += a != set(W.columns[W.loc[d] > 0])
    print("cap check vs paper_overnight.cap_by_industry: mismatching days", bad, "of", k, flush=True)


def all_picks():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    grp, th = industry_map(cols)
    Wd = {}
    for name, (study, S) in scores(cols, days[-1]).items():
        Wd[name] = (study, capped_topk(S, grp))
        if name == "live":
            check_cap(S, Wd[name][1])
        del S
        print("picks", name, flush=True)
    return P, cols, days, th, Wd


# ------------------------------------------------------------------ crosses
def legs(W, days):
    nxt = pd.Series(days[1:], index=days[:-1])
    s = W.stack()
    s = s[s > 0].reset_index()
    s.columns = ["t", "ticker", "w"]
    return pd.concat([pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                      pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]).dropna()


def cached_crosses():
    """All cached crosses (ticker Alpaca style, date, kind, px, rule); rows with a price preferred."""
    parts = []
    for f in sorted(glob.glob(os.path.join(LOC, "auctions*.parquet"))):
        b = os.path.basename(f)
        if b == "auctions.parquet":
            continue
        d = pd.read_parquet(f)
        if b == "auctions88.parquet":
            ven = S94.listing_venue()
            lv = d.ticker.map(ven)
            d = d[d.open_px.isna() | lv.isna() | (d.open_x == lv)]
            d = pd.DataFrame({"ticker": d.ticker, "date": d.date, "kind": "open", "px": d.open_px, "rule": "venue"})
        if "rule" not in d:
            d["rule"] = "venue"
        d = d[["ticker", "date", "kind", "px", "rule"]].copy()
        parts.append(d)
    x = pd.concat(parts, ignore_index=True)
    x["ticker"] = x.ticker.astype(str).str.replace("-", ".", regex=False)
    x["date"] = pd.to_datetime(x.date)
    x["rule"] = x.rule.astype(str).replace({"<NA>": "venue", "nan": "venue", "None": "venue"})
    x["has"] = x.px.notna()
    x = x.sort_values("has", ascending=False).drop_duplicates(["ticker", "date", "kind"])
    return x.drop(columns="has").reset_index(drop=True)


def fetch():
    P, cols, days, th, Wd = all_picks()
    need = pd.concat([legs(W, days) for _, W in Wd.values()], ignore_index=True)
    del P, Wd
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC166):
        have = pd.read_parquet(AUC166)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC166
    rows, t0 = [], time.time()
    from concurrent.futures import ThreadPoolExecutor

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
    if os.path.exists(AUC166):
        S94.refetch_missing()


# ------------------------------------------------------------------ scoring
def returns(P, cols, days, index):
    import study99_cross_retrain as S99
    X = cached_crosses().dropna(subset=["px"])[["ticker", "date", "kind", "px"]]
    X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
    ro, rc, err = S99.cross_ratios(P, cols, days, X)
    print("cross data errors (legs at panel price):", err, flush=True)
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=index)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex(index=index) / rc.fillna(1.0).reindex(index=index) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=index)
    return Rx, miss, C


def paired_t(a, b):
    d = (a - b).dropna()
    if len(d) < 5 or d.std() == 0:
        return np.nan
    return d.mean() / d.std() * np.sqrt(len(d))


def main():
    P, cols, days, th, Wd = all_picks()
    index = Wd["live"][1].index
    Rx, miss, C = returns(P, cols, days, index)
    del P
    nets, rows = {}, []
    for name, (study, W) in Wd.items():
        nets[name] = bt.run(W, Rx, C).net.where(W.sum(axis=1) > 0).dropna()
        h = W > 0
        print(name, "fallback share", round(float((miss & h).sum().sum() / h.sum().sum()), 4), flush=True)
    import study135_137_expectations as S135
    ref = S135.build()["nets"]["capped"]
    print("live vs study 135 build(): Sharpe here",
          {p: round(ann_stats(nets["live"].loc[a:b])["sharpe"], 2) for p, a, b in PER},
          "build()", {p: round(ann_stats(ref.loc[a:b])["sharpe"], 2) for p, a, b in PER},
          "corr", round(nets["live"].corr(ref), 4), flush=True)
    Hl = Wd["live"][1] > 0
    for name, (study, W) in Wd.items():
        H = W > 0
        nth = (H & th.reindex(cols).values).sum(axis=1)
        ov = ((H & Hl).sum(axis=1) / Hl.sum(axis=1).replace(0, np.nan))
        last = W.index[-1]
        for p, a, b in PER:
            x = nets[name].loc[a:b]
            s = ann_stats(x)
            rows.append(dict(study=study, variant=name, period=p, n=s["n"], sharpe=s["sharpe"], net_bp=1e4 * x.mean(),
                             maxdd=s["maxdd"], t_vs_live=paired_t(x, nets["live"].loc[a:b]),
                             overlap_live=ov.loc[a:b].mean(), techhw_2024=nth.loc["2024"].mean(),
                             techhw_2026=nth.loc["2026-01":"2026-09"].mean(), techhw_last=nth.loc[last]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    w = df[df.variant.isin(["live"] + [f"w{x:g}" for x in WEIGHTS if x != 2])].copy()
    first = w[w.period == "2024-25H1"].sort_values("sharpe", ascending=False)
    best = first.variant.iloc[0]
    print("\n167: best w on 2024-25H1:", best, "(Sharpe", round(first.sharpe.iloc[0], 2), ")")
    print(w[(w.variant == best) & (w.period == "2025H2-26")].round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
