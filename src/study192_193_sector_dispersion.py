"""Studies 192 and 193: ranking the overnight blend's picks within sectors, and cross-sectional dispersion as a regime
signal for the blend.

192  Rank within sector. Question: the live blend takes the overall top 10 by score S with at most 3 names per
     industry. Studies 92/172 found that caps by sector or on tech hardware did not help in both periods. Does a
     selection that spreads the picks across sectors by construction do better: (a) the best name of each of the 10
     sectors whose best name scores highest, (b) the best 2 names of each of the 5 sectors whose best name scores
     highest, (c) the top 10 by sector-demeaned score (S minus the sector's mean S that day) with the live
     3-per-industry cap? Rules fixed before looking at results; any choice is made on 2024-01..2025-06.
193  Dispersion regime. Question: is the blend's net return on night t higher when the cross-sectional dispersion of
     returns was high on day t: dispersion of the close-to-open returns (night t-1 -> t) and of the day session (open
     to close of t) across the liquid universe, and the blend's own pick-score spread (mean S of the 10 capped picks
     minus the universe median S)? Descriptive by tercile, plus one simple scale-up / scale-down rule chosen on
     2024-01..2025-06 and checked on 2025-07..2026-09.

Data: core.load_panel (split- and dividend-adjusted Yahoo daily o/c, rawc, dv). Base = the live capped blend at
official cross prices, study135_137_expectations.build() (nets 'capped'). Blend score S as study 69 lines 22-46
((2 rank(ensemble) + rank(p_jump - p_drop)) / 3; results/study23_pred.parquet, study33_pred.parquet). Sectors and
industries: data/store/sectors.parquet (one current snapshot applied to all dates); names with an empty sector form
one group '(none)' for 192 (a)-(c); industry cap groups as paper_overnight.cap_by_industry (study 166 capped_topk;
names without an industry are their own group). Tech hardware as studies 90/92 (TECH_HW industries plus 'Electrical
Products' and 'Industrial Machinery/Components' inside the Technology sector). Official crosses: every cached
data/local/auctions*.parquet (study 166 cached_crosses) plus data/local/auctions192.parquet fetched here for legs of
the new picks with study 94's rule (listing-venue 'O' / '6' print in widening windows; second pass: the largest
'O' / '6' print of any listing exchange; Alpaca SIP, 60 requests a minute). Night return at crosses as study 94:
panel night return x (open cross / panel open on t+1) / (close cross / panel close on t); legs without a usable cross
at the panel price. Costs core.exec_cost_bps(P, "auction") + 2.5 bp per side. Liquid universe for 193: raw close
> $5 and 20-day mean dollar volume > $5M on day t-1 (known before the night and day measured), with panel o and c
present.

Design 192: variants (a) sector1x10, (b) sector2x5, (c) demeaned_cap3, each equal weight over its picks, buy at the
closing cross of t, sell at the opening cross of t+1, and the base (live capped top 10) priced on the same cross
table. Per period (2024-01..2025-06, 2025-07..2026-09, 2026Q3 = 2026-07..09, 2024-01..2026-09): Sharpe, mean net bp,
max drawdown, mean tech hardware names per night, mean pairwise correlation of the picks' overnight (close-to-open
panel) returns over the 60 nights ending on the entry day (pairs with >= 40 common nights), and the paired t of the
daily net difference vs base. Pick overlap with the base.
Design 193: per day t: on_std / on_iqr = standard deviation / interquartile range of o(t)/c(t-1)-1 across the liquid
universe; day_std / day_iqr = the same for c(t)/o(t)-1 (returns clipped at +-50% for the std); score_spread = mean S
of the capped picks minus the median S of all scored names. Tercile cut points from 2024-01..2025-06, applied to all
periods (so a tercile means the same level). Per variable, tercile and period: nights, mean net bp, Sharpe, hit
rate, Welch t of the bucket against the other nights; Spearman correlation of the variable with the next-night net.
Rule candidates (10): per variable, weight 1.5 / 1.0 / 0.5 on the high / mid / low tercile ('up_high') or the
reverse ('up_low'); net return scales with the weight (costs included). The candidate with the best 2024-01..2025-06
Sharpe is chosen and checked on 2025-07..2026-09; mean weight reported (it is not 1 outside the first period).

Output: results/study192_sector_rank.csv (table = rescore / overlap / fallback / check), results/study193_dispersion.csv
(table = tercile / spearman / rule / choice / cuts).

    python src/study192_193_sector_dispersion.py fetch   # official crosses for new picks (resumable)
    python src/study192_193_sector_dispersion.py         # both studies
"""
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # other jobs share the account's limit
from core import load_panel, stock_cols, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC192 = os.path.join(LOC, "auctions192.parquet")
OUT192 = os.path.join(RES, "study192_sector_rank.csv")
OUT193 = os.path.join(RES, "study193_dispersion.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2026Q3", "2026-07", "2026-09"),
       ("2024-26", "2024-01", "2026-09")]
VARIANTS = ["base", "sector1x10", "sector2x5", "demeaned_cap3"]


# ------------------------------------------------------------------ scores and picks
def blend_score(P, cols):
    """Study 69 lines 22-46."""
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


def sector_map(cols):
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    s = sec.sector.reindex(cols)
    return s.where(s.notna() & (s != ""), "(none)")


def sector_topk(S, sec, n_sec, per_sec):
    """Rank names within sector by S; take the best per_sec names of the n_sec sectors with the highest best score."""
    W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
    codes, uniq = pd.factorize(sec.reindex(S.columns))
    vals = S.values
    for i in range(len(S)):
        v = vals[i]
        idx = np.where(~np.isnan(v))[0]
        order = idx[np.argsort(-v[idx], kind="stable")]
        best, picks = [], {}
        for j in order:
            g = codes[j]
            if g not in picks:
                picks[g] = []
                best.append(g)          # sectors in order of their best name's score
            if len(picks[g]) < per_sec:
                picks[g].append(j)
        out = [j for g in best[:n_sec] for j in picks[g]]
        if out:
            W.iloc[i, out] = 1.0 / len(out)
    return W


def all_picks(P, cols, Wbase):
    import study166_167_blend_parts as S166
    S = blend_score(P, cols)
    grp, th = S166.industry_map(cols)
    sec = sector_map(cols)
    Wc = S166.capped_topk(S, grp)
    same = ((Wc > 0) == (Wbase.reindex_like(Wc).fillna(0) > 0)).all(axis=1).mean()
    print("capped picks vs build() Wc: share of identical days", round(float(same), 4), flush=True)
    W = {"base": Wc, "sector1x10": sector_topk(S, sec, 10, 1), "sector2x5": sector_topk(S, sec, 5, 2)}
    smean = S.T.groupby(sec.reindex(S.columns).values).transform("mean").T
    W["demeaned_cap3"] = S166.capped_topk(S - smean, grp)
    return S, W, grp, th, sec, float(same)


# ------------------------------------------------------------------ crosses
def fetch():
    import study94_cross_rescore as S94
    import study166_167_blend_parts as S166
    import study135_137_expectations as S135
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    S94.A.RL = A.RL
    K = S135.build()
    P, cols = K["P"], K["cols"]
    days = P["c"].index
    _, W, *_ = all_picks(P, cols, K["Wc"])
    need = pd.concat([S166.legs(Wx, days) for Wx in W.values()], ignore_index=True)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()           # includes auctions192.parquet
    tried = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(tried, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            r = S94._day_kind(job[0], job[1], job[2], ven)
            for x in r:
                x["rule"] = "venue"
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
            if miss:               # second pass: largest print of any listing exchange
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    if x["ticker"] in lg:
                        x["rule"] = "largest"
                        if not np.isnan(lg[x["ticker"]]["px"]):
                            x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x"]})
            return r
        except RuntimeError as e:  # left uncached; a rerun retries
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC192) if os.path.exists(AUC192) else pd.DataFrame()
    new, t0 = [], time.time()
    for i in range(0, len(jobs), 12):
        with ThreadPoolExecutor(2) as exe:
            for r in exe.map(one, jobs[i:i + 12]):
                new += r
        if not new:
            continue
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC192 + ".tmp", index=False)
        os.replace(AUC192 + ".tmp", AUC192)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ helpers
def paired_t(a, b):
    d = (a - b).dropna()
    if len(d) < 5 or d.std() == 0:
        return np.nan
    return d.mean() / d.std() * np.sqrt(len(d))


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def st(x):
    x = x.dropna()
    s = ann_stats(x)
    return dict(n=s["n"], net_bp=1e4 * x.mean() if len(x) else np.nan, sharpe=s["sharpe"], maxdd=s["maxdd"],
                hit=(x > 0).mean() if len(x) else np.nan)


def pick_corr(W, N, win=60, minp=40):
    """Mean pairwise correlation of the picks' overnight returns over the win nights ending on the entry day."""
    pos = {d: i for i, d in enumerate(N.index)}
    cix = {c: j for j, c in enumerate(N.columns)}
    A = N.values
    out = pd.Series(np.nan, index=W.index)
    H = W.values > 0
    for i, d in enumerate(W.index):
        k = pos.get(d)
        names = [cix[c] for c in W.columns[H[i]]]
        if k is None or k < win or len(names) < 2:
            continue
        X = pd.DataFrame(A[k - win + 1:k + 1][:, names])
        C = X.corr(min_periods=minp).values
        iu = np.triu_indices(len(names), 1)
        out.iloc[i] = np.nanmean(C[iu])
    return out


# ------------------------------------------------------------------ 192
def study192(K, S, W, th, Rx, miss, C, same):
    P = K["P"]
    cols = K["cols"]
    o, c = P["o"][cols], P["c"][cols]
    N = (o / c.shift(1) - 1).clip(-0.5, 0.5)           # night ending on day t, known at the open of t
    nets, rows = {}, []
    ref = K["nets"]["capped"]
    for k, Wx in W.items():
        nets[k] = bt.run(Wx, Rx, C).net.where(Wx.sum(axis=1) > 0).dropna()
    b = nets["base"]
    rows.append(dict(table="check", variant="base vs build() capped", n=len(b.index.intersection(ref.index)),
                     corr=float(b.corr(ref)), mean_abs_diff_bp=float(1e4 * (b - ref).abs().mean()),
                     identical_pick_days=same,
                     **{f"sharpe_build_{p}": ann_stats(ref.loc[a0:b0])["sharpe"] for p, a0, b0 in PER}))
    Hb = W["base"] > 0
    for k, Wx in W.items():
        H = Wx > 0
        nth = (H & th.reindex(cols).values).sum(axis=1).where(H.sum(axis=1) > 0)
        nn = H.sum(axis=1).where(H.sum(axis=1) > 0)
        pc = pick_corr(Wx, N)
        nsec = None
        for p, a0, b0 in PER:
            x = nets[k].loc[a0:b0]
            rows.append(dict(table="rescore", variant=k, period=p, **st(x),
                             mean_diff_bp=1e4 * (x - b.loc[a0:b0]).mean(), paired_t=paired_t(x, b.loc[a0:b0]),
                             techhw_per_night=nth.loc[a0:b0].mean(), names_per_night=nn.loc[a0:b0].mean(),
                             pick_corr60=pc.loc[a0:b0].mean()))
            hh = H.loc[a0:b0]
            rows.append(dict(table="overlap", variant=k, period=p,
                             share_in_base=float((hh & Hb.loc[a0:b0]).values.sum() / hh.values.sum())))
            rows.append(dict(table="fallback", variant=k, period=p, legs=int(hh.values.sum()),
                             share_panel=float((miss.loc[a0:b0] & hh).values.sum() / hh.values.sum())))
        del nsec
    return pd.DataFrame(rows), nets


# ------------------------------------------------------------------ 193
def dispersion(P, cols, S, Wc):
    o, c, rc = P["o"][cols], P["c"][cols], P["rawc"][cols]
    adv = P["dv"][cols].rolling(20, min_periods=10).mean()
    liq = (rc.shift(1) > 5) & (adv.shift(1) > 5e6) & o.notna() & c.notna() & c.shift(1).notna()
    ix = S.index
    on = (o / c.shift(1) - 1).where(liq).loc[ix]
    dy = (c / o - 1).where(liq).loc[ix]

    def iqr(x):
        q = x.quantile([0.25, 0.75], axis=1)
        return q.loc[0.75] - q.loc[0.25]
    D = pd.DataFrame({"on_std": on.clip(-0.5, 0.5).std(axis=1), "on_iqr": iqr(on),
                      "day_std": dy.clip(-0.5, 0.5).std(axis=1), "day_iqr": iqr(dy), "n_liquid": liq.loc[ix].sum(1)})
    H = Wc > 0
    D["score_spread"] = S.where(H).mean(axis=1) - S.median(axis=1)
    return D


def study193(D, net):
    rows = []
    D = D.reindex(net.index)
    a1, b1 = PER[0][1], PER[0][2]
    VARS = ["on_std", "on_iqr", "day_std", "day_iqr", "score_spread"]
    lab = {}
    for v in VARS:
        q = D[v].loc[a1:b1].quantile([1 / 3, 2 / 3]).values
        lab[v] = pd.cut(D[v], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"]).astype(str)
        rows.append(dict(table="cuts", variable=v, cut_low_mid=q[0], cut_mid_high=q[1],
                         **{f"median_{p}": D[v].loc[a0:b0].median() for p, a0, b0 in PER}))
        for p, a0, b0 in PER:
            x, lb = net.loc[a0:b0], lab[v].loc[a0:b0]
            for lev in ["low", "mid", "high"]:
                sel = x[lb == lev]
                rows.append(dict(table="tercile", variable=v, bucket=lev, period=p, **st(sel),
                                 t_vs_rest=welch(sel, x[lb != lev]), var_mean=D[v].loc[sel.index].mean()))
            rows.append(dict(table="spearman", variable=v, period=p, n=len(x),
                             rho=D[v].loc[x.index].corr(x, method="spearman"),
                             t_high_vs_low=welch(x[lb == "high"], x[lb == "low"])))
    cands = {}
    for v in VARS:
        for rule, wmap in [("up_high", {"low": 0.5, "mid": 1.0, "high": 1.5}),
                           ("up_low", {"low": 1.5, "mid": 1.0, "high": 0.5})]:
            w = lab[v].map(wmap).fillna(1.0)
            r = net * w
            cands[(v, rule)] = (r, w)
            for p, a0, b0 in PER:
                x = r.loc[a0:b0]
                rows.append(dict(table="rule", variable=v, bucket=rule, period=p, **st(x),
                                 mean_weight=w.loc[a0:b0].mean(), base_sharpe=ann_stats(net.loc[a0:b0])["sharpe"],
                                 base_net_bp=1e4 * net.loc[a0:b0].mean(),
                                 paired_t=paired_t(x, net.loc[a0:b0])))
    best = max(cands, key=lambda k: ann_stats(cands[k][0].loc[a1:b1])["sharpe"])
    r, w = cands[best]
    for p, a0, b0 in PER:
        rows.append(dict(table="choice", variable=best[0], bucket=best[1], period=p, **st(r.loc[a0:b0]),
                         mean_weight=w.loc[a0:b0].mean(), base_sharpe=ann_stats(net.loc[a0:b0])["sharpe"],
                         base_net_bp=1e4 * net.loc[a0:b0].mean(), paired_t=paired_t(r.loc[a0:b0], net.loc[a0:b0])))
    return pd.DataFrame(rows), best


def main():
    import study135_137_expectations as S135
    import study166_167_blend_parts as S166
    K = S135.build()
    P, cols = K["P"], K["cols"]
    days = P["c"].index
    S, W, grp, th, sec, same = all_picks(P, cols, K["Wc"])
    index = W["base"].index
    Rx, miss, C = S166.returns(P, cols, days, index)
    t192, nets = study192(K, S, W, th, Rx, miss, C, same)
    del Rx, C, miss
    gc.collect()
    t192.to_csv(OUT192, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    r = t192[t192.table == "rescore"]
    print(r[["variant", "period", "n", "sharpe", "net_bp", "maxdd", "mean_diff_bp", "paired_t", "techhw_per_night",
             "names_per_night", "pick_corr60"]].round(3).to_string(index=False))
    print(t192[t192.table.isin(["check", "overlap", "fallback"])].dropna(axis=1, how="all").round(4).to_string())
    D = dispersion(P, cols, S, W["base"])
    t193, best = study193(D, K["nets"]["capped"])
    t193.to_csv(OUT193, index=False)
    print("liquid names per day: median", D.n_liquid.median(), "min", D.n_liquid.min())
    for tb in ["cuts", "spearman", "tercile", "rule", "choice"]:
        x = t193[t193.table == tb].dropna(axis=1, how="all")
        print(tb)
        print(x.round(3).to_string(index=False))
    print("chosen rule", best)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
