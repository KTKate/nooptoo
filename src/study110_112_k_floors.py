"""Studies 110 and 112: pick count, rank position, and price / liquidity floors of the overnight blend at official
cross prices.

Question:
  110  Study 73 found that variable pick counts did not beat a fixed 10 at panel prices, and study 88 found that the
       panel's first-trade-vs-cross bias and the post-open fall are largest in the top 3 ranks. At official cross
       prices, which number of picks k is best, and does the top of the list lose its edge?
  112  Study 88 found the bias largest in cheap, thinly traded, gapped-up names. Do higher price or dollar-volume
       floors (filling the 10 slots from the next-ranked names that pass) improve the blend at cross prices?

Data: blend scores exactly as study 69 lines 22-46 (top of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3; results/
study23_pred.parquet, study33_pred.parquet). Panel prices core.load_panel. Official crosses: caches of studies 94 / 88
/ 99 (data/local/auctions94.parquet, auctions88.parquet, auctions99.parquet) plus crosses fetched here for the new
picks into data/local/auctions110.parquet (study 94's rule: first 'O' / '6' print of the listing exchange, second pass
the largest 'O' / '6' print of any listing exchange; Alpaca SIP; resumable). Traded close for price floors from
core.traded_close (Alpaca raw daily bars; Yahoo's raw close is adjusted for later splits); where it is missing,
Yahoo raw close.
Live floors (src/paper_overnight.py): the liquid list is 20-day median dollar volume > $5M and last raw close > $5
(line 242), and ml_scores keeps raw 15:44 price > $5 and adv20 > $5M (line 151; adv20 = ml_features.build: 20-day
median of dollar volume, min 10 days); the training / prediction universe of ml_features is the same (rawc > 5,
adv20 > 5e6, line 97). In the backtest that universe uses Yahoo raw close, which is adjusted for later splits, so a
few historical picks traded below $5 (later reverse splits); the $5 floor here, on the traded close, removes them.

Design: entry at the closing cross of day t, exit at the opening cross of t+1. Returns at crosses as study 94: panel
night return x (open cross / panel open on t+1) / (close cross / panel close on t), legs without a usable cross (or a
data error, study 94's checks) fall back to the panel price (rate reported). Cost exec_cost_bps(P, "auction") + 2.5 bp
per side. Periods by entry date 2024-01..2025-06 (choice), 2025-07..2026-09 (report), 2024-01..2026-09.
  110  top k equal weight, k = 5, 8, 10, 12, 15, 20; rank buckets alone (1-3, 4-6, 7-10, 11-15, 16-20) equal weight
       within the bucket. Net bp per name (mean over nights of the bucket's equal-weight net return) at panel and at
       cross prices, bias = panel minus cross net return per pick (bp). k chosen by cross Sharpe on 2024-01..2025-06.
  112  floors on the day-t traded close: $3 / $5 / $10 / $20; on 20-day median dollar volume through t-1 (as the live
       list): $5M / $10M / $20M / $50M. Names failing the floor are removed and the 10 slots filled from the next
       ranked names. Sharpe, mean, max drawdown, average bias per pick, share of picks replaced, paired t of daily
       net return differences vs the baseline top 10, per period at cross prices; choice on 2024-01..2025-06.

Output: results/study110_pick_count.csv (rows: kind = topk / bucket, set, period, pricing, Sharpe, net bp, net bp per
name, max drawdown, bias per pick, paired diff vs top 10, fallback share), results/study112_floors.csv (rows: floor,
period, pricing, Sharpe, net bp, max drawdown, bias per pick, share replaced, paired diff and t vs baseline,
fallback share).

    python src/study110_112_k_floors.py fetch    # official crosses for the new picks (resumable)
    python src/study110_112_k_floors.py          # studies
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # another job shares the account's limit
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC110 = os.path.join(LOC, "auctions110.parquet")
AUC99 = os.path.join(LOC, "auctions99.parquet")
AUC94 = os.path.join(LOC, "auctions94.parquet")
OUT110 = os.path.join(RES, "study110_pick_count.csv")
OUT112 = os.path.join(RES, "study112_floors.csv")
START, END = "2024-01", "2026-09"
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
KS = [5, 8, 10, 12, 15, 20]
BUCKETS = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 20)]
PX_FLOORS = [3, 5, 10, 20]
ADV_FLOORS = [5e6, 10e6, 20e6, 50e6]


# ------------------------------------------------------------------ picks
def build():
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
    S = S.astype("float64")
    tc = traded_close(P)[cols]
    px = tc.where(tc.notna(), P["rawc"][cols]).reindex_like(S)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex_like(S)
    sets = {}                              # name -> 0/1 weights (equal weight), index = entry day
    rk = S.rank(axis=1, ascending=False, method="first")
    for k in KS:
        sets[("topk", f"top {k}")] = bt.select_topk(S, S.notna(), k)
    for a, b in BUCKETS:
        m = (rk >= a) & (rk <= b)
        sets[("bucket", f"ranks {a}-{b}")] = m.astype("float32").div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    for f in PX_FLOORS:
        Sf = S.where(px > f)
        sets[("floor", f"price > ${f}")] = bt.select_topk(Sf, Sf.notna(), 10)
    for f in ADV_FLOORS:
        Sf = S.where(adv > f)
        sets[("floor", f"ADV > ${f / 1e6:.0f}M")] = bt.select_topk(Sf, Sf.notna(), 10)
    return dict(P=P, cols=cols, days=days, S=S, rk=rk, px=px, adv=adv, sets=sets)


def need_list(K):
    nxt = pd.Series(K["days"][1:], index=K["days"][:-1])
    rows = []
    for Wx in K["sets"].values():
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    n = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()
    n["date"] = pd.to_datetime(n.date)
    return n.reset_index(drop=True)


# ------------------------------------------------------------------ crosses
def crosses():
    """All cached crosses (studies 94 / 88 / 99 / 110), panel-style tickers, one row per key."""
    import study99_cross_retrain as S99
    x = S99.crosses()
    if os.path.exists(AUC110):
        c = pd.read_parquet(AUC110)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
        c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
        c["date"] = pd.to_datetime(c.date)
        x = pd.concat([x, c], ignore_index=True).drop_duplicates(["ticker", "date", "kind"])
    return x


def tried():
    """Keys already looked up (found or not) by studies 94 / 99 / 110 (88's misses are re-tried by 94's seed)."""
    parts = [crosses()[["ticker", "date", "kind"]]]
    for fn in [AUC94, AUC99, AUC110]:
        if os.path.exists(fn):
            t = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t["ticker"] = t.ticker.astype(str).str.replace(".", "-", regex=False)
            parts.append(t.assign(date=pd.to_datetime(t.date)))
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def fetch():
    import study94_cross_rescore as S94
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
    S94.A.RL = A.RL
    K = build()
    need = need_list(K)
    del K
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    m = need.merge(tried(), how="left", indicator=True)
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
            for x in r:
                x["rule"] = "venue"
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
    old = pd.read_parquet(AUC110) if os.path.exists(AUC110) else pd.DataFrame()
    new = []
    t0 = time.time()
    for i in range(0, len(jobs), 12):
        with ThreadPoolExecutor(2) as exe:
            for r in exe.map(one, jobs[i:i + 12]):
                new += r
        if not new:
            continue
        df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            df[k] = df[k].astype("string")
        df.to_parquet(AUC110, index=False)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ study
def per_stats(net, a0, b0):
    x = net.loc[a0:b0].dropna()
    st = ann_stats(x)
    return dict(n=st["n"], sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"])


def main():
    import study99_cross_retrain as S99
    K = build()
    P, cols, days, S, sets = K["P"], K["cols"], K["days"], K["S"], K["sets"]
    ro, rc, err = S99.cross_ratios(P, cols, days, crosses())
    print("cross data errors (legs at the panel price):", err, flush=True)
    ro = ro.reindex(index=days, columns=cols)
    rc = rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex_like(S)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    gap = (Rp - Rx).where(~miss)          # panel minus cross, per leg with a usable cross
    nets, extra = {}, {}
    for key, Wx in sets.items():
        Wx = Wx.loc[START:END]
        for pr, R in [("panel", Rp), ("cross", Rx)]:
            nets[(key, pr)] = bt.run(Wx, R.loc[START:END], C.loc[START:END]).net
        h = Wx > 0
        hn = h.sum(axis=1).replace(0, np.nan)
        extra[key] = dict(bias=(gap.loc[START:END].where(h)).sum(axis=1) / hn,          # mean over the night's picks
                          fb=(miss.loc[START:END] & h).sum(axis=1), legs=h.sum(axis=1), nn=hn)
    base = ("topk", "top 10")
    W10 = sets[base].loc[START:END] > 0
    rows = []
    for key, Wx in sets.items():
        kind, name = key
        h = Wx.loc[START:END] > 0
        repl = (h & ~W10).sum(axis=1)
        for lab, a0, b0 in PER:
            e = extra[key]
            legs = e["legs"].loc[a0:b0].sum()
            bias = 1e4 * (gap.loc[a0:b0].where(h.loc[a0:b0])).stack().mean()
            for pr in ["panel", "cross"]:
                net = nets[(key, pr)]
                d = (net - nets[(base, pr)]).loc[a0:b0].dropna()
                r = dict(kind=kind, set=name, period=lab, pricing=pr, **per_stats(net, a0, b0),
                         names_per_night=e["nn"].loc[a0:b0].mean(), bias_bp_per_pick=bias,
                         diff_vs_top10_bp=1e4 * d.mean(),
                         t_vs_top10=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                         share_replaced=repl.loc[a0:b0].sum() / max(legs, 1) if kind == "floor" else np.nan,
                         fallback_share=e["fb"].loc[a0:b0].sum() / max(legs, 1))
                rows.append(r)
    df = pd.DataFrame(rows)
    # price / ADV profile of the picks per bucket (for the report)
    prof = []
    for key in [k for k in sets if k[0] == "bucket"]:
        h = sets[key].loc[START:END] > 0
        for lab, a0, b0 in PER:
            hh = h.loc[a0:b0].values
            pv = K["px"].loc[a0:b0].values[hh]
            av = K["adv"].loc[a0:b0].values[hh]
            pv, av = pv[~np.isnan(pv)], av[~np.isnan(av)]
            prof.append(dict(set=key[1], period=lab, median_px=np.median(pv), median_adv_m=np.median(av) / 1e6,
                             share_px_lt5=(pv < 5).mean(), share_px_lt10=(pv < 10).mean(),
                             share_adv_lt20m=(av < 20e6).mean()))
    prof = pd.DataFrame(prof)
    d110 = df[df.kind.isin(["topk", "bucket"])].drop(columns="share_replaced").merge(
        prof, on=["set", "period"], how="left")
    d112 = df[(df.kind == "floor") | ((df.kind == "topk") & (df.set == "top 10"))].copy()
    d112.loc[d112.set == "top 10", "set"] = "baseline: top 10 (no extra floor)"
    d110.to_csv(OUT110, index=False)
    d112.to_csv(OUT112, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    show = ["set", "period", "pricing", "sharpe", "net_bp", "maxdd", "names_per_night", "bias_bp_per_pick",
            "diff_vs_top10_bp", "t_vs_top10", "fallback_share"]
    print("== 110\n", d110[show + ["median_px", "median_adv_m", "share_px_lt5", "share_px_lt10", "share_adv_lt20m"]].round(3).to_string(index=False))
    print("== 112\n", d112[show + ["share_replaced"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
