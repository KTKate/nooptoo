"""Studies 118 and 127: book construction variants of the overnight blend at official cross prices.

Question:
  118  Study 108 found that holding five single-seed top-10 books at a fifth each beat averaging their predictions
       into one score. Does the same hold for the live blend's two parts? Is it better to hold separate books from the
       ensemble score and the jump score (p_jump - p_drop) than to pick the top 10 of the blended score?
  127  On nights when the signal is weak, does it pay to trade fewer names (5 to 10, rest in cash), or to shrink the
       whole book? Study 73 tested variable counts at panel prices and found a fixed 10 best; this rechecks at
       official cross prices.

Data: blend exactly as study 69 lines 22-46 (S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3, percentile ranks
within names that have both scores; results/study23_pred.parquet, study33_pred.parquet). Panel prices core.load_panel.
Official crosses: caches of studies 94 / 88 / 99 / 108 / 110 / 113 (data/local/auctions*.parquet) plus crosses fetched
here for new picks into data/local/auctions118.parquet (study 94's rule: first 'O' / '6' print of the listing
exchange, second pass the largest 'O' / '6' print of any listing exchange; Alpaca SIP; resumable).

Design: entry at the closing cross of day t, exit at the opening cross of t+1. Returns at crosses as study 94: panel
night return x (open cross / panel open on t+1) / (close cross / panel close on t); legs without a usable cross (or a
data error, study 94's checks) use the panel price (rate reported). Cost exec_cost_bps(P, "auction") + 2.5 bp per side
on the weight traded. Cash earns 0. Periods by entry date 2024-01..2025-06 (choice), 2025-07..2026-09 (holdout),
2024-01..2026-09.
  118  (a) baseline: blended score top 10, 10% each; (b1) 50% in the ensemble's top 5 + 50% in the jump score's top 5
       (10% per pick, a name in both lists gets 20%); (b2) the ensemble's top 10 and the jump score's top 10 at 5% per
       pick (overlap doubled); (c) 2/3 in the ensemble top 10 + 1/3 in the jump top 10 (6.67% / 3.33% per pick,
       matching the 2:1 score weights); (d) blended top 15 and top 20 at equal weight, for reference. "Ensemble score"
       and "jump score" are the two percentile ranks inside S.
  127  Pre-registered before looking at any returns (thresholds are quantiles of the score distribution only):
       threshold rule: hold the names whose blended score S is at or above a threshold, at least 5 and at most 10 (the
       5 best if fewer pass, the 10 best if more pass), 10% each, rest in cash. Four thresholds = the 10th / 25th /
       50th / 75th percentile of the pooled 2024-01..2025-06 distribution of the baseline top-10 picks' S.
       scale rules: the whole top-10 book times m_t = 0.5 + 0.5 x pct_t, where pct_t is the percentile of tonight's
       10th pick value within the previous 250 nights (min 60; m = 1 before that): (s1) the 10th pick's blended S,
       (s2) the 10th pick's raw ensemble prediction (the level, which carries time-series information the rank lacks).
  Per variant and period: Sharpe, mean net bp a night, max drawdown, names held, gross invested, turnover (half the
  sum of |weight change| vs the previous night, i.e. the share of the book that changes names), paired difference
  and t of nightly net returns vs the baseline, share of legs at the panel price.

Data checks: the baseline reproduces study 94 at crosses (Sharpe 2.48 / 2.27 / 2.38, 37.5 bp). 11,901 new (ticker,
day, kind) keys fetched, 99.8% found (409 by the largest-print pass). 120 opening and 120 closing legs whose Yahoo raw
close could not be put on the traded scale and 3 opening / 2 closing crosses outside 0.8..1.25 use the panel price;
weight at the panel price 0.3-1.2% per variant and period. Thresholds (S of the baseline picks, 2024-01..2025-06):
q10 0.9665, q25 0.9836, q50 0.9905, q75 0.9946.

Result (Sharpe at crosses, 2024-01..2025-06 / 2025-07..2026-09 / 2024-26; diff = nightly mean vs baseline, t paired):
  118  baseline 2.48 / 2.27 / 2.38 (37.1 / 37.9 bp); b1 2.68 / 2.44 / 2.57 (47.5 / 44.2 bp, diff +10.3 t 0.89 /
       +6.3 t 0.44, max drawdown -15% / -19% vs -17% / -22%, 9.5-9.7 names); b2 2.54 / 2.16 (-2.1 / -6.3 bp);
       c 2.52 / 2.11 (-4.7 / -8.6 bp); top 15 2.33 / 1.95; top 20 2.36 / 1.92; ensemble top 10 alone 2.10 / 1.72; jump
       top 10 alone 2.23 / 1.97 (43.0 / 38.4 bp). Chosen on the first period: b1; holdout gain not significant. Holding
       the two top-10 books (b2, c) is worse than the blended top 10 in both periods, so study 108's book averaging
       result does not carry over; b1's edge looks like concentration (5 names per score, compare study 110's top 8).
       Verdict: leads only.
  127  t10 2.56 / 2.23 (+0.6 t 0.40 / -1.1 t -0.67, 9.2 / 9.5 names); t25 2.50 / 2.29; t50 2.49 / 2.61 (-7.8 / -2.6
       bp, 6.3 / 7.0 names); t75 2.33 / 2.32; s1 2.42 / 2.53 (-5.8 / -4.7 bp); s2 2.43 / 2.28. Chosen on the first
       period: t10, which loses to the baseline in the holdout. The names the rules drop still earn +12 to +31 bp net
       a night (t25-t75, both periods), so cash is the worse use. Verdict: reject (agrees with study 73).

Output: results/study118_book_avg.csv, results/study127_threshold.csv (rows: variant, period, pricing, stats; 127 adds
share of nights cut, net bp of the dropped and kept legs).

    python src/study118_127_book_variants.py fetch    # official crosses for the new picks (resumable)
    python src/study118_127_book_variants.py          # studies
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")       # other jobs share the account's limit
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC118 = os.path.join(LOC, "auctions118.parquet")
EXTRA = [os.path.join(LOC, f"auctions{k}.parquet") for k in ["108", "110", "113"]]
OUT118 = os.path.join(RES, "study118_book_avg.csv")
OUT127 = os.path.join(RES, "study127_threshold.csv")
START, END = "2024-01", "2026-09"
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
TH_Q = [0.10, 0.25, 0.50, 0.75]
TRAIL, TRAIL_MIN = 250, 60


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
    Er = ens.where(ok).rank(axis=1, pct=True)
    Jr = (pj - pdr).where(ok).rank(axis=1, pct=True)
    S = ((2 * Er + Jr) / 3).astype("float64")
    del pj, pdr
    ens = ens.where(ok)
    sets118, sets127 = {}, {}
    W10 = bt.select_topk(S, S.notna(), 10)
    sets118["a: blended top 10 (baseline)"] = W10
    E5, J5 = bt.select_topk(Er, Er.notna(), 5), bt.select_topk(Jr, Jr.notna(), 5)
    E10, J10 = bt.select_topk(Er, Er.notna(), 10), bt.select_topk(Jr, Jr.notna(), 10)
    sets118["b1: 1/2 ensemble top 5 + 1/2 jump top 5"] = 0.5 * E5 + 0.5 * J5
    sets118["b2: 1/2 ensemble top 10 + 1/2 jump top 10"] = 0.5 * E10 + 0.5 * J10
    sets118["c: 2/3 ensemble top 10 + 1/3 jump top 10"] = (2 * E10 + J10) / 3
    sets118["d: blended top 15"] = bt.select_topk(S, S.notna(), 15)
    sets118["d: blended top 20"] = bt.select_topk(S, S.notna(), 20)
    sets118["ref: ensemble top 10 alone"] = E10
    sets118["ref: jump top 10 alone"] = J10
    # 127: thresholds from the score distribution of the baseline picks, 2024-01..2025-06 (no returns used)
    rk = S.rank(axis=1, ascending=False, method="first")
    pickS = S.where(rk <= 10).loc["2024-01":"2025-06"].values.ravel()
    th = {q: float(np.nanquantile(pickS, q)) for q in TH_Q}
    sets127["a: blended top 10 (baseline)"] = W10
    for q, t in th.items():
        keep = (rk <= 5) | ((rk <= 10) & (S >= t))
        sets127[f"t{int(q * 100)}: S >= {t:.4f} (q{int(q * 100)} of picks), 5-10 names, rest cash"] = \
            keep.astype("float64") / 10
    s10 = S.where(rk == 10).max(axis=1)
    e10 = ens.where(rk == 10).max(axis=1)
    for name, v in [("s1: scale book by 10th pick's S vs trailing 250", s10),
                    ("s2: scale book by 10th pick's raw ensemble vs trailing 250", e10)]:
        pct = pd.Series(np.nan, index=v.index)
        vals = v.values
        for i in range(len(vals)):
            h = vals[max(0, i - TRAIL):i]
            h = h[~np.isnan(h)]
            if len(h) >= TRAIL_MIN and not np.isnan(vals[i]):
                pct.iloc[i] = (h < vals[i]).mean()
        m = (0.5 + 0.5 * pct).fillna(1.0)
        sets127[name] = W10.mul(m, axis=0)
    return dict(P=P, cols=cols, days=days, S=S, sets118=sets118, sets127=sets127, th=th)


def need_list(K):
    nxt = pd.Series(K["days"][1:], index=K["days"][:-1])
    rows = []
    for Wx in list(K["sets118"].values()) + list(K["sets127"].values()):
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    n = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()
    n["date"] = pd.to_datetime(n.date)
    return n.reset_index(drop=True)


# ------------------------------------------------------------------ crosses
def _read(fn, found_only=True):
    c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]]
    c["date"] = pd.to_datetime(c.date)
    c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
    return c.dropna(subset=["px"]) if found_only else c


def crosses():
    """All cached crosses (studies 94 / 88 / 99 / 108 / 110 / 113 / 118), panel-style tickers, one row per key."""
    import study99_cross_retrain as S99
    parts = [S99.crosses()] + [_read(fn) for fn in EXTRA + [AUC118] if os.path.exists(fn)]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"]).reset_index(drop=True)


def tried():
    import study99_cross_retrain as S99
    parts = [crosses()[["ticker", "date", "kind"]]]
    for fn in [S99.AUC94, S99.AUC99] + EXTRA + [AUC118]:
        if os.path.exists(fn):
            parts.append(_read(fn, found_only=False)[["ticker", "date", "kind"]])
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
    old = pd.read_parquet(AUC118) if os.path.exists(AUC118) else pd.DataFrame()
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
        df.to_parquet(AUC118, index=False)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ study
def evaluate(sets, base_key, Rp, Rx, C, miss):
    nets, rows = {}, []
    for key, Wx in sets.items():
        Wx = Wx.loc[START:END]
        for pr, R in [("panel", Rp), ("cross", Rx)]:
            nets[(key, pr)] = bt.run(Wx, R.loc[START:END], C.loc[START:END]).net
    for key, Wx in sets.items():
        Wx = Wx.loc[START:END]
        h = Wx > 0
        names = h.sum(axis=1)
        gross = Wx.sum(axis=1)
        turn = 0.5 * Wx.diff().abs().sum(axis=1)
        fbw = (Wx.where(miss.loc[START:END].fillna(True), 0)).sum(axis=1)
        for lab, a0, b0 in PER:
            for pr in ["panel", "cross"]:
                net = nets[(key, pr)].loc[a0:b0].dropna()
                st = ann_stats(net)
                d = (nets[(key, pr)] - nets[(base_key, pr)]).loc[a0:b0].dropna()
                rows.append(dict(variant=key, period=lab, pricing=pr, n=st["n"], sharpe=st["sharpe"],
                                 net_bp=1e4 * net.mean(), ann=st["ann_ret"], vol=st["ann_vol"], maxdd=st["maxdd"],
                                 names_per_night=names.loc[a0:b0].mean(), min_names=names.loc[a0:b0].min(),
                                 gross_invested=gross.loc[a0:b0].mean(), turnover=turn.loc[a0:b0].mean(),
                                 diff_vs_base_bp=1e4 * d.mean(),
                                 t_vs_base=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                                 fallback_weight_share=fbw.loc[a0:b0].sum() / max(gross.loc[a0:b0].sum(), 1e-9)))
    return pd.DataFrame(rows)


def main():
    import study99_cross_retrain as S99
    K = build()
    P, cols, days, S = K["P"], K["cols"], K["days"], K["S"]
    print("127 thresholds (quantiles of baseline picks' S, 2024-01..2025-06):", K["th"], flush=True)
    ro, rc, err = S99.cross_ratios(P, cols, days, crosses())
    print("cross data errors (legs at the panel price):", err, flush=True)
    ro = ro.reindex(index=days, columns=cols)
    rc = rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex_like(S)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    base = "a: blended top 10 (baseline)"
    d118 = evaluate(K["sets118"], base, Rp, Rx, C, miss)
    d127 = evaluate(K["sets127"], base, Rp, Rx, C, miss)
    # 127: names held by the threshold rules split by night strength: baseline net on nights the rule cut names
    extra = []
    Wb = K["sets127"][base].loc[START:END]
    for key, Wx in K["sets127"].items():
        if not key.startswith("t"):
            continue
        cut = ((Wx.loc[START:END] > 0).sum(axis=1) < 10)
        h_cut = (Wb > 0) & ~(Wx.loc[START:END] > 0)
        for lab, a0, b0 in PER:
            r = Rx.loc[START:END].loc[a0:b0]
            cc = C.loc[START:END].loc[a0:b0]
            leg = (r - 2 * cc / 1e4).where(h_cut.loc[a0:b0]).stack().dropna()
            kept = (r - 2 * cc / 1e4).where((Wx.loc[START:END] > 0).loc[a0:b0]).stack().dropna()
            extra.append(dict(variant=key, period=lab, pricing="cross", share_nights_cut=cut.loc[a0:b0].mean(),
                              dropped_leg_net_bp=1e4 * leg.mean(), dropped_leg_t=leg.mean() / leg.std() * np.sqrt(len(leg)),
                              n_dropped_legs=len(leg), kept_leg_net_bp=1e4 * kept.mean()))
    d127 = d127.merge(pd.DataFrame(extra), on=["variant", "period", "pricing"], how="left")
    d118.to_csv(OUT118, index=False)
    d127.to_csv(OUT127, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 60)
    show = ["variant", "period", "sharpe", "net_bp", "maxdd", "names_per_night", "gross_invested", "turnover",
            "diff_vs_base_bp", "t_vs_base", "fallback_weight_share"]
    for nm, d in [("118", d118), ("127", d127)]:
        print(f"== {nm} (cross)")
        x = d[d.pricing == "cross"]
        print(x[show + (["share_nights_cut", "dropped_leg_net_bp", "dropped_leg_t"] if nm == "127" else [])]
              .round(3).to_string(index=False))
        print(f"== {nm} Sharpe by pricing")
        print(d.pivot_table(index="variant", columns=["pricing", "period"], values="sharpe", sort=False).round(2)
              .to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
