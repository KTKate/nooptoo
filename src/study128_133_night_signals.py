"""Studies 128 and 133: two overnight signals tested on their own and as additions to the overnight blend.

128  Overnight-return persistence. Question: the literature reports that stocks with high past close-to-open returns
     keep earning high close-to-open returns. Does that hold here, net of costs, and does it add anything to the
     blend, whose study-3 inputs already contain a 20-day overnight average (data/ml_frame.parquet night20 = 20-day
     mean of log(open / previous close), min 15 days; also night1, night5, night1_cs)?
133  Large-cap intraday losers. Question: among liquid large caps, do the day's worst performers earn a positive
     overnight return (buy at the close, sell at the next open), and does adding them to the blend help?

Data: core.load_panel (split- and dividend-adjusted daily o/h/l/c, rawc, v, dv; stock_cols). Universe for 128 as
ml_features: traded price (horizon_lib.TPX: Alpaca raw close, extended before 2023-12 by the Yahoo raw close scaled to
it) > $5 and 20-day median dollar volume > $5M on day t. Overnight return R(t) = open(t+1) / close(t) - 1 (panel).
Costs core.exec_cost_bps(P, "auction") + 2.5 bp per side (closing-auction buy, opening-auction sell). Blend as study
69 lines 22-46 (top 10 of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3; results/study23_pred.parquet and
study33_pred.parquet, 2024-01..2026-09 only). 15:45 prices: data/local/m5snap 5-minute bars (2024-01..2026-09), as
study 23 builds p1545 (the close of the 15:40 bar), read from the unfiltered snapshot cache
(m5snap_panels_686_none.pkl). Official crosses: caches of studies 88/94/99/108/110/113 plus crosses fetched here
(data/local/auctions128.parquet; study 94's rule: first 'O' / '6' print of the listing exchange, second pass the
largest print of any listing exchange; Alpaca SIP, resumable, ~60 requests a minute).

Design 128: signals known at the close of t from overnight returns that ended at or before today's open (R(t-1) back
to R(t-k)): mean overnight return over k = 20, 60, 120, 250 days (daily returns clipped to +-30% so one biotech
crash or a bad split does not dominate) and share of positive nights (min 80% of the window present). Per period:
daily Spearman rank IC with R(t) inside the universe, decile means and the decile 10 minus decile 1 spread (gross),
equal-weight top-decile long net of costs (and its gross excess over the universe mean). Overlap with the existing
feature: daily rank correlation of each signal with night20 (recomputed from the panel; checked equal to ml_frame);
residual signal = rank(signal) minus its daily cross-sectional regression on rank(night20), tested the same way.
On the blend (2024-01..2026-09): tilt S + w (q - 0.5) with q the signal's percentile inside the blend universe
(w = 0.15, 0.30; missing signal q = 0.5), and a filter that drops the bottom quintile of the signal; paired t of the
daily net difference vs the base blend. Choice on 2024-01..2025-06, report 2025-07..2026-09.
Design 133: universe top 500 stocks by 20-day mean dollar volume (through t-1) with traded price > $10 on t-1. Buy the
10 with the worst (a) open-to-close return (close-based proxy, all periods; selection uses the very close price it
buys at, so bid-ask bounce at the close flatters it), (b) open-to-15:45 return (p1545 / 09:30 bar open, both on
Alpaca's raw scale; 2024-26), (c) close-to-close day return (proxy) and (d) previous close (15:55 bar close) to 15:45
(2024-26). 15:45 legs whose return disagrees with the panel's by more than 15% in log terms are data errors and
dropped. Hold to the next open, equal weight, costs as above. Correlation of daily net returns with the blend, the
share of picks the blend also holds, and blend + x * leg for x = 0.25, 0.5 (extra capital). Choice of leg and x on
2024-01..2025-06 (standalone also on 2020-23), report 2025-07..2026-09. The chosen leg and the best 128 blend variant
are rescored at official crosses (study 94 / 110 method: panel night return x (open cross / panel open on t+1) /
(close cross / panel close on t), legs without a usable cross at the panel price).

Output: results/study128_night_persistence.csv (tables: check, featcorr, signal, decile, blend, cross),
results/study133_largecap_losers.csv (tables: check, leg, corr, combo, cross).

    python src/study128_133_night_signals.py fetch   # official crosses for the chosen picks (resumable)
    python src/study128_133_night_signals.py         # study
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC128 = os.path.join(LOC, "auctions128.parquet")
OUT128 = os.path.join(RES, "study128_night_persistence.csv")
OUT133 = os.path.join(RES, "study133_largecap_losers.csv")
PER = [("2020-23", "2020-01", "2023-12"), ("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]
PERB = PER[1:] + [("2024-26", "2024-01", "2026-09")]
WINS = [20, 60, 120, 250]


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def rowcorr(a, b):
    m = a.notna() & b.notna()
    a, b = a.where(m), b.where(m)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def pstats(net, a0, b0):
    x = net.loc[a0:b0].dropna()
    st = ann_stats(x)
    return dict(n=st["n"], sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"])


# ------------------------------------------------------------------ data
def base():
    import horizon_lib as H
    P = H.P
    cols, days = H.cols, H.days
    o, c = P["o"][cols], P["c"][cols]
    R = (o.shift(-1) / c - 1).loc["2019-06":"2026-09"]
    rows = R.index
    TPX = H.TPX.reindex(rows)
    adv_med = P["dv"][cols].rolling(20, min_periods=10).median().reindex(rows)
    U = (TPX > 5) & (adv_med > 5e6) & c.reindex(rows).notna()
    COST = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(rows)
    return dict(P=P, cols=cols, days=days, o=o.reindex(rows), c=c.reindex(rows), R=R, TPX=TPX, U=U, COST=COST)


def blend(K):
    """Study 69 lines 22-46."""
    P, cols, days = K["P"], K["cols"], K["days"]
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    S = S.loc[:"2026-09"]
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    W = bt.select_topk(S, S.notna(), 10)
    return S, R, C, W


def night_signals(K):
    """Signals known at the close of t from R(t-1) .. R(t-k)."""
    o, c = K["o"], K["c"]
    rn = (o / c.shift(1) - 1)                      # last night's return, known at today's open (= R(t-1))
    rc = rn.clip(-0.3, 0.3)
    pos = (rn > 0).astype("float32").where(rn.notna())
    sig = {}
    for k in WINS:
        mp = int(0.8 * k)
        sig[f"mean{k}"] = rc.rolling(k, min_periods=mp).mean().astype("float32")
        sig[f"pos{k}"] = pos.rolling(k, min_periods=mp).mean().astype("float32")
    night20 = np.log(o / c.shift(1)).rolling(20, min_periods=15).mean().astype("float32")   # ml_features night20
    return sig, night20


def residual(sig, ref, m):
    """rank(sig) minus its daily OLS fit on rank(ref), inside mask m."""
    mm = m & sig.notna() & ref.notna()
    a = sig.where(mm).rank(axis=1, pct=True)
    b = ref.where(mm).rank(axis=1, pct=True)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    beta = (a * b).sum(axis=1) / (b * b).sum(axis=1)
    return (a - b.mul(beta, axis=0)).astype("float32")


def p1545_panels(K):
    s = pd.read_pickle(os.path.join(LOC, "m5snap_panels_686_none.pkl"))
    g = lambda k: s[k].reindex(index=K["R"].index, columns=K["cols"]).astype("float64")
    return g("o09:30"), g("c15:40"), g("c15:55")


def loser_legs(K):
    """Study 133 signals (lower = worse day) and the large-cap universe."""
    P, cols, o, c = K["P"], K["cols"], K["o"], K["c"]
    rows = K["R"].index
    dv20 = P["dv"][cols].rolling(20, min_periods=15).mean().shift(1).reindex(rows)
    big = (dv20.rank(axis=1, ascending=False, method="first") <= 500) & (K["TPX"].shift(1) > 10) & c.notna() & o.notna()
    oc = c / o - 1
    cc = c / c.shift(1) - 1
    o930, c1540, c1555 = p1545_panels(K)
    oc45 = c1540 / o930 - 1
    cc45 = c1540 / c1555.shift(1) - 1
    chk = {}
    bad_oc = (np.log((1 + oc45) / (1 + oc)).abs() > np.log(1.15)) & big
    bad_cc = (np.log((1 + cc45) / (1 + cc)).abs() > np.log(1.15)) & big
    lc = big.loc["2024-01":]
    chk["oc1545_found"] = (oc45.notna() & big).loc["2024-01":].sum().sum() / lc.sum().sum()
    chk["oc1545_bad"] = int(bad_oc.sum().sum())
    chk["cc1545_found"] = (cc45.notna() & big).loc["2024-01":].sum().sum() / lc.sum().sum()
    chk["cc1545_bad"] = int(bad_cc.sum().sum())
    oc45, cc45 = oc45.mask(bad_oc), cc45.mask(bad_cc)
    legs = {"open-to-close (close proxy)": oc, "open-to-15:45": oc45,
            "close-to-close (close proxy)": cc, "prev close-to-15:45": cc45}
    return legs, big, chk


def pick_sets(K, S=None, W=None):
    """The chosen portfolios for the cross rescore (choices fixed from the 2024-01..2025-06 results)."""
    legs, big, _ = loser_legs(K)
    out = {}
    for name in ["prev close-to-15:45"]:          # the 15:45 leg with the better 2024-01..2025-06 result
        x = legs[name]
        out[f"133 {name}"] = bt.select_topk(x, big & x.notna(), 10, largest=False).loc["2024-01":"2026-09"]
    if S is not None:
        sig, n20 = night_signals(K)
        okS = S.notna()
        for key in BLEND_CHOSEN:
            nm, kind, prm = key
            Sx = blend_variant(S, sig[nm] if not nm.startswith("res_") else residual(sig[nm[4:]], n20, K["U"]), kind, prm)
            out[f"128 {nm} {kind} {prm}"] = bt.select_topk(Sx, Sx.notna(), 10)
    return out


def blend_variant(S, s, kind, prm):
    okS = S.notna()
    q = s.reindex_like(S).where(okS).rank(axis=1, pct=True)
    if kind == "tilt":
        return S + prm * (q.fillna(0.5) - 0.5)
    return S.where(~(q < prm))          # filter: drop the bottom prm share (missing kept)


BLEND_CHOSEN = []          # filled after the first run (see main); used by the cross fetch


# ------------------------------------------------------------------ crosses
def crosses():
    import study99_cross_retrain as S99
    x = S99.crosses()
    parts = [x]
    for f in ["auctions108.parquet", "auctions110.parquet", "auctions113.parquet", "auctions128.parquet"]:
        fn = os.path.join(LOC, f)
        if os.path.exists(fn):
            c = pd.read_parquet(fn)[["ticker", "date", "kind", "px"]].dropna(subset=["px"])
            c["ticker"] = c.ticker.astype(str).str.replace(".", "-", regex=False)
            c["date"] = pd.to_datetime(c.date)
            parts.append(c)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date", "kind"])


def tried():
    parts = [crosses()[["ticker", "date", "kind"]]]
    for f in ["auctions94.parquet", "auctions99.parquet", "auctions108.parquet", "auctions110.parquet",
              "auctions113.parquet", "auctions128.parquet"]:
        fn = os.path.join(LOC, f)
        if os.path.exists(fn):
            t = pd.read_parquet(fn)[["ticker", "date", "kind"]]
            t["ticker"] = t.ticker.astype(str).str.replace(".", "-", regex=False)
            parts.append(t.assign(date=pd.to_datetime(t.date)))
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def fetch():
    import study94_cross_rescore as S94
    import alpaca_data as A
    A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
    S94.A.RL = A.RL
    K = base()
    S, _, _, W = blend(K)
    sets = pick_sets(K, S, W)
    days = K["days"]
    nxt = pd.Series(days[1:], index=days[:-1])
    rows = []
    for Wx in sets.values():
        s = Wx.stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()
    need["date"] = pd.to_datetime(need.date)
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
            if miss:
                lg = {x["ticker"]: x for x in S94._day_kind_largest(job[0], job[1], miss)}
                for x in r:
                    if x["ticker"] in lg:
                        x["rule"] = "largest"
                        if not np.isnan(lg[x["ticker"]]["px"]):
                            x.update({k: lg[x["ticker"]][k] for k in ["px", "sz", "ts", "x"]})
            return r
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    old = pd.read_parquet(AUC128) if os.path.exists(AUC128) else pd.DataFrame()
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
        df.to_parquet(AUC128, index=False)
        print("jobs", min(i + 12, len(jobs)), "/", len(jobs), "rows", len(new),
              f"found {np.mean([not np.isnan(r['px']) for r in new]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ study 128
def study128(K, S, Rb, Cb, Wb, base_net):
    R, U, COST = K["R"], K["U"], K["COST"]
    rows = []
    sig, n20 = night_signals(K)
    # check: recomputed night20 equals data/ml_frame.parquet night20
    ml = pd.read_parquet(os.path.join(DATA, "ml_frame.parquet"), columns=["night20"]).reset_index()
    ml = ml.sample(200000, random_state=128)
    mine = n20.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([pd.to_datetime(ml.date), ml.ticker])).values
    ok = ~np.isnan(mine) & ml.night20.notna().values
    rows.append(dict(table="check", variant="night20 panel vs ml_frame", n=int(ok.sum()),
                     value=np.mean(np.abs(mine[ok] - ml.night20.values[ok]) < 1e-5)))
    del ml
    # residual signals
    for k in WINS:
        for b in ["mean", "pos"]:
            sig[f"res_{b}{k}"] = residual(sig[f"{b}{k}"], n20, U)
    # rank correlation with night20 and with each other
    for nm, s in sig.items():
        if nm.startswith("res_"):
            continue
        m = U & s.notna() & n20.notna()
        rc = rowcorr(s.where(m).rank(axis=1), n20.where(m).rank(axis=1))
        for lab, a0, b0 in PER:
            rows.append(dict(table="featcorr", variant=nm, period=lab, value=rc.loc[a0:b0].mean()))
    Umean = R.where(U).mean(axis=1)
    for nm, s in sig.items():
        m = U & s.notna() & R.notna()
        ic = rowcorr(s.where(m).rank(axis=1), R.where(m).rank(axis=1))
        q = s.where(U & s.notna()).rank(axis=1, pct=True)
        dec = np.ceil(q * 10)
        dm = pd.DataFrame({d: R.where(dec == d).mean(axis=1) for d in range(1, 11)})
        W = (dec == 10).astype("float32")
        W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        res = bt.run(W, R, COST)
        for lab, a0, b0 in PER:
            x = dm.loc[a0:b0]
            spread = (x[10] - x[1]).dropna()
            st = pstats(res.net, a0, b0)
            exc = (res.gross - Umean).loc[a0:b0]
            rows.append(dict(table="signal", variant=nm, period=lab, ic=ic.loc[a0:b0].mean(), ic_t=tstat(ic.loc[a0:b0]),
                             spread_bp=1e4 * spread.mean(), spread_t=tstat(spread), top_sharpe=st["sharpe"],
                             top_net_bp=st["net_bp"], top_gross_bp=1e4 * res.gross.loc[a0:b0].mean(),
                             top_excess_bp=1e4 * exc.mean(), top_excess_t=tstat(exc),
                             top_cost_bp=1e4 * res.cost.loc[a0:b0].mean(), univ_bp=1e4 * Umean.loc[a0:b0].mean(),
                             top_maxdd=st["maxdd"]))
            rows.append(dict(table="decile", variant=nm, period=lab,
                             **{f"d{d}_bp": 1e4 * x[d].mean() for d in range(1, 11)}))
    # blend tilt / filter
    for lab, a0, b0 in PERB:
        rows.append(dict(table="blend", variant="base: blend top 10", period=lab, **pstats(base_net, a0, b0)))
    for nm, s in sig.items():
        for kind, prm in [("tilt", 0.15), ("tilt", 0.30), ("filter", 0.2)]:
            Sx = blend_variant(S, s, kind, prm)
            Wx = bt.select_topk(Sx, Sx.notna(), 10)
            net = bt.run(Wx, Rb, Cb).net
            diff = net - base_net
            keep = ((Wx > 0) & (Wb > 0)).sum(axis=1).mean() / 10
            for lab, a0, b0 in PERB:
                d = diff.loc[a0:b0]
                rows.append(dict(table="blend", variant=f"{nm} {kind} {prm}", signal=nm, kind=kind, prm=prm, period=lab,
                                 **pstats(net, a0, b0), diff_bp=1e4 * d.mean(), diff_t=tstat(d), share_same=keep))
    return pd.DataFrame(rows), sig


def gapcheck(K, sig):
    """Does the signal pick stocks whose panel open sits above the official opening cross (study 88/94 bias)?
    All cached opening crosses (mostly earlier strategies' picks) on sell day t+1, grouped by the signal quintile at
    t inside the 128 universe; bias bp = panel open / cross - 1."""
    import study99_cross_retrain as S99
    P, cols, days = K["P"], K["cols"], K["days"]
    ro, _, _ = S99.cross_ratios(P, cols, days, crosses())
    gap = (1e4 * (1 / ro - 1)).reindex(index=days, columns=cols).shift(-1).reindex(K["R"].index)   # dated t
    rows = []
    for nm in ["mean20", "mean250", "pos60", "pos250"]:
        s = sig[nm]
        q = np.ceil(5 * s.where(K["U"] & s.notna()).rank(axis=1, pct=True))
        for lab, a0, b0 in PER[1:]:
            g, qq = gap.loc[a0:b0], q.loc[a0:b0]
            r = dict(table="gapcheck", variant=nm, period=lab)
            for k in range(1, 6):
                v = g.where(qq == k).stack().dropna()
                r[f"q{k}_bp"], r[f"q{k}_n"] = v.mean(), len(v)
            rows.append(r)
    return rows


# ------------------------------------------------------------------ study 133
def study133(K, Wb, base_net):
    R, COST = K["R"], K["COST"]
    legs, big, chk = loser_legs(K)
    rows = [dict(table="check", variant=k, value=v) for k, v in chk.items()]
    Ubig = R.where(big).mean(axis=1)
    nets = {}
    for name, x in legs.items():
        for k in [10]:
            W = bt.select_topk(x, big & x.notna(), k, largest=False)
            res = bt.run(W, R, COST)
            net = res.net.where(W.sum(axis=1) > 0)
            nets[name] = net
            same = ((W > 0) & (Wb.reindex_like(W).fillna(0) > 0)).sum(axis=1)
            for lab, a0, b0 in PER:
                if net.loc[a0:b0].notna().sum() < 20:
                    continue
                st = pstats(net, a0, b0)
                exc = (res.gross - Ubig).where(W.sum(axis=1) > 0).loc[a0:b0]
                sig_mean = (x.where(W > 0).sum(axis=1) / k).where(W.sum(axis=1) > 0).loc[a0:b0].mean()
                rows.append(dict(table="leg", variant=name, period=lab, k=k, **st,
                                 gross_bp=1e4 * res.gross.loc[a0:b0].where(W.sum(axis=1) > 0).mean(),
                                 cost_bp=1e4 * res.cost.loc[a0:b0].where(W.sum(axis=1) > 0).mean(),
                                 excess_bp=1e4 * exc.mean(), excess_t=tstat(exc), t=tstat(net.loc[a0:b0]),
                                 univ_bp=1e4 * Ubig.loc[a0:b0].mean(), pick_ret_mean=sig_mean,
                                 blend_overlap=same.loc[a0:b0].mean() / k if lab != "2020-23" else np.nan))
    for name, net in nets.items():
        leg = net.reindex(base_net.index).fillna(0)
        for lab, a0, b0 in PERB:
            if net.loc[a0:b0].notna().sum() < 20:
                continue
            rows.append(dict(table="corr", variant=name, period=lab,
                             value=np.corrcoef(base_net.loc[a0:b0].fillna(0), leg.loc[a0:b0])[0, 1]))
            rows.append(dict(table="combo", variant="base: blend top 10", period=lab, x=0.0, **pstats(base_net, a0, b0)))
            for xx in [0.25, 0.5]:
                comb = base_net + xx * leg
                rows.append(dict(table="combo", variant=name, period=lab, x=xx, **pstats(comb, a0, b0),
                                 diff_bp=1e4 * (xx * leg).loc[a0:b0].mean(), diff_t=tstat(leg.loc[a0:b0])))
    return pd.DataFrame(rows).drop_duplicates(subset=None), nets


# ------------------------------------------------------------------ cross rescore
def cross_rows(K, S, Wb, sets, study):
    import study99_cross_retrain as S99
    P, cols, days = K["P"], K["cols"], K["days"]
    ro, rc, err = S99.cross_ratios(P, cols, days, crosses())
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    ix = S.index
    Rp = (o.shift(-1) / c - 1).reindex(index=ix, columns=cols)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex(index=ix) / rc.fillna(1.0).reindex(index=ix) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=ix)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=ix)
    rows = [dict(table="cross", variant=f"data errors {k}", value=v["scale"] + v["ratio"], n=v["found"])
            for k, v in err.items()]
    nets = {}
    allsets = {"base: blend top 10": Wb, **sets}
    for name, Wx in allsets.items():
        Wx = Wx.reindex(index=ix, columns=cols).fillna(0)
        for pr, R in [("panel", Rp), ("cross", Rx)]:
            nets[(name, pr)] = bt.run(Wx, R, C).net.where(Wx.sum(axis=1) > 0)
        h = Wx > 0
        gap = (Rp - Rx).where(h & ~miss)
        for lab, a0, b0 in PERB:
            for pr in ["panel", "cross"]:
                rows.append(dict(table="cross", variant=name, pricing=pr, period=lab, **pstats(nets[(name, pr)], a0, b0)))
            hh = h.loc[a0:b0]
            rows.append(dict(table="cross", variant=name, pricing="fallback / gap", period=lab,
                             value=(miss.loc[a0:b0] & hh).sum().sum() / max(hh.sum().sum(), 1),
                             net_bp=1e4 * np.nanmean(gap.loc[a0:b0].values)))
    return rows, nets


def main():
    global BLEND_CHOSEN
    K = base()
    S, Rb, Cb, Wb = blend(K)
    base_net = bt.run(Wb, Rb, Cb).net
    print("base blend Sharpe", {lab: round(pstats(base_net, a0, b0)["sharpe"], 2) for lab, a0, b0 in PERB}, flush=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    # ---- 128
    df128, sig = study128(K, S, Rb, Cb, Wb, base_net)
    gc = pd.DataFrame(gapcheck(K, sig))
    del sig
    print(gc.round(2).to_string())
    df128 = pd.concat([df128, gc], ignore_index=True)
    sg = df128[df128.table == "signal"]
    print(sg.pivot_table(index="variant", columns="period", values=["ic", "ic_t", "spread_bp", "spread_t", "top_sharpe", "top_net_bp", "top_excess_bp", "top_excess_t"],
                         sort=False).round(3).to_string())
    print(df128[df128.table == "featcorr"].pivot_table(index="variant", columns="period", values="value").round(2))
    bl = df128[df128.table == "blend"]
    print(bl.pivot_table(index="variant", columns="period", values=["sharpe", "diff_bp", "diff_t"], sort=False)
          .round(2).to_string())
    v = bl[(bl.period == "2024-25H1") & bl.signal.notna()].sort_values("sharpe", ascending=False)
    best = v.iloc[0]
    BLEND_CHOSEN = [(best.signal, best.kind, best.prm)]
    print("chosen 128 blend variant on 2024-25H1:", BLEND_CHOSEN, flush=True)
    # ---- 133
    df133, nets133 = study133(K, Wb, base_net)
    print(df133[df133.table == "check"].to_string())
    print(df133[df133.table == "leg"].pivot_table(index="variant", columns="period",
                                                   values=["sharpe", "net_bp", "excess_bp", "blend_overlap"]).round(2).to_string())
    print(df133[df133.table == "corr"].pivot_table(index="variant", columns="period", values="value").round(2))
    print(df133[df133.table == "combo"].pivot_table(index=["variant", "x"], columns="period", values=["sharpe", "diff_t"])
          .round(2).to_string())
    # ---- cross rescore of the chosen portfolios
    sets = pick_sets(K, S, Wb)
    for name in list(sets):
        if name.startswith("133"):
            sets[name + " (alone)"] = sets.pop(name)
    crow, cn = cross_rows(K, S, Wb, sets, None)
    for name in [n for n in sets if n.startswith("133")]:
        for xx in [0.25, 0.5]:
            for pr in ["panel", "cross"]:
                comb = cn[("base: blend top 10", pr)] + xx * cn[(name, pr)].reindex(S.index).fillna(0)
                for lab, a0, b0 in PERB:
                    crow.append(dict(table="cross", variant=f"blend + {xx} x {name}", pricing=pr, period=lab,
                                     **pstats(comb, a0, b0)))
    cr = pd.DataFrame(crow)
    print(cr[cr.pricing.isin(["panel", "cross"])].pivot_table(index="variant", columns=["period", "pricing"],
                                                              values="sharpe", sort=False).round(2).to_string())
    print(cr[~cr.pricing.isin(["panel", "cross"])].to_string())
    is128 = cr.variant.str.startswith("128") | cr.variant.str.startswith("base") | cr.variant.str.startswith("data")
    pd.concat([df128, cr[is128]], ignore_index=True).to_csv(OUT128, index=False)
    pd.concat([df133, cr[~cr.variant.str.startswith("128")]], ignore_index=True).to_csv(OUT133, index=False)
    print("wrote", OUT128, OUT133)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        BLEND_CHOSEN = [tuple(x.split(":")[:2]) + (float(x.split(":")[2]),) for x in sys.argv[2:]]
        fetch()
    else:
        main()
