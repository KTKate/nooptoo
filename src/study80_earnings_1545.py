"""Study 80: study 77's earnings-event models for the reaction overnight, scored with inputs computed at 15:45.

Question: study 77 trained LightGBM only on earnings events to predict the reaction overnight (closing auction of t ->
opening auction of t+1). Its price inputs were the ml_frame row of t, i.e. computed with the close of t = the entry
price; with the row of t-1 the result fell sharply. Live trading decides at 15:45 ET and buys in the closing auction,
so the honest test scores with the price inputs as they were at 15:45 of t.

15:45 price inputs (2024-01 .. 2026-09; no 15:45 data for all names before 2024): exactly study 23's frame_1545
substitution (src/study23_ensemble_timing.py): the last row of the daily panels for day t is replaced by the price at
15:45 (close of the 15:40-15:45 SIP bar, m5snap, unfiltered 'none' panels as study 23; filled from m5snapx for the
small caps it lacks), day high/low widened to include it, 85% of the day's volume and dollar volume, SPY / IWM at
15:44 (m1); then ml_features.build + features_frame. Built only for the event tickers of event days: the 8
cross-sectional rank inputs (_cs) are computed on an 85-row window of all live tickers (enough for every input they
use: beta 60 rows + resid5), the other inputs on the full 261-row window of the event tickers only. Checks (section
'check'): without the substitution the builder reproduces the ml_frame rows; with it, it matches the full
261-row/all-ticker frame_1545 on sample days. ml_features' function definitions are executed without its module-level
full-panel build (memory). Data check on the 15:45 price: the m5snap 09:30 bar open must be within 15% of the daily
store's raw open (known before 15:45); m5snapx names (no 09:30 bar) must be within 40% of the previous raw close.
Events without a usable 15:45 price are dropped from every variant (coverage reported).

Events, targets, other inputs, model and walk-forward: study 77 imported unchanged (src/study77_event_models.py:
frame, add_inputs, feature set 'full': price + news (cut at 15:45 of t) + analyst (t-1) + fundamentals (t-1) +
previous report + sector + insiders + AMC flag). Samples: 'news' (first Benzinga EPS headline timing: no selection
on the reaction) and 'gap' (gap-rule timing: keeps only events whose reaction overnight moved >= 1% and >= 2x the
other overnight, i.e. selects on the reaction's size, not knowable at 15:45). Targets: quarter rank (both samples)
and raw clipped excess (gap sample, study 77's best). Test quarters 2024Q1 .. 2026Q3, training on all events
2020+ with t at least 3 trading days before the quarter (study 77).
Variants (training inputs -> scoring inputs):
  close_close  close-of-t price rows -> close-of-t rows (study 77 as published; reference, not tradable)
  close_1545   close-of-t rows -> 15:45 rows      (a: what a live system would do with study 77's training data)
  lag1_lag1    t-1 rows -> t-1 rows               (study 77 full_lag1; tradable but stale)
  lag1_1545    t-1 rows -> 15:45 rows             (b)
  (c: training on 15:45 rows needs 15:45 data before 2024: skipped.)
Books (each day >= 5 scorable events, study 77): long-short top / bottom 20% of the day's predictions (<= 10 names,
10% each; shorts only in names on Alpaca's current easy-to-borrow list: today's list, flatters the past), and
long-only top 20% (<= 10 names, 10% each) of liquid names (15:45 traded price > $5 on top of study 74's universe:
traded price of t-1 > $5, 20-day median dollar volume > $20M). Costs per side core.exec_cost_bps(P,'auction') +
2.5 bp (both legs auctions). Stats per year 2024 / 2025 / 2026 and 2024-26: net bp per trade, Sharpe (all trading
days, cash on days without trades), max drawdown, trades per year.
Overnight blend (study 68's score: (2 rank(ensemble, study 23 15:45 preds) + rank(p_jump - p_drop, study 33)) / 3,
top 10 equal weight, closing auction -> opening auction, same costs): overlap of the earnings book's names with the
blend's top 10, correlation of daily net returns (days the earnings book trades), and the combined book: on report
nights (earnings book active) w = 25% / 50% of capital goes to the earnings book and 1 - w to the blend; other nights
100% blend.
Baseline: the same books on random predictions (50 seeds) - what the model's selection adds over simply holding
the day's reporters (whose reaction overnight averages above the universe, plus the market's overnight drift).
Output: results/study80_earnings_1545.csv (sections: check, coverage, ic, book, overlap, combo, count);
cache data/local/study80_m45.parquet (15:45 input rows of the event tickers).
"""
import ast
import os
import sys
import time
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("NTHREADS", "2")
import study77_event_models as S7
import alpaca_data as A
import bt
from core import ann_stats, exec_cost_bps, RES, DATA

S = S7.S
P, cols, alld = S.P, S.cols, S.alld
OUT = os.path.join(RES, "study80_earnings_1545.csv")
CACHE = os.path.join(DATA, "local", "study80_m45.parquet")
T0 = pd.Timestamp("2024-01-02")
ROWS = []
NRAND = 50
YEARS = [("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
         ("2026", "2026-01-01", "2026-09-30"), ("2024-26", "2024-01-01", "2026-09-30")]


def emit(section, **kw):
    ROWS.append(dict(section=section, **kw))


def save():
    pd.DataFrame(ROWS).to_csv(OUT, index=False)


# ------------------------------------------------------------------ ml_features functions without its full build
def load_ml_funcs():
    """Executes src/ml_features.py without the module-level full-panel feature build (F, mkt, univ, targets)."""
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_features.py")
    tree = ast.parse(open(fn).read())
    skip = {"F", "mkt", "adv20", "univ", "T", "H"}
    body = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            names = {e.id for t in n.targets for e in ast.walk(t) if isinstance(e, ast.Name)}
            if names & skip:
                continue
        body.append(n)
    mod = types.ModuleType("ml_features_fn")
    mod.__file__ = fn
    exec(compile(ast.Module(body=body, type_ignores=[]), fn, "exec"), mod.__dict__)
    return mod


M = load_ml_funcs()
EARN = M.earnings_features()
o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
adjf = c / rawc
LIVE = [t for t in cols if P["dv"][t].loc["2023-06":].max() > 3e6]
CS = ["r1", "r5", "r20", "night1", "intra1", "vz1", "vol20", "resid5"]


def load_1545():
    """Raw 15:45 price (close of the 15:40 bar), date x ticker, m5snap first then m5snapx; and SPY/IWM 15:44."""
    from study8_exec_retest import snap_panels
    sp = snap_panels("none")
    p = sp["c15:40"].copy()
    o930 = sp["o09:30"].copy()
    del sp
    p.columns = p.columns.str.replace(".", "-", regex=False)
    o930.columns = o930.columns.str.replace(".", "-", regex=False)
    p, o930 = p.reindex(columns=cols), o930.reindex(columns=cols)
    # data check against the daily store (known before 15:45): 09:30 bar open vs raw open
    rawo = (P["o"][cols] / adjf).reindex_like(p)
    bad = (np.log(o930 / rawo).abs() > np.log(1.15)) & o930.notna()
    src = pd.DataFrame(np.where(p.notna(), 1, 0), index=p.index, columns=cols)
    p = p.where(~bad)
    nbad = int((bad & src.astype(bool)).sum().sum())
    # fill from m5snapx (small caps; 15:30-16:00 only)
    xs = []
    for m_ in pd.period_range("2024-01", "2026-09", freq="M").astype(str):
        fn = os.path.join(A.LOCAL, "m5snapx", f"{m_}.parquet")
        if os.path.exists(fn):
            d = pd.read_parquet(fn, columns=["ts", "ticker", "c"])
            d = d[d.ts.dt.strftime("%H:%M") == "15:40"]
            xs.append(d)
    x = pd.concat(xs)
    x["date"] = x.ts.dt.normalize()
    x["ticker"] = x.ticker.str.replace(".", "-", regex=False)
    px = x.pivot_table(index="date", columns="ticker", values="c", aggfunc="last").reindex(index=p.index, columns=cols)
    prev = rawc.shift(1).reindex_like(px)
    px = px.where(np.log(px / prev).abs() < np.log(1.4))
    fill = p.isna() & px.notna() & ~bad
    p = p.where(~fill, px)
    m1 = A.read("m1", start="2023-12", tickers=["SPY", "IWM"])
    m1 = m1[m1.ts.dt.strftime("%H:%M") == "15:44"]
    m1["date"] = m1.ts.dt.normalize()
    etf = m1.pivot(index="date", columns="ticker", values="c")
    return p, etf, nbad, int(fill.sum().sum())


def _window(d, tick, nrow, p45, etf, sub=True):
    i = alld.get_loc(d)
    sl = slice(max(0, i - nrow + 1), i + 1)
    oo, hh, ll, cc, rr, vv, dd = (z[tick].iloc[sl].copy() for z in (o, h, l, c, rawc, v, dv))
    spy, iwm = P["c"]["SPY"].iloc[sl].copy(), P["c"]["IWM"].iloc[sl].copy()
    ok = pd.Series(True, index=tick)
    if sub:
        px = p45.loc[d].reindex(tick)
        ok = px.notna()
        cc.iloc[-1] = np.where(ok, px * adjf.loc[d, tick], np.nan)
        rr.iloc[-1] = np.where(ok, px, np.nan)
        hh.iloc[-1] = np.maximum(hh.iloc[-1], cc.iloc[-1])
        ll.iloc[-1] = np.minimum(ll.iloc[-1], cc.iloc[-1])
        vv.iloc[-1] = vv.iloc[-1] * 0.85
        dd.iloc[-1] = dd.iloc[-1] * 0.85
        spy.iloc[-1] = etf.at[d, "SPY"] * (P["c"]["SPY"].loc[d] / P["rawc"]["SPY"].loc[d])
        iwm.iloc[-1] = etf.at[d, "IWM"] * (P["c"]["IWM"].loc[d] / P["rawc"]["IWM"].loc[d])
    F, mkt, adv20 = M.build(oo, hh, ll, cc, rr, vv, dd, spy, P["c"]["^VIX"].iloc[sl], P["c"]["^VIX3M"].iloc[sl],
                            iwm, earn=EARN)
    return F, mkt, adv20, rr, ok


def rows_fast(d, ev, p45, etf, pf, sub=True):
    """Input rows (pf columns) for event tickers ev on day d."""
    F1, _, adv1, rr1, ok1 = _window(d, LIVE, 85, p45, etf, sub)
    un = (rr1.iloc[-1] > 5) & (adv1.iloc[-1] > 5e6) & ok1
    cs = {k: F1[k].iloc[-1][un].astype("float32").rank(pct=True) for k in CS}
    F2, mkt2, _, _, ok2 = _window(d, ev, 261, p45, etf, sub)
    X = pd.DataFrame(index=ev)
    for k, f in F2.items():
        X[k] = f.iloc[-1].astype("float32")
    for k in CS:
        X[k + "_cs"] = cs[k].reindex(ev).astype("float32")
    for k in mkt2.columns:
        X[k] = np.float32(mkt2[k].iloc[-1])
    X["ok45"] = ok2.values
    X["px45"] = p45.loc[d].reindex(ev).values if sub else np.nan
    return X[pf + ["ok45", "px45"]]


def rows_full(d, ev, p45, etf, pf):
    """Reference: study 23's frame_1545 (261 rows, all live tickers) restricted to ev."""
    F, mkt, adv20, rr, ok = _window(d, LIVE, 261, p45, etf, True)
    univ = ((rr > 5) & (adv20 > 5e6)).iloc[[-1]] & ok.values
    last = {k: f.iloc[[-1]] for k, f in F.items()}
    X = M.features_frame(last, mkt.iloc[[-1]], univ)
    return X.droplevel(0).reindex(ev)[pf]


def build_m45(pairs, pf, p45, etf):
    out = []
    t = time.time()
    for n, (d, g) in enumerate(pairs.groupby("tdate")):
        if d not in p45.index or d not in etf.index:
            continue
        ev = sorted(set(g.symbol))
        X = rows_fast(d, ev, p45, etf, pf)
        X["tdate"] = d
        out.append(X.rename_axis("symbol").reset_index())
        if n % 50 == 0:
            print("m45 day", n, d.date(), f"{time.time() - t:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def checks(pairs, pf, p45, etf):
    """Builder vs ml_frame (no substitution) and vs full frame_1545 (substitution), 3 days."""
    days = sorted(pairs.tdate.unique())
    sample = [days[10], days[len(days) // 2], days[-20]]
    X = pd.read_parquet(os.path.join(DATA, "ml_frame.parquet"), filters=[("date", "in", list(sample))])
    for d in sample:
        ev = sorted(set(pairs.symbol[pairs.tdate == d]))
        a = rows_fast(d, ev, p45, etf, pf, sub=False)[pf]
        b = X.xs(d, level=0).reindex(ev)[pf]
        diff = (a.astype(float) - b.astype(float)).abs()
        emit("check", note=f"fast builder, no substitution vs ml_frame {d.date()}", n=len(ev),
             share=float((diff.max(1) < 1e-3).mean()), ic=float(np.nanmax(diff.values)))
        a = rows_fast(d, ev, p45, etf, pf)[pf]
        b = rows_full(d, ev, p45, etf, pf)
        diff = (a.astype(float) - b.astype(float)).abs()
        emit("check", note=f"fast builder vs full frame_1545 {d.date()}", n=len(ev),
             share=float((diff.max(1) < 1e-3).mean()), ic=float(np.nanmax(diff.values)))
        print(ROWS[-2], ROWS[-1], flush=True)
    save()


# ------------------------------------------------------------------ models
def walk_forward(x, tr_feats, sc_sets, ycol, target):
    """Study 77's walk-forward (test quarters 2024Q1+ only); one model per quarter scores every input set."""
    preds = {k: pd.Series(np.nan, index=x.index) for k in sc_sets}
    for q in sorted(x.q.unique()):
        if q < pd.Period("2024Q1", "Q"):
            continue
        te = (x.q == q).values & x.ok45.values
        cut = alld[max(0, alld.searchsorted(q.start_time) - 3)]
        tr = (x.tdate < cut).values & x[ycol].notna().values
        if tr.sum() < 3000 or te.sum() == 0:
            continue
        xt = x[tr]
        if target == "rank":
            y = xt.groupby("q")[ycol].rank(pct=True) - 0.5
        else:
            lo, hi = xt[ycol].quantile([0.01, 0.99])
            y = xt[ycol].clip(lo, hi)
        ds = S7.lgb.Dataset(xt[tr_feats].astype("float32").values, y.values, feature_name=tr_feats,
                            categorical_feature=["sector_code"], free_raw_data=True)
        m = S7.lgb.train(S7.PR, ds, num_boost_round=S7.NROUND)
        for k, f in sc_sets.items():
            preds[k][te] = m.predict(x.loc[te, f].astype("float32").values)
    return preds


def book_stats(b, side, a0, b0):
    d = b[side].loc[a0:b0]
    ntr = d.n.sum()
    if ntr == 0:
        return None
    s = ann_stats(d.net)
    return dict(trades=int(ntr), trades_yr=ntr / (len(d) / 252), days_active=int((d.n > 0).sum()),
                avg_names=d.n[d.n > 0].mean(), net_bp_tr=1e4 * d.tr_net.sum() / ntr,
                ex_bp_tr=1e4 * d.tr_ex.sum() / ntr, ann_ret=s["ann_ret"], sharpe=s["sharpe"], t=s["tstat"],
                maxdd=s["maxdd"])


def blend_book():
    pred = pd.read_parquet(os.path.join(RES, "study33_pred.parquet"))
    ens = pd.read_parquet(os.path.join(RES, "study23_pred.parquet"))["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < alld[-2]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    Sc = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    W = bt.select_topk(Sc, Sc.notna(), 10)
    R = (o.shift(-1) / c - 1).reindex_like(W)
    cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(W)
    r = bt.run(W, R, cost)
    return r, W


def main():
    xs = {}
    for sample, tcol in [("news", "t_news"), ("gap", "t_gap")]:
        x = S7.frame(tcol)
        x, pf = S7.add_inputs(x)
        xs[sample] = x
        print(sample, "events", len(x), flush=True)
    pairs = pd.concat([x.loc[x.tdate >= T0, ["tdate", "symbol"]] for x in xs.values()]).drop_duplicates()
    print("event pairs 2024+", len(pairs), "days", pairs.tdate.nunique(), flush=True)
    p45, etf, nbad, nfill = load_1545()
    emit("coverage", note="15:45 price: m5snap rows dropped by the 09:30 open check (all tickers)", n=nbad)
    emit("coverage", note="15:45 price: cells filled from m5snapx (all tickers)", n=nfill)
    checks(pairs, pf, p45, etf)
    if os.path.exists(CACHE):
        m45 = pd.read_parquet(CACHE)
    else:
        m45 = build_m45(pairs, pf, p45, etf)
        m45.to_parquet(CACHE)
    del p45
    m45 = m45.set_index(["tdate", "symbol"])
    FS = S7.feature_sets(pf)
    full = FS[("through", "full")]
    lag1 = FS[("through", "full_lag1")]
    m45f = ["m45_" + k[4:] if k.startswith("thr_") else k for k in full]
    configs = [("news", "rank"), ("gap", "rank"), ("gap", "raw")]
    blend, Wb = blend_book()
    emit("book", sample="blend", variant="overnight_blend_top10", side="long", period="2024-26",
         **{k: v for k, v in ann_stats(blend.net.loc["2024-01-01":"2026-09-25"]).items() if k in ("sharpe", "maxdd")})
    nvar = 0
    for sample, target in configs:
        x = xs[sample]
        k = pd.MultiIndex.from_arrays([x.tdate, x.symbol])
        mm = m45.reindex(k)
        for cname in pf:
            x["m45_" + cname] = mm[cname].values
        x["ok45"] = mm["ok45"].fillna(False).astype(bool).values & (x.tdate >= T0).values
        x["px45"] = mm["px45"].values
        n24 = int((x.tdate >= T0).sum())
        emit("coverage", sample=sample, note="events 2024-01..2026-09 with a 15:45 price (scorable)", n=n24,
             share=float(x.ok45.sum() / n24))
        print(sample, "2024+ events", n24, "with 15:45 inputs", int(x.ok45.sum()), flush=True)
        ycol, rcol, mcol = "ex_on", "r_on", "mkt_on"
        P1 = walk_forward(x, full, {"close_close": full, "close_1545": m45f}, ycol, target)
        P2 = walk_forward(x, lag1, {"lag1_lag1": lag1, "lag1_1545": m45f}, ycol, target)
        preds = {**P1, **P2}
        nvar += len(preds)
        liq = (x.px45 > 5).values
        for var, p in preds.items():
            p = p.where(x.ok45)
            icd, icd_t, nd = S7.ic_daily(x, p, ycol, "2024-01-01", "2026-09-30")
            emit("ic", sample=sample, target=target, variant=var, period="2024-26", ic_daily=icd, ic_daily_t=icd_t,
                 ic_days=nd, n=int(p.notna().sum()))
            b = S7.books(x, p, rcol, mcol)
            bl = S7.books(x[liq], p[liq], rcol, mcol)
            line = [f"{sample}|{target}|{var} dIC {icd:.4f} (t {icd_t:.2f})"]
            for per, a0, b0 in YEARS:
                for side, bb, sd in [("ls", b, "ls"), ("long", b, "long"), ("short", b, "short"),
                                     ("long_only_liquid", bl, "long")]:
                    st = book_stats(bb, sd, a0, b0)
                    if st is None:
                        continue
                    if per == "2024-26":
                        tt = bb["trades"]
                        st.update(S7.tail_stats(tt[(tt.side == sd) & (tt.tdate >= T0)]))
                    emit("book", sample=sample, target=target, variant=var, side=side, period=per, **st)
                    if per == "2024-26" or side in ("ls", "long_only_liquid"):
                        line.append(f"{per} {side}: {st['net_bp_tr']:.0f}bp SR {st['sharpe']:.2f} "
                                    f"DD {st['maxdd']:.2f} n/yr {st['trades_yr']:.0f}")
            print(" | ".join(line), flush=True)
            if var in ("close_1545", "lag1_1545"):
                # overlap / correlation with the overnight blend; combined book
                for side, bb, sd in [("ls", b, "ls"), ("long_only_liquid", bl, "long")]:
                    lt = bb["trades"]
                    lt = lt[(lt.side == "long") & (lt.tdate >= T0)]
                    inl = np.array([Wb.at[d_, s_] > 0 if (d_ in Wb.index and s_ in Wb.columns) else False
                                    for d_, s_ in zip(lt.tdate, lt.symbol)])
                    st = bb["trades"]
                    st = st[(st.side == "short") & (st.tdate >= T0)]
                    ins = np.array([Wb.at[d_, s_] > 0 if (d_ in Wb.index and s_ in Wb.columns) else False
                                    for d_, s_ in zip(st.tdate, st.symbol)])
                    e = bb[sd].net.loc["2024-01-01":"2026-09-25"]
                    act = bb[sd].n.loc["2024-01-01":"2026-09-25"] > 0
                    bn = blend.net.reindex(e.index).fillna(0.0)
                    corr = float(np.corrcoef(e[act], bn[act])[0, 1])
                    corr_all = float(np.corrcoef(e, bn)[0, 1])
                    emit("overlap", sample=sample, target=target, variant=var, side=side, period="2024-26",
                         n=int(len(lt)), share=float(inl.mean()) if len(inl) else np.nan,
                         note=f"long picks in blend top10 {inl.mean():.3f}; short picks in blend top10 "
                              f"{(ins.mean() if len(ins) else np.nan):.3f}; corr daily net on report nights "
                              f"{corr:.3f}, all nights {corr_all:.3f}; report nights {int(act.sum())}",
                         ic=corr, ic_daily=corr_all, days_active=int(act.sum()))
                    for w in [0.0, 0.25, 0.5]:
                        comb = bn.where(~act, (1 - w) * bn + w * e)
                        for per, a0, b0 in YEARS:
                            s_ = ann_stats(comb.loc[a0:b0])
                            emit("combo", sample=sample, target=target, variant=var, side=side, period=per,
                                 share=w, sharpe=s_["sharpe"], maxdd=s_["maxdd"], ann_ret=s_["ann_ret"])
                        print(f"  combo {side} w={w}: SR " + " ".join(
                            f"{per} {ann_stats(comb.loc[a0:b0])['sharpe']:.2f}" for per, a0, b0 in YEARS), flush=True)
            save()
        # baseline: random predictions (same events, same books), 50 seeds: what selection adds over being in
        # the day's reporters at all
        rs = {}
        for seed in range(NRAND):
            p = pd.Series(np.random.default_rng(seed).random(len(x)), index=x.index).where(x.ok45)
            b = S7.books(x, p, rcol, mcol)
            for per, a0, b0 in YEARS:
                for side in ["ls", "long", "short"]:
                    st = book_stats(b, side, a0, b0)
                    if st is not None:
                        rs.setdefault((per, side), []).append((st["sharpe"], st["net_bp_tr"], st["ex_bp_tr"]))
        for (per, side), vals in rs.items():
            a_ = np.array(vals)
            emit("book", sample=sample, target=target, variant=f"random_{NRAND}seeds", side=side, period=per,
                 sharpe=a_[:, 0].mean(), net_bp_tr=a_[:, 1].mean(), ex_bp_tr=a_[:, 2].mean(),
                 t=np.percentile(a_[:, 0], 95),
                 note="mean over seeds; column t = 95th percentile of the seeds' Sharpe")
            if per == "2024-26":
                print(f"{sample}|{target} random {side}: SR mean {a_[:, 0].mean():.2f} p95 "
                      f"{np.percentile(a_[:, 0], 95):.2f} net {a_[:, 1].mean():.0f}bp ex {a_[:, 2].mean():.0f}bp",
                      flush=True)
        save()
    emit("count", n=nvar, note="3 configs (news/rank, gap/rank, gap/raw) x 4 train->score variants (2 references, "
                               "2 tradable: close->15:45, t-1->15:45); full inputs; hyperparameters from study 77; "
                               "each with ls / long / short / long-only-liquid books")
    save()
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
