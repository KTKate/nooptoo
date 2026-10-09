"""Study 94: rescore the overnight blend and the candidate legs (studies 67/71 day-session long, 68 day short, the live
industry-capped blend) at the official opening and closing crosses instead of the daily panel's open and close.

Question: study 88 found that the daily panel's open (Yahoo, core.load_panel P["o"]) is usually the day's first trade,
not the listing exchange's opening cross, and for the blend's picks lies +3.5 bp (2024-01..2025-06) and +12 bp
(2025-07..2026-09) above the cross on average. Backtests that sell at P["o"] are then too optimistic, and the day
legs that buy or short at P["o"] are mis-stated the other way. (1) What are the Sharpe, mean net return and drawdown
of each strategy priced at the crosses? (2) Does the panel close also differ from the closing cross? (3) Does the
open bias hold for liquid stocks in general, or only for the picks? (4) Do two earlier variant comparisons (study 69
"today + yesterday's picks", study 65 skip Monday-night entries) change at cross prices?

Data: official crosses fetched here from Alpaca SIP trades into data/local/auctions94.parquet (one row per ticker,
day, kind): the opening cross is the first trade with condition 'O' printed by the listing exchange, the closing cross
the first with condition '6' from the listing exchange (other venues print 'O'/'6' for their own auctions, e.g. an
Arca '6' print for a Nasdaq stock; listing venue from Alpaca's asset lists, as in study 88). Windows are searched in
order (open 09:30:00-09:30:03, -09:30:20, -09:32, -09:45; close 15:59:59-16:00:30, -16:15, and 12:59:59-13:15 on
early-close days); one request covers all symbols of a day (paged). Second pass for pairs with no listing-venue
print (2.3%, mostly listing transfers such as PLTR and WMT NYSE -> Nasdaq or AZN Nasdaq -> NYSE, where today's venue
is wrong for earlier days): the largest 'O' / '6' print from any exchange (N, Q, A, P, Z), which matched the
listing-venue rule on all 110 already-found pairs checked (3 days); 0.3% of pairs remain without a cross.
Opening crosses of the uncapped blend picks' sell days are reused from study 88 (data/local/auctions88.parquet,
same rule). Panel prices: core.load_panel
(split- and dividend-adjusted Yahoo), Yahoo raw close P["rawc"] (adjusted for later splits) put back on the day's
traded scale with data/local/events/splits_yf.csv and checked against Alpaca's raw close (data/local/d1raw.parquet).
Picks: blend as study 69 (lines 22-46: top 10 of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3); the live version
with at most 3 names per industry (paper_overnight.cap_by_industry, data/store/sectors.parquet); study 68's day short
(easy-to-borrow picks of night t shorted from t+1's opening cross to t+1's closing cross, 1/10 of capital each, cost
auction + 2.5 bp per side, 30%/yr borrow fee on non-easy names); studies 67/71's day-session long (results/
study60_pred.parquet p_all, price > $5, ADV > $5M, top 10, buy at the open, sell at the close, study 25 cost column,
predictions start 2024-07). Study 80's earnings-night predictions are not saved in results/ (computed inside the
script), so that candidate is not rescored. Random sample: 3,000 liquid stock-days per period (traded close of the
previous day > $5, 20-day mean dollar volume > $5M, seed 94) for the bias outside the picks.

Design: every price is compared on the day's own traded scale: panel open in raw terms = P["o"] / P["c"] x (Yahoo raw
close x later split factor), panel close = Yahoo raw close x later split factor. Per leg the ratio cross / panel
price is taken; a leg whose panel close disagrees with Alpaca's raw close by more than 3% (unresolved split scale) or
whose ratio is outside 0.8..1.25 is a data error and falls back to the panel price. Returns at crosses = panel
return x leg ratios (night t: (1 + o(t+1)/c(t) - 1) x ratio_open(t+1) / ratio_close(t); day: x ratio_close /
ratio_open), so splits and dividends between t and t+1 stay handled by the adjusted panel; as a check the same
returns are computed raw-to-raw from the crosses and pairs that disagree by more than 1% are counted (split or
dividend between the days). Where a cross is missing the panel price is used (fallback rates reported). Costs as in
the original studies. Bias stats: mean / median / share > 0 in bp of panel / cross - 1 (positive = panel above the
cross). Periods: 2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09 by entry date (day long and study 71 from
2024-07).

Data errors found: 34 opening and 34 closing legs whose Yahoo raw close could not be put on the traded scale
(splits missing from splits_yf.csv) and 1 opening cross outside 0.8..1.25 of the panel open; these use the panel
price. Raw cross-to-cross night returns of the blend picks agree with the ratio method except 3 reverse splits on
the sell day (FCEL, HOLO, XTIA) and 14 ex-dividend days (raw return lower by the dividend), as expected.

Output: results/study94_cross_rescore.csv (tables: bias = panel vs cross by set, kind, period; rescore = Sharpe,
mean net bp, max drawdown per strategy, period and pricing; fallback = share of legs priced at the panel; variants =
study 69 / 65 comparisons; rawcheck = raw-to-raw consistency).

    python src/study94_cross_rescore.py fetch   # official crosses (incremental, resumable)
    python src/study94_cross_rescore.py         # study
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "150")))
AUC_FN = os.path.join(A.LOCAL, "auctions94.parquet")
AUC88 = os.path.join(A.LOCAL, "auctions88.parquet")
OUT = os.path.join(RES, "study94_cross_rescore.csv")
PRIMARY = {"NASDAQ": "Q", "NYSE": "N", "AMEX": "A", "ARCA": "P", "BATS": "Z"}
LISTING = set(PRIMARY.values())
EARLY = {"2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24", "2026-07-02"}
START, END = "2024-01", "2026-09"
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]


# ------------------------------------------------------------------ picks
def build_picks():
    """All strategy picks. Returns P, cols, days, S (blend score), dict of 0/1 pick frames (index = entry day t),
    day-long selection (rows date, ticker, R, cost), etb mask, random liquid sample."""
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    # blend, exactly as study 69 lines 22-46
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr
    W = bt.select_topk(S, S.notna(), 10)
    # live version: at most 3 names per industry (paper_overnight.cap_by_industry, same rule)
    ind = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index(
        "ticker").industry
    Wc = pd.DataFrame(0.0, index=S.index, columns=cols)
    for d, row in S.iterrows():
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
    a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
    etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)])      # as study 68 (today's list)
    etbm = pd.DataFrame(np.broadcast_to(np.array([t in etb for t in cols]), W.shape), index=W.index, columns=cols)
    # day-session long (studies 60/67/71)
    x = pd.read_parquet(f"{RES}/study60_pred.parquet")
    liq = (x.px > 5) & (x.adv20 > 5e6)
    dl = x[liq & x.p_all.notna()].sort_values("p_all", ascending=False).groupby("date").head(10)
    dl = dl[["date", "ticker", "R", "cost"]].reset_index(drop=True)
    del x
    # random liquid stock-days, 3000 per period (decided before the open of d: previous day's values)
    tc = traded_close(P)[cols]
    adv = P["dv"][cols].rolling(20, min_periods=10).mean()
    el = ((tc.shift(1) > 5) & (adv.shift(1) > 5e6) & P["o"][cols].notna() & P["c"][cols].notna())
    rng = np.random.default_rng(94)
    rs = []
    for lab, a0, b0 in PER[:2]:
        s = el.loc[a0:b0].stack()
        s = s[s].index.to_frame(index=False)
        s.columns = ["date", "ticker"]
        rs.append(s.iloc[np.sort(rng.choice(len(s), 3000, replace=False))].assign(per=lab))
    rnd = pd.concat(rs, ignore_index=True)
    return dict(P=P, cols=cols, days=days, S=S, W=W, Wc=Wc, etbm=etbm, dl=dl, rnd=rnd, tc=tc)


def need_list(K):
    """(ticker, date, kind) pairs needed. kind 'open' / 'close'. Tickers in panel (Yahoo) style."""
    days = K["days"]
    nxt = pd.Series(days[1:], index=days[:-1])
    nxt2 = pd.Series(days[2:], index=days[:-2])
    rows = []
    for name, Wx in [("blend", K["W"]), ("capped", K["Wc"])]:
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        t1, t2 = s.t.map(nxt), s.t.map(nxt2)
        rows += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                 pd.DataFrame({"ticker": s.ticker, "date": t1, "kind": "open"}),
                 pd.DataFrame({"ticker": s.ticker, "date": t1, "kind": "close"})]
        if name == "blend":     # study 69 night 2: close of t+1 -> open of t+2
            rows.append(pd.DataFrame({"ticker": s.ticker, "date": t2, "kind": "open"}))
    for d in [K["dl"], K["rnd"]]:
        rows += [pd.DataFrame({"ticker": d.ticker, "date": d.date, "kind": k}) for k in ["open", "close"]]
    n = pd.concat(rows, ignore_index=True).dropna()
    n["date"] = pd.to_datetime(n.date)
    return n.drop_duplicates().reset_index(drop=True)


# ------------------------------------------------------------------ fetch
def listing_venue():
    """Ticker (Alpaca style) -> SIP code of its listing exchange (Alpaca asset lists; study 88)."""
    a = pd.concat([pd.read_parquet(os.path.join(A.LOCAL, f"alpaca_assets_{k}.parquet")) for k in ["active", "inactive"]])
    a = a.drop_duplicates("symbol")
    return dict(zip(a.symbol, a.exchange.map(PRIMARY)))


def _utc(day, hms):
    return pd.Timestamp(f"{day} {hms}").tz_localize("America/New_York").tz_convert("UTC").strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _day_kind(day, kind, syms, ven):
    """Crosses for many symbols on one day: windows in order, each window one paged multi-symbol request for the
    symbols still not found. Returns rows (ticker, date, kind, px, sz, ts, x, first_px)."""
    if kind == "open":
        wins, cond = [("09:30:00", "09:30:03"), ("09:30:03", "09:30:20"), ("09:30:20", "09:32:00"),
                      ("09:32:00", "09:45:00")], "O"
    else:
        wins, cond = [("15:59:59", "16:00:30"), ("16:00:30", "16:15:00")], "6"
        if day in EARLY:
            wins = [("12:59:59", "13:15:00")] + wins
    res = {s: dict(px=np.nan, sz=np.nan, ts=None, x=None) for s in syms}
    first = {s: np.nan for s in syms}
    left = list(syms)
    for wa, wb in wins:
        if not left:
            break
        for i in range(0, len(left), 40):
            chunk = left[i:i + 40]
            p = dict(symbols=",".join(chunk), start=_utc(day, wa), end=_utc(day, wb), limit=10000, feed="sip")
            while True:
                j = A.get("trades", p)
                for t, trs in (j.get("trades") or {}).items():
                    if t not in res or not np.isnan(res[t]["px"]):
                        continue
                    v = ven.get(t)
                    for z in trs:
                        c = z.get("c", [])
                        if cond in c and (v is None or z.get("x") == v):
                            res[t] = dict(px=z["p"], sz=z["s"], ts=z["t"], x=z.get("x"))
                            break
                        if kind == "open" and np.isnan(first[t]) and not set(c) & {"I", "T", "U", "Z"}:
                            first[t] = z["p"]
                if not j.get("next_page_token"):
                    break
                p["page_token"] = j["next_page_token"]
        left = [s for s in left if np.isnan(res[s]["px"])]
    return [dict(ticker=s, date=pd.Timestamp(day), kind=kind, first_px=first[s], **res[s]) for s in syms]


def _day_kind_largest(day, kind, syms):
    """Second pass for pairs with no listing-venue cross (mostly listing transfers, e.g. PLTR and WMT moved from NYSE
    to Nasdaq, AZN from Nasdaq to NYSE, so today's listing venue is wrong for earlier days): the largest 'O' / '6'
    print of any listing exchange (N, Q, A, P, Z) in the first window that has one (the listing exchange's cross is by far the largest)."""
    if kind == "open":
        wins, cond = [("09:30:00", "09:30:20"), ("09:30:20", "09:32:00"), ("09:32:00", "09:45:00")], "O"
    else:
        wins, cond = [("15:59:59", "16:00:30"), ("16:00:30", "16:15:00")], "6"
        if day in EARLY:
            wins = [("12:59:59", "13:15:00")] + wins
    res = {s: dict(px=np.nan, sz=np.nan, ts=None, x=None) for s in syms}
    left = list(syms)
    for wa, wb in wins:
        if not left:
            break
        cand = {s: [] for s in left}
        p = dict(symbols=",".join(left), start=_utc(day, wa), end=_utc(day, wb), limit=10000, feed="sip")
        while True:
            j = A.get("trades", p)
            for t, trs in (j.get("trades") or {}).items():
                if t in cand:
                    cand[t] += [z for z in trs if cond in z.get("c", []) and z.get("x") in LISTING]
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
        for t, zs in cand.items():
            if zs:
                z = max(zs, key=lambda q: q["s"])
                res[t] = dict(px=z["p"], sz=z["s"], ts=z["t"], x=z.get("x"))
        left = [s for s in left if np.isnan(res[s]["px"])]
    return [dict(ticker=s, date=pd.Timestamp(day), kind=kind, rule="largest", **res[s]) for s in syms]


def refetch_missing():
    have = pd.read_parquet(AUC_FN)
    have["rule"] = (have["rule"] if "rule" in have else pd.Series(index=have.index, dtype="string")).fillna("venue")
    m = have.px.isna() & (have.rule == "venue")
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in have[m].groupby(["date", "kind"])]
    print("second pass (largest print):", int(m.sum()), "pairs,", len(jobs), "day-kind jobs", flush=True)
    with ThreadPoolExecutor(4) as ex:
        rows = [r for rr in ex.map(lambda jb: _day_kind_largest(*jb), jobs) for r in rr]
    new = pd.DataFrame(rows)
    keep = have[~m]
    old = have[m].drop(columns=["px", "sz", "ts", "x", "rule"]).merge(new, on=["ticker", "date", "kind"], how="left")
    old["rule"] = old.rule.fillna("largest")
    out = pd.concat([keep, old], ignore_index=True)
    out["rule"] = out.rule.astype("string")
    _save(out)
    print("found in second pass", int(new.px.notna().sum()), "of", len(new), flush=True)


def _save(df):
    df = df.copy()
    for k in ["ts", "x"]:
        df[k] = df[k].astype("string")
    df["ticker"] = df.ticker.astype("string")
    df.to_parquet(AUC_FN, index=False)


def fetch():
    K = build_picks()
    need = need_list(K)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)]
    del K
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    ven = listing_venue()
    if os.path.exists(AUC_FN):
        have = pd.read_parquet(AUC_FN)
    else:     # seed with study 88's opening crosses (same rule: listing venue, searched to 09:45)
        a = pd.read_parquet(AUC88)
        lv = a.ticker.map(ven)
        a = a[a.open_px.isna() | lv.isna() | (a.open_x == lv)]
        have = pd.DataFrame({"ticker": a.ticker, "date": pd.to_datetime(a.date), "kind": "open", "first_px": a.first_px,
                             "px": a.open_px, "sz": a.open_sz, "ts": a.open_ts, "x": a.open_x})
        _save(have)
    have["date"] = pd.to_datetime(have.date)
    m = need.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)

    def one(job):
        try:
            return _day_kind(job[0], job[1], job[2], ven)
        except RuntimeError as e:      # leave uncached; a rerun retries
            print("err", job[0], job[1], e, flush=True)
            return []
    rows = []
    t0 = time.time()
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(4) as ex:
            for r in ex.map(one, jobs[i:i + 24]):
                rows += r
        out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        _save(out)
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(rows), "found",
              f"{np.mean([not np.isnan(r['px']) for r in rows]):.3f}", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch()
    refetch_missing()
    sys.exit()


# ------------------------------------------------------------------ study
def split_factor(cols, index):
    """Product of later split ratios (data/local/events/splits_yf.csv): Yahoo raw close x factor = traded price."""
    sp = pd.read_csv(f"{DATA}/local/events/splits_yf.csv", parse_dates=["date"])
    sp = sp[sp.ticker.isin(cols) & (sp.date > index[0])]
    F = pd.DataFrame(1.0, index=index, columns=cols)
    for t, g in sp.groupby("ticker"):
        for d, r in zip(g.date, g.ratio):
            F.loc[F.index < d, t] *= r
    return F


def cross_ratios(K):
    """ro, rc: official cross / panel price on the day's traded scale (NaN = no cross or data error), plus raw
    crosses XO, XC, raw panel prices, and the error counts."""
    P, cols, days, tc = K["P"], K["cols"], K["days"], K["tc"]
    ix = days[(days >= "2023-12-01")]
    a = pd.read_parquet(AUC_FN)
    a["ticker"] = a.ticker.str.replace(".", "-", regex=False)
    a["date"] = pd.to_datetime(a.date)
    a = a[a.ticker.isin(cols) & a.date.isin(ix)]
    X = {k: g.drop_duplicates(["date", "ticker"]).pivot(index="date", columns="ticker", values="px").reindex(
        index=ix, columns=cols) for k, g in a.groupby("kind")}
    FP = {k: g.drop_duplicates(["date", "ticker"]).pivot(index="date", columns="ticker", values="first_px").reindex(
        index=ix, columns=cols) for k, g in a.groupby("kind") if k == "open"}
    F = split_factor(cols, ix)
    prc = P["rawc"][cols].reindex(ix) * F
    scale_ok = ((prc / tc.reindex(ix) - 1).abs() < 0.03) | tc.reindex(ix).isna()
    pro = P["o"][cols].reindex(ix) / P["c"][cols].reindex(ix) * prc
    out = dict(XO=X["open"], XC=X["close"], FO=FP["open"], PRO=pro, PRC=prc, fetched=a)
    err = {}
    for k, x, pr in [("ro", X["open"], pro), ("rc", X["close"], prc)]:
        r = x / pr
        bad_scale = x.notna() & ~scale_ok
        bad_ratio = x.notna() & scale_ok & ((r < 0.8) | (r > 1.25))
        err[k] = dict(scale=int(bad_scale.sum().sum()), ratio=int(bad_ratio.sum().sum()), found=int(x.notna().sum().sum()))
        out[k] = r.where(scale_ok & ~bad_ratio)
    out["err"] = err
    return out


def bias_rows(name, sel, ratio, extra=None):
    """sel: DataFrame date, ticker (price date). ratio: wide cross / panel. bp = panel / cross - 1."""
    rows = []
    r = ratio.stack(future_stack=True)
    v = r.reindex(pd.MultiIndex.from_arrays([sel.date, sel.ticker])).values
    d = pd.DataFrame({"date": sel.date.values, "r": v})
    d["bp"] = 1e4 * (1 / d.r - 1)
    for lab, a0, b0 in PER:
        x = d[(d.date >= a0) & (d.date <= pd.Timestamp(b0) + pd.offsets.MonthEnd(0))]
        y = x.bp.dropna()
        rows.append(dict(table="bias", set=name, period=lab, n=len(x), cross_found=y.size / max(len(x), 1),
                         mean_bp=y.mean(), median_bp=y.median(), share_pos=(y > 0.5).mean(), share_zero=(y.abs() <= 0.5).mean(),
                         p10_bp=y.quantile(0.1), p90_bp=y.quantile(0.9),
                         t=y.mean() / (y.std() / np.sqrt(len(y))) if len(y) > 2 else np.nan))
    return rows


def stats_rows(name, net, pricing, start=None, extra=None):
    rows = []
    for lab, a0, b0 in PER:
        if start is not None and a0 < start:
            a0 = start
            lab = lab.replace("2024-25H1", "2024H2-25H1").replace("2024-26", "2024H2-26")
        x = net.loc[a0:b0].dropna()
        st = ann_stats(x)
        rows.append(dict(table="rescore", strategy=name, pricing=pricing, period=lab, n=st["n"], sharpe=st["sharpe"],
                         net_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"], **(extra or {})))
    return rows


def fallback_rows(name, W, miss):
    """Share of held legs priced at the panel (no usable cross). miss: wide bool, True where a leg has no cross."""
    rows = []
    h = W > 0
    for lab, a0, b0 in PER:
        hh, mm = h.loc[a0:b0], (miss.reindex_like(W).fillna(True) & h).loc[a0:b0]
        rows.append(dict(table="fallback", strategy=name, period=lab, legs=int(hh.sum().sum()),
                         share_panel=mm.sum().sum() / max(hh.sum().sum(), 1)))
    return rows


def main():
    K = build_picks()
    P, cols, days, W, Wc, etbm, S = K["P"], K["cols"], K["days"], K["W"], K["Wc"], K["etbm"], K["S"]
    Q = cross_ratios(K)
    ro = Q["ro"].reindex(index=days, columns=cols)
    rc = Q["rc"].reindex(index=days, columns=cols)
    print("cross data errors (legs set to the panel price):", Q["err"], flush=True)
    rows = []
    for k, v in Q["err"].items():
        rows.append(dict(table="data_errors", set=k, n=v["found"], scale_mismatch=v["scale"], ratio_out_of_range=v["ratio"]))
    nxt = pd.Series(days[1:], index=days[:-1])

    def picks_frame(Wx, shift_days=0):
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        d = s.t
        for _ in range(shift_days):
            d = d.map(nxt)
        return pd.DataFrame({"date": d, "ticker": s.ticker}).dropna()
    # ---- (2)(3) bias of panel open / close vs the crosses
    rows += bias_rows("blend picks: sell-day open (t+1)", picks_frame(W, 1), ro)
    rows += bias_rows("blend picks: buy-day close (t)", picks_frame(W, 0), rc)
    rows += bias_rows("capped picks: sell-day open (t+1)", picks_frame(Wc, 1), ro)
    rows += bias_rows("capped picks: buy-day close (t)", picks_frame(Wc, 0), rc)
    rows += bias_rows("day short (etb picks): close of t+1", picks_frame(W.where(etbm, 0), 1), rc)
    dl = K["dl"]
    rows += bias_rows("day long top 10: open", dl, ro)
    rows += bias_rows("day long top 10: close", dl, rc)
    rnd = K["rnd"]
    rows += bias_rows("random liquid stock-days: open", rnd, ro)
    rows += bias_rows("random liquid stock-days: close", rnd, rc)
    # random sample: is Yahoo's open the first trade or the cross? (raw scale)
    FO, PRO, XO = Q["FO"], Q["PRO"], Q["XO"]
    mi = pd.MultiIndex.from_arrays([rnd.date, rnd.ticker])
    g = lambda w: w.stack(future_stack=True).reindex(mi).values
    fo, yo, xo = g(FO), g(PRO), g(XO)
    ok = ~np.isnan(xo) & ~np.isnan(yo)
    for lab in ["2024-25H1", "2025H2-26"]:
        m = ok & (rnd.per.values == lab)
        rows.append(dict(table="yahoo_open_is", set="random liquid stock-days", period=lab, n=int(m.sum()),
                         eq_cross=np.mean(np.abs(yo[m] / xo[m] - 1) < 2e-4),
                         eq_first_trade=np.nanmean(np.where(np.isnan(fo[m]), np.nan, np.abs(yo[m] / fo[m] - 1) < 2e-4)),
                         first_trade_before_cross=np.mean(~np.isnan(fo[m]))))
    # random sample bias by liquidity tercile and listing venue
    ven = listing_venue()
    rr = pd.DataFrame({"date": rnd.date, "ticker": rnd.ticker, "per": rnd.per,
                       "bp": 1e4 * (1 / g(ro) - 1)})
    adv = P["dv"][cols].rolling(20, min_periods=10).mean().shift(1)
    rr["adv"] = adv.stack(future_stack=True).reindex(mi).values
    rr["venue"] = rr.ticker.str.replace("-", ".", regex=False).map(ven).fillna("?")
    rr["adv_tercile"] = pd.qcut(rr.adv, 3, labels=["low", "mid", "high"]).astype(str)
    for key in ["venue", "adv_tercile"]:
        for (lab, lev), x in rr.dropna(subset=["bp"]).groupby(["per", key]):
            rows.append(dict(table="bias_random_by", set=f"{key}={lev}", period=lab, n=len(x), mean_bp=x.bp.mean(),
                             median_bp=x.bp.median(), share_pos=(x.bp > 0.5).mean(),
                             t=x.bp.mean() / (x.bp.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan))

    # ---- (4) rescore. Correction factors with fallback to the panel price (factor 1) where no usable cross.
    fo_ = ro.fillna(1.0)
    fc_ = rc.fillna(1.0)
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = (o.shift(-1) / c - 1).reindex_like(S)                                   # night t: close t -> open t+1
    Rn_x = ((1 + Rn_p) * fo_.shift(-1).reindex_like(S) / fc_.reindex_like(S) - 1)
    miss_n = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    C69 = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    nets = {}
    for name, Wx in [("blend top 10 (study 69 base)", W), ("blend top 10, max 3 per industry (live)", Wc)]:
        for pr, R in [("panel", Rn_p), ("cross", Rn_x)]:
            nets[(name, pr)] = bt.run(Wx, R, C69).net
            rows += stats_rows(name, nets[(name, pr)], pr)
        rows += fallback_rows(name, Wx, miss_n)
    # study 69: today + yesterday's picks; study 65: skip Monday-night entries
    okS = S.notna()
    Wy = W.shift(1).fillna(0).where(okS, 0)
    both = ((W > 0) | (Wy > 0)).astype(float)
    both = both.div(both.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    rows += fallback_rows("69: today + yesterday's picks", both, miss_n)
    for pr, R in [("panel", Rn_p), ("cross", Rn_x)]:
        base = nets[("blend top 10 (study 69 base)", pr)]
        v69 = bt.run(both, R, C69).net
        mon = pd.Series(base.index.weekday == 0, index=base.index)
        v65 = base.where(~mon, 0.0)
        for vname, net in [("base: blend top 10", base), ("69: today + yesterday's picks", v69),
                           ("65: skip Monday-night entries (cash)", v65)]:
            rows += [dict(r, table="variants") for r in stats_rows(vname, net, pr)]
        for lab, a0, b0 in PER[:2]:
            x = base.loc[a0:b0]
            for wd, v in x.groupby(x.index.weekday):
                rows.append(dict(table="weekday", strategy="blend top 10", pricing=pr, period=lab,
                                 set=["Mon", "Tue", "Wed", "Thu", "Fri"][wd], n=len(v), net_bp=1e4 * v.mean()))
    # study 68 day short (and the night leg as built in studies 68/71: cost of t+1's auction)
    Wd = bt.select_topk(S.loc[S.index < days[-2]], S.loc[S.index < days[-2]].notna(), 10)
    Wd = Wd.reindex(S.index).fillna(0)
    Rd_p = (c.shift(-1) / o.shift(-1) - 1).reindex_like(Wd)                        # day session of t+1
    Rd_x = (1 + Rd_p) * fc_.shift(-1).reindex_like(Wd) / fo_.shift(-1).reindex_like(Wd) - 1
    miss_d = (ro.shift(-1).isna() | rc.shift(-1).isna()).reindex_like(Wd)
    cost68 = (exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5).reindex_like(Wd)
    fee = pd.DataFrame(np.where(etbm.reindex_like(Wd), 0.0, 0.30 / 252 * 1e4 / 2), index=Wd.index, columns=cols)
    ws = ((Wd > 0) & etbm.reindex_like(Wd)).astype(float).div(10)
    rows += fallback_rows("68: day short, etb picks", ws, miss_d)
    night71, short71 = {}, {}
    for pr, Rn, Rd in [("panel", Rn_p, Rd_p), ("cross", Rn_x, Rd_x)]:
        night71[pr] = bt.run(Wd, Rn.reindex_like(Wd), cost68).net
        short71[pr] = bt.run(ws, Rd, cost68 + fee, side=-1).net
        rows += stats_rows("68: day short, etb picks (1/10 each, standalone)", short71[pr], pr)
        rows += stats_rows("68: blend + quarter-size day short", (night71[pr] + 0.25 * short71[pr].fillna(0)), pr)
    # studies 60/67/71 day-session long
    dk = pd.MultiIndex.from_arrays([dl.date, dl.ticker])
    fac = (fc_.stack(future_stack=True).reindex(dk).values / fo_.stack(future_stack=True).reindex(dk).values)
    dl = dl.assign(R_x=(1 + dl.R.values) * np.nan_to_num(fac, nan=1.0) - 1,
                   miss=(rc.stack(future_stack=True).reindex(dk).isna().values | ro.stack(future_stack=True).reindex(dk).isna().values))
    prev = pd.Series(days[:-1], index=days[1:])
    for lab, a0, b0 in PER:
        x = dl[(dl.date >= a0) & (dl.date <= pd.Timestamp(b0) + pd.offsets.MonthEnd(0))]
        rows.append(dict(table="fallback", strategy="67/71: day long top 10", period=lab, legs=len(x), share_panel=x.miss.mean()))
    for pr, col in [("panel", "R"), ("cross", "R_x")]:
        gq = dl.groupby("date")
        dln = gq[col].mean() - 2 * gq.cost.mean() / 1e4                               # day session of date d
        rows += stats_rows("60: day long top 10 (standalone)", dln, pr, start="2024-07")
        dla = dln.copy()
        dla.index = dla.index.map(prev)                                               # align to the preceding night
        idx = night71[pr].loc["2024-07":END].dropna().index
        n_ = night71[pr].reindex(idx)
        rows += stats_rows("71 night leg alone (same window)", n_, pr, start="2024-07")
        rows += stats_rows("67: blend + half-size day long", n_ + 0.5 * dla.reindex(idx).fillna(0), pr, start="2024-07")
        rows += stats_rows("71: blend + half day long + quarter day short",
                           n_ + 0.5 * dla.reindex(idx).fillna(0) + 0.25 * short71[pr].reindex(idx).fillna(0), pr,
                           start="2024-07")
    # ---- raw-to-raw check: night returns of the blend picks from raw crosses vs the ratio method
    XO_ = Q["XO"].reindex(index=days, columns=cols)
    XC_ = Q["XC"].reindex(index=days, columns=cols)
    raw = (XO_.shift(-1) / XC_ - 1).reindex_like(S)
    pf = picks_frame(W, 0)
    mi2 = pd.MultiIndex.from_arrays([pf.date, pf.ticker])
    a_ = raw.stack(future_stack=True).reindex(mi2).values
    b_ = Rn_x.where(~miss_n).stack(future_stack=True).reindex(mi2).values
    m = ~np.isnan(a_) & ~np.isnan(b_)
    dif = np.abs((1 + a_[m]) / (1 + b_[m]) - 1)
    sp = pd.read_csv(f"{DATA}/local/events/splits_yf.csv", parse_dates=["date"])
    spk = set(zip(sp.ticker, sp.date))
    t1 = pf.date.map(nxt).values[m]
    is_split = np.array([(t, d) in spk for t, d in zip(pf.ticker.values[m], t1)])
    rows.append(dict(table="rawcheck", set="blend night: raw cross-to-cross vs adjusted ratio method", n=int(m.sum()),
                     diff_gt_1pct=int((dif > 0.01).sum()), diff_gt_1pct_split_next_day=int(((dif > 0.01) & is_split).sum()),
                     diff_0p1_to_1pct=int(((dif > 0.001) & (dif <= 0.01)).sum()), median_abs_diff_bp=1e4 * np.median(dif)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(3).to_string(index=False))
    r = df[df.table == "rescore"].pivot_table(index="strategy", columns=["period", "pricing"],
                                              values=["sharpe", "net_bp"], sort=False)
    print(r.round(2).to_string())


if __name__ == "__main__":
    main()
